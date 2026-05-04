# -*- coding: utf-8 -*-
"""
Optuna hyperparameter search for the stacking meta-model.

Requires pre-trained LGBM/CatBoost models and OOF files.
Run after run_train.py and run_catboost.py.

Usage:
    python run_optuna_meta.py
"""

import pickle
import warnings

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
import yaml
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold

from src.config import (
    CONFIGS_DIR,
    DATA_PROCESSED,
    LAYOUT_PATH,
    LGBM_CONFIG,
    MODELS_DIR,
    TARGET_COL,
    TRAIN_PATH,
    gpu_available,
)
from src.features import (
    add_lag_features,
    add_lead_features,
    add_scenario_aggregate_features,
    add_scenario_relative_features,
    add_scenario_trajectory_features,
    build_features,
    build_time_features,
    merge_layout,
)

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

META_CONFIG   = CONFIGS_DIR / "meta.yaml"
CB_MODELS_DIR = MODELS_DIR / "catboost"
TOP_K_FEATURES = 100
META_N_SPLITS  = 5
N_TRIALS       = 60
SEED           = 42


def load_lgbm_config() -> dict:
    with open(LGBM_CONFIG, encoding="utf-8") as f:
        return yaml.safe_load(f)


def preprocess(cfg: dict) -> pd.DataFrame:
    train  = pd.read_csv(TRAIN_PATH)
    layout = pd.read_csv(LAYOUT_PATH)
    train = merge_layout(train, layout)
    train = build_time_features(train)
    train = build_features(train)
    train = add_scenario_aggregate_features(train)
    train = add_scenario_relative_features(train)
    train = add_scenario_trajectory_features(train)
    train = add_lead_features(train, leads=[1, 2, 3, 4, 5])
    lag_cfg = cfg["lag"]
    train = add_lag_features(
        train,
        lag_cols=lag_cfg["cols"],
        lags=lag_cfg["lags"],
        rolling_windows=lag_cfg.get("rolling_windows", [3, 5]),
    )
    return train


def load_lgbm_models() -> tuple[list, list[str]]:
    paths = sorted(MODELS_DIR.glob("lgbm_fold*.txt"))
    if not paths:
        raise FileNotFoundError(f"No LGBM models in {MODELS_DIR}")
    models = [lgb.Booster(model_str=open(p, encoding="utf-8").read()) for p in paths]
    with open(MODELS_DIR / "feature_cols.pkl", "rb") as f:
        feature_cols = pickle.load(f)
    return models, feature_cols


def get_top_features(lgbm_models: list, lgbm_features: list[str], top_k: int) -> list[str]:
    importance = np.zeros(len(lgbm_features))
    for m in lgbm_models:
        importance += m.feature_importance(importance_type="gain")
    importance /= len(lgbm_models)
    top_idx = np.argsort(importance)[::-1][:top_k]
    return [lgbm_features[i] for i in top_idx]


def build_meta_features(train: pd.DataFrame, top_feats: list[str]) -> tuple[pd.DataFrame, np.ndarray, pd.Series]:
    oof_lgbm = pd.read_csv(DATA_PROCESSED / "oof_lgbm.csv")
    oof_cb   = pd.read_csv(DATA_PROCESSED / "oof_catboost.csv")

    merged = oof_lgbm.merge(
        oof_cb[["ID", "oof_pred"]].rename(columns={"oof_pred": "oof_cb"}),
        on="ID",
    ).rename(columns={"oof_pred": "oof_lgbm"})

    orig_cols = ["ID", "scenario_id"] + [f for f in top_feats if f in train.columns]
    merged = merged.merge(train[orig_cols], on="ID", how="left")

    meta_cols = ["oof_lgbm", "oof_cb"] + [f for f in top_feats if f in merged.columns]
    X_meta  = merged[meta_cols]
    y_meta  = merged[TARGET_COL].values
    groups  = merged["scenario_id"]
    return X_meta, y_meta, groups


def objective(trial, X_meta, y_meta, groups):
    params = {
        "objective":         "regression_l1",
        "verbosity":         -1,
        "n_jobs":            -1,
        "seed":              SEED,
        "learning_rate":     trial.suggest_float("learning_rate",    0.005, 0.1,  log=True),
        "num_leaves":        trial.suggest_int  ("num_leaves",       15,    127),
        "min_child_samples": trial.suggest_int  ("min_child_samples", 5,    50),
        "feature_fraction":  trial.suggest_float("feature_fraction",  0.5,  1.0),
        "bagging_fraction":  trial.suggest_float("bagging_fraction",  0.5,  1.0),
        "bagging_freq":      1,
        "reg_alpha":         trial.suggest_float("reg_alpha",  1e-4, 1.0, log=True),
        "reg_lambda":        trial.suggest_float("reg_lambda", 1e-4, 1.0, log=True),
    }
    if gpu_available():
        params["device_type"] = "gpu"

    gkf = GroupKFold(n_splits=META_N_SPLITS)
    oof  = np.zeros(len(X_meta))

    for tr_idx, val_idx in gkf.split(X_meta, y_meta, groups):
        X_tr,  X_val = X_meta.iloc[tr_idx], X_meta.iloc[val_idx]
        y_tr,  y_val = y_meta[tr_idx],       y_meta[val_idx]

        dtrain = lgb.Dataset(X_tr, label=y_tr)
        dval   = lgb.Dataset(X_val, label=y_val, reference=dtrain)

        model = lgb.train(
            params, dtrain,
            num_boost_round=2000,
            valid_sets=[dval],
            callbacks=[
                lgb.early_stopping(80, verbose=False),
                lgb.log_evaluation(-1),
            ],
        )
        oof[val_idx] = model.predict(X_val)

    return mean_absolute_error(y_meta, oof)


def run() -> None:
    print("=== Optuna Meta-model hyperparameter search ===")
    cfg = load_lgbm_config()

    print("Loading LGBM models for feature importance...")
    lgbm_models, lgbm_features = load_lgbm_models()
    top_feats = get_top_features(lgbm_models, lgbm_features, TOP_K_FEATURES)

    print("Preprocessing train data...")
    train = preprocess(cfg)

    print("Building meta-features...")
    X_meta, y_meta, groups = build_meta_features(train, top_feats)
    print(f"  Meta shape: {X_meta.shape}")

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
    )

    def _obj(trial):
        return objective(trial, X_meta, y_meta, groups)

    print(f"Running {N_TRIALS} trials...")
    study.optimize(_obj, n_trials=N_TRIALS, show_progress_bar=True)

    best = study.best_trial
    print(f"\nBest Meta OOF MAE: {best.value:.4f}")
    print(f"Best params: {best.params}")

    # Save to meta.yaml
    meta_cfg = {
        "model": {
            "objective":         "regression_l1",
            "learning_rate":     best.params["learning_rate"],
            "num_leaves":        best.params["num_leaves"],
            "min_child_samples": best.params["min_child_samples"],
            "feature_fraction":  round(best.params["feature_fraction"], 4),
            "bagging_fraction":  round(best.params["bagging_fraction"],  4),
            "bagging_freq":      1,
            "reg_alpha":         round(best.params["reg_alpha"],  6),
            "reg_lambda":        round(best.params["reg_lambda"], 6),
            "n_jobs":            -1,
            "seed":              SEED,
            "verbosity":         -1,
        },
        "train": {
            "n_splits":             META_N_SPLITS,
            "num_boost_round":      2000,
            "early_stopping_rounds": 80,
        },
    }

    CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
    with open(META_CONFIG, "w", encoding="utf-8") as f:
        yaml.dump(meta_cfg, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

    print(f"\nSaved: {META_CONFIG}")
    print("Run python run_stacking.py to retrain with new meta params.")
    print("=== Search done ===")


if __name__ == "__main__":
    run()
