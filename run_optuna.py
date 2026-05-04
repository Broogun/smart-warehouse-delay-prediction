# -*- coding: utf-8 -*-
"""
Optuna hyperparameter search for LightGBM.

Usage:
    python run_optuna.py

Searches over key LGBM hyperparameters using a single held-out fold
(last GroupKFold fold) for speed.  After the search, updates
configs/lgbm.yaml with the best parameters found.
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
    DROP_COLS,
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
    get_feature_cols,
    merge_layout,
)

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

_DEVICE    = "gpu" if gpu_available() else "cpu"
N_TRIALS   = 50
N_SPLITS   = 5   # same as training CV; use last fold as proxy
SEED       = 42


def load_base_config() -> dict:
    with open(LGBM_CONFIG, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_and_preprocess() -> tuple[pd.DataFrame, pd.Series, pd.Series, list[str]]:
    cfg    = load_base_config()
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

    feature_cols = get_feature_cols(train, DROP_COLS)
    X      = train[feature_cols]
    y      = train[TARGET_COL]
    groups = train["scenario_id"]
    return X, y, groups, feature_cols


def get_last_fold_indices(
    X: pd.DataFrame, y: pd.Series, groups: pd.Series
) -> tuple[np.ndarray, np.ndarray]:
    gkf = GroupKFold(n_splits=N_SPLITS)
    tr_idx, val_idx = None, None
    for tr_idx, val_idx in gkf.split(X, y, groups):
        pass
    return tr_idx, val_idx


def objective(
    trial: optuna.Trial,
    X: pd.DataFrame,
    y: pd.Series,
    tr_idx: np.ndarray,
    val_idx: np.ndarray,
) -> float:
    params = {
        "objective":         "regression_l1",
        "verbosity":         -1,
        "n_jobs":            -1,
        "seed":              SEED,
        "device_type":       _DEVICE,
        "learning_rate":     trial.suggest_float("learning_rate",  0.005, 0.05,  log=True),
        "num_leaves":        trial.suggest_int  ("num_leaves",     63,    600),
        "min_child_samples": trial.suggest_int  ("min_child_samples", 5, 50),
        "feature_fraction":  trial.suggest_float("feature_fraction",  0.5, 1.0),
        "bagging_fraction":  trial.suggest_float("bagging_fraction",  0.5, 1.0),
        "bagging_freq":      1,
        "reg_alpha":         trial.suggest_float("reg_alpha",  1e-4, 1.0, log=True),
        "reg_lambda":        trial.suggest_float("reg_lambda", 1e-4, 1.0, log=True),
        "max_depth":         trial.suggest_int  ("max_depth", 6, 12),
    }

    X_tr,  X_val  = X.iloc[tr_idx],  X.iloc[val_idx]
    y_tr,  y_val  = y.iloc[tr_idx],  y.iloc[val_idx]

    y_tr_log  = np.log1p(y_tr)
    y_val_log = np.log1p(y_val)

    dtrain = lgb.Dataset(X_tr, label=y_tr_log)
    dval   = lgb.Dataset(X_val, label=y_val_log, reference=dtrain)

    model = lgb.train(
        params,
        dtrain,
        num_boost_round=2500,
        valid_sets=[dval],
        callbacks=[
            lgb.early_stopping(100, verbose=False),
            lgb.log_evaluation(-1),
        ],
    )

    preds = np.expm1(model.predict(X_val))
    return mean_absolute_error(y_val, preds)


def run() -> None:
    print("=== Optuna LGBM hyperparameter search ===")
    print("Loading and preprocessing data...")
    X, y, groups, feature_cols = load_and_preprocess()
    print(f"Features: {len(feature_cols)} | Samples: {len(X)}")

    tr_idx, val_idx = get_last_fold_indices(X, y, groups)
    print(f"Using last fold - train: {len(tr_idx)}, val: {len(val_idx)}")

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
    )

    def _obj(trial):
        return objective(trial, X, y, tr_idx, val_idx)

    print(f"Running {N_TRIALS} trials...")
    study.optimize(_obj, n_trials=N_TRIALS, show_progress_bar=True)

    best = study.best_trial
    print(f"\nBest MAE : {best.value:.4f}")
    print(f"Best params: {best.params}")

    # Load current config and update model section
    with open(LGBM_CONFIG, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg["model"].update({
        "learning_rate":     best.params["learning_rate"],
        "num_leaves":        best.params["num_leaves"],
        "min_child_samples": best.params["min_child_samples"],
        "feature_fraction":  round(best.params["feature_fraction"], 4),
        "bagging_fraction":  round(best.params["bagging_fraction"],  4),
        "reg_alpha":         round(best.params["reg_alpha"],   6),
        "reg_lambda":        round(best.params["reg_lambda"],  6),
        "max_depth":         best.params["max_depth"],
    })

    with open(LGBM_CONFIG, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

    print(f"\nUpdated {LGBM_CONFIG}")
    print("Run python run_train.py to retrain with new params.")
    print("=== Search done ===")


if __name__ == "__main__":
    run()
