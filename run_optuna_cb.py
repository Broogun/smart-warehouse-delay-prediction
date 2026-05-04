# -*- coding: utf-8 -*-
"""
Optuna hyperparameter search for CatBoost.

Usage:
    python run_optuna_cb.py

Searches key CatBoost hyperparameters using the last GroupKFold fold.
Updates configs/catboost.yaml with best parameters found.
"""

import warnings

import numpy as np
import optuna
import pandas as pd
import yaml
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold

from src.config import (
    CONFIGS_DIR,
    DROP_COLS,
    LAYOUT_PATH,
    LGBM_CONFIG,
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

_USE_GPU   = gpu_available()
CB_CONFIG  = CONFIGS_DIR / "catboost.yaml"
N_TRIALS   = 40
N_SPLITS   = 5
SEED       = 42


def load_and_preprocess() -> tuple[pd.DataFrame, pd.Series, pd.Series, list[str]]:
    with open(LGBM_CONFIG, encoding="utf-8") as f:
        lag_cfg = yaml.safe_load(f)

    train  = pd.read_csv(TRAIN_PATH)
    layout = pd.read_csv(LAYOUT_PATH)

    train = merge_layout(train, layout)
    train = build_time_features(train)
    train = build_features(train)
    train = add_scenario_aggregate_features(train)
    train = add_scenario_relative_features(train)
    train = add_scenario_trajectory_features(train)
    train = add_lead_features(train, leads=[1, 2, 3, 4, 5])

    lc = lag_cfg["lag"]
    train = add_lag_features(
        train,
        lag_cols=lc["cols"],
        lags=lc["lags"],
        rolling_windows=lc.get("rolling_windows", [3, 5]),
    )

    feature_cols = get_feature_cols(train, DROP_COLS)
    X      = train[feature_cols]
    y      = train[TARGET_COL]
    groups = train["scenario_id"]
    return X, y, groups, feature_cols


def get_last_fold(X, y, groups):
    gkf = GroupKFold(n_splits=N_SPLITS)
    tr_idx, val_idx = None, None
    for tr_idx, val_idx in gkf.split(X, y, groups):
        pass
    return tr_idx, val_idx


def objective(trial, X, y, tr_idx, val_idx):
    params = {
        "loss_function":      "MAE",
        "eval_metric":        "MAE",
        "random_seed":        SEED,
        "thread_count":       1 if _USE_GPU else -1,
        "verbose":            0,
        "learning_rate":      trial.suggest_float("learning_rate",    0.01,  0.1,  log=True),
        "depth":              trial.suggest_int  ("depth",            4,     10),
        "l2_leaf_reg":        trial.suggest_float("l2_leaf_reg",      1e-3,  10.0, log=True),
        "random_strength":    trial.suggest_float("random_strength",  0.0,   3.0),
        "bagging_temperature":trial.suggest_float("bagging_temperature", 0.0, 2.0),
        "border_count":       trial.suggest_categorical("border_count", [64, 128, 254]),
        "min_data_in_leaf":   trial.suggest_int  ("min_data_in_leaf", 5,     50),
    }
    iterations            = trial.suggest_int("iterations", 500, 2500, step=100)

    X_tr,  X_val  = X.iloc[tr_idx],  X.iloc[val_idx]
    y_tr,  y_val  = y.iloc[tr_idx],  y.iloc[val_idx]

    y_tr_log  = np.log1p(y_tr)
    y_val_log = np.log1p(y_val)

    model = CatBoostRegressor(**params, task_type="GPU" if _USE_GPU else "CPU",
                               iterations=iterations, early_stopping_rounds=80)
    model.fit(X_tr, y_tr_log, eval_set=(X_val, y_val_log), use_best_model=True, verbose=False)

    preds = np.expm1(model.predict(X_val))
    return mean_absolute_error(y_val, preds)


def run() -> None:
    print("=== Optuna CatBoost hyperparameter search ===")
    print("Loading and preprocessing data...")
    X, y, groups, feature_cols = load_and_preprocess()
    print(f"Features: {len(feature_cols)} | Samples: {len(X)}")

    tr_idx, val_idx = get_last_fold(X, y, groups)
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

    with open(CB_CONFIG, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    iterations = best.params.pop("iterations")
    cfg["model"].update({
        "learning_rate":       best.params["learning_rate"],
        "depth":               best.params["depth"],
        "l2_leaf_reg":         round(best.params["l2_leaf_reg"],   6),
        "random_strength":     round(best.params["random_strength"], 4),
        "bagging_temperature": round(best.params["bagging_temperature"], 4),
        "border_count":        best.params["border_count"],
        "min_data_in_leaf":    best.params["min_data_in_leaf"],
    })
    cfg["train"]["iterations"] = iterations

    with open(CB_CONFIG, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

    print(f"\nUpdated {CB_CONFIG}")
    print("Run python run_catboost.py to retrain with new params.")
    print("=== Search done ===")


if __name__ == "__main__":
    run()
