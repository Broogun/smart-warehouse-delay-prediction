# -*- coding: utf-8 -*-
"""
Optuna hyperparameter search for XGBoost.

Usage:
    python run_optuna_xgb.py

Searches key XGBoost hyperparameters using the last GroupKFold fold.
Updates configs/xgb.yaml with best parameters found.
"""

import warnings

import numpy as np
import optuna
import pandas as pd
import yaml
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold
from xgboost import XGBRegressor

from src.config import (
    CONFIGS_DIR,
    DROP_COLS,
    LAYOUT_PATH,
    LGBM_CONFIG,
    TARGET_COL,
    TRAIN_PATH,
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

XGB_CONFIG = CONFIGS_DIR / "xgb.yaml"
N_TRIALS   = 50
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
        "objective":        "reg:absoluteerror",
        "tree_method":      "hist",
        "device":           "cpu",
        "n_jobs":           -1,
        "seed":             SEED,
        "verbosity":        0,
        "learning_rate":    trial.suggest_float("learning_rate",   0.005, 0.05,  log=True),
        "max_depth":        trial.suggest_int  ("max_depth",       4,     10),
        "min_child_weight": trial.suggest_int  ("min_child_weight",5,     50),
        "subsample":        trial.suggest_float("subsample",       0.5,   1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree",0.5,   1.0),
        "reg_alpha":        trial.suggest_float("reg_alpha",       1e-4,  1.0,   log=True),
        "reg_lambda":       trial.suggest_float("reg_lambda",      1e-4,  1.0,   log=True),
        "gamma":            trial.suggest_float("gamma",           0.0,   5.0),
    }
    n_estimators = trial.suggest_int("n_estimators", 500, 3000, step=100)

    X_tr,  X_val  = X.iloc[tr_idx],  X.iloc[val_idx]
    y_tr,  y_val  = y.iloc[tr_idx],  y.iloc[val_idx]

    model = XGBRegressor(**params, n_estimators=n_estimators, early_stopping_rounds=80)
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

    preds = model.predict(X_val)
    return mean_absolute_error(y_val, preds)


def run() -> None:
    print("=== Optuna XGBoost hyperparameter search ===")
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

    with open(XGB_CONFIG, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    n_estimators = best.params.pop("n_estimators")
    cfg["model"].update({
        "learning_rate":    best.params["learning_rate"],
        "max_depth":        best.params["max_depth"],
        "min_child_weight": best.params["min_child_weight"],
        "subsample":        round(best.params["subsample"],        4),
        "colsample_bytree": round(best.params["colsample_bytree"], 4),
        "reg_alpha":        round(best.params["reg_alpha"],        6),
        "reg_lambda":       round(best.params["reg_lambda"],       6),
        "gamma":            round(best.params["gamma"],            4),
    })
    cfg["train"]["n_estimators"] = n_estimators

    with open(XGB_CONFIG, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

    print(f"\nUpdated {XGB_CONFIG}")
    print("Run python run_xgb.py to retrain with new params.")
    print("=== Search done ===")


if __name__ == "__main__":
    run()
