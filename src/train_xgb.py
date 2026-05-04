# -*- coding: utf-8 -*-
"""
XGBoost training pipeline.

Usage:
    python run_xgb.py
"""

import pickle

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold
from xgboost import XGBRegressor

from src.config import (
    CONFIGS_DIR,
    DATA_PROCESSED,
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

XGB_CONFIG    = CONFIGS_DIR / "xgb.yaml"
XGB_MODELS_DIR = MODELS_DIR / "xgb"


def load_config() -> dict:
    with open(XGB_CONFIG, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_lgbm_lag_config() -> dict:
    with open(LGBM_CONFIG, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    train  = pd.read_csv(TRAIN_PATH)
    layout = pd.read_csv(LAYOUT_PATH)
    return train, layout


def preprocess(
    train: pd.DataFrame, layout: pd.DataFrame, lag_cfg: dict
) -> pd.DataFrame:
    train = merge_layout(train, layout)
    train = build_time_features(train)
    train = build_features(train)
    train = add_scenario_aggregate_features(train)
    train = add_scenario_relative_features(train)
    train = add_scenario_trajectory_features(train)
    train = add_lead_features(train, leads=[1, 2, 3, 4, 5])
    train = add_lag_features(
        train,
        lag_cols=lag_cfg["lag"]["cols"],
        lags=lag_cfg["lag"]["lags"],
        rolling_windows=lag_cfg["lag"].get("rolling_windows", [3, 5]),
    )
    return train


def run_cv(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    feature_cols: list[str],
    cfg: dict,
) -> tuple[np.ndarray, list]:
    params   = dict(cfg["model"])
    if gpu_available():
        params["device"] = "cuda"
        print("  [GPU mode] XGBoost using CUDA")
    n_splits = cfg["train"]["n_splits"]
    gkf      = GroupKFold(n_splits=n_splits)

    oof_preds = np.zeros(len(X))
    cv_scores: list[float] = []
    models    = []

    for fold, (tr_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        X_tr, X_val = X.iloc[tr_idx][feature_cols], X.iloc[val_idx][feature_cols]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

        y_tr_log  = np.log1p(y_tr)
        y_val_log = np.log1p(y_val)

        model = XGBRegressor(
            **params,
            n_estimators=cfg["train"]["n_estimators"],
            early_stopping_rounds=cfg["train"]["early_stopping_rounds"],
        )
        model.fit(
            X_tr, y_tr_log,
            eval_set=[(X_val, y_val_log)],
            verbose=False,
        )

        val_pred = np.expm1(model.predict(X_val))
        oof_preds[val_idx] = val_pred
        mae = mean_absolute_error(y_val, val_pred)
        cv_scores.append(mae)
        print(f"  Fold {fold + 1}/{n_splits}  MAE={mae:.4f}  best_iter={model.best_iteration}")
        models.append(model)

    oof_mae = mean_absolute_error(y, oof_preds)
    print(f"\nOOF MAE : {oof_mae:.4f}")
    print(f"CV  Mean: {np.mean(cv_scores):.4f} +/- {np.std(cv_scores):.4f}")
    return oof_preds, models


def save_models(models: list, feature_cols: list[str]) -> None:
    XGB_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    # pickle 사용 — XGBoost C++ 백엔드의 비ASCII 경로 문제 회피
    for i, model in enumerate(models):
        with open(XGB_MODELS_DIR / f"xgb_fold{i}.pkl", "wb") as f:
            pickle.dump(model, f)
    with open(XGB_MODELS_DIR / "feature_cols.pkl", "wb") as f:
        pickle.dump(feature_cols, f)
    print(f"XGBoost models saved: {XGB_MODELS_DIR}")


def run() -> None:
    print("=== XGBoost training start ===")
    lag_cfg       = load_lgbm_lag_config()
    cfg           = load_config()
    train, layout = load_data()

    print("Preprocessing...")
    train = preprocess(train, layout, lag_cfg)

    feature_cols = get_feature_cols(train, DROP_COLS)
    X      = train[feature_cols]
    y      = train[TARGET_COL]
    groups = train["scenario_id"]

    print(f"Features: {len(feature_cols)} | Samples: {len(X)}")
    print("\n[XGBoost GroupKFold CV]")
    oof_preds, models = run_cv(X, y, groups, feature_cols, cfg)

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    oof_df = train[["ID", TARGET_COL]].copy()
    oof_df["oof_pred"] = oof_preds
    oof_df.to_csv(DATA_PROCESSED / "oof_xgb.csv", index=False)

    save_models(models, feature_cols)
    print("=== XGBoost training done ===")
