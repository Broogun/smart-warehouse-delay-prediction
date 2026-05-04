# -*- coding: utf-8 -*-
"""
CatBoost training pipeline.

Usage:
    python run_catboost.py
"""

import pickle

import numpy as np
import pandas as pd
import yaml
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold

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

CB_CONFIG     = CONFIGS_DIR / "catboost.yaml"
CB_MODELS_DIR = MODELS_DIR / "catboost"


def load_config() -> dict:
    with open(CB_CONFIG, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_lgbm_lag_config() -> dict:
    with open(LGBM_CONFIG, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    train  = pd.read_csv(TRAIN_PATH)
    layout = pd.read_csv(LAYOUT_PATH)
    return train, layout


def preprocess(train: pd.DataFrame, layout: pd.DataFrame, lag_cfg: dict) -> pd.DataFrame:
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


def compute_sample_weights(y: pd.Series, groups: pd.Series | None = None) -> np.ndarray:
    """Higher weight for high-delay samples + scenario-level multiplier."""
    y_arr = y.values.astype(float)

    weights = np.ones(len(y_arr))
    weights = np.where(y_arr > 20, 1.0 + 0.5 * (y_arr - 20) / 20, weights)
    weights = np.where(y_arr > 40, 1.5 + np.log1p((y_arr - 40) / 10), weights)
    weights = np.clip(weights, 1.0, 15.0)

    if groups is not None:
        scen_mean = y.groupby(groups).transform("mean").values
        scen_mult = 1.0 + np.log1p(np.clip(scen_mean - 40, 0, None) / 30)
        scen_mult = np.clip(scen_mult, 1.0, 3.0)
        weights = weights * scen_mult

    return np.clip(weights, 1.0, 20.0)


def run_cv(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    feature_cols: list[str],
    cfg: dict,
) -> tuple[np.ndarray, list]:
    params   = dict(cfg["model"])
    task_type = "GPU" if gpu_available() else "CPU"
    if task_type == "GPU":
        print("  [GPU mode] CatBoost using GPU")
    n_splits = cfg["train"]["n_splits"]
    gkf      = GroupKFold(n_splits=n_splits)

    oof_preds = np.zeros(len(X))
    cv_scores: list[float] = []
    models    = []

    for fold, (tr_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        X_tr, X_val = X.iloc[tr_idx][feature_cols], X.iloc[val_idx][feature_cols]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
        w_tr = compute_sample_weights(y_tr, groups=groups.iloc[tr_idx])

        y_tr_log  = np.log1p(y_tr)
        y_val_log = np.log1p(y_val)

        model = CatBoostRegressor(
            **params,
            task_type=task_type,
            iterations=cfg["train"]["iterations"],
            early_stopping_rounds=cfg["train"]["early_stopping_rounds"],
        )
        model.fit(X_tr, y_tr_log, eval_set=(X_val, y_val_log), use_best_model=True,
                  verbose=False, sample_weight=w_tr)

        val_pred = np.expm1(model.predict(X_val))
        oof_preds[val_idx] = val_pred
        mae = mean_absolute_error(y_val, val_pred)
        cv_scores.append(mae)
        print(f"  Fold {fold + 1}/{n_splits}  MAE={mae:.4f}  best_iter={model.best_iteration_}")
        models.append(model)

    oof_mae = mean_absolute_error(y, oof_preds)
    print(f"\nOOF MAE : {oof_mae:.4f}")
    print(f"CV  Mean: {np.mean(cv_scores):.4f} +/- {np.std(cv_scores):.4f}")
    return oof_preds, models


def save_models(models: list, feature_cols: list[str]) -> None:
    CB_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for i, model in enumerate(models):
        model.save_model(str(CB_MODELS_DIR / f"catboost_fold{i}.cbm"))
    with open(CB_MODELS_DIR / "feature_cols.pkl", "wb") as f:
        pickle.dump(feature_cols, f)
    print(f"CatBoost models saved: {CB_MODELS_DIR}")


def run() -> None:
    print("=== CatBoost training start ===")
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
    print("\n[CatBoost GroupKFold CV]")
    oof_preds, models = run_cv(X, y, groups, feature_cols, cfg)

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    oof_df = train[["ID", TARGET_COL]].copy()
    oof_df["oof_pred"] = oof_preds
    oof_df.to_csv(DATA_PROCESSED / "oof_catboost.csv", index=False)

    save_models(models, feature_cols)
    print("=== CatBoost training done ===")
