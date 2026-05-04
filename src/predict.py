# -*- coding: utf-8 -*-
"""
LightGBM inference pipeline.

Usage:
    python run_predict.py
"""

import pickle
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import yaml

from src.config import (
    DATA_SUBMISSION,
    LAYOUT_PATH,
    LGBM_CONFIG,
    MODELS_DIR,
    SAMPLE_SUB_PATH,
    TARGET_COL,
    TEST_PATH,
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


def load_config() -> dict:
    with open(LGBM_CONFIG, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_models() -> tuple[list[lgb.Booster], list[str]]:
    model_paths = sorted(MODELS_DIR.glob("lgbm_fold*.txt"))
    if not model_paths:
        raise FileNotFoundError(f"No LGBM models found in {MODELS_DIR}")
    models = [lgb.Booster(model_str=open(p, encoding="utf-8").read()) for p in model_paths]
    with open(MODELS_DIR / "feature_cols.pkl", "rb") as f:
        feature_cols = pickle.load(f)
    print(f"Loaded {len(models)} models")
    return models, feature_cols


def preprocess_test(cfg: dict) -> pd.DataFrame:
    test    = pd.read_csv(TEST_PATH)
    layout  = pd.read_csv(LAYOUT_PATH)
    test    = merge_layout(test, layout)
    test    = build_time_features(test)
    test    = build_features(test)
    test    = add_scenario_aggregate_features(test)
    test    = add_scenario_relative_features(test)
    test    = add_scenario_trajectory_features(test)
    test    = add_lead_features(test, leads=[1, 2, 3, 4, 5])
    lag_cfg = cfg["lag"]
    test    = add_lag_features(
        test,
        lag_cols=lag_cfg["cols"],
        lags=lag_cfg["lags"],
        rolling_windows=lag_cfg.get("rolling_windows", [3, 5]),
    )
    return test


def predict(models: list[lgb.Booster], X_test: pd.DataFrame) -> np.ndarray:
    preds = np.zeros(len(X_test))
    for model in models:
        preds += model.predict(X_test) / len(models)
    return preds


def save_submission(test: pd.DataFrame, preds: np.ndarray, name: str = "submission") -> Path:
    submission = pd.read_csv(SAMPLE_SUB_PATH)
    id_to_pred = dict(zip(test["ID"], np.clip(preds, 0, None)))
    submission[TARGET_COL] = submission["ID"].map(id_to_pred)
    DATA_SUBMISSION.mkdir(parents=True, exist_ok=True)
    out_path = DATA_SUBMISSION / f"{name}.csv"
    submission.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")
    print(submission[TARGET_COL].describe())
    return out_path


def run() -> None:
    print("=== Inference start ===")
    cfg = load_config()

    print("Preprocessing test data...")
    test = preprocess_test(cfg)

    models, feature_cols = load_models()

    missing = [c for c in feature_cols if c not in test.columns]
    if missing:
        print(f"Warning: {len(missing)} missing features — filling NaN")
        for c in missing:
            test[c] = np.nan

    X_test = test[feature_cols]
    preds  = predict(models, X_test)
    save_submission(test, preds, name="submission_lgbm")
    print("=== Inference done ===")
