# -*- coding: utf-8 -*-
"""
Ensemble inference: LGBM + CatBoost OOF-weighted average.

Usage:
    python run_ensemble.py
"""

import pickle

import lightgbm as lgb
import numpy as np
import pandas as pd
import yaml
from catboost import CatBoostRegressor
from scipy.optimize import minimize
from sklearn.metrics import mean_absolute_error

from src.config import (
    DATA_PROCESSED,
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

CB_MODELS_DIR = MODELS_DIR / "catboost"


def load_lgbm_config() -> dict:
    with open(LGBM_CONFIG, encoding="utf-8") as f:
        return yaml.safe_load(f)


def preprocess_test(cfg: dict) -> pd.DataFrame:
    test   = pd.read_csv(TEST_PATH)
    layout = pd.read_csv(LAYOUT_PATH)
    test   = merge_layout(test, layout)
    test   = build_time_features(test)
    test   = build_features(test)
    test   = add_scenario_aggregate_features(test)
    test   = add_scenario_relative_features(test)
    test   = add_scenario_trajectory_features(test)
    test   = add_lead_features(test, leads=[1, 2, 3, 4, 5])
    lag_cfg = cfg["lag"]
    test   = add_lag_features(
        test,
        lag_cols=lag_cfg["cols"],
        lags=lag_cfg["lags"],
        rolling_windows=lag_cfg.get("rolling_windows", [3, 5]),
    )
    return test


def load_lgbm_models() -> tuple[list, list[str]]:
    paths = sorted(MODELS_DIR.glob("lgbm_fold*.txt"))
    if not paths:
        raise FileNotFoundError(f"No LGBM models in {MODELS_DIR}")
    models = [lgb.Booster(model_str=open(p, encoding="utf-8").read()) for p in paths]
    with open(MODELS_DIR / "feature_cols.pkl", "rb") as f:
        feature_cols = pickle.load(f)
    return models, feature_cols


def load_cb_models() -> tuple[list, list[str]]:
    paths = sorted(CB_MODELS_DIR.glob("catboost_fold*.cbm"))
    if not paths:
        raise FileNotFoundError(f"No CatBoost models in {CB_MODELS_DIR}")
    models = []
    for p in paths:
        m = CatBoostRegressor()
        m.load_model(str(p))
        models.append(m)
    with open(CB_MODELS_DIR / "feature_cols.pkl", "rb") as f:
        feature_cols = pickle.load(f)
    return models, feature_cols


def find_optimal_weights(
    oof_paths: dict,
    y_col: str,
) -> dict[str, float]:
    """Find optimal 2-way blend weights from OOF predictions."""
    oofs = {}
    for name, path in oof_paths.items():
        if not path.exists():
            print(f"OOF not found for {name} - using equal weights")
            return {k: 0.5 for k in oof_paths}
        oofs[name] = pd.read_csv(path).rename(columns={"oof_pred": f"oof_{name}"})

    merged = oofs["lgbm"]
    merged = merged.merge(oofs["cb"][["ID", "oof_cb"]], on="ID")

    y_true     = merged[y_col].values
    lgbm_preds = merged["oof_lgbm"].values
    cb_preds   = merged["oof_cb"].values

    def objective(w):
        blend = w[0] * lgbm_preds + (1 - w[0]) * cb_preds
        return mean_absolute_error(y_true, blend)

    res = minimize(objective, x0=[0.5], bounds=[(0.0, 1.0)], method="L-BFGS-B")
    w_lgbm = float(res.x[0])
    w_cb   = 1.0 - w_lgbm

    weights = {"lgbm": w_lgbm, "cb": w_cb}
    print(f"Optimal weights: lgbm={w_lgbm:.3f}, cb={w_cb:.3f}  OOF MAE={res.fun:.4f}")
    return weights


def fill_missing(test: pd.DataFrame, feature_cols: list[str]) -> None:
    for col in feature_cols:
        if col not in test.columns:
            test[col] = np.nan


def run() -> None:
    print("=== Ensemble inference start ===")
    cfg = load_lgbm_config()

    print("Preprocessing test data...")
    test = preprocess_test(cfg)

    lgbm_models, lgbm_features = load_lgbm_models()
    cb_models,   cb_features   = load_cb_models()
    print(f"Loaded LGBM x{len(lgbm_models)}, CatBoost x{len(cb_models)}")

    oof_paths = {
        "lgbm": DATA_PROCESSED / "oof_lgbm.csv",
        "cb":   DATA_PROCESSED / "oof_catboost.csv",
    }
    weights = find_optimal_weights(oof_paths, TARGET_COL)

    fill_missing(test, lgbm_features)
    fill_missing(test, cb_features)

    lgbm_preds = np.expm1(np.mean([m.predict(test[lgbm_features]) for m in lgbm_models], axis=0))
    cb_preds   = np.expm1(np.mean([m.predict(test[cb_features])   for m in cb_models],   axis=0))

    preds = np.clip(
        weights["lgbm"] * lgbm_preds + weights["cb"] * cb_preds,
        0, None,
    )

    submission = pd.read_csv(SAMPLE_SUB_PATH)
    id_to_pred = dict(zip(test["ID"], preds))
    submission[TARGET_COL] = submission["ID"].map(id_to_pred)
    DATA_SUBMISSION.mkdir(parents=True, exist_ok=True)
    out_path = DATA_SUBMISSION / "submission_ensemble.csv"
    submission.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")
    print(submission[TARGET_COL].describe())
    print("=== Ensemble done ===")


if __name__ == "__main__":
    run()
