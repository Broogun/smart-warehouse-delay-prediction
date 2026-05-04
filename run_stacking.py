# -*- coding: utf-8 -*-
"""
Stacking inference: LGBM + CatBoost level-1, LightGBM meta-model level-2.
Meta-features: OOF predictions + top-N original features by LGBM importance.

Usage:
    python run_stacking.py
"""

import pickle

import lightgbm as lgb
import numpy as np
import pandas as pd
import yaml
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold
from src.config import (
    CONFIGS_DIR,
    DATA_PROCESSED,
    DATA_SUBMISSION,
    LAYOUT_PATH,
    LGBM_CONFIG,
    MODELS_DIR,
    SAMPLE_SUB_PATH,
    TARGET_COL,
    TEST_PATH,
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

CB_MODELS_DIR  = MODELS_DIR / "catboost"
META_CONFIG    = CONFIGS_DIR / "meta.yaml"
META_N_SPLITS  = 5
TOP_K_FEATURES = 100
SEED           = 42


def load_lgbm_config() -> dict:
    with open(LGBM_CONFIG, encoding="utf-8") as f:
        return yaml.safe_load(f)


def preprocess(cfg: dict, path) -> pd.DataFrame:
    df     = pd.read_csv(path)
    layout = pd.read_csv(LAYOUT_PATH)
    df = merge_layout(df, layout)
    df = build_time_features(df)
    df = build_features(df)
    df = add_scenario_aggregate_features(df)
    df = add_scenario_relative_features(df)
    df = add_scenario_trajectory_features(df)
    df = add_lead_features(df, leads=[1, 2, 3, 4, 5])
    lag_cfg = cfg["lag"]
    df = add_lag_features(
        df,
        lag_cols=lag_cfg["cols"],
        lags=lag_cfg["lags"],
        rolling_windows=lag_cfg.get("rolling_windows", [3, 5]),
    )
    return df


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


def load_xgb_models() -> tuple[list, list[str]]:
    xgb_dir = MODELS_DIR / "xgb"
    paths = sorted(xgb_dir.glob("xgb_fold*.pkl"))
    if not paths:
        raise FileNotFoundError(f"No XGB models in {xgb_dir}")
    models = []
    for p in paths:
        with open(p, "rb") as f:
            models.append(pickle.load(f))
    with open(xgb_dir / "feature_cols.pkl", "rb") as f:
        feature_cols = pickle.load(f)
    return models, feature_cols


def get_top_features(lgbm_models: list, lgbm_features: list[str], top_k: int) -> list[str]:
    """Average feature importance across all LGBM folds, return top-k names."""
    importance = np.zeros(len(lgbm_features))
    for m in lgbm_models:
        importance += m.feature_importance(importance_type="gain")
    importance /= len(lgbm_models)
    top_idx = np.argsort(importance)[::-1][:top_k]
    top_feats = [lgbm_features[i] for i in top_idx]
    print(f"Top {top_k} meta features: {top_feats[:5]} ...")
    return top_feats


def fill_missing(df: pd.DataFrame, cols: list[str]) -> None:
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan


def build_meta_train(
    train: pd.DataFrame,
    top_feats: list[str],
    use_xgb: bool = False,
) -> tuple[pd.DataFrame, np.ndarray, pd.Series]:
    """Merge OOF predictions + top original features for meta-model training."""
    oof_lgbm = pd.read_csv(DATA_PROCESSED / "oof_lgbm.csv")
    oof_cb   = pd.read_csv(DATA_PROCESSED / "oof_catboost.csv")

    merged = oof_lgbm.merge(
        oof_cb[["ID", "oof_pred"]].rename(columns={"oof_pred": "oof_cb"}),
        on="ID",
    ).rename(columns={"oof_pred": "oof_lgbm"})

    if use_xgb:
        oof_xgb = pd.read_csv(DATA_PROCESSED / "oof_xgb.csv")
        merged = merged.merge(
            oof_xgb[["ID", "oof_pred"]].rename(columns={"oof_pred": "oof_xgb"}),
            on="ID",
        )

    # Attach original features via ID
    orig_cols = ["ID", "scenario_id"] + [f for f in top_feats if f in train.columns]
    merged = merged.merge(train[orig_cols], on="ID", how="left")

    base_cols = ["oof_lgbm", "oof_cb"] + (["oof_xgb"] if use_xgb else [])
    meta_cols = base_cols + [f for f in top_feats if f in merged.columns]
    X_meta  = merged[meta_cols]
    y_meta  = merged[TARGET_COL].values
    groups  = merged["scenario_id"]
    return X_meta, y_meta, groups


def build_meta_test(
    test: pd.DataFrame,
    lgbm_models: list,
    lgbm_features: list[str],
    cb_models: list,
    cb_features: list[str],
    top_feats: list[str],
    meta_cols: list[str],
    xgb_models: list | None = None,
    xgb_features: list[str] | None = None,
) -> pd.DataFrame:
    """Build meta-features for test set."""
    fill_missing(test, lgbm_features)
    fill_missing(test, cb_features)

    lgbm_preds = np.expm1(np.mean([m.predict(test[lgbm_features]) for m in lgbm_models], axis=0))
    cb_preds   = np.expm1(np.mean([m.predict(test[cb_features])   for m in cb_models],   axis=0))

    df = pd.DataFrame({"oof_lgbm": lgbm_preds, "oof_cb": cb_preds})

    if xgb_models and xgb_features:
        fill_missing(test, xgb_features)
        xgb_preds = np.expm1(np.mean([m.predict(test[xgb_features]) for m in xgb_models], axis=0))
        df["oof_xgb"] = xgb_preds

    for f in top_feats:
        if f in test.columns:
            df[f] = test[f].values
        else:
            df[f] = np.nan

    # Ensure same column order as training
    return df[[c for c in meta_cols if c in df.columns]]


def train_meta_model(
    X_meta: pd.DataFrame,
    y_meta: np.ndarray,
    groups: pd.Series,
) -> tuple[list, float]:
    # Load tuned meta params if available, else use defaults
    if META_CONFIG.exists():
        with open(META_CONFIG, encoding="utf-8") as f:
            meta_cfg = yaml.safe_load(f)
        meta_params = meta_cfg["model"]
        print("  Using tuned meta params from meta.yaml")
    else:
        meta_params = {
            "objective":         "regression_l1",
            "learning_rate":     0.03,
            "num_leaves":        31,
            "min_child_samples": 20,
            "feature_fraction":  0.8,
            "bagging_fraction":  0.8,
            "bagging_freq":      1,
            "reg_alpha":         0.1,
            "reg_lambda":        1.0,
            "n_jobs":            -1,
            "seed":              SEED,
            "verbosity":         -1,
        }

    if gpu_available():
        meta_params = dict(meta_params)
        meta_params["device_type"] = "gpu"

    gkf = GroupKFold(n_splits=META_N_SPLITS)
    oof_meta  = np.zeros(len(X_meta))
    meta_models = []

    for fold, (tr_idx, val_idx) in enumerate(gkf.split(X_meta, y_meta, groups)):
        X_tr,  X_val = X_meta.iloc[tr_idx], X_meta.iloc[val_idx]
        y_tr,  y_val = y_meta[tr_idx],       y_meta[val_idx]

        dtrain = lgb.Dataset(X_tr, label=y_tr)
        dval   = lgb.Dataset(X_val, label=y_val, reference=dtrain)

        model = lgb.train(
            meta_params,
            dtrain,
            num_boost_round=1000,
            valid_sets=[dval],
            callbacks=[
                lgb.early_stopping(80, verbose=False),
                lgb.log_evaluation(-1),
            ],
        )
        oof_meta[val_idx] = model.predict(X_val)
        mae = mean_absolute_error(y_val, oof_meta[val_idx])
        print(f"  Meta fold {fold+1}/{META_N_SPLITS}  MAE={mae:.4f}  best_iter={model.best_iteration}")
        meta_models.append(model)

    oof_mae = mean_absolute_error(y_meta, oof_meta)
    print(f"Meta OOF MAE: {oof_mae:.4f}")
    return meta_models, oof_mae


def run() -> None:
    print("=== Stacking (LGBM + CatBoost + XGB) start ===")
    cfg = load_lgbm_config()

    print("Loading level-1 models...")
    lgbm_models, lgbm_features = load_lgbm_models()
    cb_models,   cb_features   = load_cb_models()

    use_xgb = False
    try:
        xgb_models, xgb_features = load_xgb_models()
        print(f"  LGBM x{len(lgbm_models)}, CatBoost x{len(cb_models)}, XGB x{len(xgb_models)}")
    except FileNotFoundError as e:
        print(f"  XGB models not found ({e}), using LGBM + CatBoost only")
        xgb_models, xgb_features = None, None
        use_xgb = False

    top_feats = get_top_features(lgbm_models, lgbm_features, TOP_K_FEATURES)

    print("Preprocessing train data...")
    train = preprocess(cfg, TRAIN_PATH)

    print("Building meta-features (train)...")
    X_meta, y_meta, groups = build_meta_train(train, top_feats, use_xgb=use_xgb)
    meta_cols = list(X_meta.columns)
    print(f"  Meta train shape: {X_meta.shape}  cols: {meta_cols[:5]} ...")

    print("Training meta-model...")
    meta_models, _ = train_meta_model(X_meta, y_meta, groups)

    print("Preprocessing test data...")
    test = preprocess(cfg, TEST_PATH)

    print("Building meta-features (test)...")
    X_meta_test = build_meta_test(
        test, lgbm_models, lgbm_features, cb_models, cb_features, top_feats, meta_cols,
        xgb_models=xgb_models, xgb_features=xgb_features,
    )

    print("Generating final predictions...")
    meta_preds = np.mean([m.predict(X_meta_test) for m in meta_models], axis=0)
    preds = np.clip(meta_preds, 0, None)

    submission = pd.read_csv(SAMPLE_SUB_PATH)
    id_to_pred = dict(zip(test["ID"], preds))
    submission[TARGET_COL] = submission["ID"].map(id_to_pred)
    DATA_SUBMISSION.mkdir(parents=True, exist_ok=True)
    out_path = DATA_SUBMISSION / "submission_stacking.csv"
    submission.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")
    print(submission[TARGET_COL].describe())
    print("=== Stacking done ===")


if __name__ == "__main__":
    run()
