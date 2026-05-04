# -*- coding: utf-8 -*-
"""
OOF residual analysis to find systematic error patterns.

Usage:
    python run_analysis.py
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from src.config import DATA_PROCESSED, LAYOUT_PATH, TARGET_COL, TRAIN_PATH
from src.features import build_time_features, merge_layout

pd.set_option("display.float_format", "{:.4f}".format)
pd.set_option("display.max_rows", 50)


def load_oof() -> pd.DataFrame:
    oof_lgbm = pd.read_csv(DATA_PROCESSED / "oof_lgbm.csv")
    oof_cb   = pd.read_csv(DATA_PROCESSED / "oof_catboost.csv")

    df = oof_lgbm.rename(columns={"oof_pred": "pred_lgbm"})
    df = df.merge(
        oof_cb[["ID", "oof_pred"]].rename(columns={"oof_pred": "pred_cb"}),
        on="ID",
    )
    # Ensemble prediction (use ~0.75/0.25 approximation from last run)
    df["pred_ens"] = 0.75 * df["pred_lgbm"] + 0.25 * df["pred_cb"]
    df["residual"] = df["pred_ens"] - df[TARGET_COL]  # positive = overpredict
    df["abs_error"] = df["residual"].abs()
    return df


def load_train_features() -> pd.DataFrame:
    train  = pd.read_csv(TRAIN_PATH)
    layout = pd.read_csv(LAYOUT_PATH)
    train  = merge_layout(train, layout)
    train  = build_time_features(train)
    return train


def section(title: str) -> None:
    print(f"\n{'='*55}")
    print(f"  {title}")
    print('='*55)


def run() -> None:
    print("Loading OOF predictions...")
    oof   = load_oof()
    train = load_train_features()

    df = oof.merge(
        train[["ID", "timeslot_idx", "shift_phase", "scenario_id",
               "congestion_score", "robot_utilization",
               "timeslot_ratio"]],
        on="ID",
    )

    overall_mae  = mean_absolute_error(df[TARGET_COL], df["pred_ens"])
    overall_bias = df["residual"].mean()  # positive = over, negative = under

    section("Overall")
    print(f"  OOF MAE  : {overall_mae:.4f}")
    print(f"  Mean bias: {overall_bias:.4f}  ({'overpredict' if overall_bias > 0 else 'underpredict'})")
    print(f"  Bias/MAE : {overall_bias/overall_mae*100:.1f}%")

    # --- By timeslot ---
    section("Error by timeslot_idx")
    ts = (
        df.groupby("timeslot_idx")
        .agg(mae=("abs_error", "mean"), bias=("residual", "mean"), count=("ID", "count"))
        .reset_index()
    )
    print(ts.to_string(index=False))

    # --- By shift phase ---
    section("Error by shift_phase (0=early, 1=mid, 2=late)")
    sp = (
        df.groupby("shift_phase")
        .agg(mae=("abs_error", "mean"), bias=("residual", "mean"), count=("ID", "count"))
        .reset_index()
    )
    print(sp.to_string(index=False))

    # --- By actual target bucket ---
    section("Error by actual delay bucket")
    df["target_bucket"] = pd.cut(
        df[TARGET_COL],
        bins=[0, 5, 10, 20, 40, 9999],
        labels=["0-5", "5-10", "10-20", "20-40", "40+"],
    )
    tb = (
        df.groupby("target_bucket", observed=True)
        .agg(mae=("abs_error", "mean"), bias=("residual", "mean"), count=("ID", "count"))
        .reset_index()
    )
    print(tb.to_string(index=False))

    # --- By congestion level ---
    section("Error by congestion_score bucket")
    df["cong_bucket"] = pd.qcut(df["congestion_score"], q=5, labels=["Q1","Q2","Q3","Q4","Q5"], duplicates="drop")
    cb = (
        df.groupby("cong_bucket", observed=True)
        .agg(mae=("abs_error", "mean"), bias=("residual", "mean"), count=("ID", "count"))
        .reset_index()
    )
    print(cb.to_string(index=False))

    # --- Quantile distribution ---
    section("Prediction vs Actual quantiles")
    q = [0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
    quant = pd.DataFrame({
        "quantile": q,
        "actual":   df[TARGET_COL].quantile(q).values,
        "pred_ens": df["pred_ens"].quantile(q).values,
    })
    quant["diff"] = quant["pred_ens"] - quant["actual"]
    print(quant.to_string(index=False))

    section("Summary")
    print("Key patterns to look for:")
    print("  - bias >> 0: systematic overprediction")
    print("  - bias << 0: systematic underprediction")
    print("  - MAE spikes at specific timeslots or buckets = model blind spot")
    print("  - Quantile diff growing at high values = underestimates extreme delays")


if __name__ == "__main__":
    run()
