# -*- coding: utf-8 -*-
"""
OOF 잔차 분석 — 모델이 어디서 틀리는지 파악.

Usage:
    python run_error_analysis.py
"""

import numpy as np
import pandas as pd
import yaml

from src.config import DATA_PROCESSED, LAYOUT_PATH, LGBM_CONFIG, TARGET_COL, TRAIN_PATH

BINS = [0, 5, 10, 20, 40, 80, 200, np.inf]
BIN_LABELS = ["0-5", "5-10", "10-20", "20-40", "40-80", "80-200", "200+"]


def load_oof() -> pd.DataFrame:
    lgbm = pd.read_csv(DATA_PROCESSED / "oof_lgbm.csv")
    cb   = pd.read_csv(DATA_PROCESSED / "oof_catboost.csv")
    df = lgbm.rename(columns={"oof_pred": "pred_lgbm"})
    df["pred_cb"] = cb["oof_pred"]
    df["pred_blend"] = (df["pred_lgbm"] + df["pred_cb"]) / 2
    return df


def section(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


def run() -> None:
    oof    = load_oof()
    train  = pd.read_csv(TRAIN_PATH)
    layout = pd.read_csv(LAYOUT_PATH)

    with open(LGBM_CONFIG, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # scenario_id, timeslot 정보 병합
    train["timeslot_idx"] = train.groupby("scenario_id").cumcount()
    oof = oof.merge(train[["ID", "scenario_id", "timeslot_idx", "layout_id"]], on="ID")
    oof = oof.merge(layout[["layout_id", "layout_type"]], on="layout_id", how="left")

    y    = oof[TARGET_COL].values
    pred = oof["pred_blend"].values
    res  = pred - y  # 양수 = 과예측, 음수 = 과소예측

    # ── 1. 전체 요약 ─────────────────────────────────────────────────
    section("1. 전체 잔차 요약")
    print(f"  MAE        : {np.mean(np.abs(res)):.4f}")
    print(f"  Bias(mean) : {np.mean(res):+.4f}  (양수=과예측, 음수=과소예측)")
    print(f"  RMSE       : {np.sqrt(np.mean(res**2)):.4f}")
    print(f"  과소예측 비율: {(res < 0).mean()*100:.1f}%")
    print(f"  과대예측 비율: {(res > 0).mean()*100:.1f}%")

    # ── 2. 실제값 구간별 MAE & Bias ──────────────────────────────────
    section("2. 실제값 구간별 MAE / Bias")
    oof["y_bin"] = pd.cut(oof[TARGET_COL], bins=BINS, labels=BIN_LABELS, right=False)
    print(f"  {'구간':>8}  {'샘플수':>7}  {'MAE':>8}  {'Bias':>8}  {'과소%':>6}")
    for lab in BIN_LABELS:
        mask = oof["y_bin"] == lab
        if mask.sum() == 0:
            continue
        r = res[mask.values]
        mae_  = np.mean(np.abs(r))
        bias_ = np.mean(r)
        under = (r < 0).mean() * 100
        print(f"  {lab:>8}  {mask.sum():>7}  {mae_:>8.3f}  {bias_:>+8.3f}  {under:>5.1f}%")

    # ── 3. timeslot 위치별 MAE ────────────────────────────────────────
    section("3. 시나리오 내 timeslot 위치별 MAE")
    ts_stats = oof.groupby("timeslot_idx").apply(
        lambda g: pd.Series({
            "n":    len(g),
            "mae":  np.mean(np.abs(g["pred_blend"] - g[TARGET_COL])),
            "bias": np.mean(g["pred_blend"] - g[TARGET_COL]),
        })
    )
    print(f"  {'slot':>5}  {'n':>7}  {'MAE':>8}  {'Bias':>8}")
    for idx, row in ts_stats.iterrows():
        print(f"  {int(idx):>5}  {int(row['n']):>7}  {row['mae']:>8.3f}  {row['bias']:>+8.3f}")

    # ── 4. layout_type별 MAE ─────────────────────────────────────────
    section("4. layout_type별 MAE")
    lt_stats = oof.groupby("layout_type").apply(
        lambda g: pd.Series({
            "n":    len(g),
            "mae":  np.mean(np.abs(g["pred_blend"] - g[TARGET_COL])),
            "bias": np.mean(g["pred_blend"] - g[TARGET_COL]),
            "target_mean": g[TARGET_COL].mean(),
        })
    )
    print(f"  {'type':>10}  {'n':>7}  {'MAE':>8}  {'Bias':>8}  {'실제평균':>8}")
    for ltype, row in lt_stats.iterrows():
        print(f"  {str(ltype):>10}  {int(row['n']):>7}  {row['mae']:>8.3f}  {row['bias']:>+8.3f}  {row['target_mean']:>8.3f}")

    # ── 5. 잔차 큰 시나리오 top-20 ──────────────────────────────────
    section("5. 잔차 큰 시나리오 Top-20 (scenario MAE 기준)")
    scen_stats = oof.groupby("scenario_id").apply(
        lambda g: pd.Series({
            "mae":         np.mean(np.abs(g["pred_blend"] - g[TARGET_COL])),
            "bias":        np.mean(g["pred_blend"] - g[TARGET_COL]),
            "target_mean": g[TARGET_COL].mean(),
            "target_max":  g[TARGET_COL].max(),
            "layout_type": g["layout_type"].iloc[0],
        })
    ).sort_values("mae", ascending=False)

    print(f"  {'scenario':>12}  {'MAE':>8}  {'Bias':>8}  {'실제평균':>8}  {'실제최대':>8}  {'layout':>8}")
    for sid, row in scen_stats.head(20).iterrows():
        print(f"  {str(sid):>12}  {row['mae']:>8.3f}  {row['bias']:>+8.3f}  {row['target_mean']:>8.3f}  {row['target_max']:>8.3f}  {row['layout_type']:>8}")

    # ── 6. 핵심 인사이트 요약 ─────────────────────────────────────────
    section("6. 핵심 인사이트")
    high_mask = oof[TARGET_COL] >= 40
    high_bias = np.mean(res[high_mask.values])
    low_mask  = oof[TARGET_COL] < 10
    low_bias  = np.mean(res[low_mask.values])
    early_mask = oof["timeslot_idx"] <= 5
    late_mask  = oof["timeslot_idx"] >= 19
    early_mae = np.mean(np.abs(res[early_mask.values]))
    late_mae  = np.mean(np.abs(res[late_mask.values]))

    print(f"  고지연(≥40) bias  : {high_bias:+.4f}  {'← 과소예측' if high_bias < 0 else '← 과대예측'}")
    print(f"  저지연(<10) bias  : {low_bias:+.4f}  {'← 과소예측' if low_bias < 0 else '← 과대예측'}")
    print(f"  초반 슬롯(0-5) MAE : {early_mae:.4f}")
    print(f"  후반 슬롯(19-24) MAE: {late_mae:.4f}")


if __name__ == "__main__":
    run()
