# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np


def build_layout_target_encoding(
    train: pd.DataFrame, target_col: str
) -> pd.DataFrame:
    """Compute per-layout target statistics (mean/std/median)."""
    enc = (
        train.groupby("layout_id")[target_col]
        .agg(
            layout_target_mean="mean",
            layout_target_std="std",
            layout_target_median="median",
        )
        .reset_index()
    )
    return enc


def merge_layout_target_encoding(
    df: pd.DataFrame, enc: pd.DataFrame
) -> pd.DataFrame:
    """Merge layout target encoding by layout_id."""
    return df.merge(enc, on="layout_id", how="left")


def merge_layout(df: pd.DataFrame, layout: pd.DataFrame) -> pd.DataFrame:
    """Merge layout_info by layout_id."""
    layout = layout.copy()
    layout["layout_type_code"] = layout["layout_type"].astype("category").cat.codes
    layout = layout.drop(columns=["layout_type"])
    return df.merge(layout, on="layout_id", how="left")


def build_time_features(df: pd.DataFrame, group_col: str = "scenario_id") -> pd.DataFrame:
    """Build timeslot index (0-24) and time-based derived features."""
    df = df.copy()
    df["timeslot_idx"] = df.groupby(group_col).cumcount()
    df["shift_phase"] = (df["timeslot_idx"] // 8).astype(int)
    df["is_late_scenario"] = (df["timeslot_idx"] >= 20).astype(int)
    # Normalized position within the scenario
    df["timeslot_ratio"] = df["timeslot_idx"] / 24.0
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build derived features."""
    df = df.copy()

    # Robot utilization features
    df["total_robots"]          = df["robot_active"] + df["robot_idle"] + df["robot_charging"]
    df["robots_busy_ratio"]     = df["robot_active"]   / (df["total_robots"] + 1e-6)
    df["robots_charging_ratio"] = df["robot_charging"] / (df["total_robots"] + 1e-6)
    df["charger_util"]          = df["robot_charging"] / (df["charger_count"] + 1e-6)
    df["charge_pressure"]       = df["charge_queue_length"] * df["avg_charge_wait"]
    df["battery_risk"]          = df["low_battery_ratio"]   * df["charge_queue_length"]
    df["robot_slack"]           = df["robot_idle"] / (df["total_robots"] + 1e-6)
    df["effective_robots"]      = df["robot_active"] * df["agv_task_success_rate"]

    # Order / processing load features
    df["order_per_robot"]        = df["order_inflow_15m"] / (df["robot_active"] + 1e-6)
    df["urgent_order_volume"]    = df["order_inflow_15m"] * df["urgent_order_ratio"]
    df["heavy_order_volume"]     = df["order_inflow_15m"] * df["heavy_item_ratio"]
    df["pack_load"]              = df["pack_utilization"] * df["order_inflow_15m"]
    df["pick_complexity"]        = df["pick_list_length_avg"] * df["unique_sku_15m"]
    df["order_per_pack_station"] = df["order_inflow_15m"] / (df["pack_station_count"] + 1e-6)
    df["express_order_pressure"] = df["express_lane_util"] * df["urgent_order_ratio"]

    # Congestion / blocking index features
    df["congestion_blocking"]    = df["congestion_score"]        * df["blocked_path_15m"]
    df["traffic_risk"]           = df["aisle_traffic_score"]     * df["max_zone_density"]
    df["intersection_pressure"]  = df["intersection_wait_time_avg"] * df["intersection_count"]
    df["fault_recovery_load"]    = df["fault_count_15m"] * df["avg_recovery_time"]

    # KPI inverse features
    df["kpi_miss_ratio"]         = 1.0 - df["kpi_otd_pct"] / 100.0
    df["agv_fail_ratio"]         = 1.0 - df["agv_task_success_rate"]
    df["sort_error_ratio"]       = 1.0 - df["sort_accuracy_pct"] / 100.0

    # Layout / space density features
    df["robot_density"]          = df["total_robots"]     / (df["floor_area_sqm"] + 1e-6)
    df["dock_pressure"]          = df["loading_dock_util"] * df["outbound_truck_wait_min"]
    df["area_per_robot"]         = df["floor_area_sqm"]   / (df["total_robots"] + 1e-6)

    # Composite delay risk indices
    df["delay_risk_index"]       = (
        df["robot_utilization"] * df["congestion_score"] * df["pack_utilization"]
    )
    df["system_health"]          = (
        df["agv_task_success_rate"] * df["barcode_read_success_rate"] *
        df["path_optimization_score"]
    )
    df["operational_stress"]     = (
        df["order_per_robot"] * df["congestion_score"] * (1 - df["path_optimization_score"])
    )

    # Environment / maintenance features
    df["temp_humidity_idx"]      = df["warehouse_temp_avg"] * df["humidity_pct"] / 100.0
    df["maintenance_risk"]       = df["robot_firmware_update_days"] * (1 - df["maintenance_schedule_score"])

    return df


# EDA: scenario-level aggregates correlate with target at 0.48~0.53
# Valid for test.csv since all 25 slots are provided simultaneously
_SCENARIO_AGG_COLS = [
    "low_battery_ratio",    # corr 0.51 (max), 0.50 (std)
    "battery_mean",         # corr 0.47 (std)
    "order_inflow_15m",     # corr 0.48 (mean/max)
    "congestion_score",     # corr 0.47 (max), 0.45 (std)
    "max_zone_density",     # corr 0.49 (max/std/mean)
    "charge_queue_length",  # corr 0.48 (std)
    "blocked_path_15m",     # corr 0.47 (std)
    "robot_utilization",    # corr 0.53 (std), 0.49 (max)
    "robot_idle",           # corr 0.43 (mean)
]


def add_scenario_aggregate_features(
    df: pd.DataFrame, group_col: str = "scenario_id"
) -> pd.DataFrame:
    """
    Scenario-level aggregate features (mean/max/std across all 25 slots).
    Not leakage: all 25 slots are provided simultaneously in train and test.
    """
    agg_dict: dict[str, pd.Series] = {}
    for col in _SCENARIO_AGG_COLS:
        if col not in df.columns:
            continue
        g = df.groupby(group_col)[col]
        agg_dict[f"{col}_scen_mean"] = g.transform("mean")
        agg_dict[f"{col}_scen_max"]  = g.transform("max")
        agg_dict[f"{col}_scen_std"]  = g.transform("std")

    if agg_dict:
        df = pd.concat([df, pd.DataFrame(agg_dict, index=df.index)], axis=1)
    return df


def add_scenario_relative_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Relative features: current slot value vs scenario aggregates.
    Must be called after add_scenario_aggregate_features.
    """
    pairs = [
        ("congestion_score",    "congestion_score_scen_mean",   "congestion_score_scen_max"),
        ("battery_mean",        "battery_mean_scen_mean",       "battery_mean_scen_max"),
        ("order_inflow_15m",    "order_inflow_15m_scen_mean",   "order_inflow_15m_scen_max"),
        ("low_battery_ratio",   "low_battery_ratio_scen_mean",  "low_battery_ratio_scen_max"),
        ("max_zone_density",    "max_zone_density_scen_mean",   "max_zone_density_scen_max"),
        ("charge_queue_length", "charge_queue_length_scen_mean","charge_queue_length_scen_max"),
        ("robot_utilization",   "robot_utilization_scen_mean",  "robot_utilization_scen_max"),
        ("blocked_path_15m",    "blocked_path_15m_scen_mean",   "blocked_path_15m_scen_max"),
    ]
    new_cols: dict[str, pd.Series] = {}
    for col, mean_col, max_col in pairs:
        if col not in df.columns:
            continue
        new_cols[f"{col}_vs_scen_mean"] = df[col] / (df[mean_col].abs() + 1e-6)
        new_cols[f"{col}_dev_scen_mean"] = df[col] - df[mean_col]
        new_cols[f"{col}_vs_scen_max"]  = df[col] / (df[max_col].abs() + 1e-6)
    if new_cols:
        df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
    return df


_PIVOT_COLS = [
    "congestion_score",
    "delay_risk_index",
    "robot_utilization",
    "order_inflow_15m",
    "charge_queue_length",
    "blocked_path_15m",
    "battery_mean",
]


def add_scenario_pivot_features(
    df: pd.DataFrame, group_col: str = "scenario_id"
) -> pd.DataFrame:
    """
    Pivot all 25 timeslot values for top columns into separate features.
    Each row gets access to every slot's exact value in the scenario.
    Not leakage: all 25 slots are provided simultaneously in train and test.
    """
    df = df.sort_values([group_col, "timeslot_idx"]).reset_index(drop=True)
    new_cols: dict[str, pd.Series] = {}

    for col in _PIVOT_COLS:
        if col not in df.columns:
            continue
        pivoted = (
            df.pivot_table(index=group_col, columns="timeslot_idx", values=col)
        )
        pivoted.columns = [f"{col}_slot{int(c):02d}" for c in pivoted.columns]
        merged = df[[group_col]].merge(pivoted, on=group_col, how="left")
        for feat_col in pivoted.columns:
            new_cols[feat_col] = merged[feat_col].values

    if new_cols:
        df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
    return df


_TRAJECTORY_COLS = [
    "congestion_score",
    "delay_risk_index",
    "robot_utilization",
    "order_inflow_15m",
    "charge_queue_length",
    "blocked_path_15m",
    "battery_mean",
    "pack_utilization",
    "kpi_miss_ratio",
]


def add_scenario_trajectory_features(
    df: pd.DataFrame, group_col: str = "scenario_id"
) -> pd.DataFrame:
    """
    Scenario trajectory features:
    - slope: linear trend across 25 timeslots (rising or falling)
    - peak_timeslot: which slot has the max value
    - dist_to_peak: current timeslot distance from peak
    - current_rank: rank of current value within scenario (0=lowest, 1=highest)
    """
    df = df.sort_values([group_col, "timeslot_idx"]).reset_index(drop=True)
    new_cols: dict[str, pd.Series] = {}

    for col in _TRAJECTORY_COLS:
        if col not in df.columns:
            continue

        g = df.groupby(group_col)

        # slope: (last - first) / 24 - direction of change across scenario
        first_val = g[col].transform("first")
        last_val  = g[col].transform("last")
        new_cols[f"{col}_slope"] = (last_val - first_val) / 24.0

        # peak_timeslot: timeslot index where max occurs
        def peak_ts(x):
            return pd.Series(
                np.full(len(x), x.values.argmax()), index=x.index
            )
        new_cols[f"{col}_peak_ts"] = g[col].transform(peak_ts)

        # dist_to_peak: current timeslot - peak timeslot (negative = peak ahead)
        new_cols[f"{col}_dist_to_peak"] = (
            df["timeslot_idx"] - new_cols[f"{col}_peak_ts"]
        )

        # current_rank: percentile rank within scenario (0~1)
        new_cols[f"{col}_scen_rank"] = g[col].rank(pct=True)

    if new_cols:
        df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
    return df


# EDA: lead features correlate more with target than lag0 for these columns
_LEAD_COLS = [
    "order_inflow_15m",
    "robot_utilization",
    "congestion_score",
    "pack_utilization",
    "charge_queue_length",
    "battery_mean",
    "blocked_path_15m",
    "fault_count_15m",
    "task_reassign_15m",
    "delay_risk_index",
    "order_per_robot",
    "pack_load",
    "congestion_blocking",
    "system_health",
    "kpi_miss_ratio",
    "aisle_traffic_score",
    "loading_dock_util",
]


def add_lead_features(
    df: pd.DataFrame,
    leads: list[int] = [1, 2],
    group_col: str = "scenario_id",
) -> pd.DataFrame:
    """
    Add future slot features (not leakage: all 25 slots provided simultaneously).
    shift(-n) = value n slots ahead of current slot.
    """
    new_cols: dict[str, pd.Series] = {}
    for col in _LEAD_COLS:
        if col not in df.columns:
            continue
        for n in leads:
            new_cols[f"{col}_lead{n}"] = df.groupby(group_col)[col].shift(-n)
    if new_cols:
        df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
    return df


def add_lag_features(
    df: pd.DataFrame,
    lag_cols: list[str],
    lags: list[int] = [1, 2, 3],
    rolling_windows: list[int] = [3, 5],
    group_col: str = "scenario_id",
) -> pd.DataFrame:
    """Add lag / rolling / diff features within each scenario."""
    df = df.sort_values([group_col, "timeslot_idx"]).reset_index(drop=True)
    new_cols: dict[str, pd.Series] = {}
    for col in lag_cols:
        if col not in df.columns:
            continue
        grouped = df.groupby(group_col)[col]
        for lag in lags:
            new_cols[f"{col}_lag{lag}"] = grouped.shift(lag)
        lag1 = grouped.shift(1)
        new_cols[f"{col}_diff1"] = df[col] - lag1
        for window in rolling_windows:
            new_cols[f"{col}_roll{window}_mean"] = grouped.transform(
                lambda x: x.shift(1).rolling(window, min_periods=1).mean()
            )
        new_cols[f"{col}_roll{rolling_windows[-1]}_max"] = grouped.transform(
            lambda x: x.shift(1).rolling(rolling_windows[-1], min_periods=1).max()
        )
    if new_cols:
        df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
    return df


def get_feature_cols(df: pd.DataFrame, drop_cols: list[str]) -> list[str]:
    return [c for c in df.columns if c not in drop_cols]
