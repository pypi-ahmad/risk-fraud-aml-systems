from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


REQUIRED_COLUMNS = [
    "project_name",
    "task_type",
    "library_source",
    "model_name",
    "cv_metric_mean",
    "cv_metric_std",
    "holdout_primary_metric",
    "holdout_secondary_metric",
    "holdout_tertiary_metric",
    "calibration_metric",
    "train_time_sec",
    "infer_latency_ms",
    "p95_latency_ms",
    "model_size_mb",
    "retrain_time_sec",
    "interpretability_note",
    "rank_score",
    "final_rank",
]


def _size_mb(model: Any) -> float:
    with tempfile.NamedTemporaryFile(suffix=".joblib", delete=True) as tmp:
        joblib.dump(model, tmp.name)
        size = Path(tmp.name).stat().st_size / (1024 * 1024)
    return float(size)


def _row_template(project_name: str, task_type: str, library_source: str, model_name: str):
    return {
        "project_name": project_name,
        "task_type": task_type,
        "library_source": library_source,
        "model_name": model_name,
        "cv_metric_mean": np.nan,
        "cv_metric_std": np.nan,
        "holdout_primary_metric": np.nan,
        "holdout_secondary_metric": np.nan,
        "holdout_tertiary_metric": np.nan,
        "calibration_metric": np.nan,
        "train_time_sec": np.nan,
        "infer_latency_ms": np.nan,
        "p95_latency_ms": np.nan,
        "model_size_mb": np.nan,
        "retrain_time_sec": np.nan,
        "interpretability_note": "",
        "rank_score": np.nan,
        "final_rank": np.nan,
    }


def build_leaderboard(
    project_name: str,
    task_type: str,
    lazy_results: pd.DataFrame,
    manual_results: pd.DataFrame,
    flaml_result: dict,
    pycaret_result: dict,
    baseline_rows: list[dict] | None = None,
    manual_model_objects: dict | None = None,
):
    rows = []

    baseline_rows = baseline_rows or []
    for item in baseline_rows:
        row = _row_template(project_name, task_type, "baseline", item.get("model_name", "baseline"))
        row["holdout_primary_metric"] = item.get("primary")
        row["holdout_secondary_metric"] = item.get("secondary")
        row["holdout_tertiary_metric"] = item.get("tertiary")
        row["interpretability_note"] = item.get("note", "reference baseline")
        rows.append(row)

    for _, r in lazy_results.iterrows():
        row = _row_template(project_name, task_type, "lazypredict", r.get("family", r.get("lazy_model", "unknown")))
        row["holdout_primary_metric"] = r.get("pr_auc")
        row["holdout_secondary_metric"] = r.get("roc_auc")
        row["holdout_tertiary_metric"] = r.get("brier")
        row["calibration_metric"] = r.get("brier")
        row["train_time_sec"] = r.get("train_time_sec")
        row["interpretability_note"] = r.get("eligibility_note", "LazyPredict discovery")
        rows.append(row)

    for _, r in manual_results.iterrows():
        row = _row_template(project_name, task_type, "manual", r["model_name"])
        row["holdout_primary_metric"] = r.get("pr_auc")
        row["holdout_secondary_metric"] = r.get("roc_auc")
        row["holdout_tertiary_metric"] = r.get("brier")
        row["calibration_metric"] = r.get("calibration_metric")
        row["train_time_sec"] = r.get("train_time_sec")
        row["infer_latency_ms"] = r.get("infer_latency_ms")
        row["p95_latency_ms"] = r.get("p95_latency_ms")
        row["retrain_time_sec"] = r.get("train_time_sec")
        row["interpretability_note"] = r.get("interpretability_note", "Manual engineering track")
        if manual_model_objects and r["model_name"] in manual_model_objects:
            row["model_size_mb"] = _size_mb(manual_model_objects[r["model_name"]])
        rows.append(row)

    fr = _row_template(project_name, task_type, "flaml", flaml_result.get("model_name", "flaml_best"))
    fr["holdout_primary_metric"] = flaml_result.get("pr_auc")
    fr["holdout_secondary_metric"] = flaml_result.get("roc_auc")
    fr["holdout_tertiary_metric"] = flaml_result.get("brier")
    fr["calibration_metric"] = flaml_result.get("calibration_metric")
    fr["train_time_sec"] = flaml_result.get("train_time_sec")
    fr["infer_latency_ms"] = flaml_result.get("infer_latency_ms")
    fr["p95_latency_ms"] = flaml_result.get("p95_latency_ms")
    fr["retrain_time_sec"] = flaml_result.get("train_time_sec")
    fr["interpretability_note"] = flaml_result.get("interpretability_note", "FLAML search under budget")
    rows.append(fr)

    pr = _row_template(project_name, task_type, "pycaret", pycaret_result.get("model_name", "pycaret_best"))
    pr["holdout_primary_metric"] = pycaret_result.get("pr_auc")
    pr["holdout_secondary_metric"] = pycaret_result.get("roc_auc")
    pr["holdout_tertiary_metric"] = pycaret_result.get("brier")
    pr["calibration_metric"] = pycaret_result.get("calibration_metric")
    pr["train_time_sec"] = pycaret_result.get("train_time_sec")
    pr["infer_latency_ms"] = pycaret_result.get("infer_latency_ms")
    pr["p95_latency_ms"] = pycaret_result.get("p95_latency_ms")
    pr["interpretability_note"] = pycaret_result.get("interpretability_note", "PyCaret experiment track")
    rows.append(pr)

    df = pd.DataFrame(rows)

    primary = df["holdout_primary_metric"].fillna(0)
    secondary = df["holdout_secondary_metric"].fillna(0)
    tertiary = df["holdout_tertiary_metric"].fillna(df["holdout_tertiary_metric"].max(skipna=True) or 0)
    calibration = df["calibration_metric"].fillna(df["calibration_metric"].max(skipna=True) or 0)

    df["rank_score"] = (0.62 * primary) + (0.25 * secondary) - (0.08 * tertiary) - (0.05 * calibration)
    df = df.sort_values("rank_score", ascending=False).reset_index(drop=True)
    df["final_rank"] = np.arange(1, len(df) + 1)

    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan

    return df[REQUIRED_COLUMNS]


def rerank_with_business_weights(
    leaderboard_df: pd.DataFrame,
    primary_weight: float = 0.6,
    secondary_weight: float = 0.25,
    tertiary_penalty: float = 0.1,
    latency_penalty: float = 0.05,
):
    df = leaderboard_df.copy()
    primary = df["holdout_primary_metric"].fillna(0)
    secondary = df["holdout_secondary_metric"].fillna(0)
    tertiary = df["holdout_tertiary_metric"].fillna(df["holdout_tertiary_metric"].max(skipna=True) or 0)
    latency = df["p95_latency_ms"].fillna(df["p95_latency_ms"].median(skipna=True) or 0)

    df["rank_score"] = (
        (primary_weight * primary)
        + (secondary_weight * secondary)
        - (tertiary_penalty * tertiary)
        - (latency_penalty * (latency / (latency.max() if latency.max() else 1)))
    )
    df = df.sort_values("rank_score", ascending=False).reset_index(drop=True)
    df["final_rank"] = np.arange(1, len(df) + 1)
    return df
