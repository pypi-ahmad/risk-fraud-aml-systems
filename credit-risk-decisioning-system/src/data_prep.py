from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


CORE_TABLES = [
    "application_train",
    "bureau",
    "previous_application",
    "POS_CASH_balance",
    "installments_payments",
    "credit_card_balance",
]


def _safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def load_home_credit_data(
    raw_dir: Path,
    sample_frac: float | None = None,
    random_state: int = 42,
) -> Dict[str, pd.DataFrame]:
    """Load Home Credit source tables. Optional sampling is applied on application_train."""
    raw_dir = Path(raw_dir)
    tables: Dict[str, pd.DataFrame] = {}
    for name in CORE_TABLES:
        tables[name] = _safe_read_csv(raw_dir / f"{name}.csv")

    app = tables["application_train"]
    if app.empty:
        raise FileNotFoundError(
            f"application_train.csv not found under {raw_dir}. "
            "Run scripts/download_data.sh first."
        )

    if sample_frac is not None and 0 < sample_frac < 1:
        sampled = app.sample(frac=sample_frac, random_state=random_state).copy()
        tables["application_train"] = sampled
        keep_ids = set(sampled["SK_ID_CURR"].tolist())
        for key in ("bureau", "previous_application"):
            df = tables.get(key, pd.DataFrame())
            if not df.empty and "SK_ID_CURR" in df.columns:
                tables[key] = df[df["SK_ID_CURR"].isin(keep_ids)].copy()

    return tables


def _aggregate_numeric(df: pd.DataFrame, group_key: str, prefix: str) -> pd.DataFrame:
    if df.empty or group_key not in df.columns:
        return pd.DataFrame()

    numeric_cols = [
        c
        for c in df.columns
        if c != group_key and pd.api.types.is_numeric_dtype(df[c])
    ]
    if not numeric_cols:
        return pd.DataFrame()

    # Keep a manageable subset for notebook runtime and memory stability.
    numeric_cols = numeric_cols[:40]
    agg = df.groupby(group_key)[numeric_cols].agg(["mean", "max", "min", "std"])
    agg.columns = [f"{prefix}_{c}_{s}" for c, s in agg.columns]
    return agg.reset_index()


def _attach_prev_linked_table(
    base_prev: pd.DataFrame,
    linked_df: pd.DataFrame,
    prefix: str,
) -> pd.DataFrame:
    if base_prev.empty or linked_df.empty:
        return pd.DataFrame()
    if "SK_ID_PREV" not in linked_df.columns or "SK_ID_PREV" not in base_prev.columns:
        return pd.DataFrame()

    mapper = base_prev[["SK_ID_PREV", "SK_ID_CURR"]].drop_duplicates()
    enriched = linked_df.merge(mapper, on="SK_ID_PREV", how="inner")
    return _aggregate_numeric(enriched, group_key="SK_ID_CURR", prefix=prefix)


def build_customer_level_table(tables: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Build customer-level modeling table from application and auxiliary tables."""
    app = tables["application_train"].copy()
    if "TARGET" not in app.columns:
        raise ValueError("application_train must include TARGET.")

    merged = app.copy()

    bureau_agg = _aggregate_numeric(tables.get("bureau", pd.DataFrame()), "SK_ID_CURR", "bureau")
    if not bureau_agg.empty:
        merged = merged.merge(bureau_agg, on="SK_ID_CURR", how="left")

    prev = tables.get("previous_application", pd.DataFrame())
    prev_agg = _aggregate_numeric(prev, "SK_ID_CURR", "prev")
    if not prev_agg.empty:
        merged = merged.merge(prev_agg, on="SK_ID_CURR", how="left")

    pos_agg = _attach_prev_linked_table(prev, tables.get("POS_CASH_balance", pd.DataFrame()), "pos")
    if not pos_agg.empty:
        merged = merged.merge(pos_agg, on="SK_ID_CURR", how="left")

    inst_agg = _attach_prev_linked_table(prev, tables.get("installments_payments", pd.DataFrame()), "inst")
    if not inst_agg.empty:
        merged = merged.merge(inst_agg, on="SK_ID_CURR", how="left")

    cc_agg = _attach_prev_linked_table(prev, tables.get("credit_card_balance", pd.DataFrame()), "cc")
    if not cc_agg.empty:
        merged = merged.merge(cc_agg, on="SK_ID_CURR", how="left")

    merged = merged.replace([np.inf, -np.inf], np.nan)
    return merged


def temporal_leakage_checks(df: pd.DataFrame) -> pd.DataFrame:
    """Basic leakage and quality guards before modeling."""
    work = df.copy()

    leakage_cols = [
        c
        for c in work.columns
        if c.lower().startswith("target") and c != "TARGET"
    ]
    id_like_drop = [c for c in ("SK_ID_PREV", "SK_ID_BUREAU") if c in work.columns]

    drop_cols = leakage_cols + id_like_drop
    if drop_cols:
        work = work.drop(columns=drop_cols, errors="ignore")

    # Drop near-empty columns for stability.
    missing_ratio = work.isna().mean()
    very_sparse = missing_ratio[missing_ratio > 0.995].index.tolist()
    if very_sparse:
        work = work.drop(columns=very_sparse, errors="ignore")

    return work


def stratified_split(
    df: pd.DataFrame,
    target_col: str = "TARGET",
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df, holdout_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df[target_col],
        random_state=random_state,
    )
    return train_df.reset_index(drop=True), holdout_df.reset_index(drop=True)
