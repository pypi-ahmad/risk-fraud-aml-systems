"""Feature engineering: amount, time-of-day, and velocity-style features.

The real dataset has no account/card identifier, so true per-account velocity
is impossible. Instead we build *stream-level* velocity proxies: how busy the
transaction stream is and how unusual an amount is relative to its recent
neighbours in time. These are honest, reproducible signals -- see the notebook
markdown for the caveat.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

RAW_V = [f"V{i}" for i in range(1, 29)]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df with engineered columns added.

    Added columns:
      Amount_log          log1p of amount (tames the heavy right tail)
      Hour                hour of day in [0,24)
      Hour_sin, Hour_cos  cyclic encoding of hour
      txn_count_60s        transactions in the preceding 60s window (velocity)
      txn_count_600s       transactions in the preceding 600s window (velocity)
      amount_roll_mean     mean amount over the last 100 transactions
      amount_z             z-score of amount vs that rolling window
    """
    out = df.copy()
    out["Amount_log"] = np.log1p(out["Amount"])

    out["Hour"] = (out["Time"] / 3600.0) % 24.0
    out["Hour_sin"] = np.sin(2 * np.pi * out["Hour"] / 24.0)
    out["Hour_cos"] = np.cos(2 * np.pi * out["Hour"] / 24.0)

    # Stream velocity: count of transactions within trailing time windows.
    # Data is time-ordered; searchsorted on Time gives O(n log n) windowing.
    t = out["Time"].to_numpy()
    order = np.argsort(t, kind="stable")
    t_sorted = t[order]
    for win in (60, 600):
        left = np.searchsorted(t_sorted, t_sorted - win, side="left")
        counts_sorted = np.arange(len(t_sorted)) - left + 1
        counts = np.empty_like(counts_sorted)
        counts[order] = counts_sorted
        out[f"txn_count_{win}s"] = counts

    # Rolling amount statistics over the last 100 transactions (by time order).
    amt_sorted = out["Amount"].to_numpy()[order]
    s = pd.Series(amt_sorted)
    roll_mean = s.rolling(100, min_periods=1).mean().to_numpy()
    roll_std = s.rolling(100, min_periods=1).std().fillna(0.0).to_numpy()
    z = (amt_sorted - roll_mean) / np.where(roll_std > 0, roll_std, 1.0)
    rm = np.empty_like(roll_mean)
    zz = np.empty_like(z)
    rm[order] = roll_mean
    zz[order] = z
    out["amount_roll_mean"] = rm
    out["amount_z"] = zz
    return out


def feature_columns(df: pd.DataFrame) -> list[str]:
    """All model input columns: PCA components + engineered features.

    'Time', 'Amount', 'Hour', and 'Class' are excluded ('Amount' is replaced
    by 'Amount_log'; 'Hour' is replaced by its cyclic encoding).
    """
    engineered = [
        "Amount_log", "Hour_sin", "Hour_cos",
        "txn_count_60s", "txn_count_600s",
        "amount_roll_mean", "amount_z",
    ]
    present_v = [c for c in RAW_V if c in df.columns]
    return present_v + [c for c in engineered if c in df.columns]
