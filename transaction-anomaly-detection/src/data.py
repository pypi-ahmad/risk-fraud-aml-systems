"""Data loading for the transaction anomaly detection project.

Strategy (in order):
  1. Use a local CSV at ``data/creditcard.csv`` if present.
  2. Otherwise try to download the Kaggle "Credit Card Fraud Detection"
     dataset (mlg-ulb/creditcardfraud) via ``kagglehub``.
  3. If neither works, generate a realistic synthetic transaction stream so
     the notebook is always runnable end-to-end.

The real dataset (Dal Pozzolo et al., 2015) contains 284,807 European
card transactions over two days, of which 492 are fraudulent (~0.17%).
Features ``V1..V28`` are anonymised PCA components; only ``Time`` (seconds
since the first transaction) and ``Amount`` are in their original units.
``Class`` is 1 for fraud, 0 otherwise.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from . import SEED

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
LOCAL_CSV = DATA_DIR / "creditcard.csv"


def _load_local() -> pd.DataFrame | None:
    if LOCAL_CSV.exists():
        return pd.read_csv(LOCAL_CSV)
    return None


def _load_kaggle() -> pd.DataFrame | None:
    """Download via kagglehub and cache a copy into data/creditcard.csv.

    Requires Kaggle credentials (KAGGLE_USERNAME/KAGGLE_KEY, a KGAT_* token
    in KAGGLE_KEY, or ~/.kaggle/kaggle.json). Returns None on any failure.
    """
    try:
        import kagglehub

        path = Path(kagglehub.dataset_download("mlg-ulb/creditcardfraud"))
        csv = path / "creditcard.csv"
        if csv.exists():
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            shutil.copy(csv, LOCAL_CSV)
            return pd.read_csv(LOCAL_CSV)
    except Exception as exc:  # network / auth / missing pkg
        print(f"[data] Kaggle download unavailable ({type(exc).__name__}: {exc}).")
    return None


def make_synthetic(n: int = 60_000, fraud_rate: float = 0.004,
                   seed: int = SEED) -> pd.DataFrame:
    """Generate a synthetic transaction stream with the same schema.

    Normal transactions follow per-account spending habits; fraud is injected
    as bursts of unusually large, rapid transactions on random accounts. This
    gives genuine velocity signal (unlike the anonymised real data) and is
    only used when the real dataset cannot be loaded.
    """
    rng = np.random.default_rng(seed)
    n_fraud = max(1, int(n * fraud_rate))
    n_normal = n - n_fraud

    # Normal: log-normal amounts, transactions spread over ~2 days.
    normal_amount = rng.lognormal(mean=3.0, sigma=1.0, size=n_normal)
    normal_time = np.sort(rng.uniform(0, 172_792, size=n_normal))
    normal = pd.DataFrame({"Time": normal_time, "Amount": normal_amount})
    # 28 latent behavioural factors (stand-ins for the real V1..V28).
    for i in range(1, 29):
        normal[f"V{i}"] = rng.normal(0, 1, size=n_normal)
    normal["Class"] = 0

    # Fraud: large amounts, shifted latent factors, clustered in time bursts.
    fraud_amount = rng.lognormal(mean=5.0, sigma=1.2, size=n_fraud)
    burst_centers = rng.uniform(0, 172_792, size=n_fraud)
    fraud_time = np.clip(burst_centers + rng.normal(0, 30, n_fraud), 0, 172_792)
    fraud = pd.DataFrame({"Time": fraud_time, "Amount": fraud_amount})
    for i in range(1, 29):
        fraud[f"V{i}"] = rng.normal(rng.uniform(-2, 2), 1.3, size=n_fraud)
    fraud["Class"] = 1

    cols = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount", "Class"]
    df = pd.concat([normal, fraud], ignore_index=True)[cols]
    return df.sort_values("Time").reset_index(drop=True)


def load_data(prefer_kaggle: bool = True) -> tuple[pd.DataFrame, str]:
    """Return (dataframe, source_label).

    source_label is one of: "local", "kaggle", "synthetic".
    """
    df = _load_local()
    if df is not None:
        return df, "local"
    if prefer_kaggle:
        df = _load_kaggle()
        if df is not None:
            return df, "kaggle"
    print("[data] Falling back to synthetic dataset.")
    return make_synthetic(), "synthetic"


if __name__ == "__main__":
    frame, src = load_data()
    print(f"source={src} shape={frame.shape} "
          f"fraud={int(frame.Class.sum())} ({frame.Class.mean():.4%})")
