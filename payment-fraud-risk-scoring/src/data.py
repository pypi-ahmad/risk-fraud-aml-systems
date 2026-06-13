"""Dataset acquisition, loading, and reproducible splitting.

The dataset is the public *Credit Card Fraud Detection* set from the Machine
Learning Group at ULB (Kaggle slug ``mlg-ulb/creditcardfraud``). It contains
284,807 European card transactions from two days in September 2013, of which
only 492 (0.173%) are fraudulent: a textbook severe class imbalance.

Features ``V1``-``V28`` are the output of a PCA transformation applied by the
publishers to protect confidentiality, so they are already numeric and roughly
centred. Only ``Time`` (seconds since the first transaction) and ``Amount`` are
in their original units. ``Class`` is the target: 1 = fraud, 0 = legitimate.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

LOGGER = logging.getLogger(__name__)

# --- Constants ---------------------------------------------------------------
KAGGLE_DATASET = "mlg-ulb/creditcardfraud"
CSV_NAME = "creditcard.csv"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
RAW_CSV = RAW_DIR / CSV_NAME

TARGET = "Class"
RANDOM_STATE = 42


# --- Acquisition -------------------------------------------------------------
def ensure_dataset(force: bool = False) -> Path:
    """Return local path to ``creditcard.csv``, downloading it if needed.

    If the file is already present under ``data/raw/`` it is reused. Otherwise it
    is fetched from Kaggle via ``kagglehub`` and copied into the project so that
    subsequent runs are offline and reproducible.
    """
    if RAW_CSV.exists() and not force:
        return RAW_CSV

    try:
        import kagglehub  # imported lazily so loading cached data needs no network
    except Exception as exc:  # pragma: no cover - import failure path
        raise RuntimeError(
            "kagglehub is required for automatic dataset download. Install dependencies via `uv sync`."
        ) from exc

    try:
        cache_dir = Path(kagglehub.dataset_download(KAGGLE_DATASET))
    except Exception as exc:  # pragma: no cover - network/auth dependent
        raise RuntimeError(
            "Failed to download dataset from Kaggle. "
            "Verify Kaggle credentials and network access."
        ) from exc

    source_csv = cache_dir / CSV_NAME
    if not source_csv.exists():
        matches = list(cache_dir.rglob(CSV_NAME))
        if not matches:
            raise FileNotFoundError(f"{CSV_NAME} not found under {cache_dir}")
        source_csv = matches[0]

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_csv, RAW_CSV)
    LOGGER.info("Dataset copied to %s", RAW_CSV)
    return RAW_CSV


# --- Loading -----------------------------------------------------------------
def load_data(drop_duplicates: bool = True, *, csv_path: Path | None = None) -> pd.DataFrame:
    """Load the dataset as a DataFrame.

    The raw file contains fully duplicated rows. Duplicates are dropped by
    default: leaving them in risks leaking identical records across the
    train/test boundary and inflating scores.
    """
    resolved_path = csv_path or ensure_dataset()
    df = pd.read_csv(resolved_path)
    if TARGET not in df.columns:
        raise ValueError(f"Expected target column `{TARGET}` in dataset.")

    if drop_duplicates:
        df = df.drop_duplicates().reset_index(drop=True)
    return df


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Return all columns except the target, preserving original order."""
    if TARGET not in df.columns:
        raise ValueError(f"Expected target column `{TARGET}` in input frame.")
    return [column for column in df.columns if column != TARGET]


# --- Splitting ---------------------------------------------------------------
def make_splits(
    df: pd.DataFrame,
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Create a stratified train / validation / test split.

    Stratification on ``Class`` keeps the tiny fraud proportion approximately
    stable across all three folds.

    Returns:
        Tuple ``(X_train, X_val, X_test, y_train, y_val, y_test)``.
    """
    if not (0.0 < test_size < 1.0):
        raise ValueError("test_size must be between 0 and 1.")
    if not (0.0 < val_size < 1.0):
        raise ValueError("val_size must be between 0 and 1.")
    if test_size + val_size >= 1.0:
        raise ValueError("test_size + val_size must be < 1.0.")

    X = df[feature_columns(df)]
    y = df[TARGET]

    if y.nunique() < 2:
        raise ValueError("make_splits requires at least two target classes.")

    X_rest, X_test, y_rest, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    # val_size is fraction of full dataset, so rescale relative to remainder.
    val_relative = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_rest,
        y_rest,
        test_size=val_relative,
        stratify=y_rest,
        random_state=random_state,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test
