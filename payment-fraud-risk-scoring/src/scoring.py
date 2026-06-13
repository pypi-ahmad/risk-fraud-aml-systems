"""Evaluation, threshold selection, and inference helpers for fraud scoring.

Why these metrics (and not accuracy)
------------------------------------
With a 0.173% fraud rate, a model that predicts "never fraud" is 99.83%
accurate and 100% useless. We therefore judge models on signal that ignores the
overwhelming negative class:

* **PR-AUC** (average precision) — area under the precision/recall curve. It
  summarises ranking quality across all thresholds and, unlike ROC-AUC, is not
  flattered by the huge true-negative count. This is our primary model-selection
  metric.
* **Recall** — fraction of actual fraud we catch. Missed fraud is the expensive
  error, so recall is the headline operational number.
* **Precision** — fraction of flagged transactions that are truly fraud, i.e.
  how much analyst review time is wasted on false alarms.
* **F1** — the precision/recall balance at a chosen threshold.

Because precision and recall trade off, the *operating threshold* is a business
choice, not a fixed 0.5. The helpers below make that choice explicit.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

LOGGER = logging.getLogger(__name__)


def _as_1d_array(values: Iterable[Any], *, name: str) -> np.ndarray:
    """Convert an iterable into a validated 1D ndarray."""
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be 1-dimensional.")
    return array


def _validate_binary_targets(y_true: np.ndarray) -> None:
    """Ensure target labels are binary {0, 1} with at least one row."""
    if y_true.size == 0:
        raise ValueError("y_true cannot be empty.")
    unique = np.unique(y_true)
    if not np.isin(unique, [0, 1]).all():
        raise ValueError("y_true must contain only binary labels {0, 1}.")


def _ensure_same_size(y_true: np.ndarray, y_proba: np.ndarray) -> None:
    if y_true.shape[0] != y_proba.shape[0]:
        raise ValueError("y_true and y_proba must have the same length.")


def _hash_sidecar_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".sha256")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_expected_hash(hash_path: Path) -> str:
    raw = hash_path.read_text(encoding="utf-8").strip()
    if not raw:
        raise ValueError(f"Hash sidecar is empty: {hash_path}")
    return raw.split()[0]


# --- Metrics -----------------------------------------------------------------
def evaluate(
    y_true: Iterable[int],
    y_proba: Iterable[float],
    threshold: float = 0.5,
) -> dict[str, float | int]:
    """Return a dict of fraud-relevant metrics at a given decision threshold.

    ``y_proba`` is the predicted probability of the positive (fraud) class.
    PR-AUC and ROC-AUC are threshold-independent; the rest depend on
    ``threshold``.
    """
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1.")

    y_true = _as_1d_array(y_true, name="y_true").astype(int, copy=False)
    y_proba = _as_1d_array(y_proba, name="y_proba").astype(float, copy=False)
    _validate_binary_targets(y_true)
    _ensure_same_size(y_true, y_proba)

    y_pred = (y_proba >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    roc_auc = float("nan")
    if np.unique(y_true).size >= 2:
        roc_auc = float(roc_auc_score(y_true, y_proba))

    return {
        "threshold": float(threshold),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "roc_auc": roc_auc,
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


def best_threshold(
    y_true: Iterable[int],
    y_proba: Iterable[float],
    beta: float = 1.0,
) -> tuple[float, float]:
    """Pick the threshold that maximises F-beta over the PR curve.

    ``beta > 1`` weights recall more heavily than precision — appropriate for
    fraud, where catching more fraud usually matters more than analyst workload.
    Returns ``(threshold, fbeta_at_that_threshold)``.
    """
    if beta <= 0:
        raise ValueError("beta must be > 0.")

    y_true = _as_1d_array(y_true, name="y_true").astype(int, copy=False)
    y_proba = _as_1d_array(y_proba, name="y_proba").astype(float, copy=False)
    _validate_binary_targets(y_true)
    _ensure_same_size(y_true, y_proba)

    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    if thresholds.size == 0:
        return 1.0, 0.0

    # precision_recall_curve returns one fewer threshold than precision/recall.
    precision, recall = precision[:-1], recall[:-1]

    denom = (beta**2 * precision) + recall
    with np.errstate(divide="ignore", invalid="ignore"):
        fbeta = np.where(denom > 0, (1 + beta**2) * precision * recall / denom, 0.0)
    if fbeta.size == 0:
        return 1.0, 0.0
    best = int(np.argmax(fbeta))
    return float(thresholds[best]), float(fbeta[best])


def threshold_for_recall(
    y_true: Iterable[int],
    y_proba: Iterable[float],
    target_recall: float,
) -> float:
    """Lowest-precision-cost threshold that still achieves ``target_recall``.

    Useful for SLA-style requirements such as "we must catch at least 85% of
    fraud". Among all thresholds meeting the recall target we keep the highest
    one (best precision). Falls back to 0.0 if the target is unreachable.
    """
    if not 0.0 <= target_recall <= 1.0:
        raise ValueError("target_recall must be between 0 and 1.")

    y_true = _as_1d_array(y_true, name="y_true").astype(int, copy=False)
    y_proba = _as_1d_array(y_proba, name="y_proba").astype(float, copy=False)
    _validate_binary_targets(y_true)
    _ensure_same_size(y_true, y_proba)

    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    if thresholds.size == 0:
        return 0.0

    recall, thresholds = recall[:-1], thresholds

    feasible = thresholds[recall >= target_recall]
    return float(feasible.max()) if feasible.size else 0.0


# --- Top-K review analysis ---------------------------------------------------
def topk_review(
    y_true: Iterable[int],
    y_proba: Iterable[float],
    k: int,
) -> dict[str, float | int]:
    """Simulate sending the ``k`` highest-risk transactions to manual review.

    Mirrors how a fraud team actually operates under a fixed daily review
    budget. Returns precision@k (hit rate of the review queue) and recall@k
    (share of all fraud that the queue captures).
    """
    if k < 0:
        raise ValueError("k must be non-negative.")

    y_true = _as_1d_array(y_true, name="y_true").astype(int, copy=False)
    y_proba = _as_1d_array(y_proba, name="y_proba").astype(float, copy=False)
    _validate_binary_targets(y_true)
    _ensure_same_size(y_true, y_proba)

    k = min(k, len(y_true))
    if k == 0:
        total_fraud = int(y_true.sum())
        return {
            "k": 0,
            "frauds_caught": 0,
            "precision_at_k": 0.0,
            "recall_at_k": 0.0 if total_fraud else 0.0,
        }

    top_idx = np.argsort(y_proba)[::-1][:k]
    caught = int(y_true[top_idx].sum())
    total_fraud = int(y_true.sum())
    return {
        "k": k,
        "frauds_caught": caught,
        "precision_at_k": caught / k if k else 0.0,
        "recall_at_k": caught / total_fraud if total_fraud else 0.0,
    }


# --- Risk scoring ------------------------------------------------------------
def risk_score(y_proba: Iterable[float], scale: int = 1000) -> np.ndarray:
    """Map fraud probabilities to integer risk scores in ``[0, scale]``."""
    if scale <= 0:
        raise ValueError("scale must be > 0.")
    proba = np.clip(_as_1d_array(y_proba, name="y_proba").astype(float, copy=False), 0.0, 1.0)
    return np.rint(proba * scale).astype(int)


def risk_band(
    y_proba: Iterable[float],
    low: float = 0.20,
    high: float = 0.60,
) -> np.ndarray:
    """Bucket probabilities into Low / Medium / High risk bands."""
    if not (0.0 <= low < high <= 1.0):
        raise ValueError("Expected 0 <= low < high <= 1.")
    p = np.clip(_as_1d_array(y_proba, name="y_proba").astype(float, copy=False), 0.0, 1.0)
    bands = np.where(p >= high, "High", np.where(p >= low, "Medium", "Low"))
    return bands


# --- Lightweight inference bundle -------------------------------------------
@dataclass
class FraudScorer:
    """A self-contained, serialisable fraud scorer.

    Bundles the fitted estimator with the exact feature order it was trained on
    and the chosen operating threshold, so scoring new data is unambiguous and
    reproducible. Persist with :meth:`save` / reload with :meth:`load`.
    """

    model: object
    features: list[str]
    threshold: float
    metadata: dict[str, Any] | None = None

    def validate(self) -> None:
        """Validate scorer invariants used in inference."""
        if not self.features:
            raise ValueError("FraudScorer.features must not be empty.")
        if len(set(self.features)) != len(self.features):
            raise ValueError("FraudScorer.features contains duplicates.")
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("FraudScorer.threshold must be between 0 and 1.")
        if not hasattr(self.model, "predict_proba"):
            raise TypeError("FraudScorer.model must expose predict_proba().")

    def score_frame(self, df: pd.DataFrame, *, copy_input: bool = True) -> pd.DataFrame:
        """Score a DataFrame of raw transactions.

        Returns a new frame with ``fraud_probability``, integer ``risk_score``,
        ``risk_band`` and the binary ``is_flagged`` decision at the bundled
        threshold. Input must contain (at least) the training feature columns.
        """
        self.validate()
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame.")

        missing = [c for c in self.features if c not in df.columns]
        if missing:
            raise ValueError(f"Input is missing required feature columns: {missing}")

        out = df.copy() if copy_input else df
        if df.empty:
            out["fraud_probability"] = np.array([], dtype=float)
            out["risk_score"] = np.array([], dtype=int)
            out["risk_band"] = np.array([], dtype=object)
            out["is_flagged"] = np.array([], dtype=int)
            return out

        feature_frame = df[self.features]
        if feature_frame.isnull().to_numpy().any():
            raise ValueError("Input contains null values in required feature columns.")

        try:
            proba_matrix = np.asarray(self.model.predict_proba(feature_frame), dtype=float)
        except Exception as exc:  # pragma: no cover - defensive wrapper
            raise ValueError("Model inference failed on the provided frame.") from exc

        if proba_matrix.ndim != 2 or proba_matrix.shape[1] < 2:
            raise ValueError("Model predict_proba output must have shape (n_rows, 2+).")

        proba = proba_matrix[:, 1]
        if proba.shape[0] != len(df):
            raise ValueError("Model returned a probability vector with unexpected length.")

        out["fraud_probability"] = proba
        out["risk_score"] = risk_score(proba)
        out["risk_band"] = risk_band(proba)
        out["is_flagged"] = (proba >= self.threshold).astype(int)
        return out

    def save(self, path: str | Path) -> Path:
        """Persist the scorer and write a SHA-256 sidecar for integrity checks."""
        import joblib

        self.validate()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        digest = _sha256_file(path)
        _hash_sidecar_path(path).write_text(f"{digest}  {path.name}\n", encoding="utf-8")
        return path

    @staticmethod
    def load(
        path: str | Path,
        *,
        verify_hash: bool = True,
        require_hash: bool = False,
    ) -> "FraudScorer":
        """Load a scorer, optionally validating against a SHA-256 sidecar."""
        import joblib

        scorer_path = Path(path)
        hash_path = _hash_sidecar_path(scorer_path)
        if verify_hash or require_hash:
            if hash_path.exists():
                expected = _read_expected_hash(hash_path)
                actual = _sha256_file(scorer_path)
                if actual != expected:
                    raise ValueError(
                        f"Model hash mismatch for {scorer_path}. "
                        f"Expected {expected}, got {actual}."
                    )
            elif require_hash:
                raise FileNotFoundError(
                    f"Missing model hash sidecar at {hash_path}. "
                    "Refuse to load unsigned model artifact."
                )
            elif verify_hash:
                LOGGER.warning(
                    "No model hash sidecar found at %s. Loading without integrity verification.",
                    hash_path,
                )

        scorer = joblib.load(scorer_path)
        if not isinstance(scorer, FraudScorer):
            raise TypeError(f"Expected FraudScorer artifact, got {type(scorer)!r}.")
        scorer.validate()
        return scorer
