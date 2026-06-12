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

from dataclasses import dataclass
from pathlib import Path

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


# --- Metrics -----------------------------------------------------------------
def evaluate(y_true, y_proba, threshold: float = 0.5) -> dict:
    """Return a dict of fraud-relevant metrics at a given decision threshold.

    ``y_proba`` is the predicted probability of the positive (fraud) class.
    PR-AUC and ROC-AUC are threshold-independent; the rest depend on
    ``threshold``.
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    y_pred = (y_proba >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


def best_threshold(y_true, y_proba, beta: float = 1.0) -> tuple[float, float]:
    """Pick the threshold that maximises F-beta over the PR curve.

    ``beta > 1`` weights recall more heavily than precision — appropriate for
    fraud, where catching more fraud usually matters more than analyst workload.
    Returns ``(threshold, fbeta_at_that_threshold)``.
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    # precision_recall_curve returns one fewer threshold than precision/recall.
    precision, recall = precision[:-1], recall[:-1]

    denom = (beta**2 * precision) + recall
    fbeta = np.where(denom > 0, (1 + beta**2) * precision * recall / denom, 0.0)
    best = int(np.argmax(fbeta))
    return float(thresholds[best]), float(fbeta[best])


def threshold_for_recall(y_true, y_proba, target_recall: float) -> float:
    """Lowest-precision-cost threshold that still achieves ``target_recall``.

    Useful for SLA-style requirements such as "we must catch at least 85% of
    fraud". Among all thresholds meeting the recall target we keep the highest
    one (best precision). Falls back to 0.0 if the target is unreachable.
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    recall, thresholds = recall[:-1], thresholds

    feasible = thresholds[recall >= target_recall]
    return float(feasible.max()) if feasible.size else 0.0


# --- Top-K review analysis ---------------------------------------------------
def topk_review(y_true, y_proba, k: int) -> dict:
    """Simulate sending the ``k`` highest-risk transactions to manual review.

    Mirrors how a fraud team actually operates under a fixed daily review
    budget. Returns precision@k (hit rate of the review queue) and recall@k
    (share of all fraud that the queue captures).
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    k = min(k, len(y_true))
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
def risk_score(y_proba, scale: int = 1000) -> np.ndarray:
    """Map fraud probabilities to integer risk scores in ``[0, scale]``."""
    return np.round(np.asarray(y_proba) * scale).astype(int)


def risk_band(y_proba, low: float = 0.20, high: float = 0.60) -> np.ndarray:
    """Bucket probabilities into Low / Medium / High risk bands."""
    p = np.asarray(y_proba)
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
    metadata: dict | None = None

    def score_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        """Score a DataFrame of raw transactions.

        Returns a new frame with ``fraud_probability``, integer ``risk_score``,
        ``risk_band`` and the binary ``is_flagged`` decision at the bundled
        threshold. Input must contain (at least) the training feature columns.
        """
        missing = [c for c in self.features if c not in df.columns]
        if missing:
            raise ValueError(f"Input is missing required feature columns: {missing}")

        proba = self.model.predict_proba(df[self.features])[:, 1]
        out = df.copy()
        out["fraud_probability"] = proba
        out["risk_score"] = risk_score(proba)
        out["risk_band"] = risk_band(proba)
        out["is_flagged"] = (proba >= self.threshold).astype(int)
        return out

    def save(self, path: str | Path) -> Path:
        import joblib

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        return path

    @staticmethod
    def load(path: str | Path) -> "FraudScorer":
        import joblib

        return joblib.load(path)
