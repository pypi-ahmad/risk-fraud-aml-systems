"""Ranking-oriented evaluation for anomaly detection.

When labels are available we treat the anomaly score as a ranking and ask:
how well does it surface the rare positive (fraud) class? PR-AUC is the
headline metric under extreme imbalance; ROC-AUC is reported for context but
is optimistic when positives are rare. Precision@K / Recall@K reflect the
realistic 'analysts can only review K alerts' operating point.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def precision_recall_at_k(y_true: np.ndarray, scores: np.ndarray,
                          k: int) -> tuple[float, float]:
    """Precision@K and Recall@K for the top-K highest-scoring transactions."""
    k = min(k, len(scores))
    top_idx = np.argsort(scores)[::-1][:k]
    hits = int(y_true[top_idx].sum())
    total_pos = int(y_true.sum())
    precision = hits / k if k else 0.0
    recall = hits / total_pos if total_pos else 0.0
    return precision, recall


def evaluate(y_true: np.ndarray, scores: np.ndarray,
             k: int | None = None) -> dict:
    """Return a metrics dict for one model's scores.

    k defaults to the number of true positives (a natural 'budget == #fraud'
    operating point).
    """
    y_true = np.asarray(y_true)
    if k is None:
        k = int(y_true.sum())
    p_at_k, r_at_k = precision_recall_at_k(y_true, scores, k)
    return {
        "PR_AUC": average_precision_score(y_true, scores),
        "ROC_AUC": roc_auc_score(y_true, scores),
        f"Precision@{k}": p_at_k,
        f"Recall@{k}": r_at_k,
    }


def comparison_table(results: dict[str, dict]) -> pd.DataFrame:
    """results: {model_name: metrics_dict} -> tidy comparison DataFrame."""
    return (pd.DataFrame(results).T
            .sort_values("PR_AUC", ascending=False)
            .round(4))
