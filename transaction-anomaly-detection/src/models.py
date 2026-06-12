"""Unsupervised anomaly detectors and a unified scoring interface.

Convention used everywhere in this project: a higher score == more anomalous.
scikit-learn's detectors do the opposite (higher decision_function == more
normal), so we negate as needed and expose ``anomaly_scores``.

One-Class SVM note: the exact kernel SVM is O(n^2) and does not scale to
hundreds of thousands of rows. We use a Nystroem kernel approximation feeding
an SGD-based one-class SVM, which scales linearly and is the modern sklearn
recipe for large data.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import SGDOneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import make_pipeline

from . import SEED


@dataclass
class Detector:
    name: str
    blurb: str  # one-line tradeoff note for the notebook


def build_detectors(contamination: float = 0.01, seed: int = SEED) -> dict:
    """Return {key: (estimator, Detector-meta)} for the three methods."""
    iforest = IsolationForest(
        n_estimators=200, contamination=contamination,
        random_state=seed, n_jobs=-1,
    )
    lof = LocalOutlierFactor(
        n_neighbors=20, contamination=contamination, novelty=False, n_jobs=-1,
    )
    # Nystroem gamma=None defaults to 1/n_features (RBF kernel width).
    ocsvm = make_pipeline(
        Nystroem(gamma=None, n_components=200, random_state=seed),
        SGDOneClassSVM(nu=contamination, random_state=seed),
    )
    return {
        "IsolationForest": (iforest, Detector(
            "Isolation Forest",
            "Isolates points via random splits; fast, scales well, few "
            "assumptions. Can miss anomalies in dense local regions.")),
        "LocalOutlierFactor": (lof, Detector(
            "Local Outlier Factor",
            "Density-based: flags points in lower-density regions than their "
            "neighbours. Great for local anomalies; O(n*k) and no clean "
            "out-of-sample scoring when novelty=False.")),
        "OneClassSVM": (ocsvm, Detector(
            "One-Class SVM (Nystroem + SGD)",
            "Learns a boundary around the dense 'normal' region. Flexible but "
            "sensitive to nu/gamma; exact kernel version doesn't scale, hence "
            "the kernel approximation.")),
    }


def anomaly_scores(key: str, estimator, X) -> np.ndarray:
    """Fit (where needed) and return anomaly scores, higher == more anomalous.

    LOF (novelty=False) only exposes scores for the training set via
    ``negative_outlier_factor_`` after ``fit_predict``.
    """
    if key == "LocalOutlierFactor":
        estimator.fit_predict(X)
        return -estimator.negative_outlier_factor_
    estimator.fit(X)
    # decision_function: higher == more normal -> negate for anomaly score.
    return -estimator.decision_function(X)


def normalize_scores(scores: np.ndarray) -> np.ndarray:
    """Rank-normalise scores into [0, 1] for cross-model comparison/threshold.

    Rank normalisation is robust to the very different raw score scales the
    three detectors produce.
    """
    order = scores.argsort()
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(scores))
    return ranks / max(len(scores) - 1, 1)
