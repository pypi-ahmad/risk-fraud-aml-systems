"""Helper package for the payment fraud risk-scoring project.

Modules
-------
data    : dataset download (Kaggle), loading, and reproducible train/val/test splits.
scoring : evaluation metrics tuned for imbalanced fraud detection, threshold
          selection, top-K review analysis, and a lightweight FraudScorer for
          inference / batch scoring.
"""

from . import data, scoring

__all__ = ["data", "scoring"]
