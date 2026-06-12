"""Transaction anomaly detection helper package.

Modules:
    data      -- load the Kaggle credit-card dataset (or a synthetic fallback)
    features  -- amount / time / velocity-style feature engineering
    models    -- build the unsupervised detectors and extract anomaly scores
    metrics   -- ranking metrics (PR-AUC, ROC-AUC, Precision@K, Recall@K)
"""

SEED = 42
