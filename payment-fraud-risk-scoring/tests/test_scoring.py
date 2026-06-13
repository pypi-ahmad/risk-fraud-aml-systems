from __future__ import annotations

import math
import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from src.scoring import FraudScorer, best_threshold, evaluate, topk_review


class DummyModel:
    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        p = np.clip(frame.iloc[:, 0].to_numpy(dtype=float), 0.0, 1.0)
        return np.column_stack((1.0 - p, p))


class ScoringTests(unittest.TestCase):
    def test_evaluate_single_class_returns_nan_roc_auc(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            metrics = evaluate(y_true=[0, 0, 0], y_proba=[0.1, 0.2, 0.3], threshold=0.5)
        self.assertTrue(math.isnan(metrics["roc_auc"]))
        self.assertAlmostEqual(metrics["pr_auc"], 0.0)

    def test_best_threshold_validates_beta(self) -> None:
        with self.assertRaises(ValueError):
            best_threshold(y_true=[0, 1], y_proba=[0.1, 0.9], beta=0)

    def test_topk_review_rejects_negative_k(self) -> None:
        with self.assertRaises(ValueError):
            topk_review(y_true=[0, 1], y_proba=[0.2, 0.8], k=-1)

    def test_score_frame_empty_input_is_supported(self) -> None:
        scorer = FraudScorer(model=DummyModel(), features=["f1", "f2"], threshold=0.5)
        empty = pd.DataFrame(columns=["f1", "f2"])

        scored = scorer.score_frame(empty)

        self.assertEqual(len(scored), 0)
        for col in ("fraud_probability", "risk_score", "risk_band", "is_flagged"):
            self.assertIn(col, scored.columns)

    def test_save_writes_hash_and_load_verifies_it(self) -> None:
        scorer = FraudScorer(model=DummyModel(), features=["f1", "f2"], threshold=0.5)

        with tempfile.TemporaryDirectory() as tmp_dir:
            model_path = Path(tmp_dir) / "fraud_scorer.joblib"
            scorer.save(model_path)

            hash_path = Path(f"{model_path}.sha256")
            self.assertTrue(hash_path.exists())

            loaded = FraudScorer.load(model_path, verify_hash=True, require_hash=True)
            self.assertEqual(loaded.features, scorer.features)
            self.assertEqual(loaded.threshold, scorer.threshold)

            with model_path.open("ab") as handle:
                handle.write(b"tamper")

            with self.assertRaises(ValueError):
                FraudScorer.load(model_path, verify_hash=True, require_hash=True)


if __name__ == "__main__":
    unittest.main()
