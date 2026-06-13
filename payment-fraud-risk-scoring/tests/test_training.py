from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from src import training


class DummyEstimator:
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "DummyEstimator":  # noqa: N803
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:  # noqa: N803
        p = np.clip(X.iloc[:, 0].to_numpy(dtype=float), 0.0, 1.0)
        return np.column_stack((1.0 - p, p))


class TrainingTests(unittest.TestCase):
    def test_train_and_save_writes_model_hash_and_metrics(self) -> None:
        rng = np.random.default_rng(42)
        n = 200
        df = pd.DataFrame(
            {
                "f1": rng.random(n),
                "f2": rng.random(n),
                "Class": np.array([0] * 170 + [1] * 30),
            }
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            csv_path = tmp_path / "dataset.csv"
            model_path = tmp_path / "models" / "fraud_scorer.joblib"
            metrics_path = tmp_path / "reports" / "training_metrics.json"
            df.to_csv(csv_path, index=False)

            with mock.patch.object(training, "_build_candidate_models", return_value={"Dummy": DummyEstimator()}):
                result = training.train_and_save(
                    output_model=model_path,
                    metrics_out=metrics_path,
                    data_csv=csv_path,
                    config=training.TrainingConfig(try_smote=False),
                )

            self.assertTrue(model_path.exists())
            self.assertTrue(Path(f"{model_path}.sha256").exists())
            self.assertTrue(metrics_path.exists())
            self.assertEqual(result.best_model_name, "Dummy")


if __name__ == "__main__":
    unittest.main()
