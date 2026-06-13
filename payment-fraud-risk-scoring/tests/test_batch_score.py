from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from batch_score import _render_summary, score_csv
from src.scoring import FraudScorer


class DummyModel:
    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        p = np.clip(frame.iloc[:, 0].to_numpy(dtype=float), 0.0, 1.0)
        return np.column_stack((1.0 - p, p))


class BatchScoreTests(unittest.TestCase):
    def test_score_csv_outputs_scored_columns(self) -> None:
        scorer = FraudScorer(
            model=DummyModel(),
            features=["f1", "f2"],
            threshold=0.5,
            metadata={"model_name": "dummy"},
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            input_path = Path(tmp_dir) / "input.csv"
            output_path = Path(tmp_dir) / "output.csv"
            pd.DataFrame({"f1": [0.2, 0.9], "f2": [1.0, 2.0]}).to_csv(input_path, index=False)

            total_rows, total_flagged = score_csv(input_path, output_path, scorer)

            self.assertEqual(total_rows, 2)
            self.assertEqual(total_flagged, 1)
            scored = pd.read_csv(output_path)
            self.assertIn("fraud_probability", scored.columns)
            self.assertIn("risk_score", scored.columns)
            self.assertIn("risk_band", scored.columns)
            self.assertIn("is_flagged", scored.columns)

    def test_score_csv_handles_empty_file(self) -> None:
        scorer = FraudScorer(model=DummyModel(), features=["f1", "f2"], threshold=0.5)

        with tempfile.TemporaryDirectory() as tmp_dir:
            input_path = Path(tmp_dir) / "input_empty.csv"
            output_path = Path(tmp_dir) / "output_empty.csv"
            pd.DataFrame(columns=["f1", "f2"]).to_csv(input_path, index=False)

            total_rows, total_flagged = score_csv(input_path, output_path, scorer)

            self.assertEqual(total_rows, 0)
            self.assertEqual(total_flagged, 0)
            scored = pd.read_csv(output_path)
            self.assertIn("fraud_probability", scored.columns)
            self.assertEqual(len(scored), 0)

    def test_summary_handles_missing_metadata(self) -> None:
        scorer = FraudScorer(model=DummyModel(), features=["f1"], threshold=0.5, metadata=None)
        lines = _render_summary(scorer, total_rows=0, total_flagged=0, out_path=Path("out.csv"))
        self.assertIn("unknown", lines[0])


if __name__ == "__main__":
    unittest.main()
