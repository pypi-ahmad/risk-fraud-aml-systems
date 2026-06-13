"""Train fraud model and persist the reusable scorer bundle."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.training import TrainingConfig, train_and_save

LOGGER = logging.getLogger("payment_fraud_training")

DEFAULT_MODEL = Path("models/fraud_scorer.joblib")
DEFAULT_METRICS = Path("reports/training_metrics.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and save payment fraud model bundle.")
    parser.add_argument(
        "--output-model",
        type=Path,
        default=DEFAULT_MODEL,
        help=f"Where to store FraudScorer artifact (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--metrics-out",
        type=Path,
        default=DEFAULT_METRICS,
        help=f"Where to write JSON metrics summary (default: {DEFAULT_METRICS}).",
    )
    parser.add_argument(
        "--data-csv",
        type=Path,
        default=None,
        help="Optional explicit CSV path. If omitted, Kaggle download/cache flow is used.",
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--threshold-beta", type=float, default=2.0)
    parser.add_argument(
        "--no-smote",
        action="store_true",
        help="Skip optional SMOTE comparison stage.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()

    config = TrainingConfig(
        random_state=args.random_state,
        test_size=args.test_size,
        val_size=args.val_size,
        threshold_beta=args.threshold_beta,
        try_smote=not args.no_smote,
    )

    result = train_and_save(
        output_model=args.output_model,
        metrics_out=args.metrics_out,
        data_csv=args.data_csv,
        config=config,
    )

    LOGGER.info("Model saved: %s", result.model_path)
    LOGGER.info("Metrics report: %s", result.metrics_path)
    LOGGER.info(
        "Best model: %s | smote=%s | threshold=%.4f",
        result.best_model_name,
        result.used_smote,
        result.threshold,
    )


if __name__ == "__main__":
    main()
