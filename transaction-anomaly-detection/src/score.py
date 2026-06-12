"""Lightweight batch scoring: rank transactions by anomaly score.

Trains an Isolation Forest (the recommended detector) on a CSV of transactions
and writes the rows back out sorted from most to least anomalous, with an
``anomaly_score`` column (higher == more anomalous) and an ``alert`` flag for
the top-K. Designed to be run from the command line:

    uv run python -m src.score --input data/creditcard.csv \
        --output outputs/ranked.csv --top-k 100

If --input is omitted it uses the project loader (local CSV / Kaggle / synthetic).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import RobustScaler

from . import SEED
from .data import load_data
from .features import add_features, feature_columns
from .models import build_detectors, anomaly_scores, normalize_scores


def run(input_path: str | None, output_path: str, top_k: int,
        contamination: float) -> pd.DataFrame:
    if input_path:
        df = pd.read_csv(input_path)
        print(f"[score] loaded {len(df):,} rows from {input_path}")
    else:
        df, src = load_data()
        print(f"[score] loaded {len(df):,} rows from source={src}")

    feat = add_features(df)
    cols = feature_columns(feat)
    X = RobustScaler().fit_transform(feat[cols])

    iforest, _ = build_detectors(contamination=contamination, seed=SEED)["IsolationForest"]
    scores = anomaly_scores("IsolationForest", iforest, X)

    df = df.copy()
    df["anomaly_score"] = normalize_scores(scores)
    df = df.sort_values("anomaly_score", ascending=False).reset_index(drop=True)
    df["alert"] = 0
    df.loc[: top_k - 1, "alert"] = 1

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[score] wrote ranked output -> {output_path} (top_k={top_k} flagged)")

    if "Class" in df.columns:
        hits = int(df.loc[df["alert"] == 1, "Class"].sum())
        print(f"[score] of top {top_k} alerts, {hits} are labelled fraud "
              f"(precision@{top_k}={hits / top_k:.3f})")
    return df


def main() -> None:
    p = argparse.ArgumentParser(description="Batch anomaly scoring for transactions.")
    p.add_argument("--input", default=None, help="CSV path; omit to use the project loader")
    p.add_argument("--output", default="outputs/ranked_transactions.csv")
    p.add_argument("--top-k", type=int, default=100, help="how many top rows to flag as alerts")
    p.add_argument("--contamination", type=float, default=0.01)
    args = p.parse_args()
    run(args.input, args.output, args.top_k, args.contamination)


if __name__ == "__main__":
    main()
