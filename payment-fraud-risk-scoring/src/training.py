"""Reusable training pipeline for payment fraud risk scoring.

This module extracts model training and selection logic from the notebook into
importable, testable source code.
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from . import data as dataio
from . import scoring

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainingConfig:
    """Configuration for model training and selection."""

    random_state: int = dataio.RANDOM_STATE
    test_size: float = 0.2
    val_size: float = 0.2
    threshold_beta: float = 2.0
    try_smote: bool = True


@dataclass(frozen=True)
class TrainingResult:
    """Returned training artifacts and metrics summary."""

    model_path: Path
    metrics_path: Path | None
    best_model_name: str
    used_smote: bool
    threshold: float
    validation: dict[str, Any]
    test: dict[str, Any]


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _build_candidate_models(scale_pos_weight: float, random_state: int) -> dict[str, Any]:
    return {
        "LogReg": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced",
            n_jobs=-1,
            random_state=random_state,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=400,
            learning_rate=0.1,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",
            eval_metric="aucpr",
            scale_pos_weight=scale_pos_weight,
            random_state=random_state,
            n_jobs=-1,
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
            verbose=-1,
        ),
    }


def _smote_baseline_for(best_name: str, random_state: int) -> Any:
    if best_name == "XGBoost":
        base = XGBClassifier(
            n_estimators=400,
            learning_rate=0.1,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",
            eval_metric="aucpr",
            random_state=random_state,
            n_jobs=-1,
        )
    elif best_name == "LightGBM":
        base = LGBMClassifier(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            n_jobs=-1,
            verbose=-1,
        )
    elif best_name == "RandomForest":
        base = RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=random_state,
        )
    else:
        base = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=2000, random_state=random_state)),
            ]
        )

    return ImbPipeline(
        [
            ("smote", SMOTE(random_state=random_state)),
            ("clf", clone(base)),
        ]
    )


def _to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _to_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def train_and_save(
    *,
    output_model: Path,
    metrics_out: Path | None = None,
    data_csv: Path | None = None,
    config: TrainingConfig = TrainingConfig(),
) -> TrainingResult:
    """Train candidate models, select best policy, and persist scorer artifact."""
    _set_seed(config.random_state)

    LOGGER.info("Loading dataset...")
    df = dataio.load_data(drop_duplicates=True, csv_path=data_csv)
    X_train, X_val, X_test, y_train, y_val, y_test = dataio.make_splits(
        df,
        test_size=config.test_size,
        val_size=config.val_size,
        random_state=config.random_state,
    )

    features = list(X_train.columns)
    positives = int((y_train == 1).sum())
    negatives = int((y_train == 0).sum())
    if positives == 0:
        raise ValueError("Training fold has no positive class samples.")

    scale_pos_weight = negatives / positives
    LOGGER.info("Training candidate models with scale_pos_weight=%.3f", scale_pos_weight)

    candidates = _build_candidate_models(scale_pos_weight, config.random_state)
    results_val: dict[str, dict[str, float | int]] = {}
    proba_val: dict[str, np.ndarray] = {}
    fitted: dict[str, Any] = {}

    for name, model in candidates.items():
        LOGGER.info("Fitting %s...", name)
        model.fit(X_train, y_train)
        p_val = np.asarray(model.predict_proba(X_val)[:, 1], dtype=float)
        proba_val[name] = p_val
        fitted[name] = model
        results_val[name] = scoring.evaluate(y_val, p_val, threshold=0.5)

    best_name = max(results_val, key=lambda model_name: float(results_val[model_name]["pr_auc"]))
    best_model = fitted[best_name]
    best_p_val = proba_val[best_name]

    threshold, fbeta = scoring.best_threshold(y_val, best_p_val, beta=config.threshold_beta)
    use_smote = False

    if config.try_smote:
        LOGGER.info("Evaluating optional SMOTE variant for %s...", best_name)
        smote_model = _smote_baseline_for(best_name, config.random_state)
        smote_model.fit(X_train, y_train)
        p_smote = np.asarray(smote_model.predict_proba(X_val)[:, 1], dtype=float)

        pr_auc_weighted = float(results_val[best_name]["pr_auc"])
        pr_auc_smote = float(scoring.evaluate(y_val, p_smote, threshold=0.5)["pr_auc"])
        if pr_auc_smote > pr_auc_weighted:
            use_smote = True
            best_model = smote_model
            best_p_val = p_smote
            threshold, fbeta = scoring.best_threshold(y_val, best_p_val, beta=config.threshold_beta)

    test_proba = np.asarray(best_model.predict_proba(X_test)[:, 1], dtype=float)
    test_metrics = scoring.evaluate(y_test, test_proba, threshold=threshold)
    topk = [scoring.topk_review(y_test, test_proba, k) for k in (50, 100, 200, 500, 1000)]

    metadata = {
        "model_name": best_name + (" + SMOTE" if use_smote else ""),
        "selection_metric": "PR-AUC",
        "validation_pr_auc": round(float(results_val[best_name]["pr_auc"]), 6),
        "validation_roc_auc": (
            None
            if np.isnan(float(results_val[best_name]["roc_auc"]))
            else round(float(results_val[best_name]["roc_auc"]), 6)
        ),
        "test_pr_auc": round(float(test_metrics["pr_auc"]), 6),
        "test_recall": round(float(test_metrics["recall"]), 6),
        "test_precision": round(float(test_metrics["precision"]), 6),
        "random_state": config.random_state,
        "trained_at_utc": datetime.now(UTC).isoformat(),
    }

    scorer_bundle = scoring.FraudScorer(
        model=best_model,
        features=features,
        threshold=float(threshold),
        metadata=metadata,
    )

    output_model.parent.mkdir(parents=True, exist_ok=True)
    model_path = scorer_bundle.save(output_model)

    summary: dict[str, Any] = {
        "best_model_name": best_name,
        "used_smote": use_smote,
        "threshold": float(threshold),
        "threshold_fbeta": float(fbeta),
        "validation": results_val,
        "test": test_metrics,
        "topk_test": topk,
        "metadata": metadata,
        "model_path": str(model_path),
    }

    metrics_path: Path | None = None
    if metrics_out is not None:
        metrics_out.parent.mkdir(parents=True, exist_ok=True)
        metrics_out.write_text(json.dumps(_to_builtin(summary), indent=2), encoding="utf-8")
        metrics_path = metrics_out

    LOGGER.info(
        "Training complete. best=%s smote=%s threshold=%.4f test_pr_auc=%.4f",
        best_name,
        use_smote,
        float(threshold),
        float(test_metrics["pr_auc"]),
    )

    return TrainingResult(
        model_path=model_path,
        metrics_path=metrics_path,
        best_model_name=best_name,
        used_smote=use_smote,
        threshold=float(threshold),
        validation=_to_builtin(results_val),
        test=_to_builtin(test_metrics),
    )
