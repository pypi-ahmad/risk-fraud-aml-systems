from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover - optional import
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
except Exception:  # pragma: no cover - optional import
    LGBMClassifier = None

try:
    from catboost import CatBoostClassifier
except Exception:  # pragma: no cover - optional import
    CatBoostClassifier = None


FAMILY_ALIASES = {
    "LogisticRegression": "logistic_regression",
    "LinearDiscriminantAnalysis": "logistic_regression",
    "RandomForestClassifier": "random_forest",
    "ExtraTreesClassifier": "extra_trees",
    "XGBClassifier": "xgboost",
    "LGBMClassifier": "lightgbm",
    "CatBoostClassifier": "catboost",
}


def _metrics(y_true, scores):
    return {
        "pr_auc": float(average_precision_score(y_true, scores)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "brier": float(brier_score_loss(y_true, scores)),
    }


def _measure_latency(model, X, n_rows: int = 300):
    n = min(n_rows, X.shape[0])
    if n == 0:
        return 0.0, 0.0
    times = []
    for i in range(n):
        row = X[i : i + 1]
        t0 = time.perf_counter()
        _ = model.predict_proba(row)
        times.append((time.perf_counter() - t0) * 1000)
    arr = np.array(times)
    return float(arr.mean()), float(np.percentile(arr, 95))


def make_estimator(family: str, random_state: int = 42):
    if family == "logistic_regression":
        return LogisticRegression(
            max_iter=2500,
            class_weight="balanced",
            random_state=random_state,
        )
    if family == "random_forest":
        return RandomForestClassifier(
            n_estimators=500,
            max_depth=10,
            min_samples_leaf=25,
            class_weight="balanced_subsample",
            random_state=random_state,
            n_jobs=-1,
        )
    if family == "extra_trees":
        return ExtraTreesClassifier(
            n_estimators=600,
            max_depth=12,
            min_samples_leaf=20,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        )
    if family == "xgboost" and XGBClassifier is not None:
        return XGBClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=2.0,
            eval_metric="aucpr",
            random_state=random_state,
            n_jobs=-1,
        )
    if family == "lightgbm" and LGBMClassifier is not None:
        return LGBMClassifier(
            n_estimators=600,
            learning_rate=0.04,
            num_leaves=63,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary",
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        )
    if family == "catboost" and CatBoostClassifier is not None:
        return CatBoostClassifier(
            depth=7,
            learning_rate=0.05,
            iterations=700,
            loss_function="Logloss",
            eval_metric="PRAUC",
            random_seed=random_state,
            verbose=False,
        )
    raise ValueError(f"Unsupported or unavailable family: {family}")


def run_lazypredict_discovery(X_train, X_valid, y_train, y_valid) -> pd.DataFrame:
    from lazypredict.Supervised import LazyClassifier

    train_n = min(6000, X_train.shape[0])
    valid_n = min(2500, X_valid.shape[0])
    if train_n < X_train.shape[0]:
        idx = np.random.default_rng(42).choice(X_train.shape[0], size=train_n, replace=False)
        X_train_lazy = X_train[idx]
        y_train_lazy = y_train.iloc[idx] if hasattr(y_train, "iloc") else y_train[idx]
    else:
        X_train_lazy, y_train_lazy = X_train, y_train

    if valid_n < X_valid.shape[0]:
        idxv = np.random.default_rng(43).choice(X_valid.shape[0], size=valid_n, replace=False)
        X_valid_lazy = X_valid[idxv]
        y_valid_lazy = y_valid.iloc[idxv] if hasattr(y_valid, "iloc") else y_valid[idxv]
    else:
        X_valid_lazy, y_valid_lazy = X_valid, y_valid

    lazy = LazyClassifier(verbose=0, ignore_warnings=True)
    models, _ = lazy.fit(X_train_lazy, X_valid_lazy, y_train_lazy, y_valid_lazy)
    table = models.reset_index().rename(columns={"index": "Model"})
    if "Model" not in table.columns:
        table = table.rename(columns={table.columns[0]: "Model"})

    keep_cols = [c for c in ["Model", "Accuracy", "Balanced Accuracy", "ROC AUC", "F1 Score", "Time Taken"] if c in table.columns]
    return table[keep_cols].sort_values(by=(["ROC AUC"] if "ROC AUC" in table.columns else [keep_cols[1]]), ascending=False).reset_index(drop=True)


def select_top3_eligible_families(
    lazy_table: pd.DataFrame,
    X_train,
    y_train,
    X_valid,
    y_valid,
    random_state: int = 42,
) -> tuple[pd.DataFrame, list[str]]:
    order_col = "ROC AUC" if "ROC AUC" in lazy_table.columns else lazy_table.columns[1]
    ranked = lazy_table.sort_values(order_col, ascending=False)

    chosen = []
    rows = []
    for _, row in ranked.iterrows():
        model_name = row["Model"]
        family = FAMILY_ALIASES.get(model_name)
        if family is None or family in chosen:
            continue
        try:
            est = make_estimator(family, random_state=random_state)
        except Exception:
            continue

        t0 = time.perf_counter()
        est.fit(X_train, y_train)
        train_time = time.perf_counter() - t0
        scores = est.predict_proba(X_valid)[:, 1]
        metric = _metrics(y_valid, scores)

        interpretability = {
            "logistic_regression": "high",
            "random_forest": "medium",
            "extra_trees": "medium",
            "xgboost": "medium-low",
            "lightgbm": "medium-low",
            "catboost": "medium-low",
        }.get(family, "medium")

        stable = metric["brier"] < 0.25
        eligible = stable and train_time < 600
        rows.append(
            {
                "lazy_model": model_name,
                "family": family,
                "pr_auc": metric["pr_auc"],
                "roc_auc": metric["roc_auc"],
                "brier": metric["brier"],
                "train_time_sec": train_time,
                "interpretability": interpretability,
                "eligible": eligible,
                "eligibility_note": "eligible" if eligible else "filtered: stability/speed",
            }
        )

        if eligible:
            chosen.append(family)
        if len(chosen) >= 3:
            break

    eval_table = pd.DataFrame(rows).sort_values(
        ["pr_auc", "roc_auc", "brier"], ascending=[False, False, True]
    )
    return eval_table.reset_index(drop=True), chosen[:3]


def optimize_operating_threshold(y_true, scores):
    thresholds = np.linspace(0.1, 0.9, 81)
    best_threshold = 0.5
    best_utility = -1e9
    for thr in thresholds:
        pred = (scores >= thr).astype(int)
        tp = ((pred == 1) & (y_true == 1)).sum()
        fp = ((pred == 1) & (y_true == 0)).sum()
        fn = ((pred == 0) & (y_true == 1)).sum()
        utility = (7 * tp) - (2 * fp) - (5 * fn)
        if utility > best_utility:
            best_utility = utility
            best_threshold = float(thr)
    return best_threshold, float(best_utility)


def assign_decision_band(
    scores,
    approve_threshold: float = 0.25,
    reject_threshold: float = 0.65,
):
    bands = []
    for s in scores:
        if s < approve_threshold:
            bands.append("approve")
        elif s >= reject_threshold:
            bands.append("reject")
        else:
            bands.append("manual_review")
    return np.array(bands)


def _calibrate_if_needed(family: str, estimator, X_train, y_train):
    if family in {"logistic_regression"}:
        estimator.fit(X_train, y_train)
        return estimator

    calibrated = CalibratedClassifierCV(estimator=estimator, method="sigmoid", cv=3)
    calibrated.fit(X_train, y_train)
    return calibrated


def run_manual_engineering_track(
    top3_families: list[str],
    X_train,
    y_train,
    X_holdout,
    y_holdout,
    random_state: int = 42,
):
    rows = []
    artifact_map: Dict[str, Any] = {}

    for family in top3_families:
        estimator = make_estimator(family, random_state=random_state)

        t0 = time.perf_counter()
        final_model = _calibrate_if_needed(family, estimator, X_train, y_train)
        train_time = time.perf_counter() - t0

        holdout_scores = final_model.predict_proba(X_holdout)[:, 1]
        m = _metrics(y_holdout, holdout_scores)
        thr, utility = optimize_operating_threshold(y_holdout.values, holdout_scores)
        mean_latency, p95_latency = _measure_latency(final_model, X_holdout)

        rows.append(
            {
                "model_name": family,
                "library_source": "manual",
                "pr_auc": m["pr_auc"],
                "roc_auc": m["roc_auc"],
                "brier": m["brier"],
                "train_time_sec": train_time,
                "infer_latency_ms": mean_latency,
                "p95_latency_ms": p95_latency,
                "optimized_threshold": thr,
                "policy_utility": utility,
                "holdout_scores": holdout_scores,
                "calibration_metric": m["brier"],
                "interpretability_note": f"Manual {family} with explicit thresholds and calibration",
            }
        )
        artifact_map[family] = final_model

    result = pd.DataFrame(rows).sort_values("pr_auc", ascending=False).reset_index(drop=True)
    return result, artifact_map


def run_flaml_track(
    X_train,
    y_train,
    X_holdout,
    y_holdout,
    time_budget: int = 240,
    random_state: int = 42,
):
    from flaml import AutoML

    automl = AutoML()
    t0 = time.perf_counter()
    automl.fit(
        X_train=X_train,
        y_train=y_train,
        task="classification",
        metric="ap",
        time_budget=time_budget,
        eval_method="cv",
        n_splits=3,
        estimator_list=["lgbm", "xgboost", "rf", "extra_tree", "lrl1"],
        seed=random_state,
    )
    train_time = time.perf_counter() - t0

    scores = automl.predict_proba(X_holdout)[:, 1]
    m = _metrics(y_holdout, scores)

    mean_latency, p95_latency = _measure_latency(automl.model.estimator, X_holdout)

    return {
        "model_name": str(automl.best_estimator),
        "library_source": "flaml",
        "pr_auc": m["pr_auc"],
        "roc_auc": m["roc_auc"],
        "brier": m["brier"],
        "train_time_sec": train_time,
        "infer_latency_ms": mean_latency,
        "p95_latency_ms": p95_latency,
        "calibration_metric": m["brier"],
        "best_config": automl.best_config,
        "best_loss": automl.best_loss,
        "time_budget": time_budget,
        "interpretability_note": "FLAML challenger under explicit budget",
    }


def run_pycaret_track(
    train_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    target_col: str,
    session_id: int,
    model_output_path: Path,
):
    try:
        import sys

        original_version_info = sys.version_info
        try:
            if tuple(sys.version_info) >= (3, 12):
                sys.version_info = (3, 11, 9, "final", 0)
            from pycaret.classification import (
                calibrate_model,
                compare_models,
                create_model,
                finalize_model,
                models as pycaret_models,
                predict_model,
                save_model,
                setup,
                tune_model,
            )
        finally:
            sys.version_info = original_version_info

        if len(train_df) > 30000:
            train_df = train_df.sample(n=30000, random_state=session_id).reset_index(drop=True)

        setup(
            data=train_df,
            target=target_col,
            session_id=session_id,
            fold=5,
            fold_strategy="stratifiedkfold",
            preprocess=True,
            normalize=True,
            remove_multicollinearity=True,
            multicollinearity_threshold=0.95,
            imputation_type="simple",
            numeric_imputation="median",
            categorical_imputation="most_frequent",
            html=False,
            n_jobs=1,
            verbose=False,
        )
        available_ids = set(pycaret_models().index.tolist())
        preferred_ids = ["lr", "rf", "et", "xgboost", "lightgbm", "catboost"]
        include_ids = [m for m in preferred_ids if m in available_ids]

        if include_ids:
            best = compare_models(sort="AUC", include=include_ids)
        else:
            best = compare_models(sort="AUC")
        if isinstance(best, list):
            if len(best) == 0:
                fallback_id = include_ids[0] if include_ids else "lr"
                best = create_model(fallback_id)
            else:
                best = best[0]

        try:
            tuned = tune_model(best, optimize="AUC")
        except Exception:
            tuned = best
        try:
            calibrated = calibrate_model(tuned)
        except Exception:
            calibrated = tuned
        final_model = finalize_model(calibrated)
        pred = predict_model(final_model, data=holdout_df.copy())
        score_col = "prediction_score" if "prediction_score" in pred.columns else "Score"
        scores = pred[score_col].astype(float).values
        y_true = holdout_df[target_col].astype(int).values
        m = _metrics(y_true, scores)

        save_model(final_model, str(model_output_path))
        return {
            "model_name": type(final_model).__name__,
            "library_source": "pycaret",
            "pr_auc": m["pr_auc"],
            "roc_auc": m["roc_auc"],
            "brier": m["brier"],
            "train_time_sec": np.nan,
            "infer_latency_ms": np.nan,
            "p95_latency_ms": np.nan,
            "calibration_metric": m["brier"],
            "interpretability_note": "PyCaret tuned+calibrated finalized model",
            "status": "ok",
        }
    except Exception as exc:
        return {
            "model_name": "pycaret_failed",
            "library_source": "pycaret",
            "pr_auc": np.nan,
            "roc_auc": np.nan,
            "brier": np.nan,
            "train_time_sec": np.nan,
            "infer_latency_ms": np.nan,
            "p95_latency_ms": np.nan,
            "calibration_metric": np.nan,
            "interpretability_note": f"PyCaret unavailable or failed: {exc}",
            "status": "failed",
        }


def save_inference_bundle(
    model,
    preprocessor,
    output_dir: Path,
    approve_threshold: float,
    reject_threshold: float,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_dir / "credit_risk_model.joblib")
    joblib.dump(preprocessor, output_dir / "credit_risk_preprocessor.joblib")

    with (output_dir / "decision_thresholds.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "approve_threshold": approve_threshold,
                "reject_threshold": reject_threshold,
            },
            f,
            indent=2,
        )
