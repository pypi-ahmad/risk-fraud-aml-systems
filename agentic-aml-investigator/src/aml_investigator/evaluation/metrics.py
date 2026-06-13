"""Eval metrics: decision quality, process reliability, report groundedness."""

from typing import Any

import pandas as pd

from aml_investigator.reporting import _MONEY_RE, _numbers_in


def report_groundedness(report_md: str, evidence: list[dict], risk: dict) -> dict[str, int]:
    """Count dollar claims in a report and how many trace to stored evidence."""
    grounded_pool = (
        _numbers_in([e["summary"] for e in evidence])
        | _numbers_in([e["payload"] for e in evidence])
        | {round(float(risk.get("risk_score", 0)), 2)}
    )
    claims, grounded = 0, 0
    for m in _MONEY_RE.finditer(report_md):
        claims += 1
        value = round(float(m.group(1).replace(",", "")), 2)
        if any(abs(value - g) <= max(1.0, 0.01 * abs(g)) for g in grounded_pool):
            grounded += 1
    return {"money_claims": claims, "grounded_claims": grounded}


def decision_metrics(df: pd.DataFrame) -> dict[str, Any]:
    """ESCALATE-vs-label confusion and the derived rates."""
    if "status" in df.columns:
        df = df[df.status == "ok"]
    if df.empty:
        return {
            "tp": 0, "fp": 0, "fn": 0, "tn": 0,
            "accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0,
            "typology_top1_on_escalated": 0.0, "sanctions_recall": float("nan"),
            "mean_score_suspicious": 0.0, "mean_score_clean": 0.0,
        }
    tp = len(df[(df.label == "suspicious") & (df.disposition == "ESCALATE")])
    fn = len(df[(df.label == "suspicious") & (df.disposition == "DISMISS")])
    fp = len(df[(df.label == "clean") & (df.disposition == "ESCALATE")])
    tn = len(df[(df.label == "clean") & (df.disposition == "DISMISS")])
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    sus = df[df.label == "suspicious"]
    escalated_sus = sus[sus.disposition == "ESCALATE"]
    typology_top1 = (
        (escalated_sus.typology_pred == escalated_sus.typology_true).mean()
        if len(escalated_sus) else 0.0
    )
    sanc = df[df.typology_true == "sanctioned_counterparty"]
    sanctions_recall = (
        ((sanc.sanctions_hits_found > 0) & (sanc.disposition == "ESCALATE")).mean()
        if len(sanc) else float("nan")
    )
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "accuracy": round((tp + tn) / len(df), 3) if len(df) else 0.0,
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "typology_top1_on_escalated": round(float(typology_top1), 3),
        "sanctions_recall": round(float(sanctions_recall), 3),
        "mean_score_suspicious": round(float(df[df.label == "suspicious"].risk_score.mean()), 1),
        "mean_score_clean": round(float(df[df.label == "clean"].risk_score.mean()), 1),
    }


def process_metrics(df: pd.DataFrame) -> dict[str, Any]:
    """How reliably the machinery ran (the honest-flakiness numbers)."""
    if "status" in df.columns:
        df = df[df.status == "ok"]
    if df.empty:
        return {
            "cases": 0,
            "wall_p50_s": 0.0,
            "wall_p95_s": 0.0,
            "llm_calls_per_case": 0.0,
            "tokens_per_case": 0,
            "tool_error_rate": 0.0,
            "structured_retry_rate": 0.0,
            "structured_fallback_rate": 0.0,
            "coverage_net_activation_rate": 0.0,
            "fallback_report_rate": 0.0,
            "report_retry_rate": 0.0,
            "groundedness": 0.0,
        }
    return {
        "cases": len(df),
        "wall_p50_s": round(float(df.wall_seconds.median()), 1),
        "wall_p95_s": round(float(df.wall_seconds.quantile(0.95)), 1),
        "llm_calls_per_case": round(float(df.llm_calls.mean()), 1),
        "tokens_per_case": int((df.prompt_tokens + df.completion_tokens).mean()),
        "tool_error_rate": round(float(df.tool_errors.sum() / max(1, df.tool_calls.sum())), 3),
        "structured_retry_rate": round(float((df.structured_retries > 0).mean()), 3),
        "structured_fallback_rate": round(float((df.structured_fallbacks > 0).mean()), 3),
        "coverage_net_activation_rate": round(float((df.coverage_net_checks > 0).mean()), 3),
        "fallback_report_rate": round(float(df.used_fallback_report.mean()), 3),
        "report_retry_rate": round(float((df.report_retries > 0).mean()), 3),
        "groundedness": round(float(df.grounded_claims.sum() / max(1, df.money_claims.sum())), 3),
    }
