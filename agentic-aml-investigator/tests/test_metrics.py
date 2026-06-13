import pandas as pd

from aml_investigator.evaluation.metrics import decision_metrics, process_metrics


def test_decision_metrics_ignore_failed_rows():
    df = pd.DataFrame(
        [
            {
                "status": "ok",
                "label": "suspicious",
                "disposition": "ESCALATE",
                "typology_true": "structuring",
                "typology_pred": "structuring",
                "sanctions_hits_found": 0,
                "risk_score": 70,
            },
            {
                "status": "failed",
                "label": "clean",
                "disposition": "ERROR",
                "typology_true": "none",
                "typology_pred": "unclear",
                "sanctions_hits_found": 0,
                "risk_score": None,
            },
        ]
    )
    out = decision_metrics(df)
    assert out["tp"] == 1
    assert out["accuracy"] == 1.0


def test_process_metrics_ignore_failed_rows():
    df = pd.DataFrame(
        [
            {
                "status": "ok",
                "wall_seconds": 10.0,
                "llm_calls": 2,
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "tool_errors": 0,
                "tool_calls": 4,
                "structured_retries": 0,
                "structured_fallbacks": 0,
                "coverage_net_checks": 1,
                "used_fallback_report": False,
                "report_retries": 0,
                "grounded_claims": 3,
                "money_claims": 3,
            },
            {
                "status": "failed",
                "wall_seconds": 999.0,
                "llm_calls": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "tool_errors": 0,
                "tool_calls": 0,
                "structured_retries": 0,
                "structured_fallbacks": 0,
                "coverage_net_checks": 0,
                "used_fallback_report": False,
                "report_retries": 0,
                "grounded_claims": 0,
                "money_claims": 0,
            },
        ]
    )
    out = process_metrics(df)
    assert out["cases"] == 1
    assert out["wall_p50_s"] == 10.0
