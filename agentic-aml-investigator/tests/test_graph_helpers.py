import pytest

from aml_investigator import telemetry
from aml_investigator.graph.build import (
    _normalize_human_decision,
    _safe_case_filename,
    _tool_call_succeeded,
)
from aml_investigator.tools.forensics import set_active_case


def test_tool_success_detection():
    assert _tool_call_succeeded("[EV-01] account profile collected")
    assert _tool_call_succeeded({"ok": True})
    assert not _tool_call_succeeded("ERROR: sql guard rejected query")


def test_safe_case_filename_sanitizes_path_chars():
    assert _safe_case_filename("../CASE:001") == "CASE_001"
    assert _safe_case_filename("  ") == "case"


def test_normalize_human_decision_enforces_schema():
    decision = _normalize_human_decision({"decision": "override", "disposition": "ESCALATE"})
    assert decision.decision == "override"
    assert decision.disposition == "ESCALATE"

    fallback = _normalize_human_decision({"decision": "override"})
    assert fallback.decision == "approve"
    assert fallback.disposition is None


def test_case_scope_rejects_nested_different_case():
    with telemetry.case_scope("CASE-1"):
        with pytest.raises(RuntimeError):
            with telemetry.case_scope("CASE-2"):
                pass


def test_set_active_case_checks_against_telemetry_scope():
    with telemetry.case_scope("CASE-1"):
        with pytest.raises(RuntimeError):
            set_active_case("CASE-2")
