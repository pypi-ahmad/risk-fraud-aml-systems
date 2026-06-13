from aml_investigator.reporting import render_fallback, validate_report

EVIDENCE = [
    {
        "evidence_id": "EV-01",
        "tool": "structuring_scan",
        "summary": "ACC-0001: 12 cash deposits in the $8,500-$10,000 band totalling $112,447",
        "payload": {"sub_threshold_count": 12, "sub_threshold_total": 112447},
    },
    {
        "evidence_id": "EV-02",
        "tool": "sanctions_check",
        "summary": "ACC-0001: screened 100 names. No matches.",
        "payload": {"hits": []},
    },
]
RISK = {
    "risk_score": 78,
    "typology": "structuring",
    "recommendation": "ESCALATE",
    "factors": [{"claim": "12 sub-threshold deposits", "evidence_id": "EV-01"}],
}

GOOD = """\
## Case Summary
Investigation of ACC-0001.

## Evidence
12 cash deposits totalling $112,447 [EV-01]. Sanctions screening clear [EV-02].

## Risk Assessment
Score 78, structuring.

## Recommendation
File for review. DISPOSITION: ESCALATE
"""


def test_good_report_passes():
    assert validate_report(GOOD, EVIDENCE, RISK) == {"valid": True, "errors": []}


def test_missing_section_fails():
    result = validate_report(GOOD.replace("## Evidence", "## Stuff"), EVIDENCE, RISK)
    assert not result["valid"]
    assert any("## Evidence" in e for e in result["errors"])


def test_unknown_citation_fails():
    result = validate_report(GOOD.replace("[EV-02]", "[EV-09]"), EVIDENCE, RISK)
    assert any("EV-09" in e for e in result["errors"])


def test_ungrounded_dollar_figure_fails():
    result = validate_report(GOOD.replace("$112,447", "$999,999"), EVIDENCE, RISK)
    assert any("$999,999" in e for e in result["errors"])


def test_rounding_tolerance_passes():
    result = validate_report(GOOD.replace("$112,447", "$112,000"), EVIDENCE, RISK)
    # 0.4% off -> within the 1% tolerance
    assert result["valid"]


def test_date_or_id_numbers_do_not_ground_money_claims():
    evidence = [
        {
            "evidence_id": "EV-01",
            "tool": "profile_account",
            "summary": "ACC-0001 opened 2026-05-31; no cash anomalies.",
            "payload": {"account_id": "ACC-0001", "opened_date": "2026-05-31"},
        }
    ]
    risk = {
        "risk_score": 18,
        "typology": "none",
        "recommendation": "DISMISS",
        "factors": [{"claim": "No suspicious findings", "evidence_id": "EV-01"}],
    }
    report = """\
## Case Summary
Routine review only.

## Evidence
Observed flow of $2,026 [EV-01].

## Risk Assessment
Low risk.

## Recommendation
Close. DISPOSITION: DISMISS
"""
    result = validate_report(report, evidence, risk)
    assert not result["valid"]
    assert any("$2,026" in e for e in result["errors"])


def test_wrong_disposition_fails():
    result = validate_report(GOOD.replace("ESCALATE", "DISMISS"), EVIDENCE, RISK)
    assert any("DISPOSITION" in e for e in result["errors"])


def test_fallback_template_is_always_valid():
    md = render_fallback("CASE-X", "ACC-0001", "R1: test alert", EVIDENCE, RISK, retries=2)
    assert validate_report(md, EVIDENCE, RISK)["valid"]
