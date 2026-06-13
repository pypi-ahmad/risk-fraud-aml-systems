"""Report validation and the deterministic fallback template.

``validate_report`` is the anti-hallucination gate: structure, citation
integrity, and numeric groundedness (every dollar figure in the report must
exist somewhere in the stored evidence). The Jinja fallback guarantees the
graph always finishes with a well-formed report even if the LLM never produces
a valid draft.
"""

import re
from typing import Any

from jinja2 import Template

REQUIRED_SECTIONS = ("## Case Summary", "## Evidence", "## Risk Assessment", "## Recommendation")

_MONEY_RE = re.compile(r"\$\s?([\d,]+(?:\.\d+)?)")
_EVID_RE = re.compile(r"\[(EV-\d{2})\]")
_MONEY_KEYWORDS = ("amount", "total", "balance", "volume", "sum", "paid", "received", "transferred")


def _numbers_in(obj: Any, *, key_hint: str | None = None) -> set[float]:
    """Recursively pull monetary values from evidence payloads/summaries.

    Important: groundedness is about *dollar* claims in reports. We therefore
    ignore generic numbers in free text (IDs, dates, counts) and only extract:
    - values explicitly written as ``$...`` in strings
    - numeric fields whose key names look monetary (amount/total/etc.)
    """
    found: set[float] = set()
    if isinstance(obj, bool):
        return found
    if isinstance(obj, (int, float)):
        if key_hint and any(k in key_hint.lower() for k in _MONEY_KEYWORDS):
            found.add(round(float(obj), 2))
    elif isinstance(obj, str):
        for m in _MONEY_RE.finditer(obj):
            found.add(round(float(m.group(1).replace(",", "")), 2))
    elif isinstance(obj, dict):
        for k, v in obj.items():
            found |= _numbers_in(v, key_hint=k)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            found |= _numbers_in(v, key_hint=key_hint)
    return found


def validate_report(report_md: str, evidence: list[dict], risk: dict) -> dict[str, Any]:
    """Deterministically validate a drafted report against the evidence store.

    Returns ``{"valid": bool, "errors": [str, ...]}`` — errors are written to be
    fed straight back to the model on the retry pass.
    """
    errors: list[str] = []

    for section in REQUIRED_SECTIONS:
        if section not in report_md:
            errors.append(f"Missing required section heading: '{section}'")

    known_ids = {e["evidence_id"] for e in evidence}
    cited = set(_EVID_RE.findall(report_md))
    for bad in sorted(cited - known_ids):
        errors.append(f"Cited evidence id [{bad}] does not exist (valid: {sorted(known_ids)})")
    if not cited:
        errors.append("No [EV-xx] evidence citations found; every quantitative claim needs one")

    grounded = _numbers_in([e["summary"] for e in evidence]) | _numbers_in(
        [e["payload"] for e in evidence]
    ) | {round(float(risk.get("risk_score", 0)), 2)}
    for m in _MONEY_RE.finditer(report_md):
        value = round(float(m.group(1).replace(",", "")), 2)
        ok = any(
            abs(value - g) <= max(1.0, 0.01 * abs(g))  # 1% tolerance for display rounding
            for g in grounded
        )
        if not ok:
            errors.append(
                f"Dollar figure ${m.group(1)} does not appear in any stored evidence — "
                "remove it or replace it with a figure from the evidence"
            )

    disposition = risk.get("recommendation", "ESCALATE")
    if f"DISPOSITION: {disposition}" not in report_md:
        errors.append(f"Report must end with 'DISPOSITION: {disposition}' to match the risk assessment")

    return {"valid": not errors, "errors": errors}


_FALLBACK_TEMPLATE = Template(
    """\
## Case Summary
Automated investigation of account {{ account_id }} (case {{ case_id }}).
Alert(s): {{ alerts }}. The drafted narrative failed validation {{ retries }} time(s),
so this report was rendered deterministically from the validated evidence and the
structured risk assessment.

## Evidence
{% for e in evidence -%}
- [{{ e.evidence_id }}] ({{ e.tool }}) {{ e.summary }}
{% endfor %}
## Risk Assessment
Risk score: {{ risk.risk_score }}/100. Best-matching typology: {{ risk.typology }}.
{% for f in risk.factors -%}
- {{ f.claim }} [{{ f.evidence_id }}]
{% endfor %}
## Recommendation
Based on the structured assessment above, the recommended action is {{ risk.recommendation }}.

DISPOSITION: {{ risk.recommendation }}
"""
)


def render_fallback(case_id: str, account_id: str, alerts: str, evidence: list[dict],
                    risk: dict, retries: int) -> str:
    """Render the guaranteed-valid template report (no LLM involved)."""
    return _FALLBACK_TEMPLATE.render(
        case_id=case_id, account_id=account_id, alerts=alerts,
        evidence=evidence, risk=risk, retries=retries,
    )
