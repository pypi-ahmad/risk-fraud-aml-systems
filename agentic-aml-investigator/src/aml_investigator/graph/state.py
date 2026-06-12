"""Graph state.

Everything in here must be JSON-serializable — the SQLite checkpointer
snapshots this dict at every super-step, which is what makes interrupt/resume
and kill-and-resume work. Full tool payloads deliberately live in the DuckDB
evidence store, NOT in state: the model re-reads compact summaries only
(state compression), and checkpoints stay small.
"""

from typing import TypedDict


class InvestigationState(TypedDict, total=False):
    case_id: str
    account_id: str
    alerts: list[dict]  # [{rule, details}, ...]

    triage: dict  # TriageDecision dump
    investigator_summary: str
    checks_completed: list[str]
    coverage_ran: list[str]  # checks the deterministic net had to run itself

    risk: dict  # RiskAssessment dump
    reflection_rounds: int

    report_md: str
    report_errors: list[str]
    report_retries: int
    used_fallback_report: bool

    human_decision: dict  # {decision: approve|override, disposition?, note?}
    final_disposition: str  # ESCALATE | DISMISS
    report_path: str
