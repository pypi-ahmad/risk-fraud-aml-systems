"""Domain models. Every LLM-facing payload is a validated Pydantic model.

The ``Literal`` enums double as constrained-decoding vocabulary: when a schema is
passed to Ollama's ``format=`` parameter the model physically cannot emit a value
outside the enum.
"""

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

CheckName = Literal[
    "profile_account",
    "velocity_scan",
    "structuring_scan",
    "counterparty_network",
    "sanctions_check",
]

TypologyName = Literal[
    "structuring",
    "velocity_burst",
    "circular_transfers",
    "funnel_account",
    "sanctioned_counterparty",
    "none",
    "unclear",
]

ALL_CHECKS: tuple[str, ...] = (
    "profile_account",
    "velocity_scan",
    "structuring_scan",
    "counterparty_network",
    "sanctions_check",
)

# Checks that must always run regardless of what the agents decide.
MANDATORY_CHECKS: tuple[str, ...] = ("profile_account", "sanctions_check")


class TriageDecision(BaseModel):
    """Output of the triage agent: prioritise the alert and pick forensic checks."""

    priority: Literal["high", "medium", "low"]
    checks: list[CheckName] = Field(
        min_length=1, max_length=5, description="Forensic checks to run, each at most once"
    )
    rationale: str = Field(description="2-3 sentence justification grounded in the alert facts")

    @field_validator("checks")
    @classmethod
    def _dedupe(cls, v: list[str]) -> list[str]:
        return list(dict.fromkeys(v))


class RiskFactor(BaseModel):
    """A single claim backed by a specific piece of stored evidence."""

    claim: str = Field(description="One factual statement about the account's behaviour")
    evidence_id: str = Field(description="The EV-xx id of the evidence supporting the claim")


class RiskAssessment(BaseModel):
    """Output of the risk analyst: structured verdict over the collected evidence."""

    risk_score: int = Field(ge=0, le=100)
    typology: TypologyName = Field(description="Best-matching AML typology, or none/unclear")
    factors: list[RiskFactor] = Field(min_length=1, max_length=8)
    recommendation: Literal["ESCALATE", "DISMISS"]
    needs_more_evidence: bool = Field(
        description="True only if one more targeted check would materially change the verdict"
    )
    requested_check: CheckName | None = Field(
        default=None, description="The additional check to run if needs_more_evidence"
    )


class JudgeScore(BaseModel):
    """LLM-judge rubric for a generated investigation report (1 = poor, 5 = excellent)."""

    groundedness: int = Field(ge=1, le=5, description="Claims are supported by the cited evidence")
    completeness: int = Field(ge=1, le=5, description="All material evidence is reflected")
    clarity: int = Field(ge=1, le=5, description="A compliance officer could act on it")
    justification: str = Field(description="2-3 sentences explaining the scores")


class HumanDecision(BaseModel):
    """Human-gate decision schema for interrupt/resume."""

    decision: Literal["approve", "override"] = "approve"
    disposition: Literal["ESCALATE", "DISMISS"] | None = None
    note: str | None = None

    @model_validator(mode="after")
    def _validate_override(self) -> "HumanDecision":
        if self.decision == "override" and self.disposition is None:
            raise ValueError("override decisions must include disposition ESCALATE or DISMISS")
        if self.decision == "approve" and self.disposition is not None:
            raise ValueError("approve decisions must not include an override disposition")
        return self
