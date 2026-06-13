"""Assemble the investigation graph.

Topology::

    START -> triage -> investigate -> coverage_net -> assess_risk
    assess_risk --(needs more evidence, once)--> gather_more -> assess_risk
    assess_risk --(otherwise)--> write_report -> validate
    validate --(invalid, retry budget left)--> write_report
    validate --(invalid, budget exhausted)--> fallback_report -> human_gate
    validate --(valid)--> human_gate -> finalize -> END

The reliability engineering lives in three places: every structured LLM output
goes through :func:`structured_llm_call` (escalation ladder + fallback), the
coverage net guarantees mandated checks run even if the agents forget them, and
the report validator + template fallback guarantee a well-formed final report.
"""

import time
from datetime import datetime
import re

import duckdb
from langchain.agents import create_agent
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from loguru import logger
from pydantic import ValidationError

from aml_investigator import telemetry
from aml_investigator.db import fetch_evidence
from aml_investigator.graph.state import InvestigationState
from aml_investigator.llm import get_chat_model, structured_llm_call
from aml_investigator.prompts import INVESTIGATOR_SYSTEM, REPORT_SYSTEM, RISK_SYSTEM, TRIAGE_SYSTEM
from aml_investigator.reporting import render_fallback, validate_report
from aml_investigator.schemas import (
    ALL_CHECKS,
    HumanDecision,
    MANDATORY_CHECKS,
    RiskAssessment,
    RiskFactor,
    TriageDecision,
)
from aml_investigator.settings import settings
from aml_investigator.tools.forensics import make_tools, set_active_case


def _alerts_text(state: InvestigationState) -> str:
    return "; ".join(f"{a['rule']}: {a['details']}" for a in state["alerts"]) or "manual referral"


def _tool_call_succeeded(output: object) -> bool:
    """True when a tool invocation produced evidence rather than an error string."""
    return not (isinstance(output, str) and output.strip().startswith("ERROR:"))


def _safe_case_filename(case_id: str) -> str:
    """Sanitize a case id into a safe filename stem."""
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", case_id).strip("._")
    return safe or "case"


def _normalize_human_decision(raw: object) -> HumanDecision:
    """Validate/normalize interrupt payload into a strict decision schema."""
    if isinstance(raw, str):
        candidate = {"decision": raw.strip().lower()}
    elif isinstance(raw, dict):
        candidate = raw
    else:
        candidate = {"decision": "approve"}
    try:
        return HumanDecision.model_validate(candidate)
    except ValidationError as exc:
        logger.warning(f"invalid human decision payload ({exc}); defaulting to approve")
        return HumanDecision(decision="approve")


def build_graph(
    con: duckdb.DuckDBPyConnection,
    checkpointer: BaseCheckpointSaver | None = None,
    model: str | None = None,
):
    """Compile the investigation graph against a warehouse connection.

    One compiled graph serves any number of cases (sequentially); the active
    case id is set module-globally per node so the evidence store stays per-case.
    """
    tools = make_tools(con)
    model_name = model or settings.agent_model
    react_agent = create_agent(
        get_chat_model(model_name),
        list(tools.values()),
        system_prompt=INVESTIGATOR_SYSTEM,
    )

    # ---------------- nodes ----------------

    def triage(state: InvestigationState) -> dict:
        set_active_case(state["case_id"])
        acc = con.execute(
            "SELECT holder_name, country, account_type, opened_date FROM accounts WHERE account_id = ?",
            [state["account_id"]],
        ).fetchone()
        if acc is None:
            logger.warning(f"triage received unknown account_id={state['account_id']}; using fallback triage")
            return {
                "triage": TriageDecision(
                    priority="high",
                    checks=list(MANDATORY_CHECKS),
                    rationale="Account was not found; defaulting to mandatory checks for safety.",
                ).model_dump(),
                "reflection_rounds": 0,
                "report_retries": 0,
            }
        user = (
            f"Account {state['account_id']}: '{acc[0]}', {acc[2]} account, country {acc[1]}, "
            f"opened {acc[3]}.\nAlert(s): {_alerts_text(state)}"
        )
        outcome = structured_llm_call(
            TriageDecision, TRIAGE_SYSTEM, user, model=model_name, name="triage",
            fallback=TriageDecision(
                priority="high",
                checks=list(MANDATORY_CHECKS),
                rationale="Structured triage failed; defaulting to a mandatory-check sweep.",
            ),
        )
        return {"triage": outcome.value.model_dump(), "reflection_rounds": 0, "report_retries": 0}

    def investigate(state: InvestigationState) -> dict:
        set_active_case(state["case_id"])
        checks = state["triage"]["checks"]
        task = (
            f"Investigate account {state['account_id']}.\n"
            f"Alert(s): {_alerts_text(state)}\n"
            f"Triage rationale: {state['triage']['rationale']}\n"
            f"Requested checks (run each exactly once): {', '.join(checks)}"
        )
        try:
            result = react_agent.invoke(
                {"messages": [{"role": "user", "content": task}]},
                config={
                    "recursion_limit": settings.investigator_recursion_limit,
                    "callbacks": [telemetry.LLMTimingHandler()],
                },
            )
            summary = str(result["messages"][-1].content)[:2000]
        except GraphRecursionError:
            logger.warning("investigator hit the recursion limit; proceeding with partial evidence")
            summary = "(Investigation truncated at the step limit; proceeding with collected evidence.)"
        except Exception as e:
            # A hung/errored ReAct loop must not kill the case: coverage_net runs every
            # mandated check deterministically next, so evidence is still collected.
            logger.warning(f"investigator failed ({type(e).__name__}: {str(e)[:120]}); "
                           "coverage net will collect the mandated evidence")
            summary = "(Investigation step failed; coverage net collected the evidence below.)"
        done = sorted(
            {e["tool"] for e in fetch_evidence(con, state["case_id"]) if e["tool"] != "run_sql"}
        )
        return {"investigator_summary": summary, "checks_completed": done}

    def coverage_net(state: InvestigationState) -> dict:
        """Deterministic safety net: run any mandated check the agents skipped."""
        set_active_case(state["case_id"])
        requested = set(MANDATORY_CHECKS) | set(state["triage"]["checks"])
        # Policy (v2, from eval iteration): a periodic/manual referral carries no
        # alert signal to narrow on — such reviews are full-scope by definition.
        # v1 evaluation showed every ring/funnel case dismissed because triage
        # picked minimal checks for alert-less referrals and the network scan
        # never ran.
        if any(a["rule"] == "manual_referral" for a in state["alerts"]):
            requested = set(ALL_CHECKS)
        done = set(state.get("checks_completed", []))
        missing = sorted(requested - done)
        ran: list[str] = []
        failed: list[dict[str, str]] = []
        for name in missing:
            logger.info(f"coverage net running skipped check: {name}")
            output = tools[name].invoke({"account_id": state["account_id"]})
            if _tool_call_succeeded(output):
                ran.append(name)
            else:
                failed.append({"check": name, "error": str(output)[:300]})
                logger.error(f"coverage net check failed for {name}: {str(output)[:180]}")
        return {
            "coverage_ran": ran,
            "coverage_failed": failed,
            "checks_completed": sorted(done | set(ran)),
        }

    def assess_risk(state: InvestigationState) -> dict:
        set_active_case(state["case_id"])
        evidence = fetch_evidence(con, state["case_id"])
        ev_lines = "\n".join(f"[{e['evidence_id']}] ({e['tool']}) {e['summary']}" for e in evidence)
        user = (
            f"Case {state['case_id']}, account {state['account_id']}.\n"
            f"Alert(s): {_alerts_text(state)}\n\n"
            f"Investigator summary:\n{state.get('investigator_summary', '(none)')}\n\n"
            f"Evidence:\n{ev_lines}\n\n"
            "Produce the structured risk assessment."
        )
        first_ev = evidence[0]["evidence_id"] if evidence else "EV-01"
        outcome = structured_llm_call(
            RiskAssessment, RISK_SYSTEM, user, model=model_name, name="assess_risk",
            fallback=RiskAssessment(
                risk_score=60, typology="unclear",
                factors=[RiskFactor(claim="Structured risk assessment failed; defaulting to "
                                          "human review", evidence_id=first_ev)],
                recommendation="ESCALATE", needs_more_evidence=False,
            ),
        )
        risk = outcome.value

        # Deterministic guardrail: a very-high-confidence OFAC match cannot be scored away.
        sanctions_scores = [
            h["score"]
            for e in evidence if e["tool"] == "sanctions_check"
            for h in e["payload"].get("hits", [])
        ]
        max_hit = max(sanctions_scores, default=0)
        if max_hit >= 93 and risk.risk_score < 75:
            ev_id = next(e["evidence_id"] for e in evidence if e["tool"] == "sanctions_check")
            logger.info(f"guardrail: OFAC match {max_hit} forces risk floor 75 (was {risk.risk_score})")
            risk = risk.model_copy(update={
                "risk_score": 75,
                "recommendation": "ESCALATE",
                "factors": risk.factors + [RiskFactor(
                    claim=f"Deterministic guardrail: OFAC fuzzy match at score {max_hit} "
                          "forces a minimum risk score of 75",
                    evidence_id=ev_id,
                )],
            })
        return {"risk": risk.model_dump()}

    def gather_more(state: InvestigationState) -> dict:
        """Run the one extra check the analyst asked for (bounded reflection)."""
        set_active_case(state["case_id"])
        check = state["risk"].get("requested_check")
        completed = set(state.get("checks_completed", []))
        if check and check not in set(state.get("checks_completed", [])):
            output = tools[check].invoke({"account_id": state["account_id"]})
            if _tool_call_succeeded(output):
                completed.add(check)
            else:
                logger.error(f"gather_more failed for {check}: {str(output)[:180]}")
        return {
            "reflection_rounds": state.get("reflection_rounds", 0) + 1,
            "checks_completed": sorted(completed),
        }

    def write_report(state: InvestigationState) -> dict:
        set_active_case(state["case_id"])
        evidence = fetch_evidence(con, state["case_id"])
        ev_lines = "\n".join(f"[{e['evidence_id']}] ({e['tool']}) {e['summary']}" for e in evidence)
        user = (
            f"Case {state['case_id']}, account {state['account_id']}.\n"
            f"Alert(s): {_alerts_text(state)}\n\nEvidence:\n{ev_lines}\n\n"
            f"Risk assessment: {state['risk']}\n\nWrite the report."
        )
        if state.get("report_errors"):
            user += (
                "\n\nYour previous draft failed validation with these errors — fix every one:\n- "
                + "\n- ".join(state["report_errors"])
            )
        llm = get_chat_model(model_name)
        t0 = time.perf_counter()
        response = llm.invoke([("system", REPORT_SYSTEM), ("user", user)])
        usage = response.usage_metadata or {}
        telemetry.current().record(
            "llm", "write_report", time.perf_counter() - t0,
            input_tokens=usage.get("input_tokens", 0), output_tokens=usage.get("output_tokens", 0),
        )
        return {"report_md": str(response.content)}

    def validate(state: InvestigationState) -> dict:
        evidence = fetch_evidence(con, state["case_id"])
        result = validate_report(state["report_md"], evidence, state["risk"])
        if not result["valid"]:
            logger.warning(f"report validation failed: {result['errors']}")
        return {
            "report_errors": result["errors"],
            "report_retries": state.get("report_retries", 0) + (0 if result["valid"] else 1),
        }

    def fallback_report(state: InvestigationState) -> dict:
        evidence = fetch_evidence(con, state["case_id"])
        md = render_fallback(
            state["case_id"], state["account_id"], _alerts_text(state),
            evidence, state["risk"], state.get("report_retries", 0),
        )
        return {"report_md": md, "used_fallback_report": True, "report_errors": []}

    def human_gate(state: InvestigationState) -> dict:
        decision = interrupt(
            {
                "case_id": state["case_id"],
                "account_id": state["account_id"],
                "risk": state["risk"],
                "report_md": state["report_md"],
                "question": "approve the recommendation, or override with "
                            "{'decision': 'override', 'disposition': 'ESCALATE'|'DISMISS', 'note': ...}",
            }
        )
        normalized = _normalize_human_decision(decision)
        return {"human_decision": normalized.model_dump(exclude_none=True)}

    def finalize(state: InvestigationState) -> dict:
        decision = state["human_decision"]
        if decision.get("decision") == "override" and decision.get("disposition"):
            disposition = decision["disposition"]
        else:
            disposition = state["risk"]["recommendation"]
        report_dir = settings.artifacts_dir / "reports"
        report_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{_safe_case_filename(state['case_id'])}.md"
        report_path = (report_dir / filename).resolve()
        if report_dir.resolve() not in report_path.parents:
            raise ValueError("resolved report path escapes artifacts/report directory")
        report_path.write_text(state["report_md"])
        con.execute(
            "INSERT INTO case_log VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [
                state["case_id"], state["account_id"], disposition,
                state["risk"]["risk_score"], state["risk"]["typology"],
                decision.get("decision", "approve"), str(report_path), datetime.now(),
            ],
        )
        return {"final_disposition": disposition, "report_path": str(report_path)}

    # ---------------- wiring ----------------

    def route_after_risk(state: InvestigationState) -> str:
        risk = state["risk"]
        if (
            risk.get("needs_more_evidence")
            and risk.get("requested_check")
            and state.get("reflection_rounds", 0) < settings.max_reflection_rounds
        ):
            return "gather_more"
        return "write_report"

    def route_after_validate(state: InvestigationState) -> str:
        if not state.get("report_errors"):
            return "human_gate"
        if state.get("report_retries", 0) <= settings.max_report_retries:
            return "write_report"
        return "fallback_report"

    g = StateGraph(InvestigationState)
    g.add_node("triage", triage)
    g.add_node("investigate", investigate)
    g.add_node("coverage_net", coverage_net)
    g.add_node("assess_risk", assess_risk)
    g.add_node("gather_more", gather_more)
    g.add_node("write_report", write_report)
    g.add_node("validate", validate)
    g.add_node("fallback_report", fallback_report)
    g.add_node("human_gate", human_gate)
    g.add_node("finalize", finalize)

    g.add_edge(START, "triage")
    g.add_edge("triage", "investigate")
    g.add_edge("investigate", "coverage_net")
    g.add_edge("coverage_net", "assess_risk")
    g.add_conditional_edges("assess_risk", route_after_risk,
                            {"gather_more": "gather_more", "write_report": "write_report"})
    g.add_edge("gather_more", "assess_risk")
    g.add_edge("write_report", "validate")
    g.add_conditional_edges("validate", route_after_validate,
                            {"human_gate": "human_gate", "write_report": "write_report",
                             "fallback_report": "fallback_report"})
    g.add_edge("fallback_report", "human_gate")
    g.add_edge("human_gate", "finalize")
    g.add_edge("finalize", END)

    return g.compile(checkpointer=checkpointer)
