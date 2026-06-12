"""Batch eval runner: every case through the full graph, HITL auto-approved.

Results stream to JSONL per case, so an interrupted run resumes instead of
restarting — at 2-5 minutes per case on a local 8B model that matters.
"""

import json
import time
from typing import Any

import pandas as pd
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command
from loguru import logger

from aml_investigator import db, telemetry
from aml_investigator.db import fetch_evidence
from aml_investigator.evaluation.metrics import report_groundedness
from aml_investigator.graph.build import build_graph
from aml_investigator.settings import settings


def run_eval(model: str, cases: list[dict[str, Any]], run_name: str) -> pd.DataFrame:
    """Run every case through the graph with ``model`` as backbone.

    Auto-resumes the human gate with 'approve' (so dispositions are purely the
    system's own). Writes ``artifacts/eval/<run_name>.jsonl`` incrementally and
    returns the collected rows as a DataFrame.
    """
    out_dir = settings.artifacts_dir / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / f"{run_name}.jsonl"

    done: dict[str, dict] = {}
    if jsonl_path.exists():
        for line in jsonl_path.read_text().splitlines():
            row = json.loads(line)
            done[row["account_id"]] = row
        if done:
            logger.info(f"{run_name}: resuming, {len(done)} case(s) already done")

    con = db.connect()
    graph = build_graph(con, checkpointer=MemorySaver(), model=model)
    rows: list[dict[str, Any]] = []

    with jsonl_path.open("a") as sink:
        for i, case in enumerate(cases, start=1):
            if case["account_id"] in done:
                rows.append(done[case["account_id"]])
                continue
            case_id = f"{run_name}-{case['account_id']}"
            t0 = time.perf_counter()
            state = None
            for attempt in (1, 2):  # one retry: a multi-hour run must survive transient faults
                con.execute("DELETE FROM evidence WHERE case_id = ?", [case_id])
                con.execute("DELETE FROM case_log WHERE case_id = ?", [case_id])
                config = {"configurable": {"thread_id": f"{case_id}-a{attempt}"}}
                try:
                    with telemetry.case_scope(case_id) as tel:
                        state = graph.invoke(
                            {"case_id": case_id, "account_id": case["account_id"],
                             "alerts": case["alerts"]},
                            config,
                        )
                        state = graph.invoke(Command(resume={"decision": "approve"}), config)
                    break
                except Exception as e:
                    logger.error(f"{case_id} attempt {attempt} failed: {type(e).__name__}: {e}")
            if state is None:  # both attempts failed; skip — a re-run resumes this case
                continue
            wall = time.perf_counter() - t0

            evidence = fetch_evidence(con, case_id)
            grounded = report_groundedness(state["report_md"], evidence, state["risk"])
            sanctions_hits = sum(
                len(e["payload"].get("hits", [])) for e in evidence if e["tool"] == "sanctions_check"
            )
            row = {
                "run": run_name,
                "model": model,
                "account_id": case["account_id"],
                "label": case["label"],
                "typology_true": case["typology"],
                "disposition": state["final_disposition"],
                "risk_score": state["risk"]["risk_score"],
                "typology_pred": state["risk"]["typology"],
                "sanctions_hits_found": sanctions_hits,
                "coverage_net_checks": len(state.get("coverage_ran", [])),
                "used_fallback_report": bool(state.get("used_fallback_report", False)),
                "report_retries": state.get("report_retries", 0),
                "money_claims": grounded["money_claims"],
                "grounded_claims": grounded["grounded_claims"],
                "wall_seconds": round(wall, 1),
                **{k: v for k, v in tel.summary().items() if k != "case_id"},
            }
            sink.write(json.dumps(row) + "\n")
            sink.flush()
            rows.append(row)
            logger.info(
                f"[{i}/{len(cases)}] {case['account_id']} ({case['label']}/{case['typology']}) "
                f"-> {row['disposition']} score={row['risk_score']} "
                f"pred={row['typology_pred']} {wall:.0f}s"
            )

    con.close()
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / f"{run_name}_results.csv", index=False)
    return df
