"""Cross-family LLM judge for report quality.

Each backbone's reports are scored by the OTHER model family (granite judges
qwen, qwen judges granite) to dampen self-preference bias. The judge sees the
report plus the evidence summaries it should be grounded in, and returns a
structured rubric. Scores are a secondary signal — the deterministic
groundedness checker is the primary report metric.
"""

import json
from typing import Any

import pandas as pd
from loguru import logger

from aml_investigator import db, telemetry
from aml_investigator.db import fetch_evidence
from aml_investigator.llm import structured_llm_call
from aml_investigator.schemas import JudgeScore
from aml_investigator.settings import settings

JUDGE_SYSTEM = """\
You are a senior compliance QA reviewer scoring an AML investigation report.
Score each dimension 1 (poor) to 5 (excellent):
- groundedness: every claim in the report is supported by the listed evidence;
  penalise any figure or assertion that does not appear in the evidence.
- completeness: the report reflects all material evidence (including exonerating
  evidence and 'no findings' results).
- clarity: a compliance officer could act on the report without reading anything else.
Be strict: a 5 should be rare. Judge ONLY against the evidence provided.
"""


def judge_run(run_name: str, judge_model: str) -> pd.DataFrame:
    """Score every report of an eval run. Appends to ``<run_name>_judge.jsonl``."""
    eval_dir = settings.artifacts_dir / "eval"
    rows = [json.loads(line) for line in (eval_dir / f"{run_name}.jsonl").read_text().splitlines()]
    out_path = eval_dir / f"{run_name}_judge.jsonl"
    done = set()
    if out_path.exists():
        done = {json.loads(line)["account_id"] for line in out_path.read_text().splitlines()}

    con = db.connect()
    scored: list[dict[str, Any]] = []
    with out_path.open("a") as sink:
        for row in rows:
            if row["account_id"] in done:
                continue
            case_id = f"{run_name}-{row['account_id']}"
            report_path = settings.artifacts_dir / "reports" / f"{case_id}.md"
            evidence = fetch_evidence(con, case_id)
            ev_lines = "\n".join(f"[{e['evidence_id']}] ({e['tool']}) {e['summary']}" for e in evidence)
            user = (
                f"EVIDENCE:\n{ev_lines}\n\nREPORT TO SCORE:\n{report_path.read_text()}\n\n"
                "Score the report."
            )
            with telemetry.case_scope(f"judge-{case_id}"):
                outcome = structured_llm_call(
                    JudgeScore, JUDGE_SYSTEM, user, model=judge_model, name="judge",
                    fallback=JudgeScore(
                        groundedness=3, completeness=3, clarity=3,
                        justification="Judge structured call failed; neutral default.",
                    ),
                )
            result = {
                "run": run_name, "judge_model": judge_model,
                "account_id": row["account_id"],
                **outcome.value.model_dump(),
                "judge_method": outcome.method_used,
            }
            sink.write(json.dumps(result) + "\n")
            sink.flush()
            scored.append(result)
            logger.info(
                f"judged {row['account_id']}: G{result['groundedness']} "
                f"C{result['completeness']} Cl{result['clarity']}"
            )
    con.close()

    all_rows = [json.loads(line) for line in out_path.read_text().splitlines()]
    df = pd.DataFrame(all_rows)
    df.to_csv(eval_dir / f"{run_name}_judge.csv", index=False)
    return df
