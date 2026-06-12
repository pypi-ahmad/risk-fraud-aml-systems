"""Author the tutorial notebooks programmatically (nbformat), then execute them
with nbconvert so every output in the committed notebooks is real.

Usage:
    uv run python scripts/build_notebooks.py 01   # build notebooks/01_*.ipynb
    uv run python scripts/build_notebooks.py all
"""

import sys
from pathlib import Path

import nbformat as nbf

NB_DIR = Path(__file__).resolve().parents[1] / "notebooks"


def _nb(cells: list) -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.metadata.kernelspec = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nb.cells = cells
    return nb


def md(source: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(source.strip())


def code(source: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(source.strip())


# --------------------------------------------------------------------------- #
# 01 — data foundation
# --------------------------------------------------------------------------- #
def nb01() -> nbf.NotebookNode:
    cells = [
        md("""
# 01 — Data Foundation: Agents Need a Data Layer, Not a CSV

**Project:** `agentic-aml-investigator` — a multi-agent AML (anti-money-laundering)
investigation copilot built with LangGraph and local Ollama models.

**This notebook** builds the DuckDB warehouse everything else runs on:

1. a **seeded synthetic transaction ledger** (~200 accounts, ~35k transactions, 90 days)
   with **five injected money-laundering typologies**, each recorded in a hidden
   `ground_truth` table,
2. the **real OFAC SDN sanctions list** (~19k entities, downloaded from treasury.gov),
3. **rule-based alerts** — and a look at which typologies rules *catch* vs which ones
   *only an investigation can find* (that gap is the whole reason this project exists).

> **Why synthetic data?** Real AML case data is never public, and the common public
> alternative (PaySim) contains a single fraud pattern — every investigation would look
> identical. A seeded generator gives five distinct labeled typologies, regenerates
> byte-identically on any machine (`seed=42`), and lets us *quantitatively score the
> agent's decisions against ground truth* in notebook 04. The sanctions list, however,
> is the real thing.
"""),
        code("""
import duckdb
import matplotlib.pyplot as plt
import pandas as pd

from aml_investigator import db                              # warehouse connection helper
from aml_investigator.data.generator import build_warehouse  # seeded ledger + OFAC + alerts
from aml_investigator.settings import settings               # all paths/knobs live here

pd.set_option("display.width", 140)
plt.rcParams["figure.figsize"] = (10, 3.5)

# build_warehouse(force=True): generate the synthetic ledger (seed=42), download/cache
# the real OFAC SDN list, derive rule-based alerts, and load it all into a DuckDB file.
# Returns the row count per table. Deterministic — re-running gives identical data.
counts = build_warehouse(force=True)
counts
"""),
        md("""
## The warehouse schema

Five tables matter to the agents (and one is deliberately hidden from them):

| table | rows | who sees it |
|---|---|---|
| `accounts` | KYC profiles | agents |
| `transactions` | the ledger | agents |
| `sdn` | **real** OFAC SDN list | agents |
| `alerts` | rule-engine output | agents |
| `ground_truth` | typology labels | **evaluation only — the SQL guard blocks agents from ever reading it** |
"""),
        code("""
con = db.connect()
con.execute("SELECT * FROM transactions ORDER BY ts LIMIT 5").fetch_df()
"""),
        code("""
# What does 90 days of bank activity look like?
fig, axes = plt.subplots(1, 2, figsize=(13, 3.5))
con.execute(\"\"\"
    SELECT txn_type, count(*) AS n FROM transactions GROUP BY 1 ORDER BY n DESC
\"\"\").fetch_df().plot.bar(x="txn_type", y="n", ax=axes[0], legend=False, title="Transactions by type")
con.execute(\"\"\"
    SELECT CAST(ts AS DATE) AS day, round(sum(amount)/1e6, 2) AS volume_m
    FROM transactions GROUP BY 1 ORDER BY 1
\"\"\").fetch_df().plot(x="day", y="volume_m", ax=axes[1], legend=False, title="Daily volume ($M)")
plt.tight_layout()
"""),
        md("""
## The five injected typologies

These are the labeled patterns the agent system will be evaluated against.
The `ground_truth` table records which accounts carry which typology:
"""),
        code("""
con.execute("SELECT * FROM ground_truth ORDER BY typology, account_id").fetch_df()
"""),
        md("""
### Typology 1 — Structuring (a.k.a. smurfing)

Cash deposits kept *just under* the $10,000 Currency Transaction Report threshold,
then aggregated and wired out. The histogram makes the evasion visible: a legitimate
cash-intensive business deposits across the whole range (including >$10k, which it
duly reports); the structuring account's deposits pile up in a narrow band below
the line.
"""),
        code("""
struct_acc = con.execute(
    "SELECT account_id FROM ground_truth WHERE typology = 'structuring' LIMIT 1").fetchone()[0]
cash_biz = con.execute(\"\"\"
    SELECT t.account_id FROM transactions t LEFT JOIN ground_truth g USING (account_id)
    WHERE t.txn_type = 'cash_deposit' AND g.account_id IS NULL
    GROUP BY 1 ORDER BY count(*) DESC LIMIT 1\"\"\").fetchone()[0]

fig, axes = plt.subplots(1, 2, figsize=(13, 3.5), sharex=True)
for ax, acc, title in [(axes[0], struct_acc, f"{struct_acc} (structuring)"),
                       (axes[1], cash_biz, f"{cash_biz} (legit cash business)")]:
    amounts = con.execute(
        "SELECT amount FROM transactions WHERE account_id = ? AND txn_type = 'cash_deposit'",
        [acc]).fetch_df()["amount"]
    ax.hist(amounts, bins=30)
    ax.axvline(10_000, color="red", linestyle="--", label="$10k CTR threshold")
    ax.set_title(title); ax.legend()
plt.tight_layout()
"""),
        md("""
### Typology 2 — Velocity burst

A dormant account (salary credits only) suddenly cycles ~$100k+ in and out within
three days. Daily activity tells the story at a glance:
"""),
        code("""
vel_acc = con.execute(
    "SELECT account_id FROM ground_truth WHERE typology = 'velocity_burst' LIMIT 1").fetchone()[0]
daily = con.execute(\"\"\"
    SELECT CAST(ts AS DATE) AS day, count(*) AS txns FROM transactions
    WHERE account_id = ? GROUP BY 1 ORDER BY 1\"\"\", [vel_acc]).fetch_df()
daily.plot.bar(x="day", y="txns", legend=False,
               title=f"{vel_acc}: dormant, then a 3-day burst")
plt.xticks([]); plt.tight_layout()
"""),
        md("""
### Typology 3 — Circular transfers

Three accounts pass nearly the same amount around a ring (A → B → C → A), six times.
No single hop looks odd; the *cycle* is the signal — which is why the
`counterparty_network` tool does ring detection with a recursive SQL self-join.
"""),
        code("""
ring = [r[0] for r in con.execute(
    "SELECT account_id FROM ground_truth WHERE typology = 'circular_transfers' ORDER BY 1").fetchall()]
con.execute(\"\"\"
    SELECT account_id AS src, counterparty_id AS dst, count(*) AS hops, round(sum(amount)) AS total
    FROM transactions
    WHERE account_id IN (SELECT account_id FROM ground_truth WHERE typology = 'circular_transfers')
          AND direction = 'out' AND txn_type = 'transfer' AND counterparty_id LIKE 'ACC-%'
    GROUP BY 1, 2 HAVING count(*) > 2 ORDER BY 1\"\"\").fetch_df()
"""),
        md("""
### Typology 4 — Funnel account

Seventeen unrelated senders converge on one account; the balance exits in a single
large wire. Sender concentration is the tell:
"""),
        code("""
funnel_acc = con.execute(
    "SELECT account_id FROM ground_truth WHERE typology = 'funnel_account'").fetchone()[0]
con.execute(\"\"\"
    SELECT direction, txn_type, count(*) AS n,
           count(DISTINCT coalesce(counterparty_id, counterparty_name)) AS distinct_counterparties,
           round(sum(amount)) AS total
    FROM transactions WHERE account_id = ? GROUP BY 1, 2 ORDER BY total DESC\"\"\",
    [funnel_acc]).fetch_df()
"""),
        md("""
### Typology 5 — Sanctioned counterparty (real OFAC data)

Two accounts wire money to counterparties whose names are **near-variants of real
SDN entries** (one character changed — realistic sanctions evasion). Exact-match
screening misses these; fuzzy matching catches them. Notice how close the planted
names are to the real entries:
"""),
        code("""
con.execute("SELECT details FROM ground_truth WHERE typology = 'sanctioned_counterparty'").fetch_df()
"""),
        code("""
# The real OFAC SDN list is in the warehouse for the agents to screen against
print(con.execute("SELECT count(*) FROM sdn").fetchone()[0], "real SDN entries, e.g.:")
con.execute("SELECT * FROM sdn WHERE sdn_type = 'individual' LIMIT 5").fetch_df()
"""),
        md("""
## Rule-based alerts — and what rules miss

A classic AML programme runs deterministic rules. Ours implements three (sub-threshold
deposit counting, calendar-baseline velocity spikes, large wires to high-risk
countries). The crucial result is the *coverage gap*:
"""),
        code("""
print(con.execute(\"\"\"
    SELECT a.rule, count(*) AS alerts,
           sum(CASE WHEN g.account_id IS NOT NULL THEN 1 ELSE 0 END) AS on_labeled_accounts
    FROM alerts a LEFT JOIN ground_truth g USING (account_id) GROUP BY 1\"\"\").fetch_df(), "\\n")

coverage = con.execute(\"\"\"
    SELECT g.typology,
           count(DISTINCT g.account_id) AS labeled,
           count(DISTINCT a.account_id) AS with_alert
    FROM ground_truth g LEFT JOIN alerts a USING (account_id)
    GROUP BY 1 ORDER BY 1\"\"\").fetch_df()
coverage
"""),
        md("""
**Read that table carefully — it motivates the whole project:**

- `structuring` and `velocity_burst` are caught by rules (and the rules *also* fire
  on legitimate cash businesses and payroll runs — false positives someone must triage).
- `circular_transfers`, `funnel_account` and `sanctioned_counterparty` produce
  **zero alerts**. Rings need graph analysis, funnels need counterparty concentration
  analysis, and evasive sanctions spellings need fuzzy screening — none of which a
  threshold rule does.

In the next notebooks, the agent system handles *both* halves of the problem:
investigating alerts (separating true hits from hard negatives) and running the
deeper forensics that rules can't express. The accounts with no alert enter the
eval set as "periodic KYC review" referrals — exactly how real compliance teams
open rule-less cases.
"""),
        code("""
con.close()
"""),
    ]
    return _nb(cells)


# --------------------------------------------------------------------------- #
# 02 — tools and reliability
# --------------------------------------------------------------------------- #
def nb02() -> nbf.NotebookNode:
    cells = [
        md("""
# 02 — Forensic Tools & Local-Model Reliability

The #1 killer of local-LLM agent projects is **tool-calling flakiness**: a 8-9B
quantized model that forgets the format, invents arguments, or rambles instead of
answering. This notebook shows the two layers of defense this project uses, with
measurements, not folklore:

1. **Tool design** — deterministic tools do the analytical heavy lifting (SQL,
   fuzzy matching); the model only *picks* tools and reads compact summaries.
   Full results go to an **evidence store**, not into the context window.
2. **A measured reliability ladder for structured output** — we benchmark two
   extraction methods on both local backbones and discover a genuine Ollama
   gotcha along the way.

Everything below runs against the warehouse from notebook 01.
"""),
        code("""
import json
import time

import httpx
import pandas as pd
from pydantic import ValidationError

from aml_investigator import db
from aml_investigator.settings import settings
from aml_investigator.tools.forensics import make_tools, set_active_case

pd.set_option("display.max_colwidth", 160)

con = db.connect()                                          # open the warehouse from nb01
con.execute("DELETE FROM evidence WHERE case_id = 'nb02-demo'")  # clean slate for this demo
set_active_case("nb02-demo")    # tools tag every evidence row with the active case id
tools = make_tools(con)         # build the toolset bound to this connection
list(tools)
"""),
        md("""
## The forensic toolset

Five specialised scans plus one guarded ad-hoc SQL tool. Note the design contract
visible in every output below: results come back as `[EV-xx] <compact summary>`.
The full payload (every row, every match) is persisted to the `evidence` table
under that id — the model's context stays small, and later the report validator
can check every claimed number against what the tools *actually* returned.

### `structuring_scan` — on a known structuring account
"""),
        code("""
struct_acc = con.execute(
    "SELECT account_id FROM ground_truth WHERE typology = 'structuring' LIMIT 1").fetchone()[0]
print(tools["structuring_scan"].invoke({"account_id": struct_acc}))
"""),
        md("### `velocity_scan` — dormant account, 3-day burst"),
        code("""
vel_acc = con.execute(
    "SELECT account_id FROM ground_truth WHERE typology = 'velocity_burst' LIMIT 1").fetchone()[0]
print(tools["velocity_scan"].invoke({"account_id": vel_acc}))
"""),
        md("### `counterparty_network` — ring detection via recursive SQL"),
        code("""
ring_acc = con.execute(
    "SELECT account_id FROM ground_truth WHERE typology = 'circular_transfers' LIMIT 1").fetchone()[0]
print(tools["counterparty_network"].invoke({"account_id": ring_acc}))
"""),
        md("""
Note the output contains both the **planted ring** (~$270k) *and* a tiny organic
"ring" from random baseline transfers (a few thousand dollars). Real forensic
tools produce noise; deciding which cycle matters is exactly the risk analyst's
job in notebook 03.

### `sanctions_check` — fuzzy screening against the real OFAC list
"""),
        code("""
sanc_acc = con.execute(
    "SELECT account_id FROM ground_truth WHERE typology = 'sanctioned_counterparty' LIMIT 1").fetchone()[0]
print(tools["sanctions_check"].invoke({"account_id": sanc_acc}))
"""),
        code("""
# ... and on a busy clean account, to see the false-positive behaviour of fuzzy matching
print(tools["sanctions_check"].invoke({"account_id": "ACC-0011"})
      if sanc_acc != "ACC-0011" else tools["sanctions_check"].invoke({"account_id": "ACC-0001"}))
"""),
        md("""
Fuzzy screening at cutoff 87 catches one-character evasions (scores 94-96) but can
also surface coincidental near-matches of legitimate names. That trade-off
(cutoff vs false positives) is a **policy decision**, which is why the score and
the program land in the evidence rather than a binary verdict.

### `run_sql` — guarded ad-hoc analysis

The agent gets real SQL power, fenced three ways: SELECT-only (parsed with
`sqlglot`, not regex), table allowlist, auto-`LIMIT`. The allowlist is also the
**anti-cheating mechanism**: `ground_truth` (the eval labels!) is invisible.
"""),
        code("""
print(tools["run_sql"].invoke({"query":
    "SELECT counterparty_country, count(*) n, round(sum(amount)) total "
    f"FROM transactions WHERE account_id = '{vel_acc}' AND direction = 'out' "
    "GROUP BY 1 ORDER BY total DESC"}))
"""),
        code("""
print(tools["run_sql"].invoke({"query": "DROP TABLE transactions"}))
print(tools["run_sql"].invoke({"query": "SELECT * FROM ground_truth"}))
"""),
        md("""
## The evidence store (state compression in action)

Every tool call above persisted its full payload. The model only ever saw the
short summaries — here's the size difference that keeps an 8B model's context
manageable across a multi-step investigation:
"""),
        code("""
con.execute(\"\"\"
    SELECT evidence_id, tool, length(summary) AS summary_chars, length(payload) AS payload_chars
    FROM evidence WHERE case_id = 'nb02-demo' ORDER BY evidence_id\"\"\").fetch_df()
"""),
        md("""
## Structured output on local models: measure, don't assume

Agent graphs need *validated* structured decisions (triage priorities, risk
scores). There are two mechanisms on Ollama:

- **`function_calling`** — the model emits a tool call; arguments are parsed.
- **`json_schema`** — Ollama constrains decoding with a grammar built from the
  JSON schema (the model physically cannot emit invalid JSON).

### First, a genuine gotcha we hit while building this

On Ollama 0.30.x with a thinking model (qwen3.5), **setting `think: false`
silently disables `format=` constrained decoding** — you get free-form markdown
back. With thinking left on, the thinking goes to a separate field and the
content is perfectly constrained. Demonstrated live:
"""),
        code("""
schema = {"type": "object",
          "properties": {"priority": {"type": "string", "enum": ["high", "medium", "low"]},
                          "rationale": {"type": "string"}},
          "required": ["priority", "rationale"]}
msg = [{"role": "user", "content":
        "Priority for: 14 deposits of $9,900 in 3 days then wired abroad?"}]

for think in (False, None):
    body = {"model": "qwen3.5:9b", "stream": False, "format": schema,
            "options": {"temperature": 0}, "messages": msg}
    if think is not None:
        body["think"] = think
    r = httpx.post(f"{settings.ollama_base_url}/api/chat", json=body, timeout=300).json()
    content = r["message"]["content"]
    try:
        json.loads(content)
        verdict = "VALID JSON"
    except json.JSONDecodeError:
        verdict = "NOT JSON (constrained decoding silently disabled!)"
    print(f"think={think}: {verdict}\\n  content head: {content[:110]!r}\\n")
"""),
        md("""
### The benchmark: 2 backbones x 2 methods x 3 triage prompts

Small but real: every cell below actually calls the local models. We measure
**validity** (parsed into the Pydantic schema without error) and **latency**.
"""),
        code("""
from aml_investigator.llm import get_chat_model
from aml_investigator.schemas import TriageDecision

PROMPTS = [
    "Alert: account received 14 cash deposits of $9,900 each over 3 days (just under "
    "the $10,000 reporting threshold), then wired the full balance abroad.",
    "Alert: dormant personal account suddenly moved $180,000 in and out within 72 hours "
    "via 30 transfers, destinations include high-risk countries.",
    "Alert: periodic KYC review of a business account with frequent large wires to "
    "counterparties in AE and TR; one name resembles an OFAC-listed entity.",
]
SYSTEM = ("You are an AML triage officer. Select priority and forensic checks "
          "(each at most once) from: profile_account, structuring_scan, velocity_scan, "
          "counterparty_network, sanctions_check.")

rows = []
for model in ("granite4.1:8b", "qwen3.5:9b"):
    for method in ("function_calling", "json_schema"):
        for i, prompt in enumerate(PROMPTS):
            llm = get_chat_model(model).with_structured_output(TriageDecision, method=method)
            t0 = time.perf_counter()
            try:
                result = llm.invoke([("system", SYSTEM), ("user", prompt)])
                ok, n_checks = result is not None, len(result.checks) if result else 0
            except Exception as e:
                ok, n_checks = False, 0
            rows.append({"model": model, "method": method, "prompt": i,
                         "valid": ok, "seconds": round(time.perf_counter() - t0, 1),
                         "n_checks": n_checks})
            print(rows[-1])

bench = pd.DataFrame(rows)
out_dir = settings.artifacts_dir / "reliability"
out_dir.mkdir(parents=True, exist_ok=True)
bench.to_csv(out_dir / "structured_output_benchmark.csv", index=False)
"""),
        code("""
summary = bench.groupby(["model", "method"]).agg(
    validity_rate=("valid", "mean"),
    median_seconds=("seconds", "median"),
    mean_checks=("n_checks", "mean"),
).round(2)
summary
"""),
        md("""
### What the numbers mean (and the design they led to)

- **granite4.1:8b + function_calling** is the sweet spot on this hardware —
  single-digit seconds with full validity. That's why it's the default backbone.
- **qwen3.5:9b pays a thinking tax** on every structured call; with `json_schema`
  (which requires thinking left on, per the gotcha above) it can take minutes.
- `json_schema` is the *reliability* backstop: grammar-constrained decoding cannot
  produce unparseable output, even when it's slow.

Hence the project's `structured_llm_call` escalation ladder, used by every
graph node in notebook 03:

```
function_calling  →  json_schema (+ error feedback)  →  deterministic fallback
   (fast path)          (guaranteed parseable)            (graph never dies)
```
"""),
        code("""
from aml_investigator.llm import structured_llm_call

outcome = structured_llm_call(
    TriageDecision, SYSTEM, PROMPTS[0], name="nb02-demo",
    fallback=TriageDecision(priority="high", checks=["profile_account", "sanctions_check"],
                            rationale="fallback"))
print(f"method={outcome.method_used} attempts={outcome.attempts} {outcome.seconds}s")
outcome.value.model_dump()
"""),
        code("""
con.close()
"""),
    ]
    return _nb(cells)


# --------------------------------------------------------------------------- #
# 03 — the investigation graph
# --------------------------------------------------------------------------- #
def nb03() -> nbf.NotebookNode:
    cells = [
        md("""
# 03 — The Multi-Agent Investigation Graph

This is where the pieces become a system. The graph below investigates one AML
case end-to-end: triage → forensic investigation → risk assessment → report →
**human approval gate** → audit trail.

```mermaid
flowchart TD
    START([START]) --> T[triage<br/><i>structured LLM</i>]
    T --> I[investigate<br/><i>ReAct agent + forensic tools</i>]
    I --> C[coverage_net<br/><b>deterministic</b>: runs skipped checks]
    C --> R[assess_risk<br/><i>structured LLM + guardrail floor</i>]
    R -->|needs more evidence, max 1| G[gather_more] --> R
    R --> W[write_report<br/><i>free-text LLM</i>]
    W --> V[validate<br/><b>deterministic</b>: structure, citations,<br/>numeric groundedness]
    V -->|invalid, retry| W
    V -->|invalid, budget spent| F[fallback_report<br/><b>deterministic</b> Jinja template]
    V -->|valid| H[human_gate<br/>interrupt ⏸]
    F --> H
    H --> Z[finalize<br/>report file + case_log row] --> END([END])
```

**The reliability thesis of this design:** every LLM step is paired with a
deterministic safety mechanism — the agent *proposes*, the machinery *guarantees*:

| LLM step | what guarantees correctness anyway |
|---|---|
| triage picks checks | `coverage_net` diff-and-runs anything mandated but skipped |
| ReAct investigation | tools are deterministic; evidence persisted server-side |
| risk scoring | OFAC-match guardrail floor; validated schema with fallback |
| report writing | validator (citations + every $ figure re-checked) → template fallback |

So a flaky 8B model can degrade *quality*, but it cannot crash the pipeline,
skip a sanctions screen, or fabricate an evidence number that survives validation.
"""),
        code("""
import json
import sqlite3

import pandas as pd
from IPython.display import Image, Markdown, display
from langgraph.checkpoint.sqlite import SqliteSaver  # file-backed state snapshots
from langgraph.types import Command                  # used to resume after an interrupt()

from aml_investigator import db, telemetry
from aml_investigator.graph.build import build_graph
from aml_investigator.settings import settings

con = db.connect()  # the warehouse the forensic tools query

# The checkpointer is what makes the graph pausable and resumable. We build it over a
# LONG-LIVED sqlite3 connection (check_same_thread=False) rather than the usual
# `with SqliteSaver.from_conn_string(...)` context-manager form — that form closes the
# DB at the end of the cell, which fights Jupyter's cell-by-cell execution. (This is a
# real gotcha; it's documented in the project README and CLAUDE.md.)
settings.checkpoint_db_path.parent.mkdir(parents=True, exist_ok=True)
ckpt_conn = sqlite3.connect(settings.checkpoint_db_path, check_same_thread=False)
checkpointer = SqliteSaver(ckpt_conn)   # durable: survives kernel/process restarts
graph = build_graph(con, checkpointer=checkpointer)  # compiles the StateGraph above
"""),
        code("""
# Render the actual compiled topology (not just the diagram above)
try:
    png = graph.get_graph().draw_mermaid_png()
    (settings.artifacts_dir / "graph.png").write_bytes(png)
    display(Image(png, width=420))
except Exception as e:   # offline fallback: print the mermaid source
    print(f"(mermaid render unavailable: {e})")
    print(graph.get_graph().draw_mermaid())
"""),
        md("""
## Case 1 — a structuring alert, streamed node by node

`stream_mode="updates"` shows each node finishing in real time. The run pauses
at the human gate (`__interrupt__`).
"""),
        code("""
# Pick a known structuring account (ids are looked up at runtime, never hardcoded —
# regenerating the warehouse reshuffles them) and pull its real rule alerts.
account = con.execute(
    "SELECT account_id FROM ground_truth WHERE typology = 'structuring' LIMIT 1").fetchone()[0]
alerts = [{"rule": r, "details": d} for r, d in con.execute(
    "SELECT rule, details FROM alerts WHERE account_id = ?", [account]).fetchall()]
case_id = f"NB03-{account}"
con.execute("DELETE FROM evidence WHERE case_id = ?", [case_id])   # idempotent re-runs
con.execute("DELETE FROM case_log WHERE case_id = ?", [case_id])
# thread_id is the case's durable identity in the checkpointer — one thread per case.
config = {"configurable": {"thread_id": case_id}}

print(f"Investigating {account}, alerts: {[a['rule'] for a in alerts]}\\n")
with telemetry.case_scope(case_id) as tel:
    for update in graph.stream(
        {"case_id": case_id, "account_id": account, "alerts": alerts},
        config, stream_mode="updates",
    ):
        for node, payload in update.items():
            if node == "__interrupt__":
                print("⏸  INTERRUPTED at human_gate — waiting for a human decision")
            else:
                keys = list(payload or {})
                print(f"✔ {node:<16} -> updated {keys}")
"""),
        code("""
# What is the graph waiting on? Inspect the live checkpoint.
snapshot = graph.get_state(config)
print("next node:", snapshot.next)
interrupt_payload = snapshot.interrupts[0].value
print("risk score:", interrupt_payload["risk"]["risk_score"],
      "| typology:", interrupt_payload["risk"]["typology"],
      "| recommendation:", interrupt_payload["risk"]["recommendation"])
"""),
        code("""
# The compliance officer approves. Command(resume=...) re-enters the graph
# exactly where it paused.
with telemetry.case_scope(case_id) as tel2:
    state = graph.invoke(Command(resume={"decision": "approve"}), config)
print("final disposition:", state["final_disposition"])
print("report written to:", state["report_path"])
"""),
        code("""
display(Markdown(state["report_md"]))
"""),
        md("""
Every dollar figure in that report was validated against the evidence store before
the human ever saw it (notebook 02's validator) — and the case is now in the audit
trail:
"""),
        code("""
con.execute("SELECT * FROM case_log WHERE case_id = ?", [case_id]).fetch_df()
"""),
        md("""
## Kill-and-resume: the persistence demo that matters

Investigations pause for human review — in production that can be *days*, and the
process will restart in between. Because the checkpointer is SQLite-backed, we can
**throw away every Python object, rebuild from the file, and resume the paused
case**. (Restarting the notebook kernel here would prove the same thing; rebuilding
the objects is the same code path and keeps the notebook executable top-to-bottom.)
"""),
        code("""
account2 = con.execute(
    "SELECT account_id FROM ground_truth WHERE typology = 'sanctioned_counterparty' LIMIT 1"
).fetchone()[0]
alerts2 = [{"rule": "manual_referral", "details": "Periodic KYC review"}]
case_id2 = f"NB03-{account2}"
con.execute("DELETE FROM evidence WHERE case_id = ?", [case_id2])
con.execute("DELETE FROM case_log WHERE case_id = ?", [case_id2])
config2 = {"configurable": {"thread_id": case_id2}}

with telemetry.case_scope(case_id2):
    graph.invoke({"case_id": case_id2, "account_id": account2, "alerts": alerts2}, config2)
print("paused at:", graph.get_state(config2).next)

# --- simulate process death ---
del graph, checkpointer
ckpt_conn.close()
print("…graph object destroyed, sqlite connection closed…")
"""),
        code("""
# A 'new process': fresh connection, fresh saver, fresh compiled graph
ckpt_conn = sqlite3.connect(settings.checkpoint_db_path, check_same_thread=False)
checkpointer = SqliteSaver(ckpt_conn)
graph = build_graph(con, checkpointer=checkpointer)

restored = graph.get_state(config2)
print("restored from disk — next node:", restored.next)
print("pending risk score:", restored.interrupts[0].value["risk"]["risk_score"])
"""),
        code("""
# Resume the restored case as if days had passed and a different process picked it up.
with telemetry.case_scope(case_id2):
    state2 = graph.invoke(Command(resume={"decision": "approve"}), config2)
print("disposition:", state2["final_disposition"],
      "| risk:", state2["risk"]["risk_score"], state2["risk"]["typology"])
for f in state2["risk"]["factors"][:4]:
    print(" -", f["claim"][:110], f"[{f['evidence_id']}]")
"""),
        md("""
### Time travel: the full checkpoint history

Every super-step was snapshotted. `get_state_history` is the audit trail of the
*computation* (the `case_log` table is the audit trail of the *decision*):
"""),
        code("""
history = list(graph.get_state_history(config2))
pd.DataFrame([{
    "step": s.metadata.get("step"),
    "node": next(iter(s.metadata.get("writes") or {"-": None})),
    "has_interrupt": bool(s.interrupts),
} for s in history]).head(12)
"""),
        md("""
## Case 3 — a hard negative (alert fired, account is clean)

The rules flagged this cash-intensive business for sub-threshold deposits.
A good investigator should notice deposits *above* the threshold too (structurers
never go over), the months-long stable pattern, and clear sanctions screening —
and dismiss.
"""),
        code("""
hard_neg = con.execute(\"\"\"
    SELECT a.account_id FROM alerts a
    LEFT JOIN ground_truth g USING (account_id)
    JOIN accounts acc ON acc.account_id = a.account_id
    WHERE g.account_id IS NULL AND a.rule = 'sub_threshold_deposits'
    ORDER BY a.account_id LIMIT 1\"\"\").fetchone()[0]
alerts3 = [{"rule": r, "details": d} for r, d in con.execute(
    "SELECT rule, details FROM alerts WHERE account_id = ?", [hard_neg]).fetchall()]
case_id3 = f"NB03-{hard_neg}"
con.execute("DELETE FROM evidence WHERE case_id = ?", [case_id3])
con.execute("DELETE FROM case_log WHERE case_id = ?", [case_id3])
config3 = {"configurable": {"thread_id": case_id3}}

with telemetry.case_scope(case_id3) as tel3:
    graph.invoke({"case_id": case_id3, "account_id": hard_neg, "alerts": alerts3}, config3)
    # This time the human OVERRIDES rather than approving — the gate has teeth:
    # whatever the system recommended, the officer records DISMISS with a note.
    state3 = graph.invoke(Command(resume={
        "decision": "override", "disposition": "DISMISS",
        "note": "Verified legitimate cash-intensive business; deposits exceed the "
                "threshold routinely, pattern stable for months."}), config3)
print("system recommended:", state3["risk"]["recommendation"],
      f"(score {state3['risk']['risk_score']})")
print("officer recorded:  ", state3["final_disposition"], "(human override)")
print("\\nrisk factors the system cited:")
for f in state3["risk"]["factors"]:
    print(" -", f["claim"][:110], f"[{f['evidence_id']}]")
"""),
        md("""
Whether the model dismisses this case is a *quality* question — notebook 04
measures it across all 18 labeled cases instead of anecdotes. Telemetry for this
case, as a preview of what the eval aggregates:
"""),
        code("""
tel3.summary()
"""),
        code("""
ckpt_conn.close()
con.close()
"""),
    ]
    return _nb(cells)


# --------------------------------------------------------------------------- #
# 04 — evaluation
# --------------------------------------------------------------------------- #
def nb04() -> nbf.NotebookNode:
    cells = [
        md("""
# 04 — Evaluating Agent Behaviour (Not Vibes)

A demo that works once is not evidence. This notebook runs the **full graph over
all 18 labeled cases** — 10 accounts carrying injected typologies, 8 clean
accounts including every hard negative the rule engine tripped on — for **two
local backbones**, and scores three layers:

1. **Decision quality** (vs ground truth): confusion matrix, precision/recall/F1,
   typology identification, sanctions recall, risk-score separation.
2. **Process reliability** (from telemetry): structured-output retries and
   fallbacks, coverage-net activations, tool errors, latency, tokens.
3. **Report quality**: deterministic groundedness (% of dollar claims that trace
   to stored evidence) plus a **cross-family LLM judge** (granite judges qwen's
   reports, qwen judges granite's — damping self-preference bias).

The human gate is auto-approved throughout, so dispositions are purely the
system's own.

**This notebook deliberately shows two evaluation rounds.** Round 1 scored the
system as first built — and exposed two *systematic* failure modes. We diagnose
them, apply two targeted fixes (one policy, one prompt), and re-run everything
as round 2. Both rounds' numbers are real and preserved in `artifacts/eval/`;
nothing below is curated. This is the loop that matters in agent engineering:
**evaluate → diagnose → fix → re-evaluate.**

> ⏱ Every number in this notebook comes from real local inference
> (2 rounds x 18 cases x ~6 LLM calls x 2 backbones, on an RTX 4060 Laptop, 8 GB VRAM).
"""),
        code("""
import json

import matplotlib.pyplot as plt
import pandas as pd

from aml_investigator import db
from aml_investigator.evaluation.cases import eval_cases
from aml_investigator.evaluation.judge import judge_run
from aml_investigator.evaluation.metrics import decision_metrics, process_metrics
from aml_investigator.evaluation.runner import run_eval
from aml_investigator.settings import settings

pd.set_option("display.width", 160)

con = db.connect()
# eval_cases() builds the labeled set: all 10 ground-truth accounts (suspicious) plus
# 8 clean controls (every hard negative the rules tripped on, padded with top-volume
# clean accounts). Accounts with no rule alert get a synthetic "manual_referral".
cases = eval_cases(con)
con.close()
pd.DataFrame(cases)[["account_id", "label", "typology"]].groupby(
    ["label", "typology"]).size().rename("cases").reset_index()
"""),
        md("""
## Round 1 — the system as first built

Results stream to `artifacts/eval/<run>.jsonl` case by case, so an interrupted
run resumes where it stopped (and re-executing this notebook re-loads the cached
round-1 rows instead of re-running them with the post-fix code).
"""),
        code("""
df_granite_v1 = run_eval("granite4.1:8b", cases, run_name="granite4_1_8b")
df_qwen_v1 = run_eval("qwen3.5:9b", cases, run_name="qwen3_5_9b")

decisions_v1 = pd.DataFrame({
    "granite4.1:8b (v1)": decision_metrics(df_granite_v1),
    "qwen3.5:9b (v1)": decision_metrics(df_qwen_v1),
})
decisions_v1
"""),
        code("""
# Round 1 errors, in full — these rows drive everything that follows
errors_v1 = pd.concat([df_granite_v1.assign(run="granite"), df_qwen_v1.assign(run="qwen")])
errors_v1 = errors_v1[
    ((errors_v1.label == "suspicious") & (errors_v1.disposition == "DISMISS")) |
    ((errors_v1.label == "clean") & (errors_v1.disposition == "ESCALATE"))]
errors_v1[["run", "account_id", "label", "typology_true", "typology_pred",
           "risk_score", "tool_calls", "disposition"]].sort_values(["run", "label"])
"""),
        md("""
## Diagnosis: the misses are systematic, not random

Look at the error table with the `tool_calls` column and two patterns jump out:

**Failure 1 — every `circular_transfers` and `funnel_account` case is dismissed,
always with only 2 tool calls.** These accounts produce no rule alerts (rules
can't see networks — notebook 01 showed that), so they enter as *manual
referrals*. With no alert facts to go on, triage requests only the minimal
checks; `counterparty_network` never runs; the ring evidence is never collected;
the risk analyst correctly scores the evidence it sees — which contains nothing.
The failure is in **check selection policy**, not in any model's reasoning.

**Failure 2 — the hard negatives get escalated.** The analyst sees
"22 cash deposits in the $8.5k-$10k band" and escalates — ignoring the
exonerating signal *in the same evidence*: the account also makes dozens of
deposits **over** $10k. Real structurers never cross the threshold (crossing it
triggers the report they're evading). Same story for payroll: a "velocity burst"
that recurs on the 28th of every month is payroll, not laundering. This is
**missing domain doctrine in the risk prompt**.

### The fixes (both visible in the repo diff)

1. **Policy, deterministic** (`graph/build.py`, coverage_net): a manual referral
   is full-scope by definition — all five checks run, guaranteed by the
   deterministic net rather than by hoping the LLM picks them.
2. **Doctrine, prompt** (`prompts.py`, RISK_SYSTEM): structurers never cross the
   threshold; payroll recurs monthly; weigh ring amounts vs graph noise.

Neither fix references evaluation labels — one is a compliance policy any AML
programme has, the other is textbook AML doctrine. Now we re-run **everything**.
"""),
        md("""
## Round 2 — after the fixes

Fresh run names (`*_v2`), fresh inference, identical cases.
"""),
        code("""
df_granite = run_eval("granite4.1:8b", cases, run_name="granite4_1_8b_v2")
df_granite[["account_id", "label", "typology_true", "disposition", "risk_score",
            "typology_pred", "coverage_net_checks", "wall_seconds"]]
"""),
        code("""
df_qwen = run_eval("qwen3.5:9b", cases, run_name="qwen3_5_9b_v2")
df_qwen[["account_id", "label", "typology_true", "disposition", "risk_score",
         "typology_pred", "coverage_net_checks", "wall_seconds"]]
"""),
        md("""
## Layer 1 — Decision quality, round 2 vs round 1
"""),
        code("""
decisions = pd.DataFrame({
    "granite v1": decision_metrics(df_granite_v1),
    "granite v2": decision_metrics(df_granite),
    "qwen v1": decision_metrics(df_qwen_v1),
    "qwen v2": decision_metrics(df_qwen),
})
decisions
"""),
        code("""
fig, axes = plt.subplots(1, 4, figsize=(15, 3.4))
panels = [("granite v1", df_granite_v1), ("granite v2", df_granite),
          ("qwen v1", df_qwen_v1), ("qwen v2", df_qwen)]
for ax, (name, df) in zip(axes, panels):
    cm = pd.crosstab(df.label, df.disposition).reindex(
        index=["suspicious", "clean"], columns=["ESCALATE", "DISMISS"], fill_value=0)
    ax.imshow(cm.values, cmap="Blues", vmin=0, vmax=10)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm.values[i, j], ha="center", va="center", fontsize=15)
    ax.set_xticks([0, 1], cm.columns, fontsize=8); ax.set_yticks([0, 1], cm.index, fontsize=8)
    ax.set_title(name, fontsize=10)
fig.suptitle("Decisions vs labels — suspicious row: top-left=TP; clean row: bottom-right=TN")
plt.tight_layout()
plt.savefig(settings.artifacts_dir / "eval" / "confusion_matrices.png", dpi=120,
            bbox_inches="tight")
"""),
        code("""
# Risk-score separation, round 2: do scores now separate the classes?
fig, ax = plt.subplots(figsize=(9, 3.4))
for x, (name, df) in enumerate([("granite v2", df_granite), ("qwen v2", df_qwen)]):
    for label, color in [("suspicious", "tab:red"), ("clean", "tab:green")]:
        ys = df[df.label == label].risk_score
        ax.scatter([x + (0.12 if label == "clean" else -0.12)] * len(ys), ys,
                   alpha=0.6, color=color, label=label if x == 0 else None)
ax.set_xticks([0, 1], ["granite4.1:8b v2", "qwen3.5:9b v2"]); ax.set_ylabel("risk score")
ax.legend(); ax.set_title("Risk scores by true label (round 2)")
plt.tight_layout()
plt.savefig(settings.artifacts_dir / "eval" / "score_separation.png", dpi=120,
            bbox_inches="tight")
"""),
        code("""
# Remaining round-2 errors — what the system still gets wrong
errors = pd.concat([df_granite.assign(run="granite v2"), df_qwen.assign(run="qwen v2")])
errors = errors[((errors.label == "suspicious") & (errors.disposition == "DISMISS")) |
                ((errors.label == "clean") & (errors.disposition == "ESCALATE"))]
errors[["run", "account_id", "label", "typology_true", "typology_pred",
        "risk_score", "disposition"]]
"""),
        md("""
## Layer 2 — Process reliability (round 2)

`coverage_net_activation_rate` is the honest-flakiness metric: how often the
deterministic net had to run a check the agents skipped. Every activation is a
case that still completed correctly — and an admission the LLM alone would have
missed evidence. (After the v2 policy change, manual-referral cases *by design*
route extra checks through the net, so some activation is now policy, not flake.)
"""),
        code("""
process = pd.DataFrame({
    "granite4.1:8b (v2)": process_metrics(df_granite),
    "qwen3.5:9b (v2)": process_metrics(df_qwen),
})
process
"""),
        md("""
## Layer 3 — Report quality (round 2)

Deterministic groundedness is in the table above (share of dollar figures in
reports that trace to stored evidence). The cross-family judge adds a rubric
score per report:
"""),
        code("""
judge_granite = judge_run("granite4_1_8b_v2", judge_model="qwen3.5:9b")
judge_qwen = judge_run("qwen3_5_9b_v2", judge_model="granite4.1:8b")

pd.DataFrame({
    "granite v2 reports (judged by qwen)": judge_granite[["groundedness", "completeness", "clarity"]].mean().round(2),
    "qwen v2 reports (judged by granite)": judge_qwen[["groundedness", "completeness", "clarity"]].mean().round(2),
})
"""),
        code("""
# Persist the headline numbers for the README
summary_path = settings.artifacts_dir / "eval" / "headline_summary.json"
headline = {
    "v1": {
        "granite4.1:8b": decision_metrics(df_granite_v1),
        "qwen3.5:9b": decision_metrics(df_qwen_v1),
    },
    "v2": {
        "granite4.1:8b": {**decision_metrics(df_granite), **process_metrics(df_granite)},
        "qwen3.5:9b": {**decision_metrics(df_qwen), **process_metrics(df_qwen)},
    },
    "judge_v2": {
        "granite_reports_by_qwen": judge_granite[["groundedness", "completeness", "clarity"]].mean().round(2).to_dict(),
        "qwen_reports_by_granite": judge_qwen[["groundedness", "completeness", "clarity"]].mean().round(2).to_dict(),
    },
}
summary_path.write_text(json.dumps(headline, indent=2))
print(json.dumps(headline, indent=2))
"""),
        md("""
## Reading the results honestly

**What the evaluation establishes:**

- The system completes every case end-to-end on an 8 GB laptop GPU with validated
  structured outputs and grounded reports — the reliability engineering does its job.
- Round 1 was mediocre on decisions (that's what evals are for). The misses were
  *systematic and diagnosable*, and two targeted, label-free fixes — one policy,
  one prompt — produced the round-2 numbers above. The full v1 → v2 trail is in
  `artifacts/eval/`.
- The two-backbone A/B quantifies the speed/quality trade-off between
  granite4.1:8b and qwen3.5:9b on identical cases.

**Limitations:**

- n=18 cases, two eval rounds. Enough to find and fix systematic failures; not
  enough for tight confidence intervals — per-typology recall rests on 1-3 cases.
- One iteration round against a fixed case set risks overfitting the prompt to
  these typologies; a real programme would hold out new typology instances.
- The ledger is synthetic and textbook-shaped; real laundering is adversarial.
  The OFAC list is real, but the planted variants encode one evasion style.
- The LLM judge is a 9B model with a rubric — a secondary signal next to the
  deterministic groundedness checker; cross-family judging damps, not eliminates,
  bias.
- Auto-approving the human gate measures the *system's* recommendations;
  whether the human gate catches system mistakes is a user study, not a notebook.
"""),
    ]
    return _nb(cells)


BUILDERS = {
    "01": ("01_data_foundation.ipynb", nb01),
    "02": ("02_tools_and_reliability.ipynb", nb02),
    "03": ("03_investigation_graph.ipynb", nb03),
    "04": ("04_evaluation.ipynb", nb04),
}


def main() -> None:
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    keys = BUILDERS.keys() if which == "all" else [which]
    NB_DIR.mkdir(exist_ok=True)
    for key in keys:
        filename, builder = BUILDERS[key]
        path = NB_DIR / filename
        nbf.write(builder(), path)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
