# agentic-aml-investigator

Multi-agent AML investigation copilot: LangGraph + local Ollama over a DuckDB
warehouse. Goal: real end-to-end agent system on 8 GB VRAM, evaluated against
labeled ground truth.

## Commands

```bash
uv sync                                          # env (Python 3.12)
uv run pytest -q                                 # unit tests (no LLM needed)
uv run python scripts/build_notebooks.py all     # (re)author notebooks
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/01_*.ipynb
AML_AGENT_MODEL=qwen3.5:9b uv run python -c ...  # switch backbone via env
```

## Structure

- `src/aml_investigator/` — package: `graph/build.py` (the StateGraph),
  `tools/forensics.py` (forensic tools + evidence store), `llm.py`
  (structured-output escalation ladder), `evaluation/` (runner, metrics, judge)
- `notebooks/01..04` — tutorial notebooks, authored by `scripts/build_notebooks.py`,
  executed for real; never hand-edit the .ipynb, edit the builder and re-execute
- `data/aml.duckdb` — generated warehouse (gitignored); `data/raw/sdn.csv` — real
  OFAC snapshot (committed)
- `artifacts/` — eval results, reports, checkpoints (checkpoints gitignored)

## Conventions & decisions

- All structured LLM output goes through `structured_llm_call` (function_calling
  → json_schema → deterministic fallback). Never call `with_structured_output`
  directly in graph code.
- Agents must never read `ground_truth`/`evidence`/`case_log` — enforced by
  `tools/sql_guard.py`; keep new tables in the deny-set unless agent-facing.
- Cases run sequentially: active case id + telemetry are module-level globals
  (LangGraph worker threads break ContextVar propagation).
- Default backbone granite4.1:8b (measured fastest reliable tool-caller);
  qwen3.5:9b is the A/B challenger; judges are cross-family.

## Gotchas

- Ollama 0.30.x + thinking models: `think:false` silently disables `format=`
  constrained decoding — leave thinking enabled for json_schema calls (nb02 demo).
- `SqliteSaver`: build over a long-lived `sqlite3.connect(..., check_same_thread=False)`,
  not the context-manager form (it fights Jupyter cell lifetimes).
- Regenerating the warehouse reshuffles account ids — notebooks look ids up from
  `ground_truth` at runtime, never hardcode them.

## Verify before done

- `uv run pytest -q` green; changed notebooks re-executed top-to-bottom with no
  error outputs; no `ground_truth` leakage into any agent-visible path;
  eval artifacts regenerated if metrics-affecting code changed.
