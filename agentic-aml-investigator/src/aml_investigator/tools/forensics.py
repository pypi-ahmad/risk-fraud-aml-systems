"""Deterministic forensic tools the agents call.

Design rules that make a 8B local model viable here:

- Tools do the analytical heavy lifting in SQL/rapidfuzz; the LLM only decides
  *which* tool to call and interprets compact summaries.
- Every full result is persisted to the evidence store and the agent only sees
  ``[EV-xx] <short summary>`` — state compression keeps the context small, and
  the stored payload is what reports are later verified against. The model
  cannot fabricate evidence rows.
- The active case id travels in a contextvar (cases are investigated one at a
  time), so one compiled graph serves every case.
"""

import functools
import json
import time

import duckdb
from langchain_core.tools import BaseTool, tool
from rapidfuzz import fuzz, process, utils

from aml_investigator import telemetry
from aml_investigator.db import store_evidence
from aml_investigator.settings import settings
from aml_investigator.tools.sql_guard import SQLGuardError, guard_sql

# Module-level (not a ContextVar): LangGraph may run nodes/tools in worker
# threads, where ContextVar values set in the caller don't reliably propagate.
# Cases are investigated strictly sequentially, so a global is correct here.
_ACTIVE_CASE = {"id": "interactive"}


def set_active_case(case_id: str) -> None:
    telemetry_case_id = telemetry.current().case_id
    if telemetry_case_id != "interactive" and telemetry_case_id != case_id:
        raise RuntimeError(
            "Active case mismatch between telemetry scope and tool scope. "
            "Concurrent investigations are not supported with module-level case state."
        )
    _ACTIVE_CASE["id"] = case_id


def get_active_case() -> str:
    return _ACTIVE_CASE["id"]

# Static schema documentation injected into the investigator's system prompt —
# cheaper and more reliable than list_tables/describe_table tool round-trips.
WAREHOUSE_SCHEMA_DOC = """
Tables you can query with run_sql (DuckDB SQL, SELECT only, auto-limited to 50 rows):
- accounts(account_id, holder_name, country, account_type, opened_date)
- transactions(txn_id, ts, account_id, direction['in'|'out'], txn_type['cash_deposit'|'cash_withdrawal'|'transfer'|'wire'|'card'|'salary'], amount, counterparty_id, counterparty_name, counterparty_country, channel)
- alerts(alert_id, account_id, rule, details, created_at)
- sdn(ent_num, sdn_name, sdn_type, program)  -- the real OFAC sanctions list
""".strip()


def _record(con: duckdb.DuckDBPyConnection, tool_name: str, args: dict, summary: str, payload: dict) -> str:
    case_id = get_active_case()
    evidence_id = store_evidence(con, case_id, tool_name, args, summary, payload)
    return f"[{evidence_id}] {summary}"


def make_tools(con: duckdb.DuckDBPyConnection) -> dict[str, BaseTool]:
    """Build the forensic toolset bound to a warehouse connection.

    Returns a name -> tool mapping; the graph uses the dict for deterministic
    direct execution (coverage net) and ``list(...)`` for the ReAct agent.
    """

    sdn_rows_cached = con.execute("SELECT sdn_name, program, ent_num FROM sdn").fetchall()
    sdn_names_cached = [r[0] for r in sdn_rows_cached]

    def timed_tool(fn):
        """Wrap a tool body with telemetry (latency + success/failure)."""

        @functools.wraps(fn)  # preserves the signature so @tool can infer the args schema
        def wrapper(*args, **kwargs):
            t0 = time.perf_counter()
            try:
                result = fn(*args, **kwargs)
            except SQLGuardError as e:
                telemetry.current().record("tool", fn.__name__, time.perf_counter() - t0, ok=False)
                return f"ERROR: {e}"
            except Exception as e:
                telemetry.current().record("tool", fn.__name__, time.perf_counter() - t0, ok=False)
                return f"ERROR: {fn.__name__} failed: {type(e).__name__}: {str(e)[:200]}"
            telemetry.current().record("tool", fn.__name__, time.perf_counter() - t0, ok=True)
            return result

        return wrapper

    @timed_tool
    def profile_account(account_id: str) -> str:
        """Get the KYC profile and behavioural baseline of an account: holder, type,
        age, transaction volumes by type, distinct counterparties and countries.
        Always run this first to understand what 'normal' looks like for the account.

        Args:
            account_id: The internal account id, e.g. 'ACC-0042'.
        """
        acc = con.execute(
            "SELECT account_id, holder_name, country, account_type, opened_date "
            "FROM accounts WHERE account_id = ?", [account_id]
        ).fetchone()
        if acc is None:
            return f"ERROR: account {account_id} not found"
        stats = con.execute(
            """SELECT direction, txn_type, count(*) AS n, round(sum(amount)) AS total
               FROM transactions WHERE account_id = ? GROUP BY 1, 2 ORDER BY total DESC""",
            [account_id],
        ).fetchall()
        extra = con.execute(
            """SELECT count(DISTINCT coalesce(counterparty_id, counterparty_name)) AS n_cps,
                      count(DISTINCT counterparty_country) AS n_countries,
                      count(DISTINCT CAST(ts AS DATE)) AS active_days,
                      min(ts) AS first_txn, max(ts) AS last_txn
               FROM transactions WHERE account_id = ?""",
            [account_id],
        ).fetchone()
        total_in = sum(r[3] for r in stats if r[0] == "in")
        total_out = sum(r[3] for r in stats if r[0] == "out")
        payload = {
            "account": dict(zip(["account_id", "holder_name", "country", "account_type", "opened_date"],
                                [str(v) for v in acc])),
            "by_type": [dict(zip(["direction", "txn_type", "n", "total"], r)) for r in stats],
            "distinct_counterparties": extra[0], "distinct_countries": extra[1],
            "active_days": extra[2], "first_txn": str(extra[3]), "last_txn": str(extra[4]),
            "total_in": total_in, "total_out": total_out,
        }
        summary = (
            f"{account_id} ({acc[3]}, '{acc[1]}', {acc[2]}, opened {acc[4]}): "
            f"total in ${total_in:,.0f} / out ${total_out:,.0f}, "
            f"{extra[0]} counterparties across {extra[1]} countries, active on {extra[2]} days. "
            f"Top flows: " + "; ".join(f"{r[0]} {r[1]} n={r[2]} ${r[3]:,.0f}" for r in stats[:4])
        )
        return _record(con, "profile_account", {"account_id": account_id}, summary, payload)

    @timed_tool
    def velocity_scan(account_id: str, window_days: int = 3) -> str:
        """Detect burst activity: compares the busiest window against the account's
        full-calendar median daily activity. High ratios on a dormant account
        indicate the velocity-burst typology.

        Args:
            account_id: The internal account id, e.g. 'ACC-0042'.
            window_days: Rolling window size in days (default 3).
        """
        rows = con.execute(
            """WITH cal AS (
                   SELECT unnest(generate_series((SELECT min(CAST(ts AS DATE)) FROM transactions),
                                                 (SELECT max(CAST(ts AS DATE)) FROM transactions),
                                                 INTERVAL 1 DAY))::DATE AS d
               ), active AS (
                   SELECT CAST(ts AS DATE) AS d, count(*) AS n, sum(amount) AS amt
                   FROM transactions WHERE account_id = ? GROUP BY 1
               )
               SELECT cal.d, coalesce(n, 0), coalesce(amt, 0) FROM cal
               LEFT JOIN active USING (d) ORDER BY cal.d""",
            [account_id],
        ).fetchall()
        ns = [r[1] for r in rows]
        med = sorted(ns)[len(ns) // 2]
        windows = [
            (str(rows[i][0]), sum(ns[i:i + window_days]), round(sum(r[2] for r in rows[i:i + window_days])))
            for i in range(len(rows) - window_days + 1)
        ]
        peak = max(windows, key=lambda w: w[1])
        daily_baseline = med if med > 0 else 0.5  # avoid div-by-zero on dormant accounts
        ratio = peak[1] / (daily_baseline * window_days)
        top_days = sorted(((str(r[0]), r[1], round(r[2])) for r in rows), key=lambda x: -x[1])[:3]
        payload = {"median_daily_txns": med, "peak_window_start": peak[0],
                   "peak_window_txns": peak[1], "peak_window_amount": peak[2],
                   "burst_ratio": round(ratio, 1), "top_days": top_days}
        summary = (
            f"{account_id}: median {med} txns/day (full calendar); busiest {window_days}-day window "
            f"starts {peak[0]} with {peak[1]} txns / ${peak[2]:,.0f} — burst ratio {ratio:.0f}x baseline. "
            f"Top days: {top_days}"
        )
        return _record(con, "velocity_scan", {"account_id": account_id, "window_days": window_days},
                       summary, payload)

    @timed_tool
    def structuring_scan(account_id: str) -> str:
        """Detect structuring: cash deposits kept just under the $10,000 CTR
        reporting threshold. Reports the count, sum and date span of sub-threshold
        deposits versus deposits at or above the threshold.

        Args:
            account_id: The internal account id, e.g. 'ACC-0042'.
        """
        lo = settings.structuring_band * settings.ctr_threshold
        sub = con.execute(
            """SELECT count(*), coalesce(round(sum(amount)), 0), min(CAST(ts AS DATE)), max(CAST(ts AS DATE)),
                      coalesce(max(cnt), 0)
               FROM (SELECT ts, amount, count(*) OVER (PARTITION BY CAST(ts AS DATE)) AS cnt
                     FROM transactions
                     WHERE account_id = ? AND txn_type = 'cash_deposit' AND amount >= ? AND amount < ?)""",
            [account_id, lo, settings.ctr_threshold],
        ).fetchone()
        over = con.execute(
            "SELECT count(*), coalesce(round(sum(amount)), 0) FROM transactions "
            "WHERE account_id = ? AND txn_type = 'cash_deposit' AND amount >= ?",
            [account_id, settings.ctr_threshold],
        ).fetchone()
        payload = {"sub_threshold_count": sub[0], "sub_threshold_total": sub[1],
                   "first_date": str(sub[2]), "last_date": str(sub[3]), "max_per_day": sub[4],
                   "at_or_over_count": over[0], "at_or_over_total": over[1],
                   "band": f"${lo:,.0f}-${settings.ctr_threshold:,.0f}"}
        summary = (
            f"{account_id}: {sub[0]} cash deposits in the ${lo:,.0f}-${settings.ctr_threshold:,.0f} band "
            f"totalling ${sub[1]:,.0f}"
            + (f" between {sub[2]} and {sub[3]} (max {sub[4]}/day)" if sub[0] else "")
            + f"; {over[0]} deposits >= ${settings.ctr_threshold:,.0f} (${over[1]:,.0f})."
        )
        return _record(con, "structuring_scan", {"account_id": account_id}, summary, payload)

    @timed_tool
    def counterparty_network(account_id: str) -> str:
        """Map who the account transacts with: top counterparties by volume,
        sender/receiver concentration (funnel detection), and circular transfer
        rings (A -> B -> C -> A) involving the account.

        Args:
            account_id: The internal account id, e.g. 'ACC-0042'.
        """
        top = con.execute(
            """SELECT direction, coalesce(counterparty_id, counterparty_name) AS cp,
                      max(counterparty_name) AS cp_name, max(counterparty_country) AS country,
                      count(*) AS n, round(sum(amount)) AS total
               FROM transactions WHERE account_id = ? AND counterparty_name IS NOT NULL
               GROUP BY 1, 2 ORDER BY total DESC LIMIT 10""",
            [account_id],
        ).fetchall()
        conc = con.execute(
            """SELECT count(DISTINCT CASE WHEN direction='in' THEN coalesce(counterparty_id, counterparty_name) END),
                      count(DISTINCT CASE WHEN direction='out' THEN coalesce(counterparty_id, counterparty_name) END)
               FROM transactions WHERE account_id = ?""",
            [account_id],
        ).fetchone()
        cycles = con.execute(
            """WITH edges AS (
                   SELECT DISTINCT account_id AS src, counterparty_id AS dst
                   FROM transactions
                   WHERE direction = 'out' AND txn_type = 'transfer'
                         AND counterparty_id LIKE 'ACC-%' AND counterparty_id != account_id
               )
               SELECT e1.src, e1.dst, e2.dst
               FROM edges e1 JOIN edges e2 ON e1.dst = e2.src JOIN edges e3 ON e2.dst = e3.src
               WHERE e3.dst = e1.src AND e1.src = ?
                     AND e1.src != e1.dst AND e1.src != e2.dst AND e1.dst != e2.dst""",
            [account_id],
        ).fetchall()
        ring_amounts = []
        for c in cycles[:3]:
            amt = con.execute(
                """SELECT round(sum(amount)) FROM transactions
                   WHERE direction = 'out' AND txn_type = 'transfer'
                         AND ((account_id = ? AND counterparty_id = ?) OR (account_id = ? AND counterparty_id = ?)
                              OR (account_id = ? AND counterparty_id = ?))""",
                [c[0], c[1], c[1], c[2], c[2], c[0]],
            ).fetchone()[0]
            ring_amounts.append({"ring": list(c), "total_transferred": amt})
        payload = {
            "top_counterparties": [dict(zip(["direction", "cp_id", "name", "country", "n", "total"], r)) for r in top],
            "distinct_senders": conc[0], "distinct_receivers": conc[1],
            "circular_rings": ring_amounts,
        }
        summary = (
            f"{account_id}: {conc[0]} distinct senders, {conc[1]} distinct receivers. "
            f"Top counterparties: " + "; ".join(f"{r[0]} {r[2] or r[1]} ({r[3]}) ${r[5]:,.0f}" for r in top[:4])
            + (f". CIRCULAR RING(S) DETECTED: {ring_amounts}" if ring_amounts else ". No circular rings found.")
        )
        return _record(con, "counterparty_network", {"account_id": account_id}, summary, payload)

    @timed_tool
    def sanctions_check(account_id: str) -> str:
        """Screen every counterparty of the account against the real OFAC SDN list
        using fuzzy name matching (catches transliteration/spelling variants that
        exact matching misses). Mandatory in every investigation.

        Args:
            account_id: The internal account id, e.g. 'ACC-0042'.
        """
        names = [r[0] for r in con.execute(
            "SELECT DISTINCT counterparty_name FROM transactions "
            "WHERE account_id = ? AND counterparty_name IS NOT NULL", [account_id]).fetchall()]
        hits = []
        sdn_rows = sdn_rows_cached
        sdn_names = sdn_names_cached
        for name in names:
            for sdn_name, score, idx in process.extract(
                name, sdn_names, scorer=fuzz.token_sort_ratio,
                processor=utils.default_process,  # case/punct-insensitive — SDN names are ALL CAPS
                score_cutoff=settings.sanctions_fuzzy_cutoff, limit=2,
            ):
                hits.append({"counterparty_name": name, "sdn_name": sdn_name,
                             "score": round(score, 1), "program": sdn_rows[idx][1],
                             "ent_num": sdn_rows[idx][2]})
        hits.sort(key=lambda h: -h["score"])
        payload = {"counterparties_screened": len(names),
                   "cutoff": settings.sanctions_fuzzy_cutoff, "hits": hits}
        summary = (
            f"{account_id}: screened {len(names)} counterparty names against {len(sdn_names):,} OFAC SDN entries "
            f"(fuzzy cutoff {settings.sanctions_fuzzy_cutoff}). "
            + (f"{len(hits)} HIT(S): " + "; ".join(
                f"'{h['counterparty_name']}' ~ '{h['sdn_name']}' (score {h['score']}, {h['program']})"
                for h in hits[:4]) if hits else "No matches.")
        )
        return _record(con, "sanctions_check", {"account_id": account_id}, summary, payload)

    @timed_tool
    def run_sql(query: str) -> str:
        """Run an ad-hoc read-only SQL query against the transaction warehouse for
        anything the specialised scans don't cover. SELECT only; results are
        limited to 50 rows. Use the documented schema.

        Args:
            query: A DuckDB SELECT statement.
        """
        safe_sql = guard_sql(query)  # raises SQLGuardError -> surfaced to the agent
        df = con.execute(safe_sql).fetch_df()
        rendered = df.to_string(max_rows=20, max_cols=12)
        if len(rendered) > 1500:
            rendered = rendered[:1500] + "\n... (truncated)"
        payload = {"query": query, "rows": json.loads(df.head(50).to_json(orient="records"))}
        summary = f"run_sql returned {len(df)} row(s) for: {query[:120]}"
        record_line = _record(con, "run_sql", {"query": query}, summary, payload)
        return f"{record_line}\n{rendered}"

    return {
        "profile_account": tool(profile_account),
        "velocity_scan": tool(velocity_scan),
        "structuring_scan": tool(structuring_scan),
        "counterparty_network": tool(counterparty_network),
        "sanctions_check": tool(sanctions_check),
        "run_sql": tool(run_sql),
    }
