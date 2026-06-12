"""DuckDB warehouse access and the evidence store.

One file-backed DuckDB database holds the ledger (``accounts``, ``transactions``),
the real OFAC list (``sdn``), generated ``alerts``, the hidden ``ground_truth``
labels, the per-case ``evidence`` store, and the ``case_log`` audit trail.

Agents never see ``ground_truth`` — the SQL guard in :mod:`tools.sql_guard`
rejects any query touching it.
"""

import json
from datetime import datetime
from typing import Any

import duckdb

from aml_investigator.settings import settings

SCHEMA_DDL = """
CREATE TABLE IF NOT EXISTS accounts (
    account_id   VARCHAR PRIMARY KEY,
    holder_name  VARCHAR NOT NULL,
    country      VARCHAR NOT NULL,
    account_type VARCHAR NOT NULL,          -- personal | business
    opened_date  DATE NOT NULL
);
CREATE TABLE IF NOT EXISTS transactions (
    txn_id            VARCHAR PRIMARY KEY,
    ts                TIMESTAMP NOT NULL,
    account_id        VARCHAR NOT NULL,     -- the account under our roof
    direction         VARCHAR NOT NULL,     -- in | out
    txn_type          VARCHAR NOT NULL,     -- cash_deposit | cash_withdrawal | transfer | wire | card | salary
    amount            DOUBLE NOT NULL,
    counterparty_id   VARCHAR,              -- ACC-x (internal) or EXT-x (external)
    counterparty_name VARCHAR,
    counterparty_country VARCHAR,
    channel           VARCHAR NOT NULL      -- branch | online | atm | pos
);
CREATE TABLE IF NOT EXISTS sdn (
    ent_num  INTEGER,
    sdn_name VARCHAR,
    sdn_type VARCHAR,
    program  VARCHAR
);
CREATE TABLE IF NOT EXISTS alerts (
    alert_id   VARCHAR PRIMARY KEY,
    account_id VARCHAR NOT NULL,
    rule       VARCHAR NOT NULL,
    details    VARCHAR NOT NULL,
    created_at TIMESTAMP NOT NULL
);
CREATE TABLE IF NOT EXISTS ground_truth (
    account_id VARCHAR PRIMARY KEY,
    typology   VARCHAR NOT NULL,
    details    VARCHAR NOT NULL
);
CREATE TABLE IF NOT EXISTS evidence (
    case_id     VARCHAR NOT NULL,
    evidence_id VARCHAR NOT NULL,
    tool        VARCHAR NOT NULL,
    args        VARCHAR NOT NULL,           -- JSON
    summary     VARCHAR NOT NULL,
    payload     VARCHAR NOT NULL,           -- JSON, full tool result
    created_at  TIMESTAMP NOT NULL,
    PRIMARY KEY (case_id, evidence_id)
);
CREATE TABLE IF NOT EXISTS case_log (
    case_id           VARCHAR NOT NULL,
    account_id        VARCHAR NOT NULL,
    final_disposition VARCHAR NOT NULL,
    risk_score        INTEGER,
    typology          VARCHAR,
    human_decision    VARCHAR,
    report_path       VARCHAR,
    closed_at         TIMESTAMP NOT NULL
);
"""

# Tables the investigation agents are allowed to query ad hoc.
AGENT_VISIBLE_TABLES: frozenset[str] = frozenset({"accounts", "transactions", "alerts", "sdn"})


def connect(read_only: bool = False) -> duckdb.DuckDBPyConnection:
    """Open the warehouse, creating the schema on first use."""
    settings.warehouse_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(settings.warehouse_path), read_only=read_only)
    if not read_only:
        con.execute(SCHEMA_DDL)
    return con


def store_evidence(
    con: duckdb.DuckDBPyConnection,
    case_id: str,
    tool: str,
    args: dict[str, Any],
    summary: str,
    payload: dict[str, Any],
) -> str:
    """Persist a full tool result and return its EV-xx id.

    The agent only ever sees the compact summary; the full payload stays in the
    store and is what the groundedness checker verifies reports against.
    """
    n = con.execute("SELECT count(*) FROM evidence WHERE case_id = ?", [case_id]).fetchone()[0]
    evidence_id = f"EV-{n + 1:02d}"
    con.execute(
        "INSERT INTO evidence VALUES (?, ?, ?, ?, ?, ?, ?)",
        [
            case_id,
            evidence_id,
            tool,
            json.dumps(args),
            summary,
            json.dumps(payload, default=str),
            datetime.now(),
        ],
    )
    return evidence_id


def fetch_evidence(con: duckdb.DuckDBPyConnection, case_id: str) -> list[dict[str, Any]]:
    """All evidence rows for a case, payloads parsed."""
    rows = con.execute(
        "SELECT evidence_id, tool, args, summary, payload FROM evidence "
        "WHERE case_id = ? ORDER BY evidence_id",
        [case_id],
    ).fetchall()
    return [
        {
            "evidence_id": r[0],
            "tool": r[1],
            "args": json.loads(r[2]),
            "summary": r[3],
            "payload": json.loads(r[4]),
        }
        for r in rows
    ]
