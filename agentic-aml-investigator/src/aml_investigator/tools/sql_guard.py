"""SQL guard for the agent's ad-hoc ``run_sql`` tool.

Two integrity jobs:
1. Safety — read-only, single-statement SELECTs only.
2. Anti-cheating — the agent must never see ``ground_truth`` (the labels it is
   being evaluated against) or ``evidence``/``case_log`` (its own working notes).
"""

import sqlglot
from sqlglot import exp

from aml_investigator.db import AGENT_VISIBLE_TABLES
from aml_investigator.settings import settings


class SQLGuardError(ValueError):
    """Raised when a query violates the guard; the message is shown to the agent."""


_BANNED_SUBSTRINGS = ("attach", "install", "copy ", "read_", "glob", "getenv", "pragma")


def guard_sql(query: str) -> str:
    """Validate an agent-written query and return it with a row limit applied.

    Raises:
        SQLGuardError: with an agent-readable reason on any violation.
    """
    lowered = query.lower()
    for banned in _BANNED_SUBSTRINGS:
        if banned in lowered:
            raise SQLGuardError(f"Query rejected: '{banned.strip()}' is not allowed.")

    try:
        statements = sqlglot.parse(query, read="duckdb")
    except sqlglot.errors.ParseError as e:
        raise SQLGuardError(f"Query rejected: SQL syntax error: {e}") from e

    if len(statements) != 1:
        raise SQLGuardError("Query rejected: exactly one statement is allowed.")
    stmt = statements[0]
    if not isinstance(stmt, (exp.Select, exp.Union)):
        raise SQLGuardError("Query rejected: only SELECT queries are allowed.")

    cte_names = {cte.alias_or_name.lower() for cte in stmt.find_all(exp.CTE)}
    for table in stmt.find_all(exp.Table):
        name = table.name.lower()
        if name in cte_names:
            continue
        if name not in AGENT_VISIBLE_TABLES:
            raise SQLGuardError(
                f"Query rejected: table '{name}' is not accessible. "
                f"Available tables: {', '.join(sorted(AGENT_VISIBLE_TABLES))}."
            )

    if stmt.args.get("limit") is None:
        return f"SELECT * FROM ({stmt.sql(dialect='duckdb')}) LIMIT {settings.sql_row_limit}"
    return stmt.sql(dialect="duckdb")
