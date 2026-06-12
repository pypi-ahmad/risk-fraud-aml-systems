import pytest

from aml_investigator.settings import settings
from aml_investigator.tools.sql_guard import SQLGuardError, guard_sql


def test_select_passes_and_gets_limit():
    out = guard_sql("SELECT account_id FROM transactions WHERE amount > 100")
    assert out.lower().startswith("select")
    assert f"LIMIT {settings.sql_row_limit}" in out


def test_existing_limit_is_kept():
    out = guard_sql("SELECT account_id FROM transactions LIMIT 5")
    assert "LIMIT 5" in out and str(settings.sql_row_limit) not in out


def test_cte_over_visible_tables_passes():
    out = guard_sql(
        "WITH t AS (SELECT account_id, amount FROM transactions) "
        "SELECT account_id, sum(amount) FROM t GROUP BY 1"
    )
    assert "WITH" in out.upper()


@pytest.mark.parametrize(
    "query",
    [
        "DROP TABLE transactions",
        "INSERT INTO transactions VALUES (1)",
        "UPDATE accounts SET country = 'US'",
        "DELETE FROM alerts",
        "SELECT 1; SELECT 2",
        "SELECT * FROM ground_truth",  # the anti-cheating rule
        "SELECT * FROM evidence",
        "SELECT * FROM case_log",
        "SELECT * FROM read_csv('x.csv')",
        "ATTACH 'other.db'",
    ],
)
def test_rejected(query):
    with pytest.raises(SQLGuardError):
        guard_sql(query)
