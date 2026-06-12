"""Deterministic eval case selection.

18 cases: all 10 ground-truth (suspicious) accounts, every alerted-but-clean
account (the hard negatives the rules tripped on), and enough top-volume clean
controls to reach 8 negatives. Accounts without alerts get a synthetic
"manual_referral" alert — periodic KYC reviews are exactly how real compliance
teams open cases that no rule fired on.
"""

from typing import Any

import duckdb

N_CLEAN = 8


def eval_cases(con: duckdb.DuckDBPyConnection) -> list[dict[str, Any]]:
    """Build the labeled eval set (order is deterministic)."""
    suspicious = con.execute(
        "SELECT account_id, typology FROM ground_truth ORDER BY account_id"
    ).fetchall()
    hard_negative = con.execute(
        """SELECT DISTINCT a.account_id FROM alerts a
           LEFT JOIN ground_truth g USING (account_id)
           WHERE g.account_id IS NULL ORDER BY a.account_id"""
    ).fetchall()
    clean_ids = [r[0] for r in hard_negative]
    if len(clean_ids) < N_CLEAN:
        filler = con.execute(
            """SELECT t.account_id FROM transactions t
               LEFT JOIN ground_truth g USING (account_id)
               LEFT JOIN alerts al ON al.account_id = t.account_id
               WHERE g.account_id IS NULL AND al.account_id IS NULL
               GROUP BY t.account_id ORDER BY sum(t.amount) DESC LIMIT ?""",
            [N_CLEAN - len(clean_ids)],
        ).fetchall()
        clean_ids += [r[0] for r in filler]

    cases: list[dict[str, Any]] = []
    for account_id, typology in suspicious:
        cases.append({"account_id": account_id, "label": "suspicious", "typology": typology})
    for account_id in clean_ids[:N_CLEAN]:
        cases.append({"account_id": account_id, "label": "clean", "typology": "none"})

    for case in cases:
        alerts = con.execute(
            "SELECT rule, details FROM alerts WHERE account_id = ? ORDER BY alert_id",
            [case["account_id"]],
        ).fetchall()
        case["alerts"] = (
            [{"rule": r, "details": d} for r, d in alerts]
            if alerts
            else [{"rule": "manual_referral", "details": "Periodic KYC review — no rule alert on file"}]
        )
    return cases
