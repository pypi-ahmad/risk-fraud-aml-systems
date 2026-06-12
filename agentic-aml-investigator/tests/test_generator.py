import pytest

from aml_investigator.data.generator import generate_ledger
from aml_investigator.settings import settings


@pytest.fixture(scope="module")
def ledger():
    if not settings.sdn_csv_path.exists():
        pytest.skip("SDN csv not downloaded (run data/ofac.download_sdn first)")
    return generate_ledger(seed=42)


def test_deterministic(ledger):
    again = generate_ledger(seed=42)
    assert len(again.transactions) == len(ledger.transactions)
    assert again.transactions[0] == ledger.transactions[0]
    assert again.transactions[-1] == ledger.transactions[-1]
    assert again.ground_truth == ledger.ground_truth


def test_typology_counts(ledger):
    by_typ: dict[str, int] = {}
    for row in ledger.ground_truth:
        by_typ[row["typology"]] = by_typ.get(row["typology"], 0) + 1
    assert by_typ == {
        "structuring": 2,
        "velocity_burst": 2,
        "circular_transfers": 3,
        "funnel_account": 1,
        "sanctioned_counterparty": 2,
    }


def test_flagged_accounts_are_distinct(ledger):
    ids = [row["account_id"] for row in ledger.ground_truth]
    assert len(ids) == len(set(ids))


def test_no_self_transfers(ledger):
    assert not any(
        t["counterparty_id"] == t["account_id"]
        for t in ledger.transactions
        if t["counterparty_id"]
    )
