"""Seeded synthetic transaction ledger with injected AML typologies.

Why synthetic: real SARs and real AML case data are never public, and the one
common public option (PaySim) contains a single fraud pattern, which would make
every investigation identical. A seeded generator gives five distinct, labeled
typologies and regenerates byte-identically on any machine (``seed=42``).

The five injected typologies (recorded in the hidden ``ground_truth`` table):

1. ``structuring``            — repeated cash deposits just under the $10k CTR threshold
2. ``velocity_burst``         — dormant account suddenly cycling money in and out
3. ``circular_transfers``     — a 3-account ring A -> B -> C -> A
4. ``funnel_account``         — many unrelated senders converge, one large wire exits
5. ``sanctioned_counterparty``— wires to near-variants of REAL OFAC SDN names

Plus deliberately "alert-looking but clean" hard negatives (cash-intensive
businesses, payroll bursts) so the agent cannot win by always escalating.
"""

import random
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta

import pandas as pd
from faker import Faker
from loguru import logger

from aml_investigator.data.ofac import load_sdn_frame
from aml_investigator.settings import settings

HIGH_RISK_COUNTRIES = ["IR", "KP", "MM", "SY", "PA"]
SAFE_COUNTRIES = ["US", "US", "US", "US", "CA", "GB", "DE", "FR"]  # US-weighted


@dataclass
class Ledger:
    accounts: list[dict] = field(default_factory=list)
    transactions: list[dict] = field(default_factory=list)
    ground_truth: list[dict] = field(default_factory=list)
    _txn_seq: int = 0

    def add_txn(
        self,
        ts: datetime,
        account_id: str,
        direction: str,
        txn_type: str,
        amount: float,
        counterparty_id: str | None = None,
        counterparty_name: str | None = None,
        counterparty_country: str | None = None,
        channel: str = "online",
    ) -> None:
        self._txn_seq += 1
        self.transactions.append(
            {
                "txn_id": f"TXN-{self._txn_seq:06d}",
                "ts": ts,
                "account_id": account_id,
                "direction": direction,
                "txn_type": txn_type,
                "amount": round(amount, 2),
                "counterparty_id": counterparty_id,
                "counterparty_name": counterparty_name,
                "counterparty_country": counterparty_country,
                "channel": channel,
            }
        )


def _perturb_name(name: str, rng: random.Random) -> str:
    """Create a near-variant of an SDN name (one letter swap + casing drift).

    Close enough that fuzzy matching (token_sort_ratio >= 87) catches it, far
    enough that exact-match screening would miss it — i.e. realistic evasion.
    """
    chars = list(name.title())
    letter_positions = [i for i, c in enumerate(chars) if c.isalpha()]
    i = rng.choice(letter_positions[2:]) if len(letter_positions) > 2 else letter_positions[-1]
    chars[i] = rng.choice("aeiou") if chars[i].lower() not in "aeiou" else rng.choice("bdgmnrst")
    return "".join(chars)


def _ts(d: date, rng: random.Random, start_h: int = 8, end_h: int = 20) -> datetime:
    return datetime(d.year, d.month, d.day, rng.randint(start_h, end_h), rng.randint(0, 59), rng.randint(0, 59))


def generate_ledger(seed: int | None = None) -> Ledger:
    """Build the full ledger. Deterministic for a given seed."""
    seed = seed if seed is not None else settings.seed
    rng = random.Random(seed)
    fake = Faker()
    Faker.seed(seed)

    end = date.fromisoformat(settings.ledger_end_date)
    start = end - timedelta(days=settings.n_days - 1)
    days = [start + timedelta(days=i) for i in range(settings.n_days)]

    ledger = Ledger()

    # ---------- accounts ----------
    n_business = settings.n_accounts // 5
    for i in range(settings.n_accounts):
        is_business = i < n_business
        ledger.accounts.append(
            {
                "account_id": f"ACC-{i + 1:04d}",
                "holder_name": fake.company() if is_business else fake.name(),
                "country": rng.choice(SAFE_COUNTRIES),
                "account_type": "business" if is_business else "personal",
                "opened_date": end - timedelta(days=rng.randint(120, 3000)),
            }
        )
    accounts = ledger.accounts
    personal = [a for a in accounts if a["account_type"] == "personal"]
    business = [a for a in accounts if a["account_type"] == "business"]

    # Reserve specific accounts for typologies (disjoint by construction).
    structuring_accs = [personal[5], personal[6]]
    velocity_accs = [personal[12], personal[13]]
    ring_accs = [personal[20], personal[21], personal[22]]
    funnel_acc = personal[30]
    sanctions_accs = [personal[40], business[10]]
    flagged_ids = {
        a["account_id"]
        for a in structuring_accs + velocity_accs + ring_accs + [funnel_acc] + sanctions_accs
    }
    # Hard negatives: legitimate but alert-looking. Documented, not labeled suspicious.
    cash_biz = business[:3]  # cash-intensive (laundromat-style) businesses
    payroll_biz = business[3:5]  # monthly payroll bursts

    # ---------- baseline activity for every account ----------
    for acc in accounts:
        acc_id = acc["account_id"]
        dormant = acc_id in {a["account_id"] for a in velocity_accs}
        if acc["account_type"] == "personal":
            salary = rng.uniform(2800, 8500)
            for d in days:
                if d.day == 25:  # payday
                    ledger.add_txn(_ts(d, rng), acc_id, "in", "salary", salary,
                                   counterparty_name=fake.company(), counterparty_country="US")
                if dormant:
                    continue  # dormant accounts: salary only, no spending pattern
                for _ in range(rng.randint(0, 3)):  # daily card spend
                    ledger.add_txn(_ts(d, rng), acc_id, "out", "card", rng.uniform(4, 220),
                                   counterparty_name=fake.company(), counterparty_country="US", channel="pos")
                if rng.random() < 0.08:  # occasional ATM withdrawal
                    ledger.add_txn(_ts(d, rng), acc_id, "out", "cash_withdrawal",
                                   rng.choice([40, 60, 100, 200]), channel="atm")
                if rng.random() < 0.05:  # occasional transfer to another internal account
                    other = rng.choice(accounts)
                    while other["account_id"] == acc_id:
                        other = rng.choice(accounts)
                    ledger.add_txn(_ts(d, rng), acc_id, "out", "transfer", rng.uniform(50, 1200),
                                   counterparty_id=other["account_id"],
                                   counterparty_name=other["holder_name"], counterparty_country="US")
        else:  # business
            for d in days:
                for _ in range(rng.randint(1, 4)):  # customer revenue
                    ledger.add_txn(_ts(d, rng), acc_id, "in", "transfer", rng.uniform(800, 12000),
                                   counterparty_name=fake.company(), counterparty_country="US")
                if rng.random() < 0.4:  # supplier payments
                    ledger.add_txn(_ts(d, rng), acc_id, "out", "wire", rng.uniform(2000, 25000),
                                   counterparty_name=fake.company(),
                                   counterparty_country=rng.choice(SAFE_COUNTRIES))

    # ---------- hard negatives ----------
    for acc in cash_biz:  # daily cash deposits, ABOVE and BELOW threshold, stable for months
        for d in days:
            for _ in range(rng.randint(1, 3)):
                ledger.add_txn(_ts(d, rng), acc["account_id"], "in", "cash_deposit",
                               rng.uniform(1500, 14000), channel="branch")
    for acc in payroll_biz:  # month-end burst of many outgoing salary transfers
        employees = [(fake.name(), rng.uniform(2500, 7000)) for _ in range(rng.randint(15, 25))]
        for d in days:
            if d.day == 28:
                for name, sal in employees:
                    ledger.add_txn(_ts(d, rng), acc["account_id"], "out", "transfer", sal,
                                   counterparty_name=name, counterparty_country="US")

    # ---------- typology 1: structuring ----------
    for acc in structuring_accs:
        burst_start = rng.randint(20, 60)
        n_deposits = rng.randint(12, 16)
        for j in range(n_deposits):
            d = days[burst_start + j // 2]  # ~2 per day
            ledger.add_txn(_ts(d, rng), acc["account_id"], "in", "cash_deposit",
                           rng.uniform(0.87, 0.995) * settings.ctr_threshold, channel="branch")
        wire_day = days[burst_start + n_deposits // 2 + 2]
        ledger.add_txn(_ts(wire_day, rng), acc["account_id"], "out", "wire",
                       n_deposits * 9300 * rng.uniform(0.9, 0.98),
                       counterparty_id=f"EXT-{rng.randint(1000, 9999)}",
                       counterparty_name=fake.name(),
                       counterparty_country=rng.choice(["PA", "AE", "HK"]))
        ledger.ground_truth.append(
            {"account_id": acc["account_id"], "typology": "structuring",
             "details": f"{n_deposits} cash deposits in $8.7k-$9.95k band, aggregated wire out"}
        )

    # ---------- typology 2: velocity burst ----------
    for acc in velocity_accs:
        burst_start = rng.randint(60, 80)
        total = 0.0
        for j in range(rng.randint(14, 20)):  # 3-day in/out cycle
            d = days[min(burst_start + j // 6, settings.n_days - 1)]
            amt = rng.uniform(3000, 9500)
            sender = rng.choice(accounts)
            ledger.add_txn(_ts(d, rng), acc["account_id"], "in", "transfer", amt,
                           counterparty_id=sender["account_id"],
                           counterparty_name=sender["holder_name"], counterparty_country="US")
            ledger.add_txn(_ts(d, rng, 10, 22), acc["account_id"], "out", "wire", amt * rng.uniform(0.93, 0.99),
                           counterparty_id=f"EXT-{rng.randint(1000, 9999)}",
                           counterparty_name=fake.name(),
                           counterparty_country=rng.choice(HIGH_RISK_COUNTRIES))
            total += amt
        ledger.ground_truth.append(
            {"account_id": acc["account_id"], "typology": "velocity_burst",
             "details": f"dormant account, ~${total:,.0f} cycled in/out within ~3 days"}
        )

    # ---------- typology 3: circular transfers ----------
    ring_ids = [a["account_id"] for a in ring_accs]
    for cycle in range(6):
        d = days[10 + cycle * 12]
        amt = rng.uniform(12000, 18000)
        for k in range(3):
            src, dst = ring_accs[k], ring_accs[(k + 1) % 3]
            ledger.add_txn(_ts(days[10 + cycle * 12 + k], rng), src["account_id"], "out", "transfer",
                           amt * rng.uniform(0.96, 0.995),
                           counterparty_id=dst["account_id"],
                           counterparty_name=dst["holder_name"], counterparty_country="US")
    for acc in ring_accs:
        ledger.ground_truth.append(
            {"account_id": acc["account_id"], "typology": "circular_transfers",
             "details": f"3-account ring {' -> '.join(ring_ids)}, 6 cycles of ~$12-18k"}
        )

    # ---------- typology 4: funnel account ----------
    senders = rng.sample([a for a in personal if a["account_id"] not in flagged_ids], 17)
    funnel_total = 0.0
    for j, sender in enumerate(senders):
        d = days[35 + j % 10]
        amt = rng.uniform(1800, 4900)
        ledger.add_txn(_ts(d, rng), funnel_acc["account_id"], "in", "transfer", amt,
                       counterparty_id=sender["account_id"],
                       counterparty_name=sender["holder_name"], counterparty_country="US")
        funnel_total += amt
    ledger.add_txn(_ts(days[48], rng), funnel_acc["account_id"], "out", "wire", funnel_total * 0.96,
                   counterparty_id=f"EXT-{rng.randint(1000, 9999)}",
                   counterparty_name=fake.name(), counterparty_country="AE")
    ledger.ground_truth.append(
        {"account_id": funnel_acc["account_id"], "typology": "funnel_account",
         "details": f"17 unrelated senders, ${funnel_total:,.0f} funneled then wired out"}
    )

    # ---------- typology 5: sanctioned counterparty (REAL SDN near-variants) ----------
    sdn = load_sdn_frame()
    candidates = sdn[(sdn["sdn_name"].str.len() >= 14) & (sdn["sdn_name"].str.len() <= 40)]
    picked = candidates.iloc[[100, 1500, 4000, 7000]]  # deterministic picks
    sdn_pairs = [(row.sdn_name, _perturb_name(row.sdn_name, rng)) for row in picked.itertuples()]
    for idx, acc in enumerate(sanctions_accs):
        variants = sdn_pairs[idx * 2 : idx * 2 + 2]
        for sdn_name, variant in variants:
            for _ in range(rng.randint(2, 3)):
                d = rng.choice(days[15:75])
                ledger.add_txn(_ts(d, rng), acc["account_id"], "out", "wire", rng.uniform(8000, 32000),
                               counterparty_id=f"EXT-{rng.randint(1000, 9999)}",
                               counterparty_name=variant,
                               counterparty_country=rng.choice(["AE", "TR", "HK"]))
        ledger.ground_truth.append(
            {"account_id": acc["account_id"], "typology": "sanctioned_counterparty",
             "details": "wires to near-variants of real SDN names: "
                        + "; ".join(f"'{v}' ~ '{s}'" for s, v in variants)}
        )

    logger.info(
        f"Generated {len(ledger.accounts)} accounts, {len(ledger.transactions):,} transactions, "
        f"{len(ledger.ground_truth)} labeled accounts"
    )
    return ledger


ALERT_RULES_SQL = {
    "sub_threshold_deposits": """
        SELECT account_id,
               'R1: ' || count(*) || ' cash deposits in the $8.5k-$10k band within the period' AS details
        FROM transactions
        WHERE txn_type = 'cash_deposit' AND amount BETWEEN 8500 AND 9999.99
        GROUP BY account_id HAVING count(*) >= 5
    """,
    "velocity_spike": """
        -- Baseline = median daily txn count over the FULL calendar (zero-activity
        -- days included), otherwise a dormant account's own burst inflates its
        -- baseline and the rule misses exactly the accounts it exists for.
        WITH cal AS (
            SELECT unnest(generate_series(
                (SELECT min(CAST(ts AS DATE)) FROM transactions),
                (SELECT max(CAST(ts AS DATE)) FROM transactions),
                INTERVAL 1 DAY))::DATE AS d
        ), active AS (
            SELECT account_id, CAST(ts AS DATE) AS d, count(*) AS n, sum(amount) AS amt
            FROM transactions GROUP BY 1, 2
        ), daily AS (
            SELECT acc.account_id, cal.d, coalesce(active.n, 0) AS n, coalesce(active.amt, 0) AS amt
            FROM accounts acc CROSS JOIN cal
            LEFT JOIN active ON active.account_id = acc.account_id AND active.d = cal.d
        ), base AS (
            SELECT account_id, median(n) AS med_n FROM daily GROUP BY 1
        )
        SELECT daily.account_id,
               'R2: daily activity spike (' || max(daily.n) || ' txns, $' ||
               CAST(round(max(daily.amt)) AS INTEGER) || ') vs median ' || max(base.med_n) || '/day' AS details
        FROM daily JOIN base USING (account_id)
        WHERE daily.n >= greatest(8, 6 * base.med_n) AND daily.amt > 20000
        GROUP BY daily.account_id
    """,
    "large_high_risk_wire": f"""
        SELECT account_id,
               'R3: ' || count(*) || ' outbound wires >= $8k to high-risk countries ('
               || string_agg(DISTINCT counterparty_country, ',') || ')' AS details
        FROM transactions
        WHERE direction = 'out' AND txn_type = 'wire' AND amount >= 8000
              AND counterparty_country IN ({",".join(f"'{c}'" for c in HIGH_RISK_COUNTRIES)})
        GROUP BY account_id HAVING count(*) >= 1
    """,
}


def build_warehouse(force: bool = False) -> dict[str, int]:
    """Generate ledger + load OFAC + derive alerts into a fresh warehouse.

    Returns row counts per table. Idempotent: skips work if the warehouse exists
    (pass ``force=True`` to regenerate from scratch).
    """
    from aml_investigator import db
    from aml_investigator.data.ofac import download_sdn, load_sdn_into

    if settings.warehouse_path.exists() and not force:
        con = db.connect()
        counts = {
            t: con.execute(f"SELECT count(*) FROM {t}").fetchone()[0]
            for t in ("accounts", "transactions", "sdn", "alerts", "ground_truth")
        }
        con.close()
        logger.info(f"Warehouse already built at {settings.warehouse_path}: {counts}")
        return counts

    download_sdn()
    if settings.warehouse_path.exists():
        settings.warehouse_path.unlink()
    con = db.connect()
    n_sdn = load_sdn_into(con)

    ledger = generate_ledger()
    acc_df = pd.DataFrame(ledger.accounts)
    txn_df = pd.DataFrame(ledger.transactions)
    gt_df = pd.DataFrame(ledger.ground_truth)
    con.register("acc_df", acc_df)
    con.register("txn_df", txn_df)
    con.register("gt_df", gt_df)
    con.execute("INSERT INTO accounts SELECT * FROM acc_df")
    con.execute("INSERT INTO transactions SELECT * FROM txn_df")
    con.execute("INSERT INTO ground_truth SELECT * FROM gt_df")

    # Rule-based alerts (will hit typologies AND some hard negatives — that's the point).
    alert_rows: list[tuple] = []
    for rule, sql in ALERT_RULES_SQL.items():
        for account_id, details in con.execute(sql).fetchall():
            alert_rows.append((f"ALERT-{len(alert_rows) + 1:04d}", account_id, rule, details,
                               datetime.fromisoformat(settings.ledger_end_date + "T23:59:59")))
    con.executemany("INSERT INTO alerts VALUES (?, ?, ?, ?, ?)", alert_rows)

    counts = {
        "accounts": len(acc_df),
        "transactions": len(txn_df),
        "sdn": n_sdn,
        "alerts": len(alert_rows),
        "ground_truth": len(gt_df),
    }
    con.close()
    logger.info(f"Warehouse built at {settings.warehouse_path}: {counts}")
    return counts
