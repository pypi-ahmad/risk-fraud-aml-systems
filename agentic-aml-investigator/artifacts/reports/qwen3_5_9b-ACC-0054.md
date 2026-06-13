## Case Summary
Account ACC-0054 held by 'Michael Brown', opened 2019-07-17 [EV-02]. Alerts include velocity_spike R2 and large_high_risk_wire R3 regarding outbound wires >= $8k to high-risk countries. Activity shows median 0 txns/day with burst ratio of 19x baseline starting 2026-05-11 [EV-01].

## Evidence
Velocity scan: Median 0 txns/day (full calendar) [EV-01]. Busiest window starts 2026-05-11 with 28 txns / $170,897 [EV-01]. Top day ('2026-05-11', 12, 81662) [EV-01]. Profile: Total in $100,451 and out $83,445 across 6 countries [EV-02]. Wire n=14 totals $83,445; transfer n=14 totals $87,452 [EV-02]. Network: 16 distinct senders, 14 distinct receivers. Out John White (IR) is $8,870 [EV-03]. Sanctions check screened 30 counterparty names against 19,065 OFAC SDN entries with no matches [EV-04].

## Risk Assessment
Current risk score is 72. Typology: 'sanctioned_counterparty'. Factors include dormant account velocity burst activity (19x baseline) [EV-01][EV-02], large outbound wire ($8,870) to Iranian counterparty John White [EV-03], and high-risk jurisdiction exposure (IR/Iran) [EV-01][EV-02][EV-03]. Recommendation is ESCALATE. Needs more evidence: True; requested check 'sanctions_check'.

## Recommendation
Proceed with sanctions verification for counterparties in Iran. Maintain monitoring on velocity patterns until baseline stabilizes or activity ceases. DISPOSITION: ESCALATE