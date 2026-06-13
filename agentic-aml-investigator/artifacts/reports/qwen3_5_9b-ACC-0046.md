## Case Summary
Automated investigation of account ACC-0046 (case qwen3_5_9b-ACC-0046).
Alert(s): sub_threshold_deposits: R1: 12 cash deposits in the $8.5k-$10k band within the period; large_high_risk_wire: R3: 1 outbound wires >= $8k to high-risk countries (PA). The drafted narrative failed validation 2 time(s),
so this report was rendered deterministically from the validated evidence and the
structured risk assessment.

## Evidence
- [EV-01] (structuring_scan) ACC-0046: 12 cash deposits in the $8,500-$10,000 band totalling $112,447 between 2026-04-16 and 2026-04-21 (max 2/day); 0 deposits >= $10,000 ($0).
- [EV-02] (sanctions_check) ACC-0046: screened 0 counterparty names against 19,065 OFAC SDN entries (fuzzy cutoff 87). No matches.
- [EV-03] (counterparty_network) ACC-0046: 3 distinct senders, 136 distinct receivers. Top counterparties: out Christopher Clark (PA) $108,539; in Hobbs, Gonzalez and Shaw (US) $7,871; in Martinez PLC (US) $7,871; in Smith, Alexander and Carter (US) $7,871. CIRCULAR RING(S) DETECTED: [{'ring': ['ACC-0046', 'ACC-0165', 'ACC-0093'], 'total_transferred': 1385.0}, {'ring': ['ACC-0046', 'ACC-0091', 'ACC-0056'], 'total_transferred': 2314.0}]
- [EV-04] (profile_account) ACC-0046 (personal, 'Vanessa Patel', CA, opened 2025-05-05): total in $136,059 / out $127,031, 139 counterparties across 2 countries, active on 67 days. Top flows: in cash_deposit n=12 $112,447; out wire n=1 $108,539; in salary n=3 $23,612; out card n=130 $13,270
- [EV-05] (velocity_scan) ACC-0046: median 2 txns/day (full calendar); busiest 3-day window starts 2026-04-16 with 12 txns / $56,788 — burst ratio 2x baseline. Top days: [('2026-04-16', 6, 19101), ('2026-04-19', 5, 19320), ('2026-03-03', 4, 929)]

## Risk Assessment
Risk score: 72/100. Best-matching typology: structuring.
- Confirmed structuring pattern: 12 cash deposits in $8,500-$10k band totaling $112,447 with zero deposits at or above CTR threshold ($10,000) [[EV-01]]
- Circular transfer rings detected between ACC-0046 and two other accounts (ACC-0165/ACC-0093; ACC-0091/ACC-0056) with combined transfers of $3,728 [[EV-03]]
- Funnel account pattern: 3 distinct senders converge on this personal account before single large outbound wire to high-risk jurisdiction (PA) totaling $108,539 [[EV-04], [EV-03]]
- Velocity burst: 6 transactions on busiest day ($19,101), representing 2x baseline activity during concentrated period April 16-21 [[EV-05]]

## Recommendation
Based on the structured assessment above, the recommended action is ESCALATE.

DISPOSITION: ESCALATE