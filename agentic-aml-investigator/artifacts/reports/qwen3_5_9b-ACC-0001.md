## Case Summary
Automated investigation of account ACC-0001 (case qwen3_5_9b-ACC-0001).
Alert(s): sub_threshold_deposits: R1: 22 cash deposits in the $8.5k-$10k band within the period. The drafted narrative failed validation 2 time(s),
so this report was rendered deterministically from the validated evidence and the
structured risk assessment.

## Evidence
- [EV-01] (profile_account) ACC-0001 (business, 'Rodriguez, Figueroa and Sanchez', US, opened 2025-10-21): total in $2,751,933 / out $431,565, 248 counterparties across 5 countries, active on 90 days. Top flows: in transfer n=217 $1,395,964; in cash_deposit n=182 $1,355,969; out wire n=33 $431,565
- [EV-02] (structuring_scan) ACC-0001: 22 cash deposits in the $8,500-$10,000 band totalling $203,126 between 2026-03-11 and 2026-05-29 (max 2/day); 51 deposits >= $10,000 ($609,654).
- [EV-03] (sanctions_check) ACC-0001: screened 248 counterparty names against 19,065 OFAC SDN entries (fuzzy cutoff 87). No matches.

## Risk Assessment
Risk score: 85/100. Best-matching typology: structuring.
- 22 cash deposits deliberately kept in $8,500-$10k sub-threshold band totaling $203,126 between March-May 2026 with max frequency of 2/day [[EV-02]]
- Pattern spans ~3 months (March-May 2026) showing regular clustering around sub-threshold band, indicating intentional structuring to avoid CTR reporting requirements while conducting substantial cash transactions [[EV-02]]
- Account also has 51 deposits ≥$10k totaling $609,654 which would normally trigger CTRs; combination of sub-threshold and above-threshold activity suggests intentional structuring [[EV-02]]
- No sanctions matches found against 19,065 OFAC SDN entries with fuzzy cutoff 87 across 248 counterparties [[EV-03]]

## Recommendation
Based on the structured assessment above, the recommended action is ESCALATE.

DISPOSITION: ESCALATE