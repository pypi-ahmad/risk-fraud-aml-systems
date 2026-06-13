## Case Summary
Account ACC-0005, associated with Guzman, Hoffman and Baldwin (US-based business opened on 2019-06-19), exhibited an unusual velocity spike on 2026-03-28, processing 27 transactions totaling $122,518. This activity significantly deviates from the account's typical daily transaction volume of 3, representing a ~9× increase. The spike occurred within a broader 3‑day window (2026-03-26 to 2026-03-28) where total transactions rose to 35, a 4× increase versus the median baseline.

## Evidence
[Evidence ID: EV-01] Velocity scan indicates ACC-0005 had a median of 3 transactions per day over the full calendar year. The busiest 3‑day window began on 2026-03-26 and peaked with 35 transactions totaling $185,449, resulting in a burst ratio of 4× baseline. Specific high‑activity days include:
- 2026-03-28: 27 transactions, $122,518
- 2026-04-28: 26 transactions, $116,161
- 2026-05-28: 26 transactions, $113,998

[Evidence ID: EV-02] Account profile shows typical activity:
- Total inbound transfers: 233, amounting to $1,423,409
- Total outbound transfers: 72, amounting to $314,538
- Total outbound wires: 23, amounting to $248,080
- Active across 277 counterparties in 5 countries over 90 days of activity.

[Evidence ID: EV-03] Sanctions screening against 19,065 OFAC SDN entries (fuzzy cutoff 87) found no matches for any of the 277 counterparties associated with ACC-0005.

## Risk Assessment
Risk Score: 78  
Typology: Velocity Burst  

Factors:
1. **Velocity Spike**: Account ACC-0005 exhibited a velocity spike with 27 transactions totaling $122,518 on 2026-03-28, exceeding the median daily transaction count of 3 by a factor of ~9. [EV-01]  
2. **Burst Context**: The spike occurred within a 3‑day window (2026-03-26 to 2026-03-28) where total transactions rose from the baseline to 35, representing a 4× increase versus the median. [EV-01]  
3. **Profile Deviation**: The account's normal profile shows low daily activity (median 3 txns/day) and typical flows of ~233 inbound transfers vs 72 outbound transfers; the spike far deviates from this baseline. [EV-02]  
4. **Sanctions Clearance**: No sanctioned counterparties were identified in the fuzzy sanctions check against OFAC SDN entries. [EV-03]  

Recommendation: ESCALATE  

## Recommendation
DISPOSITION: ESCALATE