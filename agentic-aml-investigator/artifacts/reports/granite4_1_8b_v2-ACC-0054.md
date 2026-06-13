## Case Summary
Account **ACC-0054** (Michael Brown, Germany) exhibited unusual activity on May 11‑13, 2026, with a significant velocity spike and outbound wire transfers to Iran.

## Evidence
- **[EV-01]**: Account overview shows $100,451 in deposits and $83,445 outflows across 30 counterparties in six countries. Notable flows include 14 inbound transfers totaling $87,452 and 14 outbound wires totaling $83,445.
- **[EV-02]**: Velocity scan reveals a median of zero transactions per day, but a three-day window (May 11‑13) recorded 28 transactions amounting to $170,897, representing a ~19× increase over the baseline.
- **[EV-03]**: Transaction network analysis identifies 16 distinct senders and 14 receivers. Two high-value outbound wires of $8,870 and $8,481 were sent to Iran (John White) on May 12.
- **[EV-04]**: Sanctions screening against OFAC SDN lists found no matches among the 30 counterparties, despite transfers to a high-risk jurisdiction.

## Risk Assessment
Risk score: **75** (moderate-high).  
Typology: **velocity_burst**.  

Factors:
1. The account experienced a velocity spike on May 11‑13 2026 with 28 transactions totaling $170,897, a ~19× increase over its baseline median of zero daily activity. [EV-02]  
2. Two outbound wires exceeding $8k were sent to Iran (high-risk jurisdiction) on May 12 2026. [EV-03]  
3. No counterparties matched OFAC SDN entries, but the high-value transfers to a sanctioned country remain concerning. [EV-04]  

Recommendation: **ESCALATE**.

## Recommendation
DISPOSITION: ESCALATE