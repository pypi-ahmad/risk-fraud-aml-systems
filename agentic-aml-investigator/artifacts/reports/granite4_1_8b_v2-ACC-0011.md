## Case Summary
Account **ACC-0011** belongs to **Blair PLC**, a business entity based in Germany, opened on 2023-08-13. The account has processed $1,443,694 in inflows and $498,511 outflows over 90 active days, involving 255 counterparties across seven countries. A manual referral for periodic KYC review was triggered due to potential sanction exposure.

## Evidence
- **[EV-01]**: Account profile indicates total inflows of $1,443,694 from 220 transfers and outflows of $498,511 via 36 wires.
- **[EV-02]**: Network analysis shows 219 distinct senders and 36 distinct receivers. Top US-based counterparties include Jennings, Rhodes and Patel ($24,682), Whitney, Mayo and Clark ($24,197), Reed Ltd ($23,458), and Morris and Sons ($22,913). No circular transaction rings were detected.
- **[EV-03]**: Sanctions screening against 19,065 OFAC SDN entries (fuzzy cutoff 87) revealed three matches:
  - 'Valfajr Shipping eompany Pjs' ~ 'VALFAJR SHIPPING COMPANY PJS' (Iran, score 96.4)
  - 'Kamyshev, Denss Valentinovich' ~ 'KAMYSHEV, Denis Valentinovich' (Russia-EO14024, score 96.4)
  - 'Ramos Group' ~ 'RAMOR GROUP' (NPWMD, score 90.9)
- **[EV-04]**: Structuring scan found no deposits in the $8,500-$10,000 band and zero deposits ≥ $10,000.
- **[EV-05]**: Velocity scan shows a median of 3 transactions per day, with a peak burst on 2026-03-28 (12 transactions totaling $63,721), maintaining a burst ratio of 1x baseline.

## Risk Assessment
The risk assessment assigns a score of **80**, categorized as **sanctioned_counterparty**. The primary factor is the identification of three counterparties matching OFAC SDN entries with high fuzzy scores (96-97), indicating sanctioned exposure ([EV-03]).

## Recommendation
DISPOSITION: ESCALATE

The presence of sanctioned counterparties necessitates immediate escalation to senior compliance and legal teams for further investigation and potential mitigation actions.