## Case Summary
Account **ACC-0081** (personal, Dean Santiago, GB) was flagged for periodic KYC review due to suspicious activity. The account, opened on 2022-08-21, has exhibited concerning patterns in its transactional behavior and counterparty relationships.

## Evidence
- **[EV-01]**: Account summary shows $13,184 received and $145,089 sent across 142 counterparties in four countries over 68 active days. Notable flows include six wires totaling $125,902, 132 card transactions amounting to $15,955, three salary inflows of $13,184, and two transfers totaling $2,032.
- **[EV-02]**: Counterparty network analysis reveals three distinct senders and 139 distinct receivers. Top counterparties include:
  - Outbound to Haqqaui, Nasiruddin (TR) for $30,315
  - Outbound to Islamic Armu Of Aden (TR) for $29,981
  - Outbound to Haqqaui, Nasiruddin (HK) twice, totaling $26,483 and $15,597 respectively. No circular rings were detected.
- **[EV-03]**: Sanctions screening against 19,065 OFAC SDN entries identified two high-fuzzy matches:
  - 'Islamic Armu Of Aden' matched with 'ISLAMIC ARMY OF ADEN' (fuzzy score 95.0)
  - 'Haqqaui, Nasiruddin' matched with 'HAQQANI, Nasiruddin' (fuzzy score 94.4)
- **[EV-04]**: Structuring scan found no deposits within the $8,500-$10,000 range or any deposits exceeding $10,000.
- **[EV-05]**: Velocity scan detected a median transaction rate of 2 transactions per day, with a notable burst on 2026-04-17 involving 11 transactions totaling $16,700, representing a 2x increase over baseline. Other busy days include 2026-05-07 (5 transactions, $26,902) and 2026-03-21 (4 transactions, $30,675).

## Risk Assessment
The risk assessment assigns a score of **80**, categorized as **sanctioned_counterparty**. Key factors contributing to this assessment are:
1. **Two counterparties match OFAC SDN entries with high fuzzy scores** ([EV-03]).
2. **Velocity scan indicates a burst of activity on specific dates**, suggesting potential structuring or unusual behavior ([EV-05]).

Recommendation: **ESCALATE**. No additional evidence is required at this time.

## Recommendation
DISPOSITION: ESCALATE