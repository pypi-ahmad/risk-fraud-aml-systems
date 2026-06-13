## Case Summary
The Suspicious Activity Report (SAR) investigates account **ACC-0029** associated with the business entity "Ross, Robinson and Bright," located in the United States and opened on November 3, 2022. The account has exhibited a total inflow of $1,611,644 and outflow of $581,099 across 291 counterparties spanning five countries over an active period of 90 days. Notably, the account's velocity scan identified a modest burst ratio of 2× baseline during specific three-day windows in March 2026.

## Evidence
- **[EV-01]**: Account profile indicates total inflows of $1,611,644 and outflows of $581,099 with activity across 291 counterparties in five countries. The account has been active for 90 days.
- **[EV-02]**: Counterparty network analysis reveals 252 distinct senders and 40 distinct receivers. Top outbound transactions include amounts to Levy, Melendez and Miller (US) ($24,614), Davis Group (GB) ($24,080), Taylor and Sons (US) ($23,225), and Peters, Hughes and Lawson (US) ($22,804). No circular transaction rings were detected.
- **[EV-03]**: Sanctions screening against 19,065 OFAC SDN entries (fuzzy cutoff 87) found no matches for any of the 291 counterparties.
- **[EV-04]**: Structuring scan identified zero cash deposits within the $8,500-$10,000 band and no deposits totaling over $10,000.
- **[EV-05]**: Velocity scan detected a median transaction rate of 3 transactions per day, with a peak burst ratio of 2× baseline during a three-day window starting April 7, 2026. Top days include March 4, 13, and 16, 2026, each with five transactions totaling $105,120.

## Risk Assessment
The risk assessment assigns a **risk score of 20**, categorized under "none." Key factors contributing to this assessment are:
1. No structuring behavior detected; no cash deposits in the $8,500-$10,000 band ([EV-04]).
2. Sanctions screening returned zero matches against the OFAC SDN list ([EV-03]).
3. Velocity scan showed a modest 2× baseline burst on specific days but remained within normal ranges for this account type ([EV-05]).

The recommendation is to **DISMISS** the case, indicating no further action is required based on the current evidence.

## Recommendation
DISPOSITION: DISMISS