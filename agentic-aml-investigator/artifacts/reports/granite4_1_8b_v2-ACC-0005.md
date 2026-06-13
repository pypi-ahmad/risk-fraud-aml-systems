## Case Summary
Account ACC-0005, belonging to Guzman, Hoffman and Baldwin (US-based business established on 2019-06-19), has exhibited a significant velocity spike in transaction activity over a three-day window starting March 26, 2026. The account processed 27 transactions totaling $122,518, which is approximately 9 times the median daily transaction count of 3 transactions/day.

## Evidence
[EV-01] (profile_account) ACC-0005: Total inflows $1,423,409 and outflows $562,618 across 277 counterparties in 5 countries, with activity on 90 days since opening.
[EV-02] (velocity_scan) ACC-0005: Median daily transaction count is 3; the busiest three-day window from March 26 to March 28, 2026, recorded 27 transactions totaling $122,518, representing a burst ratio of 9× baseline. Additional high-activity days on April 28 and May 28, 2026, logged 26 transactions each, amounting to $116,161 and $113,998 respectively.
[EV-03] (sanctions_check) ACC-0005: Screening against 19,065 OFAC SDN entries yielded no matches among the 277 counterparties.

## Risk Assessment
Risk Score: 65  
Typology: Velocity Burst  

Factors:
1. **Claim**: Account ACC-0005 exhibited a pronounced velocity spike far exceeding its normal transaction baseline.  
   Evidence ID: EV-02  
2. **Claim**: Median daily transaction count for ACC-0005 is 3 transactions/day; the burst window (2026‑03‑26 to 2026‑03‑28) recorded 27 transactions totaling $122,518, representing a ~9× increase over the median.  
   Evidence ID: EV-02  
3. **Claim**: Additional high-activity days on 2026‑04‑28 and 2026‑05‑28 also showed elevated transaction volumes (26 transactions each, $116,161 and $113,998 respectively).  
   Evidence ID: EV-02  

Recommendation: ESCALATE  
No additional evidence is required.

## Recommendation
DISPOSITION: ESCALATE