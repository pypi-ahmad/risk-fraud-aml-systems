## Case Summary
Account **ACC-0053** (Julian Chapman, personal account opened 2019-07-13 in California) exhibited a significant transaction velocity spike on May 18–20 2026, with 36 transactions totaling $185,899 across three days. This represents a 24× increase over the account's median daily activity of 0 transactions per day. The alert **velocity_spike: R2** was triggered due to this abnormal activity pattern.

## Evidence
- **[EV-01]**: Velocity scan revealed that ACC-0053 had no typical daily activity, but experienced a burst with 36 transactions totaling $185,899 during the three-day window starting May 18 2026. The busiest days were:
  - May 18: 12 transactions totaling $56,469
  - May 19: 12 transactions totaling $59,937
  - May 20: 12 transactions totaling $69,494
  
- **[EV-02]**: Account profile shows a total inflow of $124,622 and outflow of $98,522 over its lifetime. The account interacts with 40 counterparties across six countries, being active on seven days. Top flows include:
  - Inbound transfer: 19 transactions totaling $102,662
  - Outbound wire: 19 transactions totaling $98,522
  - Inbound salary: 3 transactions totaling $21,960

- **[EV-03]**: Counterparty network analysis identified 21 distinct senders and 19 distinct receivers. Notable counterparties:
  - Inbound from Gabrielle Snyder (US): $9,637
  - Inbound from Smith-Bowen (US): $8,539
  - Outbound to Gloria Frank (SY): $7,952
  - Inbound from Renee Hogan (US): $7,882
  
- **[EV-04]**: Sanctions screening against 19,065 OFAC SDN entries (fuzzy cutoff 87) found no matches for any of the 40 counterparties involved in ACC-0053's transactions.

## Risk Assessment
The risk assessment assigns a score of **65** (moderate-high risk) with typology **velocity_burst**. Key factors include:
1. The account exhibited a dramatic activity burst on May 18–20 2026, 24× higher than its median daily activity ([EV-01]).
2. The velocity spike aligns with the alert description (**velocity_spike: R2**) indicating abnormal transaction volume for an otherwise dormant account ([EV-01]).
3. No circular transfer rings were detected in the counterparty network, reducing suspicion of structured laundering loops ([EV-03]).
4. Sanctions screening found no matches against OFAC SDN entries for any counterparties involved ([EV-04]).

## Recommendation
DISPOSITION: ESCALATE

The account's sudden velocity spike warrants further investigation to determine whether this activity is legitimate or indicative of potential illicit behavior, despite the absence of sanctions matches and circular transfer patterns.