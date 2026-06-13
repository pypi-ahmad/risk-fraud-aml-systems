## Case Summary
Account ACC-0004 ('Davis and Sons', US) opened 2018-07-01 [EV-01]. Alert triggered by velocity_spike R2 showing daily activity spike of 20 txns $118336 vs median 3.0/day [EV-02].

## Evidence
Account profile shows total in $1,506,673 / out $564,975 across 266 counterparties and 4 countries active on 90 days [EV-01]. Velocity scan indicates median 3 txns/day with busiest window starting 2026-03-26 (26 txns / $154,225) burst ratio 3x baseline. Top days include ('2026-05-28', 20, 118336), ('2026-03-28', 19, 104120), ('2026-04-28', 18, 95189) [EV-02]. Sanctions check screened 266 counterparty names against 19,065 OFAC SDN entries (fuzzy cutoff 87). One HIT(S): 'Morgan LLC' ~ 'MOKRAN LLC' (score 90.0, DPRK3) [EV-03]. Network analysis shows 222 distinct senders and 45 distinct receivers with no circular rings found [EV-04].

## Risk Assessment
Risk score: 85. Typology: 'sanctioned_counterparty'. Factors include OFAC SDN entry match with DPRK3 designation despite fuzzy scoring (90.0) [EV-03] and velocity spike pattern recurs on same calendar day each month (28th), consistent with payroll run rather than laundering [EV-02]. Recommendation: ESCALATE.

## Recommendation
Further investigation required to verify counterparty identity against DPRK3 designation based on OFAC SDN entry match. DISPOSITION: ESCALATE