## Case Summary
Account ACC-0047 for Matthew Arroyo triggered a sub_threshold_deposits alert regarding 15 cash deposits totaling $138,493 between 2026-03-30 and 2026-04-06 [EV-01]. A single large wire outflow of $133,189 was observed following the deposit pattern [EV-02]. Sanctions screening identified one hit on counterparty 'Tran LLC' matching OFAC SDN entry with score 87.5 [EV-03].

## Evidence
*   **Structuring Scan:** 15 cash deposits in the $8,500-$10k band totaling $138,493 between 2026-03-30 and 2026-04-06 [EV-01]. Zero deposits >= $10,000 ($0) were recorded during this period.
*   **Account Profile:** Total inflow of $157,116 / outflow of $155,350 across 2 countries with 155 counterparties [EV-02]. Top flows include cash_deposit n=15 $138,493 and wire out n=1 $133,189.
*   **Sanctions Check:** Screened against 19,065 OFAC SDN entries with fuzzy cutoff 87 [EV-03]. One hit identified: 'Tran LLC' ~ 'STAN LLC'.

## Risk Assessment
The risk assessment assigns a score of 85 based on typology structuring. Factors include the pattern matching classic structuring behavior keeping cash deposits below $10k CTR reporting threshold [EV-01] and one sanctions hit exceeding fuzzy cutoff of 87 [EV-03]. A single large wire outflow ($133,189) suggests potential layering/funnel behavior consistent with structuring typology [EV-02].

## Recommendation
The activity exhibits high-risk characteristics including deliberate threshold avoidance and a sanctions hit. Further investigation is required to determine intent regarding the sanctioned entity and source of funds for cash deposits. Proceed with enhanced due diligence procedures immediately. DISPOSITION: ESCALATE