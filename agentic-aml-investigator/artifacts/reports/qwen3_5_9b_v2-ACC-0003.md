## Case Summary
Account ACC-0003 triggered a sub_threshold_deposits alert regarding 25 cash deposits in the $8,500-$10k band totaling $230,077 between 2026-03-04 and 2026-05-30 [EV-01]. The account belongs to business entity 'Mcclain, Miller and Henderson' opened on 2024-07-09 [EV-02].

## Evidence
- Structuring scan identified 25 cash deposits in the $8,500-$10k band totaling $230,077 between 2026-03-04 and 2026-05-30 (max 2/day); 59 deposits >= $10,000 ($721,138) [EV-01].
- Profile shows business entity 'Mcclain, Miller and Henderson', US. Total in $2,764,249 / out $523,311, 253 counterparties across 5 countries, active on 90 days [EV-02]. Top flows: cash_deposit n=180 ($1,422,161); transfer n=218 ($1,342,088) [EV-02].
- Sanctions check screened 253 counterparty names against 19,065 OFAC SDN entries (fuzzy cutoff 87). No matches found for any counterparties in the account's transaction history [EV-03].

## Risk Assessment
Risk score is 25 with typology none. Factors indicate ordinary cash-business flow rather than deliberate structuring to evade CTR reporting requirements [EV-01]. The presence of many supra-threshold deposits suggests ordinary cash-business flow rather than deliberate structuring to evade CTR reporting requirements [EV-01]. No OFAC sanctions matches found for any counterparties in the account's transaction history [EV-03]. Account is a business entity with substantial total inflows/outflows consistent with commercial activity [EV-02].

## Recommendation
Analysis indicates no suspicious typology identified. DISPOSITION: DISMISS