## Case Summary
Account ACC-0005 triggered a velocity_spike alert (R2). Daily activity spike recorded 27 txns, $122518 versus median 3.0/day [EV-01]. Business entity 'Guzman, Hoffman and Baldwin', US opened 2019-06-19 [EV-02].

## Evidence
Velocity scan shows median 3 txns/day (full calendar). Busiest window starts 2026-03-26 with 35 txns / $185,449 — burst ratio 4x baseline. Top days: [('2026-03-28', 27, 122518), ('2026-04-28', 26, 116161), ('2026-05-28', 26, 113998)] [EV-01]. Profile indicates total in $0 / out $0, 277 counterparties across 5 countries, active on 90 days [EV-02]. Sanctions check screened 277 counterparty names against 19,065 OFAC SDN entries (fuzzy cutoff 87). No matches found [EV-03].

## Risk Assessment
Current risk score is 38. Typology classified as 'velocity_burst'. Factors include velocity spike detected on specific dates (26 Mar - May) [EV-01], recurring activity pattern aligns with business cycle/payroll logic [EV-01], [EV-02], and no sanctions exposure found [EV-03]. Recommendation is DISMISS. Needs more evidence: False.

## Recommendation
The velocity burst typology presents a low risk profile given the lack of sanctions matches and alignment with established business cycles observed in data points. Final decision: DISPOSITION: DISMISS