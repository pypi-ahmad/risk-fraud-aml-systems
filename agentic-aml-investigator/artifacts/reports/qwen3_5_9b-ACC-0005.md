## Case Summary
Account ACC-0005 triggered a velocity_spike R2 alert [EV-01]. Median daily activity is 3 txns/day versus spike of 27 txns totaling $122518 on specific dates including 2026-03-28 [EV-01].

## Evidence
**Velocity Scan:** Busiest window starts 2026-03-26 with 35 txns / $185,449 (burst ratio 4x baseline) [EV-01]. Top days include ('2026-03-28', 27, 122518), ('2026-04-28', 26, 116161), and ('2026-05-28', 26, 113998) [EV-01].
**Account Profile:** Business 'Guzman, Hoffman and Baldwin' (US). Opened 2019-06-19. Total in $1,423,409 / out $562,618 across 277 counterparties [EV-02].
**Sanctions Check:** Screened 277 counterparty names against 19,065 OFAC SDN entries (fuzzy cutoff 87). No matches found [EV-03].

## Risk Assessment
Current analysis yields risk_score 45 with typology 'velocity_burst' [EV-01]. Factors include confirmed velocity spike of 4x baseline on specific dates in March-May 2026 (EV-01) and business account profile available but system error prevented KYC retrieval (EV-02). No OFAC SDN matches were identified for the screened counterparties [EV-03]. Recommendation is to escalate due to needs_more_evidence status.

## Recommendation
Further investigation required regarding velocity burst typology and account profile verification. Additional checks requested per system error noted in evidence retrieval. DISPOSITION: ESCALATE