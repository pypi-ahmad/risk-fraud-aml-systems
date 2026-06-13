## Case Summary
Account ACC-0004 ('Davis and Sons') opened 2018-07-01 [EV-01] triggered velocity_spike alert R2 (daily activity spike, 20 txns, $118336) vs median 3.0/day [EV-02]. Sanctions check identified hit: 'Morgan LLC' ~ 'MOKRAN LLC', score 90.0 linked to DPRK3 [EV-03].

## Evidence
*   **[EV-01]** Profile shows total in $1,506,673 / out $564,975 across 266 counterparties and 4 countries over 90 days. Top flows include transfer n=222 [EV-01].
*   **[EV-02]** Velocity scan indicates median 3 txns/day. Busiest window starts 2026-03-26 with 26 txns / $154,225. Top days include ('2026-05-28', 20, 118336) [EV-02].
*   **[EV-03]** Screened 266 counterparty names against 19,065 OFAC SDN entries (fuzzy cutoff 87). Hit(S): 'Morgan LLC' ~ 'MOKRAN LLC' (score 90.0) [EV-03].
*   **[EV-04]** Network shows 222 distinct senders, 45 distinct receivers. Top counterparties include Perez, Hayes and Lloyd (CA) $24,942 [EV-04] and Torres, Brown and Williams (FR) $21,790 [EV-04].

## Risk Assessment
Typology is 'sanctioned_counterparty'. Factors: Sanctions hit detected with high fuzzy match score (90.0) to DPRK3 entity 'MOKRAN LLC' [EV-03]; Velocity spike consistent with established business baseline; other high-volume days exist in March/April [EV-02]; Account profile supports legitimate US business operations (Davis and Sons, opened 2018) [EV-01]. Recommendation is ESCALATE. Needs more evidence requested: sanctions_check.

## Recommendation
Due to the sanctioned counterparty hit with a high fuzzy match score of 90.0 against DPRK3 entity 'MOKRAN LLC' [EV-03], further investigation into transaction nature and counterparties is required despite baseline velocity consistency [EV-02]. Account profile supports legitimate operations, but sanctions risk necessitates review before clearance.

DISPOSITION: ESCALATE