## Case Summary
Case granite4_1_8b-ACC-0011 involves account ACC-0011 belonging to Blair PLC, a business entity based in Germany, opened on 2023-08-13. The account has exhibited significant transaction activity with total inflows of $1,443,694 and outflows of $498,511 over 90 days, involving 220 inbound transfers and 36 outbound wires across 255 counterparties in seven countries.

## Evidence
[Evidence ID EV-01]: Account profile indicates substantial transaction volume and a broad network of counterparties.
[Evidence ID EV-02]: Sanctions screening identified three counterparties matching OFAC SDN entries with high fuzzy scores (96.4–90.9), suggesting potential sanctions exposure related to Iran, Russia, and the NPWMD.

## Risk Assessment
The risk assessment assigns a risk score of 75, categorized under "sanctioned_counterparty." Key factors include:
- Three counterparties matching OFAC SDN entries with high fuzzy scores (96–97) [EV-02].
- A deterministic guardrail mandates a minimum risk score of 75 due to the OFAC fuzzy match threshold [EV-02].

## Recommendation
DISPOSITION: ESCALATE

The presence of sanctioned counterparties necessitates immediate escalation to compliance and legal teams for further investigation and potential mitigation actions.