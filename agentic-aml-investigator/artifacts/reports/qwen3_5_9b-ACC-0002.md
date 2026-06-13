## Case Summary
Account ACC-0002 (Doyle Ltd) opened 2023-05-04 [EV-01]. Alert triggered for sub_threshold_deposits R1 involving 31 cash deposits in the $8,500-$10,000 band within the period [EV-04]. Total inflows reached $2,928,717 and outflows were $452,813 across 264 counterparties in five countries [EV-01].

## Evidence
*   **Account Activity:** Inflow total of $2,928,717; Outflow total of $452,813. Top flows include transfer n=233 ($1,591,999) and cash_deposit n=174 [EV-01].
*   **Transaction Query:** 31 rows returned for transactions where account_id='ACC-0002' and txn_type='cash_deposit' with amounts between thresholds defined in the alert logic [EV-02].
*   **Sanctions Check:** Screened against 19,065 OFAC SDN entries (fuzzy cutoff 87). No matches found among 264 counterparties [EV-03].
*   **Structuring Scan:** 31 cash deposits in the $8,500-$10,000 band totalling $287,143 between 2026-03-03 and 2026-05-31. Max frequency was 2/day [EV-04].
*   **Network Analysis:** Top counterparties include Whitney ($24,417) and Reed Group ($23,654). No circular rings found [EV-05].

## Risk Assessment
Typology is structuring with a risk score of 78. Factors indicate deliberate clustering below reporting limits within the $8,500-$9,800 range just under the CTR threshold [EV-02], totaling $287,143 over ~65 days at maximum frequency of 2/day [EV-04]. No circular transfer rings or sanctioned counterparties identified in network analysis [EV-05], [EV-03].

## Recommendation
The risk assessment recommends escalation. Further review is required to determine intent behind the deposit clustering pattern observed between March and May 2026. DISPOSITION: ESCALATE