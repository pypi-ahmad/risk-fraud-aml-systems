## Case Summary
The investigation concerns account **ACC-0036**, a business account belonging to "Rodriguez-Graham" located in California, opened on 2018-03-20. The account has exhibited typical transaction activity with $1,527,402 inflows and $593,839 outflows over 281 counterparties across five countries during its active period of 90 days. Notably, the account experienced a velocity burst on March 25, 27, and 29, 2026, with transaction counts and values roughly double its baseline median of three transactions per day.

## Evidence
- **[EV-01]**: Account profile indicates total inflows of $1,527,402 and outflows of $593,839 across 281 counterparties in five countries. The account has been active for 90 days.
- **[EV-02]**: Network analysis shows 240 distinct senders and 41 distinct receivers. Top counterparties include US entities (Soto, Wang and Blair; Evans, Hoffman and Hampton) and UK/Canadian entities (Smith-Clark, Smith-Jones), with no circular rings detected.
- **[EV-03]**: Sanctions screening against 19,065 OFAC SDN entries found no matches for the 281 counterparties.
- **[EV-04]**: Structuring scan revealed no deposits in the $8,500-$10,000 band or any deposits totaling $0 across all thresholds.
- **[EV-05]**: Velocity scan identified a burst on specific days (2026-03-25, 27, 29) with transaction counts and values roughly double the baseline median of three transactions per day.

## Risk Assessment
The risk assessment assigns a score of **40**, categorized as a **velocity_burst** typology. The primary factor is the observed velocity burst on specific days (2026-03-25, 2026-03-27, 2026-03-29) with transaction counts and values roughly double the baseline median of three transactions per day, as evidenced by [EV-05]. No matches were found in sanctions screening ([EV-03]), and no structuring patterns were detected ([EV-04]). The account's activity does not align with typical suspicious structuring or illicit financing behaviors.

## Recommendation
DISPOSITION: DISMISS

The velocity burst observed is within the account's normal operational pattern, supported by the lack of any matching sanctions entries or structuring anomalies. No additional evidence is required to dismiss this alert.