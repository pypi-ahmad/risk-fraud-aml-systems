## Case Summary
Case granite4_1_8b-ACC-0046 involves account ACC-0046 held by Vanessa Patel (personal) in California, opened on 2025-05-05. The account shows suspicious activity including structuring behavior and transactions to high-risk jurisdictions.

## Evidence
- **[EV-01]**: Account summary indicates total inflows of $136,059 and outflows of $127,031 across 139 counterparties in two countries over 67 active days. Notable flows include 12 cash deposits totaling $112,447 and one outbound wire of $108,539.
- **[EV-02]**: Structuring scan reveals 12 cash deposits within the $8,500-$10,000 range, summing to $112,447 between April 16 and April 21, 2026, with a maximum of two deposits per day and no single deposit exceeding $10,000.
- **[EV-03]**: Counterparty network analysis detects circular transfer rings involving ACC-0046:
  - Ring 1: ACC-0046 ↔ ACC-0165 ↔ ACC-0093, total transferred $1,385.0
  - Ring 2: ACC-0046 ↔ ACC-0091 ↔ ACC-0056, total transferred $2,314.0
  Notable outbound wire of $108,539 to Christopher Clark in Panama (high-risk country).
- **[EV-04]**: Sanctions screening against 19,065 OFAC SDN entries found no matches for any counterparties.

## Risk Assessment
Risk score: 78 (High). Typology: Structuring. Key factors:
1. **Structuring Behavior**: 12 cash deposits in the $8,500–$10,000 band totaling $112,447 within a short period [EV-02].
2. **High-Risk Transaction**: Outbound wire of $108,539 to Christopher Clark in Panama, a high-risk jurisdiction [EV-03].
3. **Circular Transfer Rings**: Two detected rings involving ACC-0046 suggest potential layering/funneling activity [EV-03].

Recommendation: ESCALATE.

## Recommendation
DISPOSITION: ESCALATE