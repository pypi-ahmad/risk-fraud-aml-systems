## Case Summary
The investigation concerns account **ACC-0063** (personal, Corey Herrera, DE, opened 2019-05-28). The account shows unusual activity flagged by a manual referral for periodic KYC review. Key findings include circular transfer rings involving ACC-0063 and elevated transaction velocity on specific dates.

## Evidence
1. **Account Profile ([EV-01])**:  
   - Total inflows: $19,426  
   - Total outflows: $110,813  
   - 139 counterparties across 1 country  
   - Active for 69 days  
   - Top flows: 14 outbound transfers totaling $95,688; 3 inbound salary payments totaling $19,426; 127 card transactions totaling $14,245; 7 cash withdrawals totaling $880  

2. **Counterparty Network ([EV-02])**:  
   - Detected circular rings involving ACC-0063 with three other accounts:  
     - Ring: [ACC-0063, ACC-0193, ACC-0140] – Total transferred: $3,300  
     - Ring: [ACC-0063, ACC-0067, ACC-0102] – Total transferred: $1,596  
     - Ring: [ACC-0063, ACC-0061, ACC-0062] – Total transferred: **$271,240**  

3. **Sanctions Check ([EV-03])**:  
   - Screened 139 counterparty names against 19,065 OFAC SDN entries (fuzzy cutoff 87). No matches found.

4. **Structuring Scan ([EV-04])**:  
   - No cash deposits in the $8,500-$10,000 band totaling $0.  
   - No deposits ≥ $10,000 totaling $0.

5. **Velocity Scan ([EV-05])**:  
   - Median transaction rate: 2 transactions/day (full calendar).  
   - Busiest 3-day window starting 2026-03-24 with 11 transactions totaling $8,049, representing a burst ratio of ~2× baseline.  
   - Top days: 2026-03-24 (4 txns, $458), 2026-03-25 (4 txns, $7,286), 2026-04-21 (4 txns, $549).

## Risk Assessment
- **Risk Score**: 65 (moderate to high risk)  
- **Typology**: Circular transfers  
- **Key Factors**:  
  1. Circular transfer ring detected involving ACC-0063 with three other accounts totaling **$271,240** [EV-02].  
  2. Additional circular rings identified: (ACC-0063, ACC-0193, ACC-0140) $3,300 and (ACC-0063, ACC-0067, ACC-0102) $1,596 [EV-02].  
  3. Velocity scan shows a burst on 2026-03-24 with 11 transactions totaling **$8,049**, exceeding baseline by ~2× [EV-05].  

**Recommendation**: ESCALATE  
**Needs More Evidence**: No  
**Requested Check**: None  

## Recommendation
DISPOSITION: ESCALATE