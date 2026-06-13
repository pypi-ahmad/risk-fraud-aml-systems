## Case Summary
Account **ACC-0004** associated with "Davis and Sons" (US) exhibited an unusual transaction velocity spike on **2026-05-28**, processing **20 transactions totaling $118,336**, which is approximately **6‑7 times higher** than its median daily volume of **3 transactions**. The account's activity profile indicates a diversified network of 266 counterparties across four countries, with no identified circular transfer patterns. A low-confidence sanctions match was detected for one counterparty.

## Evidence
- **[EV-01]**: Velocity scan reveals the busiest 3‑day window starting **2026-03-26** with **26 transactions totaling $154,225**, representing a burst ratio of **3× baseline**. Top transaction days include **2026-05-28** (20 txns, $118,336).
- **[EV-02]**: Account profile shows total inflows of **$1,506,673** and outflows of **$564,975**, involving **266 counterparties** across the US, Canada, France, and others, with activity on **90 days**.
- **[EV-03]**: Counterparty network analysis identifies **222 distinct senders** and **45 distinct receivers**, with notable outbound flows to entities in California and France. No circular transfer rings were detected.
- **[EV-04]**: Sanctions screening of counterparties against 19,065 OFAC SDN entries resulted in **1 low-confidence fuzzy hit** on "Morgan LLC" (~90 similarity) linked to a DPRK entity.

## Risk Assessment
- **Risk Score**: 65 (moderate-high risk)
- **Typology**: Velocity burst
- **Factors**:
  1. The account experienced a velocity spike on **2026-05-28** with **20 transactions totaling $118,336**, exceeding the median daily volume of ~3 transactions by roughly 6‑7 times. [EV-01]
  2. No sanctions matches were found among counterparties except a low-confidence fuzzy hit on 'Morgan LLC' (~90 similarity). [EV-04]
  3. The counterparty network shows no circular transfer rings; activity is broadly distributed across California and France. [EV-03]

## Recommendation
DISPOSITION: ESCALATE

The velocity spike, combined with the low-confidence sanctions match, warrants further investigation to ensure compliance with regulatory requirements and to assess potential illicit financial activities.