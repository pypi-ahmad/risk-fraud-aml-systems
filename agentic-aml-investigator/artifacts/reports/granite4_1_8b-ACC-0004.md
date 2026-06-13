## Case Summary
Case granite4_1_8b-ACC-0004 involves account ACC-0004 associated with "Davis and Sons," a U.S.-based business opened in July 2018. An alert was triggered due to a significant velocity spike on May 28, 2025, where the account processed 20 transactions totaling $118,336, markedly exceeding its median daily volume of 3 transactions.

## Evidence
- **[EV-01]**: Velocity scan indicates that ACC-0004 had a busiest 3-day window starting March 26, 2026, with 26 transactions amounting to $154,225. The burst ratio is approximately 3 times the baseline. Top transaction days include May 28, 2026, with 20 transactions totaling $118,336.
- **[EV-02]**: KYC profile shows ACC-0004 as a legitimate U.S. business with 266 counterparties across four countries. Total inflows amount to $1,506,673, and outflows total $564,975 over active days spanning 90 days.
- **[EV-03]**: Sanctions screening of the 266 counterparties against OFAC SDN entries resulted in one hit: 'Morgan LLC' closely matches 'MOKRAN LLC' (DPRK3) with a fuzzy match score of 90.0.

## Risk Assessment
The risk assessment assigns a risk score of 65, categorized as a **velocity_burst** typology. Key factors include:
1. A velocity spike on May 28, 2025, with 20 transactions totaling $118,336, exceeding the median daily volume by approximately 6.7× ([EV-01]).
2. The account's KYC profile is clean, representing a legitimate U.S. business with normal transaction patterns across multiple countries ([EV-02]).
3. One counterparty ('Morgan LLC') matches a sanctioned entity (DPRK3) with a high fuzzy match score of 90.0 ([EV-03]).
4. The velocity burst ratio is strong but requires validation against legitimate business events to rule out potential misuse.

## Recommendation
DISPOSITION: ESCALATE

The recommendation to escalate is based on the detection of a significant velocity spike and the identification of a sanctioned counterparty, necessitating further investigation to ensure compliance with regulatory requirements.