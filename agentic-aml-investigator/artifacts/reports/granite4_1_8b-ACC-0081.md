## Case Summary
The account ACC-0081, held by Dean Santiago in the UK and opened on 2022-08-21, has been flagged for a manual referral during a periodic KYC review due to potential sanctioned counterparty involvement.

## Evidence
- **[EV-01]**: Account activity summary:
  - Total inflows: $13,184
  - Total outflows: $145,089
  - Transactions with 142 counterparties across 4 countries
  - Active for 68 days
  - Top transaction types:
    - Outbound wire transfers: 6 transactions totaling $125,902
    - Outbound card payments: 132 transactions totaling $15,955
    - Inbound salary deposits: 3 transactions totaling $13,184
    - Outbound bank transfers: 2 transactions totaling $2,032

- **[EV-02]**: Sanctions screening results:
  - Screened 138 counterparties against 19,065 OFAC SDN entries using fuzzy matching with a cutoff score of 87.
  - Identified 2 high-confidence matches to sanctioned entities:
    - 'Islamic Army Of Aden' matched 'Islamic Armu Of Aden' (fuzzy score 95.0, SDGT)
    - 'Haqqaui, Nasiruddin' matched 'HAQQANI, Nasiruddin' (fuzzy score 94.4, SDGT)

## Risk Assessment
- **Risk Score**: 80 (High risk)
- **Typology**: Sanctioned Counterparty
- **Key Factor**:
  - Two counterparties in the transaction records match OFAC Specially Designated Nationals (SDN) entries with high fuzzy scores (95.0 and 94.4), indicating involvement with sanctioned entities.
- **Recommendation**: ESCALATE

## Recommendation
DISPOSITION: ESCALATE