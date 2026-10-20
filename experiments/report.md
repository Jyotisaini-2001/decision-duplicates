# Jaccard filter experiment

Pair-level comparison. Union-Find clustering is not used.
Production files and `data/decisions_vec.db` were not modified.

- Model: `BAAI/bge-small-en-v1.5` (384 dimensions)
- Cosine: L2-normalized vectors, similarity = dot product, threshold `0.75`
- Jaccard: `find_duplicates.tokens` + `find_duplicates.jaccard_sim`, cutoff `0.32`
- Primary labeled pairs: **116** (52 duplicates, 64 not duplicates)
- Ambiguous pairs excluded from metrics: **4**

## Overall

| Approach | TP | FP | TN | FN | Precision | Recall | F1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Semantic only (cosine >= 0.75) | 50 | 57 | 7 | 2 | 0.4673 | 0.9615 | 0.6289 |
| Semantic + Jaccard (<= 0.32) | 30 | 6 | 58 | 22 | 0.8333 | 0.5769 | 0.6818 |

## By category

| Category | N | A FP | A FN | A Prec | A Rec | A F1 | B FP | B FN | B Prec | B Rec | B F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A same meaning, different wording | 28 | 0 | 2 | 1.0000 | 0.9286 | 0.9630 | 0 | 4 | 1.0000 | 0.8571 | 0.9231 |
| B same meaning, similar wording | 24 | 0 | 0 | 1.0000 | 1.0000 | 1.0000 | 0 | 18 | 1.0000 | 0.2500 | 0.4000 |
| C opposite decision, similar wording | 20 | 19 | 0 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 | 0.0000 | 0.0000 | 0.0000 |
| D same template, different values | 24 | 22 | 0 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 | 0.0000 | 0.0000 | 0.0000 |
| E related topic, different decision | 20 | 16 | 0 | 0.0000 | 0.0000 | 0.0000 | 6 | 0 | 0.0000 | 0.0000 | 0.0000 |

A = semantic only. B = semantic + Jaccard.

## 1. False positives removed by Jaccard

Semantic-only accepted these as duplicates. Jaccard rejected them. Expected label is not duplicate.

Count: **51**

| Cat | Decision A | Decision B | Expected | Cosine | Jaccard | Semantic only | Semantic + Jaccard |
| --- | --- | --- | --- | --- | --- | --- | --- |
| C | We approved the migration to PostgreSQL. | We rejected the migration to PostgreSQL. | not duplicate | 0.8617 | 0.7143 | accept | reject |
| C | We decided to ship the product on Friday. | We decided not to ship the product on Friday. | not duplicate | 0.8431 | 0.8889 | accept | reject |
| C | Raise prices by 10% starting next quarter. | Do not raise prices by 10% starting next quarter. | not duplicate | 0.8699 | 0.7778 | accept | reject |
| C | Remote work is allowed three days per week. | Remote work is not allowed three days per week. | not duplicate | 0.9200 | 0.8889 | accept | reject |
| C | We will enable dark mode for the dashboard. | We will not enable dark mode for the dashboard. | not duplicate | 0.8626 | 0.8889 | accept | reject |
| C | The team approved a hiring freeze. | The team rejected a hiring freeze. | not duplicate | 0.9097 | 0.7143 | accept | reject |
| C | We chose to delay the launch until Monday. | We chose not to delay the launch until Monday. | not duplicate | 0.9041 | 0.8889 | accept | reject |
| C | Security approved rotating API keys quarterly. | Security rejected rotating API keys quarterly. | not duplicate | 0.8941 | 0.7143 | accept | reject |
| C | We agreed to keep logs for ninety days. | We agreed not to keep logs for ninety days. | not duplicate | 0.9055 | 0.8889 | accept | reject |
| C | Product decided to remove the onboarding wizard. | Product decided to keep the onboarding wizard. | not duplicate | 0.8322 | 0.7500 | accept | reject |
| C | We will migrate CI to GitHub Actions. | We will not migrate CI to GitHub Actions. | not duplicate | 0.8158 | 0.8750 | accept | reject |
| C | We decided to block public S3 buckets. | We decided to allow public S3 buckets. | not duplicate | 0.8879 | 0.7500 | accept | reject |
| C | We will sunset the legacy billing API. | We will keep the legacy billing API. | not duplicate | 0.9033 | 0.7500 | accept | reject |
| C | Leadership approved three remote days each week. | Leadership denied three remote days each week. | not duplicate | 0.7853 | 0.7500 | accept | reject |
| C | Finance approved the 10 percent price increase. | Finance rejected the 10 percent price increase. | not duplicate | 0.8272 | 0.7500 | accept | reject |
| C | We will launch the mobile app in March. | We will postpone the mobile app past March. | not duplicate | 0.8028 | 0.6000 | accept | reject |
| C | Cancel the vendor contract with Acme. | Renew the vendor contract with Acme. | not duplicate | 0.8609 | 0.7143 | accept | reject |
| C | Freeze hiring for the design team. | Continue hiring for the design team. | not duplicate | 0.8258 | 0.7143 | accept | reject |
| C | Require two-factor authentication for admins. | Two-factor authentication for admins is optional. | not duplicate | 0.8604 | 0.6250 | accept | reject |
| D | Award contract CONT-45 to vendor VND-45 for $775k. | Award contract CONT-60 to vendor VND-60 for $130k. | not duplicate | 0.8807 | 0.6364 | accept | reject |
| D | Set customer CUST-91 discount to 24% through 8/2025. | Set customer CUST-12 discount to 5% through 1/2024. | not duplicate | 0.8747 | 0.4286 | accept | reject |
| D | Promote employee EMP-3813 to level L6 on team foxtrot. | Promote employee EMP-102 to level L2 on team alpha. | not duplicate | 0.8475 | 0.5385 | accept | reject |
| D | Cap wallet WAL-4026 spend at $52/day. | Cap wallet WAL-88 spend at $900/day. | not duplicate | 0.8515 | 0.6000 | accept | reject |
| D | Schedule maintenance WIN-760 on host HST-760 at 16:00 UTC. | Schedule maintenance WIN-12 on host HST-12 at 02:00 UTC. | not duplicate | 0.9130 | 0.6923 | accept | reject |
| D | Buy domain DOM-3779.example for campaign CMP-3779. | Buy domain DOM-15.example for campaign CMP-15. | not duplicate | 0.9169 | 0.7778 | accept | reject |
| D | Ship hotfix HF-42 for bug BUG-42 tonight. | Ship hotfix HF-99 for bug BUG-99 tonight. | not duplicate | 0.9692 | 0.7500 | accept | reject |
| D | Reject RFC RFC-41 about feature FEAT-41. | Reject RFC RFC-900 about feature FEAT-900. | not duplicate | 0.9346 | 0.7143 | accept | reject |
| D | Grant role ROLE-8 to service account SA-8 only. | Grant role ROLE-77 to service account SA-77 only. | not duplicate | 0.9358 | 0.7778 | accept | reject |
| D | Close office LOC-1 in Lisbon effective 1/2. | Close office LOC-2 in Oslo effective 6/15. | not duplicate | 0.8160 | 0.5455 | accept | reject |
| D | Raise the price of plan BASIC by $5. | Raise the price of plan ENTERPRISE by $50. | not duplicate | 0.8179 | 0.6000 | accept | reject |
| D | Hire 2 backend engineers in Q1. | Hire 8 frontend engineers in Q4. | not duplicate | 0.8338 | 0.3333 | accept | reject |
| D | Set the SLA for P1 tickets to 1 hour. | Set the SLA for P3 tickets to 3 days. | not duplicate | 0.8593 | 0.5000 | accept | reject |
| D | Route region REG-1 traffic to cluster CLS-1. | Route region REG-9 traffic to cluster CLS-9. | not duplicate | 0.9249 | 0.7778 | accept | reject |
| D | Archive channel CH-10 and move history to vault VT-10. | Archive channel CH-88 and move history to vault VT-88. | not duplicate | 0.9135 | 0.8182 | accept | reject |
| D | Invoice customer CUST-3 for $1200 in March. | Invoice customer CUST-40 for $80 in November. | not duplicate | 0.8525 | 0.4545 | accept | reject |
| D | Rotate API key KEY-1 every 30 days. | Rotate API key KEY-9 every 365 days. | not duplicate | 0.9388 | 0.5556 | accept | reject |
| D | Block IP range 10.0.1.0/24 after incident INC-1. | Block IP range 10.9.9.0/24 after incident INC-9. | not duplicate | 0.9691 | 0.8182 | accept | reject |
| D | Rename project PRJ-65 to codename onyx65. | Rename project PRJ-2 to codename cedar2. | not duplicate | 0.8889 | 0.5556 | accept | reject |
| D | The launch date is Friday March 1. | The launch date is Monday June 20. | not duplicate | 0.8216 | 0.4000 | accept | reject |
| D | Discount for CUST-1 is 10 percent. | Discount for CUST-2 is 40 percent. | not duplicate | 0.8225 | 0.5556 | accept | reject |
| D | Budget for campaign CMP-3 is $10k. | Budget for campaign CMP-8 is $500k. | not duplicate | 0.8294 | 0.5556 | accept | reject |
| E | Hire two backend engineers this quarter. | Hire two designers this quarter. | not duplicate | 0.8744 | 0.5714 | accept | reject |
| E | Deprecate the legacy API by end of year. | Document the legacy API by end of year. | not duplicate | 0.8682 | 0.7778 | accept | reject |
| E | Ship the product on Friday. | Announce the product on Friday. | not duplicate | 0.8926 | 0.6667 | accept | reject |
| E | Require MFA for admin accounts. | Require MFA training for admin accounts. | not duplicate | 0.9348 | 0.8333 | accept | reject |
| E | Block public S3 buckets. | Audit public S3 buckets. | not duplicate | 0.8193 | 0.6000 | accept | reject |
| E | Approve headcount for two backend engineers. | Approve laptops for two backend engineers. | not duplicate | 0.8355 | 0.7143 | accept | reject |
| E | Three days remote each week. | Three days of on-call each week. | not duplicate | 0.8540 | 0.5000 | accept | reject |
| E | Increase prices 10 percent. | Increase support staffing 10 percent. | not duplicate | 0.8037 | 0.5000 | accept | reject |
| E | Close the Dhaka office. | Renovate the Dhaka office. | not duplicate | 0.8086 | 0.6000 | accept | reject |
| E | Cancel the Acme contract. | Renegotiate the Acme contract. | not duplicate | 0.8507 | 0.6000 | accept | reject |

## 2. Genuine duplicates rejected by Jaccard

Semantic-only accepted these, and the expected label is duplicate. Jaccard rejected them because word overlap was above 0.32 and the texts were not identical.

Count: **20**

| Cat | Decision A | Decision B | Expected | Cosine | Jaccard | Semantic only | Semantic + Jaccard |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A | Decision: add two backend engineers ASAP this quarter. | Hire two backend engineers this quarter. | duplicate | 0.9138 | 0.5556 | accept | reject |
| A | Ship the mobile app in March. | The mobile application goes live in March. | duplicate | 0.8426 | 0.4444 | accept | reject |
| B | We approved the migration to PostgreSQL. | The migration to PostgreSQL was approved. | duplicate | 0.9808 | 0.7143 | accept | reject |
| B | We decided to ship the product on Friday. | We'll put the product out on Friday. | duplicate | 0.9069 | 0.4545 | accept | reject |
| B | Decision: Friday launch for the product. | Friday is the go-live day for the product. | duplicate | 0.8236 | 0.4000 | accept | reject |
| B | Hire two backend engineers this quarter. | Recruit two backend developers this quarter. | duplicate | 0.9373 | 0.5000 | accept | reject |
| B | We will enable dark mode for the dashboard. | Dark mode will be enabled for the dashboard. | duplicate | 0.9841 | 0.6000 | accept | reject |
| B | The team approved a hiring freeze. | A hiring freeze was approved by the team. | duplicate | 0.9873 | 0.7500 | accept | reject |
| B | We chose to delay the launch until Monday. | The launch was delayed until Monday. | duplicate | 0.9249 | 0.4000 | accept | reject |
| B | Security approved rotating API keys quarterly. | Rotating API keys quarterly was approved by security. | duplicate | 0.9263 | 0.7500 | accept | reject |
| B | We agreed to keep logs for ninety days. | Logs will be kept for ninety days. | duplicate | 0.9416 | 0.3636 | accept | reject |
| B | We will migrate CI to GitHub Actions. | CI will be migrated to GitHub Actions. | duplicate | 0.9706 | 0.5556 | accept | reject |
| B | Finance approved the 10 percent price increase. | The 10 percent price increase was approved by finance. | duplicate | 0.9864 | 0.7778 | accept | reject |
| B | Leadership approved three remote days each week. | Three remote days each week were approved. | duplicate | 0.9364 | 0.7500 | accept | reject |
| B | We will sunset the legacy billing API. | The legacy billing API will be sunset. | duplicate | 0.9564 | 0.7500 | accept | reject |
| B | The board approved hiring two backend engineers. | Hiring two backend engineers was approved. | duplicate | 0.9584 | 0.6250 | accept | reject |
| B | We agreed the product ships on Friday. | The product ships on Friday, as agreed. | duplicate | 0.9545 | 0.7500 | accept | reject |
| B | We will standardize on PostgreSQL for storage. | Storage will be standardized on PostgreSQL. | duplicate | 0.9660 | 0.4444 | accept | reject |
| B | We decided to raise prices ten percent next quarter. | Next quarter prices will be raised ten percent. | duplicate | 0.9609 | 0.4167 | accept | reject |
| B | We approved closing the legacy API in December. | Closing the legacy API in December was approved. | duplicate | 0.9835 | 0.7778 | accept | reject |

## 3. False positives that passed both filters

Expected not duplicate, but both approaches accepted them.

Count: **6**

| Cat | Decision A | Decision B | Expected | Cosine | Jaccard | Semantic only | Semantic + Jaccard |
| --- | --- | --- | --- | --- | --- | --- | --- |
| E | Migrate the database to PostgreSQL. | Increase PostgreSQL database storage. | not duplicate | 0.7513 | 0.2857 | accept | accept |
| E | Raise prices by 10% starting next quarter. | Offer a 10% discount next quarter. | not duplicate | 0.8691 | 0.3000 | accept | accept |
| E | Move CI to GitHub Actions. | Add a GitHub Actions badge to the README. | not duplicate | 0.7660 | 0.3000 | accept | accept |
| E | Standardize on PostgreSQL. | Upgrade PostgreSQL to the next minor version. | not duplicate | 0.7850 | 0.1111 | accept | accept |
| E | Launch on Friday. | Run a load test on Friday. | not duplicate | 0.8018 | 0.2857 | accept | accept |
| E | Enable dark mode. | Fix the dark mode contrast bug. | not duplicate | 0.8258 | 0.2857 | accept | accept |

## Jaccard cutoff sweep (cosine fixed at 0.75)

| Jaccard cutoff | TP | FP | TN | FN | Precision | Recall | F1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| none | 50 | 57 | 7 | 2 | 0.4673 | 0.9615 | 0.6289 |
| 0.2 | 18 | 1 | 63 | 34 | 0.9474 | 0.3462 | 0.5070 |
| 0.25 | 23 | 1 | 63 | 29 | 0.9583 | 0.4423 | 0.6053 |
| 0.32 | 30 | 6 | 58 | 22 | 0.8333 | 0.5769 | 0.6818 |
| 0.4 | 33 | 8 | 56 | 19 | 0.8049 | 0.6346 | 0.7097 |
| 0.55 | 38 | 15 | 49 | 14 | 0.7170 | 0.7308 | 0.7238 |

`none` means no Jaccard filter (semantic only).

## Ambiguous pairs (excluded from metrics)

| Cat | Decision A | Decision B | Expected | Cosine | Jaccard | Semantic only | Semantic + Jaccard |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ambiguous | Ship the product on Friday. | Release a beta on Friday. | not duplicate | 0.7284 | 0.2500 | reject | reject |
| ambiguous | Deprecate the legacy API by December. | Phase out one legacy endpoint by December. | not duplicate | 0.8037 | 0.3000 | accept | accept |
| ambiguous | Hire two backend engineers. | Hire backend engineers this quarter. | duplicate | 0.8713 | 0.5000 | accept | reject |
| ambiguous | Use PostgreSQL as primary database. | Use PostgreSQL for analytics only. | not duplicate | 0.8189 | 0.2500 | accept | accept |

## Measured answers

1. Incorrect semantic matches (false positives of cosine >= 0.75): **57** of 64 negative pairs.
2. Of those, Jaccard removed **51**.
3. Genuine duplicates Jaccard rejected: **20**.
4. Precision changed by **+0.3660** (0.4673 → 0.8333).
5. Recall changed by **-0.3846** (0.9615 → 0.5769).
6. F1 changed by **+0.0529** (0.6289 → 0.6818).
7. On this labeled set, cutoff 0.32 is the sweep point to compare against 0.20, 0.25, 0.40, and 0.55 in the table above.
8. Limits: labels were written for this experiment, not drawn from a production review. Categories A–E are controlled examples, including planted paraphrases. Pair scoring ignores neighbor search and clustering. Opposite-wording and template pairs were chosen to be challenging, so the false-positive rate is not an estimate of the 10k corpus.
