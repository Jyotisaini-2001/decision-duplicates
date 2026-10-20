# Experiment 2 — critical decision checks

Exploratory evaluation. The rules were written after seeing the kinds of errors in Experiment 1 (opposites, template values, entities). They were not retuned after this run, but the 116-pair result can still be biased. The held-out sentences are new, small, and written in the same session, so they are only a weak generalization check.

Scores for the 116 pairs are reused from `experiments/pair_scores.csv`. Embeddings were not recomputed for those pairs.

## Main comparison

| Metric | Semantic only | Semantic + Jaccard | Semantic + critical checks |
| --- | --- | --- | --- |
| Precision | 0.4673 | 0.8333 | 0.7246 |
| Recall | 0.9615 | 0.5769 | 0.9615 |
| F1 | 0.6289 | 0.6818 | 0.8264 |
| False positives | 57 | 6 | 19 |
| False negatives | 2 | 22 | 2 |
| True positives | 50 | 30 | 50 |
| True negatives | 7 | 58 | 45 |

## By category

| Category | N | A FP | A FN | A F1 | B FP | B FN | B F1 | C FP | C FN | C F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | 28 | 0 | 2 | 0.9630 | 0 | 4 | 0.9231 | 0 | 2 | 0.9630 |
| B | 24 | 0 | 0 | 1.0000 | 0 | 18 | 0.4000 | 0 | 0 | 1.0000 |
| C | 20 | 19 | 0 | 0.0000 | 0 | 0 | 0.0000 | 3 | 0 | 0.0000 |
| D | 24 | 22 | 0 | 0.0000 | 0 | 0 | 0.0000 | 0 | 0 | 0.0000 |
| E | 20 | 16 | 0 | 0.0000 | 6 | 0 | 0.0000 | 16 | 0 | 0.0000 |

## What changed

- Semantic-only false positives: **57**
- Of those, critical checks removed: **38**
- Genuine duplicates in the set: **52**
- Genuine duplicates preserved by critical checks: **50**
- Genuine duplicates rejected by critical checks: **2**
- Genuine duplicates rejected by Jaccard but kept by critical checks: **20**
- Semantic false positives that still pass critical checks: **19**

## Incorrect matches removed by critical checks

| Cat | Decision A | Decision B | Expected | Cosine | Jaccard | A | B | C | Critical reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C | We approved the migration to PostgreSQL. | We rejected the migration to PostgreSQL. | not duplicate | 0.8617 | 0.7143 | accept | reject | reject | opposite actions about the same subject (shared: migration, postgresql) |
| C | We decided to ship the product on Friday. | We decided not to ship the product on Friday. | not duplicate | 0.8431 | 0.8889 | accept | reject | reject | opposite actions about the same subject (shared: decided, friday, product) |
| C | Raise prices by 10% starting next quarter. | Do not raise prices by 10% starting next quarter. | not duplicate | 0.8699 | 0.7778 | accept | reject | reject | opposite actions about the same subject (shared: prices, quarter) |
| C | Remote work is allowed three days per week. | Remote work is not allowed three days per week. | not duplicate | 0.9200 | 0.8889 | accept | reject | reject | opposite actions about the same subject (shared: days, remote, three, week, work) |
| C | We will enable dark mode for the dashboard. | We will not enable dark mode for the dashboard. | not duplicate | 0.8626 | 0.8889 | accept | reject | reject | opposite actions about the same subject (shared: dark, dashboard, mode) |
| C | The team approved a hiring freeze. | The team rejected a hiring freeze. | not duplicate | 0.9097 | 0.7143 | accept | reject | reject | different team: ['approved'] vs ['rejected'] |
| C | We chose to delay the launch until Monday. | We chose not to delay the launch until Monday. | not duplicate | 0.9041 | 0.8889 | accept | reject | reject | opposite actions about the same subject (shared: chose, monday, until) |
| C | Security approved rotating API keys quarterly. | Security rejected rotating API keys quarterly. | not duplicate | 0.8941 | 0.7143 | accept | reject | reject | opposite actions about the same subject (shared: api, keys, quarterly, rotating, security) |
| C | We agreed to keep logs for ninety days. | We agreed not to keep logs for ninety days. | not duplicate | 0.9055 | 0.8889 | accept | reject | reject | opposite actions about the same subject (shared: agreed, days, logs, ninety) |
| C | Product decided to remove the onboarding wizard. | Product decided to keep the onboarding wizard. | not duplicate | 0.8322 | 0.7500 | accept | reject | reject | opposite actions about the same subject (shared: decided, onboarding, product, wizard) |
| C | We decided to block public S3 buckets. | We decided to allow public S3 buckets. | not duplicate | 0.8879 | 0.7500 | accept | reject | reject | opposite actions about the same subject (shared: buckets, decided, public, s3) |
| C | We will sunset the legacy billing API. | We will keep the legacy billing API. | not duplicate | 0.9033 | 0.7500 | accept | reject | reject | opposite actions about the same subject (shared: api, billing, legacy) |
| C | Leadership approved three remote days each week. | Leadership denied three remote days each week. | not duplicate | 0.7853 | 0.7500 | accept | reject | reject | opposite actions about the same subject (shared: days, leadership, remote, three, week) |
| C | Finance approved the 10 percent price increase. | Finance rejected the 10 percent price increase. | not duplicate | 0.8272 | 0.7500 | accept | reject | reject | opposite actions about the same subject (shared: finance, percent, price) |
| C | We will launch the mobile app in March. | We will postpone the mobile app past March. | not duplicate | 0.8028 | 0.6000 | accept | reject | reject | opposite actions about the same subject (shared: app, march, mobile) |
| C | Cancel the vendor contract with Acme. | Renew the vendor contract with Acme. | not duplicate | 0.8609 | 0.7143 | accept | reject | reject | opposite actions about the same subject (shared: acme, contract, vendor) |
| D | Award contract CONT-45 to vendor VND-45 for $775k. | Award contract CONT-60 to vendor VND-60 for $130k. | not duplicate | 0.8807 | 0.6364 | accept | reject | reject | different money: [775000.0] vs [130000.0]; different CONT id: ['45'] vs ['60'] |
| D | Set customer CUST-91 discount to 24% through 8/2025. | Set customer CUST-12 discount to 5% through 1/2024. | not duplicate | 0.8747 | 0.4286 | accept | reject | reject | different CUST id: ['91'] vs ['12'] |
| D | Promote employee EMP-3813 to level L6 on team foxtrot. | Promote employee EMP-102 to level L2 on team alpha. | not duplicate | 0.8475 | 0.5385 | accept | reject | reject | different EMP id: ['3813'] vs ['102'] |
| D | Cap wallet WAL-4026 spend at $52/day. | Cap wallet WAL-88 spend at $900/day. | not duplicate | 0.8515 | 0.6000 | accept | reject | reject | different money: [52.0] vs [900.0]; different WAL id: ['4026'] vs ['88'] |
| D | Schedule maintenance WIN-760 on host HST-760 at 16:00 UTC. | Schedule maintenance WIN-12 on host HST-12 at 02:00 UTC. | not duplicate | 0.9130 | 0.6923 | accept | reject | reject | different time: ['16:00'] vs ['02:00']; different WIN id: ['760'] vs ['12'] |
| D | Buy domain DOM-3779.example for campaign CMP-3779. | Buy domain DOM-15.example for campaign CMP-15. | not duplicate | 0.9169 | 0.7778 | accept | reject | reject | different CMP id: ['3779'] vs ['15'] |
| D | Ship hotfix HF-42 for bug BUG-42 tonight. | Ship hotfix HF-99 for bug BUG-99 tonight. | not duplicate | 0.9692 | 0.7500 | accept | reject | reject | different HF id: ['42'] vs ['99'] |
| D | Reject RFC RFC-41 about feature FEAT-41. | Reject RFC RFC-900 about feature FEAT-900. | not duplicate | 0.9346 | 0.7143 | accept | reject | reject | different RFC id: ['41'] vs ['900'] |
| D | Grant role ROLE-8 to service account SA-8 only. | Grant role ROLE-77 to service account SA-77 only. | not duplicate | 0.9358 | 0.7778 | accept | reject | reject | different SA id: ['8'] vs ['77'] |
| D | Close office LOC-1 in Lisbon effective 1/2. | Close office LOC-2 in Oslo effective 6/15. | not duplicate | 0.8160 | 0.5455 | accept | reject | reject | different calendar date: ['1/2'] vs ['6/15']; different LOC id: ['1'] vs ['2'] |
| D | Raise the price of plan BASIC by $5. | Raise the price of plan ENTERPRISE by $50. | not duplicate | 0.8179 | 0.6000 | accept | reject | reject | different money: [5.0] vs [50.0]; different plan: ['basic'] vs ['enterprise'] |
| D | Hire 2 backend engineers in Q1. | Hire 8 frontend engineers in Q4. | not duplicate | 0.8338 | 0.3333 | accept | reject | reject | different quarter: ['1'] vs ['4'] |
| D | Set the SLA for P1 tickets to 1 hour. | Set the SLA for P3 tickets to 3 days. | not duplicate | 0.8593 | 0.5000 | accept | reject | reject | different duration: [1.0] vs [72.0] |
| D | Route region REG-1 traffic to cluster CLS-1. | Route region REG-9 traffic to cluster CLS-9. | not duplicate | 0.9249 | 0.7778 | accept | reject | reject | different REG id: ['1'] vs ['9'] |
| D | Archive channel CH-10 and move history to vault VT-10. | Archive channel CH-88 and move history to vault VT-88. | not duplicate | 0.9135 | 0.8182 | accept | reject | reject | different CH id: ['10'] vs ['88'] |
| D | Invoice customer CUST-3 for $1200 in March. | Invoice customer CUST-40 for $80 in November. | not duplicate | 0.8525 | 0.4545 | accept | reject | reject | different money: [1200.0] vs [80.0]; different month: [3] vs [11]; different CUST id: ['3'] vs ['40'] |
| D | Rotate API key KEY-1 every 30 days. | Rotate API key KEY-9 every 365 days. | not duplicate | 0.9388 | 0.5556 | accept | reject | reject | different duration: [720.0] vs [8760.0]; different KEY id: ['1'] vs ['9'] |
| D | Block IP range 10.0.1.0/24 after incident INC-1. | Block IP range 10.9.9.0/24 after incident INC-9. | not duplicate | 0.9691 | 0.8182 | accept | reject | reject | different INC id: ['1'] vs ['9'] |
| D | Rename project PRJ-65 to codename onyx65. | Rename project PRJ-2 to codename cedar2. | not duplicate | 0.8889 | 0.5556 | accept | reject | reject | different PRJ id: ['65'] vs ['2'] |
| D | The launch date is Friday March 1. | The launch date is Monday June 20. | not duplicate | 0.8216 | 0.4000 | accept | reject | reject | different weekday: ['friday'] vs ['monday'] |
| D | Discount for CUST-1 is 10 percent. | Discount for CUST-2 is 40 percent. | not duplicate | 0.8225 | 0.5556 | accept | reject | reject | different percent: [10.0] vs [40.0]; different CUST id: ['1'] vs ['2'] |
| D | Budget for campaign CMP-3 is $10k. | Budget for campaign CMP-8 is $500k. | not duplicate | 0.8294 | 0.5556 | accept | reject | reject | different money: [10000.0] vs [500000.0]; different CMP id: ['3'] vs ['8'] |

## Genuine duplicates kept by critical checks but rejected by Jaccard

| Cat | Decision A | Decision B | Expected | Cosine | Jaccard | A | B | C | Critical reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | Decision: add two backend engineers ASAP this quarter. | Hire two backend engineers this quarter. | duplicate | 0.9138 | 0.5556 | accept | reject | accept | action polarity not confident on both sides |
| A | Ship the mobile app in March. | The mobile application goes live in March. | duplicate | 0.8426 | 0.4444 | accept | reject | accept | action polarity not confident on both sides |
| B | We approved the migration to PostgreSQL. | The migration to PostgreSQL was approved. | duplicate | 0.9808 | 0.7143 | accept | reject | accept | no confident conflict |
| B | We decided to ship the product on Friday. | We'll put the product out on Friday. | duplicate | 0.9069 | 0.4545 | accept | reject | accept | action polarity not confident on both sides |
| B | Decision: Friday launch for the product. | Friday is the go-live day for the product. | duplicate | 0.8236 | 0.4000 | accept | reject | accept | action polarity not confident on both sides |
| B | Hire two backend engineers this quarter. | Recruit two backend developers this quarter. | duplicate | 0.9373 | 0.5000 | accept | reject | accept | no confident conflict |
| B | We will enable dark mode for the dashboard. | Dark mode will be enabled for the dashboard. | duplicate | 0.9841 | 0.6000 | accept | reject | accept | no confident conflict |
| B | The team approved a hiring freeze. | A hiring freeze was approved by the team. | duplicate | 0.9873 | 0.7500 | accept | reject | accept | no confident conflict |
| B | We chose to delay the launch until Monday. | The launch was delayed until Monday. | duplicate | 0.9249 | 0.4000 | accept | reject | accept | no confident conflict |
| B | Security approved rotating API keys quarterly. | Rotating API keys quarterly was approved by security. | duplicate | 0.9263 | 0.7500 | accept | reject | accept | no confident conflict |
| B | We agreed to keep logs for ninety days. | Logs will be kept for ninety days. | duplicate | 0.9416 | 0.3636 | accept | reject | accept | no confident conflict |
| B | We will migrate CI to GitHub Actions. | CI will be migrated to GitHub Actions. | duplicate | 0.9706 | 0.5556 | accept | reject | accept | action polarity not confident on both sides |
| B | Finance approved the 10 percent price increase. | The 10 percent price increase was approved by finance. | duplicate | 0.9864 | 0.7778 | accept | reject | accept | no confident conflict |
| B | Leadership approved three remote days each week. | Three remote days each week were approved. | duplicate | 0.9364 | 0.7500 | accept | reject | accept | no confident conflict |
| B | We will sunset the legacy billing API. | The legacy billing API will be sunset. | duplicate | 0.9564 | 0.7500 | accept | reject | accept | no confident conflict |
| B | The board approved hiring two backend engineers. | Hiring two backend engineers was approved. | duplicate | 0.9584 | 0.6250 | accept | reject | accept | no confident conflict |
| B | We agreed the product ships on Friday. | The product ships on Friday, as agreed. | duplicate | 0.9545 | 0.7500 | accept | reject | accept | action polarity not confident on both sides |
| B | We will standardize on PostgreSQL for storage. | Storage will be standardized on PostgreSQL. | duplicate | 0.9660 | 0.4444 | accept | reject | accept | action polarity not confident on both sides |
| B | We decided to raise prices ten percent next quarter. | Next quarter prices will be raised ten percent. | duplicate | 0.9609 | 0.4167 | accept | reject | accept | no confident conflict |
| B | We approved closing the legacy API in December. | Closing the legacy API in December was approved. | duplicate | 0.9835 | 0.7778 | accept | reject | accept | no confident conflict |

## Genuine duplicates incorrectly rejected by critical checks

_No examples in this set._

## Incorrect matches that still pass critical checks

| Cat | Decision A | Decision B | Expected | Cosine | Jaccard | A | B | C | Critical reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C | We will migrate CI to GitHub Actions. | We will not migrate CI to GitHub Actions. | not duplicate | 0.8158 | 0.8750 | accept | reject | accept | action polarity not confident on both sides |
| C | Freeze hiring for the design team. | Continue hiring for the design team. | not duplicate | 0.8258 | 0.7143 | accept | reject | accept | action polarity not confident on both sides |
| C | Require two-factor authentication for admins. | Two-factor authentication for admins is optional. | not duplicate | 0.8604 | 0.6250 | accept | reject | accept | action polarity not confident on both sides |
| E | Migrate the database to PostgreSQL. | Increase PostgreSQL database storage. | not duplicate | 0.7513 | 0.2857 | accept | accept | accept | action polarity not confident on both sides |
| E | Hire two backend engineers this quarter. | Hire two designers this quarter. | not duplicate | 0.8744 | 0.5714 | accept | reject | accept | no confident conflict |
| E | Raise prices by 10% starting next quarter. | Offer a 10% discount next quarter. | not duplicate | 0.8691 | 0.3000 | accept | accept | accept | opposite actions but the subject overlap is too small |
| E | Deprecate the legacy API by end of year. | Document the legacy API by end of year. | not duplicate | 0.8682 | 0.7778 | accept | reject | accept | action polarity not confident on both sides |
| E | Ship the product on Friday. | Announce the product on Friday. | not duplicate | 0.8926 | 0.6667 | accept | reject | accept | action polarity not confident on both sides |
| E | Require MFA for admin accounts. | Require MFA training for admin accounts. | not duplicate | 0.9348 | 0.8333 | accept | reject | accept | no confident conflict |
| E | Block public S3 buckets. | Audit public S3 buckets. | not duplicate | 0.8193 | 0.6000 | accept | reject | accept | action polarity not confident on both sides |
| E | Move CI to GitHub Actions. | Add a GitHub Actions badge to the README. | not duplicate | 0.7660 | 0.3000 | accept | accept | accept | action polarity not confident on both sides |
| E | Approve headcount for two backend engineers. | Approve laptops for two backend engineers. | not duplicate | 0.8355 | 0.7143 | accept | reject | accept | no confident conflict |
| E | Standardize on PostgreSQL. | Upgrade PostgreSQL to the next minor version. | not duplicate | 0.7850 | 0.1111 | accept | accept | accept | action polarity not confident on both sides |
| E | Three days remote each week. | Three days of on-call each week. | not duplicate | 0.8540 | 0.5000 | accept | reject | accept | action polarity not confident on both sides |
| E | Launch on Friday. | Run a load test on Friday. | not duplicate | 0.8018 | 0.2857 | accept | accept | accept | action polarity not confident on both sides |
| E | Increase prices 10 percent. | Increase support staffing 10 percent. | not duplicate | 0.8037 | 0.5000 | accept | reject | accept | no confident conflict |
| E | Close the Dhaka office. | Renovate the Dhaka office. | not duplicate | 0.8086 | 0.6000 | accept | reject | accept | action polarity not confident on both sides |
| E | Cancel the Acme contract. | Renegotiate the Acme contract. | not duplicate | 0.8507 | 0.6000 | accept | reject | accept | action polarity not confident on both sides |
| E | Enable dark mode. | Fix the dark mode contrast bug. | not duplicate | 0.8258 | 0.2857 | accept | accept | accept | action polarity not confident on both sides |

## Held-out check (14 new pairs)

| Set | A prec | A rec | A F1 | B prec | B rec | B F1 | C prec | C rec | C F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Held-out | 0.3077 | 0.8000 | 0.4444 | 1.0000 | 0.4000 | 0.5714 | 0.5714 | 0.8000 | 0.6667 |

| Cat | Decision A | Decision B | Expected | Cosine | Jaccard | A | B | C | Critical reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | The board signed off on Postgres as the system of record. | PostgreSQL will be our main database. | duplicate | 0.6540 | 0.0000 | reject | reject | reject | action polarity not confident on both sides |
| A | Staff may work remotely three days a week. | The hybrid schedule allows three work-from-home days weekly. | duplicate | 0.7929 | 0.2000 | accept | accept | accept | action polarity not confident on both sides |
| A | We will increase prices 10 percent next quarter. | A ten percent price rise starts next quarter. | duplicate | 0.9076 | 0.2308 | accept | accept | accept | action polarity not confident on both sides |
| B | The security team approved quarterly key rotation. | Quarterly key rotation was approved by the security team. | duplicate | 0.9854 | 0.7778 | accept | reject | accept | no confident conflict |
| B | We will keep customer logs for 90 days. | Customer logs will be kept for 90 days. | duplicate | 0.9795 | 0.6000 | accept | reject | accept | no confident conflict |
| C | We approved the annual budget. | We rejected the annual budget. | not duplicate | 0.7572 | 0.6667 | accept | reject | reject | opposite actions about the same subject (shared: annual, budget) |
| C | Enable audit logging for production. | Disable audit logging for production. | not duplicate | 0.8737 | 0.6667 | accept | reject | reject | opposite actions about the same subject (shared: audit, logging, production) |
| C | Continue the pilot with vendor Northwind. | Stop the pilot with vendor Northwind. | not duplicate | 0.8926 | 0.7143 | accept | reject | reject | opposite actions about the same subject (shared: northwind, pilot, vendor) |
| D | Hire 4 analysts in Berlin. | Hire 9 analysts in Tokyo. | not duplicate | 0.7687 | 0.4286 | accept | reject | reject | different headcount: [4.0] vs [9.0]; different place: ['berlin'] vs ['tokyo'] |
| D | Pay invoice INV-14 for $2,000. | Pay invoice INV-14 for $9,000. | not duplicate | 0.8991 | 0.7500 | accept | reject | reject | different money: [2.0] vs [9.0] |
| D | Close ticket T-10 after the customer confirms. | Close ticket T-11 after the customer confirms. | not duplicate | 0.9623 | 0.7778 | accept | reject | accept | action polarity not confident on both sides |
| D | Launch the billing change in April. | Launch the billing change in October. | not duplicate | 0.9143 | 0.7143 | accept | reject | reject | different month: [4] vs [10] |
| E | Add a read replica for the billing database. | Add an index to the billing database. | not duplicate | 0.8103 | 0.3636 | accept | reject | accept | action polarity not confident on both sides |
| E | Document the refund policy. | Change the refund policy. | not duplicate | 0.8523 | 0.6000 | accept | reject | accept | action polarity not confident on both sides |

## Unsupported cases

- Abbreviations and synonyms are not understood unless they share surface words or a listed action word.
- "Phase out" is handled; many other multi-word actions are not.
- A number is compared only when its unit is recognized (money, percent, headcount, duration). Bare numbers are ignored.
- "End of year" is treated as compatible with December, not as a conflict.
- If action polarity is ambiguous inside one sentence, the pair is marked unknown and not rejected.
- Entity detection covers ID prefixes, capitalized places after "in/at", plan/team/level labels, and IP addresses. Other names can be missed.
