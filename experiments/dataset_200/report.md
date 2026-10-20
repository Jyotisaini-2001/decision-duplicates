# Experiments on the 200-decision dataset

Source: `experiments/dataset_200/experiment_decision_dataset_200.json` (200 decisions).

Pairs are formed inside each group of five. Gold labels come from the group structure after reading all 40 groups, not from model scores. Three ambiguous pairs in group 1 are excluded.

- Primary pairs: **397** (120 duplicates, 277 not duplicates)
- Model: `BAAI/bge-small-en-v1.5`

## Comparison

| Metric | Semantic only | Semantic + Jaccard | Semantic + critical checks |
| --- | --- | --- | --- |
| Precision | 0.3162 | 0.4393 | 0.4932 |
| Recall | 0.9250 | 0.3917 | 0.9000 |
| F1 | 0.4713 | 0.4141 | 0.6372 |
| False positives | 240 | 60 | 111 |
| False negatives | 9 | 73 | 12 |

## By category

| Category | N | A FP | A FN | A F1 | B FP | B FN | B F1 | C FP | C FN | C F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | 80 | 0 | 7 | 0.9542 | 0 | 38 | 0.6885 | 0 | 9 | 0.9404 |
| B | 40 | 0 | 2 | 0.9744 | 0 | 35 | 0.2222 | 0 | 3 | 0.9610 |
| C | 57 | 46 | 0 | 0.0000 | 12 | 0 | 0.0000 | 25 | 0 | 0.0000 |
| D | 132 | 127 | 0 | 0.0000 | 30 | 0 | 0.0000 | 29 | 0 | 0.0000 |
| E | 88 | 67 | 0 | 0.0000 | 18 | 0 | 0.0000 | 57 | 0 | 0.0000 |

## Counts

- Semantic-only false positives: **240**
- Removed by Jaccard: **180**
- Removed by critical checks: **129**
- Genuine duplicates: **120**
- Preserved by critical checks: **108**
- Rejected by Jaccard but kept by critical checks: **63**
- Genuine duplicates that passed cosine and were rejected by critical checks: **3**
- Semantic false positives that still pass critical checks: **111**

## Examples: false positives removed by critical checks

| Cat | ID A | Decision A | ID B | Decision B | Expected | Cosine | Jaccard | A | B | C |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C | dec_00001 | We approved the migration to PostgreSQL. | dec_00003 | We rejected the migration to PostgreSQL. | not duplicate | 0.8617 | 0.7143 | accept | reject | reject |
| C | dec_00002 | The migration to PostgreSQL was approved. | dec_00003 | We rejected the migration to PostgreSQL. | not duplicate | 0.8315 | 0.5000 | accept | reject | reject |
| C | dec_00006 | Enable dark mode for the dashboard. | dec_00008 | Disable dark mode for the dashboard. | not duplicate | 0.9615 | 0.7143 | accept | reject | reject |
| C | dec_00007 | Dark mode will be enabled for the dashboard. | dec_00008 | Disable dark mode for the dashboard. | not duplicate | 0.9163 | 0.5556 | accept | reject | reject |
| C | dec_00008 | Disable dark mode for the dashboard. | dec_00010 | We approved enabling dark mode on the dashboard. | not duplicate | 0.8927 | 0.4000 | accept | reject | reject |
| C | dec_00016 | Raise subscription prices by 10% next quarter. | dec_00018 | Do not raise subscription prices by 10% next quarter. | not duplicate | 0.8576 | 0.7778 | accept | reject | reject |
| E | dec_00016 | Raise subscription prices by 10% next quarter. | dec_00019 | Offer a 10% subscription discount next quarter. | not duplicate | 0.9043 | 0.4000 | accept | reject | reject |
| C | dec_00017 | A ten percent subscription price increase starts next quarter. | dec_00018 | Do not raise subscription prices by 10% next quarter. | not duplicate | 0.7538 | 0.2000 | accept | accept | reject |
| E | dec_00017 | A ten percent subscription price increase starts next quarter. | dec_00019 | Offer a 10% subscription discount next quarter. | not duplicate | 0.8332 | 0.3333 | accept | reject | reject |
| C | dec_00018 | Do not raise subscription prices by 10% next quarter. | dec_00020 | Increase subscription pricing ten percent next quarter. | not duplicate | 0.8145 | 0.2308 | accept | accept | reject |
| E | dec_00019 | Offer a 10% subscription discount next quarter. | dec_00020 | Increase subscription pricing ten percent next quarter. | not duplicate | 0.8795 | 0.2727 | accept | accept | reject |
| C | dec_00021 | Ship the mobile app on Friday. | dec_00023 | Do not ship the mobile app on Friday. | not duplicate | 0.8432 | 0.7500 | accept | reject | reject |

_Showing 12 of 129. Full list is in pair_scores.csv._

## Examples: genuine duplicates kept by critical checks but rejected by Jaccard

| Cat | ID A | Decision A | ID B | Decision B | Expected | Cosine | Jaccard | A | B | C |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | dec_00001 | We approved the migration to PostgreSQL. | dec_00002 | The migration to PostgreSQL was approved. | duplicate | 0.9808 | 0.7143 | accept | reject | accept |
| A | dec_00006 | Enable dark mode for the dashboard. | dec_00007 | Dark mode will be enabled for the dashboard. | duplicate | 0.9435 | 0.5556 | accept | reject | accept |
| B | dec_00006 | Enable dark mode for the dashboard. | dec_00010 | We approved enabling dark mode on the dashboard. | duplicate | 0.9307 | 0.4000 | accept | reject | accept |
| A | dec_00007 | Dark mode will be enabled for the dashboard. | dec_00010 | We approved enabling dark mode on the dashboard. | duplicate | 0.9429 | 0.3333 | accept | reject | accept |
| A | dec_00011 | Hire two backend engineers this quarter. | dec_00012 | Recruit 2 backend developers this quarter. | duplicate | 0.9268 | 0.3333 | accept | reject | accept |
| B | dec_00011 | Hire two backend engineers this quarter. | dec_00015 | Two backend engineering positions were approved for this quarter. | duplicate | 0.8593 | 0.3636 | accept | reject | accept |
| A | dec_00017 | A ten percent subscription price increase starts next quarter. | dec_00020 | Increase subscription pricing ten percent next quarter. | duplicate | 0.9297 | 0.6000 | accept | reject | accept |
| A | dec_00021 | Ship the mobile app on Friday. | dec_00022 | The mobile application goes live Friday. | duplicate | 0.8415 | 0.3333 | accept | reject | accept |
| B | dec_00021 | Ship the mobile app on Friday. | dec_00025 | The mobile app will be shipped on Friday. | duplicate | 0.9610 | 0.5556 | accept | reject | accept |
| B | dec_00026 | Retire the legacy billing API by December. | dec_00030 | The legacy billing API will be retired by December. | duplicate | 0.9476 | 0.6000 | accept | reject | accept |
| B | dec_00031 | Allow employees three remote days per week. | dec_00035 | Employees may work remotely three days each week. | duplicate | 0.9302 | 0.3636 | accept | reject | accept |
| B | dec_00036 | Rotate production API keys every 30 days. | dec_00040 | Production API keys will be rotated every thirty days. | duplicate | 0.9147 | 0.4545 | accept | reject | accept |

_Showing 12 of 63. Full list is in pair_scores.csv._

## Genuine duplicates rejected by critical checks

| Cat | ID A | Decision A | ID B | Decision B | Expected | Cosine | Jaccard | A | B | C |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | dec_00121 | Award contract CONT-45 to vendor VND-45 for $775k. | dec_00122 | Vendor VND-45 wins contract CONT-45 worth $775,000. | duplicate | 0.9486 | 0.3846 | accept | reject | reject |
| A | dec_00122 | Vendor VND-45 wins contract CONT-45 worth $775,000. | dec_00125 | Contract CONT-45 for $775k was awarded to VND-45. | duplicate | 0.9299 | 0.2857 | accept | accept | reject |
| B | dec_00176 | Set campaign CMP-3 budget to $10k. | dec_00180 | The budget for CMP-3 is $10,000. | duplicate | 0.8626 | 0.2500 | accept | accept | reject |

## Examples: incorrect matches that still pass critical checks

| Cat | ID A | Decision A | ID B | Decision B | Expected | Cosine | Jaccard | A | B | C |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E | dec_00003 | We rejected the migration to PostgreSQL. | dec_00004 | We will migrate the database to PostgreSQL next quarter. | not duplicate | 0.8010 | 0.3636 | accept | reject | accept |
| E | dec_00006 | Enable dark mode for the dashboard. | dec_00009 | Fix the dark mode contrast bug. | not duplicate | 0.7914 | 0.3333 | accept | reject | accept |
| E | dec_00007 | Dark mode will be enabled for the dashboard. | dec_00009 | Fix the dark mode contrast bug. | not duplicate | 0.7595 | 0.2727 | accept | accept | accept |
| E | dec_00008 | Disable dark mode for the dashboard. | dec_00009 | Fix the dark mode contrast bug. | not duplicate | 0.7882 | 0.3333 | accept | reject | accept |
| E | dec_00009 | Fix the dark mode contrast bug. | dec_00010 | We approved enabling dark mode on the dashboard. | not duplicate | 0.7656 | 0.2727 | accept | accept | accept |
| D | dec_00011 | Hire two backend engineers this quarter. | dec_00013 | Hire eight backend engineers this quarter. | not duplicate | 0.8752 | 0.7143 | accept | reject | accept |
| D | dec_00011 | Hire two backend engineers this quarter. | dec_00014 | Hire two frontend engineers this quarter. | not duplicate | 0.9572 | 0.7143 | accept | reject | accept |
| D | dec_00012 | Recruit 2 backend developers this quarter. | dec_00013 | Hire eight backend engineers this quarter. | not duplicate | 0.8239 | 0.3333 | accept | reject | accept |
| D | dec_00012 | Recruit 2 backend developers this quarter. | dec_00014 | Hire two frontend engineers this quarter. | not duplicate | 0.8868 | 0.2000 | accept | accept | accept |
| D | dec_00013 | Hire eight backend engineers this quarter. | dec_00014 | Hire two frontend engineers this quarter. | not duplicate | 0.8199 | 0.5000 | accept | reject | accept |
| D | dec_00013 | Hire eight backend engineers this quarter. | dec_00015 | Two backend engineering positions were approved for this quarter. | not duplicate | 0.7711 | 0.2500 | accept | accept | accept |
| D | dec_00014 | Hire two frontend engineers this quarter. | dec_00015 | Two backend engineering positions were approved for this quarter. | not duplicate | 0.8133 | 0.2500 | accept | accept | accept |

_Showing 12 of 111. Full list is in pair_scores.csv._
