# Semantic + Jaccard on the 200-decision dataset

## 1. Approach and dataset

Approach 1, Semantic + Jaccard (`find_duplicates_1.py`).

Dataset A is `data/experiment_decision_dataset_200.json`: 200 synthetic decisions in 40 families of five. Slot 0 is the base decision, slot 1 is a paraphrase, slot 2 conflicts with it, slot 3 is related but different, and slot 4 is a second paraphrase. Genuine duplicate pairs are only (0, 1), (0, 4), and (1, 4). Three pairs in family 1, all involving “we will migrate next quarter,” are marked ambiguous and left out of the 397-pair score. Labels come from that family structure. They are not derived from model scores.

## 2. Algorithm configuration

- Script: `find_duplicates_1.py`
- Embeddings: `BAAI/bge-small-en-v1.5`, cosine distance on the 50 nearest neighbors
- Accept a neighbor when cosine similarity is at least 0.75 and, unless the normalized texts are identical, Jaccard overlap is at most 0.32
- Grouping: union-find. A chain of accepted pairs becomes one group
- Ranking: `(1 + log(group size)) * mean cosine * (mean Jaccard distance squared)`
- `--top 10000` only keeps every ranked group in the JSON file. It does not change which pairs are accepted. The top 5 below are the first five groups in that ranking.

## 3. Dataset description

Dataset A is `data/experiment_decision_dataset_200.json`: 200 synthetic decisions in 40 families of five. Slot 0 is the base decision, slot 1 is a paraphrase, slot 2 conflicts with it, slot 3 is related but different, and slot 4 is a second paraphrase. Genuine duplicate pairs are only (0, 1), (0, 4), and (1, 4). Three pairs in family 1, all involving “we will migrate next quarter,” are marked ambiguous and left out of the 397-pair score. Labels come from that family structure. They are not derived from model scores.

## 4. Execution summary

- Decisions loaded: 200
- Retrieved neighbor pairs: 6584
- Accepted duplicate pairs: 136
- Groups written: 27
- Execution time: 0.8 seconds
- Database reused: `data/test_200_v2.db` (no reingest)
- Output: `output/semantic_jaccard_200.json`

## 5. Precision, recall, F1, TP, FP, FN

Pair-level metrics on the 397 labeled, non-ambiguous pairs. A pair is counted as accepted only when this approach directly accepts that pair. For Semantic + Jaccard that is a neighbor edge that passes cosine 0.75 and the Jaccard gate. For Hybrid + NLI that is `prediction: duplicate` in the pair file. Transitive group membership is not counted as a pair acceptance; it is reported in the group sections.

| Metric | Value |
|---|---|
| Evaluated pairs | 397 |
| Positive labels | 120 |
| Precision | 0.4393 |
| Recall | 0.3917 |
| F1 | 0.4141 |
| True positives | 47 |
| False positives | 60 |
| False negatives | 73 |
| True negatives | 217 |

## 6. Top 5 duplicate groups

1. score 1.8896, 15 variants
   - `dec_00026` Retire the legacy billing API by December.
   - `dec_00027` Sunset the old billing API before year-end.
   - `dec_00028` Keep the legacy billing API running through December.
   - `dec_00029` Document the legacy billing API by December.
   - `dec_00030` The legacy billing API will be retired by December.
   - `dec_00061` Switch customer billing from monthly to annual.
   - … 9 more variants
2. score 1.8867, 14 variants
   - `dec_00001` We approved the migration to PostgreSQL.
   - `dec_00002` The migration to PostgreSQL was approved.
   - `dec_00004` We will migrate the database to PostgreSQL next quarter.
   - `dec_00005` Increase PostgreSQL database storage next quarter.
   - `dec_00016` Raise subscription prices by 10% next quarter.
   - `dec_00017` A ten percent subscription price increase starts next quarter.
   - … 8 more variants
3. score 1.5812, 8 variants
   - `dec_00011` Hire two backend engineers this quarter.
   - `dec_00012` Recruit 2 backend developers this quarter.
   - `dec_00013` Hire eight backend engineers this quarter.
   - `dec_00014` Hire two frontend engineers this quarter.
   - `dec_00015` Two backend engineering positions were approved for this quarter.
   - `dec_00076` Approve a hiring freeze for the design team.
   - … 2 more variants
4. score 1.4932, 3 variants
   - `dec_00066` Require MFA for administrator accounts.
   - `dec_00067` Admin users must authenticate with a second factor.
   - `dec_00070` Multi-factor authentication is mandatory for admins.
5. score 1.4286, 9 variants
   - `dec_00036` Rotate production API keys every 30 days.
   - `dec_00037` Production API credentials must rotate monthly.
   - `dec_00038` Rotate production API keys every 90 days.
   - `dec_00039` Audit production API keys every 30 days.
   - `dec_00040` Production API keys will be rotated every thirty days.
   - `dec_00041` Keep audit logs for 90 days.
   - … 3 more variants

## 7. Group completeness

20 of 40 paraphrase triples are complete: one predicted group contains every genuine member. Partial: 12. Missed entirely: 8. Split: 0.

## 8. Split groups

No labeled family was split across two predicted groups.

## 9. Missed genuine duplicates

- 2 (partial): `dec_00006` Enable dark mode for the dashboard.
- 5 (partial): `dec_00021` Ship the mobile app on Friday.
- 10 (partial): `dec_00046` Block public access to S3 buckets.
- 11 (missed): `dec_00051` Close the Dhaka office on May 1.; `dec_00052` The Dhaka location will shut down May 1.; `dec_00055` The Dhaka office will be closed on May 1.
- 16 (partial): `dec_00080` The design team hiring freeze was approved.
- 18 (partial): `dec_00087` P1 tickets must be resolved within 60 minutes.
- 20 (partial): `dec_00100` Inactive support tickets will be archived after sixty days.
- 22 (partial): `dec_00106` Ship hotfix HF-42 for bug BUG-42 tonight.
- 23 (missed): `dec_00111` Set customer CUST-91 discount to 24% through August 2025.; `dec_00112` Customer CUST-91 receives a 24 percent discount until August 2025.; `dec_00115` Apply a 24% discount to CUST-91 through August 2025.
- 24 (missed): `dec_00116` Schedule maintenance WIN-760 on host HST-760 at 16:00 UTC.; `dec_00117` Maintenance window WIN-760 for HST-760 starts at 16:00 UTC.; `dec_00120` Host HST-760 maintenance WIN-760 is booked for 16:00 UTC.
- 25 (partial): `dec_00121` Award contract CONT-45 to vendor VND-45 for $775k.
- 26 (missed): `dec_00126` Grant role ROLE-8 to service account SA-8.; `dec_00127` Service account SA-8 is assigned ROLE-8.; `dec_00130` ROLE-8 will be granted to service account SA-8.
- 27 (missed): `dec_00131` Buy domain DOM-3779.example for campaign CMP-3779.; `dec_00132` Purchase DOM-3779.example for marketing campaign CMP-3779.; `dec_00135` Campaign CMP-3779 will acquire domain DOM-3779.example.
- 28 (missed): `dec_00136` Promote employee EMP-3813 to level L6.; `dec_00137` Employee EMP-3813 will advance to grade L6.; `dec_00140` EMP-3813 was approved for promotion to L6.
- 29 (missed): `dec_00141` Archive channel CH-10 to vault VT-10.; `dec_00142` Move archived channel CH-10 history into VT-10.; `dec_00145` Channel CH-10 will be archived in vault VT-10.
- 31 (partial): `dec_00152` Deny traffic from 10.0.1.0/24 following INC-1.
- 32 (partial): `dec_00156` Set KPI KPI-313 target to 30 for squad SQ-313.
- 33 (partial): `dec_00161` Rename project PRJ-65 to codename onyx65.
- 34 (missed): `dec_00166` Route region REG-1 traffic to cluster CLS-1.; `dec_00167` Send requests from REG-1 to compute cluster CLS-1.; `dec_00170` Traffic from REG-1 will be routed to CLS-1.
- 37 (partial): `dec_00185` Checkout-api will go to staging Tuesday.

## 10. Incorrect matches and false positives

23 predicted groups contain at least one labeled non-duplicate pair. Pair-level false positives: 60.

- "Dark mode will be enabled for the dashboard." ↔ "Fix the dark mode contrast bug."
- "Fix the dark mode contrast bug." ↔ "We approved enabling dark mode on the dashboard."
- "Recruit 2 backend developers this quarter." ↔ "Hire two frontend engineers this quarter."
- "Hire eight backend engineers this quarter." ↔ "Two backend engineering positions were approved for this quarter."
- "Hire two frontend engineers this quarter." ↔ "Two backend engineering positions were approved for this quarter."
- "A ten percent subscription price increase starts next quarter." ↔ "Do not raise subscription prices by 10% next quarter."
- "Do not raise subscription prices by 10% next quarter." ↔ "Increase subscription pricing ten percent next quarter."
- "Offer a 10% subscription discount next quarter." ↔ "Increase subscription pricing ten percent next quarter."

## 11. Examples from the actual output

### Accepted genuine duplicates

- "Recruit 2 backend developers this quarter." ↔ "Two backend engineering positions were approved for this quarter."
- "Raise subscription prices by 10% next quarter." ↔ "A ten percent subscription price increase starts next quarter."
- "Raise subscription prices by 10% next quarter." ↔ "Increase subscription pricing ten percent next quarter."
- "The mobile application goes live Friday." ↔ "The mobile app will be shipped on Friday."

### Genuine duplicates missed

- "We approved the migration to PostgreSQL." ↔ "The migration to PostgreSQL was approved."
- "We approved the migration to PostgreSQL." ↔ "Increase PostgreSQL database storage next quarter."
- "The migration to PostgreSQL was approved." ↔ "Increase PostgreSQL database storage next quarter."
- "Enable dark mode for the dashboard." ↔ "Dark mode will be enabled for the dashboard."
- "Enable dark mode for the dashboard." ↔ "We approved enabling dark mode on the dashboard."
- "Dark mode will be enabled for the dashboard." ↔ "We approved enabling dark mode on the dashboard."

### Labeled or planted-family false matches

- "Dark mode will be enabled for the dashboard." ↔ "Fix the dark mode contrast bug."
- "Fix the dark mode contrast bug." ↔ "We approved enabling dark mode on the dashboard."
- "Recruit 2 backend developers this quarter." ↔ "Hire two frontend engineers this quarter."
- "Hire eight backend engineers this quarter." ↔ "Two backend engineering positions were approved for this quarter."
- "Hire two frontend engineers this quarter." ↔ "Two backend engineering positions were approved for this quarter."
- "A ten percent subscription price increase starts next quarter." ↔ "Do not raise subscription prices by 10% next quarter."


## 12. Strengths

Union-find keeps a paraphrase family together when the pairs that do pass the gates form a chain. On this labeled set the Jaccard gate removes many same-template non-duplicates that cosine alone would accept.

## 13. Limitations

The Jaccard cutoff also rejects genuine paraphrases that happen to share wording. Neighbor search only inspects 50 neighbors, so a true pair outside that list is missed. Union-find can still place two non-equivalent decisions in one group when both match a third decision.

## 14. Key observations

On 397 labeled pairs, F1 is 0.4141 with 60 false positives and 73 misses. 20 of 40 paraphrase triples are complete.
