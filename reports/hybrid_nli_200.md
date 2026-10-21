# Hybrid + NLI on the 200-decision dataset

## 1. Approach and dataset

Approach 2, Hybrid retrieval + detail guard + DeBERTa NLI (`find_duplicates_2.py`), NLI threshold 0.70.

Dataset A is `data/experiment_decision_dataset_200.json`: 200 synthetic decisions in 40 families of five. Slot 0 is the base decision, slot 1 is a paraphrase, slot 2 conflicts with it, slot 3 is related but different, and slot 4 is a second paraphrase. Genuine duplicate pairs are only (0, 1), (0, 4), and (1, 4). Three pairs in family 1, all involving “we will migrate next quarter,” are marked ambiguous and left out of the 397-pair score. Labels come from that family structure. They are not derived from model scores.

## 2. Algorithm configuration

- Script: `find_duplicates_2.py`
- Command flags: `--method hybrid --verifier nli --nli-threshold 0.70`
- Embeddings: `BAAI/bge-small-en-v1.5`
- Candidates: reciprocal-rank fusion of the 20 nearest embedding neighbors and the 20 BM25 neighbors
- A candidate is eligible for verification when cosine similarity is at least 0.75
- Detail guard: explicit number, period, identifier, month, weekday, office, or one-sided negation conflicts are rejected before the model
- Verifier: bidirectional entailment with `cross-encoder/nli-deberta-v3-base`. The pair is accepted only when the lower of the two directions is at least 0.70
- Verification budget: 2,000 pairs. Pairs past the budget would be marked uncertain and not accepted. This run did not exhaust the budget.
- Grouping: complete-link. Every pair inside a group must itself be accepted. Groups are not merged through a chain
- Ranking: the weakest cosine similarity inside the group
- `--top 10000` keeps every ranked group in the JSON. It does not change which pairs are accepted.

## 3. Dataset description

Dataset A is `data/experiment_decision_dataset_200.json`: 200 synthetic decisions in 40 families of five. Slot 0 is the base decision, slot 1 is a paraphrase, slot 2 conflicts with it, slot 3 is related but different, and slot 4 is a second paraphrase. Genuine duplicate pairs are only (0, 1), (0, 4), and (1, 4). Three pairs in family 1, all involving “we will migrate next quarter,” are marked ambiguous and left out of the 397-pair score. Labels come from that family structure. They are not derived from model scores.

## 4. Execution summary

- Decisions loaded: 200
- Retrieved candidate pairs: 2622
- Accepted duplicate pairs: 86
- NLI calls: 237
- Groups written: 41
- Execution time: 19.9 seconds
- Database reused: `data/test_200_v2.db` (no reingest)
- Output: `output/hybrid_nli_200.json`
- Pair file: `output/hybrid_nli_200_pairs.json`

## 5. Precision, recall, F1, TP, FP, FN

Pair-level metrics on the 397 labeled, non-ambiguous pairs. A pair is counted as accepted only when this approach directly accepts that pair. For Semantic + Jaccard that is a neighbor edge that passes cosine 0.75 and the Jaccard gate. For Hybrid + NLI that is `prediction: duplicate` in the pair file. Transitive group membership is not counted as a pair acceptance; it is reported in the group sections.

| Metric | Value |
|---|---|
| Evaluated pairs | 397 |
| Positive labels | 120 |
| Precision | 0.8953 |
| Recall | 0.6417 |
| F1 | 0.7476 |
| True positives | 77 |
| False positives | 9 |
| False negatives | 43 |
| True negatives | 268 |

## 6. Top 5 duplicate groups

1. score 0.9808, 2 variants
   - `dec_00001` We approved the migration to PostgreSQL.
   - `dec_00002` The migration to PostgreSQL was approved.
2. score 0.9561, 2 variants
   - `dec_00171` Pay invoice INV-14 for $2,000.
   - `dec_00175` Invoice INV-14 will be paid in the amount of $2,000.
3. score 0.9486, 2 variants
   - `dec_00121` Award contract CONT-45 to vendor VND-45 for $775k.
   - `dec_00122` Vendor VND-45 wins contract CONT-45 worth $775,000.
4. score 0.9392, 3 variants
   - `dec_00111` Set customer CUST-91 discount to 24% through August 2025.
   - `dec_00112` Customer CUST-91 receives a 24 percent discount until August 2025.
   - `dec_00115` Apply a 24% discount to CUST-91 through August 2025.
5. score 0.9334, 2 variants
   - `dec_00131` Buy domain DOM-3779.example for campaign CMP-3779.
   - `dec_00135` Campaign CMP-3779 will acquire domain DOM-3779.example.

## 7. Group completeness

16 of 40 paraphrase triples are complete: one predicted group contains every genuine member. Partial: 23. Missed entirely: 1. Split: 0.

## 8. Split groups

No labeled family was split across two predicted groups.

## 9. Missed genuine duplicates

- 1 (partial): `dec_00005` Increase PostgreSQL database storage next quarter.
- 3 (partial): `dec_00015` Two backend engineering positions were approved for this quarter.
- 4 (partial): `dec_00016` Raise subscription prices by 10% next quarter.
- 5 (partial): `dec_00025` The mobile app will be shipped on Friday.
- 7 (partial): `dec_00032` The hybrid policy permits 3 work-from-home days weekly.
- 9 (partial): `dec_00042` Audit log retention is set to three months.
- 10 (partial): `dec_00050` We decided to block public S3 bucket access.
- 12 (partial): `dec_00057` Do not renew our agreement with Acme.
- 14 (partial): `dec_00066` Require MFA for administrator accounts.
- 16 (partial): `dec_00077` Stop recruiting designers until further notice.
- 18 (partial): `dec_00087` P1 tickets must be resolved within 60 minutes.
- 20 (partial): `dec_00097` Close dormant support cases after two months.
- 21 (partial): `dec_00105` Wallet WAL-4026 must not exceed $52 per day.
- 22 (partial): `dec_00110` Hotfix HF-42 for BUG-42 goes to production tonight.
- 25 (partial): `dec_00125` Contract CONT-45 for $775k was awarded to VND-45.
- 27 (partial): `dec_00132` Purchase DOM-3779.example for marketing campaign CMP-3779.
- 29 (partial): `dec_00142` Move archived channel CH-10 history into VT-10.
- 30 (partial): `dec_00147` The proposal RFC-41 for FEAT-41 was declined.
- 31 (partial): `dec_00152` Deny traffic from 10.0.1.0/24 following INC-1.
- 32 (partial): `dec_00160` The KPI-313 target for SQ-313 is set at thirty.
- 33 (partial): `dec_00165` The new codename for PRJ-65 is onyx65.
- 34 (partial): `dec_00167` Send requests from REG-1 to compute cluster CLS-1.
- 35 (partial): `dec_00172` Authorize payment of two thousand dollars for INV-14.
- 36 (missed): `dec_00176` Set campaign CMP-3 budget to $10k.; `dec_00177` Campaign CMP-3 receives a ten-thousand-dollar budget.; `dec_00180` The budget for CMP-3 is $10,000.

## 10. Incorrect matches and false positives

3 predicted groups contain at least one labeled non-duplicate pair. Pair-level false positives: 9.

- "The mobile application goes live Friday." ↔ "Announce the mobile app on Friday."
- "Ban remote work three days per week." ↔ "Require three in-office days per week."
- "Production API credentials must rotate monthly." ↔ "Rotate production API keys every 90 days."
- "Production API credentials must rotate monthly." ↔ "Audit production API keys every 30 days."
- "Continue hiring for the design team." ↔ "Review the design team hiring plan."
- "Archive inactive support tickets after 60 days." ↔ "Tag inactive support tickets after 60 days."
- "Tag inactive support tickets after 60 days." ↔ "Inactive support tickets will be archived after sixty days."
- "Deploy hotfix HF-42 addressing BUG-42 this evening." ↔ "Document hotfix HF-42 for bug BUG-42 tonight."

## 11. Examples from the actual output

### Accepted genuine duplicates

- "We approved the migration to PostgreSQL." ↔ "The migration to PostgreSQL was approved."
- "Enable dark mode for the dashboard." ↔ "Dark mode will be enabled for the dashboard."
- "Enable dark mode for the dashboard." ↔ "We approved enabling dark mode on the dashboard."
- "Dark mode will be enabled for the dashboard." ↔ "We approved enabling dark mode on the dashboard."

### Genuine duplicates missed

- "We approved the migration to PostgreSQL." ↔ "Increase PostgreSQL database storage next quarter."
- "The migration to PostgreSQL was approved." ↔ "Increase PostgreSQL database storage next quarter."
- "Recruit 2 backend developers this quarter." ↔ "Two backend engineering positions were approved for this quarter."
- "Raise subscription prices by 10% next quarter." ↔ "A ten percent subscription price increase starts next quarter."
- "Raise subscription prices by 10% next quarter." ↔ "Increase subscription pricing ten percent next quarter."
- "The mobile application goes live Friday." ↔ "The mobile app will be shipped on Friday."

### Labeled or planted-family false matches

- "The mobile application goes live Friday." ↔ "Announce the mobile app on Friday."
- "Ban remote work three days per week." ↔ "Require three in-office days per week."
- "Production API credentials must rotate monthly." ↔ "Rotate production API keys every 90 days."
- "Production API credentials must rotate monthly." ↔ "Audit production API keys every 30 days."
- "Continue hiring for the design team." ↔ "Review the design team hiring plan."
- "Archive inactive support tickets after 60 days." ↔ "Tag inactive support tickets after 60 days."


## 12. Strengths

Requiring both directions of entailment, after the detail guard, leaves few labeled false positives. Related-but-different decisions are usually rejected even when the embedding score is high.

## 13. Limitations

Entailment is often strong in only one direction, so a genuine paraphrase is rejected. The detail guard treats some equivalent amounts as conflicts when one side is written as words (“ten percent”, “two thousand”). Complete-link grouping then leaves families partial even when some of their pairs were accepted. The 0.70 threshold was selected on this same labeled set, so this score is not a held-out estimate.

## 14. Key observations

On the same 397 pairs, F1 is 0.7476 with 9 false positives and 43 misses. 16 of 40 paraphrase triples are complete. Pair precision is high; group completeness is lower because a family is complete only when every genuine pair clears the verifier.
