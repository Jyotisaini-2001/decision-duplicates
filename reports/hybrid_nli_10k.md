# Hybrid + NLI on the 10,000-decision dataset

## 1. Approach and dataset

Approach 2, Hybrid retrieval + detail guard + DeBERTa NLI (`find_duplicates_2.py`), NLI threshold 0.70.

Dataset B is `data/experiment_decision_dataset_10k.json`: 10,000 decisions. Six planted paraphrase families are marked with `planted_group` (39 decisions, every pair inside a family treated as a genuine duplicate). The remaining decisions are template fillers. Filler pairs were not labeled one by one, so they are not treated as confirmed true negatives. Overall precision, recall, and F1 on all 10,000 decisions are therefore not reported.

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

Dataset B is `data/experiment_decision_dataset_10k.json`: 10,000 decisions. Six planted paraphrase families are marked with `planted_group` (39 decisions, every pair inside a family treated as a genuine duplicate). The remaining decisions are template fillers. Filler pairs were not labeled one by one, so they are not treated as confirmed true negatives. Overall precision, recall, and F1 on all 10,000 decisions are therefore not reported.

## 4. Execution summary

- Decisions loaded: 10,000
- Retrieved candidate pairs: 159144
- Accepted duplicate pairs: 55
- NLI calls: 89
- Groups written: 8
- Execution time: 33.8 seconds
- Database reused: `data/decisions_vec.db` (no reingest)
- Output: `output/hybrid_nli_10k.json`
- Pair file: `output/hybrid_nli_10k_pairs.json`

## 5. Precision, recall, F1, TP, FP, FN

Overall precision, recall, and F1: **N/A — insufficient ground truth.** Only the planted families are labeled. Filler-to-filler pairs are unlabeled, so they are not counted as true negatives or false positives.

Planted-family pair recovery uses the 110 pairs inside the six planted families. Recall here is the share of those pairs that the approach directly accepted. Precision is not computed, because accepted pairs among unlabeled fillers have no label.

| Metric | Value |
|---|---|
| Precision | N/A — insufficient ground truth |
| Recall on planted pairs | 0.5000 |
| F1 | N/A — insufficient ground truth |
| True positives (planted pairs accepted) | 55 |
| False negatives (planted pairs missed) | 55 |
| False positives | N/A — insufficient ground truth |
| Accepted pairs inside one planted family | 55 |
| Accepted pairs across two planted families | 0 |
| Accepted pairs linking a planted decision to an unlabeled decision | 0 |
| Accepted pairs among unlabeled decisions only | 0 |

## 6. Top 5 duplicate groups

1. score 0.8859, 4 variants
   - `dec_00022` Next quarter we increase pricing 10%.
   - `dec_00021` Raise prices by 10% starting next quarter.
   - `dec_00024` Pricing goes up 10% beginning next quarter.
   - `dec_00025` Decision: 10% price hike effective next quarter.
2. score 0.847, 2 variants
   - `dec_00015` Remote work is allowed three days per week.
   - `dec_00017` Employees may work from home three days weekly.
3. score 0.824, 5 variants
   - `dec_00012` Primary datastore will be Postgres.
   - `dec_00013` Adopt PostgreSQL as the database of record.
   - `dec_00010` We're standardizing on PostgreSQL for storage.
   - `dec_00011` Decision: PostgreSQL for the main DB.
   - `dec_00008` Use PostgreSQL as our primary database.
4. score 0.8166, 2 variants
   - `dec_00028` Legacy API sunset date: December 31.
   - `dec_00027` Deprecate the legacy API by end of year.
5. score 0.8073, 3 variants
   - `dec_00037` Recruit two backend developers this quarter.
   - `dec_00034` Hire two backend engineers this quarter.
   - `dec_00035` We're opening two backend eng roles this Q.

## 7. Group completeness

0 of 6 planted families are complete: one predicted group contains every genuine member. Partial: 4. Missed entirely: 0. Split: 2.

## 8. Split groups

- Friday ship: kept 7 of 8
- Legacy API: kept 6 of 7

## 9. Missed genuine duplicates

- Friday ship (split): `dec_00007` Ship date locked to Friday.
- Postgres (partial): `dec_00014` We picked Postgres over MySQL for the database.; `dec_00009` Postgres is the chosen database.
- WFH / remote (partial): `dec_00019` WFH allowance set at 3 days/week.; `dec_00016` Hybrid policy: 3 days WFH each week.; `dec_00020` Three days remote, two in office — that's the policy.; `dec_00018` Approved: three remote days per week.
- 10% price (partial): `dec_00023` Approved a 10 percent price increase for Q+1.; `dec_00026` We'll bump prices ten percent next quarter.
- Legacy API (split): `dec_00033` Legacy endpoints will be removed by EOY.
- Hire backend (partial): `dec_00038` Decision: add two backend engineers ASAP this quarter.; `dec_00036` Approved headcount: 2 backend engineers.

## 10. Incorrect matches and false positives

Confirmed cross-family merges: 0 groups, 0 accepted pairs.

Groups that place a planted decision with an unlabeled decision: 0. These are reported separately. The filler side has no pair label, so they are not counted as confirmed false positives.

Groups made only of unlabeled decisions: 0. N/A as false positives — those pairs were not labeled.

Groups that stay inside a single planted family: 8.

## 11. Examples from the actual output

### Accepted genuine duplicates

- "Agreed: product ships Friday." ↔ "We decided to ship the product on Friday."
- "Agreed: product ships Friday." ↔ "Release date is set for Friday."
- "Agreed: product ships Friday." ↔ "Decision: Friday launch for the product."
- "Agreed: product ships Friday." ↔ "We'll put the product out on Friday."

### Genuine duplicates missed

- "Agreed: product ships Friday." ↔ "Ship date locked to Friday."
- "Agreed: product ships Friday." ↔ "Friday is the go-live day for the product."
- "Agreed: product ships Friday." ↔ "We're launching this Friday."
- "Ship date locked to Friday." ↔ "Friday is the go-live day for the product."
- "Ship date locked to Friday." ↔ "We decided to ship the product on Friday."
- "Ship date locked to Friday." ↔ "We're launching this Friday."

### Labeled or planted-family false matches

See section 10. Unlabeled filler pairs are not listed as false positives.


## 12. Strengths

The identifier and number guard rejects almost all filler candidates before the NLI model runs, so accepted pairs stay inside the planted families. Complete-link grouping does not merge two decisions unless that exact pair was verified.

## 13. Limitations

Pair-level precision against every decision cannot be computed. Strict complete-link grouping and one-direction entailment failures split families and drop members that Semantic + Jaccard kept. BM25’s contribution is not isolated in this run: candidates still have to pass cosine 0.75, so keyword retrieval cannot admit a pair the embedding score rejects.

## 14. Key observations

Planted-pair recall is 0.5000 (55 of 110). 0 of 6 planted families are complete. Accepted pairs that stay inside one planted family: 55. Accepted pairs that only involve unlabeled decisions: 0.
