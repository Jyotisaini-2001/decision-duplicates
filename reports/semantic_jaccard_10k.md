# Semantic + Jaccard on the 10,000-decision dataset

## 1. Approach and dataset

Approach 1, Semantic + Jaccard (`find_duplicates_1.py`).

Dataset B is `data/experiment_decision_dataset_10k.json`: 10,000 decisions. Six planted paraphrase families are marked with `planted_group` (39 decisions, every pair inside a family treated as a genuine duplicate). The remaining decisions are template fillers. Filler pairs were not labeled one by one, so they are not treated as confirmed true negatives. Overall precision, recall, and F1 on all 10,000 decisions are therefore not reported.

## 2. Algorithm configuration

- Script: `find_duplicates_1.py`
- Embeddings: `BAAI/bge-small-en-v1.5`, cosine distance on the 50 nearest neighbors
- Accept a neighbor when cosine similarity is at least 0.75 and, unless the normalized texts are identical, Jaccard overlap is at most 0.32
- Grouping: union-find. A chain of accepted pairs becomes one group
- Ranking: `(1 + log(group size)) * mean cosine * (mean Jaccard distance squared)`
- `--top 10000` only keeps every ranked group in the JSON file. It does not change which pairs are accepted. The top 5 below are the first five groups in that ranking.

## 3. Dataset description

Dataset B is `data/experiment_decision_dataset_10k.json`: 10,000 decisions. Six planted paraphrase families are marked with `planted_group` (39 decisions, every pair inside a family treated as a genuine duplicate). The remaining decisions are template fillers. Filler pairs were not labeled one by one, so they are not treated as confirmed true negatives. Overall precision, recall, and F1 on all 10,000 decisions are therefore not reported.

## 4. Execution summary

- Decisions loaded: 10,000
- Retrieved neighbor pairs: 341724
- Accepted duplicate pairs: 84
- Groups written: 6
- Execution time: 4.1 seconds
- Database reused: `data/decisions_vec.db` (no reingest)
- Output: `output/semantic_jaccard_10k.json`

## 5. Precision, recall, F1, TP, FP, FN

Overall precision, recall, and F1: **N/A — insufficient ground truth.** Only the planted families are labeled. Filler-to-filler pairs are unlabeled, so they are not counted as true negatives or false positives.

Planted-family pair recovery uses the 110 pairs inside the six planted families. Recall here is the share of those pairs that the approach directly accepted. Precision is not computed, because accepted pairs among unlabeled fillers have no label.

| Metric | Value |
|---|---|
| Precision | N/A — insufficient ground truth |
| Recall on planted pairs | 0.7636 |
| F1 | N/A — insufficient ground truth |
| True positives (planted pairs accepted) | 84 |
| False negatives (planted pairs missed) | 26 |
| False positives | N/A — insufficient ground truth |
| Accepted pairs inside one planted family | 84 |
| Accepted pairs across two planted families | 0 |
| Accepted pairs linking a planted decision to an unlabeled decision | 0 |
| Accepted pairs among unlabeled decisions only | 0 |

## 6. Top 5 duplicate groups

1. score 1.9663, 7 variants
   - `dec_00012` Primary datastore will be Postgres.
   - `dec_00013` Adopt PostgreSQL as the database of record.
   - `dec_00010` We're standardizing on PostgreSQL for storage.
   - `dec_00014` We picked Postgres over MySQL for the database.
   - `dec_00011` Decision: PostgreSQL for the main DB.
   - `dec_00008` Use PostgreSQL as our primary database.
   - … 1 more variants
2. score 1.6338, 8 variants
   - `dec_00004` Agreed: product ships Friday.
   - `dec_00007` Ship date locked to Friday.
   - `dec_00003` Friday is the go-live day for the product.
   - `dec_00000` We decided to ship the product on Friday.
   - `dec_00002` We're launching this Friday.
   - `dec_00001` Release date is set for Friday.
   - … 2 more variants
3. score 1.5195, 6 variants
   - `dec_00023` Approved a 10 percent price increase for Q+1.
   - `dec_00026` We'll bump prices ten percent next quarter.
   - `dec_00022` Next quarter we increase pricing 10%.
   - `dec_00021` Raise prices by 10% starting next quarter.
   - `dec_00024` Pricing goes up 10% beginning next quarter.
   - `dec_00025` Decision: 10% price hike effective next quarter.
4. score 1.4152, 6 variants
   - `dec_00019` WFH allowance set at 3 days/week.
   - `dec_00015` Remote work is allowed three days per week.
   - `dec_00016` Hybrid policy: 3 days WFH each week.
   - `dec_00017` Employees may work from home three days weekly.
   - `dec_00020` Three days remote, two in office — that's the policy.
   - `dec_00018` Approved: three remote days per week.
5. score 1.3361, 6 variants
   - `dec_00032` Phase out the outdated API by December.
   - `dec_00029` We're retiring the old API before year-end.
   - `dec_00030` End-of-year deadline to shut down the legacy API.
   - `dec_00031` Decision: legacy API gone by year end.
   - `dec_00028` Legacy API sunset date: December 31.
   - `dec_00027` Deprecate the legacy API by end of year.

## 7. Group completeness

5 of 6 planted families are complete: one predicted group contains every genuine member. Partial: 1. Missed entirely: 0. Split: 0.

## 8. Split groups

No labeled family was split across two predicted groups.

## 9. Missed genuine duplicates

- Legacy API (partial): `dec_00033` Legacy endpoints will be removed by EOY.

## 10. Incorrect matches and false positives

Confirmed cross-family merges: 0 groups, 0 accepted pairs.

Groups that place a planted decision with an unlabeled decision: 0. These are reported separately. The filler side has no pair label, so they are not counted as confirmed false positives.

Groups made only of unlabeled decisions: 0. N/A as false positives — those pairs were not labeled.

Groups that stay inside a single planted family: 6.

## 11. Examples from the actual output

### Accepted genuine duplicates

- "Agreed: product ships Friday." ↔ "Ship date locked to Friday."
- "Agreed: product ships Friday." ↔ "Friday is the go-live day for the product."
- "Agreed: product ships Friday." ↔ "We decided to ship the product on Friday."
- "Agreed: product ships Friday." ↔ "Release date is set for Friday."

### Genuine duplicates missed

- "Agreed: product ships Friday." ↔ "We're launching this Friday."
- "Ship date locked to Friday." ↔ "Friday is the go-live day for the product."
- "Ship date locked to Friday." ↔ "We're launching this Friday."
- "Ship date locked to Friday." ↔ "Decision: Friday launch for the product."
- "Ship date locked to Friday." ↔ "We'll put the product out on Friday."
- "Friday is the go-live day for the product." ↔ "Decision: Friday launch for the product."

### Labeled or planted-family false matches

See section 10. Unlabeled filler pairs are not listed as false positives.


## 12. Strengths

The lexical gate stops same-shape filler decisions, which share many words but are different decisions, from becoming the top groups. Planted paraphrase families that pass the gate are then pulled into one group.

## 13. Limitations

Pair-level precision against every decision in the file cannot be computed. Filler pairs have no labels. Families whose paraphrases share too many words can lose members. The ranking score rewards size and wording diversity, which is a preference, not a measure of correctness.

## 14. Key observations

Planted-pair recall is 0.7636 (84 of 110). 5 of 6 planted families are complete. Unlabeled accepted pairs: 0.
