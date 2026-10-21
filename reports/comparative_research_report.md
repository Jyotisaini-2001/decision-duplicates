# Comparative Research Report: Semantic Duplicate Decision Detection

**Semantic + Jaccard vs. Hybrid Retrieval + NLI**

## 1. Executive Summary

This research compares two approaches for detecting semantically duplicate decisions in JSON datasets.

The objective is to identify the **top five meaningful duplicate-decision groups**, while minimizing incorrect matches and recovering genuine paraphrases.

Two approaches were evaluated:

1. **Semantic + Jaccard:** Semantic similarity, lexical filtering, Union-Find grouping, and group ranking.
2. **Hybrid + NLI:** Semantic and BM25 retrieval, critical-detail validation, bidirectional NLI verification, and complete-link grouping.

Experiments were conducted on datasets containing 200 and 10,000 synthetic decisions.

### Key Results

- On the 200-decision dataset, Hybrid + NLI achieved **89.53% precision and 0.7476 F1**, compared with **43.93% precision and 0.4141 F1** for Semantic + Jaccard.
- On the 10,000-decision dataset, Semantic + Jaccard recovered **84 of 110 planted duplicate pairs** and formed **5 of 6 complete families**. Hybrid + NLI recovered **55 pairs** and formed **0 of 6 complete families**.
- Semantic + Jaccard executed faster on both datasets.

**Main finding:** Higher pair-level accuracy does not necessarily produce more complete duplicate groups.

Neither approach is considered production-ready. This project is maintained as a comparative research study.

---

## 2. Research Problem

Duplicate decisions may express the same meaning using different wording.

**Example — genuine duplicates:**

- "Raise prices by 10% starting next quarter."
- "Pricing goes up 10% beginning next quarter."
- "Decision: 10% price hike effective next quarter."

However, decisions with similar wording may have different meanings.

**Example — non-duplicates:**

- "Hire two backend engineers this quarter."
- "Hire two frontend engineers this quarter."
- "Hire eight backend engineers this quarter."

The challenge is to identify meaningful equivalence rather than simple textual similarity.

### Research Questions

1. Which approach more accurately identifies genuine duplicate pairs?
2. Which approach produces more complete duplicate groups?
3. How do the approaches behave on datasets of different sizes?
4. What are the main causes of false matches and missed duplicates?
5. How useful are the top-five ranked groups produced by each approach?

---

## 3. Approaches Investigated

### 3.1 Approach 1 — Semantic + Jaccard

**Implementation:** `find_duplicates_1.py`

This approach combines semantic embeddings with lexical-overlap filtering.

**Workflow:**

1. Load JSON decisions and generate or reuse embeddings using `BAAI/bge-small-en-v1.5`.
2. Retrieve the 50 nearest semantic neighbors.
3. Filter candidates using cosine similarity ≥ 0.75.
4. Accept pairs when normalized texts are identical or Jaccard word overlap ≤ 0.32.
5. Merge accepted pairs using Union-Find.
6. Rank groups using group size, mean cosine similarity, and wording diversity.
7. Export the top-ranked groups.

**Ranking formula:**

`(1 + log(group size)) × mean cosine × (mean Jaccard distance²)`

The low-overlap Jaccard rule is an experimental heuristic intended to filter similar-looking template decisions.

**Strength:** Recovers larger connected groups of paraphrases.

**Limitation:** Union-Find can merge decisions through intermediate connections, even when some members are not equivalent.

**Figure 1: Semantic + Jaccard Workflow**

Insert: `diagrams/semantic_jaccard_workflow.png`

### 3.2 Approach 2 — Hybrid Retrieval + NLI

**Implementation:** `find_duplicates_2.py`

**Configuration:** `--method hybrid --verifier nli --nli-threshold 0.70`

This approach combines hybrid candidate retrieval with stricter semantic verification.

**Workflow:**

1. Retrieve 20 embedding neighbors and 20 BM25 keyword neighbors.
2. Combine candidates using reciprocal-rank fusion.
3. Apply cosine similarity threshold ≥ 0.75.
4. Reject candidates containing detected conflicts in critical details.
5. Verify remaining pairs using bidirectional DeBERTa NLI.
6. Accept pairs only when the weaker entailment direction scores ≥ 0.70.
7. Create complete-link groups, requiring every internal pair to be accepted.
8. Rank groups using the weakest internal cosine similarity.
9. Export the top-ranked groups.

**Strength:** Reduces many incorrect matches involving similar but different decisions.

**Limitation:** Strict verification and grouping can reject genuine paraphrases or split complete families.

**Figure 2: Hybrid + NLI Workflow**

Insert: `diagrams/hybrid_nli_workflow.png`

**Important:** The approaches differ in retrieval, verification, grouping, and ranking. The results therefore compare complete pipelines, not Jaccard versus NLI in isolation.

---

## 4. Experimental Setup

### 4.1 Dataset A — 200 Decisions

- 200 synthetic decisions.
- 40 families containing five decisions each.
- 397 labeled, non-ambiguous evaluation pairs.
- 120 genuine duplicate pairs.
- 277 non-duplicate pairs.
- Three ambiguous pairs excluded.

This dataset supports pair-level precision, recall, F1, and error analysis.

### 4.2 Dataset B — 10,000 Decisions

- 10,000 synthetic decisions.
- Six planted paraphrase families.
- 39 planted decisions.
- 110 known genuine duplicate pairs.
- Remaining decisions consist of template-generated filler records.

This dataset supports planted-family recovery analysis.

Overall precision and F1 are not calculated because filler-to-filler pairs are not exhaustively labeled.

### 4.3 Evaluation Metrics

**Pair-level evaluation:**

- Precision
- Recall
- F1 score
- True positives
- False positives
- False negatives

**Group-level evaluation:**

- Complete duplicate families
- Partial families
- Split families
- Entirely missed families
- Incorrectly merged decisions
- Top-five ranked groups

**Performance evaluation:**

- Candidate pairs retrieved
- Accepted duplicate pairs
- Groups generated
- Execution time

A pair counts as accepted only when the algorithm directly accepts it. Indirect group membership is evaluated separately.

---

## 5. Experimental Results

### 5.1 Comparison on 200 Decisions

| Metric | Semantic + Jaccard | Hybrid + NLI |
|---|---:|---:|
| Precision | 43.93% | 89.53% |
| Recall | 39.17% | 64.17% |
| F1 score | 0.4141 | 0.7476 |
| True positives | 47 | 77 |
| False positives | 60 | 9 |
| False negatives | 73 | 43 |
| True negatives | 217 | 268 |
| Complete families | 20/40 | 16/40 |
| Partial families | 12 | 23 |
| Split families | 0 | 0 |
| Entirely missed families | 8 | 1 |
| Groups containing labeled non-duplicates | 23 | 3 |
| Execution time | 0.8 seconds | 19.9 seconds |

### Observations

Hybrid + NLI achieved higher pair-level precision, recall, and F1.

It reduced labeled false positives from 60 to 9.

Semantic + Jaccard produced more complete paraphrase families, but its grouping also included incorrect matches.

Therefore, more complete families do not necessarily mean more accurate groups.

### 5.2 Comparison on 10,000 Decisions

| Metric | Semantic + Jaccard | Hybrid + NLI |
|---|---:|---:|
| Overall precision | N/A | N/A |
| Overall F1 | N/A | N/A |
| Planted-pair recall | 76.36% | 50.00% |
| Planted pairs accepted | 84/110 | 55/110 |
| Planted pairs missed | 26 | 55 |
| Complete families | 5/6 | 0/6 |
| Partial families | 1 | 4 |
| Split families | 0 | 2 |
| Entirely missed families | 0 | 0 |
| Accepted cross-family pairs | 0 | 0 |
| Accepted filler-involving pairs | 0 | 0 |
| Groups generated | 6 | 8 |
| Execution time | 4.1 seconds | 33.8 seconds |

### Observations

Semantic + Jaccard recovered more planted duplicate pairs and produced more complete families.

Hybrid + NLI recovered genuine duplicates but frequently formed smaller, incomplete groups.

Neither approach accepted filler-involving pairs in this experiment.

These results apply to the tested synthetic dataset and do not establish overall precision on real-world decisions.

### 5.3 Top Five Groups — 10,000 Decisions

| Rank | Semantic + Jaccard | Variants | Hybrid + NLI | Variants |
|---|---|---:|---|---:|
| 1 | PostgreSQL database | 7 | 10% price increase | 4 |
| 2 | Friday ship date | 8 | Remote-work policy | 2 |
| 3 | 10% price increase | 6 | PostgreSQL database | 5 |
| 4 | Remote-work policy | 6 | Legacy API retirement | 2 |
| 5 | Legacy API retirement | 6 | Backend hiring | 3 |

The ranking scores are not directly comparable because each approach uses a different scoring formula.

**Key observation:** Semantic + Jaccard surfaces more complete planted families in its top five, whereas Hybrid + NLI returns smaller verified subsets.

---

## 6. Error Analysis

### 6.1 False Matches

Semantic + Jaccard incorrectly accepted pairs such as:

- "Recruit 2 backend developers this quarter."
- "Hire two frontend engineers this quarter."

It also incorrectly accepted conflicting decisions about increasing and not increasing subscription prices.

Hybrid + NLI reduced these errors but still accepted some incorrect matches.

Example:

- "Production API credentials must rotate monthly."
- "Audit production API keys every 30 days."

These statements concern the same subject but describe different actions.

### 6.2 Missed Genuine Duplicates

Semantic + Jaccard can reject genuine paraphrases when they share too many words.

Hybrid + NLI can reject genuine paraphrases when:

- One entailment direction falls below the threshold.
- Equivalent numerical expressions are interpreted differently.
- A required pair does not pass verification, preventing complete-link grouping.

### 6.3 Grouping Differences

Semantic + Jaccard uses Union-Find.

If A matches B and B matches C, all three decisions can belong to one group even when A and C were not directly verified.

Hybrid + NLI uses complete-link grouping.

Every pair within a group must be accepted.

This reduces transitive merging but can fragment genuine families.

Because the approaches also use different verification methods, the effect of grouping alone has not been measured independently.

---

## 7. Key Research Findings

### Finding 1 — Pair accuracy and group completeness are different

Hybrid + NLI performs better on labeled pair classification.

Semantic + Jaccard produces more complete planted families on the larger dataset.

Both dimensions must be evaluated separately.

### Finding 2 — Semantic similarity does not guarantee equivalence

Two decisions can concern the same topic while expressing different actions, quantities, dates, or conditions.

### Finding 3 — Strict verification introduces a recall trade-off

Hybrid + NLI rejects many incorrect matches but can also reject valid paraphrases.

### Finding 4 — Grouping strategy influences the final result

Union-Find favors connected groups, while complete-link favors stricter internal consistency.

Neither strategy guarantees correct grouping.

### Finding 5 — Computational cost differs

Semantic + Jaccard completed both experimental runs faster than Hybrid + NLI.

These timings describe the tested runs rather than general scaling guarantees.

### Finding 6 — Top-five quality needs separate validation

The current experiments provide ranked-group examples and family-recovery measurements.

They do not establish independently validated top-five precision or usefulness on real-world decision records.

---

## 8. Research Limitations

1. Both datasets are synthetic.
2. The 10,000-decision dataset lacks exhaustive negative labels.
3. Some 200-decision ground-truth labels may be questionable.
4. The NLI threshold was selected using the same labeled benchmark, so the resulting score is not an independent held-out estimate.
5. The approaches differ in multiple components, preventing isolation of individual algorithm contributions.
6. No independently annotated real-world dataset has been evaluated.
7. Top-five group quality has not been assessed through independent human judgments.

For example, the 200-decision labels treat increasing PostgreSQL storage and approving PostgreSQL migration as paraphrases in one family, although these may represent different decisions.

This should be reviewed before treating the benchmark scores as definitive.

---

## 9. Future Research

The following are proposed investigations, not implemented changes:

1. Create a held-out evaluation dataset.
2. Review and correct questionable labels.
3. Evaluate real-world decision records.
4. Measure top-five group purity, completeness, and usefulness.
5. Compare Union-Find and complete-link using identical accepted pairs.
6. Evaluate the independent contributions of BM25, detail guards, and NLI.
7. Improve normalization of numerical expressions, dates, and identifiers.

---

## 10. Conclusion

This research compared two approaches for identifying duplicate-decision groups.

**Hybrid + NLI demonstrated stronger pair-level classification performance** on the labeled 200-decision benchmark, achieving 89.53% precision and 0.7476 F1.

**Semantic + Jaccard demonstrated stronger planted-family recovery** on the 10,000-decision dataset, recovering 84 of 110 genuine pairs and producing five complete families.

The experiments highlight a trade-off between strict pairwise verification and complete group recovery.

No single approach consistently outperformed the other across all objectives.

The research therefore does not establish a final production algorithm.

Instead, it provides a reproducible experimental baseline for understanding duplicate-detection behavior, identifying failure modes, and guiding future evaluation.

---

## Appendix A — Supporting Experimental Reports

Detailed results are available in:

- [Semantic + Jaccard — 200 Decisions](semantic_jaccard_200.md)
- [Semantic + Jaccard — 10,000 Decisions](semantic_jaccard_10k.md)
- [Hybrid + NLI — 200 Decisions](hybrid_nli_200.md)
- [Hybrid + NLI — 10,000 Decisions](hybrid_nli_10k.md)

## Appendix B — Reproduction Commands

```bash
python find_duplicates_1.py --input data/experiment_decision_dataset_200.json --db data/test_200_v2.db --output output/semantic_jaccard_200.json --top 10000

python find_duplicates_1.py --input data/experiment_decision_dataset_10k.json --db data/decisions_vec.db --output output/semantic_jaccard_10k.json --top 10000

python find_duplicates_2.py --input data/experiment_decision_dataset_200.json --db data/test_200_v2.db --method hybrid --verifier nli --nli-threshold 0.70 --top 10000 --output output/hybrid_nli_200.json

python find_duplicates_2.py --input data/experiment_decision_dataset_10k.json --db data/decisions_vec.db --method hybrid --verifier nli --nli-threshold 0.70 --top 10000 --output output/hybrid_nli_10k.json

python evaluation/evaluate.py
```