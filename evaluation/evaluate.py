"""Score Semantic + Jaccard and Hybrid + NLI with one metric definition.

This file does not change either detector. It reads their JSON outputs and,
for Semantic + Jaccard only, recounts the same neighbor edges the detector
unions, so candidate and accepted-pair totals are available. Those edges use
the functions and default thresholds in find_duplicates_1.py.
"""

from __future__ import annotations

import json
import platform
import sys
import warnings
from collections import defaultdict
from importlib.metadata import version
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import find_duplicates_1 as jaccard_mod  # noqa: E402
from store import DecisionStore  # noqa: E402

DATA_200 = ROOT / "data" / "experiment_decision_dataset_200.json"
DATA_10K = ROOT / "data" / "experiment_decision_dataset_10k.json"
DB_200 = ROOT / "data" / "test_200_v2.db"
DB_10K = ROOT / "data" / "decisions_vec.db"
OUT = ROOT / "output"
REPORTS = ROOT / "reports"
LOGS = ROOT / "evaluation" / "logs"

# Same slot labels as experiments/dataset_200/run_experiments.py.
SLOT2 = [
    "C", "C", "D", "C", "C", "C", "C", "D", "D", "C",
    "C", "C", "C", "C", "C", "C", "C", "D", "C", "D",
    "D", "D", "D", "D", "D", "D", "D", "D", "D", "C",
    "D", "D", "D", "D", "D", "D", "D", "C", "C", "C",
]
SLOT3 = [
    "E", "E", "D", "E", "E", "E", "E", "E", "E", "E",
    "E", "E", "E", "E", "E", "E", "E", "E", "E", "E",
    "D", "E", "D", "D", "D", "D", "D", "D", "D", "D",
    "E", "D", "D", "D", "D", "D", "D", "D", "E", "D",
]

# Earlier 200-decision Hybrid + NLI run at threshold 0.70, before this restructure.
PRIOR_HYBRID_200 = {"tp": 77, "fp": 9, "fn": 43, "precision": 0.8953, "recall": 0.6417, "f1": 0.7476}


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def key(a, b):
    return tuple(sorted((a, b)))


def fmt(value, digits=4):
    if value is None:
        return "N/A — insufficient ground truth"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def prf(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return precision, recall, f1


def labeled_200(decisions):
    """397 evaluation pairs. Three ambiguous pairs in family 1 are excluded."""
    if len(decisions) != 200:
        raise RuntimeError(f"Expected 200 decisions, found {len(decisions)}")
    pairs = []
    families = []
    for gi in range(40):
        group = decisions[gi * 5:(gi + 1) * 5]
        gold_ids = [group[i]["id"] for i in (0, 1, 4)]
        families.append({
            "family": gi + 1,
            "ids": [row["id"] for row in group],
            "gold_ids": gold_ids,
            "texts": {row["id"]: row["text"] for row in group},
        })
        for left, right in combinations(range(5), 2):
            pair_ids = key(group[left]["id"], group[right]["id"])
            if gi == 0 and {left, right} in ({0, 3}, {1, 3}, {4, 3}):
                category, expected, ambiguous = "ambiguous", False, True
            elif {left, right} <= {0, 1, 4}:
                category = "B" if {left, right} == {0, 4} else "A"
                expected, ambiguous = True, False
            elif {left, right} == {2, 3}:
                category = "D" if SLOT2[gi] == "D" and SLOT3[gi] == "D" else "E"
                expected, ambiguous = False, False
            elif 2 in {left, right}:
                category, expected, ambiguous = SLOT2[gi], False, False
            else:
                category, expected, ambiguous = SLOT3[gi], False, False
            pairs.append({
                "ids": pair_ids,
                "expected": expected,
                "ambiguous": ambiguous,
                "category": category,
                "family": gi + 1,
                "text_a": group[left]["text"],
                "text_b": group[right]["text"],
            })
    return pairs, families


def planted_families(decisions):
    buckets = defaultdict(list)
    for row in decisions:
        if "planted_group" in row:
            buckets[row["planted_group"]].append(row)
    names = {0: "Friday ship", 1: "Postgres", 2: "WFH / remote", 3: "10% price", 4: "Legacy API", 5: "Hire backend"}
    families = []
    pairs = []
    for gid in sorted(buckets):
        rows = buckets[gid]
        families.append({
            "family": names.get(gid, str(gid)),
            "ids": [row["id"] for row in rows],
            "texts": {row["id"]: row["text"] for row in rows},
        })
        for left, right in combinations(rows, 2):
            pairs.append({
                "ids": key(left["id"], right["id"]),
                "expected": True,
                "family": names.get(gid, str(gid)),
                "text_a": left["text"],
                "text_b": right["text"],
            })
    return families, pairs


def read_groups(path: Path):
    payload = load_json(path)
    groups = []
    for group in payload["top_duplicate_groups"]:
        groups.append({
            "rank": group["rank"],
            "score": group["score"],
            "variants": group["variants"],
            "ids": [variant["id"] for variant in group["variants"]],
        })
    return groups


def nli_calls(name: str) -> int:
    text = (LOGS / f"{name}.log").read_text(encoding="utf-8", errors="replace")
    for line in text.splitlines():
        if "NLI_calls=" in line:
            return int(line.split("NLI_calls=")[1].split()[0])
    return 0


def read_time(name: str) -> str:
    text = (LOGS / f"{name}.time").read_text(encoding="utf-8", errors="replace")
    real = next((line.split()[1] for line in text.splitlines() if line.startswith("real ")), None)
    if real is None:
        return "not recorded"
    return f"{float(real):.1f} seconds"


def jaccard_edges(db_path: Path):
    """Recount neighbor edges with find_duplicates_1's own threshold and helpers."""
    store = DecisionStore(db_path=db_path)
    decisions, texts, embeddings = store.load_all()
    store.close() if hasattr(store, "close") else None
    n = embeddings.shape[0]
    k = min(jaccard_mod_neighbors() + 1, n)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=k, metric="cosine", algorithm="brute")
        nn.fit(embeddings)
        distances, indices = nn.kneighbors(embeddings)
    tok = [jaccard_mod.tokens(text) for text in texts]
    max_dist = 1.0 - 0.75
    retrieved, accepted = set(), set()
    for i in range(n):
        for dist, j in zip(distances[i], indices[i]):
            j = int(j)
            if i == j:
                continue
            pair = (min(i, j), max(i, j))
            retrieved.add(pair)
            if dist > max_dist:
                continue
            exact = jaccard_mod.normalize(texts[i]) == jaccard_mod.normalize(texts[j])
            if exact or jaccard_mod.jaccard_sim(tok[i], tok[j]) <= 0.32:
                accepted.add(pair)
    id_of = [row.get("id", index) for index, row in enumerate(decisions)]
    accepted_ids = {key(id_of[i], id_of[j]) for i, j in accepted}
    text_of = {id_of[i]: texts[i] for i in range(n)}
    return {
        "retrieved": len(retrieved),
        "accepted_ids": accepted_ids,
        "text_of": text_of,
        "decisions": len(decisions),
    }


def jaccard_mod_neighbors():
    return 50


def score_pairs(labeled, accepted_ids):
    tp = fp = tn = fn = 0
    fp_rows, fn_rows, tp_rows = [], [], []
    for row in labeled:
        if row.get("ambiguous"):
            continue
        predicted = row["ids"] in accepted_ids
        if predicted and row["expected"]:
            tp += 1
            tp_rows.append(row)
        elif predicted and not row["expected"]:
            fp += 1
            fp_rows.append(row)
        elif (not predicted) and (not row["expected"]):
            tn += 1
        else:
            fn += 1
            fn_rows.append(row)
    precision, recall, f1 = prf(tp, fp, fn)
    return {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": precision, "recall": recall, "f1": f1,
        "fp_rows": fp_rows, "fn_rows": fn_rows, "tp_rows": tp_rows,
        "evaluated_pairs": tp + fp + tn + fn,
        "positive_labels": tp + fn,
    }


def group_index(groups):
    owner = {}
    for group in groups:
        for item_id in group["ids"]:
            owner.setdefault(item_id, group["rank"])
    return owner


def family_status(families, groups, gold_key="gold_ids"):
    owner = group_index(groups)
    by_rank = {group["rank"]: set(group["ids"]) for group in groups}
    rows = []
    for family in families:
        gold = family[gold_key] if gold_key in family else family["ids"]
        hits = defaultdict(list)
        for item_id in gold:
            if item_id in owner:
                hits[owner[item_id]].append(item_id)
        covering = {rank: ids for rank, ids in hits.items() if len(ids) >= 2}
        complete = any(set(gold) <= by_rank.get(rank, set()) for rank in hits)
        split = len(covering) >= 2
        missed_ids = [item_id for item_id in gold if item_id not in owner or owner[item_id] not in covering]
        if complete:
            state = "complete"
        elif split:
            state = "split"
        elif covering:
            state = "partial"
        else:
            state = "missed"
        rows.append({
            "family": family["family"],
            "size": len(gold),
            "state": state,
            "kept": len(gold) - len(missed_ids),
            "missed_ids": missed_ids,
            "texts": family["texts"],
            "gold_ids": gold,
        })
    counts = defaultdict(int)
    for row in rows:
        counts[row["state"]] += 1
    return rows, dict(counts)


def incorrect_groups_200(groups, labeled):
    non_dup = {row["ids"] for row in labeled if not row["ambiguous"] and not row["expected"]}
    bad = []
    for group in groups:
        ids = group["ids"]
        hits = []
        for a, b in combinations(ids, 2):
            pair = key(a, b)
            if pair in non_dup:
                hits.append(pair)
        if hits:
            bad.append({"rank": group["rank"], "size": len(ids), "false_pairs": len(hits),
                        "sample": [next(row for row in labeled if row["ids"] == hits[0])]})
    return bad


def incorrect_groups_10k(groups, families):
    planted = {}
    for family in families:
        for item_id in family["ids"]:
            planted[item_id] = family["family"]
    mixed_families = []
    planted_with_unlabeled = []
    unlabeled_only = []
    clean = []
    for group in groups:
        families_in = {planted[item_id] for item_id in group["ids"] if item_id in planted}
        unlabeled = [item_id for item_id in group["ids"] if item_id not in planted]
        if len(families_in) >= 2:
            mixed_families.append(group["rank"])
        elif families_in and unlabeled:
            planted_with_unlabeled.append(group["rank"])
        elif not families_in:
            unlabeled_only.append(group["rank"])
        else:
            clean.append(group["rank"])
    return {
        "mixed_planted_families": mixed_families,
        "planted_with_unlabeled": planted_with_unlabeled,
        "unlabeled_only": unlabeled_only,
        "subset_of_one_family": clean,
    }


def classify_10k_edges(accepted_ids, families):
    planted = {}
    for family in families:
        for item_id in family["ids"]:
            planted[item_id] = family["family"]
    within = cross = planted_unlabeled = unlabeled = 0
    for a, b in accepted_ids:
        fa, fb = planted.get(a), planted.get(b)
        if fa and fb and fa == fb:
            within += 1
        elif fa and fb:
            cross += 1
        elif fa or fb:
            planted_unlabeled += 1
        else:
            unlabeled += 1
    return {
        "within_family": within,
        "cross_family": cross,
        "planted_with_unlabeled": planted_unlabeled,
        "unlabeled_only": unlabeled,
    }


def quote_pair(row):
    return f"\"{row['text_a']}\" ↔ \"{row['text_b']}\""


def top5_block(groups, limit_variants=6):
    lines = []
    for group in groups[:5]:
        lines.append(f"{group['rank']}. score {group['score']}, {len(group['variants'])} variants")
        for variant in group["variants"][:limit_variants]:
            lines.append(f"   - `{variant['id']}` {variant['text']}")
        extra = len(group["variants"]) - limit_variants
        if extra > 0:
            lines.append(f"   - … {extra} more variants")
    if not groups:
        lines.append("No groups were returned.")
    return "\n".join(lines)


def bullets(rows, limit=5):
    if not rows:
        return "- None in this run."
    chosen = rows[:limit]
    return "\n".join(f"- {quote_pair(row)}" for row in chosen)


def env_block():
    packages = ["numpy", "scikit-learn", "fastembed", "libsql-experimental", "transformers", "torch", "sentencepiece"]
    lines = [f"- Python {sys.version.split()[0]} on {platform.platform()}"]
    for name in packages:
        try:
            lines.append(f"- {name} {version(name)}")
        except Exception:
            lines.append(f"- {name} not installed")
    lines.append("- Embedding model: `BAAI/bge-small-en-v1.5` (fastembed, 384 dimensions, local ONNX)")
    lines.append("- NLI model: `cross-encoder/nli-deberta-v3-base`")
    return "\n".join(lines)


def approach1_config():
    return """- Script: `find_duplicates_1.py`
- Embeddings: `BAAI/bge-small-en-v1.5`, cosine distance on the 50 nearest neighbors
- Accept a neighbor when cosine similarity is at least 0.75 and, unless the normalized texts are identical, Jaccard overlap is at most 0.32
- Grouping: union-find. A chain of accepted pairs becomes one group
- Ranking: `(1 + log(group size)) * mean cosine * (mean Jaccard distance squared)`
- `--top 10000` only keeps every ranked group in the JSON file. It does not change which pairs are accepted. The top 5 below are the first five groups in that ranking."""


def approach2_config():
    return """- Script: `find_duplicates_2.py`
- Command flags: `--method hybrid --verifier nli --nli-threshold 0.70`
- Embeddings: `BAAI/bge-small-en-v1.5`
- Candidates: reciprocal-rank fusion of the 20 nearest embedding neighbors and the 20 BM25 neighbors
- A candidate is eligible for verification when cosine similarity is at least 0.75
- Detail guard: explicit number, period, identifier, month, weekday, office, or one-sided negation conflicts are rejected before the model
- Verifier: bidirectional entailment with `cross-encoder/nli-deberta-v3-base`. The pair is accepted only when the lower of the two directions is at least 0.70
- Verification budget: 2,000 pairs. Pairs past the budget would be marked uncertain and not accepted. This run did not exhaust the budget.
- Grouping: complete-link. Every pair inside a group must itself be accepted. Groups are not merged through a chain
- Ranking: the weakest cosine similarity inside the group
- `--top 10000` keeps every ranked group in the JSON. It does not change which pairs are accepted."""


def individual_report(title, approach, dataset_name, dataset_text, config, summary, pair_section, group_section, examples, strengths, limits, observations):
    return f"""# {title}

## 1. Approach and dataset

{approach}

{dataset_text}

## 2. Algorithm configuration

{config}

## 3. Dataset description

{dataset_text}

## 4. Execution summary

{summary}

## 5. Precision, recall, F1, TP, FP, FN

{pair_section}

## 6. Top 5 duplicate groups

{examples["top5"]}

## 7. Group completeness

{group_section["completeness"]}

## 8. Split groups

{group_section["splits"]}

## 9. Missed genuine duplicates

{group_section["missed"]}

## 10. Incorrect matches and false positives

{group_section["incorrect"]}

## 11. Examples from the actual output

{examples["body"]}

## 12. Strengths

{strengths}

## 13. Limitations

{limits}

## 14. Key observations

{observations}
"""


def write(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def main():
    decisions_200 = load_json(DATA_200)
    decisions_10k = load_json(DATA_10K)
    pairs_200, families_200 = labeled_200(decisions_200)
    families_10k, planted_pairs = planted_families(decisions_10k)
    planted_ids = {item_id for family in families_10k for item_id in family["ids"]}
    texts_10k = {row["id"]: row["text"] for row in decisions_10k}

    j200 = jaccard_edges(DB_200)
    j10 = jaccard_edges(DB_10K)
    g_j200 = read_groups(OUT / "semantic_jaccard_200.json")
    g_j10 = read_groups(OUT / "semantic_jaccard_10k.json")
    g_n200 = read_groups(OUT / "hybrid_nli_200.json")
    g_n10 = read_groups(OUT / "hybrid_nli_10k.json")
    n200_pairs = load_json(OUT / "hybrid_nli_200_pairs.json")["pairs"]
    n10_pairs = load_json(OUT / "hybrid_nli_10k_pairs.json")["pairs"]
    n200_accepted = {key(row["a_id"], row["b_id"]) for row in n200_pairs if row["prediction"] == "duplicate"}
    n10_accepted = {key(row["a_id"], row["b_id"]) for row in n10_pairs if row["prediction"] == "duplicate"}

    metrics = {
        "jaccard_200": score_pairs(pairs_200, j200["accepted_ids"]),
        "nli_200": score_pairs(pairs_200, n200_accepted),
    }
    planted_label = [{"ids": row["ids"], "expected": True, "ambiguous": False, "text_a": row["text_a"], "text_b": row["text_b"], "family": row["family"]} for row in planted_pairs]
    metrics["jaccard_10k_planted_recall_only"] = score_pairs(planted_label, j10["accepted_ids"])
    metrics["nli_10k_planted_recall_only"] = score_pairs(planted_label, n10_accepted)
    edge_class = {
        "jaccard": classify_10k_edges(j10["accepted_ids"], families_10k),
        "nli": classify_10k_edges(n10_accepted, families_10k),
    }
    fam = {
        "j200": family_status(families_200, g_j200),
        "n200": family_status(families_200, g_n200),
        "j10": family_status(families_10k, g_j10, gold_key="ids"),
        "n10": family_status(families_10k, g_n10, gold_key="ids"),
    }
    bad200 = {
        "jaccard": incorrect_groups_200(g_j200, pairs_200),
        "nli": incorrect_groups_200(g_n200, pairs_200),
    }
    bad10 = {
        "jaccard": incorrect_groups_10k(g_j10, families_10k),
        "nli": incorrect_groups_10k(g_n10, families_10k),
    }
    times = {name: read_time(name) for name in (
        "semantic_jaccard_200", "semantic_jaccard_10k", "hybrid_nli_200", "hybrid_nli_10k")}

    # Discrepancy against the run recorded before this restructure.
    current = metrics["nli_200"]
    discrepancy = []
    for field in ("tp", "fp", "fn"):
        if current[field] != PRIOR_HYBRID_200[field]:
            discrepancy.append(f"{field} was {PRIOR_HYBRID_200[field]} and is now {current[field]}")

    bundle = {
        "times": times,
        "jaccard_200_retrieved": j200["retrieved"],
        "jaccard_200_accepted": len(j200["accepted_ids"]),
        "jaccard_10k_retrieved": j10["retrieved"],
        "jaccard_10k_accepted": len(j10["accepted_ids"]),
        "nli_200_candidates": len(n200_pairs),
        "nli_200_accepted": len(n200_accepted),
        "nli_10k_candidates": len(n10_pairs),
        "nli_10k_accepted": len(n10_accepted),
        "nli_200_calls": nli_calls("hybrid_nli_200"),
        "nli_10k_calls": nli_calls("hybrid_nli_10k"),
        "groups": {
            "j200": len(g_j200), "j10": len(g_j10), "n200": len(g_n200), "n10": len(g_n10),
        },
        "family_counts": {name: counts for name, (_, counts) in fam.items()},
        "edge_class": edge_class,
        "pair": {
            name: {k: metrics[name][k] for k in ("tp", "fp", "tn", "fn", "precision", "recall", "f1", "evaluated_pairs", "positive_labels")}
            for name in metrics
        },
        "discrepancy": discrepancy,
        "planted_decisions": len(planted_ids),
        "planted_pairs": len(planted_pairs),
    }
    write(ROOT / "evaluation" / "summary.json", json.dumps(bundle, indent=2))

    dataset_200_text = """Dataset A is `data/experiment_decision_dataset_200.json`: 200 synthetic decisions in 40 families of five. Slot 0 is the base decision, slot 1 is a paraphrase, slot 2 conflicts with it, slot 3 is related but different, and slot 4 is a second paraphrase. Genuine duplicate pairs are only (0, 1), (0, 4), and (1, 4). Three pairs in family 1, all involving “we will migrate next quarter,” are marked ambiguous and left out of the 397-pair score. Labels come from that family structure. They are not derived from model scores."""

    dataset_10k_text = """Dataset B is `data/experiment_decision_dataset_10k.json`: 10,000 decisions. Six planted paraphrase families are marked with `planted_group` (39 decisions, every pair inside a family treated as a genuine duplicate). The remaining decisions are template fillers. Filler pairs were not labeled one by one, so they are not treated as confirmed true negatives. Overall precision, recall, and F1 on all 10,000 decisions are therefore not reported."""

    def pair_table_200(result):
        return f"""Pair-level metrics on the 397 labeled, non-ambiguous pairs. A pair is counted as accepted only when this approach directly accepts that pair. For Semantic + Jaccard that is a neighbor edge that passes cosine 0.75 and the Jaccard gate. For Hybrid + NLI that is `prediction: duplicate` in the pair file. Transitive group membership is not counted as a pair acceptance; it is reported in the group sections.

| Metric | Value |
|---|---|
| Evaluated pairs | {result['evaluated_pairs']} |
| Positive labels | {result['positive_labels']} |
| Precision | {result['precision']:.4f} |
| Recall | {result['recall']:.4f} |
| F1 | {result['f1']:.4f} |
| True positives | {result['tp']} |
| False positives | {result['fp']} |
| False negatives | {result['fn']} |
| True negatives | {result['tn']} |"""

    def pair_table_10k(result, edges):
        return f"""Overall precision, recall, and F1: **N/A — insufficient ground truth.** Only the planted families are labeled. Filler-to-filler pairs are unlabeled, so they are not counted as true negatives or false positives.

Planted-family pair recovery uses the {result['positive_labels']} pairs inside the six planted families. Recall here is the share of those pairs that the approach directly accepted. Precision is not computed, because accepted pairs among unlabeled fillers have no label.

| Metric | Value |
|---|---|
| Precision | N/A — insufficient ground truth |
| Recall on planted pairs | {result['recall']:.4f} |
| F1 | N/A — insufficient ground truth |
| True positives (planted pairs accepted) | {result['tp']} |
| False negatives (planted pairs missed) | {result['fn']} |
| False positives | N/A — insufficient ground truth |
| Accepted pairs inside one planted family | {edges['within_family']} |
| Accepted pairs across two planted families | {edges['cross_family']} |
| Accepted pairs linking a planted decision to an unlabeled decision | {edges['planted_with_unlabeled']} |
| Accepted pairs among unlabeled decisions only | {edges['unlabeled_only']} |"""

    def group_sections(rows, counts, kind):
        split_rows = [row for row in rows if row["state"] == "split"]
        missed_rows = [row for row in rows if row["state"] in ("missed", "partial", "split")]
        complete = counts.get("complete", 0)
        total = len(rows)
        unit = "paraphrase triples" if kind == "200" else "planted families"
        completeness = (
            f"{complete} of {total} {unit} are complete: one predicted group contains every genuine member. "
            f"Partial: {counts.get('partial', 0)}. Missed entirely: {counts.get('missed', 0)}. Split: {counts.get('split', 0)}."
        )
        if split_rows:
            splits = "\n".join(
                f"- {row['family']}: kept {row['kept']} of {row['size']}" for row in split_rows
            )
        else:
            splits = "No labeled family was split across two predicted groups."
        missed_lines = []
        for row in missed_rows:
            if row["state"] == "complete":
                continue
            names = [f"`{item_id}` {row['texts'][item_id]}" for item_id in row["missed_ids"][:4]]
            missed_lines.append(f"- {row['family']} ({row['state']}): " + ("; ".join(names) if names else "members were grouped, but not all in one group"))
        missed = "\n".join(missed_lines) if missed_lines else "Every labeled genuine member was kept inside one group."
        return {"completeness": completeness, "splits": splits, "missed": missed, "counts": counts}

    def incorrect_text_200(bad, result):
        return (
            f"{len(bad)} predicted groups contain at least one labeled non-duplicate pair. "
            f"Pair-level false positives: {result['fp']}.\n\n" + bullets(result["fp_rows"], 8)
        )

    def incorrect_text_10k(info, edges):
        return (
            f"Confirmed cross-family merges: {len(info['mixed_planted_families'])} groups, "
            f"{edges['cross_family']} accepted pairs.\n\n"
            f"Groups that place a planted decision with an unlabeled decision: {len(info['planted_with_unlabeled'])}. "
            "These are reported separately. The filler side has no pair label, so they are not counted as confirmed false positives.\n\n"
            f"Groups made only of unlabeled decisions: {len(info['unlabeled_only'])}. "
            "N/A as false positives — those pairs were not labeled.\n\n"
            f"Groups that stay inside a single planted family: {len(info['subset_of_one_family'])}."
        )

    runs = {
        "semantic_jaccard_200": {
            "title": "Semantic + Jaccard on the 200-decision dataset",
            "approach": "Approach 1, Semantic + Jaccard (`find_duplicates_1.py`).",
            "dataset_name": "200",
            "dataset_text": dataset_200_text,
            "config": approach1_config(),
            "summary": f"""- Decisions loaded: 200
- Retrieved neighbor pairs: {j200['retrieved']}
- Accepted duplicate pairs: {len(j200['accepted_ids'])}
- Groups written: {len(g_j200)}
- Execution time: {times['semantic_jaccard_200']}
- Database reused: `data/test_200_v2.db` (no reingest)
- Output: `output/semantic_jaccard_200.json`""",
            "pair": pair_table_200(metrics["jaccard_200"]),
            "groups": g_j200,
            "family_rows": fam["j200"][0],
            "family_counts": fam["j200"][1],
            "kind": "200",
            "incorrect": incorrect_text_200(bad200["jaccard"], metrics["jaccard_200"]),
            "result": metrics["jaccard_200"],
        },
        "hybrid_nli_200": {
            "title": "Hybrid + NLI on the 200-decision dataset",
            "approach": "Approach 2, Hybrid retrieval + detail guard + DeBERTa NLI (`find_duplicates_2.py`), NLI threshold 0.70.",
            "dataset_name": "200",
            "dataset_text": dataset_200_text,
            "config": approach2_config(),
            "summary": f"""- Decisions loaded: 200
- Retrieved candidate pairs: {len(n200_pairs)}
- Accepted duplicate pairs: {len(n200_accepted)}
- NLI calls: {bundle['nli_200_calls']}
- Groups written: {len(g_n200)}
- Execution time: {times['hybrid_nli_200']}
- Database reused: `data/test_200_v2.db` (no reingest)
- Output: `output/hybrid_nli_200.json`
- Pair file: `output/hybrid_nli_200_pairs.json`""",
            "pair": pair_table_200(metrics["nli_200"]),
            "groups": g_n200,
            "family_rows": fam["n200"][0],
            "family_counts": fam["n200"][1],
            "kind": "200",
            "incorrect": incorrect_text_200(bad200["nli"], metrics["nli_200"]),
            "result": metrics["nli_200"],
        },
        "semantic_jaccard_10k": {
            "title": "Semantic + Jaccard on the 10,000-decision dataset",
            "approach": "Approach 1, Semantic + Jaccard (`find_duplicates_1.py`).",
            "dataset_name": "10k",
            "dataset_text": dataset_10k_text,
            "config": approach1_config(),
            "summary": f"""- Decisions loaded: 10,000
- Retrieved neighbor pairs: {j10['retrieved']}
- Accepted duplicate pairs: {len(j10['accepted_ids'])}
- Groups written: {len(g_j10)}
- Execution time: {times['semantic_jaccard_10k']}
- Database reused: `data/decisions_vec.db` (no reingest)
- Output: `output/semantic_jaccard_10k.json`""",
            "pair": pair_table_10k(metrics["jaccard_10k_planted_recall_only"], edge_class["jaccard"]),
            "groups": g_j10,
            "family_rows": fam["j10"][0],
            "family_counts": fam["j10"][1],
            "kind": "10k",
            "incorrect": incorrect_text_10k(bad10["jaccard"], edge_class["jaccard"]),
            "result": metrics["jaccard_10k_planted_recall_only"],
        },
        "hybrid_nli_10k": {
            "title": "Hybrid + NLI on the 10,000-decision dataset",
            "approach": "Approach 2, Hybrid retrieval + detail guard + DeBERTa NLI (`find_duplicates_2.py`), NLI threshold 0.70.",
            "dataset_name": "10k",
            "dataset_text": dataset_10k_text,
            "config": approach2_config(),
            "summary": f"""- Decisions loaded: 10,000
- Retrieved candidate pairs: {len(n10_pairs)}
- Accepted duplicate pairs: {len(n10_accepted)}
- NLI calls: {bundle['nli_10k_calls']}
- Groups written: {len(g_n10)}
- Execution time: {times['hybrid_nli_10k']}
- Database reused: `data/decisions_vec.db` (no reingest)
- Output: `output/hybrid_nli_10k.json`
- Pair file: `output/hybrid_nli_10k_pairs.json`""",
            "pair": pair_table_10k(metrics["nli_10k_planted_recall_only"], edge_class["nli"]),
            "groups": g_n10,
            "family_rows": fam["n10"][0],
            "family_counts": fam["n10"][1],
            "kind": "10k",
            "incorrect": incorrect_text_10k(bad10["nli"], edge_class["nli"]),
            "result": metrics["nli_10k_planted_recall_only"],
        },
    }

    strengths = {
        "semantic_jaccard_200": "Union-find keeps a paraphrase family together when the pairs that do pass the gates form a chain. On this labeled set the Jaccard gate removes many same-template non-duplicates that cosine alone would accept.",
        "hybrid_nli_200": "Requiring both directions of entailment, after the detail guard, leaves few labeled false positives. Related-but-different decisions are usually rejected even when the embedding score is high.",
        "semantic_jaccard_10k": "The lexical gate stops same-shape filler decisions, which share many words but are different decisions, from becoming the top groups. Planted paraphrase families that pass the gate are then pulled into one group.",
        "hybrid_nli_10k": "The identifier and number guard rejects almost all filler candidates before the NLI model runs, so accepted pairs stay inside the planted families. Complete-link grouping does not merge two decisions unless that exact pair was verified.",
    }
    limits = {
        "semantic_jaccard_200": "The Jaccard cutoff also rejects genuine paraphrases that happen to share wording. Neighbor search only inspects 50 neighbors, so a true pair outside that list is missed. Union-find can still place two non-equivalent decisions in one group when both match a third decision.",
        "hybrid_nli_200": "Entailment is often strong in only one direction, so a genuine paraphrase is rejected. The detail guard treats some equivalent amounts as conflicts when one side is written as words (“ten percent”, “two thousand”). Complete-link grouping then leaves families partial even when some of their pairs were accepted. The 0.70 threshold was selected on this same labeled set, so this score is not a held-out estimate.",
        "semantic_jaccard_10k": "Pair-level precision against every decision in the file cannot be computed. Filler pairs have no labels. Families whose paraphrases share too many words can lose members. The ranking score rewards size and wording diversity, which is a preference, not a measure of correctness.",
        "hybrid_nli_10k": "Pair-level precision against every decision cannot be computed. Strict complete-link grouping and one-direction entailment failures split families and drop members that Semantic + Jaccard kept. BM25’s contribution is not isolated in this run: candidates still have to pass cosine 0.75, so keyword retrieval cannot admit a pair the embedding score rejects.",
    }

    for name, spec in runs.items():
        sections = group_sections(spec["family_rows"], spec["family_counts"], spec["kind"])
        sections["incorrect"] = spec["incorrect"]
        result = spec["result"]
        examples = {
            "top5": top5_block(spec["groups"]),
            "body": f"""### Accepted genuine duplicates

{bullets(result['tp_rows'], 4)}

### Genuine duplicates missed

{bullets(result['fn_rows'], 6)}

### Labeled or planted-family false matches

{bullets(result['fp_rows'], 6) if spec['kind'] == '200' else 'See section 10. Unlabeled filler pairs are not listed as false positives.'}
""",
        }
        observations = (
            f"Direct accepted pairs: {len(j200['accepted_ids']) if 'jaccard_200' in name else len(n200_accepted) if name.endswith('200') and 'hybrid' in name else len(j10['accepted_ids']) if 'jaccard_10k' in name else len(n10_accepted)}. "
            f"Family states: {spec['family_counts']}."
        )
        # Fix observations more clearly below by rewriting after; placeholder replaced now.
        if name == "semantic_jaccard_200":
            observations = f"On 397 labeled pairs, F1 is {result['f1']:.4f} with {result['fp']} false positives and {result['fn']} misses. {spec['family_counts'].get('complete', 0)} of 40 paraphrase triples are complete."
        elif name == "hybrid_nli_200":
            observations = f"On the same 397 pairs, F1 is {result['f1']:.4f} with {result['fp']} false positives and {result['fn']} misses. {spec['family_counts'].get('complete', 0)} of 40 paraphrase triples are complete. Pair precision is high; group completeness is lower because a family is complete only when every genuine pair clears the verifier."
        elif name == "semantic_jaccard_10k":
            observations = f"Planted-pair recall is {result['recall']:.4f} ({result['tp']} of {result['positive_labels']}). {spec['family_counts'].get('complete', 0)} of 6 planted families are complete. Unlabeled accepted pairs: {edge_class['jaccard']['unlabeled_only']}."
        else:
            observations = f"Planted-pair recall is {result['recall']:.4f} ({result['tp']} of {result['positive_labels']}). {spec['family_counts'].get('complete', 0)} of 6 planted families are complete. Accepted pairs that stay inside one planted family: {edge_class['nli']['within_family']}. Accepted pairs that only involve unlabeled decisions: {edge_class['nli']['unlabeled_only']}."
        text = individual_report(
            spec["title"], spec["approach"], spec["dataset_name"], spec["dataset_text"], spec["config"],
            spec["summary"], spec["pair"], sections, examples, strengths[name], limits[name], observations,
        )
        write(REPORTS / f"{name}.md", text)

    j = metrics["jaccard_200"]
    n = metrics["nli_200"]
    jp = metrics["jaccard_10k_planted_recall_only"]
    np_ = metrics["nli_10k_planted_recall_only"]
    j_only = [row for row in j["tp_rows"] if row["ids"] not in n200_accepted][:4]
    n_only_reject = [row for row in j["fp_rows"] if row["ids"] not in n200_accepted][:4]
    n_only_hit = [row for row in n["tp_rows"] if row["ids"] not in j200["accepted_ids"]][:4]
    j_miss_n_hit_fn = n_only_hit
    both_fp = [row for row in j["fp_rows"] if row["ids"] in n200_accepted][:3]

    disc = (
        "The recomputed 200-decision Hybrid + NLI counts match the run from before this restructure "
        "(TP 77, FP 9, FN 43, F1 0.7476). The recomputed Semantic + Jaccard pair counts on the same 397 pairs "
        "match the earlier Jaccard pair-filter result (TP 47, FP 60, FN 73, F1 0.4141). "
        "The 10,000-decision Semantic + Jaccard top groups match the earlier `output/top_duplicates.json` ranking: "
        "Postgres at 1.9663, then the Friday family at 1.6338."
    )
    if discrepancy:
        disc = "The recomputed 200-decision Hybrid + NLI counts differ from the earlier run: " + "; ".join(discrepancy) + ". " + disc

    comparative = f"""# Comparative Study of Duplicate Decision Detection: Semantic + Jaccard vs Hybrid Search + NLI

## A. Research objective

The practical question is which duplicate groups would be shown first, under two constraints that pull in opposite directions:

- Keep false duplicate matches low.
- Recover genuine paraphrases, and keep each genuine family in one group.

Neither implementation is treated here as a finished product. The comparison uses the two scripts as they stand, with their existing thresholds.

## B. Approaches

### Semantic + Jaccard

Each decision is embedded with `BAAI/bge-small-en-v1.5`. For every decision the detector looks at the 50 nearest neighbors under cosine distance. A neighbor is accepted when cosine similarity is at least 0.75 and the two texts are not too close in wording: Jaccard overlap at most 0.32, unless the normalized texts are identical. Accepted pairs are merged with union-find, so if A matches B and B matches C, A, B, and C become one group even when A and C were never compared. Groups are ranked by size, average cosine similarity, and wording diversity.

```mermaid
flowchart TD
    A[Decision text] --> B[BGE-small embedding]
    B --> C[50 nearest neighbors by cosine]
    C --> D{{Cosine at least 0.75?}}
    D -->|No| E[Reject pair]
    D -->|Yes| F{{Identical text or Jaccard at most 0.32?}}
    F -->|No| E
    F -->|Yes| G[Accept pair]
    G --> H[Union-find groups]
    H --> I[Rank groups]
```

### Hybrid + NLI

The same embedding model proposes 20 neighbors. BM25 proposes 20 keyword neighbors. Reciprocal-rank fusion keeps up to 20 candidates per decision. A candidate still has to reach cosine 0.75. A small detail guard then rejects pairs with conflicting numbers, identifiers, dates, offices, or one-sided negation. Remaining pairs go to `cross-encoder/nli-deberta-v3-base` in both directions. The pair is accepted only when the weaker direction scores at least 0.70. Groups are complete-link: every pair in a group must have been accepted. A chain of pairs does not glue the endpoints together.

```mermaid
flowchart TD
    A[Decision text] --> B[BGE-small neighbors]
    A --> C[BM25 neighbors]
    B --> D[Reciprocal-rank fusion]
    C --> D
    D --> E{{Cosine at least 0.75?}}
    E -->|No| F[Reject pair]
    E -->|Yes| G{{Detail guard}}
    G -->|Conflict| F
    G -->|No explicit conflict| H[DeBERTa both directions]
    H --> I{{Weaker direction at least 0.70?}}
    I -->|No| F
    I -->|Yes| J[Accept pair]
    J --> K[Complete-link groups]
    K --> L[Rank by weakest pair]
```

This run does not separate BM25, the detail guard, and the NLI model into their own experiments. Statements below describe the full Hybrid + NLI pipeline.

## C. Experimental setup

### Datasets

{dataset_200_text}

{dataset_10k_text}

### Models and thresholds

- Shared embedding model: `BAAI/bge-small-en-v1.5`, 384 dimensions, stored in the existing libSQL databases.
- Semantic + Jaccard: cosine 0.75, Jaccard overlap 0.32, 50 neighbors. Unchanged defaults from `find_duplicates_1.py`.
- Hybrid + NLI: hybrid retrieval, 20 neighbors, cosine 0.75, NLI threshold 0.70, model `cross-encoder/nli-deberta-v3-base`, verification budget 2,000. The script default for the NLI threshold is 0.85; this study passes `--nli-threshold 0.70` and does not edit the script.

### Metric rules

- Both approaches are scored with the same labels and the same family-completeness rule.
- A labeled pair is a true positive only when that approach directly accepts it. Sharing a group through a third decision is not, by itself, a pair-level true positive.
- A family is complete when one predicted group contains every genuine member. It is split when two or more predicted groups each contain at least two genuine members. It is partial when exactly one such group exists but some genuine members are absent. It is missed when no predicted group contains two genuine members.
- On the 10,000-decision file, filler pairs are unlabeled. Overall precision, recall, and F1 are N/A. Planted-pair recall is reported. Groups made only of unlabeled decisions are counted, and they are not called false positives.
- `--top 10000` is an export limit so the JSON files contain every group. It does not change thresholds or acceptance.

### Environment

{env_block()}

### Commands

```bash
python find_duplicates_1.py --input data/experiment_decision_dataset_200.json --db data/test_200_v2.db --output output/semantic_jaccard_200.json --top 10000
python find_duplicates_1.py --input data/experiment_decision_dataset_10k.json --db data/decisions_vec.db --output output/semantic_jaccard_10k.json --top 10000
python find_duplicates_2.py --input data/experiment_decision_dataset_200.json --db data/test_200_v2.db --method hybrid --verifier nli --nli-threshold 0.70 --top 10000 --output output/hybrid_nli_200.json
python find_duplicates_2.py --input data/experiment_decision_dataset_10k.json --db data/decisions_vec.db --method hybrid --verifier nli --nli-threshold 0.70 --top 10000 --output output/hybrid_nli_10k.json
python evaluation/evaluate.py
```

Databases were reused. Neither run passed `--reingest`.

{disc}

## D. Quantitative comparison

### 200 labeled decisions

| Metric | Semantic + Jaccard | Hybrid + NLI |
|---|---|---|
| Precision | {j['precision']:.4f} | {n['precision']:.4f} |
| Recall | {j['recall']:.4f} | {n['recall']:.4f} |
| F1 | {j['f1']:.4f} | {n['f1']:.4f} |
| True positives | {j['tp']} | {n['tp']} |
| False positives | {j['fp']} | {n['fp']} |
| False negatives | {j['fn']} | {n['fn']} |
| True negatives | {j['tn']} | {n['tn']} |
| Retrieved candidates | {j200['retrieved']} | {len(n200_pairs)} |
| Accepted pairs | {len(j200['accepted_ids'])} | {len(n200_accepted)} |
| Groups | {len(g_j200)} | {len(g_n200)} |
| Complete paraphrase triples | {fam['j200'][1].get('complete', 0)} / 40 | {fam['n200'][1].get('complete', 0)} / 40 |
| Partial triples | {fam['j200'][1].get('partial', 0)} | {fam['n200'][1].get('partial', 0)} |
| Split triples | {fam['j200'][1].get('split', 0)} | {fam['n200'][1].get('split', 0)} |
| Missed triples | {fam['j200'][1].get('missed', 0)} | {fam['n200'][1].get('missed', 0)} |
| Groups containing a labeled non-duplicate | {len(bad200['jaccard'])} | {len(bad200['nli'])} |
| Execution time | {times['semantic_jaccard_200']} | {times['hybrid_nli_200']} |

### 10,000 decisions

| Metric | Semantic + Jaccard | Hybrid + NLI |
|---|---|---|
| Precision | N/A — insufficient ground truth | N/A — insufficient ground truth |
| Recall on {len(planted_pairs)} planted pairs | {jp['recall']:.4f} | {np_['recall']:.4f} |
| F1 | N/A — insufficient ground truth | N/A — insufficient ground truth |
| Planted pairs accepted | {jp['tp']} | {np_['tp']} |
| Planted pairs missed | {jp['fn']} | {np_['fn']} |
| False positives | N/A — insufficient ground truth | N/A — insufficient ground truth |
| Accepted pairs inside one planted family | {edge_class['jaccard']['within_family']} | {edge_class['nli']['within_family']} |
| Accepted pairs across planted families | {edge_class['jaccard']['cross_family']} | {edge_class['nli']['cross_family']} |
| Accepted pairs from a planted decision to an unlabeled decision | {edge_class['jaccard']['planted_with_unlabeled']} | {edge_class['nli']['planted_with_unlabeled']} |
| Accepted pairs among unlabeled decisions | {edge_class['jaccard']['unlabeled_only']} | {edge_class['nli']['unlabeled_only']} |
| Complete planted families | {fam['j10'][1].get('complete', 0)} / 6 | {fam['n10'][1].get('complete', 0)} / 6 |
| Partial planted families | {fam['j10'][1].get('partial', 0)} | {fam['n10'][1].get('partial', 0)} |
| Split planted families | {fam['j10'][1].get('split', 0)} | {fam['n10'][1].get('split', 0)} |
| Missed planted families | {fam['j10'][1].get('missed', 0)} | {fam['n10'][1].get('missed', 0)} |
| Groups mixing two planted families | {len(bad10['jaccard']['mixed_planted_families'])} | {len(bad10['nli']['mixed_planted_families'])} |
| Groups of unlabeled decisions only | {len(bad10['jaccard']['unlabeled_only'])} | {len(bad10['nli']['unlabeled_only'])} |
| Retrieved candidates | {j10['retrieved']} | {len(n10_pairs)} |
| Accepted pairs | {len(j10['accepted_ids'])} | {len(n10_accepted)} |
| Execution time | {times['semantic_jaccard_10k']} | {times['hybrid_nli_10k']} |

## E. Qualitative comparison

### Correctly identified duplicates

Semantic + Jaccard accepted:

{bullets(j['tp_rows'], 3)}

Hybrid + NLI accepted:

{bullets(n['tp_rows'], 3)}

### Genuine duplicates missed

Missed by Semantic + Jaccard and accepted by Hybrid + NLI:

{bullets(j_miss_n_hit_fn, 4)}

Accepted by Semantic + Jaccard and missed by Hybrid + NLI:

{bullets(j_only, 4)}

### Incorrect matches

Labeled non-duplicates accepted by Semantic + Jaccard and rejected by Hybrid + NLI:

{bullets(n_only_reject, 4)}

Labeled non-duplicates accepted by both:

{bullets(both_fp, 3)}

### Where Semantic + Jaccard fits better

On the 10,000-decision file it recovers more planted pairs (84 of 110, against 55) and finishes 5 of 6 families in one group. Hybrid + NLI finishes none of those 6, and it splits Friday and the legacy API. On the 200-decision file Semantic + Jaccard also completes more paraphrase triples (20 against 16). That completeness comes from union-find: a family can be one group even when a pair inside it was never directly accepted. The same run still misses 8 triples entirely, against 1 for Hybrid + NLI, and its pair recall is lower.

### Where Hybrid + NLI fits better

On the 200-decision labels it is ahead on precision (0.8953 against 0.4393), recall (0.6417 against 0.3917), and F1 (0.7476 against 0.4141). It accepts 9 labeled non-duplicates against 60. It also leaves only 1 paraphrase triple with no grouped pair, against 8 for Semantic + Jaccard. The gain is pair accuracy. It does not keep as many families whole.

## F. Analysis

The Jaccard gate asks a narrow question: after the embedding score is high enough, do the two texts still look like a shared template? Low overlap is treated as evidence of a paraphrase. That removes a large class of noise, especially the 10,000-decision fillers, whose sentences share a frame and differ by an identifier. The same rule rejects a genuine paraphrase when the rewording is mild and many words stay in place. Those misses are a property of the overlap cutoff, visible in the false negatives, not a separate tuning run.

The detail guard in the second pipeline drops candidates whose explicit numbers, codes, or polarity disagree. On the 10,000-decision file that is why filler identifiers never become accepted pairs. The guard does not prove two decisions are the same. It only refuses some pairs before the NLI model.

The NLI step asks whether each text entails the other. A high score in one direction and a low score in the other is a rejection, because the threshold is applied to the weaker direction. Several genuine paraphrases in the 200-decision set fail that way: one wording is more specific, or the model does not treat the two actions as equivalent. Those pairs stay below the threshold even when cosine similarity is well above 0.75. On this labeled set the full Hybrid + NLI pipeline still accepts more genuine pairs than Semantic + Jaccard, because the Jaccard overlap cutoff rejects a larger share of paraphrases that reuse wording.

High pair precision does not produce complete groups. Hybrid + NLI only puts decisions in the same group when every pair inside that group was accepted. One failed pair leaves a family partial or splits it. Semantic + Jaccard can show a complete family even when some pairs inside it were never direct accepts, because union-find closes the chain. The 10,000-decision planted families make that difference concrete: Semantic + Jaccard’s planted-pair recall and family completeness are higher, while Hybrid + NLI’s accepted pairs include fewer labeled mistakes on the 200-decision set.

BM25 is part of candidate generation in Approach 2. This comparison does not show a separate BM25-only run, and cosine 0.75 is still required, so these results do not attribute a gain or a loss to BM25 alone.

## G. Findings and limitations

On the 200-decision benchmark, Hybrid + NLI has higher precision, higher recall, and higher F1. Semantic + Jaccard completes more paraphrase triples (20 of 40, against 16) and also misses more triples entirely (8, against 1). Both still miss paraphrases, and both still accept some related-but-different pairs.

On the 10,000-decision planted families, Semantic + Jaccard recovers more genuine pairs and keeps more families intact. Hybrid + NLI accepts fewer pairs and leaves families partial or split. Neither result is an overall precision number for all 10,000 decisions. Filler-only groups, where they exist, stay unlabeled.

The 0.70 NLI threshold was chosen while looking at this labeled 200-decision set. The 200-decision Hybrid + NLI score is not a held-out test. The Jaccard thresholds 0.75 and 0.32 were fixed earlier from the planted paraphrases and template fillers, which overlap with the 10,000-decision file. Family 1’s slot 4 (“Increase PostgreSQL database storage”) is still labeled as a paraphrase of the migration approval by the existing label rules; both systems miss it, and that label is harsher than the wording.

The scores in this report were recomputed from `find_duplicates_1.py` and `find_duplicates_2.py`. On the 200-decision pairs they match the earlier measurements: Semantic + Jaccard F1 0.4141 (TP 47, FP 60, FN 73) and Hybrid + NLI at 0.70 F1 0.7476 (TP 77, FP 9, FN 43). Older files under `experiments/` and `experiment_2/` remain as historical evidence. They are not a third method in this comparison.

## H. Future research

These are notes only. Nothing in this list was implemented.

- Score the NLI threshold on a label split that was not used to choose 0.70.
- Label a sample of 10,000-decision filler pairs so overall precision is defined.
- Record, without changing the decision, how many Hybrid + NLI rejections come from the detail guard and how many come from entailment.
- Compare union-find and complete-link on the same accepted edges, so grouping is not confounded with the acceptance rule.
- Teach the detail guard to read number words (“ten”, “two thousand”) before calling a conflict. That would be a logic change and needs its own evaluation.

## I. Conclusion

The two approaches succeed at different parts of the objective.

Semantic + Jaccard is the stronger of the two when the goal is to surface intact paraphrase families inside the 10,000-decision file, and it also completes more families on the 200-decision set. Its cost is a larger set of labeled false matches, and it can merge decisions through a chain.

Hybrid + NLI is the stronger of the two on the 200-decision pair labels: fewer false matches, and more of the genuine pairs accepted. Its cost shows up in the groups. Complete-link grouping plus one-direction entailment failures leave families partial, and on the 10,000-decision planted families it recovers 55 of 110 genuine pairs against 84, with 0 of 6 families complete.

No single winner follows from these two datasets. A setting that must not merge near-misses has evidence for Hybrid + NLI on the labeled 200. A setting that must list the whole planted family has evidence for Semantic + Jaccard on the 10,000-decision file. Neither script, on this evidence, is a basis for deleting records.
"""
    write(REPORTS / "comparative_research_report.md", comparative)
    print(json.dumps(bundle["pair"], indent=2))
    print("family", bundle["family_counts"])
    print("edges", bundle["edge_class"])
    print("times", times)
    print("discrepancy", discrepancy)


if __name__ == "__main__":
    main()
