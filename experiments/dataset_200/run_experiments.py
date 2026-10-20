"""Rerun Experiments 1 and 2 on the 200-decision dataset.

The source file is 200 decisions in 40 groups of 5, with no gold labels.
After reading every group, pairs are labeled from meaning using this structure:

  0  base decision
  1  paraphrase of the base
  2  conflicting decision (opposite action, or a different amount/id)
  3  related but different decision, or another changed detail
  4  second paraphrase of the base

Duplicate pairs are only (0,1), (0,4), and (1,4).
Three pairs in group 1 are excluded as ambiguous:
"approved the migration" versus "we will migrate next quarter".

Labels are not assigned from cosine or Jaccard scores.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from embeddings import get_embedder  # noqa: E402
from experiment_2.critical_checks import critical_conflict  # noqa: E402
from find_duplicates import jaccard_sim, normalize, tokens  # noqa: E402

DATA_PATH = Path(__file__).resolve().parent / "experiment_decision_dataset_200.json"
OUT_DIR = Path(__file__).resolve().parent
COSINE_THRESHOLD = 0.75
JACCARD_CUTOFF = 0.32

# How slot 2 and slot 3 differ from the base decision.
# C = opposite action, D = same template with a different value/entity, E = related but different.
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


def build_pairs(decisions: list[dict]) -> list[dict]:
    if len(decisions) != 200:
        raise RuntimeError(f"Expected 200 decisions, found {len(decisions)}")
    pairs = []
    for group_index in range(40):
        group = decisions[group_index * 5:(group_index + 1) * 5]
        for left, right in combinations(range(5), 2):
            if group_index == 0 and {left, right} in ({0, 3}, {1, 3}, {4, 3}):
                category = "ambiguous"
                expected = False
                ambiguous = True
            elif {left, right} <= {0, 1, 4}:
                category = "B" if {left, right} == {0, 4} else "A"
                expected = True
                ambiguous = False
            elif {left, right} == {2, 3}:
                category = "D" if SLOT2[group_index] == "D" and SLOT3[group_index] == "D" else "E"
                expected = False
                ambiguous = False
            elif 2 in {left, right}:
                category = SLOT2[group_index]
                expected = False
                ambiguous = False
            else:
                category = SLOT3[group_index]
                expected = False
                ambiguous = False
            pairs.append({
                "group": group_index + 1,
                "category": category,
                "text_a": group[left]["text"],
                "text_b": group[right]["text"],
                "id_a": group[left]["id"],
                "id_b": group[right]["id"],
                "expected": expected,
                "ambiguous": ambiguous,
            })
    return pairs


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.clip(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12, None)
    return matrix / norms


def metrics(rows: list[dict], key: str) -> dict:
    tp = fp = tn = fn = 0
    for row in rows:
        pred = bool(row[key])
        exp = bool(row["expected"])
        if pred and exp:
            tp += 1
        elif pred and not exp:
            fp += 1
        elif (not pred) and (not exp):
            tn += 1
        else:
            fn += 1
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": round(precision, 4), "recall": round(recall, 4),
        "f1": round(f1, 4), "n": len(rows),
    }


def md_table(headers: list[str], body: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in body:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def esc(text: str) -> str:
    return text.replace("|", "/")


def sample_table(rows: list[dict], limit: int = 12) -> str:
    if not rows:
        return "_No examples in this set._"
    shown = rows[:limit]
    body = [[
        row["category"], row["id_a"], esc(row["text_a"]), row["id_b"], esc(row["text_b"]),
        "duplicate" if row["expected"] else "not duplicate",
        f"{row['cosine']:.4f}", f"{row['jaccard']:.4f}",
        "accept" if row["semantic_only"] else "reject",
        "accept" if row["semantic_plus_jaccard"] else "reject",
        "accept" if row["semantic_plus_critical"] else "reject",
    ] for row in shown]
    note = ""
    if len(rows) > limit:
        note = f"\n\n_Showing {limit} of {len(rows)}. Full list is in pair_scores.csv._"
    return md_table(
        ["Cat", "ID A", "Decision A", "ID B", "Decision B", "Expected", "Cosine", "Jaccard", "A", "B", "C"],
        body,
    ) + note


def main() -> None:
    decisions = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    pairs = build_pairs(decisions)
    texts = [row["text"] for row in decisions]
    embedder = get_embedder()
    vectors = l2_normalize(np.asarray(embedder.embed(texts), dtype=np.float32))
    index = {row["id"]: i for i, row in enumerate(decisions)}

    for pair in pairs:
        left, right = index[pair["id_a"]], index[pair["id_b"]]
        cosine = float(vectors[left] @ vectors[right])
        jaccard = float(jaccard_sim(tokens(pair["text_a"]), tokens(pair["text_b"])))
        equal = normalize(pair["text_a"]) == normalize(pair["text_b"])
        semantic = cosine >= COSINE_THRESHOLD
        check = critical_conflict(pair["text_a"], pair["text_b"])
        pair.update({
            "cosine": round(cosine, 6),
            "jaccard": round(jaccard, 6),
            "texts_equal": equal,
            "semantic_only": semantic,
            "semantic_plus_jaccard": semantic and (equal or jaccard <= JACCARD_CUTOFF),
            "critical_status": check["status"],
            "critical_reason": check["reason"],
            "semantic_plus_critical": semantic and check["status"] != "conflict",
        })

    primary = [pair for pair in pairs if not pair["ambiguous"]]
    counts = defaultdict(int)
    for pair in primary:
        counts[pair["category"]] += 1
    overall = {
        "semantic_only": metrics(primary, "semantic_only"),
        "semantic_plus_jaccard": metrics(primary, "semantic_plus_jaccard"),
        "semantic_plus_critical": metrics(primary, "semantic_plus_critical"),
    }
    by_category = {}
    for category in ["A", "B", "C", "D", "E"]:
        subset = [pair for pair in primary if pair["category"] == category]
        by_category[category] = {
            name: metrics(subset, name)
            for name in ("semantic_only", "semantic_plus_jaccard", "semantic_plus_critical")
        }

    semantic_fp = [pair for pair in primary if (not pair["expected"]) and pair["semantic_only"]]
    removed_by_jaccard = [pair for pair in semantic_fp if not pair["semantic_plus_jaccard"]]
    removed_by_critical = [pair for pair in semantic_fp if not pair["semantic_plus_critical"]]
    genuine = [pair for pair in primary if pair["expected"]]
    preserved_critical = [pair for pair in genuine if pair["semantic_plus_critical"]]
    recovered = [pair for pair in genuine if (not pair["semantic_plus_jaccard"]) and pair["semantic_plus_critical"]]
    critical_rejected_genuine = [
        pair for pair in genuine if pair["semantic_only"] and not pair["semantic_plus_critical"]
    ]
    still_passing = [pair for pair in semantic_fp if pair["semantic_plus_critical"]]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fields = [
        "group", "category", "id_a", "id_b", "text_a", "text_b", "expected", "ambiguous",
        "cosine", "jaccard", "texts_equal", "semantic_only", "semantic_plus_jaccard",
        "semantic_plus_critical", "critical_status", "critical_reason",
    ]
    csv_path = OUT_DIR / "pair_scores.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(pairs)

    summary = {
        "dataset": str(DATA_PATH),
        "decisions": len(decisions),
        "model": getattr(embedder, "model_name", embedder.name),
        "primary_pairs": len(primary),
        "ambiguous_pairs_excluded": len(pairs) - len(primary),
        "positive_pairs": sum(1 for pair in primary if pair["expected"]),
        "negative_pairs": sum(1 for pair in primary if not pair["expected"]),
        "category_counts": dict(counts),
        "overall": overall,
        "by_category": by_category,
        "semantic_false_positives": len(semantic_fp),
        "false_positives_removed_by_jaccard": len(removed_by_jaccard),
        "false_positives_removed_by_critical_checks": len(removed_by_critical),
        "genuine_duplicates": len(genuine),
        "genuine_duplicates_preserved_by_critical_checks": len(preserved_critical),
        "genuine_duplicates_recovered_versus_jaccard": len(recovered),
        "genuine_duplicates_rejected_by_critical_checks_after_cosine": len(critical_rejected_genuine),
        "false_positives_still_passing_critical_checks": len(still_passing),
    }
    json_path = OUT_DIR / "summary.json"
    json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    a = overall["semantic_only"]
    b = overall["semantic_plus_jaccard"]
    c = overall["semantic_plus_critical"]
    cat_body = []
    for category in ["A", "B", "C", "D", "E"]:
        block = by_category[category]
        cat_body.append([
            category, str(counts[category]),
            str(block["semantic_only"]["fp"]), str(block["semantic_only"]["fn"]), f"{block['semantic_only']['f1']:.4f}",
            str(block["semantic_plus_jaccard"]["fp"]), str(block["semantic_plus_jaccard"]["fn"]), f"{block['semantic_plus_jaccard']['f1']:.4f}",
            str(block["semantic_plus_critical"]["fp"]), str(block["semantic_plus_critical"]["fn"]), f"{block['semantic_plus_critical']['f1']:.4f}",
        ])
    report = f"""# Experiments on the 200-decision dataset

Source: `experiments/dataset_200/experiment_decision_dataset_200.json` ({len(decisions)} decisions).

Pairs are formed inside each group of five. Gold labels come from the group structure after reading all 40 groups, not from model scores. Three ambiguous pairs in group 1 are excluded.

- Primary pairs: **{len(primary)}** ({summary["positive_pairs"]} duplicates, {summary["negative_pairs"]} not duplicates)
- Model: `{summary["model"]}`

## Comparison

{md_table(
    ["Metric", "Semantic only", "Semantic + Jaccard", "Semantic + critical checks"],
    [
        ["Precision", f"{a['precision']:.4f}", f"{b['precision']:.4f}", f"{c['precision']:.4f}"],
        ["Recall", f"{a['recall']:.4f}", f"{b['recall']:.4f}", f"{c['recall']:.4f}"],
        ["F1", f"{a['f1']:.4f}", f"{b['f1']:.4f}", f"{c['f1']:.4f}"],
        ["False positives", str(a["fp"]), str(b["fp"]), str(c["fp"])],
        ["False negatives", str(a["fn"]), str(b["fn"]), str(c["fn"])],
    ],
)}

## By category

{md_table(
    ["Category", "N", "A FP", "A FN", "A F1", "B FP", "B FN", "B F1", "C FP", "C FN", "C F1"],
    cat_body,
)}

## Counts

- Semantic-only false positives: **{len(semantic_fp)}**
- Removed by Jaccard: **{len(removed_by_jaccard)}**
- Removed by critical checks: **{len(removed_by_critical)}**
- Genuine duplicates: **{len(genuine)}**
- Preserved by critical checks: **{len(preserved_critical)}**
- Rejected by Jaccard but kept by critical checks: **{len(recovered)}**
- Genuine duplicates that passed cosine and were rejected by critical checks: **{len(critical_rejected_genuine)}**
- Semantic false positives that still pass critical checks: **{len(still_passing)}**

## Examples: false positives removed by critical checks

{sample_table(removed_by_critical)}

## Examples: genuine duplicates kept by critical checks but rejected by Jaccard

{sample_table(recovered)}

## Genuine duplicates rejected by critical checks

{sample_table(critical_rejected_genuine)}

## Examples: incorrect matches that still pass critical checks

{sample_table(still_passing)}
"""
    report_path = OUT_DIR / "report.md"
    report_path.write_text(report, encoding="utf-8")
    print(report)
    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
