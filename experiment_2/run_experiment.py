"""Experiment 2: cosine vs Jaccard vs critical decision checks.

Reuses Experiment 1 pair_scores.csv (same 116 pairs and saved scores).
Does not modify production code, the original experiment, or the database.

The critical-check rules were written from the error types seen in Experiment 1,
so the 116-pair result is exploratory and may be biased. A small held-out set
is scored separately and is not mixed into the main table.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from embeddings import get_embedder  # noqa: E402
from experiment_2.critical_checks import critical_conflict  # noqa: E402

EXP1_CSV = ROOT / "experiments" / "pair_scores.csv"
OUT_DIR = Path(__file__).resolve().parent
COSINE_THRESHOLD = 0.75
JACCARD_CUTOFF = 0.32

# New pairs, not copied from Experiment 1. Labels come from meaning, not scores.
HELD_OUT = [
    {"category": "A", "a": "The board signed off on Postgres as the system of record.", "b": "PostgreSQL will be our main database.", "expected": True},
    {"category": "A", "a": "Staff may work remotely three days a week.", "b": "The hybrid schedule allows three work-from-home days weekly.", "expected": True},
    {"category": "A", "a": "We will increase prices 10 percent next quarter.", "b": "A ten percent price rise starts next quarter.", "expected": True},
    {"category": "B", "a": "The security team approved quarterly key rotation.", "b": "Quarterly key rotation was approved by the security team.", "expected": True},
    {"category": "B", "a": "We will keep customer logs for 90 days.", "b": "Customer logs will be kept for 90 days.", "expected": True},
    {"category": "C", "a": "We approved the annual budget.", "b": "We rejected the annual budget.", "expected": False},
    {"category": "C", "a": "Enable audit logging for production.", "b": "Disable audit logging for production.", "expected": False},
    {"category": "C", "a": "Continue the pilot with vendor Northwind.", "b": "Stop the pilot with vendor Northwind.", "expected": False},
    {"category": "D", "a": "Hire 4 analysts in Berlin.", "b": "Hire 9 analysts in Tokyo.", "expected": False},
    {"category": "D", "a": "Pay invoice INV-14 for $2,000.", "b": "Pay invoice INV-14 for $9,000.", "expected": False},
    {"category": "D", "a": "Close ticket T-10 after the customer confirms.", "b": "Close ticket T-11 after the customer confirms.", "expected": False},
    {"category": "D", "a": "Launch the billing change in April.", "b": "Launch the billing change in October.", "expected": False},
    {"category": "E", "a": "Add a read replica for the billing database.", "b": "Add an index to the billing database.", "expected": False},
    {"category": "E", "a": "Document the refund policy.", "b": "Change the refund policy.", "expected": False},
]


def as_bool(value: str) -> bool:
    return str(value).strip().lower() == "true"


def load_primary_pairs() -> list[dict]:
    rows = []
    with EXP1_CSV.open(encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            if as_bool(raw["ambiguous"]):
                continue
            cosine = float(raw["cosine"])
            jaccard = float(raw["jaccard"])
            texts_equal = as_bool(raw["texts_equal"])
            semantic = cosine >= COSINE_THRESHOLD
            jaccard_pred = semantic and (texts_equal or jaccard <= JACCARD_CUTOFF)
            if semantic != as_bool(raw["semantic_only"]) or jaccard_pred != as_bool(raw["semantic_plus_jaccard"]):
                raise RuntimeError(f"Saved Experiment 1 prediction mismatch for: {raw['text_a']}")
            rows.append({
                "category": raw["category"],
                "text_a": raw["text_a"],
                "text_b": raw["text_b"],
                "expected": as_bool(raw["expected"]),
                "cosine": cosine,
                "jaccard": jaccard,
                "texts_equal": texts_equal,
                "semantic_only": semantic,
                "semantic_plus_jaccard": jaccard_pred,
            })
    return rows


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
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "n": len(rows),
    }


def apply_checks(rows: list[dict]) -> None:
    for row in rows:
        check = critical_conflict(row["text_a"], row["text_b"])
        row["critical_status"] = check["status"]
        row["critical_reason"] = check["reason"]
        row["semantic_plus_critical"] = bool(row["semantic_only"]) and check["status"] != "conflict"


def md_table(headers: list[str], body: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in body:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def esc(text: str) -> str:
    return text.replace("|", "/")


def pair_table(rows: list[dict]) -> str:
    if not rows:
        return "_No examples in this set._"
    body = []
    for row in rows:
        body.append([
            row["category"],
            esc(row["text_a"]),
            esc(row["text_b"]),
            "duplicate" if row["expected"] else "not duplicate",
            f"{row['cosine']:.4f}",
            f"{row['jaccard']:.4f}",
            "accept" if row["semantic_only"] else "reject",
            "accept" if row["semantic_plus_jaccard"] else "reject",
            "accept" if row["semantic_plus_critical"] else "reject",
            esc(row["critical_reason"]),
        ])
    return md_table(
        ["Cat", "Decision A", "Decision B", "Expected", "Cosine", "Jaccard", "A", "B", "C", "Critical reason"],
        body,
    )


def score_held_out() -> list[dict]:
    texts = sorted({item["a"] for item in HELD_OUT} | {item["b"] for item in HELD_OUT})
    embedder = get_embedder()
    matrix = np.asarray(embedder.embed(texts), dtype=np.float32)
    norms = np.clip(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12, None)
    matrix = matrix / norms
    index = {text: i for i, text in enumerate(texts)}
    sys.path.insert(0, str(ROOT))
    from find_duplicates import jaccard_sim, normalize, tokens  # noqa: E402

    rows = []
    for item in HELD_OUT:
        cosine = float(matrix[index[item["a"]]] @ matrix[index[item["b"]]])
        jaccard = float(jaccard_sim(tokens(item["a"]), tokens(item["b"])))
        equal = normalize(item["a"]) == normalize(item["b"])
        rows.append({
            "category": item["category"],
            "text_a": item["a"],
            "text_b": item["b"],
            "expected": item["expected"],
            "cosine": cosine,
            "jaccard": jaccard,
            "texts_equal": equal,
            "semantic_only": cosine >= COSINE_THRESHOLD,
            "semantic_plus_jaccard": cosine >= COSINE_THRESHOLD and (equal or jaccard <= JACCARD_CUTOFF),
        })
    apply_checks(rows)
    return rows


def main() -> None:
    rows = load_primary_pairs()
    if len(rows) != 116:
        raise RuntimeError(f"Expected 116 primary pairs, found {len(rows)}")
    apply_checks(rows)

    overall = {
        "semantic_only": metrics(rows, "semantic_only"),
        "semantic_plus_jaccard": metrics(rows, "semantic_plus_jaccard"),
        "semantic_plus_critical": metrics(rows, "semantic_plus_critical"),
    }
    by_category = {}
    for category in ["A", "B", "C", "D", "E"]:
        subset = [row for row in rows if row["category"] == category]
        by_category[category] = {
            "n": len(subset),
            "semantic_only": metrics(subset, "semantic_only"),
            "semantic_plus_jaccard": metrics(subset, "semantic_plus_jaccard"),
            "semantic_plus_critical": metrics(subset, "semantic_plus_critical"),
        }

    semantic_fp = [row for row in rows if (not row["expected"]) and row["semantic_only"]]
    removed_fp = [row for row in semantic_fp if not row["semantic_plus_critical"]]
    remaining_fp = [row for row in semantic_fp if row["semantic_plus_critical"]]
    genuine = [row for row in rows if row["expected"]]
    preserved = [row for row in genuine if row["semantic_plus_critical"]]
    rejected_genuine = [row for row in genuine if not row["semantic_plus_critical"]]
    recovered = [
        row for row in genuine
        if (not row["semantic_plus_jaccard"]) and row["semantic_plus_critical"]
    ]
    critical_fn_only = [
        row for row in genuine
        if row["semantic_only"] and (not row["semantic_plus_critical"])
    ]

    held_out = score_held_out()
    held_metrics = {
        "semantic_only": metrics(held_out, "semantic_only"),
        "semantic_plus_jaccard": metrics(held_out, "semantic_plus_jaccard"),
        "semantic_plus_critical": metrics(held_out, "semantic_plus_critical"),
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / "pair_predictions.csv"
    fields = [
        "category", "text_a", "text_b", "expected", "cosine", "jaccard",
        "semantic_only", "semantic_plus_jaccard", "semantic_plus_critical",
        "critical_status", "critical_reason",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "source": str(EXP1_CSV),
        "primary_pairs": len(rows),
        "note": (
            "Exploratory. Critical-check rules were designed after seeing Experiment 1 "
            "error types, so the 116-pair numbers may be biased. Rules were not retuned "
            "after this run. Held-out pairs are separate and small."
        ),
        "overall": overall,
        "by_category": by_category,
        "semantic_false_positives": len(semantic_fp),
        "false_positives_removed_by_critical_checks": len(removed_fp),
        "genuine_duplicates": len(genuine),
        "genuine_duplicates_preserved_by_critical_checks": len(preserved),
        "genuine_duplicates_rejected_by_critical_checks": len(rejected_genuine),
        "genuine_duplicates_recovered_versus_jaccard": len(recovered),
        "false_positives_still_passing_critical_checks": len(remaining_fp),
        "held_out_pairs": len(held_out),
        "held_out": held_metrics,
    }
    json_path = OUT_DIR / "summary.json"
    json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    a, b, c = overall["semantic_only"], overall["semantic_plus_jaccard"], overall["semantic_plus_critical"]

    def metric_row(name: str, item: dict) -> list[str]:
        return [name, f"{item['precision']:.4f}", f"{item['recall']:.4f}", f"{item['f1']:.4f}", str(item["fp"]), str(item["fn"])]

    cat_body = []
    for category in ["A", "B", "C", "D", "E"]:
        block = by_category[category]
        aa, bb, cc = block["semantic_only"], block["semantic_plus_jaccard"], block["semantic_plus_critical"]
        cat_body.append([
            category, str(block["n"]),
            str(aa["fp"]), str(aa["fn"]), f"{aa['f1']:.4f}",
            str(bb["fp"]), str(bb["fn"]), f"{bb['f1']:.4f}",
            str(cc["fp"]), str(cc["fn"]), f"{cc['f1']:.4f}",
        ])

    held_body = [[
        "Held-out",
        f"{held_metrics['semantic_only']['precision']:.4f}",
        f"{held_metrics['semantic_only']['recall']:.4f}",
        f"{held_metrics['semantic_only']['f1']:.4f}",
        f"{held_metrics['semantic_plus_jaccard']['precision']:.4f}",
        f"{held_metrics['semantic_plus_jaccard']['recall']:.4f}",
        f"{held_metrics['semantic_plus_jaccard']['f1']:.4f}",
        f"{held_metrics['semantic_plus_critical']['precision']:.4f}",
        f"{held_metrics['semantic_plus_critical']['recall']:.4f}",
        f"{held_metrics['semantic_plus_critical']['f1']:.4f}",
    ]]

    report = f"""# Experiment 2 — critical decision checks

Exploratory evaluation. The rules were written after seeing the kinds of errors in Experiment 1 (opposites, template values, entities). They were not retuned after this run, but the 116-pair result can still be biased. The held-out sentences are new, small, and written in the same session, so they are only a weak generalization check.

Scores for the 116 pairs are reused from `experiments/pair_scores.csv`. Embeddings were not recomputed for those pairs.

## Main comparison

{md_table(
    ["Metric", "Semantic only", "Semantic + Jaccard", "Semantic + critical checks"],
    [
        ["Precision", f"{a['precision']:.4f}", f"{b['precision']:.4f}", f"{c['precision']:.4f}"],
        ["Recall", f"{a['recall']:.4f}", f"{b['recall']:.4f}", f"{c['recall']:.4f}"],
        ["F1", f"{a['f1']:.4f}", f"{b['f1']:.4f}", f"{c['f1']:.4f}"],
        ["False positives", str(a["fp"]), str(b["fp"]), str(c["fp"])],
        ["False negatives", str(a["fn"]), str(b["fn"]), str(c["fn"])],
        ["True positives", str(a["tp"]), str(b["tp"]), str(c["tp"])],
        ["True negatives", str(a["tn"]), str(b["tn"]), str(c["tn"])],
    ],
)}

## By category

{md_table(
    ["Category", "N", "A FP", "A FN", "A F1", "B FP", "B FN", "B F1", "C FP", "C FN", "C F1"],
    cat_body,
)}

## What changed

- Semantic-only false positives: **{len(semantic_fp)}**
- Of those, critical checks removed: **{len(removed_fp)}**
- Genuine duplicates in the set: **{len(genuine)}**
- Genuine duplicates preserved by critical checks: **{len(preserved)}**
- Genuine duplicates rejected by critical checks: **{len(rejected_genuine)}**
- Genuine duplicates rejected by Jaccard but kept by critical checks: **{len(recovered)}**
- Semantic false positives that still pass critical checks: **{len(remaining_fp)}**

## Incorrect matches removed by critical checks

{pair_table(removed_fp)}

## Genuine duplicates kept by critical checks but rejected by Jaccard

{pair_table(recovered)}

## Genuine duplicates incorrectly rejected by critical checks

{pair_table(critical_fn_only)}

## Incorrect matches that still pass critical checks

{pair_table(remaining_fp)}

## Held-out check ({len(held_out)} new pairs)

{md_table(
    ["Set", "A prec", "A rec", "A F1", "B prec", "B rec", "B F1", "C prec", "C rec", "C F1"],
    held_body,
)}

{pair_table(held_out)}

## Unsupported cases

- Abbreviations and synonyms are not understood unless they share surface words or a listed action word.
- "Phase out" is handled; many other multi-word actions are not.
- A number is compared only when its unit is recognized (money, percent, headcount, duration). Bare numbers are ignored.
- "End of year" is treated as compatible with December, not as a conflict.
- If action polarity is ambiguous inside one sentence, the pair is marked unknown and not rejected.
- Entity detection covers ID prefixes, capitalized places after "in/at", plan/team/level labels, and IP addresses. Other names can be missed.
"""
    report_path = OUT_DIR / "report.md"
    report_path.write_text(report, encoding="utf-8")
    print(report)
    print(f"\nWrote {csv_path}")
    print(f"Wrote {json_path}")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
