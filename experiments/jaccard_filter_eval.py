"""
Pair-level experiment: cosine-only vs cosine + Jaccard.

Does not modify production detection, the database, or data/decisions.json.
Reuses find_duplicates.normalize / tokens / jaccard_sim and embeddings.get_embedder.
Cosine uses L2-normalized embeddings (same as DecisionStore.load_all) and a dot product.
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
from find_duplicates import jaccard_sim, normalize, tokens  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent
COSINE_THRESHOLD = 0.75
JACCARD_CUTOFF = 0.32
JACCARD_SWEEP = [None, 0.20, 0.25, 0.32, 0.40, 0.55]

# expected: True = duplicate, False = not a duplicate.
# ambiguous pairs are stored and reported, but excluded from primary metrics.
PAIRS: list[dict] = [
    # --- A: same meaning, different wording ---
    {"category": "A", "a": "We decided to ship the product on Friday.", "b": "Release date is set for Friday.", "expected": True},
    {"category": "A", "a": "We're launching this Friday.", "b": "Ship date locked to Friday.", "expected": True},
    {"category": "A", "a": "Friday is the go-live day for the product.", "b": "We'll put the product out on Friday.", "expected": True},
    {"category": "A", "a": "Agreed: product ships Friday.", "b": "Decision: Friday launch for the product.", "expected": True},
    {"category": "A", "a": "Use PostgreSQL as our primary database.", "b": "Postgres is the chosen database.", "expected": True},
    {"category": "A", "a": "We're standardizing on PostgreSQL for storage.", "b": "Primary datastore will be Postgres.", "expected": True},
    {"category": "A", "a": "Decision: PostgreSQL for the main DB.", "b": "We picked Postgres over MySQL for the database.", "expected": True},
    {"category": "A", "a": "Adopt PostgreSQL as the database of record.", "b": "Postgres is the chosen database.", "expected": True},
    {"category": "A", "a": "Remote work is allowed three days per week.", "b": "Hybrid policy: 3 days WFH each week.", "expected": True},
    {"category": "A", "a": "Employees may work from home three days weekly.", "b": "WFH allowance set at 3 days/week.", "expected": True},
    {"category": "A", "a": "Approved: three remote days per week.", "b": "Three days remote, two in office — that's the policy.", "expected": True},
    {"category": "A", "a": "Raise prices by 10% starting next quarter.", "b": "Approved a 10 percent price increase for Q+1.", "expected": True},
    {"category": "A", "a": "Next quarter we increase pricing 10%.", "b": "We'll bump prices ten percent next quarter.", "expected": True},
    {"category": "A", "a": "Pricing goes up 10% beginning next quarter.", "b": "Decision: 10% price hike effective next quarter.", "expected": True},
    {"category": "A", "a": "Deprecate the legacy API by end of year.", "b": "Legacy API sunset date: December 31.", "expected": True},
    {"category": "A", "a": "We're retiring the old API before year-end.", "b": "Phase out the outdated API by December.", "expected": True},
    {"category": "A", "a": "End-of-year deadline to shut down the legacy API.", "b": "Legacy endpoints will be removed by EOY.", "expected": True},
    {"category": "A", "a": "Hire two backend engineers this quarter.", "b": "We're opening two backend eng roles this Q.", "expected": True},
    {"category": "A", "a": "Approved headcount: 2 backend engineers.", "b": "Recruit two backend developers this quarter.", "expected": True},
    {"category": "A", "a": "Decision: add two backend engineers ASAP this quarter.", "b": "Hire two backend engineers this quarter.", "expected": True},
    {"category": "A", "a": "Freeze hiring for the design team.", "b": "Stop recruiting designers until further notice.", "expected": True},
    {"category": "A", "a": "Move the support queue to Zendesk.", "b": "Support tickets will be handled in Zendesk.", "expected": True},
    {"category": "A", "a": "Require two-factor authentication for admins.", "b": "Admin accounts must use MFA.", "expected": True},
    {"category": "A", "a": "Cut the marketing budget by twenty percent.", "b": "Marketing spend will be reduced 20%.", "expected": True},
    {"category": "A", "a": "Ship the mobile app in March.", "b": "The mobile application goes live in March.", "expected": True},
    {"category": "A", "a": "Cancel the vendor contract with Acme.", "b": "We will not renew Acme's contract.", "expected": True},
    {"category": "A", "a": "Store customer data in the EU region.", "b": "Customer records must stay in Europe.", "expected": True},
    {"category": "A", "a": "Switch billing from monthly to annual.", "b": "Invoices will be yearly instead of every month.", "expected": True},

    # --- B: same meaning, similar wording ---
    {"category": "B", "a": "We approved the migration to PostgreSQL.", "b": "The migration to PostgreSQL was approved.", "expected": True},
    {"category": "B", "a": "We decided to ship the product on Friday.", "b": "We'll put the product out on Friday.", "expected": True},
    {"category": "B", "a": "Decision: Friday launch for the product.", "b": "Friday is the go-live day for the product.", "expected": True},
    {"category": "B", "a": "Raise prices by 10% starting next quarter.", "b": "Pricing goes up 10% beginning next quarter.", "expected": True},
    {"category": "B", "a": "Deprecate the legacy API by end of year.", "b": "We're retiring the old API before year-end.", "expected": True},
    {"category": "B", "a": "Hire two backend engineers this quarter.", "b": "Recruit two backend developers this quarter.", "expected": True},
    {"category": "B", "a": "Remote work is allowed three days per week.", "b": "Employees may work from home three days weekly.", "expected": True},
    {"category": "B", "a": "Use PostgreSQL as our primary database.", "b": "Adopt PostgreSQL as the database of record.", "expected": True},
    {"category": "B", "a": "We will enable dark mode for the dashboard.", "b": "Dark mode will be enabled for the dashboard.", "expected": True},
    {"category": "B", "a": "The team approved a hiring freeze.", "b": "A hiring freeze was approved by the team.", "expected": True},
    {"category": "B", "a": "We chose to delay the launch until Monday.", "b": "The launch was delayed until Monday.", "expected": True},
    {"category": "B", "a": "Security approved rotating API keys quarterly.", "b": "Rotating API keys quarterly was approved by security.", "expected": True},
    {"category": "B", "a": "We agreed to keep logs for ninety days.", "b": "Logs will be kept for ninety days.", "expected": True},
    {"category": "B", "a": "Product decided to remove the onboarding wizard.", "b": "The onboarding wizard will be removed.", "expected": True},
    {"category": "B", "a": "We will migrate CI to GitHub Actions.", "b": "CI will be migrated to GitHub Actions.", "expected": True},
    {"category": "B", "a": "Finance approved the 10 percent price increase.", "b": "The 10 percent price increase was approved by finance.", "expected": True},
    {"category": "B", "a": "We decided to block public S3 buckets.", "b": "Public S3 buckets will be blocked.", "expected": True},
    {"category": "B", "a": "Leadership approved three remote days each week.", "b": "Three remote days each week were approved.", "expected": True},
    {"category": "B", "a": "We will sunset the legacy billing API.", "b": "The legacy billing API will be sunset.", "expected": True},
    {"category": "B", "a": "The board approved hiring two backend engineers.", "b": "Hiring two backend engineers was approved.", "expected": True},
    {"category": "B", "a": "We agreed the product ships on Friday.", "b": "The product ships on Friday, as agreed.", "expected": True},
    {"category": "B", "a": "We will standardize on PostgreSQL for storage.", "b": "Storage will be standardized on PostgreSQL.", "expected": True},
    {"category": "B", "a": "We decided to raise prices ten percent next quarter.", "b": "Next quarter prices will be raised ten percent.", "expected": True},
    {"category": "B", "a": "We approved closing the legacy API in December.", "b": "Closing the legacy API in December was approved.", "expected": True},

    # --- C: opposite decisions, similar wording ---
    {"category": "C", "a": "We approved the migration to PostgreSQL.", "b": "We rejected the migration to PostgreSQL.", "expected": False},
    {"category": "C", "a": "We decided to ship the product on Friday.", "b": "We decided not to ship the product on Friday.", "expected": False},
    {"category": "C", "a": "Hire two backend engineers this quarter.", "b": "Do not hire two backend engineers this quarter.", "expected": False},
    {"category": "C", "a": "Raise prices by 10% starting next quarter.", "b": "Do not raise prices by 10% starting next quarter.", "expected": False},
    {"category": "C", "a": "Remote work is allowed three days per week.", "b": "Remote work is not allowed three days per week.", "expected": False},
    {"category": "C", "a": "We will enable dark mode for the dashboard.", "b": "We will not enable dark mode for the dashboard.", "expected": False},
    {"category": "C", "a": "The team approved a hiring freeze.", "b": "The team rejected a hiring freeze.", "expected": False},
    {"category": "C", "a": "We chose to delay the launch until Monday.", "b": "We chose not to delay the launch until Monday.", "expected": False},
    {"category": "C", "a": "Security approved rotating API keys quarterly.", "b": "Security rejected rotating API keys quarterly.", "expected": False},
    {"category": "C", "a": "We agreed to keep logs for ninety days.", "b": "We agreed not to keep logs for ninety days.", "expected": False},
    {"category": "C", "a": "Product decided to remove the onboarding wizard.", "b": "Product decided to keep the onboarding wizard.", "expected": False},
    {"category": "C", "a": "We will migrate CI to GitHub Actions.", "b": "We will not migrate CI to GitHub Actions.", "expected": False},
    {"category": "C", "a": "We decided to block public S3 buckets.", "b": "We decided to allow public S3 buckets.", "expected": False},
    {"category": "C", "a": "We will sunset the legacy billing API.", "b": "We will keep the legacy billing API.", "expected": False},
    {"category": "C", "a": "Leadership approved three remote days each week.", "b": "Leadership denied three remote days each week.", "expected": False},
    {"category": "C", "a": "Finance approved the 10 percent price increase.", "b": "Finance rejected the 10 percent price increase.", "expected": False},
    {"category": "C", "a": "We will launch the mobile app in March.", "b": "We will postpone the mobile app past March.", "expected": False},
    {"category": "C", "a": "Cancel the vendor contract with Acme.", "b": "Renew the vendor contract with Acme.", "expected": False},
    {"category": "C", "a": "Freeze hiring for the design team.", "b": "Continue hiring for the design team.", "expected": False},
    {"category": "C", "a": "Require two-factor authentication for admins.", "b": "Two-factor authentication for admins is optional.", "expected": False},

    # --- D: same template, different important values ---
    {"category": "D", "a": "Close office LOC-92 in Dhaka.", "b": "Close office LOC-47 in Perth.", "expected": False},
    {"category": "D", "a": "Award contract CONT-45 to vendor VND-45 for $775k.", "b": "Award contract CONT-60 to vendor VND-60 for $130k.", "expected": False},
    {"category": "D", "a": "Set customer CUST-91 discount to 24% through 8/2025.", "b": "Set customer CUST-12 discount to 5% through 1/2024.", "expected": False},
    {"category": "D", "a": "Promote employee EMP-3813 to level L6 on team foxtrot.", "b": "Promote employee EMP-102 to level L2 on team alpha.", "expected": False},
    {"category": "D", "a": "Cap wallet WAL-4026 spend at $52/day.", "b": "Cap wallet WAL-88 spend at $900/day.", "expected": False},
    {"category": "D", "a": "Schedule maintenance WIN-760 on host HST-760 at 16:00 UTC.", "b": "Schedule maintenance WIN-12 on host HST-12 at 02:00 UTC.", "expected": False},
    {"category": "D", "a": "Buy domain DOM-3779.example for campaign CMP-3779.", "b": "Buy domain DOM-15.example for campaign CMP-15.", "expected": False},
    {"category": "D", "a": "Ship hotfix HF-42 for bug BUG-42 tonight.", "b": "Ship hotfix HF-99 for bug BUG-99 tonight.", "expected": False},
    {"category": "D", "a": "Reject RFC RFC-41 about feature FEAT-41.", "b": "Reject RFC RFC-900 about feature FEAT-900.", "expected": False},
    {"category": "D", "a": "Grant role ROLE-8 to service account SA-8 only.", "b": "Grant role ROLE-77 to service account SA-77 only.", "expected": False},
    {"category": "D", "a": "Close office LOC-1 in Lisbon effective 1/2.", "b": "Close office LOC-2 in Oslo effective 6/15.", "expected": False},
    {"category": "D", "a": "Raise the price of plan BASIC by $5.", "b": "Raise the price of plan ENTERPRISE by $50.", "expected": False},
    {"category": "D", "a": "Hire 2 backend engineers in Q1.", "b": "Hire 8 frontend engineers in Q4.", "expected": False},
    {"category": "D", "a": "Set the SLA for P1 tickets to 1 hour.", "b": "Set the SLA for P3 tickets to 3 days.", "expected": False},
    {"category": "D", "a": "Route region REG-1 traffic to cluster CLS-1.", "b": "Route region REG-9 traffic to cluster CLS-9.", "expected": False},
    {"category": "D", "a": "Archive channel CH-10 and move history to vault VT-10.", "b": "Archive channel CH-88 and move history to vault VT-88.", "expected": False},
    {"category": "D", "a": "Invoice customer CUST-3 for $1200 in March.", "b": "Invoice customer CUST-40 for $80 in November.", "expected": False},
    {"category": "D", "a": "Rotate API key KEY-1 every 30 days.", "b": "Rotate API key KEY-9 every 365 days.", "expected": False},
    {"category": "D", "a": "Block IP range 10.0.1.0/24 after incident INC-1.", "b": "Block IP range 10.9.9.0/24 after incident INC-9.", "expected": False},
    {"category": "D", "a": "Rename project PRJ-65 to codename onyx65.", "b": "Rename project PRJ-2 to codename cedar2.", "expected": False},
    {"category": "D", "a": "The launch date is Friday March 1.", "b": "The launch date is Monday June 20.", "expected": False},
    {"category": "D", "a": "Discount for CUST-1 is 10 percent.", "b": "Discount for CUST-2 is 40 percent.", "expected": False},
    {"category": "D", "a": "Office in Austin closes on May 1.", "b": "Office in Pune closes on December 12.", "expected": False},
    {"category": "D", "a": "Budget for campaign CMP-3 is $10k.", "b": "Budget for campaign CMP-8 is $500k.", "expected": False},

    # --- E: related topic, different decision ---
    {"category": "E", "a": "Migrate the database to PostgreSQL.", "b": "Increase PostgreSQL database storage.", "expected": False},
    {"category": "E", "a": "Use PostgreSQL as our primary database.", "b": "Add a read replica for PostgreSQL.", "expected": False},
    {"category": "E", "a": "Hire two backend engineers this quarter.", "b": "Hire two designers this quarter.", "expected": False},
    {"category": "E", "a": "Raise prices by 10% starting next quarter.", "b": "Offer a 10% discount next quarter.", "expected": False},
    {"category": "E", "a": "Remote work is allowed three days per week.", "b": "Office snacks will be restocked three days per week.", "expected": False},
    {"category": "E", "a": "Deprecate the legacy API by end of year.", "b": "Document the legacy API by end of year.", "expected": False},
    {"category": "E", "a": "Ship the product on Friday.", "b": "Announce the product on Friday.", "expected": False},
    {"category": "E", "a": "Require MFA for admin accounts.", "b": "Require MFA training for admin accounts.", "expected": False},
    {"category": "E", "a": "Block public S3 buckets.", "b": "Audit public S3 buckets.", "expected": False},
    {"category": "E", "a": "Move CI to GitHub Actions.", "b": "Add a GitHub Actions badge to the README.", "expected": False},
    {"category": "E", "a": "Keep logs for 90 days.", "b": "Export logs to the data warehouse.", "expected": False},
    {"category": "E", "a": "Sunset the old billing API.", "b": "Add a new field to the billing API.", "expected": False},
    {"category": "E", "a": "Approve headcount for two backend engineers.", "b": "Approve laptops for two backend engineers.", "expected": False},
    {"category": "E", "a": "Standardize on PostgreSQL.", "b": "Upgrade PostgreSQL to the next minor version.", "expected": False},
    {"category": "E", "a": "Three days remote each week.", "b": "Three days of on-call each week.", "expected": False},
    {"category": "E", "a": "Launch on Friday.", "b": "Run a load test on Friday.", "expected": False},
    {"category": "E", "a": "Increase prices 10 percent.", "b": "Increase support staffing 10 percent.", "expected": False},
    {"category": "E", "a": "Close the Dhaka office.", "b": "Renovate the Dhaka office.", "expected": False},
    {"category": "E", "a": "Cancel the Acme contract.", "b": "Renegotiate the Acme contract.", "expected": False},
    {"category": "E", "a": "Enable dark mode.", "b": "Fix the dark mode contrast bug.", "expected": False},

    # Flagged for manual review. Not used in primary metrics.
    {"category": "ambiguous", "a": "Ship the product on Friday.", "b": "Release a beta on Friday.", "expected": False, "ambiguous": True, "note": "Could be the same launch or only a beta."},
    {"category": "ambiguous", "a": "Deprecate the legacy API by December.", "b": "Phase out one legacy endpoint by December.", "expected": False, "ambiguous": True, "note": "Partial overlap of the same retirement."},
    {"category": "ambiguous", "a": "Hire two backend engineers.", "b": "Hire backend engineers this quarter.", "expected": True, "ambiguous": True, "note": "Quantity is explicit in only one sentence."},
    {"category": "ambiguous", "a": "Use PostgreSQL as primary database.", "b": "Use PostgreSQL for analytics only.", "expected": False, "ambiguous": True, "note": "Same system, different scope."},
]


def l2_normalize(mat: np.ndarray) -> np.ndarray:
    """Same normalization DecisionStore.load_all applies before cosine."""
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    return mat / norms


def accept_semantic(cosine: float) -> bool:
    return cosine >= COSINE_THRESHOLD


def accept_with_jaccard(cosine: float, jaccard: float, texts_equal: bool, cutoff: float | None) -> bool:
    """Production rule. cutoff=None means cosine only."""
    if cosine < COSINE_THRESHOLD:
        return False
    if cutoff is None:
        return True
    return texts_equal or jaccard <= cutoff


def metrics(rows: list[dict], pred_key: str) -> dict:
    tp = fp = tn = fn = 0
    for row in rows:
        pred = bool(row[pred_key])
        exp = bool(row["expected"])
        if pred and exp:
            tp += 1
        elif pred and not exp:
            fp += 1
        elif not pred and not exp:
            tn += 1
        else:
            fn += 1
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "n": len(rows),
    }


def fmt(n: float) -> str:
    return f"{n:.4f}"


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def pair_table(rows: list[dict]) -> str:
    if not rows:
        return "_No examples in this set._"
    body = []
    for r in rows:
        body.append([
            r["category"],
            r["text_a"].replace("|", "/"),
            r["text_b"].replace("|", "/"),
            "duplicate" if r["expected"] else "not duplicate",
            f"{r['cosine']:.4f}",
            f"{r['jaccard']:.4f}",
            "accept" if r["semantic_only"] else "reject",
            "accept" if r["semantic_plus_jaccard"] else "reject",
        ])
    return md_table(
        ["Cat", "Decision A", "Decision B", "Expected", "Cosine", "Jaccard", "Semantic only", "Semantic + Jaccard"],
        body,
    )


def main() -> None:
    labeled = [p for p in PAIRS if not p.get("ambiguous")]
    ambiguous = [p for p in PAIRS if p.get("ambiguous")]
    counts = defaultdict(int)
    for p in labeled:
        counts[p["category"]] += 1

    texts = sorted({p["a"] for p in PAIRS} | {p["b"] for p in PAIRS})
    embedder = get_embedder()
    raw = np.asarray(embedder.embed(texts), dtype=np.float32)
    vecs = l2_normalize(raw)
    index = {t: i for i, t in enumerate(texts)}

    scored: list[dict] = []
    for p in PAIRS:
        ia, ib = index[p["a"]], index[p["b"]]
        cosine = float(vecs[ia] @ vecs[ib])
        jac = float(jaccard_sim(tokens(p["a"]), tokens(p["b"])))
        equal = normalize(p["a"]) == normalize(p["b"])
        sem = accept_semantic(cosine)
        both = accept_with_jaccard(cosine, jac, equal, JACCARD_CUTOFF)
        row = {
            "category": p["category"],
            "text_a": p["a"],
            "text_b": p["b"],
            "expected": bool(p["expected"]),
            "ambiguous": bool(p.get("ambiguous")),
            "note": p.get("note", ""),
            "cosine": round(cosine, 6),
            "jaccard": round(jac, 6),
            "texts_equal": equal,
            "semantic_only": sem,
            "semantic_plus_jaccard": both,
        }
        scored.append(row)

    primary = [r for r in scored if not r["ambiguous"]]
    amb_rows = [r for r in scored if r["ambiguous"]]

    overall_a = metrics(primary, "semantic_only")
    overall_b = metrics(primary, "semantic_plus_jaccard")

    by_cat = {}
    for cat in ["A", "B", "C", "D", "E"]:
        subset = [r for r in primary if r["category"] == cat]
        by_cat[cat] = {
            "semantic_only": metrics(subset, "semantic_only"),
            "semantic_plus_jaccard": metrics(subset, "semantic_plus_jaccard"),
        }

    removed_fp = [
        r for r in primary
        if (not r["expected"]) and r["semantic_only"] and (not r["semantic_plus_jaccard"])
    ]
    rejected_tp = [
        r for r in primary
        if r["expected"] and r["semantic_only"] and (not r["semantic_plus_jaccard"])
    ]
    remaining_fp = [
        r for r in primary
        if (not r["expected"]) and r["semantic_only"] and r["semantic_plus_jaccard"]
    ]
    semantic_fp = [r for r in primary if (not r["expected"]) and r["semantic_only"]]

    sweep = []
    for cutoff in JACCARD_SWEEP:
        preds = []
        for r in primary:
            pred = accept_with_jaccard(r["cosine"], r["jaccard"], r["texts_equal"], cutoff)
            preds.append({**r, "pred": pred})
        m = metrics(preds, "pred")
        m["jaccard_cutoff"] = "none" if cutoff is None else cutoff
        sweep.append(m)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / "pair_scores.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        fields = [
            "category", "text_a", "text_b", "expected", "ambiguous", "note",
            "cosine", "jaccard", "texts_equal", "semantic_only", "semantic_plus_jaccard",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(scored)

    summary = {
        "model": getattr(embedder, "model_name", embedder.name),
        "embedder": embedder.name,
        "dim": int(embedder.dim),
        "cosine_threshold": COSINE_THRESHOLD,
        "jaccard_cutoff": JACCARD_CUTOFF,
        "normalization": "L2, same as DecisionStore.load_all; cosine = dot product",
        "jaccard": "find_duplicates.jaccard_sim on find_duplicates.tokens",
        "primary_pairs": len(primary),
        "ambiguous_pairs_excluded": len(amb_rows),
        "category_counts": dict(counts),
        "positive_pairs": sum(1 for r in primary if r["expected"]),
        "negative_pairs": sum(1 for r in primary if not r["expected"]),
        "semantic_only": overall_a,
        "semantic_plus_jaccard": overall_b,
        "by_category": by_cat,
        "semantic_false_positives": len(semantic_fp),
        "false_positives_removed_by_jaccard": len(removed_fp),
        "true_duplicates_rejected_by_jaccard": len(rejected_tp),
        "false_positives_remaining_after_both": len(remaining_fp),
        "threshold_sweep": sweep,
    }
    json_path = OUT_DIR / "summary.json"
    json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    cat_names = {
        "A": "A same meaning, different wording",
        "B": "B same meaning, similar wording",
        "C": "C opposite decision, similar wording",
        "D": "D same template, different values",
        "E": "E related topic, different decision",
    }
    cat_rows = []
    for cat in ["A", "B", "C", "D", "E"]:
        a, b = by_cat[cat]["semantic_only"], by_cat[cat]["semantic_plus_jaccard"]
        cat_rows.append([
            cat_names[cat], str(counts[cat]),
            str(a["fp"]), str(a["fn"]), fmt(a["precision"]), fmt(a["recall"]), fmt(a["f1"]),
            str(b["fp"]), str(b["fn"]), fmt(b["precision"]), fmt(b["recall"]), fmt(b["f1"]),
        ])

    sweep_rows = []
    for m in sweep:
        sweep_rows.append([
            str(m["jaccard_cutoff"]),
            str(m["tp"]), str(m["fp"]), str(m["tn"]), str(m["fn"]),
            fmt(m["precision"]), fmt(m["recall"]), fmt(m["f1"]),
        ])

    delta_p = overall_b["precision"] - overall_a["precision"]
    delta_r = overall_b["recall"] - overall_a["recall"]
    delta_f = overall_b["f1"] - overall_a["f1"]

    report = f"""# Jaccard filter experiment

Pair-level comparison. Union-Find clustering is not used.
Production files and `data/decisions_vec.db` were not modified.

- Model: `{summary["model"]}` ({summary["dim"]} dimensions)
- Cosine: L2-normalized vectors, similarity = dot product, threshold `{COSINE_THRESHOLD}`
- Jaccard: `find_duplicates.tokens` + `find_duplicates.jaccard_sim`, cutoff `{JACCARD_CUTOFF}`
- Primary labeled pairs: **{len(primary)}** ({summary["positive_pairs"]} duplicates, {summary["negative_pairs"]} not duplicates)
- Ambiguous pairs excluded from metrics: **{len(amb_rows)}**

## Overall

{md_table(
    ["Approach", "TP", "FP", "TN", "FN", "Precision", "Recall", "F1"],
    [
        ["Semantic only (cosine >= 0.75)", str(overall_a["tp"]), str(overall_a["fp"]), str(overall_a["tn"]), str(overall_a["fn"]), fmt(overall_a["precision"]), fmt(overall_a["recall"]), fmt(overall_a["f1"])],
        ["Semantic + Jaccard (<= 0.32)", str(overall_b["tp"]), str(overall_b["fp"]), str(overall_b["tn"]), str(overall_b["fn"]), fmt(overall_b["precision"]), fmt(overall_b["recall"]), fmt(overall_b["f1"])],
    ],
)}

## By category

{md_table(
    ["Category", "N", "A FP", "A FN", "A Prec", "A Rec", "A F1", "B FP", "B FN", "B Prec", "B Rec", "B F1"],
    cat_rows,
)}

A = semantic only. B = semantic + Jaccard.

## 1. False positives removed by Jaccard

Semantic-only accepted these as duplicates. Jaccard rejected them. Expected label is not duplicate.

Count: **{len(removed_fp)}**

{pair_table(removed_fp)}

## 2. Genuine duplicates rejected by Jaccard

Semantic-only accepted these, and the expected label is duplicate. Jaccard rejected them because word overlap was above 0.32 and the texts were not identical.

Count: **{len(rejected_tp)}**

{pair_table(rejected_tp)}

## 3. False positives that passed both filters

Expected not duplicate, but both approaches accepted them.

Count: **{len(remaining_fp)}**

{pair_table(remaining_fp)}

## Jaccard cutoff sweep (cosine fixed at 0.75)

{md_table(
    ["Jaccard cutoff", "TP", "FP", "TN", "FN", "Precision", "Recall", "F1"],
    sweep_rows,
)}

`none` means no Jaccard filter (semantic only).

## Ambiguous pairs (excluded from metrics)

{pair_table(amb_rows)}

## Measured answers

1. Incorrect semantic matches (false positives of cosine >= 0.75): **{len(semantic_fp)}** of {summary["negative_pairs"]} negative pairs.
2. Of those, Jaccard removed **{len(removed_fp)}**.
3. Genuine duplicates Jaccard rejected: **{len(rejected_tp)}**.
4. Precision changed by **{delta_p:+.4f}** ({fmt(overall_a["precision"])} → {fmt(overall_b["precision"])}).
5. Recall changed by **{delta_r:+.4f}** ({fmt(overall_a["recall"])} → {fmt(overall_b["recall"])}).
6. F1 changed by **{delta_f:+.4f}** ({fmt(overall_a["f1"])} → {fmt(overall_b["f1"])}).
7. On this labeled set, cutoff 0.32 is the sweep point to compare against 0.20, 0.25, 0.40, and 0.55 in the table above.
8. Limits: labels were written for this experiment, not drawn from a production review. Categories A–E are controlled examples, including planted paraphrases. Pair scoring ignores neighbor search and clustering. Opposite-wording and template pairs were chosen to be challenging, so the false-positive rate is not an estimate of the 10k corpus.
"""
    report_path = OUT_DIR / "report.md"
    report_path.write_text(report, encoding="utf-8")
    print(report)
    print(f"\nWrote {csv_path}")
    print(f"Wrote {json_path}")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
