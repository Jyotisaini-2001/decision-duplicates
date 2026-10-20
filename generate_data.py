"""Generate a decision JSON file with planted semantic duplicates."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

OUT = Path(__file__).parent / "data" / "decisions.json"
RNG = random.Random(42)

# Planted groups: same meaning, different wording. These should surface in top-5.
PLANTED = [
    [
        "We decided to ship the product on Friday.",
        "Release date is set for Friday.",
        "We're launching this Friday.",
        "Friday is the go-live day for the product.",
        "Agreed: product ships Friday.",
        "Decision: Friday launch for the product.",
        "We'll put the product out on Friday.",
        "Ship date locked to Friday.",
    ],
    [
        "Use PostgreSQL as our primary database.",
        "Postgres is the chosen database.",
        "We're standardizing on PostgreSQL for storage.",
        "Decision: PostgreSQL for the main DB.",
        "Primary datastore will be Postgres.",
        "Adopt PostgreSQL as the database of record.",
        "We picked Postgres over MySQL for the database.",
    ],
    [
        "Remote work is allowed three days per week.",
        "Hybrid policy: 3 days WFH each week.",
        "Employees may work from home three days weekly.",
        "Approved: three remote days per week.",
        "WFH allowance set at 3 days/week.",
        "Three days remote, two in office — that's the policy.",
    ],
    [
        "Raise prices by 10% starting next quarter.",
        "Next quarter we increase pricing 10%.",
        "Approved a 10 percent price increase for Q+1.",
        "Pricing goes up 10% beginning next quarter.",
        "Decision: 10% price hike effective next quarter.",
        "We'll bump prices ten percent next quarter.",
    ],
    [
        "Deprecate the legacy API by end of year.",
        "Legacy API sunset date: December 31.",
        "We're retiring the old API before year-end.",
        "End-of-year deadline to shut down the legacy API.",
        "Decision: legacy API gone by year end.",
        "Phase out the outdated API by December.",
        "Legacy endpoints will be removed by EOY.",
    ],
    [
        "Hire two backend engineers this quarter.",
        "We're opening two backend eng roles this Q.",
        "Approved headcount: 2 backend engineers.",
        "Recruit two backend developers this quarter.",
        "Decision: add two backend engineers ASAP this quarter.",
    ],
]

# One-off unique decision shapes — each row gets its own entity IDs so
# fillers do not form giant near-duplicate clusters.
SHAPES = [
    "Award contract CONT-{i} to vendor VND-{i} for ${amt}k.",
    "Set customer CUST-{i} discount to {pct}% through {month}/{year}.",
    "Close office LOC-{i} in {city} effective {month}/{day}.",
    "Promote employee EMP-{i} to level L{level} on team {team}.",
    "Block IP range NET-{i}.0/24 after incident INC-{i}.",
    "Rename project PRJ-{i} to codename {code}.",
    "Cap wallet WAL-{i} spend at ${amt}/day.",
    "Route region REG-{i} traffic to cluster CLS-{i}.",
    "Archive channel CH-{i} and move history to vault VT-{i}.",
    "Grant role ROLE-{i} to service account SA-{i} only.",
    "Schedule maintenance WIN-{i} on host HST-{i} at {hour}:00 UTC.",
    "Reject RFC RFC-{i} about feature FEAT-{i}.",
    "Ship hotfix HF-{i} for bug BUG-{i} to production tonight.",
    "Set KPI KPI-{i} target to {pct} for squad SQ-{i}.",
    "Buy domain DOM-{i}.example for campaign CMP-{i}.",
]

CITIES = [
    "Lisbon", "Oslo", "Busan", "Medellin", "Nairobi", "Perth", "Quebec",
    "Accra", "Dhaka", "Taipei", "Krakow", "Austin", "Pune", "Recife",
]
TEAMS = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf"]
CODES = [
    "cedar", "flint", "harbor", "ivory", "jasper", "kepler", "lotus",
    "maple", "nebula", "onyx", "pebble", "quartz", "ripple", "saffron",
]


def _unique_filler(i: int) -> dict:
    shape = SHAPES[i % len(SHAPES)]
    text = shape.format(
        i=i,
        amt=10 + (i * 17) % 900,
        pct=1 + (i * 13) % 40,
        month=1 + (i % 12),
        day=1 + (i % 28),
        year=2024 + (i % 3),
        city=CITIES[i % len(CITIES)],
        level=1 + (i % 7),
        team=TEAMS[i % len(TEAMS)],
        code=f"{CODES[i % len(CODES)]}{i}",
        hour=i % 24,
    )
    return {
        "id": f"dec_{i:05d}",
        "text": text,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate decisions JSON")
    parser.add_argument(
        "--count",
        type=int,
        default=10_000,
        help="How many decisions to generate (default: 10000)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUT,
        help="Where to write the JSON file",
    )
    args = parser.parse_args()
    n = max(args.count, sum(len(g) for g in PLANTED))

    decisions: list[dict] = []
    next_id = 0

    for gi, group in enumerate(PLANTED):
        for text in group:
            decisions.append(
                {
                    "id": f"dec_{next_id:05d}",
                    "text": text,
                    "planted_group": gi,
                }
            )
            next_id += 1

    while len(decisions) < n:
        decisions.append(_unique_filler(next_id))
        next_id += 1

    RNG.shuffle(decisions)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(decisions, indent=2), encoding="utf-8")
    print(f"Wrote {len(decisions)} decisions → {args.output}")
    print(f"Planted semantic groups: {len(PLANTED)}")
    for i, g in enumerate(PLANTED):
        print(f"  group {i}: {len(g)} variants")


if __name__ == "__main__":
    main()
