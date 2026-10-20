
"""Find top-N semantic duplicate decision groups."""

from __future__ import annotations

import argparse
import json
import re
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.neighbors import NearestNeighbors
from store import DEFAULT_DB_PATH, DecisionStore

ROOT = Path(__file__).parent
DEFAULT_INPUT = ROOT / "data" / "decisions.json"
DEFAULT_OUTPUT = ROOT / "output" / "top_duplicates.json"


# Normalize text for consistent comparison.
def normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


# Extract unique words from a sentence.
def tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", normalize(text)))


# Manage groups of matching decisions.
class UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb: return
        if self.rank[ra] < self.rank[rb]: self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]: self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1


# Calculate word overlap using Jaccard similarity.
def jaccard_sim(a: set[str], b: set[str]) -> float:
    if not a and not b: return 1.0
    union = a | b
    if not union: return 0.0
    return len(a & b) / len(union)


# Find semantic duplicate groups using cosine similarity and lexical overlap.
def semantic_clusters(
    embeddings: np.ndarray,
    texts: list[str],
    threshold: float,
    top_k: int,
    max_lexical_overlap: float = 0.32,
) -> list[list[int]]:
    n = embeddings.shape[0]
    if n < 2: return []
    k = min(top_k + 1, n)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        nn = NearestNeighbors(n_neighbors=k, metric="cosine", algorithm="brute")
        nn.fit(embeddings)
        distances, indices = nn.kneighbors(embeddings)

    tok = [tokens(t) for t in texts]
    uf = UnionFind(n)
    max_dist = 1.0 - threshold

    for i in range(n):
        for dist, j in zip(distances[i], indices[i]):
            if i == j: continue
            if dist > max_dist: continue
            if normalize(texts[i]) == normalize(texts[j]):
                uf.union(i, j)
                continue
            if jaccard_sim(tok[i], tok[j]) <= max_lexical_overlap:
                uf.union(i, j)

    clusters: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        clusters[uf.find(i)].append(i)
    return [members for members in clusters.values() if len(members) > 1]


# Calculate average semantic similarity within a group.
def mean_cosine(members: list[int], embeddings: np.ndarray) -> float:
    vecs = embeddings[members]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        sim = vecs @ vecs.T
    m = len(members)
    return float((sim.sum() - m) / (m * (m - 1)))


# Calculate average wording difference within a group.
def mean_jaccard_distance(members: list[int], texts: list[str]) -> float:
    tok = [tokens(texts[i]) for i in members]
    m = len(tok)
    if m < 2: return 0.0
    total = 0.0
    pairs = 0
    for a in range(m):
        for b in range(a + 1, m):
            union = tok[a] | tok[b]
            dist = 0.0 if not union else 1.0 - (len(tok[a] & tok[b]) / len(union))
            total += dist
            pairs += 1
    return total / pairs


# Score each group using size, similarity, and wording diversity.
def cluster_score(members: list[int], embeddings: np.ndarray, texts: list[str]) -> float:
    sim = mean_cosine(members, embeddings)
    diversity = mean_jaccard_distance(members, texts)
    return (1.0 + np.log(len(members))) * sim * (diversity ** 2)


# Prepare ranked groups with decision IDs and unique text variants.
def build_result_groups(
    ranked: list[list[int]],
    decisions: list[dict],
    texts: list[str],
    embeddings: np.ndarray,
) -> list[dict]:
    groups: list[dict] = []
    for rank, members in enumerate(ranked, start=1):
        seen_norm: set[str] = set()
        variants: list[dict] = []
        for idx in members:
            key = normalize(texts[idx])
            if key in seen_norm:
                continue
            seen_norm.add(key)
            row = decisions[idx]
            variants.append({"id": row.get("id", idx), "text": texts[idx]})

        groups.append({
            "rank": rank,
            "score": round(cluster_score(members, embeddings, texts), 4),
            "variants": variants,
        })
    return groups


# Export ranked duplicate groups to a JSON file.
def export_results(groups: list[dict], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"top_duplicate_groups": groups}
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return output_path


# Load decisions, detect duplicates, rank groups, and export results.
def main() -> None:
    parser = argparse.ArgumentParser(description="Find top semantic duplicate decisions")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--reingest", action="store_true", help="Rebuild the libSQL DB from the JSON input")
    parser.add_argument("--top", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.75)
    parser.add_argument("--max-lexical-overlap", type=float, default=0.32)
    parser.add_argument("--neighbors", type=int, default=50)
    args = parser.parse_args()

    store = DecisionStore(db_path=args.db)
    store.ingest_json(args.input, force=args.reingest)
    decisions, texts, embeddings = store.load_all()
    print(f"Loaded {len(decisions)} decisions from {args.db}")

    clusters = semantic_clusters(
        embeddings, texts, args.threshold, args.neighbors,
        max_lexical_overlap=args.max_lexical_overlap,
    )

    ranked = sorted(
        clusters,
        key=lambda members: cluster_score(members, embeddings, texts),
        reverse=True,
    )[:args.top]

    groups = build_result_groups(ranked, decisions, texts, embeddings)

    print(f"\n=== Top {args.top} semantic duplicate groups ===\n")
    for g in groups:
        print(f"#{g['rank']}  score={g['score']:.3f}")
        for v in g["variants"]:
            print(f"  - [{v['id']}] {v['text']}")
        print()

    out = export_results(groups, args.output)
    print(f"Exported results → {out}")


if __name__ == "__main__":
    main()
