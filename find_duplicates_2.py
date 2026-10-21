"""Conservative duplicate-decision discovery. Keep store.py and embeddings.py unchanged.

Default: hybrid candidate retrieval -> detail guard -> bidirectional NLI ->
complete-link clustering. Run on labeled 200-decision data before production.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from decimal import Decimal, InvalidOperation
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.neighbors import NearestNeighbors
from store import DEFAULT_DB_PATH, DecisionStore

ROOT = Path(__file__).parent
DEFAULT_INPUT = ROOT / 'data' / 'experiment_decision_dataset_10k.json'
DEFAULT_OUTPUT = ROOT / 'output' / 'top_duplicates.json'


def normalize(text):
    return re.sub(r'\s+', ' ', text.lower().strip())


def tokens(text):
    return set(re.findall(r'[a-z0-9]+', normalize(text)))


def jaccard_sim(a, b):
    return len(a & b) / len(a | b) if a | b else 1.0


class BM25:
    def __init__(self, texts):
        self.n = len(texts)
        self.lengths = []
        self.postings = defaultdict(list)
        for i, text in enumerate(texts):
            words = re.findall(r'[a-z0-9]+', normalize(text))
            self.lengths.append(len(words))
            for word, count in Counter(words).items():
                self.postings[word].append((i, count))
        self.avg = sum(self.lengths) / max(self.n, 1)

    def top(self, text, k, exclude):
        scores = defaultdict(float)
        for word in tokens(text):
            hits = self.postings.get(word, ())
            if not hits:
                continue
            idf = math.log(1 + (self.n - len(hits) + .5) / (len(hits) + .5))
            for j, freq in hits:
                if j == exclude:
                    continue
                norm = 1.5 * (.25 + .75 * self.lengths[j] / max(self.avg, 1))
                scores[j] += idf * freq * 2.5 / (freq + norm)
        return [j for j, _ in sorted(scores.items(), key=lambda x: (-x[1], x[0]))[:k]]


def candidates(vectors, texts, neighbors, hybrid):
    n = len(texts)
    if n < 2:
        return set()
    nn = NearestNeighbors(n_neighbors=min(neighbors + 1, n), metric='cosine', algorithm='brute')
    nn.fit(vectors)
    _, indices = nn.kneighbors(vectors)
    bm25 = BM25(texts) if hybrid else None
    found = set()
    for i in range(n):
        semantic = [int(j) for j in indices[i] if int(j) != i][:neighbors]
        if bm25:
            lexical = bm25.top(texts[i], neighbors, i)
            fusion = defaultdict(float)
            for ranking in (semantic, lexical):
                for rank, j in enumerate(ranking, 1):
                    fusion[j] += 1 / (60 + rank)
            selected = sorted(fusion, key=lambda j: (-fusion[j], j))[:neighbors]
        else:
            selected = semantic
        for j in selected:
            found.add((min(i, j), max(i, j)))
    return found


# Conservative, explicit mismatch rules. They do not attempt full entity resolution.
_NUMBER = re.compile(r'(?<![a-z0-9])\d+(?:[.,]\d+)*(?:%|k|m|b)?(?![a-z0-9])', re.I)
_CODE = re.compile(r'\b[A-Z]{2,}[A-Z0-9]*[-_][A-Z0-9_-]+\b')
_MONTH = re.compile(r'\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\b', re.I)
_WEEKDAY = re.compile(r'\b(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b', re.I)
_OFFICE = re.compile(r'\b(?:close|closing|shut|shutdown|reopen|relocate)\s+(?:the\s+)?([a-z][a-z\s-]{1,35}?)\s+office\b', re.I)
_NEGATION = re.compile(r'\b(?:not|never|reject|rejected|cancel|cancelled|deny|denied|prohibit|prohibited)\b', re.I)


# Canonicalize explicit numeric formats, without assuming missing values are equal.
# Examples: 24% == 24 percent; $775k == $775,000; Q+1 == next quarter.
_VALUE = re.compile(r'(?<![a-z0-9])(?P<currency>\$)?(?P<num>\d[\d,]*(?:\.\d+)?)\s*(?P<scale>k|m|b)?\s*(?P<pct>%|percent|percentage)?(?![a-z0-9])', re.I)
_QUARTER = re.compile(r'\bq\s*\+\s*1\b|\bnext quarter\b', re.I)

def numeric_values(text):
    text = _QUARTER.sub(' NEXT_QUARTER ', text)
    result = set()
    for match in _VALUE.finditer(text):
        try:
            value = Decimal(match.group('num').replace(',', ''))
        except InvalidOperation:
            continue
        scale = (match.group('scale') or '').lower()
        value *= {'': 1, 'k': 1000, 'm': 1000000, 'b': 1000000000}[scale]
        unit = 'percent' if match.group('pct') else ('currency' if match.group('currency') else 'number')
        result.add((unit, str(value.normalize())))
    if 'NEXT_QUARTER' in text:
        result.add(('period', 'next_quarter'))
    return result


def detail_conflict(a, b):
    """Reject only explicit incompatible details, not missing details.

    A missing value is left to the verifier. Normalization is intentionally narrow.
    """
    if normalize(a) == normalize(b):
        return None
    aa, bb = numeric_values(a), numeric_values(b)
    # Treat Q+1 / next quarter as one period, not a stray number.
    if aa and bb and aa != bb:
        return f'number/period mismatch: {sorted(aa)} vs {sorted(bb)}'
    for label, pattern in [('identifier', _CODE), ('month', _MONTH),
                           ('weekday', _WEEKDAY), ('office location', _OFFICE)]:
        aa = {x.lower().strip() for x in pattern.findall(a)}
        bb = {x.lower().strip() for x in pattern.findall(b)}
        if aa and bb and aa != bb:
            return f'{label} mismatch: {sorted(aa)} vs {sorted(bb)}'
    if bool(_NEGATION.search(a)) != bool(_NEGATION.search(b)):
        return 'possible polarity conflict'
    return None


class NLIVerifier:
    """A->B AND B->A entailment; relatedness alone is insufficient."""
    def __init__(self, model_name, threshold):
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise SystemExit('Install NLI dependencies: pip install transformers torch') from exc
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()
        labels = {str(v).lower(): int(k) for k, v in self.model.config.id2label.items()}
        matches = [v for k, v in labels.items() if 'entail' in k]
        if len(matches) != 1:
            raise SystemExit(f'NLI model must expose an entailment label; got {labels}')
        self.entailment_index = matches[0]
        self.threshold = threshold

    def entailment(self, premise, hypothesis):
        inputs = self.tokenizer(premise, hypothesis, return_tensors='pt',
                                truncation=True, max_length=384)
        with self.torch.inference_mode():
            logits = self.model(**inputs).logits[0]
            probs = self.torch.softmax(logits, dim=-1)
        return float(probs[self.entailment_index])

    def verify(self, a, b):
        ab = self.entailment(a, b)
        ba = self.entailment(b, a)
        confidence = min(ab, ba)
        return ('duplicate' if confidence >= self.threshold else 'not_duplicate',
                f'bidirectional entailment min={confidence:.3f} (A->B={ab:.3f}, B->A={ba:.3f})')


def conservative_groups(n, accepted, max_size):
    """Complete-link groups: every pair in a group must be positively verified.

    Greedy clique partition, not optimal clique cover. No transitive chain merges.
    """
    edges = {(min(a, b), max(a, b)) for a, b in accepted}
    neighbors = defaultdict(set)
    for a, b in edges:
        neighbors[a].add(b)
        neighbors[b].add(a)
    remaining = set(neighbors)
    groups = []
    while remaining:
        seed = max(remaining, key=lambda x: (len(neighbors[x] & remaining), -x))
        group = [seed]
        remaining.remove(seed)
        for j in sorted(neighbors[seed] & remaining,
                        key=lambda x: (-len(neighbors[x] & remaining), x)):
            if len(group) >= max_size:
                break
            if all((min(i, j), max(i, j)) in edges for i in group):
                group.append(j)
                remaining.remove(j)
        if len(group) > 1:
            groups.append(group)
    return groups


def score_group(members, vectors):
    """Rank by weakest pair similarity; do not reward size or word diversity."""
    sims = [float(vectors[i] @ vectors[j]) for p, i in enumerate(members)
            for j in members[p+1:]]
    return min(sims) if sims else 0.0


def main():
    p = argparse.ArgumentParser(description='Compare duplicate retrieval and verification methods')
    p.add_argument('--input', type=Path, default=DEFAULT_INPUT)
    p.add_argument('--db', type=Path, default=DEFAULT_DB_PATH)
    p.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    p.add_argument('--reingest', action='store_true')
    p.add_argument('--top', type=int, default=20)
    p.add_argument('--neighbors', type=int, default=20)
    p.add_argument('--threshold', type=float, default=.75)
    p.add_argument('--max-lexical-overlap', type=float, default=.32)
    p.add_argument('--method', choices=['semantic', 'jaccard', 'hybrid'], default='hybrid')
    p.add_argument('--verifier', choices=['nli', 'none'], default='nli')
    p.add_argument('--nli-model', default='cross-encoder/nli-deberta-v3-base')
    p.add_argument('--nli-threshold', type=float, default=.85,
                   help='Tune on development labels; evaluate once on held-out labels')
    p.add_argument('--max-verifications', type=int, default=2000,
                   help='Maximum NLI pair calls; remaining pairs are uncertain, not accepted')
    p.add_argument('--max-group-size', type=int, default=25)
    p.add_argument('--pairs-output', type=Path, default=None)
    args = p.parse_args()
    if min(args.top, args.neighbors, args.max_group_size) < 1 or args.max_verifications < 0:
        p.error('top/neighbors/max-group-size must be positive; max-verifications >= 0')
    if not all(0 <= x <= 1 for x in (args.threshold, args.max_lexical_overlap, args.nli_threshold)):
        p.error('thresholds must be between 0 and 1')

    store = DecisionStore(db_path=args.db)
    store.ingest_json(args.input, force=args.reingest)
    decisions, texts, vectors = store.load_all()
    vectors = np.asarray(vectors, dtype=np.float32)
    vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
    pair_ids = candidates(vectors, texts, args.neighbors, args.method == 'hybrid')
    wordsets = [tokens(t) for t in texts]
    # Most promising candidates first; cap is explicit and reported.
    # NOTE: for fair threshold sweeps, use the same candidate set and verification budget.
    ordered = sorted(pair_ids, key=lambda ij: (-float(vectors[ij[0]] @ vectors[ij[1]]), ij))
    verifier = NLIVerifier(args.nli_model, args.nli_threshold) if args.verifier == 'nli' else None
    accepted, rows = [], []
    verified = 0
    for i, j in ordered:
        cos = float(vectors[i] @ vectors[j])
        jac = jaccard_sim(wordsets[i], wordsets[j])
        exact = normalize(texts[i]) == normalize(texts[j])
        label, reason = 'not_duplicate', 'below cosine threshold'
        if exact:
            label, reason = 'duplicate', 'exact normalized text'
        elif cos >= args.threshold:
            if args.method == 'jaccard' and jac > args.max_lexical_overlap:
                reason = 'Jaccard baseline rejected high word overlap'
            else:
                conflict = detail_conflict(texts[i], texts[j])
                if conflict:
                    reason = conflict
                elif verifier is None:
                    label, reason = 'duplicate', 'UNVERIFIED cosine candidate (baseline only)'
                elif verified < args.max_verifications:
                    verified += 1
                    label, reason = verifier.verify(texts[i], texts[j])
                else:
                    label, reason = 'uncertain', 'verification budget exhausted'
        if label == 'duplicate':
            accepted.append((i, j))
        rows.append({'a_index': i, 'b_index': j,
                     'a_id': decisions[i].get('id', i), 'b_id': decisions[j].get('id', j),
                     'cosine': round(cos, 5), 'jaccard': round(jac, 5),
                     'prediction': label, 'reason': reason})
    groups = conservative_groups(len(texts), accepted, args.max_group_size)
    groups.sort(key=lambda g: (-score_group(g, vectors), -len(g), min(g)))
    output = []
    for rank, group in enumerate(groups[:args.top], 1):
        variants, seen = [], set()
        for i in sorted(group):
            norm = normalize(texts[i])
            if norm in seen:
                continue
            seen.add(norm)
            variants.append({'id': decisions[i].get('id', i), 'text': texts[i]})
        output.append({'rank': rank, 'score': round(score_group(group, vectors), 4),
                       'variants': variants})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({'top_duplicate_groups': output}, indent=2,
                                      ensure_ascii=False) + '\n', encoding='utf-8')
    pairs_path = args.pairs_output or args.output.with_name(args.output.stem + '_pairs.json')
    pairs_path.parent.mkdir(parents=True, exist_ok=True)
    pairs_path.write_text(json.dumps({'settings': vars(args) | {'input': str(args.input),
                               'db': str(args.db), 'output': str(args.output),
                               'pairs_output': str(pairs_path)},
                               'pairs': rows}, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    counts = Counter(r['prediction'] for r in rows)
    print(f'Decisions={len(texts)} candidates={len(rows)} accepted={counts["duplicate"]} '
          f'rejected={counts["not_duplicate"]} uncertain={counts["uncertain"]} '
          f'NLI_calls={verified} groups={len(groups)}')
    print(f'Groups: {args.output}\nPairs: {pairs_path}')
    print('No ground-truth metrics calculated. Audit labels and sample false positives/negatives.')
    if args.verifier == 'none':
        print('WARNING: no NLI verification; outputs are candidate matches, NOT confirmed duplicates.')


if __name__ == '__main__':
    main()
