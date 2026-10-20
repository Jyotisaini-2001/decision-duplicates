# Decision duplicates

## 1) Generate demo decisions and run

```bash
cd ~/Desktop/learn/decision-duplicates
source .venv/bin/activate

# any number, e.g. 100 / 1000 / 10000
python generate_data.py --count 10000

# rebuild DB from that JSON and find top duplicates
rm -f data/decisions_vec.db
python find_duplicates.py --reingest
```

Results: `output/top_duplicates.json`

## 2) Run with your own JSON

Your file must be a JSON array like:

```json
[
  { "id": "dec_001", "text": "We will ship on Friday." },
  { "id": "dec_002", "text": "Launch date is Friday." }
]
```

```bash
cd ~/Desktop/learn/decision-duplicates
source .venv/bin/activate

python find_duplicates.py --input /path/to/your.json --reingest
```

Results: `output/top_duplicates.json`

## 3) Prove why threshold=0.75 and max-lexical-overlap=0.32

Needs demo JSON with `planted_group` (from `generate_data.py`):

```bash
cd ~/Desktop/learn/decision-duplicates
source .venv/bin/activate

python generate_data.py --count 100
python prove_thresholds.py
```

This prints cosine scores for known paraphrases (why 0.75) and word-overlap for paraphrases vs templates (why 0.32).
