# How to run

From this folder:

```bash
source .venv/bin/activate
```

If the environment is not set up yet:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Data files:

- 200 decisions: `data/experiment_decision_dataset_200.json`
- 10,000 decisions: `data/experiment_decision_dataset_10k.json`

## Semantic + Jaccard

200 decisions:

```bash
python find_duplicates_1.py \
  --input data/experiment_decision_dataset_200.json \
  --db data/test_200_v2.db \
  --output output/semantic_jaccard_200.json \
  --top 10000
```

10,000 decisions:

```bash
python find_duplicates_1.py \
  --input data/experiment_decision_dataset_10k.json \
  --db data/decisions_vec.db \
  --output output/semantic_jaccard_10k.json \
  --top 10000
```

## Hybrid + NLI

200 decisions:

```bash
python find_duplicates_2.py \
  --input data/experiment_decision_dataset_200.json \
  --db data/test_200_v2.db \
  --method hybrid \
  --verifier nli \
  --nli-threshold 0.70 \
  --top 10000 \
  --output output/hybrid_nli_200.json
```

10,000 decisions:

```bash
python find_duplicates_2.py \
  --input data/experiment_decision_dataset_10k.json \
  --db data/decisions_vec.db \
  --method hybrid \
  --verifier nli \
  --nli-threshold 0.70 \
  --top 10000 \
  --output output/hybrid_nli_10k.json
```

The first Hybrid + NLI run downloads the NLI model. Results are written in `output/`.
