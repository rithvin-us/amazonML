# Business Entity Resolution Pipeline

Matches Source 1 business records to Source 2/Source 3 records using
TF-IDF blocking + XGBoost classification with string-similarity features.

## Requirements

- Python 3.11+
- Dependencies: `pip install -r requirements.txt`
- Optional: CUDA-capable GPU for XGBoost (falls back to CPU automatically)

## Data Setup

Place the challenge dataset under `student_resource/dataset/`:

```
student_resource/dataset/
  train/
    train_source1.tsv
    train_source2.tsv
    train_source3.tsv
    train_ground_truth.tsv
  test/
    test_source1.tsv
    test_source2.tsv
    test_source3.tsv
```

## Running

All commands run from the repository root.

```bash
# Full end-to-end: normalise, train, tune, predict on test, validate
python src/pipeline.py all

# Dev run on 2% sample (fast iteration)
python src/pipeline.py all --sample 0.02

# Train only (no test prediction)
python src/pipeline.py train

# Predict with a previously trained model
python src/pipeline.py predict --run <run_id>

# Build submission zip
python src/pipeline.py submit --run <run_id> --team <team_name>
```

## Key Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--sample` | 1.0 | Fraction of S1 train entities to use |
| `--model` | xgb | `xgb` (GPU-capable) or `lgb` (LightGBM/CPU) |
| `--device` | cuda | `cuda` or `cpu` for XGBoost |
| `--max-cands` | 40 | Max candidates per S1 after blocking |
| `--max-df` | 150 | Max document frequency for blocking keys |
| `--threshold` | (tuned) | Override the tuned decision threshold |

## Pipeline Stages

1. **Prep** (`prep.py`): Normalise names/addresses (ASCII, legal suffixes,
   abbreviations) and cache as partitioned parquet.
2. **Blocking** (`blocking.py`): IDF-weighted inverted index over hashed
   name/address tokens, bigrams, prefixes, and postcodes. Built per country.
3. **Features** (`features.py`): 30 pairwise features via `rapidfuzz` string
   similarities (ratio, token-set, Jaro-Winkler, partial), postcode match,
   numeric Jaccard, blocking rank/score, and group context.
4. **Train** (`pipeline.py`): XGBoost binary classifier with early stopping
   on validation logloss.
5. **Tune**: Grid-search over global threshold vs. per-S1 expected-F0.5
   decision, selecting whichever maximises macro F0.5 on the val split.
6. **Predict**: Score test candidates, apply the tuned decision rule, and
   write `matching_results.tsv` + `candidate_pairs.tsv`.
7. **Validate**: Run the official `validate_submission.py` on outputs.

## Output

- `output/matching_results.tsv` — final matches (upload to leaderboard)
- `output/candidate_pairs.tsv` — blocking candidate set
- `runs/<run_id>/` — logs, metrics, model, feature importance, training curve
