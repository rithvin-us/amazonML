# Amazon ML Challenge 2026 — Business Entity Resolution

Goal: for each Source1 (S1) record find matching S2/S3 records. Metric: macro F0.5 per S1 entity
(singletons count; empty prediction on a singleton = 1.0). Precision matters 2x recall.
Rules: no external lookups/APIs/geocoding; final model MIT/Apache ≤8B; `country` is open-set
(test has France, unseen in train) — never hardcode/filter countries.

## Layout
- `student_resource/` — official data + `utils/validate_submission.py` (do not modify)
- `code/business_entity_resolution/src/` — pipeline (entry: `pipeline.py`)
- `cache/` parquet caches · `runs/<run_id>/` run registry · `output/` submission TSVs
- `notebooks/dashboard.ipynb` — human UI over runs/ + hardware
- Python: `python3` (3.11+). On Windows: `.venv311\Scripts\python.exe`.

## Commands (run from repo root)
```
PY=python3; SRC=code/business_entity_resolution/src
$PY $SRC/profile_data.py                 # aggregate stats -> runs/profile.json
$PY $SRC/pipeline.py all --sample 0.02   # dev run
$PY $SRC/pipeline.py all                 # full run (train+val+test+validate)
$PY $SRC/pipeline.py lb <run_id> <score> # record portal score
$PY $SRC/hwmon.py 30                     # 30s hardware snapshot
```

## Watching progress (for Claude)
- `runs/LATEST` -> current run id
- `runs/<id>/status.json` stage/pct/eta/state; `log.txt`; `metrics.json`; `hw.csv` (CPU/RAM/GPU/throttle)
- `runs/leaderboard.csv` — all runs incl. `lb_score` from portal
- Long jobs: launch with Bash run_in_background, watch with Monitor on log.txt
  (grep `stage|metric|Traceback|Error|MemoryError|run finished`).

## Token rules
- NEVER cat/Read the dataset TSVs or big parquet. Use `head -3`, `runs/profile.json`, or polars aggregates.
- Read logs with `tail`, not whole files.
