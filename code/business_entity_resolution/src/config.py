"""Central paths and knobs. Override via CLI flags in pipeline.py."""
from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from pathlib import Path

# Project root = D:\amazon-ml (four levels up from this file), overridable with ER_ROOT.
ROOT = Path(os.environ.get("ER_ROOT", Path(__file__).resolve().parents[3]))
DATA_DIR = Path(os.environ.get("ER_DATA", ROOT / "student_resource" / "dataset"))
CACHE_DIR = ROOT / "cache"
RUNS_DIR = ROOT / "runs"
OUTPUT_DIR = ROOT / "output"
VALIDATOR = ROOT / "student_resource" / "utils" / "validate_submission.py"


@dataclass
class Config:
    run_name: str = "baseline"
    seed: int = 42
    sample: float = 1.0            # fraction of S1 train entities used (dev mode < 1)
    val_frac: float = 0.2          # held-out S1 train entities for validation
    # blocking
    top_k_tfidf: int = 30          # char-ngram TF-IDF neighbours per S1 (per source)
    top_k_token: int = 20          # rare-token block neighbours per S1
    max_candidates: int = 60       # hard cap after union
    tfidf_min_sim: float = 0.2
    chunk_size: int = 2000
    # model
    lgb_rounds: int = 600
    lgb_lr: float = 0.05
    lgb_leaves: int = 63
    neg_per_pos_cap: int = 0       # 0 = keep all negatives
    threshold: float = 0.5         # overwritten by tuning
    n_jobs: int = max(1, (os.cpu_count() or 4) - 2)
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


for d in (CACHE_DIR, RUNS_DIR, OUTPUT_DIR):
    d.mkdir(parents=True, exist_ok=True)
