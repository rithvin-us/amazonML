"""Central paths and knobs. Override via CLI flags in pipeline.py."""
from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field, asdict
from pathlib import Path

# Project root = D:\amazon-ml (four levels up from this file), overridable with ER_ROOT.
ROOT = Path(os.environ.get("ER_ROOT", Path(__file__).resolve().parents[3]))
DATA_DIR = Path(os.environ.get("ER_DATA", ROOT / "student_resource" / "dataset"))
CACHE_DIR = ROOT / "cache"
RUNS_DIR = ROOT / "runs"
OUTPUT_DIR = ROOT / "output"
TMP_DIR = ROOT / "tmp"
LOCAL_CACHE = ROOT / ".cache"
VALIDATOR = ROOT / "student_resource" / "utils" / "validate_submission.py"
NORM_VERSION = "v5"  # bump when normalize.py output changes -> fresh cache/norm_<split>_<kind>_<ver>

# Keep every temp/cache write on the project drive (C: is full). Child processes inherit these.
_ENV_DIRS = {
    "TMP": TMP_DIR, "TEMP": TMP_DIR, "TMPDIR": TMP_DIR, "POLARS_TEMP_DIR": TMP_DIR / "polars",
    "CUDA_CACHE_PATH": LOCAL_CACHE / "nv", "XDG_CACHE_HOME": LOCAL_CACHE, "HF_HOME": LOCAL_CACHE / "hf",
    "TORCH_HOME": LOCAL_CACHE / "torch", "PIP_CACHE_DIR": LOCAL_CACHE / "pip", "MPLCONFIGDIR": LOCAL_CACHE / "mpl",
}
for _k, _v in _ENV_DIRS.items():
    _v.mkdir(parents=True, exist_ok=True)
    os.environ[_k] = str(_v)
tempfile.tempdir = str(TMP_DIR)


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
    chunk_size: int = 4000
    s1_chunk: int = 20_000         # S1 rows featurised per step (bounds peak RAM)
    # model
    model: str = "xgb"             # xgb (GPU-capable) | lgb
    xgb_rounds: int = 4000
    xgb_lr: float = 0.05
    xgb_depth: int = 8
    xgb_early_stop: int = 100
    device: str = "cuda"           # XGBoost on local GPU (RTX 4050); "cpu" fallback
    neg_per_pos_cap: int = 0       # 0 = keep all negatives
    threshold: float = 0.5         # overwritten by tuning
    p_floor: float = 0.02          # test pairs below this are never kept for the decision stage
    n_jobs: int = max(1, (os.cpu_count() or 4) - 2)
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


for d in (CACHE_DIR, RUNS_DIR, OUTPUT_DIR):
    d.mkdir(parents=True, exist_ok=True)
