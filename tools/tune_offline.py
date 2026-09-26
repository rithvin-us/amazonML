"""Offline decision tuning on a finished run's val_scored + val_truth (no retraining)."""
import sys
import time

sys.path.insert(0, r"D:\amazon-ml\code\business_entity_resolution\src")
import polars as pl  # noqa: E402

import pipeline as pp  # noqa: E402


class _Log:
    def log(self, m):
        print(m)


run_dir = pp.RUNS_DIR / sys.argv[1]
va = pl.read_parquet(run_dir / "val_scored.parquet")
tvc = pl.read_parquet(run_dir / "val_truth.parquet").select("s1_idx", "n_true")
t = time.time()
res = pp.tune_decision(va, tvc, 0.02, _Log())
best = max(res.values(), key=lambda d: d["f05"])
print(f"best: {best['mode']} {best['param']} excl={best['excl']} f05={best['f05']:.5f}  ({time.time() - t:.0f}s)")
