"""Average stage-1 probabilities of runs that share candidates (same blocking/test feats), into a synthetic run
dir that ce-apply can consume: val_scored, comp_scored, val_truth, metrics.json, pred/{scored,cand}.

  python ens.py <name> <trainA> <predA> <trainB> <predB> [<trainC> <predC> ...]
Pairs below a model's p_floor (0.02) are missing from its files: they count as 0.01 in the average.
"""
import json
import shutil
import sys
import time
from pathlib import Path

import polars as pl

RUNS = Path(r"D:\amazon-ml\runs")
name, rest = sys.argv[1], sys.argv[2:]
pairs = [(RUNS / rest[i], RUNS / rest[i + 1]) for i in range(0, len(rest), 2)]
out = RUNS / f"{time.strftime('%Y%m%d-%H%M%S')}-{name}"
out.mkdir()
FLOOR = 0.01


def avg(frames: list[pl.DataFrame], keys: list[str], keep: list[str]) -> pl.DataFrame:
    base = pl.concat([f.select(*keys, *keep) for f in frames]).unique(subset=keys, keep="first")
    for i, f in enumerate(frames):
        base = base.join(f.select(*keys, pl.col("p").alias(f"p{i}")), on=keys, how="left")
    n = len(frames)
    return base.with_columns((sum(pl.col(f"p{i}").fill_null(FLOOR) for i in range(n)) / n).cast(pl.Float32).alias("p")
                             ).drop([f"p{i}" for i in range(n)]).filter(pl.col("p") >= 0.02)


va = avg([pl.read_parquet(t / "val_scored.parquet") for t, _ in pairs], ["s1_idx", "cand_idx"], ["label"])
va.select("s1_idx", "cand_idx", "label", "p").write_parquet(out / "val_scored.parquet")
comps = [t / "comp_scored.parquet" for t, _ in pairs if (t / "comp_scored.parquet").exists()]
if len(comps) == len(pairs):
    avg([pl.read_parquet(c) for c in comps], ["s1_idx", "cand_idx"], []).write_parquet(out / "comp_scored.parquet")
shutil.copy(pairs[0][0] / "val_truth.parquet", out / "val_truth.parquet")
shutil.copy(pairs[0][0] / "metrics.json", out / "metrics.json")
(out / "pred").mkdir()
te = avg([pl.read_parquet(p / "pred" / "scored-*.parquet") for _, p in pairs], ["s1_idx", "cand_idx"], ["cid"])
te.select("s1_idx", "cand_idx", "cid", "p").write_parquet(out / "pred" / "scored-00000.parquet")
for f in (pairs[0][1] / "pred").glob("cand-*.parquet"):
    shutil.copy(f, out / "pred" / f.name)
(out / "ensemble.json").write_text(json.dumps({"members": [[t.name, p.name] for t, p in pairs]}, indent=2))
print(out.name, f"val pairs {va.height:,} test pairs {te.height:,}")
