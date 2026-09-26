"""Profile BlockIndex build phases on a pool slice."""
import sys
import time

sys.path.insert(0, r"D:\amazon-ml\code\business_entity_resolution\src")
import polars as pl  # noqa: E402

from blocking import build_keys  # noqa: E402
from pipeline import NORM_COLS, scan_norm  # noqa: E402

pool = scan_norm("train", "pool").filter(pl.col("country_n") == "india").select(NORM_COLS).head(1_000_000).collect()
chunk = 100_000
t = time.time()
parts = [build_keys(pool.slice(i, chunk)) for i in range(0, pool.height, chunk)]
t_keys = time.time() - t
t = time.time()
dfc = None
for p in parts:
    c = p.group_by("key").len()
    dfc = c if dfc is None else pl.concat([dfc, c]).group_by("key").agg(pl.col("len").sum())
t_merge = time.time() - t
t = time.time()
one = pl.concat([p.select("key") for p in parts]).group_by("key").len()
t_single = time.time() - t
print(f"build_keys x{len(parts)}: {t_keys:.1f}s | incremental merge: {t_merge:.1f}s | single group_by: {t_single:.1f}s "
      f"| keys {dfc.height:,} rows {sum(p.height for p in parts):,}")
t = time.time()
k = build_keys(pool.slice(0, chunk))
print(f"one build_keys chunk: {time.time() - t:.2f}s")

import cProfile, pstats  # noqa: E401,E402
from blocking import BlockIndex  # noqa: E402
t = time.time()
pr = cProfile.Profile(); pr.enable()
ix = BlockIndex(pool, max_df=600, chunk=100_000)
pr.disable()
print(f"BlockIndex(1M): {time.time() - t:.1f}s keys {len(ix.keys):,} postings {ix.n_postings:,}")
pstats.Stats(pr).sort_stats("cumulative").print_stats(12)
