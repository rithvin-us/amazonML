"""Where do blocking-missed true pairs rank? Re-query the cached index with a huge top_k for val S1 that
have misses.  python blocking_debug.py <run_id> [country]
"""
import sys

import numpy as np
import polars as pl

sys.path.insert(0, r"D:\amazon-ml\code\business_entity_resolution\src")
import pipeline as pp  # noqa: E402
from blocking import BLOCK_VERSION, BlockIndex  # noqa: E402

R = pp.RUNS_DIR / sys.argv[1]
country = sys.argv[2] if len(sys.argv) > 2 else "us"
vt = pl.read_parquet(R / "val_truth.parquet").filter(pl.col("country_n") == country)
va = pl.read_parquet(R / "val_scored.parquet", columns=["s1_idx", "cand_idx", "label"])
gt = pl.read_parquet(pp.CACHE_DIR / "gt_long.parquet").drop_nulls().filter(pl.col("s1_id").is_in(vt["s1_id"].implode()))
pool_ids = (pp.scan_norm("train", "pool").filter(pl.col("entity_id").is_in(gt["match_id"].implode()))
            .select(pl.col("entity_id").alias("match_id"), pl.col("idx").alias("cand_idx")).collect())
pos = gt.join(vt.select("s1_idx", "s1_id"), on="s1_id").join(pool_ids, on="match_id")
miss = pos.join(va, on=["s1_idx", "cand_idx"], how="anti")
print(f"{country}: positives {pos.height}, block misses {miss.height}")

s1 = pp.scan_norm("train", "s1").filter(pl.col("idx").is_in(miss["s1_idx"].unique().implode())).select(pp.NORM_COLS).collect()
d = pp.CACHE_DIR / "index" / f"train_{country}_{pp.NORM_VERSION}_b{BLOCK_VERSION}_df600"
ix = BlockIndex.load(d)
q = ix.query(s1, top_k=100_000, k_name=0, k_addr=0)
r = miss.join(q, on=["s1_idx", "cand_idx"], how="left")
print("not retrievable at all (no shared kept key):", r["brank"].null_count())
rr = r.filter(pl.col("brank").is_not_null())
for k in (40, 60, 80, 100, 150, 200, 400):
    print(f"  total-rank <= {k}: {(rr['brank'] <= k).sum()}")
for k in (10, 20, 40, 80):
    print(f"  name-rank <= {k}: {(rr['brank_name'] <= k).sum()}   addr-rank <= {k}: {(rr['brank_addr'] <= k).sum()}")
both = rr.with_columns(pl.min_horizontal(pl.col("brank_name").fill_null(10**9), pl.col("brank_addr").fill_null(10**9)).alias("chan"))
for k in (20, 40, 80):
    print(f"  either channel rank <= {k}: {(both['chan'] <= k).sum()}")
print("rank quantiles total/name/addr:", [rr[c].quantile(x) for c in ("brank", "brank_name", "brank_addr") for x in (0.25, 0.5, 0.75)])
# per-S1 candidate volume at these depths
n = q.group_by("s1_idx").len()["len"]
print("candidates available per S1 (median/p90):", n.median(), n.quantile(0.9))
nk = r.filter(pl.col("brank").is_null()).join(s1.rename({"idx": "s1_idx"}), on="s1_idx").head(8)
print(nk.select("s1_idx", "name_core", "addr"))
