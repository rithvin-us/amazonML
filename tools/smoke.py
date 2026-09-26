"""Smoke test: v3 blocking + features on a small India slice from the v3 train cache."""
import sys
import time

sys.path.insert(0, r"D:\amazon-ml\code\business_entity_resolution\src")
import polars as pl  # noqa: E402

from blocking import BlockIndex  # noqa: E402
from features import FEATURES, add_name_counts, build_features  # noqa: E402
from pipeline import NORM_COLS, scan_norm  # noqa: E402

pool = scan_norm("train", "pool").filter(pl.col("country_n") == "india").select(NORM_COLS).head(300_000).collect()
gt = pl.read_parquet(r"D:\amazon-ml\cache\gt_long.parquet").drop_nulls()
ids = gt.join(pool.select(pl.col("entity_id").alias("match_id")), on="match_id", how="semi")["s1_id"].unique().head(2000)
s1 = scan_norm("train", "s1").filter(pl.col("entity_id").is_in(ids.implode())).select(NORM_COLS).collect()
print("pool", pool.height, "s1", s1.height)
t = time.time()
ix = BlockIndex(pool, max_df=600, chunk=100_000)
pairs = ix.query(s1, top_k=40, k_name=10, k_addr=5)
print(f"query {time.time() - t:.1f}s pairs {pairs.height} per s1 {pairs.height / s1.height:.1f}")
print(pairs.describe().select("statistic", "brank", "brank_name", "brank_addr", "bname", "baddr"))
s1, pool = add_name_counts([s1, pool], pool, scan_norm("train", "s1").filter(pl.col("country_n") == "india"))
f = build_features(pairs, s1, pool, workers=4)
missing = [c for c in FEATURES if c not in f.columns]
print("missing", missing, "nulls", {c: f[c].null_count() for c in FEATURES if f[c].null_count()})
pos = gt.join(s1.select(pl.col("entity_id").alias("s1_id"), pl.col("idx").alias("s1_idx")), on="s1_id") \
    .join(pool.select(pl.col("entity_id").alias("match_id"), pl.col("idx").alias("cand_idx")), on="match_id")
hit = pos.join(pairs, on=["s1_idx", "cand_idx"], how="semi").height
old = pos.join(pairs.filter(pl.col("brank") <= 40), on=["s1_idx", "cand_idx"], how="semi").height
print(f"recall(slice) all-channels {hit / pos.height:.4f} vs top40-only {old / pos.height:.4f} ({pos.height} true pairs)")
print(f.select(["ncc_partial", "nsk_tset", "hn_eq", "c_name_ratio", "s1_same_name", "bname_norm", "brank_name"]).describe())
