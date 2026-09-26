import sys
import polars as pl
sys.path.insert(0, r"D:\amazon-ml\code\business_entity_resolution\src")
import pipeline as pp
pl.Config.set_tbl_rows(25); pl.Config.set_fmt_str_lengths(48); pl.Config.set_tbl_width_chars(250)
R = pp.RUNS_DIR / sys.argv[1]
vt = pl.read_parquet(R / "val_truth.parquet"); va = pl.read_parquet(R / "val_scored.parquet")
gt = pl.read_parquet(pp.CACHE_DIR / "gt_long.parquet").drop_nulls()
sel = pp.apply_decision(va, {"mode": "threshold", "param": 0.7, "excl": True}, 0.02)
fp = sel.filter((pl.col("label") == 0) & (pl.col("p") > 0.9))
pool = pp.scan_norm("train", "pool").filter(pl.col("idx").is_in(fp["cand_idx"].implode())).select(
    pl.col("idx").alias("cand_idx"), pl.col("entity_id").alias("mid"), pl.col("name_full").alias("c_name"), pl.col("addr").alias("c_addr")).collect()
fp = fp.join(pool, on="cand_idx").join(gt.select(pl.col("match_id").alias("mid"), "s1_id"), on="mid", how="anti")
s1 = pp.scan_norm("train", "s1").filter(pl.col("idx").is_in(fp["s1_idx"].implode())).select(
    pl.col("idx").alias("s1_idx"), "name_full", "addr").collect()
# the S1's own true matches (to compare distractor vs real)
print(f"distractor FPs p>0.9: {fp.height}")
print(fp.join(s1, on="s1_idx").select(pl.col("p").round(3), "name_full", "c_name", "addr", "c_addr").sample(18, seed=3))
# uncertain band size
for lo, hi in ((0.05, 0.95), (0.05, 0.99), (0.02, 0.995)):
    b = va.filter((pl.col("p") >= lo) & (pl.col("p") < hi))
    print(f"band [{lo},{hi}): pairs {b.height:,} ({b.height / vt.height:.2f}/S1) pos {b['label'].sum():,} neg {(b.height - b['label'].sum()):,}")
