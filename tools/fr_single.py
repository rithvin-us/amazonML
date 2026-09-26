import sys, polars as pl
sys.path.insert(0, r"D:\amazon-ml\code\business_entity_resolution\src")
import pipeline as pp
pl.Config.set_tbl_rows(30); pl.Config.set_fmt_str_lengths(44); pl.Config.set_tbl_width_chars(230)
PR = pp.RUNS_DIR / sys.argv[1]
c = sys.argv[2] if len(sys.argv) > 2 else "france"
s1 = pp.scan_norm("test", "s1").filter(pl.col("country_n") == c).select(pl.col("idx").alias("s1_idx"), pl.col("name_full").alias("n1"), pl.col("addr").alias("a1")).collect()
sc = pl.read_parquet(PR / "pred" / "scored-*.parquet").join(s1.select("s1_idx"), on="s1_idx")
sel = pp.apply_decision(sc, {"mode": "thr_top1", "param": [0.575, 0.5], "excl": True}, 0.02)
one = sel.group_by("s1_idx").agg(pl.len().alias("k")).filter(pl.col("k") == 1)["s1_idx"]
ex = sc.filter(pl.col("s1_idx").is_in(one.sample(10, seed=3).implode()) & (pl.col("p") >= 0.05)).sort(["s1_idx", "p"], descending=[False, True])
po = pp.scan_norm("test", "pool").filter(pl.col("idx").is_in(ex["cand_idx"].implode())).select(pl.col("idx").alias("cand_idx"), pl.col("name_full").alias("n2"), pl.col("addr").alias("a2"), "src").collect()
print(ex.join(po, on="cand_idx").join(s1, on="s1_idx").select("s1_idx", pl.col("p").round(3), "src", "n1", "n2", "a1", "a2"))
