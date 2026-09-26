import sys
import polars as pl
sys.path.insert(0, r"D:\amazon-ml\code\business_entity_resolution\src")
import pipeline as pp
pl.Config.set_tbl_rows(40); pl.Config.set_fmt_str_lengths(46); pl.Config.set_tbl_width_chars(240)
sc = pl.read_parquet(r"runs/20260926-094122-v7/pred/scored-*.parquet")
s1 = pp.scan_norm("test", "s1").select(pl.col("idx").alias("s1_idx"), "country_n", pl.col("name_full").alias("n1"), pl.col("addr").alias("a1")).collect()
sc = sc.join(s1.select("s1_idx", "country_n"), on="s1_idx")
g = sc.group_by("s1_idx", "country_n").agg(pl.col("p").max().alias("pmax"), ((pl.col("p") > 0.2) & (pl.col("p") < 0.8)).sum().alias("unc"),
                                            (pl.col("p") >= 0.7).sum().alias("k"))
print(g.group_by("country_n").agg(pl.col("unc").mean().round(3), (pl.col("pmax") < 0.7).mean().round(4).alias("no_confident"),
      pl.col("k").mean().round(3)).sort("country_n"))
fr = sc.filter((pl.col("country_n") == "france") & (pl.col("p") > 0.25) & (pl.col("p") < 0.75))
ids = fr["s1_idx"].unique().sample(8, seed=11)
ex = sc.filter(pl.col("s1_idx").is_in(ids.implode()) & (pl.col("p") > 0.05))
po = pp.scan_norm("test", "pool").filter(pl.col("idx").is_in(ex["cand_idx"].implode())).select(pl.col("idx").alias("cand_idx"), pl.col("name_full").alias("n2"), pl.col("addr").alias("a2")).collect()
ex = ex.join(po, on="cand_idx").join(s1.select("s1_idx", "n1", "a1"), on="s1_idx").sort(["s1_idx", "p"], descending=[False, True])
print(ex.select("s1_idx", pl.col("p").round(3), "n1", "n2", "a1", "a2"))
