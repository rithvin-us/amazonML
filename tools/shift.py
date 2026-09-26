import sys
import polars as pl
sys.path.insert(0, r"D:\amazon-ml\code\business_entity_resolution\src")
import pipeline as pp
pl.Config.set_tbl_cols(20); pl.Config.set_tbl_width_chars(250)
R = pp.RUNS_DIR / "20260926-100316-v7_test"
cols = ["s1_idx", "rr", "rr_rank", "bscore", "bscore_norm", "bname", "baddr", "c_name_cnt", "s1_same_name", "c_oov_frac",
        "nc_tset", "ad_tset", "ad_core_tset", "n_cands", "hn_both", "addr_empty_any", "anc_nc_mean"]
df = pl.concat([pl.read_parquet(f, columns=cols) for f in sorted((R / "test_feats").glob("part-*.parquet"))])
cty = pp.scan_norm("test", "s1").select(pl.col("idx").alias("s1_idx"), "country_n").collect()
df = df.join(cty, on="s1_idx")
top = df.filter(pl.col("rr_rank") == 1)
agg = lambda d: d.group_by("country_n").agg([pl.col(c).median().round(3).alias(c) for c in cols[1:] if c != "rr_rank"]).sort("country_n")
print("median over rank-1 (best re-ranked) candidate per S1:"); print(agg(top))
print("median over ALL candidates:"); print(agg(df))
