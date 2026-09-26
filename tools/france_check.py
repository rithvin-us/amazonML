"""Label-free per-country confidence on test predictions + uncertain France examples."""
import sys

import polars as pl

run = sys.argv[1] if len(sys.argv) > 1 else "20260926-005609-v2gpu_lite"
sc = pl.read_parquet(f"runs/{run}/pred/scored-*.parquet")
s1 = pl.read_parquet("cache/raw_test_s1.parquet").with_row_index("s1_idx").with_columns(pl.col("s1_idx").cast(pl.Int64))
sc = sc.join(s1.select("s1_idx", "country"), on="s1_idx")
top = sc.group_by("s1_idx", "country").agg(pl.col("p").max().alias("pmax"), (pl.col("p") >= 0.7).sum().alias("k"),
                                           ((pl.col("p") > 0.3) & (pl.col("p") < 0.7)).sum().alias("unc"))
allc = s1.group_by("country").len()
print(top.group_by("country").agg(
    pl.len().alias("s1_with_p>=.02"), (pl.col("pmax") >= 0.9).mean().alias("top>=.9"),
    ((pl.col("pmax") >= 0.3) & (pl.col("pmax") < 0.7)).mean().alias("top_in_.3-.7"),
    pl.col("unc").mean().alias("uncertain_pairs/S1"), pl.col("k").mean().alias("k_pred"))
    .join(allc, on="country").sort("country"))
# uncertain France pairs with text
pool = pl.concat([pl.scan_parquet("cache/raw_test_s2.parquet"), pl.scan_parquet("cache/raw_test_s3.parquet")])
fr = sc.filter((pl.col("country") == "France") & (pl.col("p") > 0.3) & (pl.col("p") < 0.7)).sample(12, seed=7)
txt = pool.join(fr.lazy().select(pl.col("cid").alias("entity_id")), on="entity_id", how="semi").collect()
fr = (fr.join(s1.select("s1_idx", pl.col("business_name").alias("n1"), pl.col("business_address").alias("a1")), on="s1_idx")
      .join(txt.select(pl.col("entity_id").alias("cid"), pl.col("business_name").alias("n2"),
                       pl.col("business_address").alias("a2")), on="cid"))
pl.Config.set_tbl_rows(20); pl.Config.set_fmt_str_lengths(55); pl.Config.set_tbl_width_chars(260)
print(fr.select(pl.col("p").round(2), "n1", "n2", "a1", "a2"))
