import polars as pl
s1 = pl.read_parquet("cache/raw_test_s1.parquet", columns=["entity_id", "country"])
m = pl.read_csv("output/matching_results.tsv", separator="\t", quote_char=None, infer_schema=False)
m = m.rename({m.columns[0]: "entity_id", m.columns[1]: "ids"}).join(s1, on="entity_id")
m = m.with_columns(k=pl.col("ids").fill_null("").str.split(",").list.eval(pl.element().filter(pl.element() != "")).list.len())
print(m.group_by("country").agg(pl.len(), (pl.col("k") == 0).mean().alias("empty"), pl.col("k").mean().alias("k_mean"),
      pl.col("k").quantile(0.5).alias("k_p50")).sort("country"))
# pool per-country size vs S1 -> expected matches per S1 (train: 7.64M pairs / 2.2M s1 = 3.46)
for sp in ["train", "test"]:
    p = pl.concat([pl.scan_parquet(f"cache/raw_{sp}_s2.parquet"), pl.scan_parquet(f"cache/raw_{sp}_s3.parquet")]).group_by("country").len().collect()
    s = pl.read_parquet(f"cache/raw_{sp}_s1.parquet", columns=["country"]).group_by("country").len()
    print(sp, p.join(s, on="country", suffix="_s1").with_columns(ratio=pl.col("len") / pl.col("len_s1")).sort("country"))
