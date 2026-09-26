import sys, polars as pl
rd = lambda f: pl.read_csv(f, separator="\t", quote_char=None, infer_schema=False).with_columns(
    pl.col("matched_entity_ids").fill_null("").str.split(",").list.eval(pl.element().filter(pl.element() != "")).list.sort())
a, b = rd(sys.argv[1]), rd(sys.argv[2])
s1 = pl.read_parquet("cache/raw_test_s1.parquet", columns=["entity_id", "country"]).rename({"entity_id": "source1_entity_id"})
j = a.join(b, on="source1_entity_id", suffix="_b").join(s1, on="source1_entity_id")
print(j.group_by("country").agg((pl.col("matched_entity_ids") != pl.col("matched_entity_ids_b")).mean().round(4).alias("changed"),
      pl.col("matched_entity_ids").list.len().mean().round(3).alias("k_a"), pl.col("matched_entity_ids_b").list.len().mean().round(3).alias("k_b")).sort("country"))
