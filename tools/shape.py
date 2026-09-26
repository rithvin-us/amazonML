import polars as pl
rd = lambda f: pl.read_csv(f, separator="\t", quote_char=None, infer_schema=False).with_columns(
    pl.col("matched_entity_ids").fill_null("").str.split(",").list.eval(pl.element().filter(pl.element() != "")).alias("m"))
t = rd("output/submissions/v7fr_ce_matching_results.tsv").join(
    pl.read_parquet("cache/raw_test_s1.parquet", columns=["entity_id", "country"]).rename({"entity_id": "source1_entity_id"}), on="source1_entity_id")
t = t.with_columns(pl.col("m").list.len().alias("k"), pl.col("m").list.eval(pl.element().str.starts_with("S2")).list.sum().alias("k2"))
gt = pl.read_csv("student_resource/dataset/train/train_ground_truth.tsv", separator="\t", quote_char=None, infer_schema=False).rename({"matched_entity_ids": "matched_entity_ids"})
gt = rd.__call__ if False else gt.with_columns(pl.col("matched_entity_ids").fill_null("").str.split(",").list.eval(pl.element().filter(pl.element() != "")).alias("m"))
s1c = pl.read_parquet("cache/raw_train_s1.parquet", columns=["entity_id", "country"]).rename({"entity_id": "source1_entity_id"})
gt = gt.join(s1c, on="source1_entity_id").with_columns(pl.col("m").list.len().alias("k"), pl.col("m").list.eval(pl.element().str.starts_with("S2")).list.sum().alias("k2"))
dist = lambda d, tag: d.group_by("country").agg([(pl.col("k").clip(0, 7) == i).mean().round(3).alias(str(i)) for i in range(8)] +
        [(pl.col("k2") / pl.col("k")).mean().round(3).alias("S2share"), pl.col("k").mean().round(2).alias("mean_k")]).sort("country").with_columns(pl.lit(tag).alias("set"))
pl.Config.set_tbl_cols(14); pl.Config.set_tbl_width_chars(200)
print(pl.concat([dist(gt, "TRAIN truth"), dist(t, "TEST pred")]))
