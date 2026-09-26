import polars as pl
rd = lambda f, c: pl.read_csv(f, separator="\t", quote_char=None, infer_schema=False).with_columns(
    pl.col(c).fill_null("").str.split(",").list.eval(pl.element().filter(pl.element() != "")).alias(c))
m = rd("output/submissions/v7ce_matching_results.tsv", "matched_entity_ids")
c = rd("output/submissions/v7ce_candidate_pairs.tsv", "candidate_entity_ids")
s1 = pl.read_parquet("cache/raw_test_s1.parquet", columns=["entity_id", "country"])
j = m.join(c, on="source1_entity_id").join(s1.rename({"entity_id": "source1_entity_id"}), on="source1_entity_id")
print("rows", m.height, "unique S1", m["source1_entity_id"].n_unique(), "all test S1 present", j.height == s1.height)
print("matches not in candidates:", j.select(pl.col("matched_entity_ids").list.set_difference("candidate_entity_ids").list.len().sum()).item())
print("S1-prefixed ids in matches:", j.select(pl.col("matched_entity_ids").list.eval(pl.element().str.starts_with("S1-")).list.any().sum()).item())
ex = j.select("source1_entity_id", pl.col("matched_entity_ids").alias("id")).explode("id").drop_nulls()
print("pool ids matched to >1 S1 (exclusivity):", ex.group_by("id").len().filter(pl.col("len") > 1).height)
print(j.group_by("country").agg(pl.len().alias("s1"), pl.col("matched_entity_ids").list.len().mean().round(3).alias("matches_per_s1"),
      (pl.col("matched_entity_ids").list.len() == 0).mean().round(4).alias("empty_frac"),
      pl.col("candidate_entity_ids").list.len().mean().round(2).alias("cands_per_s1")).sort("country"))
v5 = rd("output/submissions/v5_matching_results.tsv", "matched_entity_ids").rename({"matched_entity_ids": "v5"})
a = m.join(v5, on="source1_entity_id")
print("identical lists vs v5 (LB 0.9685):", round(a.select((pl.col("matched_entity_ids").list.sort() == pl.col("v5").list.sort()).mean()).item(), 4))
