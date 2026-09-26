import polars as pl
gt = pl.read_parquet("cache/gt_long.parquet").drop_nulls()
s1 = pl.read_parquet("cache/raw_train_s1.parquet", columns=["entity_id"]).with_row_index("r1")
s2 = pl.read_parquet("cache/raw_train_s2.parquet", columns=["entity_id"]).with_row_index("r2")
s3 = pl.read_parquet("cache/raw_train_s3.parquet", columns=["entity_id"]).with_row_index("r3")
pool = pl.concat([s2.rename({"r2": "r"}).with_columns(pl.lit("S2").alias("src")), s3.rename({"r3": "r"}).with_columns(pl.lit("S3").alias("src"))])
j = gt.join(s1.rename({"entity_id": "s1_id"}), on="s1_id").join(pool.rename({"entity_id": "match_id"}), on="match_id")
n1 = s1.height
for src, n in (("S2", s2.height), ("S3", s3.height)):
    d = j.filter(pl.col("src") == src).with_columns((pl.col("r1") / n1).alias("q1"), (pl.col("r") / n).alias("q2"))
    print(src, "row-position corr:", round(d.select(pl.corr("q1", "q2")).item(), 4))
num = lambda c: pl.col(c).str.extract(r"(\d+)$").cast(pl.Int64)
d = j.with_columns(num("s1_id").alias("a"), num("match_id").alias("b"))
print("id-number corr:", round(d.select(pl.corr("a", "b")).item(), 4), " same last3 digits:", round(d.select(((pl.col("a") % 1000) == (pl.col("b") % 1000)).mean()).item(), 5))
# sibling pool records of same S1: adjacent rows within a source?
g = j.filter(pl.col("src") == "S2").group_by("s1_id").agg(pl.col("r").sort().diff().drop_nulls().alias("gaps")).explode("gaps").drop_nulls()
print("S2 siblings row gap median:", g["gaps"].median(), " frac gap==1:", round((g["gaps"] == 1).mean(), 4))
