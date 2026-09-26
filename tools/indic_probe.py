import polars as pl
from unidecode import unidecode
INDIC = r"[\x{0900}-\x{0DFF}]"
gt = pl.scan_parquet("cache/gt_long.parquet").drop_nulls()
pool = pl.concat([pl.scan_parquet("cache/raw_train_s2.parquet"), pl.scan_parquet("cache/raw_train_s3.parquet")]) \
    .filter(pl.col("business_name").str.contains(INDIC)).select(pl.col("entity_id").alias("match_id"), pl.col("business_name").alias("pn"))
s1 = pl.scan_parquet("cache/raw_train_s1.parquet").select(pl.col("entity_id").alias("s1_id"), pl.col("business_name").alias("sn"))
d = pool.join(gt, on="match_id").join(s1, on="s1_id").head(200000).collect()
print("pairs", d.height)
d = d.with_columns(nt_p=pl.col("pn").str.split(" ").list.len(), nt_s=pl.col("sn").str.split(" ").list.len())
print("same token count frac", (d["nt_p"] == d["nt_s"]).mean())
for r in d.sample(12, seed=3).iter_rows(named=True):
    print(f"{r['sn'][:40]:40s} | {r['pn'][:40]} | {unidecode(r['pn'])[:45]}")
