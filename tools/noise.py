import polars as pl
for split in ["train", "test"]:
    pool = pl.concat([pl.scan_parquet(f"cache/raw_{split}_s2.parquet"), pl.scan_parquet(f"cache/raw_{split}_s3.parquet")])
    n = pool.select(pl.col("business_name").str.to_lowercase().alias("n"), pl.col("business_address").fill_null("").alias("a"))
    r = n.select(
        leet=pl.col("n").str.contains(r"[a-z][01][a-z]|[a-z][01]\b|\b[01][a-z]{2}").mean(),
        domain=pl.col("n").str.contains(r"\.(com|net|org|in|co|fr|io|biz)\b|^@|www\.").mean(),
        name_trail_num=pl.col("n").str.contains(r"\d{6,}\s*$").mean(),
        dotted=pl.col("n").str.contains(r"\b([a-z]\.){2,}").mean(),
        brackets=pl.col("n").str.contains(r"[\[\(]").mean(),
        addr_lead0=pl.col("a").str.contains(r"(^|[\s,#])0\d+").mean(),
        addr_null_tok=pl.col("a").str.to_lowercase().str.contains(r"\bnull\b").mean(),
        addr_empty=(pl.col("a").str.strip_chars() == "").mean(),
    ).collect()
    print(split, r.to_dicts()[0])
