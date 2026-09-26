import polars as pl, sys
p = sys.argv[1]
df = pl.scan_csv(p, separator="\t", quote_char=None, infer_schema=False).select("business_name", "business_address", "country")
pats = {
    "devanagari": r"[\x{0900}-\x{097F}]", "bengali": r"[\x{0980}-\x{09FF}]", "tamil": r"[\x{0B80}-\x{0BFF}]",
    "telugu": r"[\x{0C00}-\x{0C7F}]", "kannada": r"[\x{0C80}-\x{0CFF}]", "gujarati": r"[\x{0A80}-\x{0AFF}]",
    "gurmukhi": r"[\x{0A00}-\x{0A7F}]", "malayalam": r"[\x{0D00}-\x{0D7F}]", "arabic": r"[\x{0600}-\x{06FF}]",
    "cjk": r"[\x{4E00}-\x{9FFF}]", "latin_accent": r"[\x{00C0}-\x{017F}]",
    "mojibake": r"(Ã.|à¤|à¥|Â.)",
}
out = df.select(
    [pl.col("business_name").str.contains(v).mean().alias(f"name_{k}") for k, v in pats.items()]
    + [pl.col("business_address").str.contains(pats["mojibake"]).mean().alias("addr_mojibake"),
       pl.col("business_address").str.contains(pats["devanagari"]).mean().alias("addr_devanagari"),
       pl.len().alias("n")]
).collect()
for c in out.columns:
    v = out[c][0]
    if v: print(f"{c:18s} {v:.4f}" if isinstance(v, float) else f"{c:18s} {v}")
