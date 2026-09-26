"""Val loss decomposition for latest run. Aggregates + a few examples only."""
import sys
import polars as pl

R = sys.argv[1]
TH = float(sys.argv[2]) if len(sys.argv) > 2 else 0.7
INDIC = r"[\x{0900}-\x{0DFF}]"

v = pl.read_parquet(f"{R}/val_scored.parquet")
s1n = pl.scan_parquet("cache/norm_train_s1_v3/*.parquet")
pool = pl.scan_parquet("cache/norm_train_pool_v3/*.parquet")
val_s1 = v.select("s1_idx").unique()
s1 = s1n.join(val_s1.lazy(), left_on="idx", right_on="s1_idx", how="semi").collect()
print("val s1", s1.height)

gt = (pl.scan_parquet("cache/gt_long.parquet")
      .join(s1.lazy().select(pl.col("entity_id").alias("s1_id"), pl.col("idx").alias("s1_idx")), on="s1_id")
      .drop_nulls("match_id").collect())
pid = pool.select("idx", "entity_id").join(gt.lazy().select(pl.col("match_id").alias("entity_id")), on="entity_id", how="semi").collect()
gt = gt.join(pid.rename({"entity_id": "match_id", "idx": "cand_idx"}), on="match_id", how="left")
print("gt pairs", gt.height, "unmapped", gt["cand_idx"].null_count())

pos = gt.select("s1_idx", "cand_idx").with_columns(pl.lit(1).alias("is_true"))
allp = (v.join(pos, on=["s1_idx", "cand_idx"], how="full", coalesce=True)
        .with_columns(pl.col("is_true").fill_null(0), pl.col("p").fill_null(-1.0)))
allp = allp.with_columns(
    pl.when((pl.col("is_true") == 1) & (pl.col("p") < 0)).then(pl.lit("block_miss"))
    .when((pl.col("is_true") == 1) & (pl.col("p") < TH)).then(pl.lit("model_fn"))
    .when((pl.col("is_true") == 1)).then(pl.lit("tp"))
    .when(pl.col("p") >= TH).then(pl.lit("fp"))
    .otherwise(pl.lit("tn")).alias("cat"))
print(allp.group_by("cat").len().sort("cat"))

# ---- oracle decomposition (macro F0.5 per S1, singleton empty=1)
def macro(df):
    g = df.group_by("s1_idx").agg(tp=pl.col("sel").cast(pl.Int32).dot(pl.col("is_true")),
                                  k=pl.col("sel").sum(), n=pl.col("is_true").sum())
    g = val_s1.join(g, on="s1_idx", how="left").fill_null(0)
    f = pl.when((pl.col("n") == 0) & (pl.col("k") == 0)).then(1.0).otherwise(
        1.25 * pl.col("tp") / (0.25 * pl.col("n") + pl.col("k")).clip(lower_bound=1e-9))
    return g.select(f.mean()).item()

base = allp.with_columns(sel=pl.col("cat").is_in(["tp", "fp"]))
a = macro(base)
b = macro(base.with_columns(sel=pl.col("cat") == "tp"))
c = macro(base.with_columns(sel=pl.col("cat").is_in(["tp", "model_fn"])))
d = macro(base.with_columns(sel=pl.col("is_true") == 1))
print(f"\nmacroF0.5 now={a:.4f} | -FP={b:.4f} (+{b-a:.4f}) | +modelFN={c:.4f} (+{c-b:.4f}) | +blockMiss={d:.4f} (+{d-c:.4f})")

# ---- attach attributes to error pairs
err = allp.filter(pl.col("cat").is_in(["block_miss", "model_fn", "fp", "tp"]))
pinfo = pool.join(err.lazy().select(pl.col("cand_idx").alias("idx")).unique(), on="idx", how="semi").collect()
raw = pl.concat([pl.scan_parquet("cache/raw_train_s2.parquet"), pl.scan_parquet("cache/raw_train_s3.parquet")])
raw = raw.join(pinfo.lazy().select("entity_id"), on="entity_id", how="semi").select(
    "entity_id", pl.col("business_name").alias("raw_name_c"), pl.col("business_address").alias("raw_addr_c")).collect()
pinfo = pinfo.join(raw, on="entity_id", how="left")
s1raw = pl.scan_parquet("cache/raw_train_s1.parquet").join(s1.lazy().select("entity_id"), on="entity_id", how="semi").select(
    "entity_id", pl.col("business_name").alias("raw_name_s1"), pl.col("business_address").alias("raw_addr_s1")).collect()
s1i = s1.join(s1raw, on="entity_id").select(pl.col("idx").alias("s1_idx"), pl.col("name_core").alias("nc_s1"),
                                           pl.col("addr").alias("ad_s1"), pl.col("postcode").alias("pc_s1"),
                                           pl.col("country_n").alias("cty_s1"), "raw_name_s1", "raw_addr_s1")
pi = pinfo.select(pl.col("idx").alias("cand_idx"), pl.col("name_core").alias("nc_c"), pl.col("addr").alias("ad_c"),
                  pl.col("postcode").alias("pc_c"), pl.col("country_n").alias("cty_c"), "src", "raw_name_c", "raw_addr_c")
e = err.join(s1i, on="s1_idx", how="left").join(pi, on="cand_idx", how="left")

tok = lambda c: pl.col(c).str.split(" ").list.unique()
e = e.with_columns(
    indic=pl.col("raw_name_c").str.contains(INDIC),
    indic_addr=pl.col("raw_addr_c").str.contains(INDIC),
    addr_empty=pl.col("raw_addr_c").fill_null("").str.strip_chars() == "",
    cty_diff=pl.col("cty_s1") != pl.col("cty_c"),
    name_tok_overlap=tok("nc_s1").list.set_intersection(tok("nc_c")).list.len(),
    pc_eq=(pl.col("pc_s1") == pl.col("pc_c")) & (pl.col("pc_s1") != ""),
)
print("\nper category (fractions):")
print(e.group_by("cat").agg(pl.len(), pl.col("indic").mean(), pl.col("indic_addr").mean(), pl.col("addr_empty").mean(),
                            pl.col("cty_diff").mean(), (pl.col("name_tok_overlap") == 0).mean().alias("no_name_tok"),
                            pl.col("pc_eq").mean(), (pl.col("src") == "s3").mean().alias("s3"),
                            (pl.col("cty_s1") == "india").mean().alias("india")).sort("cat"))

bm = e.filter(pl.col("cat") == "block_miss")
print("\nblock_miss buckets:")
print(bm.with_columns(
    b=pl.when(pl.col("indic")).then(pl.lit("indic_name"))
    .when(pl.col("cty_diff")).then(pl.lit("country_diff"))
    .when(pl.col("name_tok_overlap") == 0).then(pl.lit("latin_no_name_tok"))
    .otherwise(pl.lit("shares_name_tok(topK/maxdf cut)"))).group_by("b").len().sort("len", descending=True))

pl.Config.set_tbl_rows(40); pl.Config.set_fmt_str_lengths(70); pl.Config.set_tbl_width_chars(250)
for cat, filt in [("block_miss latin no-tok", (pl.col("cat") == "block_miss") & ~pl.col("indic") & (pl.col("name_tok_overlap") == 0)),
                  ("block_miss latin shares-tok", (pl.col("cat") == "block_miss") & ~pl.col("indic") & (pl.col("name_tok_overlap") > 0)),
                  ("model_fn", pl.col("cat") == "model_fn"), ("fp", pl.col("cat") == "fp")]:
    print(f"\n== {cat}")
    print(e.filter(filt).sample(n=min(8, e.filter(filt).height), seed=1).select(
        "p", "raw_name_s1", "raw_name_c", "raw_addr_s1", "raw_addr_c"))
