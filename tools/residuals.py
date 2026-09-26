"""Residual anatomy on a run's val set (decision = threshold t, exclusivity on).

  python residuals.py <run_id> [thr]
FP origin (belongs to another S1 vs pure distractor), zero-TP S1s, FN p-bins, by source / n_true.
"""
import sys

import polars as pl

sys.path.insert(0, r"D:\amazon-ml\code\business_entity_resolution\src")
import pipeline as pp  # noqa: E402

pl.Config.set_tbl_rows(30)
pl.Config.set_tbl_width_chars(200)
R = pp.RUNS_DIR / sys.argv[1]
thr = float(sys.argv[2]) if len(sys.argv) > 2 else 0.7
vt = pl.read_parquet(R / "val_truth.parquet")
va = pl.read_parquet(R / "val_scored.parquet")
gt_all = pl.read_parquet(pp.CACHE_DIR / "gt_long.parquet").drop_nulls()
sel = pp.apply_decision(va, {"mode": "threshold", "param": thr, "excl": True}, 0.02)

# ---- FP origin
fp = sel.filter(pl.col("label") == 0)
ids = (pp.scan_norm("train", "pool").filter(pl.col("idx").is_in(fp["cand_idx"].unique().implode()))
       .select(pl.col("idx").alias("cand_idx"), pl.col("entity_id").alias("match_id"), "src").collect())
owner = gt_all.select(pl.col("match_id"), pl.col("s1_id").alias("owner"))
fp = fp.join(ids, on="cand_idx").join(owner, on="match_id", how="left")
val_ids = vt["s1_id"].implode()
fp = fp.with_columns(pl.when(pl.col("owner").is_null()).then(pl.lit("distractor(no S1)"))
                     .when(pl.col("owner").is_in(val_ids)).then(pl.lit("owned by other VAL S1"))
                     .otherwise(pl.lit("owned by non-val S1")).alias("fp_kind"))
print(f"FP pairs {fp.height} of {sel.height} selected")
print(fp.group_by("fp_kind").agg(pl.len(), pl.col("p").mean().alias("mean_p")).sort("len", descending=True))
print("FP by src:", dict(fp.group_by("src").len().iter_rows()))

# ---- per-S1
agg = (vt.select("s1_idx", "n_true", "country_n")
       .join(sel.group_by("s1_idx").agg(pl.len().alias("n_pred"), pl.col("label").sum().alias("tp")), on="s1_idx", how="left")
       .join(va.group_by("s1_idx").agg(pl.col("label").sum().alias("in_cands"), pl.col("p").max().alias("pmax"),
                                        pl.col("p").filter(pl.col("label") == 1).max().alias("pmax_pos")), on="s1_idx", how="left")
       .fill_null(0))
z = agg.filter((pl.col("n_true") > 0) & (pl.col("tp") == 0))
print(f"\nzero-TP S1 with truth: {z.height}")
print(z.with_columns(pl.when(pl.col("n_pred") > 0).then(pl.lit("pred wrong"))
                     .when(pl.col("in_cands") == 0).then(pl.lit("empty: none in cands"))
                     .otherwise(pl.lit("empty: pos in cands, p low")).alias("k"))
      .group_by("k").agg(pl.len(), pl.col("n_true").mean().alias("n_true"), pl.col("pmax_pos").mean().alias("mean_pmax_pos"))
      .sort("len", descending=True))
print("zero-TP n_true dist:", dict(z.group_by("n_true").len().sort("n_true").iter_rows()))

# ---- FN p bins
pos = va.filter(pl.col("label") == 1).join(sel.select("s1_idx", "cand_idx", pl.lit(1).alias("ch")), on=["s1_idx", "cand_idx"], how="left")
fn = pos.filter(pl.col("ch").is_null())
print(f"\nin-cand positives {pos.height}, missed by decision {fn.height}")
print(fn.with_columns(pl.col("p").cut([0.05, 0.2, 0.4, 0.6, 0.7], labels=["<.05", ".05-.2", ".2-.4", ".4-.6", ".6-.7", ">=.7(excl lost)"]).alias("bin"))
      .group_by("bin").len().sort("bin"))
# how many FNs lost only to exclusivity (p >= thr but another S1 won the cand)
print("FN with p>=thr (lost to exclusivity):", fn.filter(pl.col("p") >= thr).height)

# ---- per n_true bucket
f = agg.with_columns(pl.when((pl.col("n_true") == 0) & (pl.col("n_pred") == 0)).then(1.0)
                     .otherwise(1.25 * pl.col("tp") / (0.25 * pl.col("n_true") + pl.col("n_pred")).clip(1e-9)).alias("f"))
print("\nF by n_true:")
print(f.group_by(pl.col("n_true").clip(0, 7)).agg(pl.len(), pl.col("f").mean(), (1 - pl.col("f")).sum().alias("loss")).sort("n_true"))
print("F by country:", dict(f.group_by("country_n").agg(pl.col("f").mean()).iter_rows()))
