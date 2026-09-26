"""Loss breakdown on a run's val set: blocking misses vs model FN vs FP, + examples.

  python err_analysis.py <run_id> [thr] [excl]
"""
import sys

import polars as pl

sys.path.insert(0, r"D:\amazon-ml\code\business_entity_resolution\src")
import pipeline as pp  # noqa: E402

pl.Config.set_tbl_rows(40)
pl.Config.set_fmt_str_lengths(60)
pl.Config.set_tbl_width_chars(250)
R = pp.RUNS_DIR / sys.argv[1]
thr = float(sys.argv[2]) if len(sys.argv) > 2 else 0.7
excl = (sys.argv[3] == "1") if len(sys.argv) > 3 else True

vt = pl.read_parquet(R / "val_truth.parquet")
va = pl.read_parquet(R / "val_scored.parquet")
gt = pl.read_parquet(pp.CACHE_DIR / "gt_long.parquet").filter(pl.col("s1_id").is_in(vt["s1_id"].implode())).drop_nulls()
need = pl.concat([gt["match_id"], pl.Series(values=[], dtype=pl.String)]).unique()
pool = (pp.scan_norm("train", "pool").select("idx", "entity_id", "name_full", "addr", "src")
        .filter(pl.col("entity_id").is_in(need.implode()) | pl.col("idx").is_in(va["cand_idx"].unique().implode()))
        .collect())
s1 = pp.scan_norm("train", "s1").filter(pl.col("entity_id").is_in(vt["s1_id"].implode())).select(
    pl.col("idx").alias("s1_idx"), "name_full", "addr").collect()

pos = (gt.join(vt.select("s1_idx", "s1_id", "country_n"), on="s1_id")
       .join(pool.select(pl.col("entity_id").alias("match_id"), pl.col("idx").alias("cand_idx"), "src"), on="match_id"))
cand_pos = va.filter(pl.col("label") == 1).select("s1_idx", "cand_idx", "p")
pos = pos.join(cand_pos, on=["s1_idx", "cand_idx"], how="left")
sel = pp.apply_decision(va, {"mode": "threshold", "param": thr, "excl": excl}, 0.02)
selk = sel.select("s1_idx", "cand_idx").with_columns(pl.lit(True).alias("chosen"))
pos = pos.join(selk, on=["s1_idx", "cand_idx"], how="left").with_columns(pl.col("chosen").fill_null(False))
pos = pos.with_columns(pl.when(pl.col("p").is_null()).then(pl.lit("block_miss"))
                       .when(~pl.col("chosen")).then(pl.lit("model_fn")).otherwise(pl.lit("tp")).alias("kind"))
print("positives by kind/src:\n", pos.group_by("kind", "src").len().sort("kind", "src"))
print("positives by kind/country:\n", pos.group_by("kind", "country_n").len().sort("kind", "country_n"))

# siblings: missed positive whose S1 has >=1 chosen TP (-> transitive pool-pool expansion could recover it)
tp_s1 = pos.filter(pl.col("kind") == "tp").select("s1_idx").unique().with_columns(pl.lit(True).alias("has_tp"))
bm = pos.filter(pl.col("kind") == "block_miss").join(tp_s1, on="s1_idx", how="left").with_columns(pl.col("has_tp").fill_null(False))
print(f"block misses {bm.height}: with a TP sibling {bm['has_tp'].sum()} ({bm['has_tp'].mean():.3f})")
fn = pos.filter(pl.col("kind") == "model_fn").join(tp_s1, on="s1_idx", how="left").with_columns(pl.col("has_tp").fill_null(False))
print(f"model FN {fn.height}: with a TP sibling {fn['has_tp'].sum()} ({fn['has_tp'].mean():.3f}); p quantiles",
      fn["p"].quantile(0.25), fn["p"].median(), fn["p"].quantile(0.75))

# per-S1 loss attribution
fp = sel.filter(pl.col("label") == 0)
agg = (vt.select("s1_idx", "n_true", "country_n")
       .join(sel.group_by("s1_idx").agg(pl.len().alias("n_pred"), pl.col("label").sum().alias("tp")), on="s1_idx", how="left")
       .fill_null(0)
       .join(pos.group_by("s1_idx").agg((pl.col("kind") == "block_miss").sum().alias("n_bm"),
                                        (pl.col("kind") == "model_fn").sum().alias("n_fn")), on="s1_idx", how="left")
       .fill_null(0)
       .with_columns(pl.when((pl.col("n_true") == 0) & (pl.col("n_pred") == 0)).then(1.0)
                     .otherwise(1.25 * pl.col("tp") / (0.25 * pl.col("n_true") + pl.col("n_pred")).clip(1e-9)).alias("f"))
       .with_columns((pl.col("n_pred") - pl.col("tp")).alias("n_fp")))
n = agg.height
print(f"val S1 {n}  F0.5 {agg['f'].mean():.5f}  total loss {(1 - agg['f']).sum():.1f}")
cat = agg.with_columns(pl.when(pl.col("n_true") == 0).then(pl.lit("singleton_fp"))
                       .when(pl.col("tp") == 0).then(pl.lit("no_tp"))
                       .when(pl.col("n_fp") > 0).then(pl.lit("has_fp"))
                       .when(pl.col("n_bm") > 0).then(pl.lit("block_miss_only"))
                       .when(pl.col("n_fn") > 0).then(pl.lit("model_fn_only")).otherwise(pl.lit("perfect")).alias("cat"))
print(cat.group_by("cat").agg(pl.len(), (1 - pl.col("f")).sum().alias("loss"), (1 - pl.col("f")).mean().alias("mean_loss"))
      .with_columns((pl.col("loss") / n).alias("F_pts_lost")).sort("loss", descending=True))
print("singletons:", agg.filter(pl.col("n_true") == 0).height, " n_true dist:",
      dict(agg.group_by("n_true").len().sort("n_true").head(8).iter_rows()))

# examples
ex = (bm.join(s1, on="s1_idx").join(pool.select(pl.col("entity_id").alias("match_id"), pl.col("name_full").alias("m_name"),
                                                pl.col("addr").alias("m_addr")), on="match_id"))
print("\nBLOCK MISS examples:")
print(ex.select("country_n", "src", "name_full", "m_name", "addr", "m_addr").sample(min(14, ex.height), seed=1))
fpx = (fp.join(s1, on="s1_idx").join(pool.select(pl.col("idx").alias("cand_idx"), pl.col("name_full").alias("c_name"),
                                                 pl.col("addr").alias("c_addr")), on="cand_idx"))
print("\nFALSE POSITIVE examples:")
print(fpx.select("p", "name_full", "c_name", "addr", "c_addr").sample(min(10, fpx.height), seed=1))
fnx = (fn.join(s1, on="s1_idx").join(pool.select(pl.col("entity_id").alias("match_id"), pl.col("name_full").alias("m_name"),
                                                 pl.col("addr").alias("m_addr")), on="match_id"))
print("\nMODEL FN examples:")
print(fnx.select("p", "has_tp", "name_full", "m_name", "addr", "m_addr").sample(min(10, fnx.height), seed=1))
