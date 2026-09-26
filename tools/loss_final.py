"""Loss breakdown of the FINAL pipeline (stage-1 + CE stack + competitor-aware decision) on val.

  python loss_final.py <train_run> <ce_dir>
Buckets per val S1 (F0.5 points lost / n_val): block miss, model FN, FP (pure distractor vs owned by another S1),
zero-TP, singleton FP; plus examples of the biggest bucket.
"""
import sys
from pathlib import Path

import numpy as np
import polars as pl
import xgboost as xgb
from sklearn.model_selection import GroupKFold

sys.path.insert(0, r"D:\amazon-ml\code\business_entity_resolution\src")
import cross_encoder as ce  # noqa: E402
import pipeline as pp  # noqa: E402

pl.Config.set_tbl_rows(40)
pl.Config.set_fmt_str_lengths(46)
pl.Config.set_tbl_width_chars(240)
TR, CE = pp.RUNS_DIR / sys.argv[1], Path(sys.argv[2])
score = ce._Scorer(CE)


class _Q:
    def log(self, m):
        pass


def with_ce(df, split):
    b = ce.texts(pp, ce._band(df).select("s1_idx", "cand_idx", "p", *[c for c in ("label",) if c in df.columns]), split)
    return b.with_columns(pl.Series("ce", score(b), dtype=pl.Float32)).drop("t1", "t2")


va, vt = pl.read_parquet(TR / "val_scored.parquet"), pl.read_parquet(TR / "val_truth.parquet")
comp = pl.read_parquet(TR / "comp_scored.parquet")
vb = with_ce(va, "train").join(vt.select("s1_idx", "block"), on="s1_idx", how="left")
cb = with_ce(comp, "train")
yb = vb["label"].to_numpy()
oof = np.zeros(len(yb), np.float32)
for trn, tst in GroupKFold(n_splits=5).split(ce._x(vb), yb, vb["block"].fill_null("?").to_numpy()):
    oof[tst] = xgb.train(ce.STACK, xgb.DMatrix(ce._x(vb)[trn], yb[trn]), 200).predict(xgb.DMatrix(ce._x(vb)[tst]))
full = xgb.train(ce.STACK, xgb.DMatrix(ce._x(vb), yb), 200)
va2 = ce._restack(va, vb, oof)
comp2 = ce._restack(comp, cb, full.predict(xgb.DMatrix(ce._x(cb))))
tvc = vt.select("s1_idx", "n_true")
uni = pl.concat([va2, comp2.select("s1_idx", "cand_idx", pl.lit(0, pl.Int8).alias("label"), "p")])
dec = max(pp.tune_decision(uni, tvc, 0.02, _Q()).values(), key=lambda d: d["f05"])
sel = pp.apply_decision(uni, dec, 0.02).join(vt.select("s1_idx"), on="s1_idx", how="semi")
print(f"final val F0.5 {dec['f05']:.5f}  decision {dec['mode']}:{dec['param']} excl={dec['excl']}")

# positives: blocked or not, selected or not
gt = pl.read_parquet(pp.CACHE_DIR / "gt_long.parquet").drop_nulls().join(vt.select("s1_idx", "s1_id"), on="s1_id")
ids = (pp.scan_norm("train", "pool").filter(pl.col("entity_id").is_in(gt["match_id"].implode()))
       .select(pl.col("entity_id").alias("match_id"), pl.col("idx").alias("cand_idx")).collect())
pos = gt.join(ids, on="match_id").select("s1_idx", "cand_idx")
inc = va2.filter(pl.col("label") == 1).select("s1_idx", "cand_idx", "p")
pos = (pos.join(inc, on=["s1_idx", "cand_idx"], how="left")
       .join(sel.select("s1_idx", "cand_idx", pl.lit(True).alias("chosen")), on=["s1_idx", "cand_idx"], how="left")
       .with_columns(pl.when(pl.col("p").is_null()).then(pl.lit("block_miss"))
                     .when(pl.col("chosen").is_null()).then(pl.lit("model_fn")).otherwise(pl.lit("tp")).alias("kind")))
agg = (vt.select("s1_idx", "n_true", "country_n")
       .join(sel.group_by("s1_idx").agg(pl.len().alias("n_pred"), pl.col("label").sum().alias("tp")), on="s1_idx", how="left")
       .fill_null(0)
       .join(pos.group_by("s1_idx").agg((pl.col("kind") == "block_miss").sum().alias("n_bm"),
                                        (pl.col("kind") == "model_fn").sum().alias("n_fn")), on="s1_idx", how="left").fill_null(0)
       .with_columns(pl.when((pl.col("n_true") == 0) & (pl.col("n_pred") == 0)).then(1.0)
                     .otherwise(1.25 * pl.col("tp") / (0.25 * pl.col("n_true") + pl.col("n_pred")).clip(1e-9)).alias("f"),
                     (pl.col("n_pred") - pl.col("tp")).alias("n_fp")))
n = agg.height
cat = agg.with_columns(pl.when(pl.col("n_true") == 0).then(pl.lit("singleton_fp"))
                       .when(pl.col("tp") == 0).then(pl.lit("zero_tp"))
                       .when(pl.col("n_fp") > 0).then(pl.lit("has_fp"))
                       .when(pl.col("n_bm") > 0).then(pl.lit("block_miss_only"))
                       .when(pl.col("n_fn") > 0).then(pl.lit("model_fn_only")).otherwise(pl.lit("perfect")).alias("cat"))
print(cat.group_by("cat").agg(pl.len(), ((1 - pl.col("f")).sum() / n).round(5).alias("F_lost")).sort("F_lost", descending=True))
print("positives:", dict(pos.group_by("kind").len().iter_rows()))
fp = sel.filter(pl.col("label") == 0)
print("selected FP pairs:", fp.height, " FN (in cands, not chosen):", pos.filter(pl.col("kind") == "model_fn").height)
s1t = pp.scan_norm("train", "s1").filter(pl.col("idx").is_in(vt["s1_idx"].implode())).select(
    pl.col("idx").alias("s1_idx"), pl.col("name_full").alias("n1"), pl.col("addr").alias("a1")).collect()


def show(df, title):
    po = pp.scan_norm("train", "pool").filter(pl.col("idx").is_in(df["cand_idx"].implode())).select(
        pl.col("idx").alias("cand_idx"), pl.col("name_full").alias("n2"), pl.col("addr").alias("a2")).collect()
    print(title)
    print(df.join(po, on="cand_idx").join(s1t, on="s1_idx").select(pl.col("p").round(3), "n1", "n2", "a1", "a2"))


show(pos.filter(pl.col("kind") == "model_fn").sample(10, seed=2), "MODEL FN examples:")
show(fp.sample(min(10, fp.height), seed=2), "FP examples:")
show(pos.filter(pl.col("kind") == "block_miss").sample(8, seed=2).with_columns(pl.lit(None, pl.Float32).alias("p")), "BLOCK MISS examples:")
