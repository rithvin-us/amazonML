"""Offline: does S1 context in the CE stacker help? Same val band + competitors as ce-apply; compares
stacker [lp, ce] vs [lp, ce, context...] by competitor-aware tuned val F0.5 (OOF by block).

  python stack_exp.py <train_run> <ce_dir>
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

TR, CE = pp.RUNS_DIR / sys.argv[1], Path(sys.argv[2])
score = ce._Scorer(CE)


class _Q:
    def log(self, m):
        pass


def with_ce(df, split):
    b = ce.texts(pp, ce._band(df).select("s1_idx", "cand_idx", "p", *[c for c in ("label",) if c in df.columns]), split)
    return b.with_columns(pl.Series("ce", score(b), dtype=pl.Float32)).drop("t1", "t2")


def ctx(full: pl.DataFrame, band: pl.DataFrame) -> pl.DataFrame:
    """S1 context from ALL scored pairs (p) and the band (ce)."""
    s1p = full.group_by("s1_idx").agg(pl.col("p").max().alias("s1_pmax"), (pl.col("p") >= 0.5).sum().alias("s1_nhi"))
    return (band.join(s1p, on="s1_idx", how="left").with_columns(
        (pl.col("p").log() - (1 - pl.col("p")).log()).alias("lp"),
        (pl.col("s1_pmax") - pl.col("p")).alias("p_gap"),
        pl.col("ce").rank("average", descending=True).over("s1_idx").alias("ce_rank"),
        pl.col("ce").max().over("s1_idx").alias("ce_max"),
        pl.len().over("s1_idx").alias("n_band")).with_columns((pl.col("ce_max") - pl.col("ce")).alias("ce_gap")))


BASE = ["lp", "ce"]
RICH = ["lp", "ce", "p_gap", "ce_rank", "ce_gap", "n_band", "s1_nhi"]
va = pl.read_parquet(TR / "val_scored.parquet")
vt = pl.read_parquet(TR / "val_truth.parquet")
comp = pl.read_parquet(TR / "comp_scored.parquet")
vb = ctx(va, with_ce(va, "train")).join(vt.select("s1_idx", "block"), on="s1_idx", how="left")
cb = ctx(comp, with_ce(comp, "train"))
tvc = vt.select("s1_idx", "n_true")
lab0 = lambda d: d.select("s1_idx", "cand_idx", pl.lit(0, pl.Int8).alias("label"), "p")  # noqa: E731
b0 = max(pp.tune_decision(pl.concat([va, lab0(comp)]), tvc, 0.02, _Q()).values(), key=lambda d: d["f05"])
print(f"stage-1 only: {b0['f05']:.5f}")
yb, grp = vb["label"].to_numpy(), vb["block"].fill_null("?").to_numpy()
for name, feats in (("base [lp, ce]", BASE), ("rich", RICH)):
    mono = "(" + ",".join("1" if f in ("lp", "ce") else "0" for f in feats) + ")"
    prm = {**ce.STACK, "monotone_constraints": mono}
    X, Xc = vb.select(feats).to_numpy(), cb.select(feats).to_numpy()
    oof = np.zeros(len(yb), np.float32)
    for trn, tst in GroupKFold(n_splits=5).split(X, yb, grp):
        oof[tst] = xgb.train(prm, xgb.DMatrix(X[trn], yb[trn]), 200).predict(xgb.DMatrix(X[tst]))
    full = xgb.train(prm, xgb.DMatrix(X, yb), 200)
    va2 = ce._restack(va, vb, oof)
    comp2 = ce._restack(comp, cb, full.predict(xgb.DMatrix(Xc)))
    b = max(pp.tune_decision(pl.concat([va2, lab0(comp2)]), tvc, 0.02, _Q()).values(), key=lambda d: d["f05"])
    print(f"{name}: {b['f05']:.5f} ({b['mode']}:{b['param']} excl={b['excl']})")
