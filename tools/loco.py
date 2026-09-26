"""Leave-one-country-out proxy for France: train on SRC country only, evaluate on TGT country val.
Then self-training: pseudo-label TGT's non-val (unlabelled here) pairs with the SRC model and retrain.

  python loco.py <run_with_train_feats> [src=us] [tgt=india]
All decisions: threshold + exclusivity tuned on SRC val (the only labelled val we would have for France).
"""
import sys

import numpy as np
import polars as pl
import xgboost as xgb

sys.path.insert(0, r"D:\amazon-ml\code\business_entity_resolution\src")
import pipeline as pp  # noqa: E402
from features import FEATURES  # noqa: E402

R = pp.RUNS_DIR / sys.argv[1]
SRC = sys.argv[2] if len(sys.argv) > 2 else "us"
TGT = sys.argv[3] if len(sys.argv) > 3 else "india"
cols = ["s1_idx", "cand_idx", "label", "is_val", "is_es", *FEATURES]
df = pl.concat([pl.read_parquet(f, columns=cols) for f in sorted((R / "train_feats").glob("part-*.parquet"))])
cty = pp.scan_norm("train", "s1").select(pl.col("idx").alias("s1_idx"), "country_n").collect()
df = df.join(cty, on="s1_idx")
vt = pl.read_parquet(R / "val_truth.parquet")
PRM = dict(objective="binary:logistic", eval_metric="logloss", tree_method="hist", device="cuda", eta=0.05, max_depth=8,
           min_child_weight=5, subsample=0.8, colsample_bytree=0.8, max_bin=256, seed=42)


class _Q:
    def log(self, m):
        pass


def fit(tr: pl.DataFrame, es: pl.DataFrame, w: np.ndarray | None = None):
    d = xgb.DMatrix(tr.select(FEATURES).to_numpy(), tr["label"].to_numpy(), weight=w, feature_names=FEATURES)
    e = xgb.DMatrix(es.select(FEATURES).to_numpy(), es["label"].to_numpy(), feature_names=FEATURES)
    return xgb.train(PRM, d, 4000, evals=[(e, "es")], early_stopping_rounds=100, verbose_eval=False)


def pred(b, d: pl.DataFrame) -> pl.DataFrame:
    p = b.predict(xgb.DMatrix(d.select(FEATURES).to_numpy(), feature_names=FEATURES), iteration_range=(0, b.best_iteration + 1))
    return d.select("s1_idx", "cand_idx", "label").with_columns(pl.Series("p", p, dtype=pl.Float32))


def f05(b, name: str) -> None:
    """decision tuned on SRC val, applied to TGT val (what happens to France); plus TGT-oracle tuning for reference."""
    out = {}
    for c in (SRC, TGT):
        v = df.filter(pl.col("is_val") & (pl.col("country_n") == c))
        tvc = vt.filter(pl.col("country_n") == c).select("s1_idx", "n_true")
        out[c] = (pred(b, v), tvc)
    dec = max(pp.tune_decision(out[SRC][0], out[SRC][1], 0.02, _Q()).values(), key=lambda d: d["f05"])
    tgt = pp.eval_selection(pp.apply_decision(out[TGT][0], dec, 0.02), out[TGT][1])["f05"]
    oracle = max(pp.tune_decision(out[TGT][0], out[TGT][1], 0.02, _Q()).values(), key=lambda d: d["f05"])["f05"]
    print(f"{name:28s} {SRC} val {dec['f05']:.5f} | {TGT} val with {SRC}-tuned decision {tgt:.5f} "
          f"(oracle-tuned {oracle:.5f})", flush=True)


tr_all = df.filter(~pl.col("is_val") & ~pl.col("is_es"))
es_all = df.filter(pl.col("is_es"))
src_tr, src_es = tr_all.filter(pl.col("country_n") == SRC), es_all.filter(pl.col("country_n") == SRC)
tgt_pool = tr_all.filter(pl.col("country_n") == TGT)  # labels exist but are NOT used below except as reference
print(f"src train pairs {src_tr.height:,}, tgt unlabelled pairs {tgt_pool.height:,}", flush=True)

b_all = fit(tr_all, es_all)
f05(b_all, "trained on both (reference)")
b_src = fit(src_tr, src_es)
f05(b_src, f"trained on {SRC} only")

# self-training: pseudo-label confident TGT pairs with the SRC model, add them, retrain
pt = pred(b_src, tgt_pool)
for lo, hi, wgt in ((0.03, 0.97, 1.0), (0.1, 0.9, 1.0), (0.03, 0.97, 0.5)):
    ps = tgt_pool.join(pt.select("s1_idx", "cand_idx", "p"), on=["s1_idx", "cand_idx"]).filter(
        (pl.col("p") <= lo) | (pl.col("p") >= hi)).with_columns((pl.col("p") >= hi).cast(pl.Int8).alias("label"))
    acc = (ps["label"] == tgt_pool.join(ps.select("s1_idx", "cand_idx"), on=["s1_idx", "cand_idx"], how="semi")
           .sort(["s1_idx", "cand_idx"])["label"]).mean() if False else None
    both = pl.concat([src_tr.select(cols + ["country_n"]), ps.select(cols + ["country_n"])])
    w = np.concatenate([np.ones(src_tr.height, np.float32), np.full(ps.height, wgt, np.float32)])
    f05(fit(both, src_es, w), f"self-train p<={lo}|>={hi} w={wgt}")
