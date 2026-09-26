"""Covariate-shift weighting on the unseen-country proxy (train SRC, evaluate TGT val).

Domain classifier on features only (no labels): P(TGT | x) from SRC train pairs vs TGT non-val pairs.
w = odds ratio, clipped. (1) weighted matcher training on SRC; (2) decision tuned on w-weighted SRC val
(S1 weight = mean pair weight) instead of plain SRC val.

  python loco_w.py <run_with_train_feats> [src=us] [tgt=india]
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
df = df.join(pp.scan_norm("train", "s1").select(pl.col("idx").alias("s1_idx"), "country_n").collect(), on="s1_idx")
vt = pl.read_parquet(R / "val_truth.parquet")
PRM = dict(objective="binary:logistic", eval_metric="logloss", tree_method="hist", device="cuda", eta=0.05, max_depth=8,
           min_child_weight=5, subsample=0.8, colsample_bytree=0.8, max_bin=256, seed=42)
# features that describe the pair / the S1 group; blocking ranks and counts carry most of the country shift
DOM = FEATURES


class _Q:
    def log(self, m):
        pass


def X(d):
    return d.select(FEATURES).to_numpy()


src_tr = df.filter(~pl.col("is_val") & ~pl.col("is_es") & (pl.col("country_n") == SRC))
src_es = df.filter(pl.col("is_es") & (pl.col("country_n") == SRC))
tgt_un = df.filter(~pl.col("is_val") & (pl.col("country_n") == TGT))  # labels NOT used
src_val = df.filter(pl.col("is_val") & (pl.col("country_n") == SRC))
tgt_val = df.filter(pl.col("is_val") & (pl.col("country_n") == TGT))

# ---- domain classifier (features only)
n = min(src_tr.height, tgt_un.height, 1_500_000)
a, b = src_tr.sample(n=n, seed=1), tgt_un.sample(n=n, seed=1)
Xd = np.vstack([a.select(DOM).to_numpy(), b.select(DOM).to_numpy()])
yd = np.r_[np.zeros(n), np.ones(n)]
perm = np.random.default_rng(0).permutation(len(yd))
cut = int(0.8 * len(yd))
dom = xgb.train({**PRM, "eval_metric": "auc"}, xgb.DMatrix(Xd[perm[:cut]], yd[perm[:cut]]), 300,
                evals=[(xgb.DMatrix(Xd[perm[cut:]], yd[perm[cut:]]), "ho")], verbose_eval=False)
print("domain classifier holdout AUC:", dom.eval(xgb.DMatrix(Xd[perm[cut:]], yd[perm[cut:]])), flush=True)


def weights(d: pl.DataFrame, clip=(0.1, 10.0)) -> np.ndarray:
    q = np.clip(dom.predict(xgb.DMatrix(d.select(DOM).to_numpy())), 1e-4, 1 - 1e-4)
    return np.clip(q / (1 - q), *clip).astype(np.float32)


def fit(tr, es, w=None):
    return xgb.train(PRM, xgb.DMatrix(X(tr), tr["label"].to_numpy(), weight=w), 4000,
                     evals=[(xgb.DMatrix(X(es), es["label"].to_numpy()), "es")], early_stopping_rounds=100, verbose_eval=False)


def scored(bst, d):
    return d.select("s1_idx", "cand_idx", "label").with_columns(
        pl.Series("p", bst.predict(xgb.DMatrix(X(d)), iteration_range=(0, bst.best_iteration + 1)), dtype=pl.Float32))


def weighted_tune(sv: pl.DataFrame, tvc: pl.DataFrame, s1w: pl.DataFrame) -> dict:
    """tune_decision, but the per-S1 F0.5 average is weighted by S1 weights (France-like validation)."""
    best = None
    for excl in (False, True):
        for t in pp.THRESHOLD_GRID:
            sel = pp.apply_decision(sv, {"mode": "threshold", "param": float(t), "excl": excl}, 0.02)
            agg = sel.group_by("s1_idx").agg(pl.len().alias("n_pred"), pl.col("label").sum().alias("tp"))
            f = (tvc.join(agg, on="s1_idx", how="left").fill_null(0).join(s1w, on="s1_idx", how="left")
                 .with_columns(pl.col("w").fill_null(1.0),
                               pl.when((pl.col("n_true") == 0) & (pl.col("n_pred") == 0)).then(1.0)
                               .otherwise(1.25 * pl.col("tp") / (0.25 * pl.col("n_true") + pl.col("n_pred")).clip(1e-9)).alias("f")))
            v = float((f["f"] * f["w"]).sum() / f["w"].sum())
            if best is None or v > best["f05"]:
                best = {"mode": "threshold", "param": float(t), "excl": excl, "f05": v}
    return best


def report(bst, name, s1w=None):
    tv_s = vt.filter(pl.col("country_n") == SRC).select("s1_idx", "n_true")
    tv_t = vt.filter(pl.col("country_n") == TGT).select("s1_idx", "n_true")
    sv, tvp = scored(bst, src_val), scored(bst, tgt_val)
    dec = weighted_tune(sv, tv_s, s1w) if s1w is not None else max(pp.tune_decision(sv, tv_s, 0.02, _Q()).values(), key=lambda d: d["f05"])
    f = pp.eval_selection(pp.apply_decision(tvp, dec, 0.02), tv_t)["f05"]
    print(f"{name:34s} {TGT} val {f:.5f}  (decision {dec['mode']}:{dec['param']} excl={dec['excl']})", flush=True)


b0 = fit(src_tr, src_es)
report(b0, f"{SRC}-only, plain")
s1w = src_val.select("s1_idx").with_columns(pl.Series("w", weights(src_val))).group_by("s1_idx").agg(pl.col("w").mean())
report(b0, f"{SRC}-only, France-like tuning", s1w)
bw = fit(src_tr, src_es, weights(src_tr))
report(bw, f"{SRC}-only weighted train")
report(bw, f"{SRC}-only weighted train + tuning", s1w)
