"""Stage 2: re-score stage-1 probabilities with group context (post-hoc on a finished run, no re-blocking).

  python pipeline.py stage2 --run <run_id> [--promote]

Inputs (from the stage-1 run dir): val_scored.parquet (s1_idx, cand_idx, label, p), val_truth.parquet
(every val S1 with n_true + block), pred/scored-*.parquet (test pairs with p >= p_floor) and the norm caches.

Features per (S1, cand) pair with p1 >= P_MIN:
  S1 context     rank of p1 within the S1, p1/max, max, sum, #(p1>0.5), #cands, gap to next-lower p1
  competition    among S1s of the same block (country|city|name prefix, pipeline.s1_blocks): #S1 claiming the
                 pool record, this S1's rank among them, best p1 of any OTHER S1, #claimers with p1>0.3.
                 Block-local on val AND test so the feature means the same thing on both (val samples whole
                 blocks). Ground truth is exclusive: a pool record belongs to at most one S1.
  consistency    name/address similarity of the cand to the S1's best other candidate + that candidate's p1
                 (true matches of one entity look alike: an S1 has ~3.5 matches across S2/S3)
M2 = small monotone XGBoost on the block-sampled val S1, GroupKFold by block with an inner early-stopping
split -> out-of-fold p2 tunes the decision; test p2 = mean of the fold models (same calibration as OOF).
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.process import cpdist

from config import OUTPUT_DIR

P_MIN = 0.05
S2_FEATURES = [
    "p", "r_s1", "p_max", "p_sum", "n_hi", "n_s1", "p_ratio", "gap_next",
    "c_n", "c_rank", "c_other", "c_n_hi",
    "p_ref", "ref_nc_ratio", "ref_nc_tset", "ref_ad_tset", "ref_nf_tsort",
]
# p2 must not fall when p1 rises, nor rise when a competing S1 claims the record more strongly
MONOTONE = {"p": 1, "c_other": -1, "c_rank": -1}
TEXT = ["name_full", "name_core", "addr"]


def context_features(df: pl.DataFrame) -> pl.DataFrame:
    """df: s1_idx, cand_idx, p, block (+ passthrough). Adds S1-context, block-local competition, reference cand."""
    df = df.filter(pl.col("p") >= P_MIN).sort(["s1_idx", "p"], descending=[False, True])
    comp = ["cand_idx", "block"]
    df = df.with_columns(
        pl.int_range(1, pl.len() + 1).over("s1_idx").cast(pl.Float32).alias("r_s1"),
        pl.col("p").max().over("s1_idx").alias("p_max"),
        pl.col("p").sum().over("s1_idx").alias("p_sum"),
        (pl.col("p") > 0.5).sum().over("s1_idx").cast(pl.Float32).alias("n_hi"),
        pl.len().over("s1_idx").cast(pl.Float32).alias("n_s1"),
        (pl.col("p") - pl.col("p").shift(-1).over("s1_idx").fill_null(0.0)).alias("gap_next"),
        pl.len().over(comp).cast(pl.Float32).alias("c_n"),
        pl.col("p").rank("ordinal", descending=True).over(comp).cast(pl.Float32).alias("c_rank"),
        pl.col("p").max().over(comp).alias("_cmax"),
        pl.col("p").top_k(2).min().over(comp).alias("_c2"),
        (pl.col("p") > 0.3).sum().over(comp).cast(pl.Float32).alias("c_n_hi"),
    )
    df = df.with_columns(
        (pl.col("p") / pl.col("p_max")).alias("p_ratio"),
        pl.when(pl.col("c_n") == 1).then(0.0).when(pl.col("c_rank") == 1).then(pl.col("_c2"))
        .otherwise(pl.col("_cmax")).alias("c_other"),
    ).drop("_cmax", "_c2")
    # reference = the S1's best candidate other than this one
    tops = df.filter(pl.col("r_s1") <= 2).select("s1_idx", "r_s1", pl.col("cand_idx").alias("ref"),
                                                 pl.col("p").alias("p_ref"))
    t1 = tops.filter(pl.col("r_s1") == 1).drop("r_s1")
    t2 = tops.filter(pl.col("r_s1") == 2).drop("r_s1").rename({"ref": "ref2", "p_ref": "p_ref2"})
    df = df.join(t1, on="s1_idx", how="left").join(t2, on="s1_idx", how="left")
    is_top = pl.col("cand_idx") == pl.col("ref")
    return df.with_columns(pl.when(is_top).then(pl.col("ref2")).otherwise(pl.col("ref")).alias("ref"),
                           pl.when(is_top).then(pl.col("p_ref2")).otherwise(pl.col("p_ref")).fill_null(0.0)
                           .alias("p_ref")).drop("ref2", "p_ref2")


def _text(norm_pool: pl.LazyFrame, ids: pl.Series) -> pl.DataFrame:
    """Pool text for the given global idx; is_in filter pushes into the parquet scan (idx is monotonic)."""
    u = ids.unique().sort()
    return (norm_pool.select(["idx"] + TEXT).filter(pl.col("idx").is_between(u.min(), u.max()))
            .filter(pl.col("idx").is_in(u.implode())).collect())


def consistency_features(df: pl.DataFrame, norm_pool: pl.LazyFrame, workers: int,
                         chunk: int = 1_500_000) -> pl.DataFrame:
    """Similarity of each cand to its S1's reference cand (pool text looked up by idx, chunked by rows)."""
    out = []
    for i in range(0, df.height, chunk):
        d = df.slice(i, chunk)
        txt = _text(norm_pool, pl.concat([d["cand_idx"], d["ref"].drop_nulls()]))
        a = (d.select("cand_idx", "ref")
             .join(txt.rename({c: c + "_c" for c in ["idx"] + TEXT}), left_on="cand_idx", right_on="idx_c",
                   how="left", maintain_order="left")
             .join(txt.rename({c: c + "_r" for c in ["idx"] + TEXT}), left_on="ref", right_on="idx_r",
                   how="left", maintain_order="left"))
        g = lambda c: a[c].fill_null("").to_list()  # noqa: E731
        has = a["ref"].is_not_null().to_numpy()
        f = {}
        for name, col, scorer in (("ref_nc_ratio", "name_core", fuzz.ratio),
                                  ("ref_nc_tset", "name_core", fuzz.token_set_ratio),
                                  ("ref_ad_tset", "addr", fuzz.token_set_ratio),
                                  ("ref_nf_tsort", "name_full", fuzz.token_sort_ratio)):
            v = cpdist(g(col + "_c"), g(col + "_r"), scorer=scorer, workers=workers, dtype=np.float32)
            f[name] = np.where(has, v, -1.0).astype(np.float32)
        out.append(d.with_columns([pl.Series(k, v) for k, v in f.items()]))
        del txt, a
    return pl.concat(out)


def build(df: pl.DataFrame, norm_pool: pl.LazyFrame, workers: int) -> pl.DataFrame:
    df = consistency_features(context_features(df), norm_pool, workers)
    return df.with_columns(pl.col(S2_FEATURES).cast(pl.Float32))


def _params(dev: str, seed: int) -> dict:
    mono = "(" + ",".join(str(MONOTONE.get(f, 0)) for f in S2_FEATURES) + ")"
    return dict(objective="binary:logistic", eval_metric="logloss", tree_method="hist", device=dev, eta=0.05,
                max_depth=6, min_child_weight=10, subsample=0.8, colsample_bytree=0.9, reg_lambda=2.0,
                monotone_constraints=mono, seed=seed)


def run_stage2(src_dir: Path, run, device: str = "cuda", seed: int = 42, workers: int = 8,
               promote: bool = False, folds: int = 5, rounds: int = 3000) -> None:
    import xgboost as xgb
    from sklearn.model_selection import GroupKFold

    import pipeline as pp

    dev = pp.xgb_device(device)
    s1m = json.loads((src_dir / "metrics.json").read_text())
    run.set_metrics(stage1_run=src_dir.name, stage1_val_f05=s1m.get("val_f05"))

    # ---------------- val: features, CV, OOF decision
    run.start_stage("s2_val_features")
    vt = pl.read_parquet(src_dir / "val_truth.parquet")
    tvc = vt.select("s1_idx", "n_true")
    va = pl.read_parquet(src_dir / "val_scored.parquet").join(vt.select("s1_idx", "block"), on="s1_idx", how="left")
    va = build(va.with_columns(pl.col("block").fill_null("")), pp.scan_norm("train", "pool"), workers)
    run.log(f"val pairs (p1>={P_MIN}) {va.height:,}  pos {int(va['label'].sum()):,}  S1 {tvc.height:,}")
    run.end_stage()

    run.start_stage("s2_cv")
    X, y = va.select(S2_FEATURES).to_numpy(), va["label"].to_numpy()
    groups = va["block"].to_numpy()
    oof = np.zeros(len(y), np.float32)
    models, best_iters = [], []
    for k, (tr, te) in enumerate(GroupKFold(n_splits=folds).split(X, y, groups)):
        # inner early-stopping split by block hash (~10% of the training fold), never the held-out fold
        inner = (pl.Series(groups[tr]).hash(seed) % 10 == 0).to_numpy()
        dtr = xgb.DMatrix(X[tr][~inner], y[tr][~inner], feature_names=S2_FEATURES)
        des = xgb.DMatrix(X[tr][inner], y[tr][inner], feature_names=S2_FEATURES)
        b = xgb.train(_params(dev, seed), dtr, rounds, evals=[(des, "es")], early_stopping_rounds=100,
                      verbose_eval=False)
        b = b[: b.best_iteration + 1]
        models.append(b)
        best_iters.append(b.num_boosted_rounds())
        oof[te] = b.predict(xgb.DMatrix(X[te], feature_names=S2_FEATURES))
        run.progress((k + 1) / folds, f"fold {k + 1}/{folds} rounds {best_iters[-1]}")
    run.end_stage()

    run.start_stage("s2_tune")
    base = va.select("s1_idx", "cand_idx", "label", "p")
    r1 = pp.tune_decision(base, tvc, P_MIN, run)
    d1 = max(r1.values(), key=lambda d: d["f05"])
    s2 = base.with_columns(pl.Series("p", oof))
    r2 = pp.tune_decision(s2, tvc, P_MIN, run)
    d2 = max(r2.values(), key=lambda d: d["f05"])
    ex1 = pp.eval_selection(pp.exclusive(pp.apply_decision(base, d1, P_MIN)), tvc)
    ex2 = pp.eval_selection(pp.exclusive(pp.apply_decision(s2, d2, P_MIN)), tvc)
    run.set_metrics(val_f05_s1=round(d1["f05"], 5), val_f05_s1_excl=round(ex1["f05"], 5),
                    val_f05=round(d2["f05"], 5), val_f05_excl=round(ex2["f05"], 5),
                    val_precision=round(d2["precision"], 4), val_recall=round(d2["recall"], 4),
                    decision_mode=d2["mode"], decision_param=d2["param"], threshold=r2["threshold"]["param"],
                    s2_rounds=best_iters)
    run.log(f"metric stage1 val_f05={d1['f05']:.5f} excl={ex1['f05']:.5f} | stage2 OOF val_f05={d2['f05']:.5f} "
            f"excl={ex2['f05']:.5f} P={d2['precision']:.4f} R={d2['recall']:.4f} decision={d2['mode']}:{d2['param']}")
    gain: dict[str, float] = {}
    for b in models:
        for f_, v in b.get_score(importance_type="total_gain").items():
            gain[f_] = gain.get(f_, 0.0) + v / len(models)
    (run.dir / "feature_importance.json").write_text(json.dumps(
        dict(sorted({f: round(gain.get(f, 0.0), 1) for f in S2_FEATURES}.items(), key=lambda x: -x[1])), indent=2))
    for k, b in enumerate(models):
        b.save_model(str(run.dir / f"model_s2_fold{k}.json"))
    del va, X, y, oof, base, s2
    run.end_stage()

    # ---------------- test
    if not (src_dir / "pred").exists():
        run.log("no pred/ in stage-1 run (train-only) -> skipping test")
        return
    run.start_stage("s2_test")
    s1n = pp.scan_norm("test", "s1").select(pl.col("idx").alias("s1_idx"), "entity_id").collect()
    blocks = pp.s1_blocks("test").select("entity_id", "block").join(s1n, on="entity_id").drop("entity_id")
    te = pl.read_parquet(src_dir / "pred" / "scored-*.parquet")
    cids = te.select("cand_idx", "cid").unique("cand_idx")
    te = build(te.drop("cid").join(blocks, on="s1_idx", how="left").with_columns(pl.col("block").fill_null("")),
               pp.scan_norm("test", "pool"), workers)
    p2 = np.zeros(te.height, np.float32)
    for i in range(0, te.height, 2_000_000):
        dm = xgb.DMatrix(te.slice(i, 2_000_000).select(S2_FEATURES).to_numpy(), feature_names=S2_FEATURES)
        p2[i:i + dm.num_row()] = np.mean([b.predict(dm) for b in models], axis=0)
    te = te.select("s1_idx", "cand_idx").with_columns(pl.Series("p", p2)).join(cids, on="cand_idx", how="left")
    s1_map = s1n.select("s1_idx", pl.col("entity_id").alias("source1_entity_id"))
    sel = pp.apply_decision(te, d2, P_MIN)
    excl = pp.exclusive(sel)
    out = run.dir / "output"
    out.mkdir(exist_ok=True)
    pp._write_ids(s1_map, pp._join_ids(sel), "matched_entity_ids", out / "matching_results.tsv")
    pp._write_ids(s1_map, pp._join_ids(excl), "matched_entity_ids", out / "matching_results_excl.tsv")
    (OUTPUT_DIR / "variants").mkdir(exist_ok=True)
    shutil.copy(out / "matching_results.tsv", OUTPUT_DIR / "variants" / "matching_results_s2.tsv")
    shutil.copy(out / "matching_results_excl.tsv", OUTPUT_DIR / "variants" / "matching_results_s2_excl.tsv")
    if promote:
        shutil.copy(out / "matching_results.tsv", OUTPUT_DIR / "matching_results.tsv")
    run.set_metrics(test_decision=f"{d2['mode']}:{d2['param']}", test_s1=s1_map.height,
                    test_nonempty=sel["s1_idx"].n_unique(), test_pred_pairs=sel.height,
                    test_pred_pairs_excl=excl.height, test_excl_dropped=sel.height - excl.height)
    run.log(f"metric test nonempty={sel['s1_idx'].n_unique():,}/{s1_map.height:,} pred_pairs={sel.height:,} "
            f"excl_dropped={sel.height - excl.height:,}")
    run.end_stage()
