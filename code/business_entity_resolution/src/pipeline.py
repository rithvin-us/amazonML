"""End-to-end CLI (memory-lean: streamed prep, per-country blocking, chunked scoring).

  python pipeline.py prep                 # normalise all sources -> cache/norm_<split>_<kind>/part-*.parquet
  python pipeline.py all [--sample 0.02]  # prep + train/val + tune + test predict + validate
  python pipeline.py train [--sample f]   # train/val only (no test)
  python pipeline.py predict --run <id>   # test inference with a trained run's model
  python pipeline.py submit --run <id> --team NAME   # build final zip
  python pipeline.py lb <run_id> <score>  # record portal leaderboard score

Model: XGBoost on CUDA by default (--model lgb for LightGBM/CPU). Decision: per-S1 plug-in
expected-F0.5 or global threshold, whichever wins on validation.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import polars as pl
import polars.selectors as cs
import psutil

from blocking import BLOCK_VERSION, BlockIndex
from config import CACHE_DIR, NORM_VERSION, OUTPUT_DIR, ROOT, RUNS_DIR, VALIDATOR, Config
from features import (FEATURES, RR_FEATURES, TEXT_COLS, add_name_counts, build_features, country_context,
                      rerank_sims)
from hwmon import HwMonitor
from io_utils import load_ground_truth
from tracking import Run, record_lb_score

SRC_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SRC_DIR.parent
NORM_COLS = ["idx", "entity_id"] + TEXT_COLS
THRESHOLD_GRID = np.round(np.arange(0.05, 0.96, 0.025), 3)
ALPHA_GRID = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0]
TOP1_GRID = [0.2, 0.3, 0.4, 0.5, 0.6]  # "top-1 rescue" floors for S1s with nothing above the main threshold


def avail_gb() -> float:
    return psutil.virtual_memory().available / 1e9


# ---------------------------------------------------------------- prep (streamed, separate process)
def norm_dir(split: str, kind: str) -> Path:
    return CACHE_DIR / f"norm_{split}_{kind}_{NORM_VERSION}"


def prep(split: str, run: Run, cfg: Config) -> None:
    """Runs prep.py in a child process (light imports -> small multiprocessing workers)."""
    if all((norm_dir(split, k) / "_DONE").exists() for k in ("s1", "pool")):
        run.log(f"cache hit norm_{split}_*_{NORM_VERSION}")
        return
    workers = cfg.extra.get("prep_workers", 6)
    proc = subprocess.Popen([sys.executable, str(SRC_DIR / "prep.py"), split, "--workers", str(workers)],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            encoding="utf-8", errors="replace")
    for line in proc.stdout:
        line = line.rstrip()
        if line.startswith("PROGRESS "):
            _, pct, note = line.split(" ", 2)
            run.progress(float(pct), note)
        elif line:
            run.log(line)
    if proc.wait() != 0:
        raise RuntimeError(f"prep.py {split} failed (exit {proc.returncode})")


def scan_norm(split: str, kind: str) -> pl.LazyFrame:
    return pl.scan_parquet(norm_dir(split, kind) / "part-*.parquet")


def countries(split: str) -> list[str]:
    return scan_norm(split, "s1").select(pl.col("country_n").unique()).collect()["country_n"].sort().to_list()


# ---------------------------------------------------------------- candidates + features
def country_index(split: str, c: str, pool_c: pl.DataFrame, cfg: Config) -> tuple[BlockIndex, bool]:
    max_df = cfg.extra.get("max_df", 150)
    idx_dir = CACHE_DIR / "index" / f"{split}_{re.sub(r'[^a-z0-9]+', '_', c)}_{NORM_VERSION}_b{BLOCK_VERSION}_df{max_df}"
    return BlockIndex.cached(pool_c, idx_dir, max_df=max_df,
                             chunk=cfg.extra.get("index_chunk", 100_000))  # smaller key batches = lower peak RAM


def _rr_matrix(w: pl.DataFrame, names: list[str] | None = None) -> np.ndarray:
    return w.select(pl.col(names or RR_FEATURES).cast(pl.Float32).fill_null(-1.0)).to_numpy()


def rr_keep_expr(cfg: Config) -> pl.Expr:
    """Final candidate set = what the matching model scores (candidate_pairs.tsv). Adaptive: the top rr_min
    by re-ranker score always, then any candidate with rr >= rr_tau, capped at rr_keep. rr_tau=0 -> plain top-K."""
    keep, kmin, tau = cfg.extra.get("rr_keep", 40), cfg.extra.get("rr_min", 3), cfg.extra.get("rr_tau", 0.0)
    return (pl.col("rr_rank") <= keep) & ((pl.col("rr_rank") <= kmin) | (pl.col("rr") >= tau))


def wide_query(index: BlockIndex, s1: pl.DataFrame, pool_c: pl.DataFrame, cfg: Config) -> pl.DataFrame:
    """Wide retrieval (top-W by key score + WN/WA channel extras) with the re-ranker's fuzzy sims attached."""
    w_top, w_name, w_addr = cfg.extra.get("rr_wide", (300, 60, 60))
    w = index.query(s1, top_k=w_top, chunk=cfg.chunk_size, k_name=w_name, k_addr=w_addr)
    return rerank_sims(w, s1, pool_c, workers=cfg.n_jobs)


def fit_reranker(cfg: Config, run: Run, s1_ids: pl.Series, gt: pl.DataFrame):
    """Blocking re-ranker: small XGB on wide candidates of S1 that are in neither train nor val (so its score is
    an honest feature for the main model). One global model -> also used for unseen countries (France)."""
    import xgboost as xgb
    n_per = cfg.extra.get("rr_fit_s1", 5000)
    parts, n_true = [], 0
    for c in countries("train"):
        s1_c = (scan_norm("train", "s1").filter((pl.col("country_n") == c) & pl.col("entity_id").is_in(s1_ids.implode()))
                .select(NORM_COLS).collect())
        s1_c = s1_c.sample(n=min(n_per, s1_c.height), seed=cfg.seed)
        pool_c = scan_norm("train", "pool").filter(pl.col("country_n") == c).select(NORM_COLS).collect()
        index, _ = country_index("train", c, pool_c, cfg)
        w = wide_query(index, s1_c, pool_c, cfg)
        w = (w.join(s1_c.select(pl.col("idx").alias("s1_idx"), pl.col("entity_id").alias("s1_id")), on="s1_idx")
             .join(pool_c.select(pl.col("idx").alias("cand_idx"), pl.col("entity_id").alias("match_id")), on="cand_idx"))
        g = gt.join(s1_c.select(pl.col("entity_id").alias("s1_id")), on="s1_id").drop_nulls()
        n_true += g.height
        w = w.join(g.with_columns(pl.lit(1, pl.Int8).alias("label")), on=["s1_id", "match_id"], how="left")
        parts.append(w.select(*RR_FEATURES, "s1_idx", pl.col("label").fill_null(0)))
        run.log(f"reranker [{c}] fit S1 {s1_c.height:,} wide pairs {w.height:,} (RAM avail {avail_gb():.1f}GB)")
        del pool_c, index, w
        gc.collect()
    d = pl.concat(parts)
    dev = xgb_device(cfg.device)
    bst = xgb.train({"objective": "binary:logistic", "tree_method": "hist", "device": dev, "max_depth": 6, "eta": 0.1,
                     "seed": cfg.seed, "nthread": cfg.n_jobs},
                    xgb.DMatrix(_rr_matrix(d), d["label"].to_numpy(), feature_names=RR_FEATURES), 300)
    bst.save_model(str(run.dir / "reranker.json"))
    d = d.with_columns(pl.Series("rr", bst.predict(xgb.DMatrix(_rr_matrix(d), feature_names=RR_FEATURES))))
    d = d.with_columns(pl.col("rr").rank("ordinal", descending=True).over("s1_idx").alias("rr_rank"))
    kept = d.with_columns(pl.col("rr").cast(pl.Float32), pl.col("rr_rank").cast(pl.Float32)).filter(rr_keep_expr(cfg))
    wide_r = d["label"].sum() / max(n_true, 1)
    kept_r = kept["label"].sum() / max(n_true, 1)
    per_s1 = kept.height / max(d["s1_idx"].n_unique(), 1)
    run.log(f"metric reranker (in-sample) wide recall={wide_r:.4f} kept recall={kept_r:.4f} at {per_s1:.1f} cands/S1 "
            f"on {n_true:,} true pairs")
    run.set_metrics(rr_wide_recall=round(float(wide_r), 4), rr_keep_recall_insample=round(float(kept_r), 4),
                    rr_cands_per_s1_insample=round(per_s1, 2))
    return bst


def load_reranker(run_dir: Path, device: str):
    p = run_dir / "reranker.json"
    if not p.exists():
        return None
    import xgboost as xgb
    b = xgb.Booster()
    b.load_model(str(p))
    b.set_param({"device": xgb_device(device)})
    return b


def rerank_query(index: BlockIndex, part: pl.DataFrame, pool_c: pl.DataFrame, reranker, cfg: Config) -> pl.DataFrame:
    """Wide retrieval -> re-ranker score rr -> keep top rr_keep per S1 (in sub-chunks: ~350 cands per S1)."""
    import xgboost as xgb
    sub, out = cfg.extra.get("rr_sub", 2500), []
    for j in range(0, part.height, sub):
        w = wide_query(index, part.slice(j, sub), pool_c, cfg)
        if not w.height:
            continue
        names = reranker.feature_names or RR_FEATURES
        w = (w.with_columns(pl.Series("rr", reranker.predict(xgb.DMatrix(_rr_matrix(w, names), feature_names=names)),
                                      dtype=pl.Float32))
             .with_columns(pl.col("rr").rank("ordinal", descending=True).over("s1_idx").cast(pl.Float32).alias("rr_rank"))
             .filter(rr_keep_expr(cfg)).drop(cs.starts_with("r_")))
        out.append(w)
    if not out:
        return index.query(part.slice(0, 0)).with_columns(pl.lit(None, pl.Float32).alias("rr"),
                                                           pl.lit(None, pl.Float32).alias("rr_rank"))
    return pl.concat(out)


def iter_country_blocks(split: str, s1_all: pl.DataFrame, cfg: Config, run: Run, reranker=None):
    """Yield (country, s1_part, pool_c, pairs_features) per S1 chunk, one country pool in RAM at a time."""
    cs = s1_all["country_n"].unique().sort().to_list()
    done, n_all = 0, s1_all.height
    for c in cs:
        s1_c = s1_all.filter(pl.col("country_n") == c)
        pool_c = scan_norm(split, "pool").filter(pl.col("country_n") == c).select(NORM_COLS).collect()
        s1_pop = scan_norm(split, "s1").filter(pl.col("country_n") == c)
        s1_c, pool_c = add_name_counts([s1_c, pool_c], pool_c, s1_pop)
        ctx = country_context(s1_pop)  # address stop tokens + S1 name vocabulary (unlabelled S1 population)
        run.log(f"[{c}] S1 {s1_c.height:,}  pool {pool_c.height:,}: building index (RAM avail {avail_gb():.1f}GB)")
        index, hit = country_index(split, c, pool_c, cfg)
        run.log(f"[{c}] index {'loaded from cache' if hit else 'built + cached'}: keys kept {len(index.idf):,}/"
                f"{index.n_keys_total:,}, postings {index.n_postings:,} (RAM avail {avail_gb():.1f}GB)")
        for i in range(0, s1_c.height, cfg.s1_chunk):
            part = s1_c.slice(i, cfg.s1_chunk)
            if reranker is not None:
                pairs = rerank_query(index, part, pool_c, reranker, cfg)
            else:
                pairs = index.query(part, top_k=cfg.max_candidates, chunk=cfg.chunk_size,
                                    k_name=cfg.extra.get("k_name", 10), k_addr=cfg.extra.get("k_addr", 5)
                                    ).with_columns(pl.lit(None, pl.Float32).alias("rr"), pl.lit(None, pl.Float32).alias("rr_rank"))
            feats = build_features(pairs, part, pool_c, workers=cfg.n_jobs, ctx=ctx)
            done += part.height
            run.progress(done / n_all, f"[{c}] {done:,}/{n_all:,} S1 blocked+featurised")
            yield c, part, pool_c, feats
        del index, pool_c, s1_c
        gc.collect()


# ---------------------------------------------------------------- model (XGBoost GPU / LightGBM)
def xgb_device(want: str) -> str:
    """'cuda' only if a tiny CUDA fit actually works; else 'cpu'."""
    if want != "cuda":
        return "cpu"
    try:
        import xgboost as xgb
        xgb.train({"device": "cuda", "tree_method": "hist"},
                  xgb.DMatrix(np.zeros((8, 1), np.float32), label=np.zeros(8)), 1)
        return "cuda"
    except Exception:  # noqa: BLE001
        return "cpu"


class Model:
    """Thin wrapper so train/predict don't care which booster is inside."""

    def __init__(self, kind: str, booster, device: str = "cpu"):
        self.kind, self.booster, self.device = kind, booster, device

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.kind == "xgb":
            import xgboost as xgb
            return self.booster.predict(xgb.DMatrix(X, feature_names=FEATURES))
        return self.booster.predict(X, num_iteration=self.booster.best_iteration or None)

    def save(self, run_dir: Path) -> None:
        if self.kind == "xgb":
            self.booster.save_model(str(run_dir / "model.json"))
        else:
            self.booster.save_model(str(run_dir / "model.txt"))
        (run_dir / "model_meta.json").write_text(json.dumps({"kind": self.kind, "device": self.device}))

    @staticmethod
    def load(run_dir: Path, device: str = "cuda") -> "Model":
        meta = run_dir / "model_meta.json"
        kind = json.loads(meta.read_text())["kind"] if meta.exists() else "lgb"
        if kind == "xgb":
            import xgboost as xgb
            b = xgb.Booster()
            b.load_model(str(run_dir / "model.json"))
            dev = xgb_device(device)
            b.set_param({"device": dev})
            return Model("xgb", b, dev)
        import lightgbm as lgb
        return Model("lgb", lgb.Booster(model_file=str(run_dir / "model.txt")))


class Curve:
    """runs/<id>/train_curve.csv, read live by the dashboard."""

    def __init__(self, path: Path):
        self.path = path
        path.write_text("iter,train_logloss,val_logloss,val_aucpr\n")

    def add(self, it: int, tr_ll: float, va_ll: float, va_ap: float) -> None:
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(f"{it},{tr_ll:.5f},{va_ll:.5f},{va_ap:.5f}\n")


def _train_parts(files: list[Path]) -> pl.DataFrame:
    return pl.concat([pl.read_parquet(f).filter(~pl.col("is_val") & ~pl.col("is_es")) for f in files])


def train_model(cfg: Config, run: Run, train_files: list[Path], va: pl.DataFrame) -> Model:
    """train_files: per-chunk feature parquet on D: (val / early-stop rows skipped); va: in-RAM early-stop frame."""
    Xva, yva = va.select(FEATURES).to_numpy(), va["label"].to_numpy()
    curve = Curve(run.dir / "train_curve.csv")
    if cfg.model == "xgb":
        import xgboost as xgb
        dev = xgb_device(cfg.device)

        class PartIter(xgb.DataIter):
            """Streams train parts from disk into the quantile sketch: host RAM ~ one part at a time."""

            def __init__(self):
                self.i = 0
                super().__init__()

            def next(self, input_data) -> bool:
                if self.i == len(train_files):
                    return False
                df = pl.read_parquet(train_files[self.i]).filter(~pl.col("is_val") & ~pl.col("is_es"))
                input_data(data=df.select(FEATURES).to_numpy(), label=df["label"].to_numpy(),
                           feature_names=FEATURES)
                self.i += 1
                return True

            def reset(self) -> None:
                self.i = 0

        # logloss last -> early stopping tracks calibration (the decision stage relies on calibrated p)
        params = dict(objective="binary:logistic", eval_metric=["aucpr", "logloss"], tree_method="hist",
                      device=dev, eta=cfg.xgb_lr, max_depth=cfg.xgb_depth, min_child_weight=5, subsample=0.8,
                      colsample_bytree=0.8, reg_lambda=1.0, max_bin=256, seed=cfg.seed, nthread=cfg.n_jobs)
        dtr = xgb.QuantileDMatrix(PartIter(), max_bin=256)
        dva = xgb.QuantileDMatrix(Xva, yva, ref=dtr, feature_names=FEATURES)
        run.log(f"xgboost {xgb.__version__} device={dev} train {dtr.num_row():,} rows (streamed from "
                f"{len(train_files)} parts) val {len(yva):,} pos_rate_val {yva.mean():.4f}")

        class Live(xgb.callback.TrainingCallback):
            def after_iteration(self, model, epoch, evals_log):
                if epoch % 10 == 0:
                    t_, v_ = evals_log["train"], evals_log["val"]
                    curve.add(epoch, t_["logloss"][-1], v_["logloss"][-1], v_["aucpr"][-1])
                    run.progress(min(epoch / cfg.xgb_rounds, 1.0), f"iter {epoch} val_logloss={v_['logloss'][-1]:.4f}"
                                 f" val_aucpr={v_['aucpr'][-1]:.4f} [{dev}]")
                return False

        bst = xgb.train(params, dtr, cfg.xgb_rounds, evals=[(dtr, "train"), (dva, "val")],
                        early_stopping_rounds=cfg.xgb_early_stop, callbacks=[Live()], verbose_eval=False)
        best = bst.best_iteration
        bst = bst[: best + 1]
        gain = bst.get_score(importance_type="total_gain")
        imp = {f: round(float(gain.get(f, 0.0)), 1) for f in FEATURES}
        model = Model("xgb", bst, dev)
    else:
        import lightgbm as lgb
        tr = _train_parts(train_files)
        Xtr, ytr = tr.select(FEATURES).to_numpy(), tr["label"].to_numpy()
        del tr
        params = dict(objective="binary", metric=["binary_logloss", "average_precision"], learning_rate=cfg.xgb_lr,
                      num_leaves=cfg.extra.get("lgb_leaves", 63), min_data_in_leaf=50, feature_fraction=0.8, bagging_fraction=0.8,
                      bagging_freq=1, lambda_l2=1.0, num_threads=cfg.n_jobs, verbose=-1, seed=cfg.seed)
        dtr = lgb.Dataset(Xtr, ytr, feature_name=FEATURES)
        dva = lgb.Dataset(Xva, yva, reference=dtr)

        def _cb(env):
            if env.iteration % 10 == 0:
                r = {(e[0], e[1]): e[2] for e in env.evaluation_result_list}
                curve.add(env.iteration, r[("train", "binary_logloss")], r[("val", "binary_logloss")],
                          r[("val", "average_precision")])
                run.progress(min(env.iteration / cfg.xgb_rounds, 1.0),
                             f"iter {env.iteration} val_logloss={r[('val', 'binary_logloss')]:.4f} [cpu]")

        bst = lgb.train(params, dtr, cfg.xgb_rounds, valid_sets=[dtr, dva], valid_names=["train", "val"],
                        callbacks=[lgb.early_stopping(cfg.xgb_early_stop, first_metric_only=True, verbose=False), _cb])
        best = bst.best_iteration
        imp = dict(zip(FEATURES, bst.feature_importance("gain").round(1).tolist()))
        model = Model("lgb", bst, "cpu")
    model.save(run.dir)
    imp = dict(sorted(imp.items(), key=lambda x: -x[1]))
    (run.dir / "feature_importance.json").write_text(json.dumps(imp, indent=2))
    run.set_metrics(model=model.kind, device=model.device, best_iter=best)
    run.log(f"best iter {best} ({model.kind} on {model.device})")
    return model


# ---------------------------------------------------------------- decision / tuning
def eval_selection(sel: pl.DataFrame, truth_counts: pl.DataFrame) -> dict:
    """sel: chosen (s1_idx, label) rows. truth_counts: s1_idx, n_true for every eval S1 (incl. 0).

    Per-S1 F0.5 = 1.25*tp / (0.25*n_true + n_pred); 1.0 for a correctly empty singleton.
    """
    agg = sel.group_by("s1_idx").agg(pl.len().alias("n_pred"), pl.col("label").sum().alias("tp"))
    df = truth_counts.join(agg, on="s1_idx", how="left").fill_null(0).with_columns(
        pl.when((pl.col("n_true") == 0) & (pl.col("n_pred") == 0)).then(1.0)
        .otherwise(1.25 * pl.col("tp") / (0.25 * pl.col("n_true") + pl.col("n_pred")).clip(1e-9)).alias("f"))
    return {"f05": float(df["f"].mean()),
            "precision": float(df["tp"].sum() / max(df["n_pred"].sum(), 1)),
            "recall": float(df["tp"].sum() / max(df["n_true"].sum(), 1))}


def select_expected_f(scored: pl.DataFrame, alpha: float) -> pl.DataFrame:
    """Per S1 keep the top-m candidates (by p) maximising plug-in E[F0.5] = 1.25*E[tp] / (0.25*E[k] + m).

    E[k] = sum(p) + alpha, alpha = expected true matches missed by blocking. The empty list scores
    P(no match) ~ prod(1-p) * exp(-alpha); it wins for likely singletons.
    """
    s = scored.sort(["s1_idx", "p"], descending=[False, True]).with_columns(
        pl.col("p").cum_sum().over("s1_idx").alias("_ctp"),
        pl.int_range(1, pl.len() + 1).over("s1_idx").alias("_m"),
        pl.col("p").sum().over("s1_idx").alias("_ek"),
        (1 - pl.col("p")).clip(1e-6, 1.0).log().sum().over("s1_idx").alias("_lp0"))
    s = s.with_columns((1.25 * pl.col("_ctp") / (0.25 * (pl.col("_ek") + alpha) + pl.col("_m"))).alias("_ef"))
    best = s.group_by("s1_idx").agg(pl.col("_ef").max().alias("_efb"),
                                    pl.col("_m").sort_by("_ef", descending=True).first().alias("_mb"),
                                    pl.col("_lp0").first())
    best = best.select("s1_idx", pl.when(pl.col("_efb") > (pl.col("_lp0") - alpha).exp())
                       .then(pl.col("_mb")).otherwise(0).alias("_msel"))
    return (s.join(best, on="s1_idx").filter(pl.col("_m") <= pl.col("_msel"))
            .drop("_ctp", "_m", "_ek", "_lp0", "_ef", "_msel"))


def exclusive(sel: pl.DataFrame) -> pl.DataFrame:
    """GT is one-to-many: a pool record matches at most one S1. Keep each cand only for its best-p S1."""
    return sel.filter(pl.col("p").rank("ordinal", descending=True).over("cand_idx") == 1)


def apply_decision(scored: pl.DataFrame, dec: dict, floor: float) -> pl.DataFrame:
    """dec: {mode, param, excl}. excl -> each pool record is first assigned to its best-p S1 only (GT is
    exclusive), then the per-S1 rule runs. Modes: threshold (p >= t); thr_top1 (p >= t, plus the S1's best
    candidate when it has p >= t1 < t); expected_f (per-S1 plug-in expected-F0.5 optimum)."""
    if dec.get("excl"):
        scored = exclusive(scored.filter(pl.col("p") >= floor))
    if dec["mode"] == "threshold":
        return scored.filter(pl.col("p") >= dec["param"])
    if dec["mode"] == "thr_top1":
        t, t1 = dec["param"]
        top1 = pl.col("p").rank("ordinal", descending=True).over("s1_idx") == 1
        return scored.filter((pl.col("p") >= t) | (top1 & (pl.col("p") >= t1)))
    return select_expected_f(scored.filter(pl.col("p") >= floor), dec["param"])


def tune_decision(scored: pl.DataFrame, truth_counts: pl.DataFrame, floor: float, run: Run) -> dict:
    """Best param per (mode, excl) on labelled pairs -> {name: {mode, param, excl, f05, precision, recall}}."""
    res = {}
    cands = ([("threshold", float(t)) for t in THRESHOLD_GRID] + [("expected_f", a) for a in ALPHA_GRID]
             + [("thr_top1", [float(t), t1]) for t in THRESHOLD_GRID if t >= 0.5 for t1 in TOP1_GRID if t1 < t])
    for excl in (False, True):
        base = exclusive(scored.filter(pl.col("p") >= floor)) if excl else scored
        for mode, param in cands:
            m = eval_selection(apply_decision(base, {"mode": mode, "param": param}, floor), truth_counts)
            name = mode + ("+excl" if excl else "")
            if m["f05"] > res.get(name, {"f05": -1})["f05"]:
                res[name] = {"mode": mode, "param": param, "excl": excl, **m}
    for k, v in res.items():
        run.log(f"decision {k}: param={v['param']} f05={v['f05']:.5f} P={v['precision']:.4f} R={v['recall']:.4f}")
    return res


# ---------------------------------------------------------------- train
def score_competitors(cfg: Config, run: Run, model: Model, reranker, va: pl.DataFrame, val_ids: pl.Series) -> pl.DataFrame:
    """Non-val S1 that own (by GT) a pool record some val S1 puts p >= 0.05 on. On test every S1 competes for
    every pool record; without these owners, exclusivity on val is weaker than on test and the decision is
    tuned on the wrong precision. They are scored like test S1 and never enter the metric."""
    hot = va.filter(pl.col("p") >= 0.05)["cand_idx"].unique()
    mids = (scan_norm("train", "pool").filter(pl.col("idx").is_in(hot.implode()))
            .select(pl.col("entity_id").alias("match_id")).collect())
    owners = (load_ground_truth().drop_nulls().join(mids, on="match_id", how="semi")
              .filter(~pl.col("s1_id").is_in(val_ids.implode()))["s1_id"].unique())
    s1o = scan_norm("train", "s1").filter(pl.col("entity_id").is_in(owners.implode())).select(NORM_COLS).collect()
    run.log(f"val competitors: {s1o.height:,} non-val S1 own {mids.height:,} hot val candidates")
    out = [pl.DataFrame(schema={"s1_idx": pl.Int64, "cand_idx": pl.Int64, "p": pl.Float32})]
    if s1o.height:
        for c, part, pool_c, feats in iter_country_blocks("train", s1o, cfg, run, reranker):
            out.append(feats.select(pl.col("s1_idx").cast(pl.Int64), pl.col("cand_idx").cast(pl.Int64)).with_columns(
                pl.Series("p", model.predict(feats.select(FEATURES).to_numpy()), dtype=pl.Float32)))
    return pl.concat(out).filter(pl.col("p") >= cfg.p_floor)


def s1_blocks(split: str, max_size: int = 2000) -> pl.DataFrame:
    """entity_id -> block = country | city (2nd-to-last comma part of raw S1 address, digits dropped) | name_core[:1].

    Blocks over max_size are split by name_core[:3]. Validation samples whole blocks so sibling S1s (chains,
    duplicates) competing for the same pool records are co-present, as on test; stage 2 computes its
    competition features within blocks on both val and test.
    """
    raw = pl.scan_parquet(CACHE_DIR / f"raw_{split}_s1.parquet").select(
        "entity_id", pl.col("business_address").fill_null("").str.split(",")
        .list.eval(pl.element().str.to_lowercase().str.replace_all(r"\d+", "").str.strip_chars())
        .list.get(-2, null_on_oob=True).fill_null("").alias("city"))
    key = lambda n: pl.concat_str([pl.col("country_n"), pl.col("city"), pl.col("name_core").str.slice(0, n)],  # noqa: E731
                                  separator="|")
    return (scan_norm(split, "s1").select("entity_id", "country_n", "name_core")
            .join(raw, on="entity_id", how="left").with_columns(pl.col("city").fill_null(""))
            .with_columns(key(1).alias("b1"))
            .select("entity_id", "country_n",
                    pl.when(pl.len().over("b1") > max_size).then(key(3)).otherwise(pl.col("b1")).alias("block"))
            .collect())


def train_stage(cfg: Config, run: Run) -> None:
    run.start_stage("prep_train")
    prep("train", run, cfg)
    run.end_stage()

    gt = load_ground_truth()
    blk = s1_blocks("train")
    n_use = min(int(blk.height * cfg.sample), cfg.extra.get("train_max_s1", 150_000))
    n_val = min(int(n_use * cfg.val_frac), cfg.extra.get("val_max_s1", 40_000))
    # val = whole blocks (random order, huge metro blocks skipped) until n_val; train = random from the rest
    bl = (blk.group_by("block").len().filter(pl.col("len") <= cfg.extra.get("val_block_max", 2000))
          .sort("block").sample(fraction=1.0, shuffle=True, seed=cfg.seed)
          .filter(pl.col("len").cum_sum() <= n_val))
    val_ids = blk.join(bl.select("block"), on="block", how="semi")["entity_id"]
    rest = blk.join(bl.select("block"), on="block", how="anti")["entity_id"]
    use = pl.concat([val_ids, rest.sample(n=min(n_use - val_ids.len(), rest.len()), seed=cfg.seed, shuffle=True)])
    reranker = None
    if cfg.extra.get("use_rr", True):
        run.start_stage("fit_reranker")
        reranker = fit_reranker(cfg, run, rest.filter(~rest.is_in(use)), gt)
        run.end_stage()
    blk = blk.join(bl.select("block"), on="block", how="semi")  # keep only val rows (for val_truth.parquet)
    del rest
    # is_es: ~5% of the non-val train S1 held out for early stopping (val stays untouched by model selection)
    s1 = (scan_norm("train", "s1").filter(pl.col("entity_id").is_in(use.implode())).select(NORM_COLS)
          .collect().with_columns(pl.col("entity_id").is_in(val_ids.implode()).alias("is_val"))
          .with_columns((~pl.col("is_val") & (pl.col("entity_id").hash(cfg.seed) % 20 == 0)).alias("is_es")))
    mix = lambda f: dict(s1.filter(f).group_by("country_n").len().sort("country_n").iter_rows())  # noqa: E731
    run.log(f"S1 used {s1.height:,} (val {val_ids.len():,} in {bl.height:,} blocks, es {int(s1['is_es'].sum()):,}) "
            f"country mix val={mix(pl.col('is_val'))} train={mix(~pl.col('is_val'))}")

    idmap_s1 = s1.select(pl.col("idx").alias("s1_idx"), pl.col("entity_id").alias("s1_id"), "is_val", "is_es")
    gt_use = gt.join(idmap_s1, on="s1_id")
    truth_counts = (idmap_s1.join(gt_use.group_by("s1_id").agg(pl.col("match_id").drop_nulls().len().alias("n_true")),
                                  on="s1_id", how="left").fill_null(0))
    del gt
    gc.collect()

    # labelled feature chunks spill to D: (runs/<id>/train_feats) -> GPU training streams them back
    run.start_stage("blocking+features_train")
    fdir = run.dir / "train_feats"
    shutil.rmtree(fdir, ignore_errors=True)
    fdir.mkdir()
    files, val_parts, es_parts = [], [], []
    n_pairs = n_pos = 0
    pos_all = gt_use.drop_nulls()
    for c, part, pool_c, feats in iter_country_blocks("train", s1, cfg, run, reranker):
        ids = pool_c.select(pl.col("idx").alias("cand_idx"), pl.col("entity_id").alias("match_id"))
        pos = (pos_all.join(ids, on="match_id")
               .select("s1_idx", "cand_idx").with_columns(pl.lit(1, pl.Int8).alias("label")))
        f = (feats.join(pos, on=["s1_idx", "cand_idx"], how="left").with_columns(pl.col("label").fill_null(0))
             .join(idmap_s1.select("s1_idx", "is_val", "is_es"), on="s1_idx"))
        n_pairs += f.height
        n_pos += int(f["label"].sum())
        val_parts.append(f.filter(pl.col("is_val")))
        es_parts.append(f.filter(pl.col("is_es")))
        files.append(fdir / f"part-{len(files):05d}.parquet")
        f.write_parquet(files[-1])
        del f, feats
    va, es = pl.concat(val_parts), pl.concat(es_parts)
    del val_parts, es_parts
    gc.collect()
    run.end_stage()

    tv = truth_counts.filter(pl.col("is_val"))
    br = float(va["label"].sum() / max(tv["n_true"].sum(), 1))
    avg_c = va.height / max(tv.height, 1)
    zero_c = tv.height - va["s1_idx"].n_unique()
    run.log(f"metric val block_recall={br:.4f} avg_cands={avg_c:.1f} s1_without_cands={zero_c:,} "
            f"pos_rate={n_pos / max(n_pairs, 1):.4f} train_pairs={n_pairs - va.height:,}")
    run.set_metrics(block_recall=round(br, 4), avg_candidates=round(avg_c, 2), val_s1_without_cands=zero_c,
                    n_pairs=n_pairs, n_s1=s1.height, n_val_s1=tv.height)

    run.start_stage("train_model")
    model = train_model(cfg, run, files, es)
    del es
    gc.collect()
    run.end_stage()

    vtruth = tv.select("s1_idx", "s1_id", "n_true").join(blk.rename({"entity_id": "s1_id"}), on="s1_id", how="left")
    score_and_tune(cfg, run, model, reranker, va, vtruth, val_ids)


def score_and_tune(cfg: Config, run: Run, model: Model, reranker, va: pl.DataFrame, vtruth: pl.DataFrame,
                   val_ids: pl.Series) -> None:
    """Score val features, score competitor S1, tune the decision. vtruth: every val S1 (s1_idx, s1_id, n_true,
    country_n, block), incl. those without candidates."""
    va = va.select("s1_idx", "cand_idx", "label").with_columns(
        pl.Series("p", model.predict(va.select(FEATURES).to_numpy()), dtype=pl.Float32))
    va.write_parquet(run.dir / "val_scored.parquet")  # for offline error analysis / re-decisions
    scored = va
    if cfg.extra.get("use_comp", True):
        run.start_stage("val_competitors")
        comp = score_competitors(cfg, run, model, reranker, va, val_ids)
        comp.write_parquet(run.dir / "comp_scored.parquet")
        scored = pl.concat([va, comp.select("s1_idx", "cand_idx", pl.lit(0, pl.Int8).alias("label"), "p")])
        run.end_stage()

    run.start_stage("tune")
    tvc = vtruth.select("s1_idx", "n_true")
    vtruth.write_parquet(run.dir / "val_truth.parquet")  # stage 2 / ce-apply need every val S1 incl. no-cand ones
    if scored is not va:  # decision tuned WITH competitors (as on test); val-only numbers logged for reference
        solo = max(tune_decision(va, tvc, cfg.p_floor, run).values(), key=lambda d: d["f05"])
        run.set_metrics(val_f05_no_comp=round(solo["f05"], 5))
        run.log(f"metric val_f05 without competitors={solo['f05']:.5f} ({solo['mode']}:{solo['param']} excl={solo['excl']})")
    res = tune_decision(scored, tvc, cfg.p_floor, run)
    dec = max(res.values(), key=lambda d: d["f05"])
    run.set_metrics(decision_mode=dec["mode"], decision_param=dec["param"], decision_excl=dec["excl"],
                    threshold=res["threshold"]["param"], val_f05=round(dec["f05"], 5),
                    val_precision=round(dec["precision"], 4), val_recall=round(dec["recall"], 4),
                    val_f05_by_decision={k: round(v["f05"], 5) for k, v in res.items()},
                    val_f05_threshold=round(res["threshold"]["f05"], 5),
                    val_f05_excl=round(res["threshold+excl"]["f05"], 5),
                    val_singleton_frac=round(vtruth.filter(pl.col("n_true") == 0).height / max(vtruth.height, 1), 4))
    # per-country val breakdown (tells us how France-like generalisation may behave)
    per_c = {}
    sel_all = apply_decision(scored, dec, cfg.p_floor)
    for c in vtruth["country_n"].drop_nulls().unique().to_list():
        ids_c = vtruth.filter(pl.col("country_n") == c)["s1_idx"].implode()
        per_c[c] = round(eval_selection(sel_all.filter(pl.col("s1_idx").is_in(ids_c)),
                                        tvc.filter(pl.col("s1_idx").is_in(ids_c)))["f05"], 5)
    run.set_metrics(val_f05_by_country=per_c)
    run.log(f"metric val_f05={dec['f05']:.5f} P={dec['precision']:.4f} R={dec['recall']:.4f} "
            f"decision={dec['mode']}:{dec['param']} excl={dec['excl']} by_country={per_c}")
    run.end_stage()


def retrain_stage(cfg: Config, run: Run, src: Path) -> None:
    """New matcher on another run's saved train_feats (same candidates/features, e.g. other depth/lr/seed), then
    the usual val scoring + competitor-aware tuning. Test: `rescore --run <this> --feats-run <run with test_feats>`."""
    files = sorted((src / "train_feats").glob("part-*.parquet"))
    for name in ("reranker.json",):  # predict/rescore of this run must block exactly like the source run
        if (src / name).exists():
            shutil.copy(src / name, run.dir / name)
    reranker = load_reranker(src, cfg.device)
    es = pl.concat([pl.read_parquet(f).filter(pl.col("is_es")) for f in files])
    va = pl.concat([pl.read_parquet(f).filter(pl.col("is_val")) for f in files])
    vtruth = pl.read_parquet(src / "val_truth.parquet")
    run.log(f"retrain from {src.name}: {len(files)} parts, val pairs {va.height:,}, early-stop pairs {es.height:,}")
    run.set_metrics(retrain_from=src.name, **{k: v for k, v in json.loads((src / "metrics.json").read_text()).items()
                                              if k in ("block_recall", "avg_candidates", "n_s1", "n_val_s1")})
    run.start_stage("train_model")
    model = train_model(cfg, run, files, es)
    del es
    gc.collect()
    run.end_stage()
    score_and_tune(cfg, run, model, reranker, va, vtruth, vtruth["s1_id"])


# ---------------------------------------------------------------- predict
def _write_ids(s1_map: pl.DataFrame, lists: pl.DataFrame, col: str, path: Path) -> None:
    """One row per S1 in source order; lists: s1_idx, ids (comma-joined)."""
    (s1_map.join(lists, on="s1_idx", how="left", maintain_order="left")
     .select("source1_entity_id", pl.col("ids").fill_null("").alias(col))
     .write_csv(path, separator="\t", quote_style="never"))


def _join_ids(sel: pl.DataFrame) -> pl.DataFrame:
    return (sel.sort(["s1_idx", "p"], descending=[False, True])
            .group_by("s1_idx", maintain_order=True).agg(pl.col("cid").str.join(",").alias("ids")))


def _check_id_file(path: Path, col: str, n_s1: int, batch: int = 100_000) -> list[str]:
    """Streamed per-row format check (the official validator, and any global unique over ~70M ids, OOM here)."""
    issues, rows, s1_seen = set(), 0, []
    lf = pl.scan_csv(path, separator="\t", quote_char=None, infer_schema=False)
    if lf.collect_schema().names() != ["source1_entity_id", col]:
        return [f"header {lf.collect_schema().names()}"]
    for b in lf.collect_batches(chunk_size=batch):
        rows += b.height
        s1_seen.append(b["source1_entity_id"])
        v = b[col].fill_null("")
        if (~v.str.contains(r"^(S[23]-[^,]+(,S[23]-[^,]+)*)?$")).any():
            issues.add("malformed list or non S2/S3 id")
        lst = v.str.split(",")
        if (lst.list.n_unique() != lst.list.len()).any():
            issues.add("duplicate id within a list")
    n_unique = pl.concat(s1_seen).n_unique()
    if rows != n_s1 or n_unique != n_s1:
        issues.add(f"rows {rows} unique {n_unique} expected {n_s1}")
    return sorted(issues)


def predict_stage(cfg: Config, run: Run, model_run_dir: Path) -> None:
    metrics = json.loads((model_run_dir / "metrics.json").read_text())
    dec = {"mode": metrics.get("decision_mode", "threshold"),
           "param": metrics.get("decision_param", metrics.get("threshold", 0.5)),
           "excl": metrics.get("decision_excl", False)}
    if cfg.extra.get("threshold_override"):
        dec = {"mode": "threshold", "param": cfg.extra["threshold_override"], "excl": dec.get("excl", False)}
    model = Model.load(model_run_dir, cfg.device)
    reranker = load_reranker(model_run_dir, cfg.device)
    run.log(f"model {model.kind} on {model.device}, decision {dec['mode']}:{dec['param']} excl={dec.get('excl')} "
            f"reranker={'yes' if reranker is not None else 'no'}")

    run.start_stage("prep_test")
    prep("test", run, cfg)
    run.end_stage()
    s1 = scan_norm("test", "s1").select(NORM_COLS).collect()

    # per-chunk results spill to D: (run dir) instead of accumulating in RAM; kept for re-decisions
    spill = run.dir / "pred"
    shutil.rmtree(spill, ignore_errors=True)
    spill.mkdir()
    run.start_stage("blocking+features+predict_test")
    n_pairs = part_no = 0
    fdir = run.dir / "test_feats" if cfg.extra.get("save_test_feats") else None
    if fdir:
        fdir.mkdir(exist_ok=True)
    for c, part, pool_c, feats in iter_country_blocks("test", s1, cfg, run, reranker):
        feats = feats.join(pool_c.select(pl.col("idx").alias("cand_idx"), pl.col("entity_id").alias("cid")), on="cand_idx")
        if fdir:  # lets `rescore` apply any later model to the same test candidates in minutes
            feats.select("s1_idx", "cand_idx", "cid", *FEATURES).write_parquet(fdir / f"part-{part_no:05d}.parquet")
        n_pairs += _spill_scored(feats, model, spill, part_no, cfg.p_floor)
        part_no += 1
        del feats
    run.end_stage()
    del s1
    gc.collect()
    decide_stage(cfg, run, spill, dec, n_pairs)


def _spill_scored(feats: pl.DataFrame, model: Model, spill: Path, part_no: int, p_floor: float) -> int:
    """Score one chunk; write its candidate lists (cand-*) and p >= floor pairs (scored-*) to the spill dir."""
    f = feats.select("s1_idx", "cand_idx", "cid", "brank").with_columns(
        pl.Series("p", model.predict(feats.select(FEATURES).to_numpy()), dtype=pl.Float32))
    (f.sort(["s1_idx", "brank"]).group_by("s1_idx", maintain_order=True)
     .agg(pl.col("cid").str.join(",").alias("ids")).write_parquet(spill / f"cand-{part_no:05d}.parquet"))
    f.filter(pl.col("p") >= p_floor).select("s1_idx", "cand_idx", "cid", "p").write_parquet(
        spill / f"scored-{part_no:05d}.parquet")
    return f.height


def rescore_stage(cfg: Config, run: Run, model_run_dir: Path, feats_run_dir: Path) -> None:
    """Score saved test features (another run's test_feats/, same FEATURES) with model_run_dir's model + decision."""
    metrics = json.loads((model_run_dir / "metrics.json").read_text())
    dec = {"mode": metrics.get("decision_mode", "threshold"),
           "param": metrics.get("decision_param", metrics.get("threshold", 0.5)), "excl": metrics.get("decision_excl", False)}
    model = Model.load(model_run_dir, cfg.device)
    run.log(f"rescoring {feats_run_dir.name}/test_feats with {model_run_dir.name} ({model.kind}), decision {dec}")
    spill = run.dir / "pred"
    shutil.rmtree(spill, ignore_errors=True)
    spill.mkdir()
    run.start_stage("rescore_test")
    parts = sorted((feats_run_dir / "test_feats").glob("part-*.parquet"))
    n_pairs = 0
    for i, fp in enumerate(parts):
        n_pairs += _spill_scored(pl.read_parquet(fp), model, spill, i, cfg.p_floor)  # brank is a saved FEATURE
        run.progress((i + 1) / len(parts), f"{i + 1}/{len(parts)} parts")
    run.end_stage()
    decide_stage(cfg, run, spill, dec, n_pairs)


def decide_stage(cfg: Config, run: Run, spill: Path, dec: dict, n_pairs: int | None = None) -> None:
    """Decision + TSV writing + validation from a run's pred/ spill (scored-*.parquet, cand-*.parquet).

    Also used standalone (`pipeline.py decide --run <id>`) to re-decide without re-blocking. s1_idx is the
    test S1 row number in raw source order (the norm caches assign idx the same way).
    """
    run.start_stage("decide+write_test")
    s1_map = (pl.read_parquet(CACHE_DIR / "raw_test_s1.parquet", columns=["entity_id"]).with_row_index("s1_idx")
              .select(pl.col("s1_idx").cast(pl.Int64), pl.col("entity_id").alias("source1_entity_id")))
    n_s1 = s1_map.height
    out_dir = run.dir / "output"
    out_dir.mkdir(exist_ok=True)
    _write_ids(s1_map, pl.read_parquet(spill / "cand-*.parquet"), "candidate_entity_ids",
               out_dir / "candidate_pairs.tsv")
    gc.collect()
    scored = pl.read_parquet(spill / "scored-*.parquet")
    # main = the val-best decision (incl. its exclusivity choice); alt = same rule with exclusivity flipped
    sel = apply_decision(scored, dec, cfg.p_floor)
    alt = apply_decision(scored, {**dec, "excl": not dec.get("excl")}, cfg.p_floor)
    del scored
    _write_ids(s1_map, _join_ids(sel), "matched_entity_ids", out_dir / "matching_results.tsv")
    _write_ids(s1_map, _join_ids(alt), "matched_entity_ids", out_dir / "matching_results_alt.tsv")
    for fn in ("matching_results.tsv", "candidate_pairs.tsv"):
        shutil.copy(out_dir / fn, OUTPUT_DIR / fn)
    (OUTPUT_DIR / "variants").mkdir(exist_ok=True)
    shutil.copy(out_dir / "matching_results_alt.tsv", OUTPUT_DIR / "variants" / "matching_results_alt.tsv")
    nonempty = sel["s1_idx"].n_unique()
    run.set_metrics(test_decision=f"{dec['mode']}:{dec['param']} excl={dec.get('excl')}", test_s1=n_s1,
                    test_nonempty=nonempty, test_pred_pairs=sel.height, test_pred_pairs_alt=alt.height)
    if n_pairs is not None:
        run.set_metrics(test_avg_candidates=round(n_pairs / max(n_s1, 1), 2))
    run.log(f"metric test nonempty={nonempty:,}/{n_s1:,} pred_pairs={sel.height:,} alt_pairs={alt.height:,}")
    del sel, alt
    gc.collect()
    run.end_stage()

    run.start_stage("validate")
    test_dir = ROOT / "student_resource" / "dataset" / "test"
    ok = True
    report = []
    for fn in ("matching_results.tsv", "variants/matching_results_alt.tsv"):
        # --candidate at a missing path: the official validator holds ~70M candidate ids in Python sets (OOM
        # here); candidate_pairs.tsv gets the streamed polars check below instead
        r = subprocess.run([sys.executable, str(VALIDATOR), "--matching", str(OUTPUT_DIR / fn),
                            "--candidate", str(OUTPUT_DIR / "__skip__.tsv"), "--test-dir", str(test_dir)],
                           capture_output=True, text=True, encoding="utf-8", errors="replace",
                           env={**os.environ, "PYTHONIOENCODING": "utf-8"})  # validator prints non-cp1252 chars
        report.append(f"== {fn}\n{r.stdout}{r.stderr}")
        tail = (r.stdout.strip().splitlines() or [r.stderr[-300:]])[-1]
        run.log(f"metric validator {fn} exit={r.returncode}: {tail}")
        ok &= r.returncode == 0
    cand_issues = _check_id_file(OUTPUT_DIR / "candidate_pairs.tsv", "candidate_entity_ids", n_s1)
    report.append(f"== candidate_pairs.tsv (polars check)\n{cand_issues or 'OK'}")
    run.log(f"metric candidate_pairs check: {cand_issues or 'OK'}")
    (run.dir / "validate.txt").write_text("\n".join(report), encoding="utf-8")
    run.set_metrics(validator_pass=ok and not cand_issues)
    run.end_stage()


# ---------------------------------------------------------------- submit zip
def submit_stage(run_dir: Path, team: str) -> Path:
    zpath = ROOT / f"{team}_submission.zip"
    doc = ROOT / "docs" / "Documentation_template.md"  # the filled-in methodology document
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for fn in ("matching_results.tsv", "candidate_pairs.tsv"):
            z.write(run_dir / "output" / fn, f"output/{fn}")
        for p in PROJECT_DIR.rglob("*"):
            if p.is_file() and "__pycache__" not in p.parts:
                z.write(p, f"code/business_entity_resolution/{p.relative_to(PROJECT_DIR).as_posix()}")
        z.write(doc, "Documentation_template.md")
    print(f"wrote {zpath}")
    return zpath


# ---------------------------------------------------------------- main
def main() -> None:
    for stream in (sys.stdout, sys.stderr):  # Windows console is cp1252: never crash a run on a log line
        stream.reconfigure(errors="backslashreplace")
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prep", "train", "predict", "all", "submit", "lb", "stage2", "decide", "rescore", "ce-train", "ce-apply", "retrain"])
    ap.add_argument("args", nargs="*")
    ap.add_argument("--sample", type=float, default=1.0)
    ap.add_argument("--name", default=None)
    ap.add_argument("--run", default=None, help="run id holding the model (predict/submit)")
    ap.add_argument("--team", default="team")
    ap.add_argument("--max-cands", type=int, default=40)
    ap.add_argument("--train-max-s1", type=int, default=150_000)
    ap.add_argument("--val-max-s1", type=int, default=40_000)
    ap.add_argument("--max-df", type=int, default=150)
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--prep-workers", type=int, default=6)
    ap.add_argument("--model", choices=["xgb", "lgb"], default="xgb")
    ap.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    ap.add_argument("--s1-chunk", type=int, default=20_000)
    ap.add_argument("--k-name", type=int, default=10, help="extra candidates kept by name-key score rank")
    ap.add_argument("--k-addr", type=int, default=5, help="extra candidates kept by address-key score rank")
    ap.add_argument("--rounds", type=int, default=None, help="override xgb_rounds")
    ap.add_argument("--promote", action="store_true", help="stage2: also write output/matching_results.tsv")
    ap.add_argument("--no-rr", action="store_true", help="disable the blocking re-ranker (v4 behaviour)")
    ap.add_argument("--rr-keep", type=int, default=40, help="candidates kept per S1 after re-ranking")
    ap.add_argument("--rr-wide", default="300,60,60", help="wide retrieval: top_k,k_name,k_addr before re-ranking")
    ap.add_argument("--rr-fit-s1", type=int, default=5000, help="held-out S1 per country for fitting the re-ranker")
    ap.add_argument("--rr-tau", type=float, default=0.0, help="keep candidates with re-ranker score >= tau (0 = top-K)")
    ap.add_argument("--rr-min", type=int, default=3, help="always keep this many top re-ranked candidates")
    ap.add_argument("--ce-pairs", type=int, default=600_000, help="ce-train: training pairs")
    ap.add_argument("--ce-epochs", type=int, default=2, help="ce-train: epochs")
    ap.add_argument("--ce-dir", default=None, help="ce-apply: fine-tuned model dir (default models/ce_<run>)")
    ap.add_argument("--save-test-feats", action="store_true", help="predict: keep test features for `rescore`")
    ap.add_argument("--feats-run", default=None, help="rescore: run id holding test_feats/")
    ap.add_argument("--no-comp", action="store_true", help="tune the decision on val S1 only (no competitor S1)")
    ap.add_argument("--depth", type=int, default=None, help="override xgb max_depth")
    ap.add_argument("--lr", type=float, default=None, help="override xgb learning rate")
    a = ap.parse_args()

    if a.cmd == "lb":
        record_lb_score(a.args[0], float(a.args[1]))
        print("recorded")
        return
    if a.cmd == "submit":
        submit_stage(RUNS_DIR / a.run, a.team)
        return

    cfg = Config(run_name=a.name or (f"{a.cmd}_{a.run}" if a.cmd in ("stage2", "decide", "rescore", "ce-train", "ce-apply") else f"dev{a.sample}" if a.sample < 1 else a.cmd),
                 sample=a.sample)
    cfg.max_candidates = a.max_cands
    cfg.model, cfg.device, cfg.s1_chunk = a.model, a.device, a.s1_chunk
    if a.rounds:
        cfg.xgb_rounds = a.rounds
    if a.depth:
        cfg.xgb_depth = a.depth
    if a.lr:
        cfg.xgb_lr = a.lr
    cfg.extra.update(train_max_s1=a.train_max_s1, val_max_s1=a.val_max_s1, max_df=a.max_df, threshold_override=a.threshold,
                     prep_workers=a.prep_workers, k_name=a.k_name, k_addr=a.k_addr, use_rr=not a.no_rr, use_comp=not a.no_comp, save_test_feats=a.save_test_feats,
                     rr_keep=a.rr_keep, rr_wide=tuple(int(x) for x in a.rr_wide.split(",")), rr_fit_s1=a.rr_fit_s1, rr_tau=a.rr_tau, rr_min=a.rr_min)
    run = Run(cfg.run_name, cfg.to_dict())
    try:
        with HwMonitor(run.dir / "hw.csv"):
            if a.cmd == "prep":
                run.start_stage("prep")
                prep("train", run, cfg)
                prep("test", run, cfg)
                run.end_stage()
            if a.cmd in ("train", "all"):
                train_stage(cfg, run)
                gc.collect()
            if a.cmd == "predict":
                predict_stage(cfg, run, RUNS_DIR / a.run)
            if a.cmd == "all":
                predict_stage(cfg, run, run.dir)
            if a.cmd == "retrain":
                retrain_stage(cfg, run, RUNS_DIR / a.run)
            if a.cmd in ("ce-train", "ce-apply"):
                import cross_encoder as ce
                me = sys.modules[__name__]
                ce_dir = Path(a.ce_dir) if a.ce_dir else ROOT / "models" / f"ce_{a.run}"
                if a.cmd == "ce-train":
                    ce.train_ce(me, run, RUNS_DIR / a.run, ce_dir, n_pairs=a.ce_pairs, epochs=a.ce_epochs, seed=cfg.seed)
                else:
                    ce.apply_ce(me, cfg, run, RUNS_DIR / a.run, RUNS_DIR / (a.feats_run or a.run), ce_dir)
            if a.cmd == "rescore":
                rescore_stage(cfg, run, RUNS_DIR / a.run, RUNS_DIR / a.feats_run)
            if a.cmd == "decide":
                src = RUNS_DIR / a.run
                m = json.loads((src / "metrics.json").read_text())
                dec = {"mode": m.get("decision_mode", "threshold"),
                       "param": m.get("decision_param", m.get("threshold", 0.5)), "excl": m.get("decision_excl", False)}
                if a.threshold:
                    dec = {"mode": "threshold", "param": a.threshold, "excl": dec["excl"]}
                run.log(f"re-deciding {src.name} pred/ with {dec}")
                decide_stage(cfg, run, src / "pred", dec)
            if a.cmd == "stage2":
                from stage2 import run_stage2
                run_stage2(RUNS_DIR / a.run, run, device=cfg.device, seed=cfg.seed, workers=cfg.n_jobs,
                           promote=a.promote)
        m = run.metrics
        run.finish(cfg.sample, notes=f"{m.get('model', '')}/{m.get('device', '')} {m.get('decision_mode', '')}:"
                                     f"{m.get('decision_param', '')}")
    except BaseException as e:  # noqa: BLE001
        run.log(f"FAILED: {type(e).__name__}: {e}")
        run.finish(cfg.sample, state="failed")
        raise


if __name__ == "__main__":
    main()
