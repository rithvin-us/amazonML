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
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import polars as pl
import psutil

from blocking import BlockIndex
from config import CACHE_DIR, OUTPUT_DIR, ROOT, RUNS_DIR, VALIDATOR, Config
from features import FEATURES, TEXT_COLS, build_features
from hwmon import HwMonitor
from io_utils import load_ground_truth
from tracking import Run, record_lb_score

SRC_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SRC_DIR.parent
NORM_COLS = ["idx", "entity_id"] + TEXT_COLS
THRESHOLD_GRID = np.round(np.arange(0.05, 0.96, 0.025), 3)
ALPHA_GRID = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0]


def avail_gb() -> float:
    return psutil.virtual_memory().available / 1e9


# ---------------------------------------------------------------- prep (streamed, separate process)
def norm_dir(split: str, kind: str) -> Path:
    return CACHE_DIR / f"norm_{split}_{kind}"


def prep(split: str, run: Run, cfg: Config) -> None:
    """Runs prep.py in a child process (light imports -> small multiprocessing workers)."""
    if all((norm_dir(split, k) / "_DONE").exists() for k in ("s1", "pool")):
        run.log(f"cache hit norm_{split}_*")
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
def iter_country_blocks(split: str, s1_all: pl.DataFrame, cfg: Config, run: Run):
    """Yield (country, s1_part, pool_c, pairs_features) per S1 chunk, one country pool in RAM at a time."""
    cs = s1_all["country_n"].unique().sort().to_list()
    done, n_all = 0, s1_all.height
    for c in cs:
        s1_c = s1_all.filter(pl.col("country_n") == c)
        pool_c = scan_norm(split, "pool").filter(pl.col("country_n") == c).select(NORM_COLS).collect()
        run.log(f"[{c}] S1 {s1_c.height:,}  pool {pool_c.height:,}: building index (RAM avail {avail_gb():.1f}GB)")
        index = BlockIndex(pool_c, max_df=cfg.extra.get("max_df", 150),
                           chunk=cfg.extra.get("index_chunk", 100_000))  # smaller key-build batches = lower peak RAM
        run.log(f"[{c}] index keys kept {len(index.idf):,}/{index.n_keys_total:,}, postings {index.n_postings:,}"
                f" (RAM avail {avail_gb():.1f}GB)")
        for i in range(0, s1_c.height, cfg.s1_chunk):
            part = s1_c.slice(i, cfg.s1_chunk)
            pairs = index.query(part, top_k=cfg.max_candidates, chunk=cfg.chunk_size)
            feats = build_features(pairs, part, pool_c, workers=cfg.n_jobs)
            done += part.height
            run.progress(done / n_all, f"[{c}] {done:,}/{n_all:,} S1 blocked+featurised")
            yield c, part, pool_c, feats
        del index, pool_c
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
    return pl.concat([pl.read_parquet(f).filter(~pl.col("is_val")) for f in files])


def train_model(cfg: Config, run: Run, train_files: list[Path], va: pl.DataFrame) -> Model:
    """train_files: per-chunk feature parquet on D: (is_val rows skipped); va: in-RAM validation frame."""
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
                df = pl.read_parquet(train_files[self.i]).filter(~pl.col("is_val"))
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


def apply_decision(scored: pl.DataFrame, dec: dict, floor: float) -> pl.DataFrame:
    if dec["mode"] == "threshold":
        return scored.filter(pl.col("p") >= dec["param"])
    return select_expected_f(scored.filter(pl.col("p") >= floor), dec["param"])


def exclusive(sel: pl.DataFrame) -> pl.DataFrame:
    """GT is one-to-many: a pool record matches at most one S1. Keep each cand only for its best-p S1."""
    return sel.filter(pl.col("p").rank("ordinal", descending=True).over("cand_idx") == 1)


def tune_decision(scored: pl.DataFrame, truth_counts: pl.DataFrame, floor: float, run: Run) -> dict:
    res = {}
    for t in THRESHOLD_GRID:
        m = eval_selection(apply_decision(scored, {"mode": "threshold", "param": float(t)}, floor), truth_counts)
        if m["f05"] > res.get("threshold", {"f05": -1})["f05"]:
            res["threshold"] = {"mode": "threshold", "param": float(t), **m}
    for a in ALPHA_GRID:
        m = eval_selection(apply_decision(scored, {"mode": "expected_f", "param": a}, floor), truth_counts)
        if m["f05"] > res.get("expected_f", {"f05": -1})["f05"]:
            res["expected_f"] = {"mode": "expected_f", "param": a, **m}
    for k, v in res.items():
        run.log(f"decision {k}: param={v['param']} f05={v['f05']:.5f} P={v['precision']:.4f} R={v['recall']:.4f}")
    return res


# ---------------------------------------------------------------- train
def train_stage(cfg: Config, run: Run) -> None:
    run.start_stage("prep_train")
    prep("train", run, cfg)
    run.end_stage()

    gt = load_ground_truth()
    s1_ids = scan_norm("train", "s1").select("entity_id").collect()["entity_id"]
    n_use = min(int(s1_ids.len() * cfg.sample), cfg.extra.get("train_max_s1", 150_000))
    use = s1_ids.sample(n=n_use, seed=cfg.seed, shuffle=True)
    val_ids = use.head(min(int(n_use * cfg.val_frac), cfg.extra.get("val_max_s1", 40_000)))
    s1 = (scan_norm("train", "s1").filter(pl.col("entity_id").is_in(use.implode())).select(NORM_COLS)
          .collect().with_columns(pl.col("entity_id").is_in(val_ids.implode()).alias("is_val")))
    run.log(f"S1 used {s1.height:,} (val {val_ids.len():,})")

    idmap_s1 = s1.select(pl.col("idx").alias("s1_idx"), pl.col("entity_id").alias("s1_id"), "is_val")
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
    files, val_parts = [], []
    n_pairs = n_pos = 0
    pos_all = gt_use.drop_nulls()
    for c, part, pool_c, feats in iter_country_blocks("train", s1, cfg, run):
        ids = pool_c.select(pl.col("idx").alias("cand_idx"), pl.col("entity_id").alias("match_id"))
        pos = (pos_all.join(ids, on="match_id")
               .select("s1_idx", "cand_idx").with_columns(pl.lit(1, pl.Int8).alias("label")))
        f = (feats.join(pos, on=["s1_idx", "cand_idx"], how="left").with_columns(pl.col("label").fill_null(0))
             .join(idmap_s1.select("s1_idx", "is_val"), on="s1_idx"))
        n_pairs += f.height
        n_pos += int(f["label"].sum())
        val_parts.append(f.filter(pl.col("is_val")))
        files.append(fdir / f"part-{len(files):05d}.parquet")
        f.write_parquet(files[-1])
        del f, feats
    va = pl.concat(val_parts)
    del val_parts
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
    model = train_model(cfg, run, files, va)
    gc.collect()
    run.end_stage()

    run.start_stage("tune")
    va = va.select("s1_idx", "cand_idx", "label").with_columns(
        pl.Series("p", model.predict(va.select(FEATURES).to_numpy()), dtype=pl.Float32))
    va.write_parquet(run.dir / "val_scored.parquet")  # for offline error analysis / re-decisions
    tvc = tv.select("s1_idx", "n_true")
    res = tune_decision(va, tvc, cfg.p_floor, run)
    dec = max(res.values(), key=lambda d: d["f05"])
    ex = eval_selection(exclusive(apply_decision(va, dec, cfg.p_floor)), tvc)
    run.set_metrics(decision_mode=dec["mode"], decision_param=dec["param"],
                    threshold=res["threshold"]["param"], val_f05=round(dec["f05"], 5),
                    val_precision=round(dec["precision"], 4), val_recall=round(dec["recall"], 4),
                    val_f05_threshold=round(res["threshold"]["f05"], 5),
                    val_f05_expected_f=round(res["expected_f"]["f05"], 5),
                    val_f05_excl=round(ex["f05"], 5),
                    val_singleton_frac=round(tv.filter(pl.col("n_true") == 0).height / max(tv.height, 1), 4))
    # per-country val breakdown (tells us how France-like generalisation may behave)
    cmap = s1.select(pl.col("idx").alias("s1_idx"), "country_n")
    per_c = {}
    for c in cmap["country_n"].unique().to_list():
        ids_c = cmap.filter(pl.col("country_n") == c)["s1_idx"].implode()
        per_c[c] = round(eval_selection(apply_decision(va.filter(pl.col("s1_idx").is_in(ids_c)), dec, cfg.p_floor),
                                        tvc.filter(pl.col("s1_idx").is_in(ids_c)))["f05"], 5)
    run.set_metrics(val_f05_by_country=per_c)
    run.log(f"metric val_f05={dec['f05']:.5f} P={dec['precision']:.4f} R={dec['recall']:.4f} "
            f"decision={dec['mode']}:{dec['param']} by_country={per_c}")
    run.end_stage()


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
           "param": metrics.get("decision_param", metrics.get("threshold", 0.5))}
    if cfg.extra.get("threshold_override"):
        dec = {"mode": "threshold", "param": cfg.extra["threshold_override"]}
    model = Model.load(model_run_dir, cfg.device)
    run.log(f"model {model.kind} on {model.device}, decision {dec['mode']}:{dec['param']}")

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
    for c, part, pool_c, feats in iter_country_blocks("test", s1, cfg, run):
        p = model.predict(feats.select(FEATURES).to_numpy())
        f = (feats.select("s1_idx", "cand_idx", "brank").with_columns(pl.Series("p", p, dtype=pl.Float32))
             .join(pool_c.select(pl.col("idx").alias("cand_idx"), pl.col("entity_id").alias("cid")), on="cand_idx"))
        n_pairs += f.height
        (f.sort(["s1_idx", "brank"]).group_by("s1_idx", maintain_order=True)
         .agg(pl.col("cid").str.join(",").alias("ids")).write_parquet(spill / f"cand-{part_no:05d}.parquet"))
        f.filter(pl.col("p") >= cfg.p_floor).select("s1_idx", "cand_idx", "cid", "p").write_parquet(
            spill / f"scored-{part_no:05d}.parquet")
        part_no += 1
        del p, f, feats
    run.end_stage()

    run.start_stage("decide+write_test")
    s1_map = s1.select(pl.col("idx").alias("s1_idx"), pl.col("entity_id").alias("source1_entity_id"))
    n_s1 = s1_map.height
    del s1
    gc.collect()
    out_dir = run.dir / "output"
    out_dir.mkdir(exist_ok=True)
    _write_ids(s1_map, pl.read_parquet(spill / "cand-*.parquet"), "candidate_entity_ids",
               out_dir / "candidate_pairs.tsv")
    gc.collect()
    scored = pl.read_parquet(spill / "scored-*.parquet")
    sel = apply_decision(scored, dec, cfg.p_floor)
    excl = exclusive(sel)
    del scored
    _write_ids(s1_map, _join_ids(sel), "matched_entity_ids", out_dir / "matching_results.tsv")
    _write_ids(s1_map, _join_ids(excl), "matched_entity_ids", out_dir / "matching_results_excl.tsv")
    for fn in ("matching_results.tsv", "candidate_pairs.tsv"):
        shutil.copy(out_dir / fn, OUTPUT_DIR / fn)
    (OUTPUT_DIR / "variants").mkdir(exist_ok=True)
    shutil.copy(out_dir / "matching_results_excl.tsv", OUTPUT_DIR / "variants" / "matching_results_excl.tsv")
    nonempty = sel["s1_idx"].n_unique()
    run.set_metrics(test_decision=f"{dec['mode']}:{dec['param']}", test_s1=n_s1, test_nonempty=nonempty,
                    test_pred_pairs=sel.height, test_pred_pairs_excl=excl.height,
                    test_excl_dropped=sel.height - excl.height,
                    test_avg_candidates=round(n_pairs / max(n_s1, 1), 2))
    run.log(f"metric test nonempty={nonempty:,}/{n_s1:,} pred_pairs={sel.height:,} "
            f"excl_dropped={sel.height - excl.height:,} avg_cands={n_pairs / max(n_s1, 1):.1f}")
    del sel, excl
    gc.collect()
    run.end_stage()

    run.start_stage("validate")
    test_dir = ROOT / "student_resource" / "dataset" / "test"
    ok = True
    report = []
    for fn in ("matching_results.tsv", "variants/matching_results_excl.tsv"):
        r = subprocess.run([sys.executable, str(VALIDATOR), "--matching", str(OUTPUT_DIR / fn),
                            "--test-dir", str(test_dir)],
                           capture_output=True, text=True, encoding="utf-8", errors="replace")
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
    doc = ROOT / "student_resource" / "Documentation_template.md"
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
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prep", "train", "predict", "all", "submit", "lb"])
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
    a = ap.parse_args()

    if a.cmd == "lb":
        record_lb_score(a.args[0], float(a.args[1]))
        print("recorded")
        return
    if a.cmd == "submit":
        submit_stage(RUNS_DIR / a.run, a.team)
        return

    cfg = Config(run_name=a.name or (f"dev{a.sample}" if a.sample < 1 else a.cmd), sample=a.sample)
    cfg.max_candidates = a.max_cands
    cfg.model, cfg.device, cfg.s1_chunk = a.model, a.device, a.s1_chunk
    cfg.extra.update(train_max_s1=a.train_max_s1, val_max_s1=a.val_max_s1, max_df=a.max_df, threshold_override=a.threshold,
                     prep_workers=a.prep_workers)
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
        m = run.metrics
        run.finish(cfg.sample, notes=f"{m.get('model', '')}/{m.get('device', '')} {m.get('decision_mode', '')}:"
                                     f"{m.get('decision_param', '')}")
    except BaseException as e:  # noqa: BLE001
        run.log(f"FAILED: {type(e).__name__}: {e}")
        run.finish(cfg.sample, state="failed")
        raise


if __name__ == "__main__":
    main()
