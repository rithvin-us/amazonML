"""End-to-end CLI (memory-lean: streamed prep, per-country blocking, chunked scoring).

  python pipeline.py prep                 # normalise all sources -> cache/norm_<split>_<kind>/part-*.parquet
  python pipeline.py all [--sample 0.02]  # prep + train/val + tune + test predict + validate
  python pipeline.py train [--sample f]   # train/val only (no test)
  python pipeline.py predict --run <id>   # test inference with a trained run's model
  python pipeline.py submit --run <id> --team NAME   # build final zip
  python pipeline.py lb <run_id> <score>  # record portal leaderboard score
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

import lightgbm as lgb
import numpy as np
import polars as pl

from blocking import BlockIndex
from config import CACHE_DIR, OUTPUT_DIR, ROOT, RUNS_DIR, VALIDATOR, Config
from features import FEATURES, TEXT_COLS, build_features
from hwmon import HwMonitor
from io_utils import load_ground_truth
from tracking import Run, record_lb_score

SRC_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SRC_DIR.parent
NORM_COLS = ["idx", "entity_id"] + TEXT_COLS


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
def iter_country_blocks(split: str, s1_all: pl.DataFrame, cfg: Config, run: Run, s1_chunk: int = 50_000):
    """Yield (country, s1_part, pool_c, pairs_features) per S1 chunk, one country pool in RAM at a time."""
    cs = s1_all["country_n"].unique().sort().to_list()
    done, n_all = 0, s1_all.height
    for c in cs:
        s1_c = s1_all.filter(pl.col("country_n") == c)
        pool_c = scan_norm(split, "pool").filter(pl.col("country_n") == c).select(NORM_COLS).collect()
        run.log(f"[{c}] S1 {s1_c.height:,}  pool {pool_c.height:,}: building index")
        index = BlockIndex(pool_c, max_df=cfg.extra.get("max_df", 150))
        run.log(f"[{c}] index keys kept {index.idf.height:,}/{index.n_keys_total:,}, postings {index.pk.height:,}")
        for i in range(0, s1_c.height, s1_chunk):
            part = s1_c.slice(i, s1_chunk)
            pairs = index.query(part, top_k=cfg.max_candidates, chunk=cfg.chunk_size)
            feats = build_features(pairs, part, pool_c, workers=cfg.n_jobs)
            done += part.height
            run.progress(done / n_all, f"[{c}] {done:,}/{n_all:,} S1 blocked+featurised")
            yield c, part, pool_c, feats
        del index, pool_c
        gc.collect()


# ---------------------------------------------------------------- decision / tuning
def macro_f05_frame(scored: pl.DataFrame, truth_counts: pl.DataFrame, t: float) -> dict:
    """scored: s1_idx, label, p. truth_counts: s1_idx, n_true (all eval S1, incl. 0)."""
    agg = (scored.filter(pl.col("p") >= t).group_by("s1_idx")
           .agg(pl.len().alias("n_pred"), pl.col("label").sum().alias("tp")))
    df = truth_counts.join(agg, on="s1_idx", how="left").fill_null(0).with_columns(
        (pl.col("tp") / pl.col("n_pred").clip(1)).alias("P"),
        (pl.col("tp") / pl.col("n_true").clip(1)).alias("R"))
    df = df.with_columns(
        pl.when((pl.col("n_true") == 0) & (pl.col("n_pred") == 0)).then(1.0)
        .when(pl.col("tp") == 0).then(0.0)
        .otherwise(1.25 * pl.col("P") * pl.col("R") / (0.25 * pl.col("P") + pl.col("R"))).alias("f"))
    return {"f05": float(df["f"].mean()),
            "precision": float(df["tp"].sum() / max(df["n_pred"].sum(), 1)),
            "recall": float(df["tp"].sum() / max(df["n_true"].sum(), 1))}


def tune_threshold(scored, truth_counts, run: Run) -> tuple[float, dict]:
    best = (0.5, {"f05": -1})
    for t in np.round(np.arange(0.05, 0.96, 0.025), 3):
        m = macro_f05_frame(scored, truth_counts, float(t))
        if m["f05"] > best[1]["f05"]:
            best = (float(t), m)
    run.log(f"best threshold {best[0]} -> {best[1]}")
    return best


# ---------------------------------------------------------------- train
def train_stage(cfg: Config, run: Run) -> None:
    run.start_stage("prep_train")
    prep("train", run, cfg)
    run.end_stage()

    gt = load_ground_truth()
    s1_ids = scan_norm("train", "s1").select("entity_id").collect()["entity_id"]
    n_use = min(int(s1_ids.len() * cfg.sample), cfg.extra.get("train_max_s1", 150_000))
    use = s1_ids.sample(n=n_use, seed=cfg.seed, shuffle=True)
    val_ids = use.head(int(n_use * cfg.val_frac))
    s1 = (scan_norm("train", "s1").filter(pl.col("entity_id").is_in(use.implode())).select(NORM_COLS)
          .collect().with_columns(pl.col("entity_id").is_in(val_ids.implode()).alias("is_val")))
    run.log(f"S1 used {s1.height:,} (val {val_ids.len():,})")

    idmap_s1 = s1.select(pl.col("idx").alias("s1_idx"), pl.col("entity_id").alias("s1_id"), "is_val")
    gt_use = gt.join(idmap_s1, on="s1_id")
    truth_counts = (idmap_s1.join(gt_use.group_by("s1_id").agg(pl.col("match_id").drop_nulls().len().alias("n_true")),
                                  on="s1_id", how="left").fill_null(0))

    run.start_stage("blocking+features_train")
    chunks = []
    for c, part, pool_c, feats in iter_country_blocks("train", s1, cfg, run):
        ids = pool_c.select(pl.col("idx").alias("cand_idx"), pl.col("entity_id").alias("match_id"))
        pos = (gt_use.drop_nulls().join(ids, on="match_id")
               .select("s1_idx", "cand_idx").with_columns(pl.lit(1, pl.Int8).alias("label")))
        chunks.append(feats.join(pos, on=["s1_idx", "cand_idx"], how="left")
                      .with_columns(pl.col("label").fill_null(0)))
    feats = pl.concat(chunks).join(idmap_s1.select("s1_idx", "is_val"), on="s1_idx")
    del chunks
    gc.collect()
    run.end_stage()

    tv = truth_counts.filter(pl.col("is_val"))
    br = float(feats.filter(pl.col("is_val"))["label"].sum() / max(tv["n_true"].sum(), 1))
    avg_c = feats.filter(pl.col("is_val")).height / max(tv.height, 1)
    run.log(f"metric val block_recall={br:.4f} avg_cands={avg_c:.1f} pos_rate={feats['label'].mean():.4f}")
    run.set_metrics(block_recall=round(br, 4), avg_candidates=round(avg_c, 2),
                    n_pairs=feats.height, n_s1=s1.height)

    run.start_stage("train_lgbm")
    tr, va = feats.filter(~pl.col("is_val")), feats.filter(pl.col("is_val"))
    params = dict(objective="binary", learning_rate=cfg.lgb_lr, num_leaves=cfg.lgb_leaves,
                  min_data_in_leaf=50, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
                  lambda_l2=1.0, num_threads=cfg.n_jobs, verbose=-1, seed=cfg.seed)
    dtr = lgb.Dataset(tr.select(FEATURES).to_numpy(), tr["label"].to_numpy(), feature_name=FEATURES)
    dva = lgb.Dataset(va.select(FEATURES).to_numpy(), va["label"].to_numpy(), reference=dtr)

    def _cb(env):
        if env.iteration % 25 == 0:
            msg = " ".join(f"{e[1]}={e[2]:.4f}" for e in env.evaluation_result_list)
            run.progress(env.iteration / cfg.lgb_rounds, f"iter {env.iteration} {msg}")

    model = lgb.train(params, dtr, cfg.lgb_rounds, valid_sets=[dva], valid_names=["val"],
                      callbacks=[lgb.early_stopping(50, verbose=False), _cb])
    model.save_model(str(run.dir / "model.txt"))
    imp = dict(sorted(zip(FEATURES, model.feature_importance("gain").round(1).tolist()), key=lambda x: -x[1]))
    (run.dir / "feature_importance.json").write_text(json.dumps(imp, indent=2))
    run.log(f"best iter {model.best_iteration}")
    run.end_stage()

    run.start_stage("tune")
    va = va.with_columns(pl.Series("p", model.predict(va.select(FEATURES).to_numpy(),
                                                      num_iteration=model.best_iteration)))
    t, m = tune_threshold(va.select("s1_idx", "label", "p"), tv.select("s1_idx", "n_true"), run)
    run.set_metrics(threshold=t, val_f05=round(m["f05"], 5), val_precision=round(m["precision"], 4),
                    val_recall=round(m["recall"], 4), best_iter=model.best_iteration,
                    val_singleton_frac=round(tv.filter(pl.col("n_true") == 0).height / max(tv.height, 1), 4))
    # per-country val breakdown (tells us how France-like generalisation may behave)
    cmap = s1.select(pl.col("idx").alias("s1_idx"), "country_n")
    per_c = {}
    for c in cmap["country_n"].unique().to_list():
        ids_c = cmap.filter(pl.col("country_n") == c)["s1_idx"].implode()
        per_c[c] = round(macro_f05_frame(va.filter(pl.col("s1_idx").is_in(ids_c)).select("s1_idx", "label", "p"),
                                         tv.filter(pl.col("s1_idx").is_in(ids_c)).select("s1_idx", "n_true"), t)["f05"], 5)
    run.set_metrics(val_f05_by_country=per_c)
    run.log(f"metric val_f05={m['f05']:.5f} P={m['precision']:.4f} R={m['recall']:.4f} t={t} by_country={per_c}")
    run.end_stage()


# ---------------------------------------------------------------- predict
def predict_stage(cfg: Config, run: Run, model_run_dir: Path) -> None:
    metrics = json.loads((model_run_dir / "metrics.json").read_text())
    t = cfg.extra.get("threshold_override") or metrics["threshold"]
    model = lgb.Booster(model_file=str(model_run_dir / "model.txt"))

    run.start_stage("prep_test")
    prep("test", run, cfg)
    run.end_stage()
    s1 = scan_norm("test", "s1").select(NORM_COLS).collect()

    run.start_stage("blocking+features+predict_test")
    cand_rows, match_rows = [], []
    n_pairs = 0
    for c, part, pool_c, feats in iter_country_blocks("test", s1, cfg, run):
        p = model.predict(feats.select(FEATURES).to_numpy(), num_iteration=model.best_iteration or None)
        f = (feats.select("s1_idx", "cand_idx", "brank").with_columns(pl.Series("p", p))
             .join(part.select(pl.col("idx").alias("s1_idx"), pl.col("entity_id").alias("s1_id")), on="s1_idx")
             .join(pool_c.select(pl.col("idx").alias("cand_idx"), pl.col("entity_id").alias("cid")), on="cand_idx"))
        n_pairs += f.height
        cand_rows.append(f.sort("brank").group_by("s1_id").agg(pl.col("cid").str.join(",").alias("ids")))
        match_rows.append(f.filter(pl.col("p") >= t).sort("p", descending=True)
                          .group_by("s1_id").agg(pl.col("cid").str.join(",").alias("ids")))
    run.end_stage()

    run.start_stage("write_test")
    all_s1 = s1.select(pl.col("entity_id").alias("source1_entity_id"))
    cands = all_s1.join(pl.concat(cand_rows), left_on="source1_entity_id", right_on="s1_id", how="left",
                        maintain_order="left").fill_null("")
    matches = all_s1.join(pl.concat(match_rows), left_on="source1_entity_id", right_on="s1_id", how="left",
                          maintain_order="left").fill_null("")
    out_dir = run.dir / "output"
    out_dir.mkdir(exist_ok=True)
    cands.rename({"ids": "candidate_entity_ids"}).write_csv(out_dir / "candidate_pairs.tsv", separator="\t",
                                                            quote_style="never")
    matches.rename({"ids": "matched_entity_ids"}).write_csv(out_dir / "matching_results.tsv", separator="\t",
                                                            quote_style="never")
    for fn in ("matching_results.tsv", "candidate_pairs.tsv"):
        shutil.copy(out_dir / fn, OUTPUT_DIR / fn)
    nonempty = int((matches["ids"] != "").sum())
    npred = int(matches["ids"].str.split(",").list.len().filter(matches["ids"] != "").sum())
    run.set_metrics(test_threshold=t, test_s1=all_s1.height, test_nonempty=nonempty, test_pred_pairs=npred,
                    test_avg_candidates=round(n_pairs / max(all_s1.height, 1), 2))
    run.log(f"metric test nonempty={nonempty:,}/{all_s1.height:,} pred_pairs={npred:,}")
    run.end_stage()

    run.start_stage("validate")
    r = subprocess.run([sys.executable, str(VALIDATOR), "--matching", str(OUTPUT_DIR / "matching_results.tsv"),
                        "--candidate", str(OUTPUT_DIR / "candidate_pairs.tsv"),
                        "--test-dir", str(ROOT / "student_resource" / "dataset" / "test")],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    (run.dir / "validate.txt").write_text(r.stdout + r.stderr, encoding="utf-8")
    tail = (r.stdout.strip().splitlines() or [r.stderr[-300:]])[-1]
    run.log(f"metric validator exit={r.returncode}: {tail}")
    run.set_metrics(validator_pass=r.returncode == 0)
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
    ap.add_argument("--run", default=None, help="run id holding model.txt (predict/submit)")
    ap.add_argument("--team", default="team")
    ap.add_argument("--max-cands", type=int, default=40)
    ap.add_argument("--train-max-s1", type=int, default=150_000)
    ap.add_argument("--max-df", type=int, default=150)
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--prep-workers", type=int, default=6)
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
    cfg.extra.update(train_max_s1=a.train_max_s1, max_df=a.max_df, threshold_override=a.threshold,
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
            if a.cmd == "predict":
                predict_stage(cfg, run, RUNS_DIR / a.run)
            if a.cmd == "all":
                predict_stage(cfg, run, run.dir)
        run.finish(cfg.sample)
    except BaseException as e:  # noqa: BLE001
        run.log(f"FAILED: {type(e).__name__}: {e}")
        run.finish(cfg.sample, state="failed")
        raise


if __name__ == "__main__":
    main()
