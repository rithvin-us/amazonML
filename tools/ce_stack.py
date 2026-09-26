"""Apply a fine-tuned cross-encoder to a finished stage-1 run: rescore the uncertain band on val, competitors
and test, stack with p (monotone XGB on [logit p, ce]), re-tune the decision with competitors, re-decide test.

  python ce_stack.py <train_run_id> <test_run_id> <ce_model_dir>
Writes a new run dir (<name>_ce) with output/ + validation; baseline vs stacked val F0.5 printed first.
"""
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch
import xgboost as xgb
from sklearn.model_selection import GroupKFold
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, r"D:\amazon-ml\code\business_entity_resolution\src")
import pipeline as pp  # noqa: E402
from config import Config  # noqa: E402
from tracking import Run  # noqa: E402

TR, TE, CE = pp.RUNS_DIR / sys.argv[1], pp.RUNS_DIR / sys.argv[2], Path(sys.argv[3])
BAND, MAXLEN = (0.02, 0.995), 96
dev = "cuda" if torch.cuda.is_available() else "cpu"
t0 = time.time()
log = lambda m: print(f"[{time.time() - t0:6.0f}s] {m}", flush=True)  # noqa: E731
tok = AutoTokenizer.from_pretrained(CE)
model = AutoModelForSequenceClassification.from_pretrained(CE).to(dev).eval()


def texts(pairs: pl.DataFrame, split: str) -> pl.DataFrame:
    s1 = (pp.scan_norm(split, "s1").filter(pl.col("idx").is_in(pairs["s1_idx"].unique().implode()))
          .select(pl.col("idx").alias("s1_idx"), pl.concat_str(["name_full", "addr"], separator=" | ").alias("t1")).collect())
    po = (pp.scan_norm(split, "pool").filter(pl.col("idx").is_in(pairs["cand_idx"].unique().implode()))
          .select(pl.col("idx").alias("cand_idx"), pl.concat_str(["name_full", "addr"], separator=" | ").alias("t2")).collect())
    return pairs.join(s1, on="s1_idx", how="left").join(po, on="cand_idx", how="left").with_columns(
        pl.col("t1").fill_null(""), pl.col("t2").fill_null(""))


@torch.no_grad()
def ce_score(df: pl.DataFrame, bs: int = 512) -> np.ndarray:
    a, b, out = df["t1"].to_list(), df["t2"].to_list(), []
    for i in range(0, len(a), bs):
        enc = tok(a[i:i + bs], b[i:i + bs], truncation=True, max_length=MAXLEN, padding=True, return_tensors="pt").to(dev)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=dev == "cuda"):
            out.append(model(**enc).logits.squeeze(-1).float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0, np.float32)


def band_of(df: pl.DataFrame) -> pl.DataFrame:
    return df.filter((pl.col("p") >= BAND[0]) & (pl.col("p") < BAND[1]))


def with_ce(df: pl.DataFrame, split: str, what: str) -> pl.DataFrame:
    b = texts(band_of(df).select("s1_idx", "cand_idx", "p", *[c for c in ("label",) if c in df.columns]), split)
    ts = time.time()
    b = b.with_columns(pl.Series("ce", ce_score(b), dtype=pl.Float32)).drop("t1", "t2")
    log(f"{what}: band {b.height:,} pairs scored in {time.time() - ts:.0f}s")
    return b


X = lambda d: d.select((pl.col("p").log() - (1 - pl.col("p")).log()).alias("lp"), "ce").to_numpy()  # noqa: E731
PARAMS = {"objective": "binary:logistic", "max_depth": 3, "eta": 0.1, "monotone_constraints": "(1,1)", "tree_method": "hist"}


def restack(full: pl.DataFrame, band: pl.DataFrame, p2: np.ndarray) -> pl.DataFrame:
    """Replace p by the stacked p2 on band pairs; everything else keeps stage-1 p."""
    b = band.select("s1_idx", "cand_idx").with_columns(pl.Series("p2", p2, dtype=pl.Float32))
    return (full.join(b, on=["s1_idx", "cand_idx"], how="left", maintain_order="left")
            .with_columns(pl.coalesce("p2", "p").alias("p")).drop("p2"))


class _Quiet:
    def log(self, m):
        pass


# ---- val (labelled) + competitors (unlabelled, exclusivity only)
va = pl.read_parquet(TR / "val_scored.parquet")
vt = pl.read_parquet(TR / "val_truth.parquet")
comp = pl.read_parquet(TR / "comp_scored.parquet") if (TR / "comp_scored.parquet").exists() else None
vb = with_ce(va, "train", "val").join(vt.select("s1_idx", "block"), on="s1_idx", how="left")
yb = vb["label"].to_numpy()
oof = np.zeros(vb.height, np.float32)
for trn, tst in GroupKFold(n_splits=5).split(X(vb), yb, vb["block"].fill_null("?").to_numpy()):
    oof[tst] = xgb.train(PARAMS, xgb.DMatrix(X(vb)[trn], yb[trn]), 200).predict(xgb.DMatrix(X(vb)[tst]))
stacker = xgb.train(PARAMS, xgb.DMatrix(X(vb), yb), 200)
va2 = restack(va, vb, oof)
tvc = vt.select("s1_idx", "n_true")
if comp is not None:
    cb = with_ce(comp, "train", "competitors")
    comp2 = restack(comp, cb, stacker.predict(xgb.DMatrix(X(cb))))
    lab0 = lambda d: d.select("s1_idx", "cand_idx", pl.lit(0, pl.Int8).alias("label"), "p")  # noqa: E731
    base_u, new_u = pl.concat([va, lab0(comp)]), pl.concat([va2, lab0(comp2)])
else:
    base_u, new_u = va, va2
b0 = max(pp.tune_decision(base_u, tvc, 0.02, _Quiet()).values(), key=lambda d: d["f05"])
res = pp.tune_decision(new_u, tvc, 0.02, _Quiet())
b1 = max(res.values(), key=lambda d: d["f05"])
log(f"RESULT val F0.5 (with competitors) stage1 {b0['f05']:.5f} ({b0['mode']}:{b0['param']} excl={b0['excl']}) -> "
    f"+CE {b1['f05']:.5f} ({b1['mode']}:{b1['param']} excl={b1['excl']})  gain {b1['f05'] - b0['f05']:+.5f}")
dec = {"mode": b1["mode"], "param": b1["param"], "excl": b1["excl"]}

# ---- test: rescore band, re-decide with the stacked-tuned decision
spill_old = TE / "pred"
sc = pl.read_parquet(spill_old / "scored-*.parquet")
tb = with_ce(sc, "test", "test")
sc2 = restack(sc, tb, stacker.predict(xgb.DMatrix(X(tb))))
cfg = Config(run_name=f"{TE.name.split('-', 2)[-1]}_ce")
run = Run(cfg.run_name, {**cfg.to_dict(), "ce_model": str(CE), "decision": dec, "train_run": TR.name, "test_run": TE.name})
spill = run.dir / "pred"
spill.mkdir()
for f in spill_old.glob("cand-*.parquet"):
    shutil.copy(f, spill / f.name)
sc2.write_parquet(spill / "scored-00000.parquet")
run.set_metrics(val_f05=round(b1["f05"], 5), val_f05_stage1=round(b0["f05"], 5), decision_mode=dec["mode"],
                decision_param=dec["param"], decision_excl=dec["excl"], test_band_pairs=tb.height)
pp.decide_stage(cfg, run, spill, dec)
run.finish(1.0, notes=f"CE stack {dec['mode']}:{dec['param']}")
log(f"done -> {run.dir}")
