"""Segment 6 gate: fine-tune a MiniLM cross-encoder on train pairs, rescore the val uncertain band, stack with
stage-1 p (OOF by block), and compare tuned val F0.5 against the stage-1 baseline.

  python ce_experiment.py <run_id> [n_train_pairs] [epochs]
Model: cross-encoder/ms-marco-MiniLM-L6-v2 (Apache-2.0). Saved to D:/amazon-ml/models/ce_<run>.
"""
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, r"D:\amazon-ml\code\business_entity_resolution\src")
import pipeline as pp  # noqa: E402

RUN = sys.argv[1]
N_TRAIN = int(sys.argv[2]) if len(sys.argv) > 2 else 300_000
EPOCHS = int(sys.argv[3]) if len(sys.argv) > 3 else 1
TAG = sys.argv[4] if len(sys.argv) > 4 else ""
BAND = (0.02, 0.995)
MAXLEN, BS, LR = 96, 64, 3e-5
BASE = "cross-encoder/ms-marco-MiniLM-L6-v2"
R = pp.RUNS_DIR / RUN
OUT = pp.ROOT / "models" / f"ce_{RUN}{TAG}"
dev = "cuda" if torch.cuda.is_available() else "cpu"
t0 = time.time()
log = lambda m: print(f"[{time.time() - t0:6.0f}s] {m}", flush=True)  # noqa: E731


def texts(pairs: pl.DataFrame) -> pl.DataFrame:
    """Attach 'name | addr' strings for the S1 side (t1) and the candidate side (t2)."""
    s1 = (pp.scan_norm("train", "s1").filter(pl.col("idx").is_in(pairs["s1_idx"].unique().implode()))
          .select(pl.col("idx").alias("s1_idx"), pl.concat_str(["name_full", "addr"], separator=" | ").alias("t1")).collect())
    po = (pp.scan_norm("train", "pool").filter(pl.col("idx").is_in(pairs["cand_idx"].unique().implode()))
          .select(pl.col("idx").alias("cand_idx"), pl.concat_str(["name_full", "addr"], separator=" | ").alias("t2")).collect())
    return pairs.join(s1, on="s1_idx", how="left").join(po, on="cand_idx", how="left").with_columns(
        pl.col("t1").fill_null(""), pl.col("t2").fill_null(""))


# ---- training pairs: non-val train S1, "hard" region by re-ranker score (out-of-sample for train S1)
parts = sorted((R / "train_feats").glob("part-*.parquet"))
tr = pl.concat([pl.read_parquet(f, columns=["s1_idx", "cand_idx", "label", "rr", "nc_tset", "ad_tset", "is_val"])
                .filter(~pl.col("is_val")) for f in parts])
hard = tr.filter((pl.col("rr") >= 0.02) | (pl.col("nc_tset") >= 70) | (pl.col("ad_tset") >= 80))
pos, neg = hard.filter(pl.col("label") == 1), hard.filter(pl.col("label") == 0)
n_pos = min(pos.height, N_TRAIN // 2)
tr = pl.concat([pos.sample(n=n_pos, seed=1), neg.sample(n=min(neg.height, N_TRAIN - n_pos), seed=1)]).sample(fraction=1.0, shuffle=True, seed=2)
log(f"train pairs {tr.height:,} (pos {n_pos:,}) from {hard.height:,} hard / {len(parts)} parts")
tr = texts(tr.select("s1_idx", "cand_idx", "label"))

tok = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForSequenceClassification.from_pretrained(BASE, num_labels=1).to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
steps = EPOCHS * (tr.height // BS)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=steps, pct_start=0.06)
scaler = torch.amp.GradScaler(enabled=dev == "cuda")
lossf = torch.nn.BCEWithLogitsLoss()
t1, t2, y = tr["t1"].to_list(), tr["t2"].to_list(), tr["label"].to_numpy().astype(np.float32)
model.train()
step = 0
for ep in range(EPOCHS):
    order = np.random.default_rng(ep).permutation(len(y))
    for i in range(0, len(y) - BS + 1, BS):
        b = order[i:i + BS]
        enc = tok([t1[j] for j in b], [t2[j] for j in b], truncation=True, max_length=MAXLEN, padding=True,
                  return_tensors="pt").to(dev)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=dev == "cuda"):
            logit = model(**enc).logits.squeeze(-1)
            loss = lossf(logit.float(), torch.from_numpy(y[b]).to(dev))
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        sched.step()
        step += 1
        if step % 500 == 0:
            log(f"ep {ep} step {step}/{steps} loss {loss.item():.4f}")
OUT.mkdir(parents=True, exist_ok=True)
model.save_pretrained(OUT)
tok.save_pretrained(OUT)
log(f"saved {OUT}")


@torch.no_grad()
def score(df: pl.DataFrame, bs: int = 512) -> np.ndarray:
    model.eval()
    a, b, out = df["t1"].to_list(), df["t2"].to_list(), []
    for i in range(0, len(a), bs):
        enc = tok(a[i:i + bs], b[i:i + bs], truncation=True, max_length=MAXLEN, padding=True, return_tensors="pt").to(dev)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=dev == "cuda"):
            out.append(model(**enc).logits.squeeze(-1).float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0, np.float32)


# ---- val: rescore band, stack OOF by block, compare tuned F0.5
va = pl.read_parquet(R / "val_scored.parquet")
vt = pl.read_parquet(R / "val_truth.parquet")
band = va.filter((pl.col("p") >= BAND[0]) & (pl.col("p") < BAND[1]))
ts = time.time()
band = texts(band)
band = band.with_columns(pl.Series("ce", score(band), dtype=pl.Float32))
log(f"val band {band.height:,} pairs scored in {time.time() - ts:.0f}s ({band.height / max(time.time() - ts, 1e-6):.0f}/s)")
band = band.join(vt.select("s1_idx", "block"), on="s1_idx", how="left")

import xgboost as xgb  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402

X = band.select(pl.col("p").log() - (1 - pl.col("p")).log(), "ce").to_numpy()
yb = band["label"].to_numpy()
oof = np.zeros(len(yb), np.float32)
groups = band["block"].fill_null("?").to_numpy()
for trn, tst in GroupKFold(n_splits=5).split(X, yb, groups):
    m = xgb.train({"objective": "binary:logistic", "max_depth": 3, "eta": 0.1, "monotone_constraints": "(1,1)",
                   "tree_method": "hist"}, xgb.DMatrix(X[trn], yb[trn]), 200)
    oof[tst] = m.predict(xgb.DMatrix(X[tst]))
band = band.with_columns(pl.Series("p2", oof))
new = va.join(band.select("s1_idx", "cand_idx", "p2"), on=["s1_idx", "cand_idx"], how="left").with_columns(
    pl.coalesce("p2", "p").alias("p")).drop("p2")
tvc = vt.select("s1_idx", "n_true")


class _R:
    def log(self, m):
        pass


b0 = max(pp.tune_decision(va, tvc, 0.02, _R()).values(), key=lambda d: d["f05"])
b1 = max(pp.tune_decision(new, tvc, 0.02, _R()).values(), key=lambda d: d["f05"])
from sklearn.metrics import roc_auc_score  # noqa: E402
log(f"band AUC stage1 p {roc_auc_score(yb, band['p'].to_numpy()):.4f}  ce {roc_auc_score(yb, band['ce'].to_numpy()):.4f}  "
    f"stacked OOF {roc_auc_score(yb, oof):.4f}")
log(f"RESULT val F0.5 stage1 {b0['f05']:.5f} ({b0['mode']}:{b0['param']})  ->  +CE {b1['f05']:.5f} ({b1['mode']}:{b1['param']})"
    f"  gain {b1['f05'] - b0['f05']:+.5f}")
