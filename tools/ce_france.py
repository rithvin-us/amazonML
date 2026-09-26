"""Adapt a fine-tuned cross-encoder to France: continue training on hard France pseudo-labelled test pairs mixed
50/50 with labelled train pairs (no forgetting), low LR, 1 epoch.

  python ce_france.py <init_ce_dir> <out_dir> <pseudo_dir> <train_run> [n_each=200000]
"""
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, r"D:\amazon-ml\code\business_entity_resolution\src")
import cross_encoder as ce  # noqa: E402
import pipeline as pp  # noqa: E402

INIT, OUT, PSEUDO, TR = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), pp.RUNS_DIR / sys.argv[4]
N = int(sys.argv[5]) if len(sys.argv) > 5 else 200_000
LR, BS, SEED = 1e-5, 64, 7
torch, Model, Tok = ce._torch()
dev = "cuda"
t0 = time.time()


def hard(df):
    return df.filter((pl.col("rr") >= 0.02) | (pl.col("nc_tset") >= 70) | (pl.col("ad_tset") >= 80))


def balanced(df, n):
    pos, neg = df.filter(pl.col("label") == 1), df.filter(pl.col("label") == 0)
    k = min(pos.height, n // 2)
    return pl.concat([pos.sample(n=k, seed=SEED), neg.sample(n=min(neg.height, n - k), seed=SEED)])


cols = ["s1_idx", "cand_idx", "label", "rr", "nc_tset", "ad_tset"]
fr = balanced(hard(pl.concat([pl.read_parquet(f, columns=cols) for f in sorted(PSEUDO.glob("part-*.parquet"))])), N)
trn = balanced(hard(pl.concat([pl.read_parquet(f, columns=cols + ["is_val"]).filter(~pl.col("is_val")).drop("is_val")
                               for f in sorted((TR / "train_feats").glob("part-*.parquet"))])), N)
data = pl.concat([ce.texts(pp, fr.select("s1_idx", "cand_idx", "label"), "test"),
                  ce.texts(pp, trn.select("s1_idx", "cand_idx", "label"), "train")]).sample(fraction=1.0, shuffle=True, seed=SEED)
print(f"[{time.time() - t0:.0f}s] france pseudo {fr.height:,} + train {trn.height:,} pairs", flush=True)

tok = Tok.from_pretrained(INIT)
model = Model.from_pretrained(INIT).to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
steps = data.height // BS
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=steps, pct_start=0.05)
scaler = torch.amp.GradScaler()
lossf = torch.nn.BCEWithLogitsLoss()
t1, t2, y = data["t1"].to_list(), data["t2"].to_list(), data["label"].to_numpy().astype(np.float32)
model.train()
for s in range(steps):
    b = slice(s * BS, (s + 1) * BS)
    enc = tok(t1[b], t2[b], truncation=True, max_length=ce.MAXLEN, padding=True, return_tensors="pt").to(dev)
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        loss = lossf(model(**enc).logits.squeeze(-1).float(), torch.from_numpy(y[b]).to(dev))
    opt.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.step(opt)
    scaler.update()
    sched.step()
    if s % 1000 == 0:
        print(f"[{time.time() - t0:.0f}s] step {s}/{steps} loss {loss.item():.4f}", flush=True)
OUT.mkdir(parents=True, exist_ok=True)
model.save_pretrained(OUT)
tok.save_pretrained(OUT)
print(f"[{time.time() - t0:.0f}s] saved {OUT}", flush=True)
