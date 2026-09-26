"""Cross-encoder re-scoring of the uncertain band (stage 3).

A MiniLM cross-encoder (cross-encoder/ms-marco-MiniLM-L6-v2, Apache-2.0, 22M params) is fine-tuned on
("name | address" of S1, "name | address" of candidate) pairs from a finished run's training S1, then
re-scores only pairs whose stage-1 p lies in [0.02, 0.995) (~1 pair per S1). A small monotone XGBoost
stacks [logit p, ce] on the validation band; the decision is re-tuned with competitor S1 and applied to test.

  python pipeline.py ce-train --run <train_run>                     # -> models/ce_<train_run>
  python pipeline.py ce-apply --run <train_run> --feats-run <run holding pred/>   # -> new run + output/
"""
from __future__ import annotations

import shutil
import time
from pathlib import Path

import numpy as np
import polars as pl

BASE = "cross-encoder/ms-marco-MiniLM-L6-v2"
BAND = (0.02, 0.995)
MAXLEN, BS, LR = 96, 64, 3e-5
STACK = {"objective": "binary:logistic", "max_depth": 3, "eta": 0.1, "monotone_constraints": "(1,1)",
         "tree_method": "hist"}


def _torch(base: str = BASE):
    import os
    hub = Path(os.environ.get("HF_HOME", "")) / "hub" / ("models--" + base.replace("/", "--"))
    if hub.exists():  # base weights already downloaded once -> never touch the network again
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    return torch, AutoModelForSequenceClassification, AutoTokenizer


def texts(pp, pairs: pl.DataFrame, split: str) -> pl.DataFrame:
    """Attach 't1' / 't2' = 'name_full | addr' of the S1 and of the candidate (normalised caches)."""
    s1 = (pp.scan_norm(split, "s1").filter(pl.col("idx").is_in(pairs["s1_idx"].unique().implode()))
          .select(pl.col("idx").alias("s1_idx"), pl.concat_str(["name_full", "addr"], separator=" | ").alias("t1")).collect())
    po = (pp.scan_norm(split, "pool").filter(pl.col("idx").is_in(pairs["cand_idx"].unique().implode()))
          .select(pl.col("idx").alias("cand_idx"), pl.concat_str(["name_full", "addr"], separator=" | ").alias("t2")).collect())
    return pairs.join(s1, on="s1_idx", how="left").join(po, on="cand_idx", how="left").with_columns(
        pl.col("t1").fill_null(""), pl.col("t2").fill_null(""))


def train_ce(pp, run, src_run: Path, out: Path, n_pairs: int = 600_000, epochs: int = 2, seed: int = 42,
             base: str = BASE) -> None:
    """Fine-tune on non-val training pairs of src_run (hard region: re-ranker score >= 0.02 or strong name/address
    overlap), balanced 50/50. The re-ranker never saw these S1, so rr is an honest selector here."""
    torch, Model, Tok = _torch(base)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    parts = sorted((src_run / "train_feats").glob("part-*.parquet"))
    tr = pl.concat([pl.read_parquet(f, columns=["s1_idx", "cand_idx", "label", "rr", "nc_tset", "ad_tset", "is_val"])
                    .filter(~pl.col("is_val")) for f in parts])
    hard = tr.filter((pl.col("rr") >= 0.02) | (pl.col("nc_tset") >= 70) | (pl.col("ad_tset") >= 80))
    pos, neg = hard.filter(pl.col("label") == 1), hard.filter(pl.col("label") == 0)
    n_pos = min(pos.height, n_pairs // 2)
    tr = (pl.concat([pos.sample(n=n_pos, seed=seed), neg.sample(n=min(neg.height, n_pairs - n_pos), seed=seed)])
          .sample(fraction=1.0, shuffle=True, seed=seed + 1))
    run.log(f"ce-train: base {base}, {tr.height:,} pairs (pos {n_pos:,}) from {hard.height:,} hard pairs, device {dev}")
    tr = texts(pp, tr.select("s1_idx", "cand_idx", "label"), "train")
    tok = Tok.from_pretrained(base)
    model = Model.from_pretrained(base, num_labels=1).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    steps = epochs * (tr.height // BS)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=steps, pct_start=0.06)
    scaler = torch.amp.GradScaler(enabled=dev == "cuda")
    lossf = torch.nn.BCEWithLogitsLoss()
    t1, t2, y = tr["t1"].to_list(), tr["t2"].to_list(), tr["label"].to_numpy().astype(np.float32)
    torch.manual_seed(seed)
    model.train()
    step = 0
    for ep in range(epochs):
        order = np.random.default_rng(seed + ep).permutation(len(y))
        for i in range(0, len(y) - BS + 1, BS):
            b = order[i:i + BS]
            enc = tok([t1[j] for j in b], [t2[j] for j in b], truncation=True, max_length=MAXLEN, padding=True,
                      return_tensors="pt").to(dev)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=dev == "cuda"):
                loss = lossf(model(**enc).logits.squeeze(-1).float(), torch.from_numpy(y[b]).to(dev))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
            if step % 250 == 0:
                run.progress(step / steps, f"epoch {ep} step {step}/{steps} loss {loss.item():.4f} [{dev}]")
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    tok.save_pretrained(out)
    run.log(f"ce-train: saved {out}")


class _Scorer:
    def __init__(self, ce_dir: Path):
        self.torch, Model, Tok = _torch()
        self.dev = "cuda" if self.torch.cuda.is_available() else "cpu"
        self.tok = Tok.from_pretrained(ce_dir)
        self.model = Model.from_pretrained(ce_dir).to(self.dev).eval()

    def __call__(self, df: pl.DataFrame, bs: int = 512) -> np.ndarray:
        torch, out = self.torch, []
        a, b = df["t1"].to_list(), df["t2"].to_list()
        with torch.no_grad():
            for i in range(0, len(a), bs):
                enc = self.tok(a[i:i + bs], b[i:i + bs], truncation=True, max_length=MAXLEN, padding=True,
                               return_tensors="pt").to(self.dev)
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.dev == "cuda"):
                    out.append(self.model(**enc).logits.squeeze(-1).float().cpu().numpy())
        return np.concatenate(out) if out else np.zeros(0, np.float32)


def _band(df: pl.DataFrame) -> pl.DataFrame:
    return df.filter((pl.col("p") >= BAND[0]) & (pl.col("p") < BAND[1]))


def _x(d: pl.DataFrame) -> np.ndarray:
    return d.select((pl.col("p").log() - (1 - pl.col("p")).log()).alias("lp"), "ce").to_numpy()


def _restack(full: pl.DataFrame, band: pl.DataFrame, p2: np.ndarray) -> pl.DataFrame:
    b = band.select("s1_idx", "cand_idx").with_columns(pl.Series("p2", p2, dtype=pl.Float32))
    return (full.join(b, on=["s1_idx", "cand_idx"], how="left", maintain_order="left")
            .with_columns(pl.coalesce("p2", "p").alias("p")).drop("p2"))


def apply_ce(pp, cfg, run, src_run: Path, pred_run: Path, ce_dir: Path) -> None:
    """Band re-scoring + stacking + competitor-aware re-tuning on val, then the same on src/pred_run test scores."""
    import xgboost as xgb
    from sklearn.model_selection import GroupKFold
    score = _Scorer(ce_dir)

    def with_ce(df: pl.DataFrame, split: str, what: str) -> pl.DataFrame:
        b = texts(pp, _band(df).select("s1_idx", "cand_idx", "p", *[c for c in ("label",) if c in df.columns]), split)
        t = time.time()
        b = b.with_columns(pl.Series("ce", score(b), dtype=pl.Float32)).drop("t1", "t2")
        run.log(f"ce-apply: {what} band {b.height:,} pairs scored in {time.time() - t:.0f}s")
        return b

    run.start_stage("ce_val")
    va = pl.read_parquet(src_run / "val_scored.parquet")
    vt = pl.read_parquet(src_run / "val_truth.parquet")
    vb = with_ce(va, "train", "val").join(vt.select("s1_idx", "block"), on="s1_idx", how="left")
    yb, xb = vb["label"].to_numpy(), _x(vb)
    oof = np.zeros(vb.height, np.float32)  # out-of-fold by block: honest val estimate of the stack
    for trn, tst in GroupKFold(n_splits=5).split(xb, yb, vb["block"].fill_null("?").to_numpy()):
        oof[tst] = xgb.train(STACK, xgb.DMatrix(xb[trn], yb[trn]), 200).predict(xgb.DMatrix(xb[tst]))
    stacker = xgb.train(STACK, xgb.DMatrix(xb, yb), 200)
    stacker.save_model(str(run.dir / "ce_stacker.json"))
    va2 = _restack(va, vb, oof)
    lab0 = lambda d: d.select("s1_idx", "cand_idx", pl.lit(0, pl.Int8).alias("label"), "p")  # noqa: E731
    base_u, new_u = va, va2
    if (src_run / "comp_scored.parquet").exists():
        comp = pl.read_parquet(src_run / "comp_scored.parquet")
        cb = with_ce(comp, "train", "competitors")
        comp2 = _restack(comp, cb, stacker.predict(xgb.DMatrix(_x(cb))))
        base_u, new_u = pl.concat([va, lab0(comp)]), pl.concat([va2, lab0(comp2)])
    tvc = vt.select("s1_idx", "n_true")
    b0 = max(pp.tune_decision(base_u, tvc, cfg.p_floor, run).values(), key=lambda d: d["f05"])
    res = pp.tune_decision(new_u, tvc, cfg.p_floor, run)
    b1 = max(res.values(), key=lambda d: d["f05"])
    dec = {"mode": b1["mode"], "param": b1["param"], "excl": b1["excl"]}
    run.set_metrics(val_f05=round(b1["f05"], 5), val_precision=round(b1["precision"], 4),
                    val_recall=round(b1["recall"], 4), val_f05_stage1=round(b0["f05"], 5),
                    decision_mode=dec["mode"], decision_param=dec["param"], decision_excl=dec["excl"],
                    stage1_run=src_run.name, ce_model=str(ce_dir))
    run.log(f"metric ce val_f05 {b0['f05']:.5f} -> {b1['f05']:.5f} ({dec['mode']}:{dec['param']} excl={dec['excl']})")
    run.end_stage()

    run.start_stage("ce_test")
    spill_old = pred_run / "pred"
    sc = pl.read_parquet(spill_old / "scored-*.parquet")
    tb = with_ce(sc, "test", "test")
    spill = run.dir / "pred"
    shutil.rmtree(spill, ignore_errors=True)
    spill.mkdir()
    for f in spill_old.glob("cand-*.parquet"):
        shutil.copy(f, spill / f.name)
    _restack(sc, tb, stacker.predict(xgb.DMatrix(_x(tb)))).write_parquet(spill / "scored-00000.parquet")
    run.set_metrics(test_band_pairs=tb.height)
    run.end_stage()
    pp.decide_stage(cfg, run, spill, dec, labelled=pp.labelled_countries(src_run))
