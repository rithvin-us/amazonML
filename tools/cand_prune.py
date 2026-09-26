"""How small can the candidate set get? On a run's val rows (train_feats, is_val) sweep adaptive pruning rules
keep = rr_rank <= kmin OR rr >= tau, and report avg cands/S1, block recall, and how many TPs the stage-1
decision would lose (pairs selected at p >= thr that the rule drops).

  python cand_prune.py <run_id> [thr]
"""
import sys

import polars as pl

sys.path.insert(0, r"D:\amazon-ml\code\business_entity_resolution\src")
import pipeline as pp  # noqa: E402

R = pp.RUNS_DIR / sys.argv[1]
thr = float(sys.argv[2]) if len(sys.argv) > 2 else 0.675
parts = sorted((R / "train_feats").glob("part-*.parquet"))
va = pl.concat([pl.read_parquet(f, columns=["s1_idx", "cand_idx", "label", "rr", "rr_rank", "is_val"]).filter(pl.col("is_val"))
                for f in parts]).drop("is_val")
vs = pl.read_parquet(R / "val_scored.parquet", columns=["s1_idx", "cand_idx", "p"])
va = va.join(vs, on=["s1_idx", "cand_idx"], how="left")
vt = pl.read_parquet(R / "val_truth.parquet")
n_s1, n_true = vt.height, int(vt["n_true"].sum())
sel_tp = va.filter((pl.col("p") >= thr) & (pl.col("label") == 1)).height
sel_fp = va.filter((pl.col("p") >= thr) & (pl.col("label") == 0)).height
print(f"val S1 {n_s1:,}  true pairs {n_true:,}  current cands/S1 {va.height / n_s1:.1f}  recall {va['label'].sum() / n_true:.4f}"
      f"  selected TP {sel_tp:,} FP {sel_fp:,}")
print("rr quantiles of positives:", [round(va.filter(pl.col("label") == 1)["rr"].quantile(q), 5) for q in (0.001, 0.005, 0.01, 0.05)])
rows = []
for kmin in (1, 2, 3, 5):
    for tau in (0.001, 0.003, 0.01, 0.02, 0.05):
        k = va.filter((pl.col("rr_rank") <= kmin) | (pl.col("rr") >= tau))
        rows.append({"kmin": kmin, "tau": tau, "cands_per_s1": round(k.height / n_s1, 2),
                     "recall": round(k["label"].sum() / n_true, 4),
                     "lost_sel_TP": sel_tp - k.filter((pl.col("p") >= thr) & (pl.col("label") == 1)).height,
                     "lost_sel_FP": sel_fp - k.filter((pl.col("p") >= thr) & (pl.col("label") == 0)).height})
pl.Config.set_tbl_rows(30)
print(pl.DataFrame(rows))
