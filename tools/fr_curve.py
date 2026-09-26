import sys, polars as pl
sys.path.insert(0, r"D:\amazon-ml\code\business_entity_resolution\src")
import pipeline as pp
PR = pp.RUNS_DIR / sys.argv[1]
sc = pl.read_parquet(PR / "pred" / "scored-*.parquet").join(
    pp.scan_norm("test", "s1").select(pl.col("idx").alias("s1_idx"), "country_n").collect(), on="s1_idx").filter(pl.col("country_n") == "france")
n = 259452
for t in (0.575, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.965):
    sel = pp.apply_decision(sc, {"mode": "thr_top1", "param": [t, 0.5], "excl": True}, 0.02) if t < 0.6 else pp.apply_decision(sc, {"mode": "threshold", "param": t, "excl": True}, 0.02)
    k = sel.group_by("s1_idx").len()
    print(f"t={t:.3f} mean_k {k['len'].sum() / n:.3f} empty {1 - k.height / n:.4f}")
# how confident are the matches of S1s that have exactly one predicted match (singleton suspects)?
sel = pp.apply_decision(sc, {"mode": "thr_top1", "param": [0.575, 0.5], "excl": True}, 0.02)
one = sel.group_by("s1_idx").agg(pl.len(), pl.col("p").max()).filter(pl.col("len") == 1)
print("S1 with exactly 1 match:", one.height, " p quantiles", [round(one["p"].quantile(q), 3) for q in (0.1, 0.25, 0.5, 0.75)])
