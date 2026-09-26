"""France pseudo-labels: v7 test features + v7+CE stacked p. p>=HI -> 1, p<=LO (incl. unscored, p<0.02) -> 0.
Negatives subsampled; rows weighted W. Output: tmp/pseudo_fr/part-*.parquet in train_feats format."""
import sys, shutil
from pathlib import Path
import polars as pl
sys.path.insert(0, r"D:\amazon-ml\code\business_entity_resolution\src")
import pipeline as pp
from features import FEATURES
FEATS_RUN = pp.RUNS_DIR / "20260926-100316-v7_test"
P_RUN = pp.RUNS_DIR / (sys.argv[1] if len(sys.argv) > 1 else "20260926-104131-v7_ce")
LO, HI = 0.03, 0.97
W = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
NEG_PER_POS = 2.0
out = pp.ROOT / "tmp" / (sys.argv[3] if len(sys.argv) > 3 else "pseudo_fr"); shutil.rmtree(out, ignore_errors=True); out.mkdir(parents=True)
fr = pp.scan_norm("test", "s1").filter(pl.col("country_n") == "france").select(pl.col("idx").alias("s1_idx")).collect()
p = pl.read_parquet(P_RUN / "pred" / "scored-*.parquet").select("s1_idx", "cand_idx", "p")
n_pos = n_neg = 0
for i, f in enumerate(sorted((FEATS_RUN / "test_feats").glob("part-*.parquet"))):
    d = pl.read_parquet(f).join(fr, on="s1_idx", how="semi")
    if not d.height:
        continue
    d = d.join(p, on=["s1_idx", "cand_idx"], how="left").with_columns(pl.col("p").fill_null(0.0))
    pos = d.filter(pl.col("p") >= HI)
    neg = d.filter(pl.col("p") <= LO)
    neg = neg.sample(n=min(neg.height, int(NEG_PER_POS * pos.height)), seed=i)
    part = pl.concat([pos.with_columns(pl.lit(1, pl.Int8).alias("label")), neg.with_columns(pl.lit(0, pl.Int8).alias("label"))])
    part.select("s1_idx", "cand_idx", *FEATURES, "label", pl.lit(False).alias("is_val"), pl.lit(False).alias("is_es"),
                pl.lit(W, pl.Float32).alias("w")).write_parquet(out / f"part-{i:05d}.parquet")
    n_pos += pos.height; n_neg += neg.height
print(f"France pseudo: pos {n_pos:,} neg {n_neg:,} weight {W} -> {out}")
