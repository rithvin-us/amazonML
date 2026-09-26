"""US-only -> India val: which feature groups hurt cross-country transfer? (drop each group, retrain)"""
import sys
import numpy as np
import polars as pl
import xgboost as xgb
sys.path.insert(0, r"D:\amazon-ml\code\business_entity_resolution\src")
import pipeline as pp
from features import FEATURES
R = pp.RUNS_DIR / sys.argv[1]; SRC, TGT = "us", "india"
cols = ["s1_idx", "cand_idx", "label", "is_val", "is_es", *FEATURES]
df = pl.concat([pl.read_parquet(f, columns=cols) for f in sorted((R / "train_feats").glob("part-*.parquet"))])
df = df.join(pp.scan_norm("train", "s1").select(pl.col("idx").alias("s1_idx"), "country_n").collect(), on="s1_idx")
vt = pl.read_parquet(R / "val_truth.parquet")
PRM = dict(objective="binary:logistic", eval_metric="logloss", tree_method="hist", device="cuda", eta=0.08, max_depth=8,
           min_child_weight=5, subsample=0.8, colsample_bytree=0.8, max_bin=256, seed=42)
class _Q:
    def log(self, m): pass
tr = df.filter(~pl.col("is_val") & ~pl.col("is_es") & (pl.col("country_n") == SRC))
es = df.filter(pl.col("is_es") & (pl.col("country_n") == SRC))
sv = df.filter(pl.col("is_val") & (pl.col("country_n") == SRC)); tv = df.filter(pl.col("is_val") & (pl.col("country_n") == TGT))
tvs = vt.filter(pl.col("country_n") == SRC).select("s1_idx", "n_true"); tvt = vt.filter(pl.col("country_n") == TGT).select("s1_idx", "n_true")
GROUPS = {
    "none": [],
    "raw_block_scores": ["bscore", "bname", "baddr", "bscore_gap_top"],
    "counts": ["c_name_cnt", "s1_name_cnt", "s1_same_name", "c_name_ratio"],
    "lengths": ["len_name_s1", "len_name_c", "len_addr_s1", "len_addr_c"],
    "ranks_ncands": ["brank", "brank_name", "brank_addr", "rr_rank", "n_cands", "nf_tset_rank", "ad_tset_rank", "hn_sim_rank", "ncc_ratio_rank", "ad_core_rank"],
    "all_four": ["bscore", "bname", "baddr", "bscore_gap_top", "c_name_cnt", "s1_name_cnt", "s1_same_name", "c_name_ratio",
                 "len_name_s1", "len_name_c", "len_addr_s1", "len_addr_c"],
}
for g, drop in GROUPS.items():
    F = [f for f in FEATURES if f not in drop]
    b = xgb.train(PRM, xgb.DMatrix(tr.select(F).to_numpy(), tr["label"].to_numpy()), 4000,
                  evals=[(xgb.DMatrix(es.select(F).to_numpy(), es["label"].to_numpy()), "es")], early_stopping_rounds=80, verbose_eval=False)
    sc = lambda d: d.select("s1_idx", "cand_idx", "label").with_columns(pl.Series("p", b.predict(xgb.DMatrix(d.select(F).to_numpy()), iteration_range=(0, b.best_iteration + 1)), dtype=pl.Float32))
    s, t = sc(sv), sc(tv)
    dec = max(pp.tune_decision(s, tvs, 0.02, _Q()).values(), key=lambda d: d["f05"])
    print(f"drop {g:18s} us val {dec['f05']:.5f} | india val {pp.eval_selection(pp.apply_decision(t, dec, 0.02), tvt)['f05']:.5f}", flush=True)
