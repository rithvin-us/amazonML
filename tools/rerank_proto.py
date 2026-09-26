"""Prototype: learned re-ranker for blocking. Wide retrieval (top-R by key score + channel extras) ->
cheap fuzzy sims -> small XGB -> keep top-K. Compares block recall vs the current rule at equal budget.

  python rerank_proto.py <run_id> <country> [n_s1]
"""
import sys
import time

import numpy as np
import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.process import cpdist

sys.path.insert(0, r"D:\amazon-ml\code\business_entity_resolution\src")
import pipeline as pp  # noqa: E402
from blocking import BLOCK_VERSION, BlockIndex  # noqa: E402

R = pp.RUNS_DIR / sys.argv[1]
country = sys.argv[2]
N = int(sys.argv[3]) if len(sys.argv) > 3 else 5000
WIDE, WIDE_N, WIDE_A = (int(x) for x in (sys.argv[4] if len(sys.argv) > 4 else "300,60,60").split(","))
LITE = len(sys.argv) > 5
t0 = time.time()

vt = pl.read_parquet(R / "val_truth.parquet").filter(pl.col("country_n") == country)
val_ids = vt["s1_id"]
s1_all = pp.scan_norm("train", "s1").filter(pl.col("country_n") == country).select(pp.NORM_COLS)
s1_va = s1_all.filter(pl.col("entity_id").is_in(val_ids.implode())).collect().sample(n=min(N, vt.height), seed=1)
s1_tr = s1_all.filter(~pl.col("entity_id").is_in(val_ids.implode())).collect().sample(n=N, seed=2)
gt = pl.read_parquet(pp.CACHE_DIR / "gt_long.parquet").drop_nulls()

ix = BlockIndex.load(pp.CACHE_DIR / "index" / f"train_{country}_{pp.NORM_VERSION}_b{BLOCK_VERSION}_df600")


def cc(e):
    return e.str.replace_all(" ", "", literal=True)


def wide(s1):
    q = ix.query(s1, top_k=WIDE, k_name=WIDE_N, k_addr=WIDE_A)
    ids = q["cand_idx"].unique()
    pool = (pp.scan_norm("train", "pool").filter(pl.col("idx").is_in(ids.implode()))
            .select("idx", "entity_id", "name_core", "name_full", "addr").collect())
    a = (q.join(s1.select(pl.col("idx").alias("s1_idx"), pl.col("entity_id").alias("s1_id"),
                          pl.col("name_core").alias("nc1"), pl.col("name_full").alias("nf1"), pl.col("addr").alias("ad1")),
                on="s1_idx")
         .join(pool.select(pl.col("idx").alias("cand_idx"), pl.col("entity_id").alias("match_id"),
                           pl.col("name_core").alias("nc2"), pl.col("name_full").alias("nf2"), pl.col("addr").alias("ad2")),
               on="cand_idx"))
    lab = gt.join(a.select("s1_id").unique(), on="s1_id").with_columns(pl.lit(1, pl.Int8).alias("label"))
    a = a.join(lab, on=["s1_id", "match_id"], how="left").with_columns(pl.col("label").fill_null(0))
    g = lambda c: a[c].fill_null("").to_list()  # noqa: E731
    sims = {
        "r_nc_tset": cpdist(g("nc1"), g("nc2"), scorer=fuzz.token_set_ratio, workers=-1, dtype=np.float32),
        "r_cc_ratio": cpdist(a["nc1"].fill_null("").str.replace_all(" ", "").to_list(),
                             a["nc2"].fill_null("").str.replace_all(" ", "").to_list(), scorer=fuzz.ratio, workers=-1, dtype=np.float32),
        "r_nf_part": cpdist(g("nf1"), g("nf2"), scorer=fuzz.partial_ratio, workers=-1, dtype=np.float32),
        "r_ad_tset": cpdist(g("ad1"), g("ad2"), scorer=fuzz.token_set_ratio, workers=-1, dtype=np.float32),
        "r_ad_ratio": cpdist(g("ad1"), g("ad2"), scorer=fuzz.ratio, workers=-1, dtype=np.float32),
    }
    a = a.with_columns([pl.Series(k, v) for k, v in sims.items()])
    # total truth per S1 (for recall incl. pairs never retrieved)
    nt = gt.join(s1.select(pl.col("entity_id").alias("s1_id")), on="s1_id").group_by("s1_id").len()
    return a, int(nt["len"].sum())


RF_ALL = ["bscore", "bscore_norm", "bname", "baddr", "bname_norm", "baddr_norm", "brank", "brank_name", "brank_addr",
      "r_nc_tset", "r_cc_ratio", "r_nf_part", "r_ad_tset", "r_ad_ratio"]
RF = [f for f in RF_ALL if not (LITE and f in ("r_nc_tset", "r_ad_ratio"))]
tr, _ = wide(s1_tr)
va, n_true_va = wide(s1_va)
print(f"[{country}] wide pairs train {tr.height:,} val {va.height:,} ({va.height / s1_va.height:.0f}/S1); "
      f"val wide recall {va['label'].sum() / n_true_va:.4f}  ({time.time() - t0:.0f}s)")

import xgboost as xgb  # noqa: E402

X = lambda d: d.select(RF).fill_null(-1).to_numpy()  # noqa: E731
bst = xgb.train({"objective": "binary:logistic", "tree_method": "hist", "device": "cuda", "max_depth": 6, "eta": 0.1,
                 "eval_metric": "aucpr"}, xgb.DMatrix(X(tr), tr["label"].to_numpy()), 300)
va = va.with_columns(pl.Series("rr", bst.predict(xgb.DMatrix(X(va)))))
imp = bst.get_score(importance_type="total_gain")
print("rerank importance:", {k: round(v) for k, v in sorted(imp.items(), key=lambda x: -x[1])[:8]})

base = va.with_columns(  # exact current rule: top-40 total + top-10 name / top-5 addr among the rest
    *[pl.when((pl.col("brank") > 40) & (pl.col(c) > 0)).then(pl.col(c)).rank("ordinal", descending=True)
      .over("s1_idx").alias(r) for c, r in (("bname", "_rn"), ("baddr", "_ra"))]
).filter((pl.col("brank") <= 40) | (pl.col("_rn") <= 10) | (pl.col("_ra") <= 5))
print(f"current-rule approx: cands/S1 {base.height / s1_va.height:.1f} recall {base['label'].sum() / n_true_va:.4f}")
va = va.with_columns(pl.col("rr").rank("ordinal", descending=True).over("s1_idx").alias("rr_rank"))
for k in (30, 40, 50, 60, 80):
    s = va.filter(pl.col("rr_rank") <= k)
    print(f"rerank top-{k}: cands/S1 {s.height / s1_va.height:.1f} recall {s['label'].sum() / n_true_va:.4f}")
bst.save_model(str(pp.ROOT / "tmp" / "scratch" / f"rerank_{country}.json"))
print(f"done {time.time() - t0:.0f}s")
