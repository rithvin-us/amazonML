"""Label-free decision for countries without validation labels: pick the threshold at which the country's predicted
match-count shape (mean matches per S1, share of empty S1) matches the shape the SAME model produces on the
labelled countries (whose thresholds are val-tuned). The generator's match-count distribution is identical across
train countries (US == India), so an unseen country should look the same.

  python prior_match.py <train_run (val-tuned decision)> <pred_run with pred/scored> <out_name>
"""
import json
import shutil
import subprocess
import sys
import time

import polars as pl

sys.path.insert(0, r"D:\amazon-ml\code\business_entity_resolution\src")
import pipeline as pp  # noqa: E402

TR, PR, NAME = pp.RUNS_DIR / sys.argv[1], pp.RUNS_DIR / sys.argv[2], sys.argv[3]
m = json.loads((PR / "metrics.json").read_text()) if (PR / "metrics.json").exists() else {}
mt = json.loads((TR / "metrics.json").read_text())
src = m if "decision_mode" in m else mt
dec = {"mode": src["decision_mode"], "param": src["decision_param"], "excl": src["decision_excl"]}
labelled = set(pl.read_parquet(TR / "val_truth.parquet")["country_n"].drop_nulls().unique().to_list())
sc = pl.read_parquet(PR / "pred" / "scored-*.parquet")
s1 = pp.scan_norm("test", "s1").select(pl.col("idx").alias("s1_idx"), "country_n").collect()
sc = sc.join(s1, on="s1_idx")
n_s1 = dict(s1.group_by("country_n").len().iter_rows())


def shape(sel: pl.DataFrame, c: str) -> tuple[float, float]:
    k = sel.group_by("s1_idx").len()
    return k["len"].sum() / n_s1[c], 1 - k.height / n_s1[c]


sels, target = [], []
for c in sorted(n_s1):
    if c in labelled:
        sel = pp.apply_decision(sc.filter(pl.col("country_n") == c), dec, 0.02)
        mk, e = shape(sel, c)
        target.append((mk, e, n_s1[c]))
        print(f"{c}: val-tuned {dec}  mean_k {mk:.3f} empty {e:.4f}")
        sels.append(sel)
tk = sum(a * w for a, _, w in target) / sum(w for *_, w in target)
te = sum(b * w for _, b, w in target) / sum(w for *_, w in target)
print(f"target shape: mean_k {tk:.3f} empty {te:.4f}")
for c in sorted(n_s1):
    if c in labelled:
        continue
    part = sc.filter(pl.col("country_n") == c)
    mk0, e0 = shape(pp.apply_decision(part, dec, 0.02), c)
    if len(sys.argv) > 4:  # fixed threshold for unlabelled countries (e.g. chosen from the shape curve)
        d = {"mode": "threshold", "param": float(sys.argv[4]), "excl": dec["excl"]}
        mk, e = shape(pp.apply_decision(part, d, 0.02), c)
        print(f"{c}: fixed {d} mean_k {mk:.3f} empty {e:.4f}")
        sels.append(pp.apply_decision(part, d, 0.02))
        continue
    best = None
    for t in [x / 1000 for x in range(500, 991, 5)]:
        for t1 in (None, 0.5, 0.6, 0.7, 0.8):
            d = ({"mode": "threshold", "param": t, "excl": dec["excl"]} if t1 is None or t1 >= t
                 else {"mode": "thr_top1", "param": [t, t1], "excl": dec["excl"]})
            mk, e = shape(pp.apply_decision(part, d, 0.02), c)
            err = abs(mk - tk) / tk + abs(e - te) / max(te, 1e-3)
            if best is None or err < best[0]:
                best = (err, d, mk, e)
    print(f"{c}: default {dec} mean_k {mk0:.3f} empty {e0:.4f}  ->  prior-matched {best[1]} mean_k {best[2]:.3f} empty {best[3]:.4f}")
    sels.append(pp.apply_decision(part, best[1], 0.02))
sel = pl.concat(sels)

out = pp.RUNS_DIR / f"{time.strftime('%Y%m%d-%H%M%S')}-{NAME}"
(out / "output").mkdir(parents=True)
s1_map = (pl.read_parquet(pp.CACHE_DIR / "raw_test_s1.parquet", columns=["entity_id"]).with_row_index("s1_idx")
          .select(pl.col("s1_idx").cast(pl.Int64), pl.col("entity_id").alias("source1_entity_id")))
pp._write_ids(s1_map, pp._join_ids(sel), "matched_entity_ids", out / "output" / "matching_results.tsv")
shutil.copy(PR / "output" / "candidate_pairs.tsv", out / "output" / "candidate_pairs.tsv")
r = subprocess.run([sys.executable, str(pp.VALIDATOR), "--matching", str(out / "output" / "matching_results.tsv"),
                    "--candidate", str(out / "output" / "candidate_pairs.tsv"), "--test-dir",
                    str(pp.ROOT / "student_resource" / "dataset" / "test")], capture_output=True, text=True,
                   encoding="utf-8", errors="replace")
print((r.stdout.strip().splitlines() or [r.stderr[-300:]])[-1])
print(out)
