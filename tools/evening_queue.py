"""Unattended evening queue (runs detached). Waits for the running v8bigce variant and v9 chain, then:
  1. plain-decision alternative of v8bigce (France gets the val-tuned rule instead of the shape rule)
  2. after v9 chain: plain-decision alternative of v9_ce
  3. after v9 chain: v9 + France-adapted cross-encoder (models/ce_v7_l12_fr)
  4. full validation (--check-ids) of every candidate, FINAL_REPORT.md, best file -> output/SUBMIT_THIS
Every step is guarded: a failure is logged and the queue continues. Log: runs/evening_queue.log
"""
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import polars as pl

ROOT = Path(r"D:\amazon-ml")
PY = str(ROOT / ".venv311" / "Scripts" / "python.exe")
PIPE = str(ROOT / "code" / "business_entity_resolution" / "src" / "pipeline.py")
RUNS = ROOT / "runs"
LOG = RUNS / "evening_queue.log"
BLOCK = ["--max-df", "600", "--rr-tau", "0.002", "--rr-min", "3"]
DEADLINE = time.mktime(time.strptime(time.strftime("%Y-%m-%d") + " 18:45", "%Y-%m-%d %H:%M"))


def log(msg):
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")


def newest(name):
    ds = sorted(d for d in RUNS.glob(f"*-{name}") if d.is_dir())
    return ds[-1] if ds else None


def wait_for(logfile, marker, fail="STOP", poll=30):
    while True:
        txt = (RUNS / logfile).read_text(encoding="utf-8") if (RUNS / logfile).exists() else ""
        if marker in txt:
            return True
        if fail in txt or time.time() > DEADLINE:
            return False
        time.sleep(poll)


def run(tag, args):
    t = time.time()
    log(f"start {tag}: {' '.join(args)}")
    with open(RUNS / f"q_{tag}.out", "w", encoding="utf-8") as out:
        rc = subprocess.run([PY, PIPE, *args], cwd=ROOT, stdout=out, stderr=subprocess.STDOUT).returncode
    log(f"done {tag} rc={rc} in {time.time() - t:.0f}s")
    return rc == 0


def metrics(d):
    p = d / "metrics.json" if d else None
    return json.loads(p.read_text()) if p and p.exists() else {}


# ---- 1. v8bigce plain alternative
if wait_for("variant_v8bigce.log", "variant v8bigce finished"):
    ce = newest("v8bigce_ce")
    run("v8bigce_plain", ["decide", "--run", ce.name, "--name", "v8bigce_plain", *BLOCK])
else:
    log("v8bigce did not finish")

# ---- 2/3. v9
v9_ok = wait_for("chain_v9.log", "chain v9 finished", poll=60)
if v9_ok:
    v9, v9ce = newest("v9"), newest("v9_ce")
    run("v9_plain", ["decide", "--run", v9ce.name, "--name", "v9_plain", *BLOCK])
    if time.time() < DEADLINE - 45 * 60:
        run("v9_cefr", ["ce-apply", "--run", v9.name, "--ce-dir", str(ROOT / "models" / "ce_v7_l12_fr"),
                        "--name", "v9_cefr", *BLOCK])
    else:
        log("skip v9_cefr: not enough time before deadline")
else:
    log("v9 chain did not finish -> fallback candidates only")

# ---- 4. validate + report + pick
rows = []
for name, note in (("v9_ce", "v9 (v5 norm, 1.6M S1, L12 CE on 1M pairs) + France shape rule  [MAIN]"),
                   ("v9_plain", "v9, val-tuned rule for France too"),
                   ("v9_cefr", "v9 + France-adapted CE + France shape rule"),
                   ("v8bigce_ce", "v8big (1.3M S1) + L12 CE + France shape rule  [FALLBACK]"),
                   ("v8bigce_plain", "v8big + L12 CE, val-tuned rule for France too")):
    d = newest(name)
    f = d / "output" / "matching_results.tsv" if d else None
    if not f or not f.exists():
        rows.append({"file": name, "note": note, "status": "missing"})
        continue
    r = subprocess.run([sys.executable, str(ROOT / "student_resource" / "utils" / "validate_submission.py"),
                        "--matching", str(f), "--candidate", str(d / "output" / "candidate_pairs.tsv"),
                        "--test-dir", str(ROOT / "student_resource" / "dataset" / "test"), "--check-ids"],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    ok = r.returncode == 0
    m = pl.read_csv(f, separator="\t", quote_char=None, infer_schema=False).with_columns(
        pl.col("matched_entity_ids").fill_null("").str.split(",").list.eval(pl.element().filter(pl.element() != "")).list.len().alias("k"))
    s1 = pl.read_parquet(ROOT / "cache" / "raw_test_s1.parquet", columns=["entity_id", "country"])
    by = (m.join(s1.rename({"entity_id": "source1_entity_id"}), on="source1_entity_id").group_by("country")
          .agg(pl.col("k").mean().round(3).alias("k"), (pl.col("k") == 0).mean().round(4).alias("empty")).sort("country"))
    shape = "; ".join(f"{c}: {k}/S1 empty {e}" for c, k, e in by.iter_rows())
    mm = metrics(d)
    rows.append({"file": name, "note": note, "status": "PASS" if ok else "FAIL", "val_f05": mm.get("val_f05"),
                 "shape": shape, "path": str(f), "dir": d})
    log(f"validated {name}: {'PASS' if ok else 'FAIL'} val {mm.get('val_f05')} {shape}")

good = [r for r in rows if r.get("status") == "PASS"]
pick = next((r for r in good if r["file"] == "v9_ce"), None) or next((r for r in good if r["file"] == "v8bigce_ce"), None)
if pick:
    dst = ROOT / "output" / "SUBMIT_THIS"
    dst.mkdir(parents=True, exist_ok=True)
    shutil.copy(pick["dir"] / "output" / "matching_results.tsv", dst / "matching_results.tsv")
    shutil.copy(pick["dir"] / "output" / "candidate_pairs.tsv", dst / "candidate_pairs.tsv")
    (dst / "WHAT_IS_THIS.txt").write_text(f"{pick['file']}: {pick['note']}\nval F0.5 {pick.get('val_f05')}\n"
                                          f"{pick.get('shape')}\nvalidator --check-ids PASS\n{time.ctime()}\n", encoding="utf-8")
    for r in good:  # keep every candidate under a stable name
        shutil.copy(r["dir"] / "output" / "matching_results.tsv", ROOT / "output" / "submissions" / f"{r['file']}_matching_results.tsv")
lines = ["# Evening run report", f"generated {time.ctime()}", "",
         f"**Recommended file:** `output/SUBMIT_THIS/matching_results.tsv` = **{pick['file'] if pick else 'NONE'}**", "",
         "| file | what | validator | val F0.5 (US/India) | test shape by country |", "|---|---|---|---|---|"]
for r in rows:
    lines.append(f"| {r['file']} | {r['note']} | {r['status']} | {r.get('val_f05', '')} | {r.get('shape', '')} |")
lines += ["", "Reference: v7+CE LB 0.973901 (val 0.98457); v7fr+L12 LB 0.973905 (val 0.98499).",
          "Stable copies of every passing file: output/submissions/<file>_matching_results.tsv", "",
          "Queue log: runs/evening_queue.log"]
(RUNS / "FINAL_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
log(f"report written, picked {pick['file'] if pick else None}")
