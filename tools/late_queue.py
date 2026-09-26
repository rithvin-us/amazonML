"""Late queue: after the evening queue's GPU work, train a depth-10 v9 variant from saved features, ensemble it
with v9 (same candidates), apply the v9 L12 CE (+ France shape rule), and replace SUBMIT_THIS only if val F0.5
beats v9_ce by >= 0.0003. Appends to runs/FINAL_REPORT.md. Log: runs/late_queue.log
"""
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(r"D:\amazon-ml")
PY = str(ROOT / ".venv311" / "Scripts" / "python.exe")
PIPE = str(ROOT / "code" / "business_entity_resolution" / "src" / "pipeline.py")
RUNS = ROOT / "runs"
LOG = RUNS / "late_queue.log"
BLOCK = ["--max-df", "600", "--rr-tau", "0.002", "--rr-min", "3"]
STOP_AT = time.mktime(time.strptime(time.strftime("%Y-%m-%d") + " 18:25", "%Y-%m-%d %H:%M"))


def log(m):
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(f"[{time.strftime('%H:%M:%S')}] {m}\n")


def newest(name):
    ds = sorted(d for d in RUNS.glob(f"*-{name}") if d.is_dir())
    return ds[-1] if ds else None


def val(d):
    p = d / "metrics.json" if d else None
    return json.loads(p.read_text()).get("val_f05") if p and p.exists() else None


def run(tag, args, py_script=None):
    t = time.time()
    log(f"start {tag}")
    cmd = [PY, py_script, *args] if py_script else [PY, PIPE, *args]
    with open(RUNS / f"late_{tag}.out", "w", encoding="utf-8") as out:
        rc = subprocess.run(cmd, cwd=ROOT, stdout=out, stderr=subprocess.STDOUT).returncode
    log(f"done {tag} rc={rc} in {time.time() - t:.0f}s")
    return rc == 0


# wait until the evening queue has finished its GPU work (report written)
while "report written" not in ((RUNS / "evening_queue.log").read_text(encoding="utf-8") if (RUNS / "evening_queue.log").exists() else ""):
    if time.time() > STOP_AT:
        log("evening queue not done in time; abort")
        sys.exit(0)
    time.sleep(30)
v9, v9test, ce_dir = newest("v9"), newest("v9_test"), ROOT / "models" / f"ce_{newest('v9').name}"
lines = ["", "## Late queue (depth-10 variant + ensemble)"]
ok = run("v9d10", ["retrain", "--run", v9.name, "--name", "v9d10", "--depth", "10", "--rounds", "8000", *BLOCK])
d10 = newest("v9d10")
lines.append(f"- v9d10 stage-1 val F0.5: {val(d10)} (v9 stage-1: {val(v9)})")
if ok and time.time() < STOP_AT - 40 * 60:
    ok = run("v9d10_test", ["rescore", "--run", d10.name, "--feats-run", v9test.name, "--name", "v9d10_test", *BLOCK])
if ok and time.time() < STOP_AT - 30 * 60:
    ok = run("ens", ["ens_v9", v9.name, v9.name, d10.name, newest("v9d10_test").name], py_script=str(ROOT / "tmp" / "scratch" / "ens.py"))
if ok and time.time() < STOP_AT - 22 * 60:
    ens = newest("ens_v9")
    ok = run("ens_ce", ["ce-apply", "--run", ens.name, "--ce-dir", str(ce_dir), "--name", "ens_v9_ce", *BLOCK])
    ec, base = val(newest("ens_v9_ce")), val(newest("v9_ce"))
    lines.append(f"- ensemble(v9, v9d10) + L12 CE + France rule val F0.5: {ec} (v9_ce: {base})")
    if ok and ec and base and ec >= base + 0.0003:
        src = newest("ens_v9_ce") / "output"
        dst = ROOT / "output" / "SUBMIT_THIS"
        shutil.copy(src / "matching_results.tsv", dst / "matching_results.tsv")
        shutil.copy(src / "candidate_pairs.tsv", dst / "candidate_pairs.tsv")
        (dst / "WHAT_IS_THIS.txt").write_text(f"ens_v9_ce: ensemble(v9, v9d10) + L12 CE + France shape rule\nval F0.5 {ec} "
                                              f"(v9_ce {base})\nvalidator PASS (pipeline)\n{time.ctime()}\n", encoding="utf-8")
        lines.append("- **SUBMIT_THIS replaced by the ensemble** (val gain >= 0.0003)")
    else:
        lines.append("- SUBMIT_THIS unchanged (ensemble gain < 0.0003 or failed)")
    if ok:
        shutil.copy(newest("ens_v9_ce") / "output" / "matching_results.tsv", ROOT / "output" / "submissions" / "ens_v9_ce_matching_results.tsv")
else:
    lines.append("- stopped early (failure or not enough time)")
with open(RUNS / "FINAL_REPORT.md", "a", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
log("late queue finished")
