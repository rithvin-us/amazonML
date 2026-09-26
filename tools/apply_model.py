"""Trained model -> rescore saved test feats -> ce-apply (existing CE). No retraining.

  python apply_model.py <name> <train_run> <test_feats_run> <ce_dir> [extra pipeline args]
Log: runs/variant_<name>.log
"""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(r"D:\amazon-ml")
PY = str(ROOT / ".venv311" / "Scripts" / "python.exe")
PIPE = str(ROOT / "code" / "business_entity_resolution" / "src" / "pipeline.py")
RUNS = ROOT / "runs"
NAME, TR, FEATS, CE, EXTRA = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5:]
LOG = RUNS / f"variant_{NAME}.log"


def log(msg):
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(f"[{time.strftime('%H:%M:%S')}] VARIANT {msg}\n")


def step(tag, args):
    run_name = args[args.index("--name") + 1]
    t = time.time()
    log(f"start {tag}: {' '.join(args)}")
    with open(RUNS / f"{NAME}_{tag}.out", "w", encoding="utf-8") as out:
        rc = subprocess.run([PY, PIPE, *args], cwd=ROOT, stdout=out, stderr=subprocess.STDOUT).returncode
    mine = [d for d in RUNS.glob(f"*-{run_name}") if d.is_dir() and d.stat().st_ctime >= t - 5]
    rid = max(mine, key=lambda d: d.stat().st_ctime).name if mine else ""
    log(f"done {tag} rc={rc} run={rid} in {time.time() - t:.0f}s")
    if rc != 0:
        log(f"STOP failed at {tag}")
        sys.exit(rc)
    return rid


te = step("rescore", ["rescore", "--run", TR, "--feats-run", FEATS, "--name", f"{NAME}_test", *EXTRA])
step("ce_apply", ["ce-apply", "--run", TR, "--feats-run", te, "--ce-dir", CE, "--name", f"{NAME}_ce", *EXTRA])
log(f"variant {NAME} finished")
