"""Full v4 chain, each step in its own process so RAM is released between steps.

train (450k S1) -> predict test (fresh process) -> move pred/ into train run -> stage2.
Log: runs/chain_v4.log (one line per step; watched by Monitor).
"""
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(r"D:\amazon-ml")
PY = str(ROOT / ".venv311" / "Scripts" / "python.exe")
PIPE = str(ROOT / "code" / "business_entity_resolution" / "src" / "pipeline.py")
RUNS = ROOT / "runs"
LOG = RUNS / "chain_v4.log"
MAX_DF = "600"
TRAIN_S1 = sys.argv[1] if len(sys.argv) > 1 else "450000"


def log(msg: str) -> None:
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(f"[{time.strftime('%H:%M:%S')}] CHAIN {msg}\n")


def step(name: str, args: list[str]) -> str:
    before = (RUNS / "LATEST").read_text().strip()
    log(f"start {name}: {' '.join(args)}")
    t = time.time()
    with open(RUNS / f"{name}.out", "w", encoding="utf-8") as out:
        rc = subprocess.run([PY, PIPE, *args], cwd=ROOT, stdout=out, stderr=subprocess.STDOUT).returncode
    rid = (RUNS / "LATEST").read_text().strip()
    if rid == before:
        rid = ""
    log(f"done {name} rc={rc} run={rid} in {time.time() - t:.0f}s")
    if rc != 0:
        log(f"STOP chain failed at {name}")
        sys.exit(rc)
    return rid


def main() -> None:
    log(f"chain v4 begin (train_max_s1={TRAIN_S1}, max_df={MAX_DF})")
    tr = step("v4_train", ["train", "--name", "v4", "--train-max-s1", TRAIN_S1, "--max-df", MAX_DF])
    te = step("v4_test", ["predict", "--run", tr, "--name", "v4_test", "--max-df", MAX_DF])
    # stage2 reads val_scored/val_truth/metrics AND pred/ from one run dir
    src, dst = RUNS / te / "pred", RUNS / tr / "pred"
    shutil.rmtree(dst, ignore_errors=True)
    shutil.move(str(src), str(dst))
    log(f"moved {src} -> {dst}")
    step("v4_stage2", ["stage2", "--run", tr, "--name", "v4_s2"])
    log("chain v4 finished")


if __name__ == "__main__":
    main()
