"""Generic chain, each step in its own process so RAM is released between steps.

  python chain.py <name> <train_max_s1> [extra pipeline args...] [--no-stage2] [--with-ce]
train -> predict test (fresh process; with --with-ce, ce-train runs on the GPU meanwhile) -> ce-apply
-> move pred/ into train run -> stage2 (unless --no-stage2).
Log: runs/chain_<name>.log (one line per step; watched by Monitor).
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
NAME, TRAIN_S1, EXTRA = sys.argv[1], sys.argv[2], sys.argv[3:]
SKIP_S2 = "--no-stage2" in EXTRA
WITH_CE = "--with-ce" in EXTRA
EXTRA = [a for a in EXTRA if a not in ("--no-stage2", "--with-ce")]
LOG = RUNS / f"chain_{NAME}.log"
MAX_DF = "600"


def log(msg: str) -> None:
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(f"[{time.strftime('%H:%M:%S')}] CHAIN {msg}\n")


def launch(name: str, args: list[str]):
    log(f"start {name}: {' '.join(args)}")
    out = open(RUNS / f"{name}.out", "w", encoding="utf-8")
    return subprocess.Popen([PY, PIPE, *args], cwd=ROOT, stdout=out, stderr=subprocess.STDOUT), time.time()


def finish(name: str, args: list[str], proc, t: float) -> str:
    rc = proc.wait()
    run_name = args[args.index("--name") + 1]
    # runs/LATEST is shared with any other run started meanwhile: find this step's run dir by its name
    mine = [d for d in RUNS.glob(f"*-{run_name}") if d.is_dir() and d.stat().st_ctime >= t - 5]
    rid = max(mine, key=lambda d: d.stat().st_ctime).name if mine else ""
    log(f"done {name} rc={rc} run={rid} in {time.time() - t:.0f}s")
    if rc != 0:
        log(f"STOP chain failed at {name}")
        sys.exit(rc)
    return rid


def step(name: str, args: list[str]) -> str:
    return finish(name, args, *launch(name, args))


def main() -> None:
    log(f"chain {NAME} begin (train_max_s1={TRAIN_S1}, max_df={MAX_DF}, extra={EXTRA}, ce={WITH_CE})")
    tr = step(f"{NAME}_train", ["train", "--name", NAME, "--train-max-s1", TRAIN_S1, "--max-df", MAX_DF, *EXTRA])
    te_args = ["predict", "--run", tr, "--name", f"{NAME}_test", "--max-df", MAX_DF, *EXTRA]
    te_proc = launch(f"{NAME}_test", te_args)
    if WITH_CE:  # GPU-bound, overlaps with the CPU-bound test featurisation
        step(f"{NAME}_ce_train", ["ce-train", "--run", tr, "--name", f"{NAME}_cetrain", *EXTRA])
    te = finish(f"{NAME}_test", te_args, *te_proc)
    # stage2 / ce-apply read val_scored/val_truth/metrics AND pred/ from one run dir
    src, dst = RUNS / te / "pred", RUNS / tr / "pred"
    shutil.rmtree(dst, ignore_errors=True)
    shutil.copytree(src, dst)
    log(f"copied {src} -> {dst}")
    if WITH_CE:
        step(f"{NAME}_ce_apply", ["ce-apply", "--run", tr, "--name", f"{NAME}_ce", *EXTRA])
    if not SKIP_S2:
        step(f"{NAME}_stage2", ["stage2", "--run", tr, "--name", f"{NAME}_s2"])
    log(f"chain {NAME} finished")


if __name__ == "__main__":
    main()
