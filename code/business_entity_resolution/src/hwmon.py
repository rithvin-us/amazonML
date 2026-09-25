"""Background hardware sampler: CPU/RAM/GPU + GPU throttle reasons -> CSV.

Usage:
    with HwMonitor(run_dir / "hw.csv"):
        ...work...
Standalone:  python hwmon.py [seconds]   (prints samples, writes runs/hw_live.csv)
"""
from __future__ import annotations

import csv
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import psutil

GPU_FIELDS = ["utilization.gpu", "memory.used", "memory.total", "temperature.gpu", "power.draw",
              "clocks.sm", "clocks.max.sm", "clocks_throttle_reasons.active"]
# Decoded bits of clocks_throttle_reasons.active
THROTTLE_BITS = {0x1: "idle", 0x2: "app_clock", 0x4: "sw_power_cap", 0x8: "hw_slowdown",
                 0x20: "sw_thermal", 0x40: "hw_thermal", 0x80: "hw_power_brake"}
COLS = ["time", "cpu_pct", "cpu_freq_mhz", "ram_used_gb", "ram_pct", "proc_rss_gb",
        "gpu_util", "gpu_mem_used_mb", "gpu_mem_total_mb", "gpu_temp_c", "gpu_power_w",
        "gpu_sm_clock", "gpu_sm_clock_max", "gpu_throttle"]


def gpu_sample() -> list:
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={','.join(GPU_FIELDS)}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5).stdout.strip().splitlines()[0]
        v = [x.strip() for x in out.split(",")]
        mask = int(v[-1], 16) if v[-1].startswith("0x") else 0
        reasons = "|".join(n for b, n in THROTTLE_BITS.items() if mask & b and n != "idle") or "none"
        return v[:-1] + [reasons]
    except Exception:
        return [""] * (len(GPU_FIELDS) - 1) + ["n/a"]


def sample(proc: psutil.Process | None = None) -> list:
    vm = psutil.virtual_memory()
    freq = psutil.cpu_freq()
    rss = 0.0
    if proc is not None:
        try:
            rss = proc.memory_info().rss + sum(c.memory_info().rss for c in proc.children(recursive=True))
        except psutil.Error:
            pass
    return [datetime.now().isoformat(timespec="seconds"), psutil.cpu_percent(None),
            round(freq.current) if freq else "", round(vm.used / 1e9, 2), vm.percent,
            round(rss / 1e9, 2)] + gpu_sample()


class HwMonitor:
    def __init__(self, path: Path, interval: float = 5.0):
        self.path, self.interval = Path(path), interval
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._loop, daemon=True)
        self.proc = psutil.Process()

    def _loop(self) -> None:
        new = not self.path.exists()
        with open(self.path, "a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(COLS)
            psutil.cpu_percent(None)
            while not self._stop.wait(self.interval):
                w.writerow(sample(self.proc))
                f.flush()

    def __enter__(self):
        self._t.start()
        return self

    def __exit__(self, *a):
        self._stop.set()
        self._t.join(timeout=10)


if __name__ == "__main__":
    secs = float(sys.argv[1]) if len(sys.argv) > 1 else 30
    from config import RUNS_DIR
    path = RUNS_DIR / "hw_live.csv"
    psutil.cpu_percent(None)
    with HwMonitor(path, interval=2):
        time.sleep(secs)
    print(open(path).read()[-1500:])
