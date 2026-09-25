"""Run registry so humans and Claude can watch progress from files.

runs/<run_id>/status.json   live: stage, pct, eta_s, updated
runs/<run_id>/log.txt       append-only log
runs/<run_id>/config.json   config snapshot
runs/<run_id>/metrics.json  final metrics
runs/leaderboard.csv        one row per finished run
"""
from __future__ import annotations

import csv
import json
import time
from datetime import datetime
from pathlib import Path

from config import RUNS_DIR

LEADERBOARD = RUNS_DIR / "leaderboard.csv"
LB_FIELDS = ["run_id", "date", "sample", "val_f05", "val_precision", "val_recall",
             "block_recall", "avg_candidates", "threshold", "lb_score", "notes"]


class Run:
    def __init__(self, name: str, config: dict | None = None, run_id: str | None = None):
        self.id = run_id or f"{datetime.now():%Y%m%d-%H%M%S}-{name}"
        self.dir = RUNS_DIR / self.id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.t0 = time.time()
        self._stage_t0 = self.t0
        self.stage = "init"
        self.metrics: dict = {}
        if config is not None:
            (self.dir / "config.json").write_text(json.dumps(config, indent=2))
        (RUNS_DIR / "LATEST").write_text(self.id)
        self.log(f"run {self.id} started")

    def log(self, msg: str) -> None:
        line = f"[{datetime.now():%H:%M:%S}] [{self.stage}] {msg}"
        print(line, flush=True)
        with open(self.dir / "log.txt", "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def start_stage(self, stage: str) -> None:
        self.stage = stage
        self._stage_t0 = time.time()
        self.progress(0.0)
        self.log("stage start")

    def progress(self, pct: float, note: str = "") -> None:
        el = time.time() - self._stage_t0
        eta = el / pct * (1 - pct) if pct > 0 else None
        status = {"run_id": self.id, "stage": self.stage, "pct": round(pct, 4),
                  "stage_elapsed_s": round(el, 1), "eta_s": None if eta is None else round(eta, 1),
                  "total_elapsed_s": round(time.time() - self.t0, 1), "note": note,
                  "updated": datetime.now().isoformat(timespec="seconds"), "state": "running"}
        (self.dir / "status.json").write_text(json.dumps(status, indent=2))

    def end_stage(self) -> None:
        self.log(f"stage done in {time.time() - self._stage_t0:.1f}s")

    def set_metrics(self, **kw) -> None:
        self.metrics.update(kw)
        (self.dir / "metrics.json").write_text(json.dumps(self.metrics, indent=2))

    def finish(self, sample: float, notes: str = "", state: str = "done") -> None:
        st = json.loads((self.dir / "status.json").read_text()) if (self.dir / "status.json").exists() else {}
        st.update(state=state, updated=datetime.now().isoformat(timespec="seconds"))
        (self.dir / "status.json").write_text(json.dumps(st, indent=2))
        if state != "done":
            return
        new = not LEADERBOARD.exists()
        with open(LEADERBOARD, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=LB_FIELDS)
            if new:
                w.writeheader()
            row = {k: self.metrics.get(k, "") for k in LB_FIELDS}
            row.update(run_id=self.id, date=datetime.now().isoformat(timespec="minutes"),
                       sample=sample, notes=notes)
            w.writerow(row)
        self.log("run finished")


def record_lb_score(run_id: str, score: float) -> None:
    """Fill the portal score the user reports back."""
    rows = list(csv.DictReader(open(LEADERBOARD, encoding="utf-8")))
    for r in rows:
        if r["run_id"] == run_id:
            r["lb_score"] = score
    with open(LEADERBOARD, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=LB_FIELDS)
        w.writeheader()
        w.writerows(rows)
