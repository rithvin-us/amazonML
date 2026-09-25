import csv
import json
import sys
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
INDEX = Path(__file__).resolve().parent / "index.html"

sys.path.insert(0, str(ROOT / "code" / "business_entity_resolution" / "src"))
try:
    import psutil
    psutil.cpu_percent(None)
except Exception:
    psutil = None


def read_json(p):
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except Exception:
        return None


def read_csv(p, last=None):
    try:
        with open(p, newline="", encoding="utf-8", errors="replace") as f:
            rows = csv.DictReader(f)
            return list(deque(rows, maxlen=last)) if last else list(rows)
    except Exception:
        return None


def tail(p, n=60):
    try:
        with open(p, encoding="utf-8", errors="replace") as f:
            return "".join(deque(f, maxlen=n))
    except Exception:
        return None


def list_runs():
    try:
        ds = [d for d in RUNS.iterdir() if d.is_dir() and (d / "status.json").exists()]
        ds.sort(key=lambda d: (d / "status.json").stat().st_mtime, reverse=True)
        return [d.name for d in ds]
    except Exception:
        return []


def safe_run(rid, runs):
    return rid if rid and rid in runs else None


def state(rid):
    runs = list_runs()
    rid = safe_run(rid, runs)
    if rid is None:
        try:
            rid = safe_run((RUNS / "LATEST").read_text(encoding="utf-8").strip(), runs)
        except Exception:
            rid = None
    if rid is None and runs:
        rid = runs[0]
    d = RUNS / rid if rid else None
    return {
        "runs": runs,
        "run": rid,
        "status": read_json(d / "status.json") if d else None,
        "metrics": read_json(d / "metrics.json") if d else None,
        "log_tail": tail(d / "log.txt", 150) if d else None,
        "hw": (read_csv(d / "hw.csv", 400) or []) if d else [],
        "curve": (read_csv(d / "train_curve.csv", 2000) or []) if d else [],
        "leaderboard": read_csv(RUNS / "leaderboard.csv") or [],
        "feature_importance": read_json(d / "feature_importance.json") if d else None,
        "config": read_json(d / "config.json") if d else None,
    }


def hw_now():
    try:
        from hwmon import sample, COLS
        return dict(zip(COLS, sample()))
    except Exception as e:
        return {"error": str(e)}


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def send(self, code, body, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_json(self, obj):
        self.send(200, json.dumps(obj, default=str).encode("utf-8"), "application/json; charset=utf-8")

    def do_GET(self):
        u = urlparse(self.path)
        try:
            if u.path in ("/", "/index.html"):
                self.send(200, INDEX.read_bytes(), "text/html; charset=utf-8")
            elif u.path == "/api/state":
                q = parse_qs(u.query)
                self.send_json(state((q.get("run") or [None])[0]))
            elif u.path == "/api/hw_now":
                self.send_json(hw_now())
            else:
                self.send(404, b"not found", "text/plain")
        except Exception as e:
            try:
                self.send(500, str(e).encode("utf-8"), "text/plain")
            except Exception:
                pass


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    srv = ThreadingHTTPServer(("127.0.0.1", port), H)
    print(f"dashboard on http://127.0.0.1:{port}", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
