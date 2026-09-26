"""Kill any running dashboard/server.py and start a fresh detached one (logs in runs/dashboard.*)."""
import os
import subprocess
import time

import psutil

ROOT = r"D:\amazon-ml"
for p in psutil.process_iter(["pid", "cmdline"]):
    cl = p.info["cmdline"] or []
    if p.pid != os.getpid() and len(cl) > 1 and cl[1].replace("/", "\\").endswith("dashboard\\server.py"):
        p.kill()
        print("killed", p.pid)
time.sleep(1)
flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
subprocess.Popen([os.path.join(ROOT, ".venv311", "Scripts", "python.exe"), r"dashboard\server.py"], cwd=ROOT,
                 stdout=open(os.path.join(ROOT, "runs", "dashboard.out"), "w"),
                 stderr=open(os.path.join(ROOT, "runs", "dashboard.err"), "w"), creationflags=flags, close_fds=True)
print("started")
