import math
import os
import subprocess
import time
import uuid


class Clock:
    def __init__(self, error_seconds=None):
        self.boot = uuid.uuid4().hex
        self.source = os.environ.get("TIMEPREDICT_SOURCE", os.uname().nodename if hasattr(os, "uname") else os.environ.get("COMPUTERNAME", "host")) + f":{os.getpid()}"
        self.error_seconds = error_seconds
        self.last = None
        self.jumped = False
        self.last_probe = 0.0

    def refresh(self):
        if time.monotonic() - self.last_probe < 10:
            return
        self.last_probe = time.monotonic()
        try:
            row = subprocess.check_output(["chronyc", "-c", "tracking"], timeout=2, text=True).strip().split(",")
            # chronyc CSV: system offset, root delay, root dispersion, leap status.
            self.error_seconds = abs(float(row[4])) + abs(float(row[10])) / 2 + abs(float(row[11])) if row[-1] == "Normal" else None
        except (OSError, subprocess.SubprocessError, ValueError, IndexError):
            self.error_seconds = None

    def stamp(self):
        wall, mono = time.time(), time.monotonic()
        if self.last and abs((wall - self.last[0]) - (mono - self.last[1])) > 0.005:
            self.jumped = True
        self.last = (wall, mono)
        return {"utc": wall, "mono": mono, "source": self.source, "boot": self.boot,
                "error": self.error_seconds, "jumped": self.jumped}


def elapsed(start, end, budget=0.005):
    if start.get("jumped") or end.get("jumped"):
        raise ValueError("clock_jump")
    if (start["source"], start["boot"]) == (end["source"], end["boot"]):
        delta = end["mono"] - start["mono"]
    else:
        errors = [start.get("error"), end.get("error")]
        if any(e is None or not math.isfinite(e) or e < 0 for e in errors) or sum(errors) > budget:
            raise ValueError("clock_uncertain")
        delta = end["utc"] - start["utc"]
    if not math.isfinite(delta) or delta < 0:
        raise ValueError("invalid_interval")
    return delta
