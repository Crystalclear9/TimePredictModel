# 文件作用：统一时间戳与跨来源耗时计算：同来源同启动用单调时间，跨来源用受校时约束的 UTC。
import math
import os
import subprocess
import time
import uuid


# 跟踪进程来源、启动身份、chrony 误差与系统时间跳变。
class Clock:
    # 为当前进程生成启动标识；初始校时误差可显式注入。
    def __init__(self, error_seconds=None, *, source_id=None):
        if source_id is not None and (not isinstance(source_id, str) or not source_id):
            raise ValueError("source_id must be a nonempty string")
        self.boot = uuid.uuid4().hex
        self.source = source_id or (os.environ.get("TIMEPREDICT_SOURCE", os.uname().nodename if hasattr(os, "uname") else os.environ.get("COMPUTERNAME", "host")) + f":{os.getpid()}")
        self.error_seconds = error_seconds
        self.last = None
        self.jumped = False
        self.last_probe = 0.0

    # 最多每十秒查询一次 chronyc；无同步证明时把误差设为未知。
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

    # 同时记录 UTC 与单调秒数；两者增量差超过 5ms 时永久标记本实例已跳时。
    def stamp(self):
        wall, mono = time.time(), time.perf_counter_ns() / 1_000_000_000.0
        if self.last and abs((wall - self.last[0]) - (mono - self.last[1])) > 0.005:
            self.jumped = True
        self.last = (wall, mono)
        return {"utc": wall, "mono": mono, "source": self.source, "boot": self.boot,
                "error": self.error_seconds, "jumped": self.jumped}


# 计算秒数差；跨来源误差总和必须不超过预算，负值/非有限值直接拒绝。
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
