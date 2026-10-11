# 文件作用：本地 HTTP 受控回放：真实测量网络/工具/固定延时的时间，样本始终标记为合成回放。
"""Measured HTTP round trips to synthetic, controlled tools (never production labels)."""
import argparse
import concurrent.futures
import hashlib
import json
import random
import sqlite3
import tempfile
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

VERSION = "replay-1"
NAMES = ("sleep_tool", "http_tool", "file_tool", "database_tool")

# 使用高分辨率单调计时器，秒为单位。
def _tick():
    """High resolution monotonic seconds; Windows monotonic() is too coarse."""
    return time.perf_counter_ns() / 1_000_000_000.0


# 构造四类工具与参数，启动本地端点并并发执行，保存样本、时间划分及数据清单。
def generate(output, rows=5000, seed=42, concurrency=4):
    if rows < 100 or concurrency < 1:
        raise ValueError("need >=100 rows and positive concurrency")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "samples.jsonl").exists():
        raise ValueError("refusing to overwrite an existing replay")
    rng = random.Random(seed)
    jobs = []
    # Multiple calls belong to the same workflow.  This is intentional: the
    # temporal splitter must be able to quarantine a whole workflow and the
    # test set must contain workflows never seen during fitting.
    calls_per_workflow = 10
    workflow_count = max(500, (rows + calls_per_workflow - 1) // calls_per_workflow)
    workflow_starts = {}
    workflow_lock = threading.Lock()
    for i in range(rows):
        name = NAMES[i % 4]
        size = rng.randint(1, 64)
        args = ({"items": size} if name == "sleep_tool" else
                {"payload_bytes": size * 128} if name == "http_tool" else
                {"file_bytes": size * 256, "operation": "write_read"} if name == "file_tool" else
                {"rows": size * 4, "complexity": 1})
        workflow = f"replay-{seed}-wf-{i // calls_per_workflow:04d}"
        jobs.append((i, workflow, name, args, rng.uniform(-0.0015, 0.0015)))

    # 模拟工具调用完整响应的 HTTP 端点；并非真实 SGLang。
    class Handler(BaseHTTPRequestHandler):
        # 关闭回放服务器的默认访问日志。
        def log_message(self, *args):
            pass

        # 接收模拟调用，在响应写出前记录 t_out，并通过 JSON 交给本地执行器。
        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            # The endpoint represents SGLang's completed response.  It emits
            # t_out before the Agent starts the tool, so the measured target
            # is exactly the post-response return interval.
            t_out = _tick()
            body = json.dumps({"t_out": t_out}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    # 执行文件/SQLite 或模拟工作量，加参数控制延时；本地 tool_end 同时作为 t_return。
    def execute(job):
        i, workflow, name, args, jitter = job
        utc, start = time.time(), _tick()
        with workflow_lock:
            workflow_starts.setdefault(workflow, utc)
        payload = json.dumps({"name": name, "arguments": args, "jitter": jitter}).encode()
        req = urllib.request.Request(f"http://127.0.0.1:{server.server_port}/", payload)
        with urllib.request.urlopen(req, timeout=30) as response:
            stamps = json.load(response)
        tool_start = _tick()
        if name == "file_tool":
            with tempfile.TemporaryFile() as f:
                f.write(b"x" * args["file_bytes"]); f.seek(0); f.read()
            work = args["file_bytes"] / 256
        elif name == "database_tool":
            with sqlite3.connect(":memory:") as db:
                db.execute("create table items (x integer)")
                db.executemany("insert into items values (?)", [(i,) for i in range(args["rows"])])
                db.execute("select sum(x) from items").fetchone()
            work = args["rows"] / 4
        elif name == "http_tool":
            work = args["payload_bytes"] / 128
        else:
            work = args["items"]
        delay = 0.008 + (NAMES.index(name) + 1) * 0.0008 + work * 0.00055
        time.sleep(max(0, delay + jitter))
        tool_end = _tick()
        end = tool_end
        call = {"name": name, "arguments": args, "description": f"Controlled replay {name}",
                "schema": {"type": "object", "properties": {k: {"type": "integer" if isinstance(v, int) else "string"} for k, v in args.items()}}}
        server_out = stamps.get("t_out", start)
        duration = end - server_out
        return {"generation": f"replay-{seed}-{i}", "workflow": workflow,
                "workflow_start": workflow_starts[workflow], "workflow_end": utc + duration, "call": call,
                "raw_arguments": json.dumps(args), "lower": duration, "upper": duration,
                "label_kind": "observed_replay", "data_kind": "synthetic_replay",
                "production_releasable": False, "converter_version": VERSION,
                "boundaries": {"t_out": server_out, "t_tool_start": tool_start,
                               "t_tool_end": tool_end, "t_return": end,
                               "clock": "same_process_monotonic"}}

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
            samples = list(executor.map(execute, jobs))
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
    samples.sort(key=lambda r: r["workflow_start"])
    workflow_ends = {}
    for sample in samples:
        workflow_ends[sample["workflow"]] = max(workflow_ends.get(sample["workflow"], 0.0), sample["workflow_end"])
    for sample in samples:
        sample["workflow_end"] = workflow_ends[sample["workflow"]]
    data = "".join(json.dumps(r, sort_keys=True) + "\n" for r in samples)
    (output / "samples.jsonl").write_text(data, encoding="utf-8")
    from .training import split_rows
    groups = split_rows(samples)
    for name, group in groups.items():
        (output / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in group), encoding="utf-8")
    manifest = {"version": VERSION, "seed": seed, "rows": rows, "concurrency": concurrency,
                "label_kind": "observed_replay", "production_releasable": False,
                "sha256": hashlib.sha256(data.encode()).hexdigest(),
                "warning": "Measured round trips with parameter-controlled synthetic delays; not real SGLang or hardware performance.",
                "split_counts": {k: len(v) for k, v in groups.items()}}
    (output / "dataset-manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
