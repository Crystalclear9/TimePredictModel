"""Measure the supported state envelope on target hardware, using a private Redis prefix.

This creates synthetic state; it is capacity evidence only for the supplied sizes/workload.
"""
import argparse
import asyncio
import json
import math
import platform
import resource
import time
import uuid
from pathlib import Path

from redis.asyncio import Redis

from timepredict.state import empty_state, reduce_event
from timepredict.store import RedisStore
from timepredict.training import evaluate
from timepredict.service import window_rows
import numpy as np


async def run(args):
    redis = Redis.from_url(args.redis_url, decode_responses=True)
    reports = []
    try:
        for size in args.sizes:
            store = RedisStore(redis, "timepredict-scale-"+uuid.uuid4().hex)
            state = empty_state()
            now = time.time()
            def event(i, kind="out"):
                return {"event_id": uuid.uuid4().hex, "kind": kind, "tenant": "scale", "workflow_id": str(i),
                        "agent_id": "scale", "request_id": str(i), "stamp": {"utc": now, "mono": 1., "source": "scale", "boot": "fixture", "error": .001},
                        "call": {"id": str(i), "name": "search", "arguments": {"count": 1}, "schema": {}}}
            for i in range(size):
                state, _ = reduce_event(state, event(i), copy_state=False)
            stamp = {**event(0)["stamp"], "utc": now+.1, "mono": 1.1}
            state["processor"] = stamp
            for record in state["records"].values():
                record["prediction_status"] = "ready"
                record["prediction"] = {"mu": math.log(3), "sigma": .5, "model_version": "synthetic-scale"}
            await redis.set(store.state_key, json.dumps(state))
            latencies = []
            sem = asyncio.Semaphore(args.concurrency)
            async def mutation(i):
                async with sem:
                    start = time.perf_counter()
                    await store.apply(event(i % size, "heartbeat"))
                    latencies.append(time.perf_counter()-start)
            started = time.perf_counter()
            await asyncio.gather(*(mutation(i) for i in range(args.events)))
            throughput = args.events/(time.perf_counter()-started)
            p99 = sorted(latencies)[int((len(latencies)-1)*.99)]
            memory = await redis.memory_usage(store.state_key)
            started = time.perf_counter()
            view = window_rows(await store.read(), stamp, 2.)
            refresh_seconds = time.perf_counter()-started
            ready_rows = sum(row["status"] == "ready" for row in view["records"])
            reports.append({"active_records": size, "events": args.events, "concurrency": args.concurrency,
                            "window_read_compute_seconds": refresh_seconds, "window_rows": len(view["records"]), "ready_rows": ready_rows,
                            "mutation_p99_seconds": p99, "mutations_per_second": throughput, "redis_state_memory_bytes": memory,
                            "store": store.metrics.copy(),
                            "passed": ready_rows == size and p99 <= args.max_mutate_p99 and store.metrics["state_bytes"] <= args.max_state_bytes and refresh_seconds <= args.max_window_seconds})
            keys = [key async for key in redis.scan_iter(store.prefix+":*")]
            if keys:
                await redis.delete(*keys)
        rows = [{"lower": 1.+i % 10, "upper": math.inf if i % 5 == 0 else 1.+i % 10} for i in range(args.training_rows)]
        started = time.perf_counter()
        evaluate(rows, np.full(len(rows), math.log(3)), .5)
        result = {"kind": "scale_acceptance", "host": platform.platform(), "created_at": time.time(), "online": reports,
                  "offline": {"rows": len(rows), "evaluation_seconds": time.perf_counter()-started,
                              "process_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss},
                  "budgets": {"max_mutate_p99_seconds": args.max_mutate_p99, "max_state_bytes": args.max_state_bytes, "max_window_seconds": args.max_window_seconds},
                  "passed": all(r["passed"] for r in reports),
                  "limits": "CAS document contains active and retained metadata. Repeat with representative retention/arguments before capacity commitment; no GPU throughput claim."}
        Path(args.output).write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
        return result["passed"]
    finally:
        await redis.aclose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--redis-url", required=True)
    parser.add_argument("--sizes", type=int, nargs="+", required=True)
    parser.add_argument("--events", type=int, default=200)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--training-rows", type=int, default=100000)
    parser.add_argument("--max-mutate-p99", type=float, required=True)
    parser.add_argument("--max-state-bytes", type=int, required=True)
    parser.add_argument("--max-window-seconds", type=float, required=True)
    parser.add_argument("--output", default="scale-acceptance.json")
    args = parser.parse_args()
    if platform.system() != "Linux" or min(*args.sizes, args.events, args.concurrency, args.training_rows, args.max_state_bytes, args.max_mutate_p99, args.max_window_seconds) <= 0:
        parser.error("requires Linux and positive sizes/budgets")
    raise SystemExit(0 if asyncio.run(run(args)) else 1)
