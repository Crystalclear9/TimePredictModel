# 文件作用：Linux 服务器验收：比较基线与采集部署的完整单工具往返，汇总关联、时钟和性能证据。
"""Full single-tool roundtrip acceptance; run against isolated Linux acceptance servers.

The tool execution is a fixed delay/result; generated code is never executed.
"""
import argparse
import asyncio
import json
import math
import os
import platform
import statistics
import time
import uuid
from pathlib import Path

import httpx
from redis.asyncio import Redis

from timepredict.clock import elapsed
from timepredict.sglang import Capture
from timepredict.store import RedisStore


# 统一非流式响应与 SSE 拼接；不完整流或异常工具输出不能继续工具轮次。
async def completion(client, url, body, headers):
    if not body["stream"]:
        response = await client.post(url.rstrip("/")+"/v1/chat/completions", json=body, headers=headers)
        response.raise_for_status()
        data = response.json()
        if data["choices"][0].get("finish_reason") in ("length", "abort"):
            raise ValueError("incomplete response")
        return data["choices"][0]["message"]
    capture, content, done = Capture({}, {}), [], False
    async with client.stream("POST", url.rstrip("/")+"/v1/chat/completions", json=body, headers=headers) as response:
        response.raise_for_status()
        async for line in response.aiter_lines():
            if not line.startswith("data: "):
                continue
            if line[6:] == "[DONE]":
                done = True
                break
            data = json.loads(line[6:])
            capture.observe(data, True)
            for choice in data.get("choices", []):
                content.append((choice.get("delta") or {}).get("content") or "")
    if not done or capture.reason:
        raise ValueError("incomplete/error SSE stream")
    result = {"role": "assistant", "content": "".join(content) or None}
    if capture.saw_tool:
        payload = capture.payload()
        if payload["reason"]:
            raise ValueError(payload["reason"])
        call = payload["call"]
        result["tool_calls"] = [{"id": call["id"], "type": "function", "function": {"name": call["name"], "arguments": call["arguments"]}}]
    return result


# 校验健康状态、来源新鲜度和跨主机校时误差总和不超过 5ms。
def clock_ok(health):
    now = health.get("clock", {})
    if health.get("status") != "ok" or not health.get("sources"):
        return False
    for source in health["sources"].values():
        if source.get("retired"):
            continue
        stamp = source["last"]
        errors = [now.get("error"), stamp.get("error")]
        if source.get("bad") or any(v is None or not math.isfinite(v) or v < 0 for v in errors) or sum(errors) > .005:
            return False
        try:
            if elapsed(stamp, now) > 10:
                return False
        except ValueError:
            return False
    return True


# 预热并交替测试两组部署，以 Redis 闭合记录和遥测构建 server_acceptance 证据。
async def run(args):
    payloads = [json.loads(line) for line in Path(args.requests).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not payloads or any(not p.get("tools") for p in payloads):
        raise ValueError("each workload row must define a tool and messages")
    auth = {"Authorization": "Bearer "+os.environ["SGLANG_API_KEY"]}
    internal = {"Authorization": "Bearer "+os.environ["TIMEPREDICT_API_KEY"]}
    redis = Redis.from_url(args.redis_url, decode_responses=True)
    store = RedisStore(redis, args.prefix)
    trials, health_samples, probe_errors, tracked = {"baseline": [], "instrumented": []}, [], [], {}
    async with httpx.AsyncClient(timeout=args.timeout, limits=httpx.Limits(max_connections=args.concurrency+4)) as client:
        # 读取经过认证的预测服务健康响应。
        async def health():
            response = await client.get(args.predictor.rstrip("/")+"/internal/health", headers=internal)
            response.raise_for_status()
            return response.json()
        before = await health()
        # 每秒采集一次健康状态，探测异常单独累计。
        async def monitor():
            while True:
                try:
                    health_samples.append(await health())
                except Exception as exc:
                    probe_errors.append(type(exc).__name__)
                await asyncio.sleep(1)
        monitor_task = asyncio.create_task(monitor())

        # 按并发限额执行一组完整往返，计算吞吐与 P99，比较区间之外再清理生命周期。
        async def measure(urls, count, observed):
            sem, durations, errors = asyncio.Semaphore(args.concurrency), [], []
            # 发出单工具请求、等待固定延时，并在下一入口携带匹配工具 ID 续接。
            async def request(i):
                async with sem:
                    payload = {**payloads[i % len(payloads)], "stream": bool(i % 2), "n": 1, "parallel_tool_calls": False}
                    workflow = "acceptance-"+uuid.uuid4().hex
                    headers = {**auth, "X-Workflow-ID": workflow, "X-Agent-ID": "benchmark", "X-Request-ID": uuid.uuid4().hex}
                    started = time.perf_counter()
                    try:
                        first = await completion(client, urls[i % len(urls)], payload, headers)
                        calls = first.get("tool_calls") or []
                        if len(calls) != 1:
                            raise ValueError("workload must generate exactly one tool call")
                        if observed:
                            tracked[workflow] = {"stream": payload["stream"], "call_id": calls[0]["id"], "returned": False}
                        await asyncio.sleep(args.tool_delay)
                        followup = {**payload, "tool_choice": "none", "messages": [*payload["messages"], first,
                                    {"role": "tool", "tool_call_id": calls[0]["id"], "content": args.tool_result}]}
                        headers["X-Request-ID"] = uuid.uuid4().hex
                        last = await completion(client, urls[(i+1) % len(urls)], followup, headers)
                        if last.get("tool_calls"):
                            raise ValueError("unexpected extra tool round")
                        durations.append(time.perf_counter()-started)
                        if observed:
                            tracked[workflow]["returned"] = True
                    except Exception as exc:
                        errors.append(type(exc).__name__+": "+str(exc)[:160])
            started = time.perf_counter()
            await asyncio.gather(*(request(i) for i in range(count)))
            duration = time.perf_counter()-started
            # Lifecycle cleanup is outside the compared HTTP throughput interval.
            if observed:
                for workflow, info in tracked.items():
                    if info.get("terminated"):
                        continue
                    try:
                        result = await client.post(args.predictor.rstrip("/")+"/internal/lifecycle/terminate", headers=internal,
                                                   json={"workflow_id": workflow})
                        result.raise_for_status()
                        info["terminated"] = True
                    except Exception:
                        errors.append("terminate failed")
            return {"roundtrips": count, "success": len(durations), "errors": errors, "elapsed_seconds": duration,
                    "throughput_rps": 2*len(durations)/duration,
                    "roundtrip_p99_seconds": sorted(durations)[int((len(durations)-1)*.99)] if durations else None}
        try:
            await measure(args.baseline, args.warmup, False)
            await measure(args.instrumented, args.warmup, True)
            for repeat in range(args.repeats):
                order = [("baseline", args.baseline), ("instrumented", args.instrumented)]
                for name, urls in (order if repeat % 2 == 0 else reversed(order)):
                    trials[name].append(await measure(urls, args.requests_per_trial, name == "instrumented"))
            await asyncio.sleep(6)  # queue drain and fresh ingress telemetry heartbeat
            after = await health()
            state = await store.read()
        finally:
            monitor_task.cancel()
            await asyncio.gather(monitor_task, return_exceptions=True)
            await redis.aclose()
    records = [r for r in state["records"].values() if r["workflow_id"] in tracked]
    matched = {r["workflow_id"] for r in records if r["status"] == "returned" and r["valid"]
               and r["call"]["id"] == tracked[r["workflow_id"]]["call_id"]}
    # 确认指定流式模式的所有工作流在客户端及 Redis 状态中都完成回返。
    def mode_pass(stream):
        workflows = {w for w, v in tracked.items() if v["stream"] == stream}
        return bool(workflows) and workflows <= matched and all(tracked[w]["returned"] for w in workflows)
    cross = [r for r in records if r["workflow_id"] in matched and r["out"]["source"] != r["return"]["source"]]
    base = statistics.median(t["throughput_rps"] for t in trials["baseline"])
    observed = statistics.median(t["throughput_rps"] for t in trials["instrumented"])
    relevant_sources = {r["out"]["source"]+"/"+r["out"]["boot"] for r in records}
    hooks, drops, queues = [], [], []
    for source_id in relevant_sources:
        telemetry = after["sources"].get(source_id, {}).get("telemetry", {})
        hook = telemetry.get("hook_seconds", {}).get("accepted", {}).get("p99")
        if hook is not None:
            hooks.append(hook)
        drops.append(telemetry.get("dropped"))
    for sample in health_samples:
        queues += [s.get("telemetry", {}).get("queue_depth", 0) for s in sample.get("sources", {}).values()]
    failover = json.loads(Path(args.failover_evidence).read_text(encoding="utf-8")) if args.failover_evidence else {}
    checks = {"sglang_nonstream": mode_pass(False), "sglang_stream": mode_pass(True),
              "prediction_observed": before.get("model_version") == after.get("model_version") and bool(after.get("model_version")) and
                  any(r.get("prediction", {}) and r["prediction"].get("model_version") == after.get("model_version") for r in records),
              "cross_ingress_roundtrip": bool(cross) and len(args.instrumented) >= 2,
              "clock_bounds": bool(health_samples) and not probe_errors and all(clock_ok(h) for h in health_samples),
              "worker_failover": failover.get("kind") == "worker_failover" and failover.get("passed") is True
                  and failover.get("prefix") == args.prefix and failover.get("model_version") == after.get("model_version")
                  and 0 <= time.time()-failover.get("created_at", 0) < 86400}
    metrics = {"throughput_drop_fraction": 1-observed/base if base else None,
               "entry_hook_p99_seconds": max(hooks) if len(hooks) == len(relevant_sources) and hooks else None,
               "prediction_available_p99_seconds": after.get("latency_seconds", {}).get("prediction_available_seconds", {}).get("p99"),
               "dropped_events": sum(drops) if drops and all(d is not None for d in drops) else None,
               "request_errors": sum(len(t["errors"]) for group in trials.values() for t in group)+len(probe_errors),
               "peak_observed_queue_depth": max(queues, default=0)}
    result = {"kind": "server_acceptance", "created_at": time.time(), "model_version": after.get("model_version"),
              "checks": checks, "metrics": metrics, "trials": trials, "health_before": before, "health_after": after,
              "probe_errors": probe_errors, "matched_roundtrips": len(matched), "tracked_roundtrips": len(tracked),
              "prediction_status_counts": {status: sum(r["prediction_status"] == status for r in records) for status in {r["prediction_status"] for r in records}},
              "failover": failover, "host": platform.platform(),
              "scope": "Fixed-delay synthetic tool workload; no accuracy or KV benefit claim. Identical baseline hardware/model/config required."}
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"checks": checks, "metrics": metrics}, indent=2))
    return all(checks.values()) and metrics["request_errors"] == 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", nargs="+", required=True)
    parser.add_argument("--instrumented", nargs="+", required=True)
    parser.add_argument("--predictor", required=True, help="leader API URL for process-local latency telemetry")
    parser.add_argument("--redis-url", required=True)
    parser.add_argument("--prefix", default="timepredict")
    parser.add_argument("--failover-evidence")
    parser.add_argument("--requests", required=True)
    parser.add_argument("--output", default="server-acceptance.json")
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--requests-per-trial", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--tool-delay", type=float, default=.5)
    parser.add_argument("--tool-result", default='{"ok":true}')
    args = parser.parse_args()
    if platform.system() != "Linux" or min(args.concurrency, args.requests_per_trial, args.warmup, args.repeats) < 1 or not 0 <= args.tool_delay < 25:
        parser.error("run on Linux with positive counts and tool delay in [0,25)")
    raise SystemExit(0 if asyncio.run(run(args)) else 1)
