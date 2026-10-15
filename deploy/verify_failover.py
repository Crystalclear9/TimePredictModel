# 文件作用：独立 Linux 验收部署的 leader 故障演练；执行操作者显式提供的停止命令，并验证新 leader 消费事件。
"""Controlled predictor failover drill on a dedicated acceptance deployment.

--stop-command is a JSON argv array explicitly supplied by the operator. No shell.
The operator restarts the stopped replica afterwards.
"""
import argparse
import asyncio
import json
import os
import platform
import time
import uuid
from pathlib import Path

import httpx
from redis.asyncio import Redis


# 记录旧 leader，执行 JSON argv 停止命令，等待新实例接管并在 history 中找到探测心跳。
async def run(args):
    redis = Redis.from_url(args.redis_url, decode_responses=True)
    headers = {"Authorization": "Bearer "+os.environ["TIMEPREDICT_API_KEY"]}
    async with httpx.AsyncClient(timeout=5) as client:
        # 通过内部令牌读取指定实例健康与模型版本。
        async def health(url):
            response = await client.get(url.rstrip("/")+"/internal/health", headers=headers)
            response.raise_for_status()
            return response.json()
        before = await health(args.before)
        old = before.get("processor")
        if not old or before["status"] != "ok" or not before.get("model_version"):
            raise ValueError("old leader must be healthy with a model before the drill")
        tail = await redis.xrevrange(args.prefix+":history", count=1)
        cursor = tail[0][0] if tail else "0-0"
        argv = json.loads(args.stop_command)
        if not isinstance(argv, list) or not argv or not all(isinstance(a, str) for a in argv):
            raise ValueError("stop-command must be a nonempty JSON string array")
        started = time.monotonic()
        process = await asyncio.create_subprocess_exec(*argv)
        if await asyncio.wait_for(process.wait(), 30) != 0:
            raise RuntimeError("stop command failed")
        recovered, event_id = None, None
        deadline = time.monotonic()+args.timeout
        while time.monotonic() < deadline:
            try:
                current = await health(args.after)
                processor = current.get("processor") or {}
                if (processor.get("source"), processor.get("boot")) != (old["source"], old["boot"]) and current["status"] == "ok" and current.get("model_version") == before["model_version"]:
                    response = await client.post(args.after.rstrip("/")+"/internal/lifecycle/heartbeat", headers=headers,
                                                 json={"workflow_id": "failover-"+uuid.uuid4().hex, "agent_id": "drill"})
                    response.raise_for_status()
                    event_id, recovered = response.json()["event_id"], current
                    break
            except (httpx.HTTPError, KeyError):
                pass
            await asyncio.sleep(.5)
        consumed = False
        while event_id and time.monotonic() < deadline:
            batches = await redis.xread({args.prefix+":history": cursor}, count=1000, block=500)
            for _, entries in batches:
                for ident, fields in entries:
                    cursor = ident
                    if json.loads(fields["data"]).get("event_id") == event_id:
                        consumed = True
            if consumed:
                break
        result = {"kind": "worker_failover", "created_at": time.time(), "passed": bool(recovered and consumed),
                  "prefix": args.prefix, "model_version": before["model_version"], "before_processor": old,
                  "after_processor": recovered.get("processor") if recovered else None,
                  "recovery_seconds": time.monotonic()-started, "post_failover_event_consumed": consumed}
        Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2))
    await redis.aclose()
    return result["passed"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", required=True, help="current leader API")
    parser.add_argument("--after", required=True, help="standby API")
    parser.add_argument("--redis-url", required=True)
    parser.add_argument("--prefix", default="timepredict")
    parser.add_argument("--stop-command", required=True)
    parser.add_argument("--timeout", type=float, default=45)
    parser.add_argument("--output", default="failover.json")
    args = parser.parse_args()
    if platform.system() != "Linux" or args.timeout <= 0:
        parser.error("requires Linux and positive timeout")
    raise SystemExit(0 if asyncio.run(run(args)) else 1)
