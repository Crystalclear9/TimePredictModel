"""Exercise late OS-thread publication after task cancellation or lease takeover."""
import asyncio
import json
import threading
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("failure", ["cancel_task", "replace_lease"])
def test_late_publication_thread_cannot_commit_after_trainer_loses_authority(tmp_path, monkeypatch, failure):
    fakeredis = pytest.importorskip("fakeredis.aioredis")
    pytest.importorskip("lupa")
    from redis.asyncio import Redis
    from timepredict import pipeline, release
    started, resume, finished = threading.Event(), threading.Event(), threading.Event()
    commits = []
    # This substitutes package IO only. The real retrainer's final publication guard runs.
    def paused_publish(*args, before_commit=None):
        started.set()
        assert resume.wait(5), "test did not release publication thread"
        try:
            before_commit()
            commits.append("committed")
            return {"decision": "approved"}
        finally:
            finished.set()
    monkeypatch.setattr(release, "promote", paused_publish)
    monkeypatch.setattr(pipeline, "train", lambda *a, **kw: ("fixture-version", {}))
    async def run():
        redis = fakeredis.FakeRedis(decode_responses=True)
        monkeypatch.setattr(Redis, "from_url", lambda *a, **kw: redis)
        for i in range(24):
            t, wf = i*20., f"w{i}"
            for kind, at, request in (("out", t+1, "r1"), ("ingress", t+4, "r2"), ("terminate", t+5, "r2")):
                event = {"kind": kind, "event_id": f"{wf}/{kind}", "tenant": "t", "workflow_id": wf, "agent_id": "a",
                         "request_id": request, "stamp": {"utc": at, "mono": at, "source": "s", "boot": "b", "error": .001, "jumped": False},
                         "source_kind": "sglang", "serial_protocol": True, "latest_tool_ids": ["c"],
                         "call": {"id": "c", "name": "x", "arguments": "{}"}}
                await redis.xadd("cancel-test:history", {"data": json.dumps(event)})
        args = SimpleNamespace(redis_url="redis://fixture", prefix="cancel-test", output=tmp_path,
                               serving_dir=tmp_path/"serving", evidence="fixture-only", policy="fixture-only",
                               threads=1, rounds=1, feature_mode="structured", holdout_tool=[])
        task = asyncio.create_task(pipeline.retrain_once(args))
        try:
            assert await asyncio.to_thread(started.wait, 5)
            if failure == "cancel_task":
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
                resume.set()
            else:
                await redis.set("cancel-test:training-lease", "another-trainer")
                resume.set()
                with pytest.raises(RuntimeError, match="lease lost"):
                    await task
                assert await redis.get("cancel-test:training-lease") == "another-trainer"
            assert await asyncio.to_thread(finished.wait, 5)
            assert not commits
        finally:
            resume.set()
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            await redis.aclose()
    asyncio.run(run())
