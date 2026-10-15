"""Candidate reuse must be bound to the actual data, implementation and options."""
import asyncio
import json
from types import SimpleNamespace

import pytest

from timepredict import pipeline, training


def observed_events(duration=3.):
    events = []
    for i in range(12):
        for kind, t, request in (("out", i*20.+1, "r1"), ("ingress", i*20.+1+duration, "r2"),
                                 ("terminate", i*20.+2+duration, "r2")):
            events.append({"event_id": f"w{i}/{kind}", "kind": kind, "tenant": "t", "workflow_id": f"w{i}",
                           "agent_id": "a", "request_id": request, "source_kind": "sglang", "serial_protocol": True,
                           "latest_tool_ids": ["c"], "call": {"id": "c", "name": "search", "arguments": {"n": 2}},
                           "stamp": {"utc": t, "mono": t, "source": "fixture", "boot": "b", "error": .001}})
    return events


async def setup(tmp_path, monkeypatch):
    fakeredis = pytest.importorskip("fakeredis.aioredis")
    from redis.asyncio import Redis
    redis = fakeredis.FakeRedis(decode_responses=True)
    monkeypatch.setattr(Redis, "from_url", lambda *a, **kw: redis)
    for event in observed_events():
        await redis.xadd("reuse:history", {"data": json.dumps(event)})
    args = SimpleNamespace(redis_url="redis://fixture", prefix="reuse", output=tmp_path,
                           serving_dir=None, evidence=None, policy=None, rounds=1, threads=1,
                           feature_mode="structured", holdout_tool=[])
    return redis, args


def test_dependency_update_invalidates_unchanged_data_reuse(tmp_path, monkeypatch):
    async def run():
        redis, args = await setup(tmp_path, monkeypatch)
        first = await pipeline.retrain_once(args)
        assert (await pipeline.retrain_once(args))["status"] == "unchanged"
        actual = training.versions
        # Explicit metadata fixture; no package is upgraded and no release is promoted.
        monkeypatch.setattr(training, "versions", lambda: {**actual(), "numpy": "fixture-upgraded"})
        second = await pipeline.retrain_once(args)
        assert second["status"] == "candidate_trained"
        assert second["version"] != first["version"]
        assert (await pipeline.retrain_once(args))["status"] == "unchanged"
        await redis.aclose()
    asyncio.run(run())


@pytest.mark.parametrize("contents", ["{broken", "null", "[]"])
def test_bad_retraining_receipt_is_preserved_and_rebuilt(tmp_path, monkeypatch, contents):
    async def run():
        redis, args = await setup(tmp_path, monkeypatch)
        receipt = tmp_path/"last-run.json"
        receipt.write_text(contents, encoding="utf-8")
        result = await pipeline.retrain_once(args)
        assert result["status"] == "candidate_trained"
        archived = list(tmp_path.glob(".invalid-last-run-*.json"))
        assert len(archived) == 1 and archived[0].read_text(encoding="utf-8") == contents
        assert json.loads(receipt.read_text(encoding="utf-8"))["version"] == result["version"]
        assert (await pipeline.retrain_once(args))["status"] == "unchanged"
        await redis.aclose()
    asyncio.run(run())


@pytest.mark.parametrize("mismatch", ["data", "rounds"])
def test_valid_package_for_other_training_cannot_be_reused(tmp_path, monkeypatch, mismatch):
    async def run():
        redis, args = await setup(tmp_path, monkeypatch)
        first = await pipeline.retrain_once(args)
        rows, _ = training.snapshot(observed_events(4. if mismatch == "data" else 3.), 1000.)
        other, _ = training.train(rows, tmp_path/"candidates", rounds=2 if mismatch == "rounds" else 1,
                                  threads=1, data_kind="observed")
        assert other != first["version"]
        receipt = tmp_path/"last-run.json"
        previous = json.loads(receipt.read_text(encoding="utf-8"))
        previous["version"] = other
        receipt.write_text(json.dumps(previous), encoding="utf-8")
        result = await pipeline.retrain_once(args)
        assert result["status"] == "candidate_trained" and result["version"] == first["version"]
        await redis.aclose()
    asyncio.run(run())
