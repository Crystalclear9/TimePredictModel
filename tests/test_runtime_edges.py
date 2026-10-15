"""Regression tests for streaming and source-backed administrative censoring."""
from timepredict.sglang import Capture
from timepredict.training import snapshot

import asyncio
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


def stamp(t):
    return {"utc": t, "mono": t, "source": "observer", "boot": "boot", "error": .001, "jumped": False}


def test_identical_stream_name_fragments_are_concatenated():
    capture = Capture({}, {"nana": {"description": "expected tool", "parameters": {}}})
    for delta in (
        {"id": "call", "function": {"name": "na", "arguments": "{"}},
        {"function": {"name": "na", "arguments": "}"}},
    ):
        capture.observe({"choices": [{"delta": {"tool_calls": [{"index": 0, **delta}]}}]}, True)
    result = capture.payload()
    assert result["call"]["name"] == "nana"
    assert result["call"]["description"] == "expected tool"


def test_snapshot_never_exports_censored_row_without_source_heartbeat():
    identity = {"tenant": "t", "workflow_id": "w", "agent_id": "a", "request_id": "r"}
    events = [
        {**identity, "event_id": "out", "kind": "out", "stamp": stamp(1), "seq": 1,
         "source_kind": "sglang", "serial_protocol": True,
         "call": {"id": "c", "name": "tool", "arguments": "{}"}},
        {**identity, "event_id": "heartbeat", "kind": "heartbeat", "stamp": stamp(2)},
    ]
    rows, excluded = snapshot(events, 3, cutoff_stamp=stamp(3))
    assert rows == []
    assert excluded["missing_censoring_heartbeats"] == 1


@pytest.mark.parametrize("require_promoted", [False, True])
def test_removed_direct_quantile_format_has_explicit_error(tmp_path, require_promoted):
    from timepredict.model import Model
    (tmp_path/"manifest.json").write_text(json.dumps({"model_type": "xgboost_direct_log_time"}), encoding="utf-8")
    with pytest.raises(ValueError, match="direct_quantile_package_not_supported"):
        Model(tmp_path, require_promoted=require_promoted)


def test_scale_probe_exercises_ready_distribution_math(tmp_path, monkeypatch):
    fakeredis = pytest.importorskip("fakeredis.aioredis")
    from redis.asyncio import Redis

    path = Path(__file__).resolve().parents[1]/"deploy"/"benchmark_scale.py"
    spec = importlib.util.spec_from_file_location("capacity_probe", path)
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)

    async def run():
        redis = fakeredis.FakeRedis(decode_responses=True)
        monkeypatch.setattr(Redis, "from_url", lambda *a, **kw: redis)
        async def memory_usage(key):
            return len((await redis.get(key)).encode())
        monkeypatch.setattr(redis, "memory_usage", memory_usage)
        args = SimpleNamespace(redis_url="redis://fixture", sizes=[2], events=2, concurrency=1,
                               training_rows=10, max_mutate_p99=10, max_state_bytes=10**7,
                               max_window_seconds=10, output=tmp_path/"scale.json")
        assert await probe.run(args)
        report = json.loads(args.output.read_text(encoding="utf-8"))
        assert report["kind"] == "scale_acceptance"
        assert report["online"][0]["ready_rows"] == 2
        assert not [k async for k in redis.scan_iter("timepredict-scale-*")]
        await redis.aclose()
    asyncio.run(run())
