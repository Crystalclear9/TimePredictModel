"""A bad Redis stream must not leave an event seen without its history."""
import asyncio
import json

import pytest
from redis.exceptions import ResponseError

from timepredict.store import RedisStore


def out_event():
    return {"event_id": "event1", "kind": "out", "tenant": "t", "workflow_id": "w", "agent_id": "a",
            "request_id": "r1", "stamp": {"utc": 1., "mono": 1., "source": "s", "boot": "b", "error": .001, "jumped": False},
            "source_kind": "sglang", "serial_protocol": True, "call": {"id": "c", "name": "search", "arguments": {}}}


@pytest.mark.parametrize("stream", ["history", "notices"])
@pytest.mark.parametrize("race", [False, True])
def test_stream_type_failure_has_no_partial_state_and_retry_recovers_once(monkeypatch, stream, race):
    fakeredis = pytest.importorskip("fakeredis.aioredis")
    async def run():
        redis = fakeredis.FakeRedis(decode_responses=True)
        store = RedisStore(redis, "write-recovery")
        broken = getattr(store, stream)
        if race:
            original_pipeline = redis.pipeline
            injected = False
            def pipeline(*args, **kwargs):
                pipe = original_pipeline(*args, **kwargs)
                execute = pipe.execute
                async def change_before_exec(*args, **kwargs):
                    nonlocal injected
                    if not injected:
                        injected = True
                        await redis.set(broken, "fixture wrong type")
                    return await execute(*args, **kwargs)
                monkeypatch.setattr(pipe, "execute", change_before_exec)
                return pipe
            monkeypatch.setattr(redis, "pipeline", pipeline)
        else:
            await redis.set(broken, "fixture wrong type")
        with pytest.raises(ResponseError):
            await store.apply(out_event())
        assert not await redis.exists(store.state_key)
        other = store.notices if stream == "history" else store.history
        assert not await redis.exists(other)
        await redis.delete(broken)
        await store.apply(out_event())
        await store.apply(out_event())
        assert len((await store.read())["records"]) == 1
        history = await redis.xrange(store.history)
        assert len(history) == 1 and json.loads(history[0][1]["data"])["event_id"] == "event1"
        assert await redis.xlen(store.notices) == 1
        if race:
            assert store.metrics["cas_retries"] == 1
        await redis.aclose()
    asyncio.run(run())
