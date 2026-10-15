"""A lease-fenced replacement must reclaim pending events before reading new ones."""

import asyncio
import json
import time
from types import SimpleNamespace

import pytest

from timepredict.service import Worker
from timepredict.store import RedisStore


def test_consumer_cannot_run_without_election_context():
    worker = Worker(SimpleNamespace(redis=None, required_lease=None))

    async def run():
        with pytest.raises(RuntimeError, match="elected worker lease"):
            await worker.consume()

    asyncio.run(run())


def test_displaced_worker_cannot_commit_reclaimed_event(monkeypatch):
    fakeredis = pytest.importorskip("fakeredis.aioredis")

    async def run():
        redis = fakeredis.FakeRedis(decode_responses=True)
        store = RedisStore(redis, "lease-fence")
        lease = store.prefix + ":worker-lease"
        await redis.set(lease, "old-leader")
        store.required_lease = (lease, "old-leader")
        pipeline = redis.pipeline
        replaced = False

        def injected_pipeline(*args, **kwargs):
            pipe = pipeline(*args, **kwargs)
            execute = pipe.execute

            async def replace_before_commit(*args, **kwargs):
                nonlocal replaced
                if not replaced:
                    replaced = True
                    await redis.set(lease, "new-leader")
                return await execute(*args, **kwargs)

            monkeypatch.setattr(pipe, "execute", replace_before_commit)
            return pipe

        monkeypatch.setattr(redis, "pipeline", injected_pipeline)
        item = {
            "kind": "source_heartbeat",
            "event_id": "reclaimed",
            "seq": 1,
            "stamp": {
                "utc": time.time(),
                "mono": 1.0,
                "source": "observer",
                "boot": "b",
                "error": 0.001,
            },
        }
        with pytest.raises(RuntimeError, match="worker lease lost"):
            await store.apply(item)
        assert not await redis.exists(store.state_key, store.history, store.notices)
        assert store.metrics["cas_retries"] == 1
        store.required_lease = (lease, "new-leader")
        await store.apply(item)
        assert await redis.xlen(store.history) == 1
        assert not (await store.read())["sources"]["observer/b"]["bad"]
        await redis.aclose()

    asyncio.run(run())


@pytest.mark.parametrize("pending_count", [1, 101])
def test_immediate_leader_takeover_drains_pending_before_new_messages(monkeypatch, pending_count):
    fakeredis = pytest.importorskip("fakeredis.aioredis")

    async def run():
        redis = fakeredis.FakeRedis(decode_responses=True)
        store = RedisStore(redis, "immediate-takeover")
        origin = time.time() - 1000

        def event(seq):
            return {
                "kind": "source_heartbeat",
                "event_id": f"e{seq}",
                "seq": seq,
                "stamp": {
                    "utc": origin + seq,
                    "mono": float(seq),
                    "source": "observer",
                    "boot": "b",
                    "error": 0.001,
                },
            }

        await redis.xgroup_create(store.events, "predictor", id="0", mkstream=True)
        for seq in range(1, pending_count + 2):
            await redis.xadd(store.events, {"data": json.dumps(event(seq))})
        reserved = await redis.xreadgroup(
            "predictor", "old-leader", {store.events: ">"}, count=pending_count + 1
        )
        await store.apply(event(1))
        await redis.xack(store.events, "predictor", reserved[0][1][0][0])
        # All remaining old-leader entries are fresh pending messages, not 10s idle.
        newest = pending_count + 2
        await redis.xadd(store.events, {"data": json.dumps(event(newest))})
        lease = store.prefix + ":worker-lease"
        await redis.set(lease, "new-leader")
        store.required_lease = (lease, "new-leader")
        applied, complete = [], asyncio.Event()
        apply = store.apply

        async def observed(item):
            result = await apply(item)
            applied.append(item["seq"])
            if item["seq"] == newest:
                complete.set()
            return result

        monkeypatch.setattr(store, "apply", observed)
        clock = SimpleNamespace(
            stamp=lambda: {
                "utc": origin + 1000,
                "mono": 1000.0,
                "source": "worker",
                "boot": "b",
                "error": 0.001,
            }
        )
        worker = Worker(store, clock=clock)
        task = asyncio.create_task(worker.consume())
        try:
            await asyncio.wait_for(complete.wait(), 5)
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        assert applied == list(range(2, newest + 1))
        state = await store.read()
        assert state["sources"]["observer/b"]["seq"] == newest
        assert not state["sources"]["observer/b"]["bad"]
        assert await redis.xlen(store.history) == newest
        await redis.aclose()

    asyncio.run(run())
