"""History replay must respect current state and cache ownership."""
import asyncio
import json
import math
from types import SimpleNamespace

import httpx
import pytest

from timepredict.client import WindowConsumer
from timepredict.service import create_app, window_rows
from timepredict.store import RedisStore


def stamp(t):
    return {"utc": t, "mono": t, "source": "fixture", "boot": "b", "error": .001, "jumped": False}


@pytest.mark.parametrize("closing", [None, "ingress", "cancel", "processor_stale"])
def test_notice_page_cannot_offer_a_round_already_closed_in_shared_state(closing):
    fakeredis = pytest.importorskip("fakeredis.aioredis")

    async def run():
        redis = fakeredis.FakeRedis(decode_responses=True)
        store = RedisStore(redis, "notice-recovery")
        out = {"event_id": "out", "kind": "out", "tenant": "tenant", "workflow_id": "w", "agent_id": "a",
               "request_id": "r", "stamp": stamp(1), "call": {"id": "c", "name": "tool", "arguments": "{}"}}
        state, _ = await store.apply(out)
        record = next(iter(state["records"].values()))
        await store.prediction(record["generation"], {"mu": math.log(3), "sigma": .5, "model_version": "fixture",
                                                    "distribution_calibrated": True}, "ready")
        def processor(state):
            state["processor"] = stamp(2)
            return state, []
        await store.mutate(processor)
        row = window_rows(await store.read(), stamp(2), 2)["records"][0]
        cursor = (await redis.xrevrange(store.notices, count=1))[0][0]
        notice = {"action": "candidate_update", "tenant": "tenant", "key": record["key"],
                  "generation": record["generation"], "revision": row["revision"], "H_seconds": 2.,
                  "snapshot_stamp": stamp(2), "snapshot_utc": 2., "candidate": row}
        await redis.xadd(store.notices, {"data": json.dumps(notice)})
        if closing in ("ingress", "cancel"):
            await store.apply({**out, "kind": closing, "event_id": "closing", "request_id": "r2",
                               "stamp": stamp(3), "latest_tool_ids": ["c"]})
        elif closing == "processor_stale":
            def stale(state):
                state["processor"] = stamp(0)
                return state, []
            await store.mutate(stale)
        app = create_app(store, {"secret": "tenant"}, clock=SimpleNamespace(stamp=lambda: stamp(3)))
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://fixture") as client:
            response = await client.get("/internal/notices", params={"after": cursor, "limit": 1},
                                        headers={"Authorization": "Bearer secret"})
            assert response.status_code == 200
            candidate = response.json()["events"][0]["candidate"]
            if closing:
                assert candidate["status"] == "unknown" and not candidate["in_window"]
                assert candidate["valid_for_seconds"] == 0
            else:
                assert candidate["status"] == "ready" and candidate["in_window"]
        await redis.aclose()
    asyncio.run(run())


@pytest.mark.parametrize("cursor", ["١-٠", "18446744073709551616-0", "0-18446744073709551616", "00-0"])
def test_noncanonical_or_overflowing_notice_cursor_is_rejected_before_redis(cursor):
    async def unavailable(*args, **kwargs):
        raise AssertionError("invalid cursor must not reach Redis")
    async def run():
        store = SimpleNamespace(redis=SimpleNamespace(xrange=unavailable), notices="notices")
        app = create_app(store, {"secret": "tenant"})
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app, raise_app_exceptions=False),
                                     base_url="http://fixture") as client:
            response = await client.get("/internal/notices", params={"after": cursor},
                                        headers={"Authorization": "Bearer secret"})
            assert response.status_code == 422
    asyncio.run(run())


@pytest.mark.parametrize("stage", ["xrange", "state"])
def test_notice_transport_outage_returns_503(stage):
    async def unavailable(*args, **kwargs):
        raise ConnectionError("fixture outage")
    async def run():
        async def empty(*args, **kwargs): return []
        store = SimpleNamespace(redis=SimpleNamespace(xrange=unavailable if stage == "xrange" else empty),
                                notices="notices", read=unavailable)
        app = create_app(store, {"secret": "tenant"})
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app, raise_app_exceptions=False),
                                     base_url="http://fixture") as client:
            response = await client.get("/internal/notices", headers={"Authorization": "Bearer secret"})
            assert response.status_code == 503
    asyncio.run(run())


@pytest.mark.parametrize("cursor", ["0-0", "1-0", "18446744073709551615-18446744073709551615"])
def test_canonical_notice_cursor_boundaries_remain_accepted(cursor):
    fakeredis = pytest.importorskip("fakeredis.aioredis")
    async def run():
        redis = fakeredis.FakeRedis(decode_responses=True)
        app = create_app(RedisStore(redis, "cursor-bounds"), {"secret": "tenant"})
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://fixture") as client:
            response = await client.get("/internal/notices", params={"after": cursor},
                                        headers={"Authorization": "Bearer secret"})
            assert response.status_code == 200 and response.json() == {"cursor": cursor, "events": []}
        await redis.aclose()
    asyncio.run(run())


@pytest.mark.parametrize("mutate", ["input", "output"])
def test_candidate_distribution_cannot_be_changed_through_an_external_alias(mutate):
    consumer = WindowConsumer(lambda: 0.)
    row = {"key": "k", "generation": "g", "revision": 1, "status": "ready", "in_window": True,
           "age_seconds": 2., "valid_for_seconds": 1.,
           "distribution": {"mu": math.log(3), "sigma": .5, "distribution_calibrated": True}}
    consumer.snapshot({"revision": 1, "snapshot_utc": 10., "H_seconds": 2., "records": [row]})
    first = consumer.candidates()[0]
    target = row if mutate == "input" else first
    target["distribution"]["distribution_calibrated"] = False
    after = consumer.candidates()
    assert len(after) == 1 and after[0]["distribution"]["distribution_calibrated"] is True
