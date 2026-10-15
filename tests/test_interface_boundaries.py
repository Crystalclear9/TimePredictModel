"""Authentication and local-processing boundaries must fail closed."""
import asyncio
import math

import httpx
import pytest

from timepredict.candidates import CandidateCache
from timepredict.client import WindowConsumer
from timepredict.service import create_app


@pytest.mark.parametrize("tokens", [{"secret": None}, {"secret": ""}, {"": "tenant"},
                                  {1: "tenant"}, {"secret": 1}, ["secret"], "secret", {"密钥": "tenant"}])
def test_invalid_tenant_mapping_is_rejected_at_startup(tokens):
    with pytest.raises(ValueError, match="authentication token"):
        create_app(None, tokens)


def test_non_ascii_authorization_returns_401_instead_of_500():
    async def run():
        app = create_app(None, {"secret": "tenant"})
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app, raise_app_exceptions=False),
                                     base_url="http://fixture") as client:
            response = await client.get("/internal/contract", headers={"Authorization": b"Bearer \xff"})
            assert response.status_code == 401
            assert (await client.get("/internal/contract", headers={"Authorization": "Bearer secret"})).status_code == 200
    asyncio.run(run())


def test_caller_cannot_mutate_validated_tenant_bindings():
    from timepredict.store import RedisStore
    fakeredis = pytest.importorskip("fakeredis.aioredis")

    async def run():
        redis = fakeredis.FakeRedis(decode_responses=True)
        store = RedisStore(redis, "auth-config")
        for tenant in ("allowed", "other"):
            await store.apply({"event_id": tenant, "kind": "out", "tenant": tenant, "workflow_id": "w", "agent_id": "a",
                               "request_id": "r", "call": {"name": "tool", "arguments": "{}"},
                               "stamp": {"utc": 1., "mono": 1., "source": "fixture", "boot": "b", "error": .001}})
        tokens = {"secret": "allowed"}
        app = create_app(store, tokens)
        tokens["secret"] = None  # The original dictionary no longer controls this app's authorization.
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://fixture") as client:
            response = await client.get("/internal/window?H_seconds=2", headers={"Authorization": "Bearer secret"})
            assert response.status_code == 200
            assert [row["tenant"] for row in response.json()["records"]] == ["allowed"]
        await redis.aclose()
    asyncio.run(run())


@pytest.mark.parametrize("mode", ["snapshot", "notice"])
def test_client_processing_cannot_extend_candidate_lease(monkeypatch, mode):
    now = [0.]
    consumer = WindowConsumer(lambda: now[0])
    original = CandidateCache._put

    def slow_put(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        now[0] += .6  # Deterministic time spent parsing/copying a large candidate batch.
        return result

    monkeypatch.setattr(CandidateCache, "_put", slow_put)
    rows = [{"key": f"k{i}", "generation": f"g{i}", "revision": 10+i, "status": "ready", "in_window": True,
             "age_seconds": 2., "valid_for_seconds": 1.,
             "distribution": {"mu": math.log(3), "sigma": .5, "distribution_calibrated": True}}
            for i in range(2)]
    if mode == "snapshot":
        consumer.snapshot({"revision": 11, "snapshot_utc": 100., "H_seconds": 2., "records": rows})
        assert now[0] == 1.2 and consumer.candidates() == []
    else:
        consumer.notice({"action": "candidate_update", "key": "k0", "generation": "g0", "revision": 10,
                         "H_seconds": 2., "candidate": rows[0]})
        assert consumer.candidates()[0]["age_seconds"] == pytest.approx(2.6)
        assert consumer.candidates()[0]["valid_for_seconds"] == pytest.approx(.4)
