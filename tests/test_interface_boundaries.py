"""Authentication and local-processing boundaries must fail closed."""

import asyncio
import math

import httpx
import pytest

from timepredict.candidates import CandidateCache
from timepredict.client import WindowConsumer
from timepredict.service import create_app


@pytest.mark.parametrize(
    "tokens",
    [
        {"secret": None},
        {"secret": ""},
        {"": "tenant"},
        {1: "tenant"},
        {"secret": 1},
        ["secret"],
        "secret",
        {"密钥": "tenant"},
    ],
)
def test_invalid_tenant_mapping_is_rejected_at_startup(tokens):
    with pytest.raises(ValueError, match="authentication token"):
        create_app(None, tokens)


def test_non_ascii_authorization_returns_401_instead_of_500():
    async def run():
        app = create_app(None, {"secret": "tenant"})
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app, raise_app_exceptions=False),
            base_url="http://fixture",
        ) as client:
            response = await client.get(
                "/internal/contract", headers={"Authorization": b"Bearer \xff"}
            )
            assert response.status_code == 401
            assert (
                await client.get("/internal/contract", headers={"Authorization": "Bearer secret"})
            ).status_code == 200

    asyncio.run(run())


def test_caller_cannot_mutate_validated_tenant_bindings():
    from timepredict.store import RedisStore

    fakeredis = pytest.importorskip("fakeredis.aioredis")

    async def run():
        redis = fakeredis.FakeRedis(decode_responses=True)
        store = RedisStore(redis, "auth-config")
        for tenant in ("allowed", "other"):
            await store.apply(
                {
                    "event_id": tenant,
                    "kind": "out",
                    "tenant": tenant,
                    "workflow_id": "w",
                    "agent_id": "a",
                    "request_id": "r",
                    "call": {"name": "tool", "arguments": "{}"},
                    "stamp": {
                        "utc": 1.0,
                        "mono": 1.0,
                        "source": "fixture",
                        "boot": "b",
                        "error": 0.001,
                    },
                }
            )
        tokens = {"secret": "allowed"}
        app = create_app(store, tokens)
        tokens["secret"] = (
            None  # The original dictionary no longer controls this app's authorization.
        )
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://fixture"
        ) as client:
            response = await client.get(
                "/internal/window?H_seconds=2", headers={"Authorization": "Bearer secret"}
            )
            assert response.status_code == 200
            assert [row["tenant"] for row in response.json()["records"]] == ["allowed"]
        await redis.aclose()

    asyncio.run(run())


@pytest.mark.parametrize("mode", ["snapshot", "notice"])
def test_client_processing_cannot_extend_candidate_lease(monkeypatch, mode):
    now = [0.0]
    consumer = WindowConsumer(lambda: now[0])
    original = CandidateCache._put

    def slow_put(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        now[0] += 0.6  # Deterministic time spent parsing/copying a large candidate batch.
        return result

    monkeypatch.setattr(CandidateCache, "_put", slow_put)
    rows = [
        {
            "key": f"k{i}",
            "generation": f"g{i}",
            "revision": 10 + i,
            "status": "ready",
            "in_window": True,
            "age_seconds": 2.0,
            "valid_for_seconds": 1.0,
            "distribution": {"mu": math.log(3), "sigma": 0.5, "distribution_calibrated": True},
        }
        for i in range(2)
    ]
    if mode == "snapshot":
        consumer.snapshot(
            {"revision": 11, "snapshot_utc": 100.0, "H_seconds": 2.0, "records": rows}
        )
        assert now[0] == 1.2 and consumer.candidates() == []
    else:
        consumer.notice(
            {
                "action": "candidate_update",
                "key": "k0",
                "generation": "g0",
                "revision": 10,
                "H_seconds": 2.0,
                "candidate": rows[0],
            }
        )
        assert consumer.candidates()[0]["age_seconds"] == pytest.approx(2.6)
        assert consumer.candidates()[0]["valid_for_seconds"] == pytest.approx(0.4)


@pytest.mark.parametrize("count", [1, 2])
@pytest.mark.parametrize("mode", ["snapshot", "notice"])
def test_distribution_computation_consumes_lease_before_returning_batch(monkeypatch, count, mode):
    from timepredict.distribution import LogNormal

    now = [0.0]
    consumer = WindowConsumer(lambda: now[0])
    rows = [
        {
            "key": f"k{i}",
            "generation": f"g{i}",
            "revision": 10 + i,
            "status": "ready",
            "in_window": True,
            "age_seconds": 2.0,
            "valid_for_seconds": 1.0,
            "distribution": {"mu": math.log(3), "sigma": 0.5, "distribution_calibrated": True},
        }
        for i in range(count)
    ]
    if mode == "snapshot":
        consumer.snapshot(
            {"revision": 10 + count, "snapshot_utc": 100.0, "H_seconds": 2.0, "records": rows}
        )
    else:
        for row in rows:
            consumer.notice(
                {
                    "action": "candidate_update",
                    "key": row["key"],
                    "generation": row["generation"],
                    "revision": row["revision"],
                    "H_seconds": 2.0,
                    "candidate": row,
                }
            )
    summarize = LogNormal.summarize

    def slow_summary(self, *args, **kwargs):
        value = summarize(self, *args, **kwargs)
        now[0] += 0.6
        return value

    monkeypatch.setattr(LogNormal, "summarize", slow_summary)
    result = consumer.candidates()
    if count == 1:
        assert result[0]["valid_for_seconds"] == pytest.approx(0.4)
    else:
        assert now[0] == 1.2 and result == []
