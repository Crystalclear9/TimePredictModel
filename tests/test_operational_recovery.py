"""Poison isolation, explicit observer fencing and prediction/readiness recovery."""

import asyncio
import json
import time
from types import SimpleNamespace

import httpx
import pytest

from timepredict.events import decode_event
from timepredict.freshness import observation_failure
from timepredict.service import Worker, create_app
from timepredict.service import Retirement, Recovery
from timepredict.state import apply_prediction, empty_state, reduce_event
from timepredict.store import RedisStore


def stamp(t, source="B"):
    return {"utc": t, "mono": t, "source": source, "boot": "b", "error": 0.001, "jumped": False}


def out(t, workflow="w", source="B"):
    return {
        "kind": "out",
        "event_id": workflow,
        "stamp": stamp(t, source),
        "tenant": "t",
        "workflow_id": workflow,
        "agent_id": "a",
        "request_id": "r",
        "call": {"id": "c", "name": "tool", "arguments": "{}"},
    }


@pytest.mark.parametrize(
    "raw",
    [
        "{}",
        "null",
        "[]",
        '{"kind":"out"}',
        '{"kind":NaN}',
        '{"kind":"out","x":1e999}',
        '{"kind":[]}',
    ],
)
def test_bad_envelopes_are_rejected(raw):
    with pytest.raises(ValueError, match="invalid_event_envelope"):
        decode_event(raw)


@pytest.mark.parametrize(
    "field,value", [("request_id", "\ud800"), ("reason", []), ("reason", "waiting")]
)
def test_invalid_fields_cannot_pass_validation_then_poison_reducer(field, value):
    item = {**out(1), field: value}
    with pytest.raises(ValueError, match="invalid_event_envelope"):
        decode_event(json.dumps(item))


@pytest.mark.parametrize("field", ["tool_ids", "description", "telemetry", "object_key"])
def test_invalid_unicode_in_nested_event_data_is_rejected(field):
    item = out(1)
    if field == "tool_ids":
        item.update(kind="ingress", latest_tool_ids=["c", "\ud800"])
    elif field == "description":
        item["call"]["description"] = "\ud800"
    elif field == "telemetry":
        item["telemetry"] = {"note": "\ud800"}
    else:
        item["call"]["schema"] = {"\ud800": "invalid key"}
    with pytest.raises(ValueError, match="invalid_event_envelope"):
        decode_event(json.dumps(item))


@pytest.mark.parametrize("nested", [False, True])
def test_duplicate_json_keys_are_rejected(nested):
    raw = json.dumps(out(1))
    if nested:
        duplicate = raw.replace('"utc": 1', '"utc": 1, "utc": 2', 1)
    else:
        duplicate = raw.replace('"kind": "out"', '"kind": "out", "kind": "out"', 1)
    assert duplicate != raw
    with pytest.raises(ValueError, match="invalid_event_envelope"):
        decode_event(duplicate)


@pytest.mark.parametrize("model", [Retirement, Recovery])
@pytest.mark.parametrize("assertion", [1, 1.0, False, "true", None])
def test_operator_assertions_require_literal_json_true(model, assertion):
    from pydantic import ValidationError

    if model is Retirement:
        data = {
            "source_key": "A/b",
            "expected_seq": 1,
            "routing_fenced": assertion,
            "reason": "fixture",
        }
    else:
        data = {"gap_id": "1-0", "observation_reestablished": assertion, "reason": "fixture"}
    with pytest.raises(ValidationError):
        model.model_validate(data)


def test_poison_message_is_isolated_once_and_valid_successor_is_consumed():
    fakeredis = pytest.importorskip("fakeredis.aioredis")

    async def run():
        redis = fakeredis.FakeRedis(decode_responses=True)
        store = RedisStore(redis, "poison")
        lease = store.prefix + ":worker-lease"
        await redis.set(lease, "leader")
        store.required_lease = (lease, "leader")
        now = time.time()
        bad = await redis.xadd(store.events, {"data": "{}"})
        await redis.xadd(
            store.events,
            {
                "data": json.dumps(
                    {"kind": "source_heartbeat", "event_id": "valid", "seq": 1, "stamp": stamp(now)}
                )
            },
        )
        worker = Worker(store, clock=SimpleNamespace(stamp=lambda: stamp(now + 1, "worker")))
        task = asyncio.create_task(worker.consume())
        try:
            for _ in range(100):
                if "valid" in (await store.read())["seen"]:
                    break
                await asyncio.sleep(0.01)
            else:
                pytest.fail("valid message was blocked behind poison")
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        state = await store.read()
        assert bad in state["transport_failures"]
        assert observation_failure(state, {"out": stamp(now + 2)}) == "invalid_event"
        await store.quarantine(bad, {"data": "{}"}, stamp(now + 1))
        assert await redis.xlen(store.deadletters) == 1
        assert (await redis.xpending(store.events, "predictor"))["pending"] == 0
        state, _ = await store.apply(
            {
                "kind": "transport_recovered",
                "event_id": "recovered",
                "gap_id": bad,
                "observation_reestablished": True,
                "stamp": stamp(now + 3, "worker"),
            }
        )
        assert observation_failure(state, {"out": stamp(now + 4)}) is None
        assert observation_failure(state, {"out": stamp(now + 2)}) == "invalid_event"
        await redis.aclose()

    asyncio.run(run())


@pytest.mark.parametrize("failure", ["wrong_type", "lost_lease", "full"])
def test_quarantine_failure_cannot_commit_partial_state(failure):
    fakeredis = pytest.importorskip("fakeredis.aioredis")

    async def run():
        redis = fakeredis.FakeRedis(decode_responses=True)
        store = RedisStore(redis, "quarantine-failure", deadletter_limit=1)
        if failure == "wrong_type":
            await redis.set(store.deadletters, "fixture wrong type")
        elif failure == "lost_lease":
            store.required_lease = ("lease", "old-token")
            await redis.set("lease", "new-token")
        else:
            await redis.xadd(store.deadletters, {"data": "fixture full queue"})
        with pytest.raises(Exception):
            await store.quarantine("1-0", {"data": "{}"}, stamp(1))
        assert not await redis.exists(store.state_key, store.history, store.notices)
        await redis.aclose()

    asyncio.run(run())


@pytest.mark.parametrize(
    "failure",
    ["model_unavailable", "prediction_dependency_unavailable", "model_acceptance_expired"],
)
def test_transient_prediction_failure_recovers_but_closed_or_ready_records_do_not_change(failure):
    state, _ = reduce_event(empty_state(), out(1))
    record = next(iter(state["records"].values()))
    gen = record["generation"]
    apply_prediction(state, gen, None, failure, stamp(2))
    assert record["prediction_retry_at"] == 4.0
    prediction = {"mu": 1.0, "sigma": 0.5}
    assert apply_prediction(state, gen, prediction, "ready", stamp(5))
    assert record["prediction"] == prediction and "prediction_retry_at" not in record
    assert not apply_prediction(state, gen, None, failure, stamp(6))
    record["status"] = "cancelled"
    record["prediction_status"] = failure
    assert not apply_prediction(state, gen, prediction, "ready", stamp(7))


def test_tick_retries_after_backoff_when_model_appears():
    fakeredis = pytest.importorskip("fakeredis.aioredis")

    async def run():
        redis = fakeredis.FakeRedis(decode_responses=True)
        store = RedisStore(redis, "recover-model")
        t = [time.time()]
        state, _ = await store.apply(out(t[0]))
        record = next(iter(state["records"].values()))
        worker = Worker(store, clock=SimpleNamespace(stamp=lambda: stamp(t[0])), refresh=0.005)
        await worker.predict_one(record)

        class Model:
            version = "restored"

            def predict(self, call):
                return {"mu": 1.0, "sigma": 0.5, "model_version": self.version}, "ready"

        worker.model = Model()
        t[0] += 3
        task = asyncio.create_task(worker.tick())
        try:
            for _ in range(100):
                if (await store.read())["records"][record["generation"]][
                    "prediction_status"
                ] == "ready":
                    break
                await asyncio.sleep(0.01)
            else:
                pytest.fail("tick never retried recoverable prediction")
        finally:
            task.cancel()
            await asyncio.gather(task, *worker.prediction_tasks, return_exceptions=True)
        await redis.aclose()

    asyncio.run(run())


def test_retirement_allows_new_calls_only_and_reappearance_reopens_fault():
    state = empty_state()
    events = [
        {"kind": "source_heartbeat", "event_id": "A", "seq": 1, "stamp": stamp(1, "A")},
        out(2, "old"),
        {"kind": "source_lost", "event_id": "lost", "lost_source": "A/b", "stamp": stamp(32)},
        {
            "kind": "source_retired",
            "event_id": "retire",
            "retired_source": "A/b",
            "expected_seq": 1,
            "routing_fenced": True,
            "stamp": stamp(33, "operator"),
        },
        out(34, "new"),
    ]
    for event in events:
        state, _ = reduce_event(state, event)
    by_workflow = {r["workflow_id"]: r for r in state["records"].values()}
    assert observation_failure(state, by_workflow["old"]) == "observation_lost"
    assert observation_failure(state, by_workflow["new"]) is None
    state, _ = reduce_event(
        state, {"kind": "source_heartbeat", "event_id": "A-back", "seq": 2, "stamp": stamp(35, "A")}
    )
    assert observation_failure(state, by_workflow["new"]) == "source_retirement_violated"


def test_admin_endpoints_require_separate_credentials_and_verified_clock():
    fakeredis = pytest.importorskip("fakeredis.aioredis")

    async def run():
        redis = fakeredis.FakeRedis(decode_responses=True)
        store = RedisStore(redis, "admin")
        await store.apply(
            {"kind": "source_heartbeat", "event_id": "A", "seq": 1, "stamp": stamp(1, "A")}
        )
        clock = SimpleNamespace(stamp=lambda: stamp(3, "operator"))
        app = create_app(store, {"tenant-secret": "t"}, clock, admin_tokens=["operator-secret"])
        body = {
            "source_key": "A/b",
            "expected_seq": 1,
            "routing_fenced": True,
            "reason": "removed from routing",
        }
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://fixture"
        ) as client:
            path = "/internal/admin/retire-source"
            assert (
                await client.post(
                    path, json=body, headers={"Authorization": "Bearer tenant-secret"}
                )
            ).status_code == 403
            assert (
                await client.post(
                    path,
                    json={**body, "expected_seq": 2},
                    headers={"Authorization": "Bearer operator-secret"},
                )
            ).status_code == 409
            assert (
                await client.post(
                    path, json=body, headers={"Authorization": "Bearer operator-secret"}
                )
            ).status_code == 200
            clock.stamp = lambda: {**stamp(4), "error": None}
            assert (
                await client.post(
                    path, json=body, headers={"Authorization": "Bearer operator-secret"}
                )
            ).status_code == 503
        await redis.aclose()

    asyncio.run(run())


def test_readiness_rejects_live_processor_with_failed_source_or_missing_model():
    fakeredis = pytest.importorskip("fakeredis.aioredis")

    async def run():
        redis = fakeredis.FakeRedis(decode_responses=True)
        store = RedisStore(redis, "ready")
        state = empty_state()
        state.update(
            processor=stamp(3),
            processor_model={
                "version": "v",
                "calibrated": True,
                "expires_at": 100.0,
                "shadow": False,
            },
        )
        state["sources"]["A/b"] = {"last": stamp(2, "A"), "bad": False}
        app = create_app(store, {"secret": "t"}, SimpleNamespace(stamp=lambda: stamp(3)))
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://fixture",
            headers={"Authorization": "Bearer secret"},
        ) as client:
            await redis.set(store.state_key, json.dumps(state))
            assert (await client.get("/internal/ready")).status_code == 200
            state["sources"]["A/b"]["bad"] = True
            state["processor_model"] = {}
            await redis.set(store.state_key, json.dumps(state))
            health = (await client.get("/internal/health")).json()
            assert health["live"] and not health["ready"] and health["status"] == "degraded"
            assert {"model_unavailable", "observer_unavailable"} <= set(health["readiness_reasons"])
            assert (await client.get("/internal/ready")).status_code == 503
        await redis.aclose()

    asyncio.run(run())
