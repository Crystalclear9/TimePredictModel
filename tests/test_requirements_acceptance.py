"""Requirement-level tests, with independent mathematical and event-order oracles."""

import hashlib
import asyncio
import copy
import json
import time
import itertools
import math
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import lognorm

from timepredict.distribution import LogNormal
from timepredict.features import FeatureSpec, UnsupportedInput
from timepredict.sglang import accepted
from timepredict.state import empty_state, reduce_event
from timepredict.training import snapshot, train
from timepredict.dataset import validate_label_provenance


def stamp(t):
    return {
        "utc": t,
        "mono": t,
        "source": "fixture",
        "boot": "boot",
        "error": 0.001,
        "jumped": False,
    }


def event(kind, t, request_id="r1"):
    return {
        "event_id": f"{kind}-{t}",
        "kind": kind,
        "tenant": hashlib.sha256(b"key").hexdigest(),
        "workflow_id": "w",
        "agent_id": "a",
        "request_id": request_id,
        "stamp": stamp(t),
        "source_kind": "sglang",
        "serial_protocol": True,
        "latest_tool_ids": ["A"],
        "call": {"id": "A", "name": "search", "arguments": "{}"},
    }


def test_wrong_result_id_is_not_silently_treated_as_missing_id():
    events = []
    observer = SimpleNamespace(
        emit=lambda kind, payload, s: events.append(
            {**payload, "kind": kind, "stamp": s, "event_id": "ingress"}
        )
    )
    raw = SimpleNamespace(
        scope={"state": {"timepredict_observer": observer, "timepredict_received": stamp(3)}},
        headers={
            "authorization": "Bearer key",
            "x-workflow-id": "w",
            "x-agent-id": "a",
            "x-request-id": "r2",
        },
        app=SimpleNamespace(
            state=SimpleNamespace(
                openai_serving_chat=SimpleNamespace(
                    tokenizer_manager=SimpleNamespace(server_args=SimpleNamespace(api_key="key"))
                )
            )
        ),
    )
    request = SimpleNamespace(
        model_dump=lambda **kw: {
            "messages": [
                {"role": "assistant", "tool_calls": [{"id": "A"}]},
                {"role": "tool", "tool_call_id": "WRONG"},
            ],
            "n": 1,
            "parallel_tool_calls": False,
        }
    )
    accepted(request, raw)
    assert events[0]["tool_call_ids_present"] and not events[0]["latest_tool_ids"]
    state, _ = reduce_event(empty_state(), event("out", 1))
    state, _ = reduce_event(state, events[0])
    record = next(iter(state["records"].values()))
    assert record["status"] == "waiting" and not record.get("return")


@pytest.mark.parametrize("kind", ["cancel", "terminate"])
@pytest.mark.parametrize("order", list(itertools.permutations(range(3))))
def test_completed_return_label_is_independent_of_terminal_event_delivery_order(kind, order):
    events = [event("out", 1), event("ingress", 3, "r2"), event(kind, 4)]
    state = empty_state()
    for index in order:
        state, _ = reduce_event(state, events[index])
    record = next(iter(state["records"].values()))
    assert record["status"] == "returned" and record["valid"] and record["duration"] == 2
    rows, _ = snapshot([events[i] for i in order], 5)
    assert len(rows) == 1 and rows[0]["lower"] == rows[0]["upper"] == 2


@pytest.mark.parametrize("kind", ["cancel", "terminate"])
def test_return_after_terminal_boundary_never_becomes_normal_label(kind):
    state, _ = reduce_event(empty_state(), event("out", 1))
    state, _ = reduce_event(state, event(kind, 2))
    state, _ = reduce_event(state, event("ingress", 3, "r2"))
    record = next(iter(state["records"].values()))
    assert not record["valid"] and record["status"] == (
        "cancelled" if kind == "cancel" else "terminated"
    )


@pytest.mark.parametrize("order", list(itertools.permutations(range(4))))
def test_repeated_cancel_cannot_shift_the_first_terminal_boundary(order):
    events = [event("out", 1), event("ingress", 5, "r2"), event("cancel", 4), event("cancel", 8)]
    state = empty_state()
    for index in order:
        state, _ = reduce_event(state, events[index])
    record = next(iter(state["records"].values()))
    assert not record["valid"] and record["status"] == "cancelled"


def test_delayed_old_cancel_does_not_cancel_a_newer_admitted_round():
    state, _ = reduce_event(empty_state(), event("ingress", 5))
    state, _ = reduce_event(state, event("out", 6))
    state, _ = reduce_event(state, event("cancel", 4))
    assert next(iter(state["records"].values()))["status"] == "waiting"


@pytest.mark.parametrize(
    "mu,sigma,age_ratio",
    list(itertools.product((-4.0, 0.0, 2.0, 6.0), (0.1, 0.5, 1.5), (0.0, 0.5, 1.0, 2.0))),
)
def test_distribution_against_independent_scipy_oracle(mu, sigma, age_ratio):
    oracle = lognorm(s=sigma, scale=math.exp(mu))
    age, h = math.exp(mu) * age_ratio, math.exp(mu) * 0.8
    survival = oracle.sf(age)
    if survival < 1e-12:
        return  # This case is intentionally outside the plugin's support boundary.
    result = LogNormal(mu, sigma).summarize(age, h)
    assert result["prob_return_within_H"] == pytest.approx(
        1 - oracle.sf(age + h) / survival, abs=1e-10
    )
    for p, field in ((0.5, "remaining_p50_seconds"), (0.9, "remaining_p90_seconds")):
        assert result[field] == pytest.approx(
            oracle.isf(survival * (1 - p)) - age, rel=1e-8, abs=1e-10
        )
    integral = quad(lambda u: oracle.sf(age + u) / survival, 0, h)[0]
    assert result["expected_capped_remaining_seconds"] == pytest.approx(integral, rel=1e-8)
    z = (math.log(age) - mu - sigma * sigma) / sigma if age else -math.inf
    from scipy.stats import norm

    expected = math.exp(mu + sigma * sigma / 2) * norm.sf(z) / survival - age
    assert result["expected_remaining_seconds"] == pytest.approx(expected, rel=1e-8)


def test_feature_schema_bool_path_identity_and_no_id_leakage():
    call = {
        "name": "tool",
        "schema": {"properties": {"n": True}},
        "arguments": {"n": 10, "x/y": {"~": False}},
    }
    spec = FeatureSpec.fit([call])
    assert any("/x~1y/~0" in field for field in spec.fields)
    assert np.array_equal(
        spec.transform(call), spec.transform({**call, "workflow_id": "different", "t_return": 9999})
    )


def test_tool_holdout_is_removed_from_dictionary_and_entire_workflow(tmp_path):
    rows = []
    for i in range(30):
        rows.append(
            {
                "generation": f"g{i}",
                "workflow": f"w{i}",
                "workflow_start": float(i),
                "workflow_end": float(i) + 0.1,
                "lower": 2.0 + i % 3,
                "upper": 2.0 + i % 3,
                "call": {"name": "unseen" if i % 10 == 0 else "seen", "arguments": {"n": i % 3}},
            }
        )
    rows.append(
        {
            **rows[0],
            "generation": "g0-extra",
            "call": {"name": "seen", "arguments": {"heldout_only": 99}},
        }
    )
    version, report = train(
        rows,
        tmp_path,
        rounds=2,
        threads=1,
        sigmas=(0.5,),
        data_kind="synthetic",
        holdout_tools=["unseen"],
    )
    from timepredict.model import Model

    model = Model(tmp_path / version)
    assert model.spec.tools == ["seen"]
    assert not any("heldout_only" in f for f in model.spec.fields)
    assert report["unknown_tool_test"]["rows"] == 3


def test_censoring_requires_correct_source_agent_and_cutoff_clock_evidence():
    events = [event("out", 1), {**event("source_heartbeat", 5), "seq": 1}, event("heartbeat", 5)]
    rows, excluded = snapshot(events, 6)
    assert not rows and excluded["cutoff_clock_unverified"] == 1
    rows, _ = snapshot(events, 6, cutoff_stamp=stamp(6))
    validate_label_provenance(rows, data_kind="observed")
    broken = copy.deepcopy(rows)
    for e in broken[0]["event_evidence"]["observation"]:
        if e["kind"] == "heartbeat":
            e["agent_id"] = "wrong-agent"
    evidence = broken[0]["event_evidence"]
    broken[0]["source_manifest"]["evidence_sha256"] = hashlib.sha256(
        json.dumps(evidence, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()
    with pytest.raises(ValueError, match="matching_fresh_heartbeats"):
        validate_label_provenance(broken, data_kind="observed")


@pytest.mark.parametrize("bad_value", [float("nan"), "\ud800", "deeply_nested"])
def test_bad_json_event_does_not_block_following_events(bad_value):
    from timepredict.publisher import Publisher

    if bad_value == "deeply_nested":
        bad_value = []
        for _ in range(2000):
            bad_value = [bad_value]

    async def run():
        delivered = []

        class RedisFixture:
            async def xadd(self, stream, fields):
                delivered.append(json.loads(fields["data"]))

        publisher = Publisher(RedisFixture(), SimpleNamespace(stamp=lambda: stamp(1)))
        publisher.emit("bad", {"value": bad_value})
        publisher.emit("source_heartbeat")
        task = asyncio.create_task(publisher.run())
        try:
            await asyncio.wait_for(publisher.queue.join(), 1)
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        assert publisher.dropped == 1 and [e["seq"] for e in delivered] == [2]

    asyncio.run(run())


@pytest.mark.parametrize(
    "schema",
    [{"properties": None}, {"properties": {"n": {"enum": None}}}, {"properties": {"n": []}}],
)
def test_invalid_schema_is_unknown_input_and_does_not_abort_snapshot(schema):
    call = {"name": "tool", "arguments": {"n": 1}, "schema": schema}
    with pytest.raises(UnsupportedInput):
        FeatureSpec.fit([call])
    out, incoming = event("out", 1), event("ingress", 3, "r2")
    out["call"] = {**call, "id": "A"}
    rows, excluded = snapshot([out, incoming], 4)
    assert not rows and excluded["unsupported_input"] == 1


def test_publisher_elected_worker_stream_prediction_and_return_loop():
    fakeredis = pytest.importorskip("fakeredis.aioredis")
    pytest.importorskip("lupa")
    from timepredict.publisher import Publisher
    from timepredict.store import RedisStore
    from timepredict.service import Worker
    from timepredict.clock import Clock

    async def run():
        redis = fakeredis.FakeRedis(decode_responses=True)

        class ClockFixture(Clock):
            def refresh(self):
                pass

        class ModelFixture:
            def predict(self, call):
                return {
                    "mu": math.log(0.05),
                    "sigma": 0.5,
                    "model_version": "fixture",
                    "distribution_calibrated": False,
                }, "ready"

        clock = ClockFixture(error_seconds=0.001)
        store = RedisStore(redis, "stream-integration")
        publisher = Publisher(redis, clock, store.events)
        worker = Worker(store, clock=clock, shadow=True, refresh=0.01)
        worker.model = ModelFixture()
        publisher.start()
        task = asyncio.create_task(worker.run())

        async def await_record(predicate):
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                state = await store.read()
                records = list(state["records"].values())
                if records and predicate(records[0]):
                    return records[0]
                await asyncio.sleep(0.01)
            raise AssertionError(
                f"event stream did not reach required state: records={records!r}; metrics={worker.metrics!r}"
            )

        payload = {
            k: v for k, v in event("out", 1).items() if k not in ("kind", "stamp", "event_id")
        }
        try:
            publisher.emit("prepared", payload)
            publisher.emit("out", payload)
            publisher.emit("heartbeat", payload)
            record = await await_record(lambda r: r["prediction_status"] == "ready")
            assert record["prediction"]["shadow"]
            publisher.emit("ingress", {**payload, "request_id": "r2"})
            returned = await await_record(lambda r: r["status"] == "returned")
            assert returned["valid"] and returned["duration"] > 0
            assert worker.metrics["predictions"] == 1
            assert await redis.xlen(store.notices) > 0
        finally:
            await publisher.stop()
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            await redis.aclose()

    asyncio.run(run())


def test_server_stamp_uses_high_resolution_monotonic_counter(monkeypatch):
    from timepredict.clock import Clock, elapsed

    ticks = iter((1_000_000_000, 1_001_000_000))
    walls = iter((1.0, 1.001))
    monkeypatch.setattr(time, "monotonic", lambda: 1.0)  # Emulate a coarse unchanged Windows tick.
    monkeypatch.setattr(time, "perf_counter_ns", lambda: next(ticks))
    monkeypatch.setattr(time, "time", lambda: next(walls))
    clock = Clock()
    before, after = clock.stamp(), clock.stamp()
    assert elapsed(before, after) == pytest.approx(0.001)
    assert not after["jumped"]
