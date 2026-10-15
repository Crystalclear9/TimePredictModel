"""Any ingress can receive a continuation; its lost telemetry affects other origins."""

import itertools
import math

import pytest

from timepredict.service import window_rows
from timepredict.state import apply_prediction, empty_state, reduce_event
from timepredict.training import snapshot


def stamp(t, source="A", boot="b"):
    return {"utc": t, "mono": t, "source": source, "boot": boot, "error": 0.001, "jumped": False}


def event(kind, t, *, source="A", boot="b", workflow="old", **extra):
    return {
        "kind": kind,
        "event_id": f"{kind}/{t}/{source}/{boot}/{workflow}",
        "stamp": stamp(t, source, boot),
        "tenant": "t",
        "workflow_id": workflow,
        "agent_id": "a",
        "request_id": "r2" if kind == "ingress" else "r1",
        "source_kind": "sglang",
        "serial_protocol": True,
        "latest_tool_ids": ["c"],
        "call": {"id": "c", "name": "tool", "arguments": "{}"},
        **extra,
    }


def predict(state):
    for record in state["records"].values():
        apply_prediction(
            state,
            record["generation"],
            {
                "mu": math.log(10),
                "sigma": 0.5,
                "model_version": "fixture",
                "distribution_calibrated": True,
            },
            "ready",
        )
    state["processor"] = stamp(6, "worker")
    return {
        row["workflow_id"]: row for row in window_rows(state, stamp(6, "worker"), 20)["records"]
    }


@pytest.mark.parametrize("fault", ["gap", "lost"])
@pytest.mark.parametrize("returned", [False, True])
def test_other_ingress_failure_fences_candidates_and_ambiguous_return_labels(fault, returned):
    events = [event("source_heartbeat", 0, source="B", seq=1), event("out", 1)]
    if returned:
        events.append(event("ingress", 2))
    events.append(
        event("source_heartbeat", 3, source="B", seq=3)
        if fault == "gap"
        else event("source_lost", 3, source="worker", lost_source="B/b", last_seen=0)
    )
    state = empty_state()
    for item in events:
        state, _ = reduce_event(state, item)
    if returned:
        rows, excluded = snapshot(events, 6)
        assert not rows and excluded
    else:
        row = predict(state)["old"]
        assert row["status"] == "unknown" and not row["in_window"]


@pytest.mark.parametrize("delayed_out", [False, True])
def test_replacement_observer_does_not_rehabilitate_affected_round_but_allows_new_round(
    delayed_out,
):
    events = [event("source_heartbeat", 0, source="B", seq=1)]
    if not delayed_out:
        events.append(event("out", 1))
    events.extend(
        [
            event("source_heartbeat", 3, source="B", seq=3),
            event("source_heartbeat", 4, source="B", boot="replacement", seq=1),
        ]
    )
    if delayed_out:
        events.append(event("out", 1))
    events.append(event("out", 5, workflow="new"))
    state = empty_state()
    for item in events:
        state, _ = reduce_event(state, item)
    view = predict(state)
    assert view["old"]["status"] == "unknown" and not view["old"]["in_window"]
    assert view["new"]["status"] == "ready" and view["new"]["in_window"]


def test_proven_return_before_last_complete_observer_event_survives_delivery_order():
    events = [
        event("out", 1),
        event("ingress", 3),
        event("source_heartbeat", 4, source="B", seq=1),
        event("source_heartbeat", 5, source="B", seq=3),
    ]
    for order in itertools.permutations(range(4)):
        if order.index(2) > order.index(3):
            continue  # A publisher preserves its own sequence; independent sources can interleave.
        rows, _ = snapshot([events[i] for i in order], 6)
        assert len(rows) == 1 and rows[0]["lower"] == 2.0, order


def test_earlier_return_proof_can_clear_a_provisional_observation_failure():
    events = [
        event("out", 1),
        event("source_heartbeat", 4, source="B", seq=1),
        event("ingress", 6),
        event("source_heartbeat", 5, source="B", seq=3),
    ]
    state = empty_state()
    for item in events:
        state, _ = reduce_event(state, item)
    assert next(iter(state["records"].values())).get("observation_failure")
    events.append(event("ingress", 3))
    state, _ = reduce_event(state, events[-1])
    assert not next(iter(state["records"].values())).get("observation_failure")
    rows, _ = snapshot(events, 7)
    assert len(rows) == 1 and rows[0]["lower"] == 2.0


@pytest.mark.parametrize("boundary", ["return", "out"])
def test_cross_clock_error_bounds_do_not_clear_an_ambiguous_gap_boundary(boundary):
    events = [
        event("source_heartbeat", 0, source="B", seq=1),
        event("source_heartbeat", 4, source="B", seq=2),
        event("source_heartbeat", 5, source="B", seq=4),
        event("source_heartbeat", 6, source="B", boot="replacement", seq=1),
    ]
    if boundary == "return":
        events.extend([event("out", 1), event("ingress", 3.999)])
        rows, _ = snapshot(events, 7)
        assert not rows
    else:
        events.append(event("out", 6.001))
        state = empty_state()
        for item in events:
            state, _ = reduce_event(state, item)
        assert next(iter(state["records"].values())).get("observation_failure")


def test_late_event_from_retired_boot_cannot_retire_replacement():
    state = empty_state()
    for item in [
        event("source_heartbeat", 0, source="B", seq=1),
        event("source_heartbeat", 3, source="B", seq=3),
        event("source_heartbeat", 4, source="B", boot="replacement", seq=1),
        event("source_heartbeat", 3.5, source="B", seq=4),
        event("out", 5, workflow="new"),
    ]:
        state, _ = reduce_event(state, item)
    assert state["sources"]["B/b"]["retired"]
    assert not state["sources"]["B/replacement"].get("retired")
    assert predict(state)["new"]["status"] == "ready"


def test_observer_identity_survives_process_restart_and_defaults_are_replica_specific(monkeypatch):
    from types import SimpleNamespace
    from timepredict.clock import Clock
    from timepredict.sglang import install

    monkeypatch.setenv("TIMEPREDICT_REDIS_URL", "redis://fixture")
    monkeypatch.delenv("TIMEPREDICT_OBSERVER_ID", raising=False)
    captured = []
    app = SimpleNamespace(add_middleware=lambda middleware, **kw: captured.append(kw["source_id"]))
    for port in (30000, 30000, 30001):
        install(app, SimpleNamespace(tokenizer_worker_num=1, host="127.0.0.1", port=port))
    assert captured[0] == captured[1] != captured[2]
    clocks = [Clock(source_id=captured[0]) for _ in range(2)]
    assert clocks[0].source == clocks[1].source and clocks[0].boot != clocks[1].boot
    monkeypatch.setenv("TIMEPREDICT_OBSERVER_ID", "stable-replica-B")
    install(app, SimpleNamespace(tokenizer_worker_num=1))
    assert captured[-1] == "stable-replica-B"


@pytest.mark.parametrize("first,second", [("host/worker", "host"), ("host", "host/worker")])
def test_observer_ids_are_compared_exactly_not_as_prefixes(first, second):
    state = empty_state()
    for item in [
        event("source_heartbeat", 0, source=first, seq=1),
        event("source_heartbeat", 1, source=second, seq=1),
        event("source_heartbeat", 2, source=first, seq=2),
    ]:
        state, _ = reduce_event(state, item)
    assert not any(
        source.get("retired") or source.get("bad") for source in state["sources"].values()
    )


def test_first_late_event_from_older_boot_cannot_retire_new_boot():
    # Consumer-group takeover can encounter a new boot before reclaiming an old pending event.
    state = empty_state()
    for item in [
        event("source_heartbeat", 4, source="B", boot="new", seq=1),
        event("source_heartbeat", 1, source="B", boot="old", seq=1),
        event("out", 5, workflow="new"),
    ]:
        state, _ = reduce_event(state, item)
    assert state["sources"]["B/old"].get("retired")
    assert not state["sources"]["B/new"].get("retired")
    assert predict(state)["new"]["status"] == "ready"


def test_boot_order_is_independent_of_transport_order():
    heartbeats = [event("source_heartbeat", t, source="B", boot=str(t), seq=1) for t in (0, 2, 4)]
    for order in itertools.permutations(heartbeats):
        state = empty_state()
        for item in order:
            state, _ = reduce_event(state, item)
        active = [key for key, value in state["sources"].items() if not value.get("retired")]
        assert active == ["B/4"]


@pytest.mark.parametrize("clock_case", ["overlap", "unverified", "same_instant"])
def test_ambiguous_boot_order_is_fenced_until_proven_replacement(clock_case):
    first = event("source_heartbeat", 1, source="B", boot="first", seq=1)
    second = event("source_heartbeat", 1.001, source="B", boot="second", seq=1)
    if clock_case == "unverified":
        second["stamp"]["utc"] = second["stamp"]["mono"] = 2
        first["stamp"]["error"] = None
    elif clock_case == "same_instant":
        second["stamp"]["utc"] = second["stamp"]["mono"] = 1
        first["stamp"]["error"] = second["stamp"]["error"] = 0.0
    state = empty_state()
    for item in [first, second, event("out", 3, workflow="uncertain")]:
        state, _ = reduce_event(state, item)
    assert predict(state)["uncertain"]["status"] == "unknown"
    if clock_case != "unverified":
        # A later clock-verified boot starts beyond both ambiguous intervals.
        for item in [
            event("source_heartbeat", 4, source="B", boot="replacement", seq=1),
            event("out", 5, workflow="new"),
        ]:
            state, _ = reduce_event(state, item)
        assert predict(state)["new"]["status"] == "ready"


def test_retired_boot_cannot_emit_an_actionable_new_out():
    state = empty_state()
    for item in [
        event("source_heartbeat", 1, source="B", boot="old", seq=1),
        event("source_heartbeat", 3, source="B", boot="new", seq=1),
        event("out", 4, source="B", boot="old", seq=2),
    ]:
        state, _ = reduce_event(state, item)
    record = next(iter(state["records"].values()))
    assert not record["valid"] and record["status"] == "source_restart" and not state["pending"]
