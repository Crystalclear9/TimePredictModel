"""Associations use event boundary times, rather than transport delivery order."""
import itertools

import pytest

from timepredict.state import empty_state, reduce_event


def event(kind, t, request_id, call_id, *, supplied=True):
    return {"event_id": f"{kind}-{t}-{request_id}", "kind": kind, "tenant": "t", "workflow_id": "w", "agent_id": "a",
            "request_id": request_id, "stamp": {"utc": t, "mono": t, "source": "s", "boot": "b", "error": .001, "jumped": False},
            "serial_protocol": True, "call": {"id": call_id, "name": "tool", "arguments": "{}"},
            "latest_tool_ids": [call_id] if kind == "ingress" and supplied else []}


@pytest.mark.parametrize("supplied", [False, True])
@pytest.mark.parametrize("rounds", [2, 3])
def test_all_serial_round_delivery_permutations(supplied, rounds):
    events = []
    for i in range(rounds):
        events.extend([event("out", 1.+i*3, f"r{i}", f"c{i}", supplied=supplied),
                       event("ingress", 3.+i*3, f"r{i+1}", f"c{i}", supplied=supplied)])
    for order in itertools.permutations(range(len(events))):
        state = empty_state()
        for index in order:
            state, _ = reduce_event(state, events[index])
        records = {r["request_id"]: r for r in state["records"].values()}
        assert not state["pending"], order
        for i in range(rounds):
            record = records[f"r{i}"]
            assert (record["status"], record["valid"], record.get("duration"), record.get("return_request_id")) == (
                "returned", True, 2., f"r{i+1}"), (order, records)


def test_repeated_matching_result_history_cannot_replace_earliest_real_return():
    state, _ = reduce_event(empty_state(), event("out", 1., "r0", "c0"))
    state, _ = reduce_event(state, event("ingress", 5., "repeated-history", "c0"))
    state, _ = reduce_event(state, event("ingress", 3., "r1", "c0"))
    record = next(iter(state["records"].values()))
    assert record["duration"] == 2. and record["return_request_id"] == "r1" and record["valid"]


def test_retry_with_same_request_id_uses_earliest_proven_server_entry():
    state, _ = reduce_event(empty_state(), event("out", 1., "r0", "c0"))
    state, _ = reduce_event(state, event("ingress", 5., "r1", "c0"))
    state, _ = reduce_event(state, event("ingress", 3., "r1", "c0"))
    state, _ = reduce_event(state, event("ingress", 7., "r1", "c0"))
    record = next(iter(state["records"].values()))
    assert record["duration"] == 2. and record["return"]["utc"] == 3.


@pytest.mark.parametrize("kind", ["cancel", "terminate"])
def test_earlier_matching_return_corrects_later_history_even_after_terminal_classification(kind):
    state, _ = reduce_event(empty_state(), event("out", 1., "r0", "c0"))
    state, _ = reduce_event(state, event("ingress", 5., "repeated-history", "c0"))
    state, _ = reduce_event(state, event(kind, 4., "lifecycle", ""))
    state, _ = reduce_event(state, event("ingress", 3., "r1", "c0"))
    record = next(iter(state["records"].values()))
    assert record["status"] == "returned" and record["valid"] and record["duration"] == 2.


def test_corrected_return_does_not_hide_proven_concurrent_calls():
    state = empty_state()
    for e in (event("out", 1., "r0", "c0"), event("out", 2., "r1", "c1"),
              event("ingress", 4., "r2", "c0"), event("ingress", 3., "r3", "c1"),
              event("ingress", 2.5, "early", "c0")):
        state, _ = reduce_event(state, e)
    assert not state["pending"]
    assert all(not r["valid"] and r["status"] == "concurrent_conflict" for r in state["records"].values())
