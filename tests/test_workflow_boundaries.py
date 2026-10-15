"""Observed label boundaries must not be hidden by workflow metadata."""

import copy

import pytest

from timepredict.dataset import validate_label_provenance, validate_rows
from timepredict.training import snapshot, split_rows


def event(kind, t, request):
    return {
        "event_id": kind,
        "kind": kind,
        "tenant": "t",
        "workflow_id": "w",
        "agent_id": "a",
        "request_id": request,
        "stamp": {"utc": t, "mono": t, "source": "s", "boot": "b", "error": 0.001, "jumped": False},
        "source_kind": "sglang",
        "serial_protocol": True,
        "call": {"id": "c", "name": "search", "arguments": {}},
        "latest_tool_ids": ["c"],
    }


def observed_row():
    rows, _ = snapshot(
        [event("out", 1.0, "r1"), event("ingress", 10.0, "r2"), event("terminate", 11.0, "r2")],
        12.0,
    )
    return rows[0]


@pytest.mark.parametrize("field,value", [("workflow_start", 2.0), ("workflow_end", 2.0)])
@pytest.mark.parametrize("entry", ["rows", "provenance"])
def test_inconsistent_workflow_boundaries_are_rejected(field, value, entry):
    row = observed_row()
    row[field] = value
    with pytest.raises(ValueError, match="workflow.*boundar"):
        if entry == "rows":
            validate_rows([row])
        else:
            validate_label_provenance([row], data_kind="observed")


def test_split_extends_end_to_observed_return_even_with_small_utc_skew():
    rows = [
        {"workflow": str(i), "workflow_start": float(i), "workflow_end": i + 0.1} for i in range(10)
    ]
    # 70% boundary = 7.0. A small clock skew must not admit a label observed at it.
    rows[0].update(workflow_end=6.999, t_out=1.0, t_return=7.0)
    rows.append(copy.deepcopy(rows[0]))
    groups = split_rows(rows)
    assert all(r["workflow"] != "0" for r in groups["train"])
    assert len([r for r in groups["quarantine"] if r["workflow"] == "0"]) == 2


@pytest.mark.parametrize("field,value", [("workflow_start", 9.0), ("workflow_end", 5.0)])
def test_direct_split_rejects_conflicting_metadata_within_one_workflow(field, value):
    rows = [
        {"workflow": str(i), "workflow_start": float(i), "workflow_end": i + 0.1} for i in range(10)
    ]
    extra = {**rows[0], field: value}
    with pytest.raises(ValueError, match="inconsistent.*workflow"):
        split_rows([*rows, extra])


def test_small_end_clock_skew_is_accepted_but_not_used_to_shorten_split():
    row = observed_row()
    row["workflow_end"] = row["t_return"] - 0.001
    validate_label_provenance(validate_rows([row]), data_kind="observed")


def test_ended_workflow_is_not_administrative_censoring():
    events = [
        event("out", 1.0, "r1"),
        {**event("source_heartbeat", 2.0, "r1"), "seq": 1},
        event("heartbeat", 2.0, "r1"),
    ]
    rows, _ = snapshot(events, 2.5, cutoff_stamp=event("cutoff", 2.5, "r1")["stamp"])
    assert len(rows) == 1
    rows[0]["workflow_end"] = 2.5
    with pytest.raises(ValueError, match="censoring.*workflow"):
        validate_label_provenance(validate_rows(rows), data_kind="observed")
