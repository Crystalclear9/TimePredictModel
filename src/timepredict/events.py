"""Validate untrusted stream envelopes before deterministic state reduction."""

import json
import math


def decode_event(raw):
    def invalid(value):
        raise ValueError("nonfinite JSON")

    def finite_float(value):
        number = float(value)
        if not math.isfinite(number):
            invalid(value)
        return number

    def unique_object(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError("duplicate JSON key")
            value[key] = item
        return value

    try:
        if (
            isinstance(raw, (str, bytes))
            and len(raw.encode("utf-8") if isinstance(raw, str) else raw) > 1024 * 1024
        ):
            raise ValueError("event exceeds parsing budget")
        event = json.loads(
            raw, parse_constant=invalid, parse_float=finite_float, object_pairs_hook=unique_object
        )
        # JSON escapes can conceal lone surrogates even in an ASCII envelope.
        # Verify nested values and keys before they reach history/provenance encoding.
        json.dumps(event, ensure_ascii=False, allow_nan=False).encode("utf-8")
        validate_event(event)
        return event
    except (KeyError, TypeError, OverflowError, RecursionError, ValueError) as exc:
        raise ValueError("invalid_event_envelope") from exc


def validate_event(event):
    if not isinstance(event, dict):
        raise ValueError("object required")
    kind = event["kind"]
    if kind not in {
        "prepared",
        "out",
        "ingress",
        "heartbeat",
        "cancel",
        "terminate",
        "expire",
        "source_heartbeat",
        "gap",
        "source_lost",
        "source_retired",
        "transport_gap",
        "transport_recovered",
    }:
        raise ValueError("unknown event kind")

    def text(value):
        if not isinstance(value, str) or not value or len(value) > 4096:
            raise ValueError("invalid identifier")
        value.encode("utf-8")

    def number(value):
        if type(value) not in (float, int) or not math.isfinite(value):
            raise ValueError("invalid number")

    text(event["event_id"])
    stamp = event["stamp"]
    for name in ("source", "boot"):
        text(stamp[name])
    for name in ("utc", "mono"):
        number(stamp[name])
    if stamp.get("error") is not None:
        number(stamp["error"])
        if stamp["error"] < 0:
            raise ValueError("negative clock error")
    if "jumped" in stamp and type(stamp["jumped"]) is not bool:
        raise ValueError("invalid jumped flag")
    if kind in {"source_retired", "transport_recovered"} and (
        stamp.get("jumped") or stamp.get("error") is None or not 0 <= stamp["error"] <= 0.0025
    ):
        raise ValueError("verified administrative clock required")
    if "seq" in event and (type(event["seq"]) is not int or event["seq"] <= 0):
        raise ValueError("invalid sequence")
    for flag in ("serial_protocol", "tool_call_ids_present"):
        if flag in event and type(event[flag]) is not bool:
            raise ValueError("invalid protocol flag")
    if event.get("reason") is not None:
        text(event["reason"])
        if kind in {"out", "prepared"} and event["reason"] in {
            "waiting",
            "returned",
            "prepared",
            "concurrent_conflict",
        }:
            raise ValueError("reason cannot impersonate lifecycle state")
    if "telemetry" in event and not isinstance(event["telemetry"], dict):
        raise ValueError("telemetry object required")
    if kind in {"source_heartbeat", "gap"} and "seq" not in event:
        raise ValueError("sequence required")
    if kind == "source_lost":
        text(event["lost_source"])
    elif kind == "source_retired":
        text(event["retired_source"])
        if type(event["expected_seq"]) is not int or event["routing_fenced"] is not True:
            raise ValueError("invalid retirement fence")
    elif kind in {"transport_gap", "transport_recovered"}:
        text(event["gap_id"])
        if kind == "transport_recovered" and event["observation_reestablished"] is not True:
            raise ValueError("recovery assertion required")
    elif kind not in {"source_heartbeat", "gap"}:
        for name in ("tenant", "workflow_id", "agent_id"):
            text(event[name])
        if kind in {"out", "prepared", "ingress"}:
            text(event["request_id"])
        if kind in {"out", "prepared"}:
            if not isinstance(event["call"], dict):
                raise ValueError("call object required")
            text(event["call"]["name"])
            if event["call"].get("id") is not None:
                text(event["call"]["id"])
        if kind == "ingress" and (
            not isinstance(event.get("latest_tool_ids", []), list)
            or any(not isinstance(i, str) for i in event.get("latest_tool_ids", []))
        ):
            raise ValueError("invalid tool IDs")
