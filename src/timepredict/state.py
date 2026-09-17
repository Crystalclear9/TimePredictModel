"""Deterministic event reducer. Redis owns serialization, never frontend processes."""
import copy
import hashlib
import json

from .clock import elapsed


def key_of(event):
    return json.dumps([event["tenant"], event["workflow_id"], event["agent_id"]], separators=(",", ":"))


def empty_state():
    return {"records": {}, "pending": {}, "seen": {}, "sources": {}, "orphans": {}, "revision": 0, "heartbeats": {}, "workflows": {}, "ingress_seen": {}, "cancelled_agents": {}, "tombstones": {}}


def workflow_key(e):
    return json.dumps([e["tenant"], e["workflow_id"]], separators=(",", ":"))


def reduce_event(original, event, *, copy_state=True):
    state = copy.deepcopy(original) if copy_state else original
    if event["event_id"] in state["seen"]:
        return state, []
    state["seen"][event["event_id"]] = event["stamp"]["utc"]
    notices = []
    state.setdefault("tombstones", {})
    if event.get("observed_at", event["stamp"]["utc"]) - event["stamp"]["utc"] > 7*86400:
        return state, notices

    def notice(record, action):
        state["revision"] += 1
        record["revision"] = state["revision"]
        notices.append({"action": action, "key": record["key"], "generation": record["generation"], "revision": state["revision"], "tenant": record["tenant"]})

    def invalidate(record, status):
        record["status"] = status
        record["valid"] = False
        record["closed_at"] = event["stamp"]["utc"]
        if state["pending"].get(record["key"]) == record["generation"]:
            del state["pending"][record["key"]]
        notice(record, "remove")

    stamp = event["stamp"]
    if event["kind"] == "source_lost":
        source = state["sources"].get(event["lost_source"])
        if not source or source.get("retired") or source["last"]["utc"] != event.get("last_seen", source["last"]["utc"]):
            return state, notices
        if source:
            source["bad"], source["lost"] = True, True
        for record in list(state["records"].values()):
            if record["status"] == "waiting":
                invalidate(record, "observation_lost")
        return state, notices
    source_key = stamp["source"] + "/" + stamp["boot"]
    if "seq" in event:
        source = state["sources"].get(source_key, {"seq": 0, "bad": False})
        gap = event["seq"] != source["seq"] + 1 or event["kind"] == "gap"
        if gap:
            source["bad"] = True
            for record in list(state["records"].values()):
                if record["status"] == "waiting":
                    # A missing ingress event could have closed any active record.
                    invalidate(record, "observation_gap")
        source["seq"] = max(source["seq"], event["seq"])
        source["last"] = stamp
        if "telemetry" in event:
            source["telemetry"] = event["telemetry"]
        state["sources"][source_key] = source
        for old_key, old in state["sources"].items():
            if old_key.startswith(stamp["source"] + "/") and old_key != source_key and not old.get("retired"):
                old["retired"] = True
                for record in list(state["records"].values()):
                    if record["status"] == "waiting" and record["out"]["source"] == stamp["source"]:
                        invalidate(record, "source_restart")
    if event["kind"] in ("source_heartbeat", "gap"):
        return state, notices

    key = key_of(event)
    state.setdefault("ingress_seen", {})
    state.setdefault("cancelled_agents", {})
    wf = workflow_key(event)
    state["workflows"].setdefault(wf, {"start": stamp["utc"], "end": None})
    state["workflows"][wf]["start"] = min(state["workflows"][wf]["start"], stamp["utc"])
    if event["kind"] == "heartbeat":
        state["heartbeats"][key] = stamp["utc"]
        state.setdefault("heartbeat_stamps", {})[key] = stamp
        return state, notices
    if event["kind"] in ("cancel", "terminate", "expire"):
        if event["kind"] == "cancel":
            state["cancelled_agents"][key] = stamp["utc"]
        if event["kind"] == "terminate":
            state["workflows"][wf]["end"] = stamp["utc"]
        for record in list(state["records"].values()):
            generation_matches = event["kind"] != "expire" or event.get("generation", record["generation"]) == record["generation"]
            if generation_matches and (record["status"] == "waiting" or (record["status"] == "concurrent_conflict" and not record.get("return"))) and (record["key"] == key or (event["kind"] == "terminate" and workflow_key(record) == wf)):
                invalidate(record, {"cancel": "cancelled", "terminate": "terminated", "expire": "observation_lost"}[event["kind"]])
        return state, notices

    def close(record, incoming):
        if record["status"] not in ("waiting", "concurrent_conflict") or record.get("return") or incoming["request_id"] == record["request_id"]:
            return False
        if record["call"].get("id") not in incoming.get("latest_tool_ids", []):
            return False
        cancelled = state["cancelled_agents"].get(key)
        admitted = state["ingress_seen"].get(json.dumps([key, record["request_id"]]), -float("inf"))
        if cancelled is not None and admitted <= cancelled <= incoming["stamp"]["utc"]:
            invalidate(record, "cancelled")
            return True
        ended = state["workflows"][wf]["end"]
        if ended is not None and ended <= incoming["stamp"]["utc"]:
            invalidate(record, "terminated")
            return True
        # Old requests must never consume a newer pending generation.
        if incoming["stamp"]["utc"] + 0.005 < record["out"]["utc"]:
            return False
        record["return"] = incoming["stamp"]
        record["return_request_id"] = incoming["request_id"]
        record["closed_at"] = incoming["stamp"]["utc"]
        try:
            duration = elapsed(record["out"], incoming["stamp"])
            if duration <= 0:
                raise ValueError("nonpositive_interval")
            record["duration"] = duration
            record["status"] = "returned"
        except ValueError as exc:
            record["status"], record["valid"] = str(exc), False
        incoming_source = incoming["stamp"]["source"] + "/" + incoming["stamp"]["boot"]
        if state["sources"].get(incoming_source, {}).get("bad"):
            record["status"], record["valid"] = "observation_gap", False
        if state["pending"].get(key) == record["generation"]:
            state["pending"].pop(key, None)
        notice(record, "remove")
        return True

    def reconcile_intervals():
        # Conflicts stop predictions immediately, but delayed ingress can prove a
        # seemingly overlapping interval was actually serial. Preserve both OUTs.
        records = sorted((r for r in state["records"].values() if r["key"] == key and
                          r["status"] in ("waiting", "returned", "concurrent_conflict")), key=lambda r: r["out"]["utc"])
        groups, group, end = [], [], None
        for record in records:
            separate = False
            if group and end is not None:
                try:
                    separate = elapsed(end, record["out"]) >= 0
                except ValueError as exc:
                    if str(exc) == "invalid_interval":
                        for member in [*group, record]:
                            member["conflict_confirmed"] = True
            if separate:
                groups.append(group)
                group = []
            if not group:
                end = record.get("return")
            elif end is not None:
                returned = record.get("return")
                end = max((end, returned), key=lambda s: s["utc"]) if returned else None
            group.append(record)
        if group:
            groups.append(group)
        for group in groups:
            if len(group) > 1:
                for record in group:
                    if record["status"] != "concurrent_conflict":
                        invalidate(record, "concurrent_conflict")
            else:
                record = group[0]
                if record.get("conflict_confirmed"):
                    if record["status"] != "concurrent_conflict":
                        invalidate(record, "concurrent_conflict")
                elif record["status"] == "concurrent_conflict":
                    record["valid"] = True
                    record["status"] = "returned" if record.get("return") else "waiting"
                    if record["status"] == "waiting":
                        record.pop("closed_at", None)
                        state["pending"][key] = record["generation"]
                        notice(record, "update")
                    else:
                        notice(record, "remove")
                elif record["status"] == "returned":
                    record["valid"] = True

    kind = event["kind"]
    if kind == "ingress":
        request_key = json.dumps([key, event["request_id"]])
        if request_key in state["ingress_seen"]:
            return state, notices
        state["ingress_seen"][request_key] = stamp["utc"]
        candidates = sorted((r for r in state["records"].values() if r["key"] == key), key=lambda r: r["out"]["utc"], reverse=True)
        for record in candidates:
            if close(record, event):
                reconcile_intervals()
                return state, notices
        # Buffer unmatched returns, including returns arriving before their OUT event.
        if event.get("latest_tool_ids"):
            state["orphans"].setdefault(key, []).append(event)
        return state, notices
    if kind == "out":
        gen = hashlib.sha256((key + "/" + event["request_id"]).encode()).hexdigest()
        if gen in state["records"] or gen in state["tombstones"]:
            return state, notices
        record = {"key": key, "generation": gen, "tenant": event["tenant"], "workflow_id": event["workflow_id"],
                  "agent_id": event["agent_id"], "request_id": event["request_id"], "call": event["call"],
                  "out": stamp, "status": "waiting", "valid": True, "prediction": None,
                  "prediction_status": "pending_prediction", "workflow_start": state["workflows"][wf]["start"]}
        state["records"][gen] = record
        if state["sources"].get(source_key, {}).get("bad"):
            invalidate(record, "observation_gap")
            return state, notices
        if event.get("reason"):
            invalidate(record, event["reason"])
            return state, notices
        # An older OUT may arrive after a successor OUT from another ingress.
        # Reconcile known returns before deciding whether waiting intervals overlap.
        orphans = state["orphans"].pop(key, [])
        unmatched = []
        for incoming in sorted(orphans, key=lambda e: e["stamp"]["utc"]):
            if not close(record, incoming):
                unmatched.append(incoming)
        if unmatched:
            state["orphans"][key] = unmatched
        if record["status"] != "waiting":
            reconcile_intervals()
            return state, notices
        cancelled = state["cancelled_agents"].get(key)
        admitted = state["ingress_seen"].get(json.dumps([key, event["request_id"]]), -float("inf"))
        if cancelled is not None and admitted <= cancelled:
            invalidate(record, "cancelled")
            return state, notices
        if state["workflows"][wf]["end"] is not None:
            invalidate(record, "terminated")
            return state, notices
        reconcile_intervals()
        if record["status"] == "waiting":
            state["pending"][key] = gen
            notice(record, "update")
    return state, notices


def apply_prediction(state, generation, prediction, status, available_at=None):
    record = state["records"].get(generation)
    if not record or record["status"] != "waiting" or record["prediction_status"] != "pending_prediction":
        return []
    record["prediction"], record["prediction_status"] = prediction, status
    if available_at is not None:
        record["prediction_available_at"] = available_at
    state["revision"] += 1
    record["revision"] = state["revision"]
    return [{"action": "update", "tenant": record["tenant"], "key": record["key"], "generation": generation, "revision": state["revision"]}]


def compact(state, now):
    """Bound online history; the immutable Redis history stream remains the training source."""
    state.setdefault("tombstones", {})
    for gen, record in list(state["records"].items()):
        if record["status"] != "waiting" and now-record.get("closed_at", record["out"]["utc"]) > 3600:
            state["tombstones"][gen] = record["out"]["utc"]
            del state["records"][gen]
    for name in ("seen", "tombstones", "ingress_seen", "cancelled_agents"):
        for key, timestamp in list(state.get(name, {}).items()):
            if now-timestamp > 7*86400:
                del state[name][key]
    for key, events in list(state["orphans"].items()):
        retained = [e for e in events if now-e["stamp"]["utc"] <= 7*86400]
        if retained:
            state["orphans"][key] = retained
        else:
            del state["orphans"][key]
    active_workflows = {workflow_key(state["records"][g]) for g in state["pending"].values()}
    for key, workflow in list(state["workflows"].items()):
        if key not in active_workflows and now-(workflow["end"] or workflow["start"]) > 7*86400:
            del state["workflows"][key]
    for key, timestamp in list(state["heartbeats"].items()):
        if now-timestamp > 7*86400:
            del state["heartbeats"][key]
            state.get("heartbeat_stamps", {}).pop(key, None)
    return state, []
