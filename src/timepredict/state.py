# 文件作用：确定性事件状态机：业务键标识 Agent，generation 标识轮次，revision 标识可消费状态版本。
"""Deterministic event reducer. Redis owns serialization, never frontend processes."""
import copy
import hashlib
import json

from .clock import elapsed


# 将 tenant/workflow/agent 编码为稳定业务键。
def key_of(event):
    return json.dumps([event["tenant"], event["workflow_id"], event["agent_id"]], separators=(",", ":"))


# 建立记录、待回返索引、幂等表、来源、孤立入口、心跳与工作流状态。
def empty_state():
    return {"records": {}, "pending": {}, "seen": {}, "sources": {}, "orphans": {}, "revision": 0, "heartbeats": {}, "workflows": {}, "ingress_seen": {}, "cancelled_agents": {}, "tombstones": {}}


# 生成租户隔离的 workflow 键。
def workflow_key(e):
    return json.dumps([e["tenant"], e["workflow_id"]], separators=(",", ":"))


# 按 event_id 幂等归约事件，维护来源缺口、生命周期、回返关联和待回返状态。
def reduce_event(original, event, *, copy_state=True):
    state = copy.deepcopy(original) if copy_state else original
    # Redis 可能重复投递；幂等归约避免重复闭合与重复通知。
    if event["event_id"] in state["seen"]:
        return state, []
    state["seen"][event["event_id"]] = event["stamp"]["utc"]
    notices = []
    state.setdefault("tombstones", {})
    if event.get("observed_at", event["stamp"]["utc"]) - event["stamp"]["utc"] > 7*86400:
        return state, notices

    # 递增全局及记录 revision，收集更新或移除通知。
    def notice(record, action):
        state["revision"] += 1
        record["revision"] = state["revision"]
        notices.append({"action": action, "key": record["key"], "generation": record["generation"], "revision": state["revision"], "tenant": record["tenant"]})

    # 关闭记录并取消有效性；只有 pending 指向当前 generation 才删除索引。
    def invalidate(record, status, closed_stamp=None):
        closing = closed_stamp or event["stamp"]
        record["status"] = status
        record["valid"] = False
        record["closed_at"] = closing["utc"]
        record["closed_stamp"] = closing
        if state["pending"].get(record["key"]) == record["generation"]:
            del state["pending"][record["key"]]
        notice(record, "remove")

    def cancellation_for(record):
        admitted = state.get("ingress_seen", {}).get(json.dumps([record["key"], record["request_id"]]), -float("inf"))
        history = state.get("cancel_history", {}).get(record["key"], [])
        applicable = [s for s in history if s["utc"] >= admitted]
        if applicable:
            return min(applicable, key=lambda s: s["utc"])
        old = state.get("cancelled_agent_stamps", {}).get(record["key"])
        return old if old and old["utc"] >= admitted else None

    stamp = event["stamp"]
    if event["kind"] == "source_lost":
        source = state["sources"].get(event["lost_source"])
        if not source or source.get("retired") or source["last"]["utc"] != event.get("last_seen", source["last"]["utc"]):
            return state, notices
        if source:
            source["bad"], source["lost"] = True, True
        for record in list(state["records"].values()):
            out_stamp = record.get("out", {})
            if (record["status"] == "waiting" and
                    out_stamp.get("source", "") + "/" + out_stamp.get("boot", "") == event.get("lost_source")):
                invalidate(record, "observation_lost")
        return state, notices
    source_key = stamp["source"] + "/" + stamp["boot"]
    # 来源序号断层或进程重启意味着观察不完整，相关等待标签必须失效。
    if "seq" in event:
        source = state["sources"].get(source_key, {"seq": 0, "bad": False})
        gap = event["seq"] != source["seq"] + 1 or event["kind"] == "gap"
        if gap:
            source["bad"] = True
            for record in list(state["records"].values()):
                if record["status"] == "waiting" and record.get("out", {}).get("source", "") + "/" + record.get("out", {}).get("boot", "") == source_key:
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
                    if record["status"] == "waiting" and record["out"].get("source", "") + "/" + record["out"].get("boot", "") == old_key:
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
            state.setdefault("cancelled_agent_stamps", {})[key] = stamp
            state.setdefault("cancel_history", {}).setdefault(key, []).append(stamp)
        if event["kind"] == "terminate":
            workflow = state["workflows"][wf]
            if workflow["end"] is None or stamp["utc"] < workflow["end"]:
                workflow["end"], workflow["end_stamp"] = stamp["utc"], stamp
        for record in list(state["records"].values()):
            generation_matches = event["kind"] != "expire" or event.get("generation", record["generation"]) == record["generation"]
            eligible = record["status"] == "waiting" or (record["status"] == "concurrent_conflict" and not record.get("return")) or (record["status"] == "returned" and event["kind"] in ("cancel", "terminate"))
            if generation_matches and eligible and (record["key"] == key or (event["kind"] == "terminate" and workflow_key(record) == wf)):
                if event["kind"] == "cancel":
                    closing = cancellation_for(record)
                    if closing is None:
                        continue  # A delayed cancellation of an earlier admitted round.
                else:
                    closing = state["workflows"][wf].get("end_stamp", stamp) if event["kind"] == "terminate" else stamp
                if record["status"] == "returned":
                    try:
                        elapsed(closing, record["return"])
                    except ValueError as exc:
                        if str(exc) != "invalid_interval":
                            invalidate(record, "lifecycle_clock_uncertain", closing)
                        continue  # The proven return precedes this terminal event.
                invalidate(record, {"cancel": "cancelled", "terminate": "terminated", "expire": "observation_lost"}[event["kind"]], closing)
        return state, notices

    # 关联续接请求并计算时间标签；明确 ID 优先，无 ID 时允许串行降级且标记 degraded。
    def close(record, incoming):
        recovered = False
        if record["status"] in ("cancelled", "terminated") and record.get("closed_stamp"):
            try:
                # Event transport order cannot override a proven earlier ingress boundary.
                recovered = elapsed(incoming["stamp"], record["closed_stamp"]) > 0
            except ValueError:
                pass
        if (record["status"] not in ("waiting", "concurrent_conflict") and not recovered) or record.get("return") or incoming["request_id"] == record["request_id"]:
            return False
        # Under the serial protocol the tenant/workflow/agent key identifies
        # the only outstanding round. If the continuation includes tool IDs,
        # require the recorded ID to be present; older clients may omit the
        # list, in which case the serial invariant is the association proof.
        supplied = bool(incoming.get("tool_call_ids_present", bool(incoming.get("latest_tool_ids"))))
        incoming_ids = incoming.get("latest_tool_ids") or []
        if record.get("association_mismatch") and not supplied:
            return False
        if supplied and record["call"].get("id") not in incoming_ids:
            record["association_mismatch"] = {"request_id": incoming["request_id"], "tool_ids": incoming_ids,
                                               "at": incoming["stamp"]}
            record["status_reason"] = "historical_tool_result" if incoming.get("historical_tool_ids") else "association_mismatch"
            return False
        cancelled = cancellation_for(record)
        if cancelled is not None and cancelled["utc"] <= incoming["stamp"]["utc"]:
            invalidate(record, "cancelled", cancelled)
            return True
        ended = state["workflows"][wf]["end"]
        if ended is not None and ended <= incoming["stamp"]["utc"]:
            invalidate(record, "terminated", state["workflows"][wf].get("end_stamp"))
            return True
        # Old requests must never consume a newer pending generation.
        if incoming["stamp"]["utc"] + 0.005 < record["out"]["utc"]:
            return False
        record["return"] = incoming["stamp"]
        if recovered:
            record["valid"] = True
        record["return_event_id"] = incoming["event_id"]
        record["return_event"] = copy.deepcopy(incoming)
        record["return_source_kind"] = incoming.get("source_kind")
        record["return_request_id"] = incoming["request_id"]
        record["association_mode"] = "tool_call_id" if supplied else "serial_fallback"
        record["association_confidence"] = "verified" if supplied else "degraded"
        record["tool_call_id_verified"] = supplied
        record["serial_verified"] = bool(record.get("serial_protocol") and incoming.get("serial_protocol"))
        if not supplied and record["serial_verified"]:
            record["association_confidence"] = "verified_serial_protocol"
        record["closed_at"] = incoming["stamp"]["utc"]
        try:
            duration = elapsed(record["out"], incoming["stamp"])
            if duration <= 0:
                raise ValueError("nonpositive_interval")
            record["duration"] = duration
            record["status"] = "returned"
            record["lifecycle_state"] = "returned"
            record.setdefault("lifecycle", []).append("returned")
        except ValueError as exc:
            record["status"], record["valid"] = str(exc), False
        incoming_source = incoming["stamp"]["source"] + "/" + incoming["stamp"]["boot"]
        if state["sources"].get(incoming_source, {}).get("bad"):
            record["status"], record["valid"] = "observation_gap", False
        if state["pending"].get(key) == record["generation"]:
            state["pending"].pop(key, None)
        notice(record, "remove")
        return True

    # 重建同 Agent 各轮等待区间；重叠时冲突，延迟入口证明串行时按现有规则恢复。
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
                        # Retain the evidence pair so a corrected serial association can
                        # withdraw only conflicts derived from that provisional pairing.
                        prior = next((r for r in group if r.get("return") == end), group[-1])
                        cause = [prior["generation"], record["generation"]]
                        for member in [*group, record]:
                            member["conflict_confirmed"] = True
                            causes = member.setdefault("conflict_causes", [])
                            if cause not in causes:
                                causes.append(cause)
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

    def correct_association(incoming):
        # With no tool ID, transport delivery may present a later continuation first.
        # The earliest proven ingress after OUT is the serial successor; revise a
        # provisional later match and return it to the orphan pool for its own round.
        for record in state["records"].values():
            if (record["key"] != key or record.get("association_mode") not in ("serial_fallback", "tool_call_id")
                    or not record.get("return_event") or record["request_id"] == incoming["request_id"]
                    or record["status"] not in ("returned", "concurrent_conflict", "cancelled", "terminated")):
                continue
            supplied = incoming.get("tool_call_ids_present", bool(incoming.get("latest_tool_ids")))
            if record.get("association_mode") == "tool_call_id" and not supplied:
                continue
            if supplied and record["call"].get("id") not in (incoming.get("latest_tool_ids") or []):
                continue
            try:
                if elapsed(record["out"], incoming["stamp"]) <= 0 or elapsed(incoming["stamp"], record["return"]) <= 0:
                    continue
            except ValueError:
                continue
            state["orphans"].setdefault(key, []).append(record["return_event"])
            for field in ("return", "return_event", "return_event_id", "return_request_id", "return_source_kind",
                          "duration", "association_mode", "association_confidence", "tool_call_id_verified", "serial_verified"):
                record.pop(field, None)
            record["status"], record["valid"] = "waiting", True
            for member in state["records"].values():
                causes = member.get("conflict_causes")
                if causes is not None:
                    member["conflict_causes"] = [c for c in causes if record["generation"] not in c]
                    member["conflict_confirmed"] = bool(member["conflict_causes"])
            break

    kind = event["kind"]
    if kind == "ingress":
        request_key = json.dumps([key, event["request_id"]])
        ingress_stamps = state.setdefault("ingress_stamps", {})
        if request_key in state["ingress_seen"]:
            previous = ingress_stamps.get(request_key)
            if previous is None:
                return state, notices
            try:
                if elapsed(stamp, previous) <= 0:
                    return state, notices
            except ValueError:
                return state, notices
        state["ingress_seen"][request_key] = stamp["utc"]
        ingress_stamps[request_key] = stamp
        correct_association(event)
        candidates = sorted((r for r in state["records"].values() if r["key"] == key), key=lambda r: r["out"]["utc"], reverse=True)
        for record in candidates:
            if close(record, event):
                reconcile_intervals()
                leftovers = []
                queue = sorted(state["orphans"].pop(key, []), key=lambda e: e["stamp"]["utc"])
                while queue:
                    orphan = queue.pop(0)
                    correct_association(orphan)
                    choices = sorted((r for r in state["records"].values() if r["key"] == key),
                                     key=lambda r: r["out"]["utc"], reverse=True)
                    if any(close(r, orphan) for r in choices):
                        reconcile_intervals()
                    else:
                        leftovers.append(orphan)
                    queue.extend(state["orphans"].pop(key, []))
                    queue.sort(key=lambda e: e["stamp"]["utc"])
                if leftovers:
                    state["orphans"][key] = leftovers
                return state, notices
        # Buffer unmatched returns, including returns arriving before their OUT event.
        state["orphans"].setdefault(key, []).append(event)
        return state, notices
    # prepared 只是可观测发送准备状态，尚不能生成有效等待预测。
    if kind == "prepared":
        # Prepared is an observable lifecycle event before the response is sent.
        gen = hashlib.sha256((key + "/" + event["request_id"]).encode()).hexdigest()
        if gen not in state["records"] and gen not in state["tombstones"]:
            state["records"][gen] = {"key": key, "generation": gen, "tenant": event["tenant"], "workflow_id": event["workflow_id"],
                "agent_id": event["agent_id"], "request_id": event["request_id"], "call": event["call"], "out": stamp,
                "status": "prepared", "valid": False, "prediction": None, "prediction_status": "not_ready",
                "workflow_start": state["workflows"][wf]["start"], "lifecycle": ["prepared"], "lifecycle_state": "prepared"}
        return state, notices
    if kind == "out":
        gen = hashlib.sha256((key + "/" + event["request_id"]).encode()).hexdigest()
        prepared = None
        if gen in state["records"]:
            existing = state["records"][gen]
            if existing.get("status") == "prepared":
                prepared = existing
            else:
                return state, notices
        if gen in state["tombstones"]: return state, notices
        record = {"key": key, "generation": gen, "tenant": event["tenant"], "workflow_id": event["workflow_id"],
                  "agent_id": event["agent_id"], "request_id": event["request_id"], "call": event["call"],
                  "out": stamp, "status": "waiting", "valid": True, "prediction": None,
                  "prediction_status": "pending_prediction", "workflow_start": state["workflows"][wf]["start"],
                  "lifecycle": ["waiting"], "lifecycle_state": "waiting"}
        record.update(out_event_id=event["event_id"], source_kind=event.get("source_kind"),
                      serial_protocol=event.get("serial_protocol", False))
        if prepared:
            record["lifecycle"] = prepared["lifecycle"] + ["waiting"]
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
        cancelled = cancellation_for(record)
        if cancelled is not None:
            invalidate(record, "cancelled", cancelled)
            return state, notices
        if state["workflows"][wf]["end"] is not None:
            invalidate(record, "terminated", state["workflows"][wf].get("end_stamp"))
            return state, notices
        reconcile_intervals()
        if record["status"] == "waiting":
            state["pending"][key] = gen
            notice(record, "update")
    return state, notices


# 仅给仍 waiting 且 pending_prediction 的轮次写预测，防止迟到结果复活旧候选。
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


# 清理一小时以上闭合热记录，用七天 tombstone 保留幂等；训练历史流独立保留。
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
    for key in list(state.get("cancelled_agent_stamps", {})):
        if key not in state["cancelled_agents"]:
            del state["cancelled_agent_stamps"][key]
    for key in list(state.get("ingress_stamps", {})):
        if key not in state["ingress_seen"]:
            del state["ingress_stamps"][key]
    for key, stamps in list(state.get("cancel_history", {}).items()):
        recent = [s for s in stamps if now-s["utc"] <= 7*86400]
        if recent:
            state["cancel_history"][key] = recent
        else:
            del state["cancel_history"][key]
    return state, []
