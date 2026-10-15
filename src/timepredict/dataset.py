# 文件作用：训练样本契约：先校验标签、工作流元数据、参数和来源证据，再允许拟合特征。
"""Validate the training contract before fitting any feature dictionary or model."""
import math
import hashlib
import json
from collections import Counter

from .features import extract
from .clock import elapsed


# 判断有限数值，并排除 Python 中也属于 int 的布尔值。
def finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


# 复制并规范化样本；只接受精确标签 [T,T] 或右删失 [a,null]，拒绝重复 generation。
def validate_rows(rows):
    if not rows:
        raise ValueError("dataset is empty")
    generations, workflows, normalized = set(), {}, []
    for index, original in enumerate(rows, 1):
        try:
            if not isinstance(original, dict):
                raise ValueError("sample must be an object")
            row = dict(original)
            for name in ("generation", "workflow"):
                if not isinstance(row.get(name), str) or not row[name]:
                    raise ValueError(name+" must be a nonempty string")
            if row["generation"] in generations:
                raise ValueError("duplicate generation")
            generations.add(row["generation"])
            start, end = row["workflow_start"], row["workflow_end"]
            if not finite_number(start) or (end is not None and (not finite_number(end) or end < start)):
                raise ValueError("invalid workflow UTC timestamps")
            metadata = (start, end)
            previous = workflows.setdefault(row["workflow"], metadata)
            if previous != metadata:
                raise ValueError("inconsistent timestamps within a workflow")
            lower, upper = row["lower"], row["upper"]
            if upper is None:
                upper = math.inf
            if not finite_number(lower) or lower <= 0:
                raise ValueError("lower must be positive finite seconds")
            if not ((finite_number(upper) and upper == lower) or (isinstance(upper, float) and upper == math.inf)):
                raise ValueError("only exact [T,T] or right-censored [a,null] labels are supported")
            row["upper"] = upper
            call = row["call"]
            if not isinstance(call, dict) or not isinstance(call.get("name"), str) or not call["name"]:
                raise ValueError("call.name must be a nonempty string")
            if not isinstance(call.get("schema", {}), dict):
                raise ValueError("call.schema must be an object")
            extract(call)
            normalized.append(row)
        except (KeyError, TypeError, ValueError, AttributeError, OverflowError) as exc:
            raise ValueError(f"sample {index}: {exc}") from exc
    return normalized


# 正式标签必须携带串行协议或工具 ID 关联证据，以及真实边界；代理/合成数据不能用于生产。
def validate_label_provenance(rows, *, data_kind, production=False):
    """Enforce the source evidence contract before any fitting.

    A caller cannot turn arbitrary JSON into a production observation by
    passing ``data_kind=observed``.  Formal SGLang rows carry both boundaries,
    a verified tool id, and an event/source manifest.  Replay and proxy rows
    remain explicitly separate experiments.
    """
    if data_kind == "observed":
        required = "sglang_return_observed"
        for i, row in enumerate(rows, 1):
            if row.get("label_kind") != required:
                raise ValueError(f"sample {i}: observed_requires_sglang_return_observed")
            mode = row.get("association_mode")
            exact = math.isfinite(row["upper"])
            verified = (mode == "tool_call_id" and row.get("tool_call_id_verified") is True) or (
                mode in ("serial_fallback", "pending_serial") and row.get("serial_verified") is True)
            if not verified or (exact and mode == "pending_serial"):
                raise ValueError(f"sample {i}: formal_label_requires_verified_association")
            if not finite_number(row.get("t_out")) or (exact and not finite_number(row.get("t_return"))):
                raise ValueError(f"sample {i}: formal_label_missing_boundaries")
            manifest, evidence = row.get("source_manifest"), row.get("event_evidence")
            if not isinstance(manifest, dict) or not isinstance(evidence, dict):
                raise ValueError(f"sample {i}: formal_label_missing_event_evidence")
            digest = hashlib.sha256(json.dumps(evidence, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()
            out, incoming = evidence.get("out"), evidence.get("return")
            if manifest.get("kind") != "sglang" or manifest.get("evidence_sha256") != digest or not isinstance(out, dict):
                raise ValueError(f"sample {i}: formal_evidence_mismatch")
            if out.get("kind") != "out" or out.get("source_kind") != "sglang" or out.get("reason"):
                raise ValueError(f"sample {i}: formal_out_invalid")
            if out.get("call") != row["call"] or out.get("stamp") != row.get("out_stamp") or out["stamp"]["utc"] != row["t_out"]:
                raise ValueError(f"sample {i}: formal_call_or_boundary_mismatch")
            identity = [out.get(k) for k in ("tenant", "workflow_id", "agent_id")]
            expected_generation = hashlib.sha256((json.dumps(identity, separators=(",", ":")) + "/" + out["request_id"]).encode()).hexdigest()
            if expected_generation != row["generation"] or row["workflow"] != json.dumps(identity[:2], separators=(",", ":")):
                raise ValueError(f"sample {i}: formal_identity_mismatch")
            if exact:
                if not isinstance(incoming, dict) or incoming.get("kind") != "ingress" or incoming.get("source_kind") != "sglang":
                    raise ValueError(f"sample {i}: formal_return_invalid")
                if any(incoming.get(k) != out.get(k) for k in ("tenant", "workflow_id", "agent_id")) or incoming.get("request_id") == out.get("request_id"):
                    raise ValueError(f"sample {i}: formal_association_mismatch")
                if mode == "tool_call_id" and (not out["call"].get("id") or out["call"]["id"] not in incoming.get("latest_tool_ids", [])):
                    raise ValueError(f"sample {i}: formal_tool_id_mismatch")
                if mode == "serial_fallback" and not (out.get("serial_protocol") and incoming.get("serial_protocol")):
                    raise ValueError(f"sample {i}: formal_serial_protocol_missing")
                if incoming.get("stamp") != row.get("return_stamp") or incoming["stamp"]["utc"] != row["t_return"] or not math.isclose(elapsed(out["stamp"], incoming["stamp"]), row["lower"], rel_tol=1e-9, abs_tol=1e-9):
                    raise ValueError(f"sample {i}: formal_duration_mismatch")
            else:
                if incoming is not None or row.get("t_return") is not None or not out.get("serial_protocol"):
                    raise ValueError(f"sample {i}: invalid_censoring_boundary")
                cutoff = row.get("observation_cutoff")
                if not finite_number(cutoff) or not math.isclose(cutoff-row["t_out"], row["lower"], abs_tol=1e-9):
                    raise ValueError(f"sample {i}: invalid_observation_cutoff")
                if not evidence.get("observation"):
                    raise ValueError(f"sample {i}: missing_censoring_observation_evidence")
                ttl = row.get("observation_ttl_seconds")
                if not finite_number(ttl) or ttl <= 0:
                    raise ValueError(f"sample {i}: invalid_observation_ttl")
                observations = evidence["observation"]
                if not isinstance(observations, list):
                    raise ValueError(f"sample {i}: invalid_observation_evidence")
                def fresh(e):
                    s = e.get("stamp", {})
                    error = s.get("error")
                    return (finite_number(s.get("utc")) and 0 <= cutoff-s["utc"] <= ttl
                            and finite_number(error) and 0 <= error <= .0025 and not s.get("jumped")
                            and e.get("observed_at", s["utc"]) <= cutoff)
                source_seen = any(isinstance(e, dict) and e.get("kind") == "source_heartbeat" and fresh(e)
                                  and all(e["stamp"].get(k) == out["stamp"].get(k) for k in ("source", "boot"))
                                  for e in observations)
                agent_seen = any(isinstance(e, dict) and e.get("kind") == "heartbeat" and fresh(e)
                                 and all(e.get(k) == out.get(k) for k in ("tenant", "workflow_id", "agent_id"))
                                 for e in observations)
                if not source_seen or not agent_seen:
                    raise ValueError(f"sample {i}: censoring_requires_matching_fresh_heartbeats")
                for boundary in (out["stamp"], evidence.get("cutoff_stamp", {})):
                    if not isinstance(boundary, dict):
                        raise ValueError(f"sample {i}: censoring_clock_unverified")
                    error = boundary.get("error")
                    if not finite_number(error) or not 0 <= error <= .0025 or boundary.get("jumped"):
                        raise ValueError(f"sample {i}: censoring_clock_unverified")
                if evidence["cutoff_stamp"].get("utc") != cutoff:
                    raise ValueError(f"sample {i}: censoring_cutoff_mismatch")
    elif data_kind in ("proxy", "synthetic", "synthetic_replay", "proxy_dataset"):
        if production:
            raise ValueError("proxy_or_synthetic_data_not_production_releasable")
    elif data_kind in ("unspecified", None):
        if production:
            raise ValueError("data_kind_required_for_production")
    else:
        raise ValueError("invalid_data_kind")


# 统计样本、工作流、精确/删失标签及工具分布。
def describe(rows):
    return {"rows": len(rows), "workflows": len({r["workflow"] for r in rows}),
            "observed": sum(math.isfinite(r["upper"]) for r in rows),
            "right_censored": sum(math.isinf(r["upper"]) for r in rows),
            "tools": dict(Counter(r["call"]["name"] for r in rows))}
