# 文件作用：训练样本契约：先校验标签、工作流元数据、参数和来源证据，再允许拟合特征。
"""Validate the training contract before fitting any feature dictionary or model."""
import math
from collections import Counter

from .features import extract


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


# 正式 observed 标签必须携带工具 ID、边界与事件证据；代理/合成数据不能用于生产。
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
            if row.get("association_mode") != "tool_call_id" or row.get("tool_call_id_verified") is not True:
                raise ValueError(f"sample {i}: formal_label_requires_verified_tool_call_id")
            if not isinstance(row.get("t_out"), (int, float)) or not isinstance(row.get("t_return"), (int, float)):
                raise ValueError(f"sample {i}: formal_label_missing_boundaries")
            if row.get("source_manifest") is None or row.get("event_evidence") is None:
                raise ValueError(f"sample {i}: formal_label_missing_event_evidence")
        if production and any(row.get("association_mode") != "tool_call_id" for row in rows):
            raise ValueError("serial_fallback_not_allowed_for_production")
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
