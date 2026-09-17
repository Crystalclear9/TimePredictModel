"""Validate the training contract before fitting any feature dictionary or model."""
import math
from collections import Counter

from .features import extract


def finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


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


def describe(rows):
    return {"rows": len(rows), "workflows": len({r["workflow"] for r in rows}),
            "observed": sum(math.isfinite(r["upper"]) for r in rows),
            "right_censored": sum(math.isinf(r["upper"]) for r in rows),
            "tools": dict(Counter(r["call"]["name"] for r in rows))}
