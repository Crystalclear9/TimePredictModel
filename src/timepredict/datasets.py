"""Small Hugging Face adapters for controlled proxy-label experiments.

The public traces do not expose the SGLang response-send and agent-return
boundaries, so every emitted sample is explicitly marked proxy_tool_duration.
"""
import json
import hashlib
import math
import ast
from pathlib import Path
from datetime import datetime


DATASETS = {
    "exgentic": ("Exgentic/agent-llm-traces", "70036b93a04e61b0ea2706a68b962f4f26774587"),
    "agenttrace": ("pagarsky/agent-trace", "4b05b2f00eea267a5bb4d841c228059d1bf9ac0c"),
    # Claude Code sessions with timestamped tool_use/tool_result events.  The
    # interval is a useful proxy for agent-visible return latency, but does
    # not prove the SGLang t_out boundary.
    "tracecommons": ("trace-commons/agent-traces", "112ebd4d03ce852b00e935d523107c3d0c9a65bf"),
    "betterwright": ("ProCreations/betterwright-agent-traces", "7e81937f8d968b61fafeb942cfc1f6079a813807"),
}
CONVERTER_VERSION = "2026-09-22.1"


def write_local_manifest(cache_dir, name):
    """Create the required manifest for an explicitly downloaded Parquet cache."""
    if name not in DATASETS:
        raise ValueError(f"unknown dataset: {name}")
    root = Path(cache_dir) / name
    files = sorted(root.glob("*.parquet"))
    if not files:
        raise ValueError(f"no parquet files in {root}")
    manifest = {"dataset": name, "revision": DATASETS[name][1], "converter_version": CONVERTER_VERSION,
                "files": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in files}}
    (root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def _json(value, default):
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (ValueError, TypeError):
            return default
    return value if value is not None else default


def _unix(value):
    if isinstance(value, (int, float)) and math.isfinite(value):
        return float(value) / (1e9 if value > 1e12 else 1.0)
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()
    except (ValueError, TypeError, OverflowError):
        return None


def _call(name, arguments, schema=None):
    if not isinstance(name, str) or not name:
        name = "unknown_tool"
    if isinstance(arguments, dict):
        arguments = json.dumps(arguments, ensure_ascii=False, separators=(",", ":"))
    elif not isinstance(arguments, str):
        raise ValueError("unsupported_input")
    try:
        parsed = json.loads(arguments)
        if not isinstance(parsed, dict):
            raise ValueError("arguments_not_object")
    except (ValueError, TypeError):
        # AgentTrace records Python-style tool wrappers such as
        # ``kwargs={'query': 'x'}``. Parse these explicitly and preserve the
        # conversion provenance instead of silently replacing the input.
        try:
            candidate = arguments.split("=", 1)[1] if "=" in arguments else arguments
            parsed = ast.literal_eval(candidate)
            if not isinstance(parsed, dict):
                raise ValueError("arguments_not_object")
            if set(parsed) == {"kwargs"} and isinstance(parsed["kwargs"], dict):
                parsed = parsed["kwargs"]
            arguments = json.dumps(parsed, ensure_ascii=False, separators=(",", ":"))
        except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
            raise ValueError("unsupported_input")
    return {"name": name, "arguments": arguments, "schema": schema if isinstance(schema, dict) else {}, "description": ""}


def _exgentic(row, prefix):
    # Exgentic does not prove the SGLang send-to-Agent-return interval. It is
    # intentionally excluded from time training; use _exgentic_calls for
    # semantic coverage only.
    return []


def _exgentic_calls(row, prefix):
    """Extract tool call descriptors without inventing a duration label."""
    spans = _json(row.get("spans"), [])
    if not isinstance(spans, list):
        return []
    calls_out = []
    for index, span in enumerate(spans):
        if not isinstance(span, dict):
            continue
        attrs = span.get("attributes") if isinstance(span.get("attributes"), dict) else {}
        outputs = _json(attrs.get("gen_ai.output.messages"), [])
        definitions = _json(attrs.get("gen_ai.tool.definitions"), [])
        definitions = definitions if isinstance(definitions, list) else []
        for call_index, call in enumerate(part for message in outputs if isinstance(message, dict)
                                          for part in (message.get("parts") or [])
                                          if isinstance(part, dict) and part.get("type") == "tool_call"):
            name = call.get("name")
            schema = next((d.get("parameters", {}) for d in definitions
                           if isinstance(d, dict) and d.get("name") == name), {})
            try:
                normalized = _call(name, call.get("arguments"), schema)
            except ValueError:
                continue
            normalized["description"] = next((d.get("description", "") for d in definitions
                                               if isinstance(d, dict) and d.get("name") == name), "")
            calls_out.append({"generation": f"{prefix}:{row.get('session_id', 'trace')}:{span.get('span_id', index)}:{call_index}",
                              "call": normalized, "source_name": name})
    return calls_out


def _agenttrace(row, prefix):
    spans = _json(row.get("spans_json"), [])
    if not isinstance(spans, list):
        return []
    base = _unix(row.get("timestamp_utc"))
    total = float(row.get("total_duration_ms", 0.0) or 0.0) / 1000.0
    out = []
    for index, span in enumerate(spans):
        if not isinstance(span, dict) or str(span.get("type", "")).upper() != "TOOL":
            continue
        duration = float(span.get("duration_ms", 0.0)) / 1000.0
        end_ns = span.get("end_ns")
        start_ns = span.get("start_ns")
        if end_ns is None or start_ns is None:
            continue
        try:
            boundary_duration = (float(end_ns) - float(start_ns)) / 1e9
        except (TypeError, ValueError):
            continue
        if not math.isfinite(boundary_duration) or boundary_duration <= 0 or abs(boundary_duration - duration) > max(1e-6, duration * 0.02):
            continue
        start = float(start_ns) / 1e9
        if start is None or not math.isfinite(duration) or duration <= 0:
            continue
        name = span.get("tool_name", "tool")
        args = span.get("tool_input")
        if args is None:
            continue
        try:
            normalized_call = _call(name, args)
        except ValueError:
            continue
        out.append({"generation": f"{prefix}:{row.get('trace_id', 'trace')}:{span.get('span_id', index)}",
                    "workflow": f"{prefix}:{row.get('trace_id', 'trace')}", "workflow_start": base or start,
                    "workflow_end": (base + total) if base is not None and total > 0 else start + duration, "call": normalized_call, "lower": duration,
                    "upper": duration, "label_kind": "proxy_tool_duration"})
    return out


def _tracecommons(row, prefix):
    """Pair timestamped tool_use/tool_result events by their stable id.

    The parquet schema stores each trace event as a JSON string.  We never
    infer a duration from adjacent events: both endpoints and the matching id
    are required, and malformed or duplicate pairs are excluded.
    """
    raw_trace = row.get("trace", [])
    if not isinstance(raw_trace, list):
        return []
    tools = row.get("tools", [])
    if isinstance(tools, str):
        tools = _json(tools, [])
    schemas = {}
    if isinstance(tools, list):
        for item in tools:
            if not isinstance(item, dict):
                continue
            fn = item.get("function", item)
            if isinstance(fn, dict) and isinstance(fn.get("name"), str):
                schemas[fn["name"]] = {
                    "schema": fn.get("parameters", {}) if isinstance(fn.get("parameters"), dict) else {},
                    "description": fn.get("description", "") if isinstance(fn.get("description", ""), str) else "",
                }
    uses, results = {}, {}
    timestamps = []
    for raw in raw_trace:
        event = raw
        if isinstance(raw, str):
            try:
                event = json.loads(raw)
            except (ValueError, TypeError):
                continue
        if not isinstance(event, dict):
            continue
        stamp = _unix(event.get("timestamp"))
        if stamp is not None:
            timestamps.append(stamp)
        message = event.get("message") if isinstance(event.get("message"), dict) else {}
        content = message.get("content", []) if isinstance(message, dict) else []
        if not isinstance(content, list):
            continue
        for item in content:
            if not isinstance(item, dict):
                continue
            kind = item.get("type")
            if kind == "tool_use":
                ident = item.get("id")
                if ident and stamp is not None and ident not in uses:
                    uses[ident] = (stamp, item.get("name"), item.get("input"))
            elif kind == "tool_result":
                ident = item.get("tool_use_id") or item.get("toolUseID")
                if ident and stamp is not None and ident not in results:
                    results[ident] = stamp
    session = str(row.get("session_id") or row.get("sessionId") or "session")
    workflow_start = min(timestamps) if timestamps else None
    workflow_end = max(timestamps) if timestamps else workflow_start
    out = []
    for index, (ident, (start, name, arguments)) in enumerate(uses.items()):
        end = results.get(ident)
        if end is None or not math.isfinite(start) or not math.isfinite(end) or end <= start:
            continue
        try:
            normalized = _call(name, arguments, schemas.get(name, {}).get("schema", {}))
        except (ValueError, TypeError):
            continue
        if name in schemas:
            normalized["description"] = schemas[name]["description"]
        out.append({
            "generation": f"{prefix}:{session}:{ident}",
            "workflow": f"{prefix}:{session}",
            "workflow_start": workflow_start or start,
            "workflow_end": workflow_end or end,
            "call": normalized,
            "lower": end - start,
            "upper": end - start,
            "label_kind": "proxy_agent_tool_return",
            "source_name": name,
        })
    return out


def _betterwright(row, prefix):
    """Extract per-turn latency telemetry from Betterwright.

    The source exposes model-turn latency, not SGLang send/return timestamps;
    therefore rows are explicitly marked ``proxy_model_turn_latency``.
    """
    # Betterwright records browser telemetry per step.  These are the most
    # direct executable-tool durations in this corpus, so prefer them over
    # the surrounding model-turn latency when present.
    browser_steps = row.get("browser_steps", [])
    if isinstance(browser_steps, list) and browser_steps:
        finish = _unix(row.get("finished_at"))
        wall = float(row.get("wall_seconds") or 0.0)
        start = finish - wall if finish is not None and math.isfinite(wall) else None
        session = str(row.get("task_id") or row.get("id") or "task")
        out = []
        for index, step in enumerate(browser_steps):
            if not isinstance(step, dict):
                continue
            try:
                duration = float(step.get("ms")) / 1000.0
            except (TypeError, ValueError):
                continue
            if not math.isfinite(duration) or duration <= 0:
                continue
            args = {"url": step.get("url", ""), "note": step.get("note", ""), "step": step.get("step", index)}
            normalized = _call("browser_step", args)
            out.append({"generation": f"{prefix}:{session}:browser:{index}",
                        "workflow": f"{prefix}:{session}", "workflow_start": start or float(index + 1),
                        "workflow_end": finish or (start or 0.0) + duration, "call": normalized,
                        "lower": duration, "upper": duration, "label_kind": "proxy_browser_step_duration",
                        "source_name": "browser_step"})
        if out:
            return out
    messages = row.get("messages", [])
    turns = row.get("turns", [])
    if not isinstance(messages, list) or not isinstance(turns, list):
        return []
    finish = _unix(row.get("finished_at"))
    wall = float(row.get("wall_seconds") or 0.0)
    start = finish - wall if finish is not None and math.isfinite(wall) else None
    out, assistant_index = [], 0
    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        calls = message.get("tool_calls") or []
        if not calls:
            continue
        turn = turns[assistant_index] if assistant_index < len(turns) and isinstance(turns[assistant_index], dict) else {}
        assistant_index += 1
        latency = turn.get("latency_ms")
        try:
            duration = float(latency) / 1000.0
        except (TypeError, ValueError):
            continue
        if not math.isfinite(duration) or duration <= 0:
            continue
        for index, call in enumerate(calls):
            if not isinstance(call, dict):
                continue
            name = call.get("name") or "unknown_tool"
            args = call.get("arguments", {})
            try:
                normalized = _call(name, args)
            except (ValueError, TypeError):
                continue
            session = str(row.get("task_id") or row.get("id") or "task")
            out.append({"generation": f"{prefix}:{session}:{assistant_index}:{index}",
                        "workflow": f"{prefix}:{session}", "workflow_start": start or float(assistant_index),
                        "workflow_end": finish or (start or 0.0) + duration, "call": normalized,
                        "lower": duration, "upper": duration, "label_kind": "proxy_model_turn_latency",
                        "source_name": name})
    return out


def load_proxy_rows(name, cache_dir=None, max_rows=None):
    """Stream a pinned dataset and return normalized proxy rows."""
    if name not in DATASETS:
        raise ValueError(f"unknown dataset: {name}")
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("install experiment dependencies with: pip install datasets pyarrow") from exc
    repo, revision = DATASETS[name]
    local_root = Path(cache_dir) if cache_dir else None
    local_files = []
    if local_root:
        local_files = sorted((local_root / name).glob("*.parquet"))
    if local_files:
        manifest_path = local_root / name / "manifest.json"
        if not manifest_path.exists():
            raise ValueError(f"local cache for {name} is missing revision/hash manifest")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("revision") != revision:
            raise ValueError(f"local cache revision mismatch for {name}")
        expected = manifest.get("files", {})
        if sorted(expected) != [path.name for path in local_files]:
            raise ValueError(f"local cache file list mismatch for {name}")
        for path in local_files:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != expected[path.name]:
                raise ValueError(f"local cache hash mismatch for {path.name}")
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise RuntimeError("local Parquet input requires pyarrow") from exc
        stream = (row for path in local_files for batch in pq.ParquetFile(path).iter_batches(batch_size=512)
                  for row in batch.to_pylist())
    else:
        kwargs = {"path": repo, "split": "train", "revision": revision, "streaming": True}
        if cache_dir:
            kwargs["cache_dir"] = cache_dir
        stream = load_dataset(**kwargs)
    rows = []
    for row in stream:
        if name == "exgentic":
            converted = _exgentic(row, name)
        elif name == "agenttrace":
            converted = _agenttrace(row, name)
        elif name == "tracecommons":
            converted = _tracecommons(row, name)
        elif name == "betterwright":
            converted = _betterwright(row, name)
        else:
            converted = []
        rows.extend(converted)
        if max_rows and len(rows) >= max_rows:
            rows = rows[:max_rows]
            break
    for item in rows:
        item["dataset"] = name
        item["dataset_revision"] = revision
        item["converter_version"] = CONVERTER_VERSION
    if not rows:
        raise ValueError(f"{name} produced no usable tool spans")
    return rows


def load_semantic_rows(name, cache_dir=None, max_rows=None):
    """Load descriptors for coverage analysis without time labels."""
    if name != "exgentic":
        return []
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("install experiment dependencies with: pip install datasets pyarrow") from exc
    repo, revision = DATASETS[name]
    kwargs = {"path": repo, "split": "train", "revision": revision, "streaming": True}
    if cache_dir:
        kwargs["cache_dir"] = cache_dir
    rows = []
    for index, row in enumerate(load_dataset(**kwargs)):
        if max_rows and index >= max_rows:
            break
        rows.extend(_exgentic_calls(row, name))
    for item in rows:
        item["dataset"] = name
        item["dataset_revision"] = revision
        item["converter_version"] = CONVERTER_VERSION
    return rows
