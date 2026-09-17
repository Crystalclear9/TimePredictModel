import json
import math
from collections import Counter
from dataclasses import asdict, dataclass

import numpy as np

VERSION = 1
MAX_BYTES = 256 * 1024
MAX_DEPTH = 32


class UnsupportedInput(ValueError):
    pass


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def parse_arguments(arguments):
    try:
        if isinstance(arguments, str):
            if len(arguments.encode("utf-8")) > MAX_BYTES:
                raise UnsupportedInput("arguments_too_large")
            arguments = json.loads(arguments, parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
        if not isinstance(arguments, dict):
            raise UnsupportedInput("arguments_not_object")
        if len(canonical(arguments).encode("utf-8")) > MAX_BYTES:
            raise UnsupportedInput("arguments_too_large")
        def check(v, depth):
            if depth > MAX_DEPTH:
                raise UnsupportedInput("arguments_too_deep")
            if isinstance(v, dict):
                for child in v.values():
                    check(child, depth + 1)
            elif isinstance(v, list):
                for child in v:
                    check(child, depth + 1)
        check(arguments, 0)
        return arguments
    except (ValueError, TypeError, RecursionError, UnicodeError) as exc:
        raise UnsupportedInput(str(exc)) from exc


def extract(call):
    args = parse_arguments(call["arguments"])
    tool = call["name"]
    fields, stats = {}, Counter()
    stats["json_bytes"] = len(canonical(args).encode("utf-8"))

    def put(path, kind, value):
        fields[canonical([tool, path, kind])] = value

    def walk(value, path, schema, depth):
        stats["max_depth"] = max(stats["max_depth"], depth)
        if value is None:
            kind = "null"
            put(path, "null", 1.0)
        elif isinstance(value, bool):
            kind = "boolean"
            put(path, "boolean", float(value))
        elif isinstance(value, (int, float)):
            kind = "number"
            try:
                number = float(value)
            except OverflowError as exc:
                raise UnsupportedInput("number_out_of_range") from exc
            if not math.isfinite(number) or abs(number) > float(np.finfo(np.float32).max):
                raise UnsupportedInput("number_out_of_range")
            put(path, "number", number)
        elif isinstance(value, str):
            kind = "string"
            put(path, "chars", len(value))
            put(path, "bytes", len(value.encode("utf-8")))
        elif isinstance(value, list):
            kind = "array"
            put(path, "length", len(value))
            # Count structure inside arrays without generating position-specific fields.
            def count_array(v, d):
                stats["max_depth"] = max(stats["max_depth"], d)
                typ = "null" if v is None else "boolean" if isinstance(v, bool) else "object" if isinstance(v, dict) else "array" if isinstance(v, list) else "string" if isinstance(v, str) else "number"
                stats[typ] += 1
                if isinstance(v, dict):
                    stats["fields"] += len(v)
                    for child in v.values():
                        count_array(child, d + 1)
                elif isinstance(v, list):
                    for child in v:
                        count_array(child, d + 1)
            for child in value:
                count_array(child, depth + 1)
        elif isinstance(value, dict):
            kind = "object"
            stats["fields"] += len(value)
            for key in sorted(value):
                pointer = path + "/" + key.replace("~", "~0").replace("/", "~1")
                walk(value[key], pointer, schema.get("properties", {}).get(key, {}), depth + 1)
        else:
            raise UnsupportedInput("unsupported_type")
        stats[kind] += 1
        if "enum" in schema and not isinstance(value, (list, dict)):
            # Include zero columns for every declared enum, not just the observed value.
            for choice in schema["enum"]:
                put(path, "enum:" + canonical(choice), float(canonical(value) == canonical(choice)))
            put(path, "enum:unknown", float(all(canonical(value) != canonical(c) for c in schema["enum"])))

    walk(args, "", call.get("schema", {}), 0)
    return fields, stats


STATS = ("json_bytes", "max_depth", "fields", "null", "boolean", "number", "string", "array", "object")


@dataclass
class FeatureSpec:
    tools: list[str]
    fields: list[str]
    version: int = VERSION

    @classmethod
    def fit(cls, calls, max_fields=4096):
        counts, tools = Counter(), set()
        for call in calls:
            fields, _ = extract(call)
            counts.update(fields.keys())
            tools.add(call["name"])
        return cls(sorted(tools), sorted(counts, key=lambda k: (-counts[k], k))[:max_fields])

    def transform(self, call):
        if self.version != VERSION:
            raise ValueError("feature_version_mismatch")
        fields, stats = extract(call)
        values = [float(call["name"] == t) for t in self.tools]
        values += [float(call["name"] not in self.tools)]
        values += [float(stats[k]) for k in STATS]
        values += [fields.get(k, float("nan")) for k in self.fields]
        return np.asarray(values, dtype=np.float32)

    def to_dict(self):
        return asdict(self)
