# 文件作用：结构特征与语义向量契约：训练建立词典，在线按相同顺序转换为 float32 向量。
import json
import math
from collections import Counter
from dataclasses import asdict, dataclass

import numpy as np
from .semantic import SemanticEncoder

VERSION = 4
MAX_BYTES = 256 * 1024
MAX_DEPTH = 32


# 表示参数超限、非 JSON 对象或数值不可表达，调用不能进入预测。
class UnsupportedInput(ValueError):
    pass


# 生成稳定且严格的 JSON 表示；排序键并拒绝 NaN/Infinity。
def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


# 把对象/JSON 字符串规范化为字典，检查 256KiB 大小及 32 层深度上限。
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
        # 递归检查对象和数组的嵌套深度。
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


# 提取数值、布尔、字符串长度、数组长度、枚举指示和通用结构统计。
def extract(call, use_action_family=True):
    if not isinstance(call, dict) or not isinstance(call.get("name"), str) or not call["name"] or "arguments" not in call:
        raise UnsupportedInput("invalid_call")
    args = parse_arguments(call["arguments"])
    tool = call.get("action_family", call["name"]) if use_action_family else call["name"]
    fields, stats = {}, Counter()
    stats["json_bytes"] = len(canonical(args).encode("utf-8"))

    # 用动作族、JSON Pointer 和特征类型组成字段键。
    def put(path, kind, value):
        fields[canonical([tool, path, kind])] = value

    # 遍历参数树；布尔值先于数值判断，嵌套对象按排序后的字段处理。
    def walk(value, path, schema, depth):
        if isinstance(schema, bool):
            schema = {}  # Boolean JSON Schema nodes add no enum or property hints.
        if not isinstance(schema, dict):
            raise UnsupportedInput("invalid_schema")
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
            # 数组内部只累计类型、字段及深度统计，不建立位置相关特征列。
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

    try:
        walk(args, "", call.get("schema", {}), 0)
    except UnsupportedInput:
        raise
    except (AttributeError, TypeError, ValueError, UnicodeError) as exc:
        raise UnsupportedInput("invalid_schema") from exc
    return fields, stats


STATS = ("json_bytes", "max_depth", "fields", "null", "boolean", "number", "string", "array", "object")


# 保存工具/动作族词典、结构字段和固定语义编码器契约。
@dataclass
class FeatureSpec:
    tools: list[str]
    fields: list[str]
    version: int = VERSION
    embedding_dim: int = 384
    embedding_version: int = 3
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_revision: str = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
    embedding_kind: str = "sentence_transformers"
    # Legacy packages omit this field. New fits select structured explicitly.
    feature_mode: str = "legacy_semantic"

    # 仅从传入的训练调用统计字段频率，最多保留 4096 列。
    @classmethod
    def fit(cls, calls, max_fields=4096, feature_mode="structured"):
        if feature_mode not in ("structured", "semantic", "legacy_semantic"):
            raise ValueError("invalid_feature_mode")
        counts, tools = Counter(), set()
        for call in calls:
            fields, _ = extract(call, use_action_family=feature_mode == "legacy_semantic")
            counts.update(fields.keys())
            tools.add(call.get("action_family", call["name"]) if feature_mode == "legacy_semantic" else call["name"])
        if feature_mode == "structured":
            return cls(sorted(tools), sorted(counts, key=lambda k: (-counts[k], k))[:max_fields],
                       embedding_dim=0, embedding_kind="none", feature_mode=feature_mode)
        encoder = SemanticEncoder()
        spec = cls(sorted(tools), sorted(counts, key=lambda k: (-counts[k], k))[:max_fields],
                   embedding_dim=encoder.dim, embedding_model=encoder.model_name,
                   embedding_revision=encoder.revision, embedding_kind=encoder.backend,
                   feature_mode=feature_mode)
        object.__setattr__(spec, "_encoder", encoder)
        return spec

    # 按工具指示、未知工具列、语义向量、统计、结构字段顺序编码；缺失字段用 NaN。
    def transform(self, call):
        if self.version not in (3, VERSION):
            raise ValueError("feature_version_mismatch")
        fields, stats = extract(call, use_action_family=self.feature_mode == "legacy_semantic")
        action = call.get("action_family", call["name"]) if self.feature_mode == "legacy_semantic" else call["name"]
        values = [float(action == t) for t in self.tools]
        values += [float(action not in self.tools)]
        if self.feature_mode != "structured":
            encoder = getattr(self, "_encoder", None)
            if encoder is None:
                encoder = SemanticEncoder(self.embedding_model, revision=self.embedding_revision,
                                          device="cpu")
                object.__setattr__(self, "_encoder", encoder)
            if encoder.dim != self.embedding_dim or encoder.backend != self.embedding_kind:
                raise ValueError("embedding_dimension_mismatch")
            values += (encoder.encode_call(call) if self.feature_mode == "semantic" else encoder.encode(call)).tolist()
        values += [float(stats[k]) for k in STATS]
        values += [fields.get(k, float("nan")) for k in self.fields]
        return np.asarray(values, dtype=np.float32)

    # 导出 dataclass 配置，不包含运行时缓存的编码器。
    def to_dict(self):
        return asdict(self)
