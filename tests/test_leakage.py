# 文件作用：防泄漏测试：语义向量不使用参数值，特征不引用标签，工作流不跨数据集合。
import json

from timepredict.features import FeatureSpec
from timepredict.semantic import descriptor_text
from timepredict.training import split_rows


# 构造结构相同但数值可变的工具调用。
def _call(value):
    return {
        "name": "sleep_tool",
        "description": "controlled sleep operation",
        "arguments": {"items": value},
        "schema": {"type": "object", "properties": {"items": {"type": "integer"}}},
    }


# 比较不同参数值的操作描述文本，确保值没有进入描述符。
def test_semantic_descriptor_excludes_argument_values():
    assert descriptor_text(_call(1)) == descriptor_text(_call(999))


# 检查测试调用生成的词典中没有耗时、结果和工作流标签字段。
def test_feature_spec_has_no_label_fields():
    spec = FeatureSpec.fit([_call(1), _call(2)])
    forbidden = ("t_out", "t_return", "lower", "upper", "duration", "result", "workflow")
    assert not any(any(token in field.lower() for token in forbidden) for field in spec.fields)


# 验证训练、验证、测试三组 workflow 集合互不相交。
def test_temporal_split_has_disjoint_workflows():
    rows = []
    for workflow in range(12):
        for call in range(2):
            start = float(workflow)
            rows.append({"workflow": str(workflow), "workflow_start": start,
                         "workflow_end": start + .1, "generation": f"{workflow}-{call}",
                         "call": _call(call + 1), "lower": 1., "upper": 1.})
    groups = split_rows(rows)
    sets = [{r["workflow"] for r in groups[name]} for name in ("train", "validation", "test")]
    assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
