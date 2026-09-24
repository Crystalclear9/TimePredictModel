import json

from timepredict.features import FeatureSpec
from timepredict.semantic import descriptor_text
from timepredict.training import split_rows


def _call(value):
    return {
        "name": "sleep_tool",
        "description": "controlled sleep operation",
        "arguments": {"items": value},
        "schema": {"type": "object", "properties": {"items": {"type": "integer"}}},
    }


def test_semantic_descriptor_excludes_argument_values():
    assert descriptor_text(_call(1)) == descriptor_text(_call(999))


def test_feature_spec_has_no_label_fields():
    spec = FeatureSpec.fit([_call(1), _call(2)])
    forbidden = ("t_out", "t_return", "lower", "upper", "duration", "result", "workflow")
    assert not any(any(token in field.lower() for token in forbidden) for field in spec.fields)


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
