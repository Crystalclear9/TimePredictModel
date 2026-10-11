# 文件作用：受控回放契约测试：边界有序、标签来源明确，时间与结果不进入参数特征。
import json
from pathlib import Path

from timepredict.features import extract
from timepredict.replay import generate


# 生成 100 条真实本地回放样本并检查第一条的边界和特征排除约定。
def test_replay_boundaries_and_feature_exclusion(tmp_path):
    manifest = generate(tmp_path, rows=100, seed=11, concurrency=8)
    assert manifest["label_kind"] == "observed_replay"
    row = json.loads((tmp_path / "samples.jsonl").read_text().splitlines()[0])
    boundaries = row["boundaries"]
    assert set(boundaries) == {"t_out", "t_tool_start", "t_tool_end", "t_return", "clock"}
    assert boundaries["t_out"] <= boundaries["t_tool_start"] <= boundaries["t_tool_end"] <= boundaries["t_return"]
    fields, _ = extract(row["call"])
    assert all("t_return" not in key and "t_out" not in key for key in fields)
    assert "result" not in row["call"]["arguments"]
