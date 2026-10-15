# 文件作用：基础契约测试：关联降级、参数拒绝、流式拼接、来源失联、动作匹配及分布数学。
import math
import unittest
from unittest.mock import patch

from timepredict.distribution import LogNormal
from timepredict.features import FeatureSpec, UnsupportedInput
from timepredict.semantic import FrozenSemanticEncoder
from timepredict.state import empty_state, reduce_event
from timepredict.sglang import Capture
from timepredict.actions import ActionFamilies


# 构造可控时钟戳，避免测试依赖真实 chrony。
def stamp(utc, source="a", boot="b", mono=None, error=0.001):
    return {"utc": utc, "mono": utc if mono is None else mono, "source": source,
            "boot": boot, "error": error, "jumped": False}


# 用 unittest 组织基础确定性契约断言。
class ContractTests(unittest.TestCase):
    # 验证无工具 ID 时仍可按串行业务键闭合，并标记 serial_fallback。
    def test_serial_return_without_tool_id_uses_business_key(self):
        state = empty_state()
        out = {"event_id": "out", "kind": "out", "tenant": "t", "workflow_id": "w", "agent_id": "a",
               "request_id": "r1", "call": {"id": "tc", "name": "search", "arguments": "{}"},
               "stamp": stamp(1, mono=1)}
        state, _ = reduce_event(state, out)
        incoming = {"event_id": "in", "kind": "ingress", "tenant": "t", "workflow_id": "w", "agent_id": "a",
                    "request_id": "r2", "latest_tool_ids": [], "stamp": stamp(3, mono=3)}
        state, _ = reduce_event(state, incoming)
        record = next(iter(state["records"].values()))
        self.assertEqual(record["status"], "returned")
        self.assertEqual(record["association_mode"], "serial_fallback")

    # 验证非 JSON 参数在特征拟合阶段被拒绝。
    def test_invalid_arguments_are_rejected(self):
        with self.assertRaises(UnsupportedInput):
            FeatureSpec.fit([{"name": "x", "arguments": "not-json", "schema": {}}])

    # 验证显式测试编码器的确定性和输出维度。
    def test_embedding_is_stable(self):
        encoder = FrozenSemanticEncoder()
        call = {"name": "find_files", "description": "search files", "schema": {"properties": {"query": {"type": "string"}}}, "arguments": '{"query":"x"}'}
        self.assertTrue((encoder.encode(call) == encoder.encode(call)).all())
        self.assertEqual(encoder.encode(call).shape[0], 32)

    # 验证条件窗口概率数值与 P90/P50 顺序。
    def test_conditional_probability_is_bounded(self):
        result = LogNormal(math.log(3), 0.5).summarize(2, 2)
        self.assertAlmostEqual(result["prob_return_within_H"], 0.643, delta=0.01)
        self.assertGreater(result["remaining_p90_seconds"], result["remaining_p50_seconds"])

    # 验证两段 SSE 参数可拼成完整 JSON。
    def test_stream_fragments_complete_one_call(self):
        capture = Capture({"tenant": "t", "workflow_id": "w", "agent_id": "a", "request_id": "r"},
                          {"search": {"description": "search", "parameters": {}}})
        first = {"choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "id": "tc", "function": {"name": "search", "arguments": '{"q":"'}}]}}]}
        second = {"choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "function": {"arguments": 'x"}'}}]}, "finish_reason": "tool_calls"}]}
        capture.observe(first, True)
        capture.observe(second, True)
        self.assertEqual(capture.payload()["call"]["arguments"], '{"q":"x"}')

    # 验证来源失联只关闭该来源的等待记录。
    def test_source_loss_only_invalidates_matching_source(self):
        state = empty_state()
        for i, source in enumerate(("s1", "s2"), 1):
            event = {"event_id": f"out{i}", "kind": "out", "tenant": "t", "workflow_id": f"w{i}", "agent_id": "a",
                     "request_id": f"r{i}", "call": {"id": f"tc{i}", "name": "search", "arguments": "{}"},
                     "stamp": stamp(i, source=source, boot="b", mono=i), "seq": 1}
            state, _ = reduce_event(state, event)
        state, _ = reduce_event(state, {"event_id": "lost", "kind": "source_lost", "lost_source": "s1/b",
                                        "last_seen": 1, "stamp": stamp(3, source="s1", boot="b", mono=3)})
        statuses = {record["workflow_id"]: record["status"] for record in state["records"].values()}
        self.assertEqual(statuses["w1"], "observation_lost")
        self.assertEqual(statuses["w2"], "waiting")

    # 验证不相关操作输出未知/歧义与诊断排名。
    def test_action_family_reports_unknown_or_match(self):
        with patch.dict("os.environ", {"TIMEPREDICT_TEST_ENCODER": "1"}):
            families = ActionFamilies.fit([{"name": "search_documents", "description": "search documents",
                                        "schema": {"properties": {"query": {"type": "string"}}},
                                        "arguments": {"query": "x"}}])
        family, ranked, _ = families.match({"name": "unrelated_operation", "description": "rotate keys",
                                            "schema": {"properties": {"key_id": {"type": "string"}}},
                                            "arguments": {"key_id": "x"}})
        self.assertIn(family, {"unknown_action", "ambiguous_action"})
        self.assertIsInstance(ranked, list)


if __name__ == "__main__":
    unittest.main()
