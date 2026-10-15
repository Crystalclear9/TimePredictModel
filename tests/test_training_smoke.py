# 文件作用：小规模 AFT 打包冒烟测试：几轮训练后检查产物、恢复模型并完成预测。
import tempfile
import unittest
from pathlib import Path

from timepredict.training import train
from timepredict.model import Model


# 组织候选训练与独立加载的基础断言。
class TrainingSmoke(unittest.TestCase):
    # 以十个合成工作流训练三轮，检查包文件和恢复后的模型版本。
    def test_model_package_contract(self):
        rows = []
        for index in range(10):
            value = 1.0 + (index % 3) * 0.1
            rows.append({"generation": f"g{index}", "workflow": f"w{index}",
                         "workflow_start": float(index), "workflow_end": float(index) + 0.1,
                         "call": {"name": "search", "arguments": {}, "schema": {}},
                         "lower": value, "upper": value})
        root = Path(tempfile.mkdtemp())
        version, report = train(rows, root, threads=1, rounds=3, sigmas=(0.5,), data_kind="synthetic")
        files = {path.name for path in (root / version).iterdir()}
        self.assertTrue({"model.ubj", "features.json", "embedding-manifest.json",
                         "manifest.json", "report.json", "dataset-manifest.json"} <= files)
        self.assertIn("accuracy_curves", report)
        loaded = Model(root / version)
        prediction, status = loaded.predict({"name": "search", "arguments": {}, "schema": {}})
        self.assertEqual(status, "ready")
        self.assertTrue(prediction["model_version"] == loaded.version)
        self.assertEqual(loaded.spec.feature_mode, "structured")
        self.assertEqual(loaded.spec.embedding_dim, 0)


if __name__ == "__main__":
    unittest.main()
