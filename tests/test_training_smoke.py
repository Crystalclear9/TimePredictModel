import tempfile
import unittest
from pathlib import Path

from timepredict.training import train
from timepredict.model import Model


class TrainingSmoke(unittest.TestCase):
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
        self.assertTrue({"model.ubj", "features.json", "action-families.json", "embedding-manifest.json",
                         "manifest.json", "report.json", "dataset-manifest.json"} <= files)
        self.assertIn("accuracy_curves", report)
        loaded = Model(root / version)
        prediction, status = loaded.predict({"name": "search", "arguments": {}, "schema": {}})
        self.assertEqual(status, "ready")
        self.assertTrue(prediction["model_version"] == loaded.version)


if __name__ == "__main__":
    unittest.main()
