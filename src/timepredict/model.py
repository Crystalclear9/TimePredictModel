import hashlib
import json
import os
import platform
import tempfile
from pathlib import Path

import numpy as np
import scipy
import xgboost as xgb

from .features import STATS, FeatureSpec, UnsupportedInput, canonical


def versions():
    return {"python": platform.python_version(), "xgboost": xgb.__version__, "numpy": np.__version__, "scipy": scipy.__version__}


class Model:
    def __init__(self, directory, *, require_promoted=False):
        root = Path(directory)
        pointer = root / "CURRENT"
        if pointer.exists():
            root = root / pointer.read_text().strip()
        manifest = json.loads((root / "manifest.json").read_text())
        if require_promoted:
            promotion = json.loads((root / "promotion.json").read_text())
            if promotion.get("decision") != "approved" or promotion.get("manifest_sha256") != hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest():
                raise ValueError("model_not_approved")
            for name, field in (("acceptance.json", "evidence_sha256"), ("gate-policy.json", "policy_sha256")):
                if promotion.get(field) != hashlib.sha256((root / name).read_bytes()).hexdigest():
                    raise ValueError("promotion_evidence_mismatch")
        for filename, digest in manifest["sha256"].items():
            if hashlib.sha256((root / filename).read_bytes()).hexdigest() != digest:
                raise ValueError("model_checksum_mismatch")
        if manifest["dependencies"]["xgboost"].split(".")[0] != xgb.__version__.split(".")[0]:
            raise ValueError("xgboost_major_version_mismatch")
        self.spec = FeatureSpec(**json.loads((root / "features.json").read_text()))
        if self.spec.version != 1:
            raise ValueError("feature_version_mismatch")
        self.booster = xgb.Booster()
        self.booster.load_model(root / "model.ubj")
        self.sigma, self.version = manifest["sigma"], manifest["version"]
        self.best_iteration = manifest["best_iteration"]
        expected = len(self.spec.tools) + 1 + len(STATS) + len(self.spec.fields)
        if self.booster.num_features() != expected:
            raise ValueError("feature_dimension_mismatch")

    def predict(self, call):
        try:
            row = self.spec.transform(call)
        except UnsupportedInput:
            return None, "unsupported_input"
        if call["name"] not in self.spec.tools:
            return None, "unknown_tool"
        mu = float(self.booster.predict(xgb.DMatrix(row.reshape(1, -1)), output_margin=True, iteration_range=(0, self.best_iteration + 1))[0])
        if not np.isfinite(mu):
            return None, "nonfinite_result"
        return {"mu": mu, "sigma": self.sigma, "model_version": self.version}, "ready"


def publish(root, booster, spec, sigma, best_iteration, report):
    """Write an immutable candidate. Training never updates a serving pointer."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    model_bytes = bytes(booster.save_raw(raw_format="ubj"))
    version = hashlib.sha256(model_bytes + canonical({"report": report, "features": spec.to_dict(), "sigma": sigma,
                                                    "best_iteration": best_iteration, "dependencies": versions()}).encode()).hexdigest()[:16]
    target = root / version
    if not target.exists():
        stage = Path(tempfile.mkdtemp(prefix=".staging-", dir=root))
        booster.save_model(stage / "model.ubj")
        (stage / "features.json").write_text(canonical(spec.to_dict()), encoding="utf-8")
        (stage / "report.json").write_text(canonical(report), encoding="utf-8")
        manifest = {"version": version, "sigma": sigma, "best_iteration": best_iteration, "dependencies": versions(),
                    "sha256": {name: hashlib.sha256((stage / name).read_bytes()).hexdigest() for name in ("model.ubj", "features.json", "report.json")}}
        (stage / "manifest.json").write_text(canonical(manifest), encoding="utf-8")
        os.replace(stage, target)
    return version
