import hashlib
import json
import math
import os
import platform
import tempfile
from pathlib import Path

import numpy as np
import scipy
import xgboost as xgb

from .features import STATS, FeatureSpec, UnsupportedInput, canonical
from .actions import ActionFamilies


def _fsync_directory(path):
    try:
        fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        pass

def _fsync_file(path):
    try:
        with open(path, "rb") as handle:
            os.fsync(handle.fileno())
    except OSError:
        # Some Windows filesystems reject fsync on read-only handles; Linux
        # publication still performs the required durability barrier.
        pass


def versions():
    return {"python": platform.python_version(), "xgboost": xgb.__version__, "numpy": np.__version__, "scipy": scipy.__version__}


class Model:
    def __init__(self, directory, *, require_promoted=False):
        root = Path(directory)
        pointer = root / "CURRENT"
        if pointer.exists():
            root = root / pointer.read_text().strip()
        manifest = json.loads((root / "manifest.json").read_text())
        if manifest.get("model_type") == "xgboost_direct_log_time":
            from .regression import RegressionModel
            self._regression = RegressionModel(root)
            self.version = manifest.get("package_version", 1)
            self.prediction_mode = "direct_quantiles_uncalibrated"
            return
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
        if manifest.get("feature_sha256") and manifest["feature_sha256"] != hashlib.sha256((root / "features.json").read_bytes()).hexdigest():
            raise ValueError("feature_checksum_mismatch")
        if manifest["dependencies"]["xgboost"].split(".")[0] != xgb.__version__.split(".")[0]:
            raise ValueError("xgboost_major_version_mismatch")
        self.spec = FeatureSpec(**json.loads((root / "features.json").read_text()))
        embedding_path = root / "embedding-manifest.json"
        if embedding_path.exists():
            embedding = json.loads(embedding_path.read_text())
            if (embedding.get("version") != self.spec.embedding_version or
                    embedding.get("dim") != self.spec.embedding_dim or
                    embedding.get("model") != self.spec.embedding_model or
                    embedding.get("revision") != self.spec.embedding_revision or
                    embedding.get("kind") != self.spec.embedding_kind):
                raise ValueError("embedding_manifest_mismatch")
        self.action_families = ActionFamilies(**json.loads((root / "action-families.json").read_text())) if (root / "action-families.json").exists() else None
        if self.action_families and (self.action_families.embedding_kind != self.spec.embedding_kind or self.action_families.embedding_dim != self.spec.embedding_dim or self.action_families.embedding_revision != self.spec.embedding_revision):
            raise ValueError("action_embedding_backend_mismatch")
        if self.spec.version != 3:
            raise ValueError("feature_version_mismatch")
        self.booster = xgb.Booster()
        self.booster.load_model(root / "model.ubj")
        self.sigma, self.version = manifest["sigma"], manifest["version"]
        self.time_scale = float(manifest.get("time_scale", 1.0))
        self.prediction_mode = manifest.get("prediction_mode", "xgboost_aft")
        self.family_priors = manifest.get("family_priors", {})
        self.best_iteration = manifest["best_iteration"]
        expected = len(self.spec.tools) + 1 + self.spec.embedding_dim + len(STATS) + len(self.spec.fields)
        if self.booster.num_features() != expected:
            raise ValueError("feature_dimension_mismatch")

    def predict(self, call):
        if hasattr(self, "_regression"):
            try:
                p50, p90 = self._regression.quantiles(call)
                if not (math.isfinite(float(p50)) and math.isfinite(float(p90)) and p50 > 0 and p90 >= p50):
                    return None, "nonfinite_prediction"
                return {"model_version": str(self.version), "p50_seconds": float(p50),
                        "p90_seconds": float(p90), "prediction_mode": self.prediction_mode,
                        "distribution_calibrated": False,
                        "unknown_reason": "distribution_not_calibrated"}, "ready"
            except ValueError as exc:
                return None, str(exc)
        try:
            call = self.action_families.enrich(call) if self.action_families else call
            row = self.spec.transform(call)
        except UnsupportedInput:
            return None, "unsupported_input"
        if call["name"] not in self.spec.tools:
            action = call.get("action_family")
            if not action or action not in self.spec.tools:
                return None, "unknown_tool"
        action = call.get("action_family", call["name"])
        if self.prediction_mode == "family_prior" and action in self.family_priors:
            mu = float(self.family_priors[action]["mu"])
            sigma = float(self.family_priors[action]["sigma"])
            return {"mu": mu, "sigma": sigma, "model_version": self.version}, "ready"
        mu = float(self.booster.predict(xgb.DMatrix(row.reshape(1, -1)), output_margin=True, iteration_range=(0, self.best_iteration + 1))[0]) - float(np.log(self.time_scale))
        if not np.isfinite(mu):
            return None, "nonfinite_result"
        return {"mu": mu, "sigma": self.sigma, "model_version": self.version}, "ready"


def publish(root, booster, spec, sigma, best_iteration, report, action_families=None, prediction_mode=None, family_priors=None, curve_svg=None, artifacts=None):
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
        (stage / "embedding-manifest.json").write_text(canonical({
            "kind": spec.embedding_kind, "version": spec.embedding_version,
            "model": spec.embedding_model, "revision": spec.embedding_revision, "dim": spec.embedding_dim,
        }), encoding="utf-8")
        (stage / "report.json").write_text(canonical(report), encoding="utf-8")
        if action_families:
            (stage / "action-families.json").write_text(canonical(action_families.to_dict()), encoding="utf-8")
        if curve_svg:
            (stage / "training_curve.svg").write_text(curve_svg, encoding="utf-8")
        files = ["model.ubj", "features.json", "embedding-manifest.json", "report.json"]
        if action_families:
            files.append("action-families.json")
        if curve_svg:
            files.append("training_curve.svg")
        for name, content in (artifacts or {}).items():
            (stage / name).write_text(content, encoding="utf-8")
            files.append(name)
        if not (stage / "dataset-manifest.json").exists():
            (stage / "dataset-manifest.json").write_text(canonical(report.get("dataset_manifest", {"data_digest": report.get("data_digest"), "label_kinds": report.get("label_kinds")})), encoding="utf-8")
            files.append("dataset-manifest.json")
        manifest = {"package_version": 3, "version": version, "sigma": sigma, "best_iteration": best_iteration, "time_scale": float(report.get("time_scale", 1.0)), "prediction_mode": prediction_mode or "xgboost_aft",
                    "family_priors": family_priors or {}, "dependencies": versions(),
                    "feature_sha256": hashlib.sha256((stage / "features.json").read_bytes()).hexdigest(),
                    "sha256": {name: hashlib.sha256((stage / name).read_bytes()).hexdigest() for name in files}}
        (stage / "manifest.json").write_text(canonical(manifest), encoding="utf-8")
        for filename in files + ["manifest.json"]:
            _fsync_file(stage / filename)
        _fsync_directory(stage)
        os.replace(stage, target)
        _fsync_directory(root)
        link = root / "best-model"
        tmp_link = root / (".best-model-" + version)
        try:
            tmp_link.symlink_to(target.name, target_is_directory=True)
            os.replace(tmp_link, link)
        except OSError:
            if tmp_link.exists(): tmp_link.unlink()
    return version
