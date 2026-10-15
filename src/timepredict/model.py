# 文件作用：AFT 模型加载/预测入口与不可变候选打包。
import hashlib
import json
import math
import os
import platform
import tempfile
import time
from pathlib import Path

import numpy as np
import scipy
import xgboost as xgb

from .features import STATS, FeatureSpec, UnsupportedInput, canonical
from .actions import ActionFamilies


# POSIX 发布要求持久化目录元数据；Windows 不提供此目录 fsync 路径。
def _fsync_directory(path):
    if os.name == "nt":
        return
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)

# 写模式避免 Windows 只读句柄限制；真实同步错误必须传给发布方。
def _fsync_file(path):
    with open(path, "r+b") as handle:
        os.fsync(handle.fileno())


# 记录 Python、XGBoost、NumPy 和 SciPy 版本。
def versions():
    return {"python": platform.python_version(), "xgboost": xgb.__version__, "numpy": np.__version__, "scipy": scipy.__version__}


# 加载 CURRENT 指向的包或直接目录，并根据 manifest 选择预测路径。
class Model:
    # 检查包摘要、特征维度和依赖契约；AFT 正式加载还检查 promotion 证据。
    def __init__(self, directory, *, require_promoted=False):
        root = Path(directory)
        pointer = root / "CURRENT"
        if pointer.exists():
            root = root / pointer.read_text(encoding="utf-8").strip()
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("model_type") == "xgboost_direct_log_time":
            raise ValueError("direct_quantile_package_not_supported; retrain with survival:AFT")
        self.distribution_calibrated = False
        self.acceptance_expires_at = 0.0
        if (root / "promotion.json").exists():
            promotion = json.loads((root / "promotion.json").read_text(encoding="utf-8"))
            if promotion.get("decision") != "approved" or promotion.get("manifest_sha256") != hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest():
                raise ValueError("model_not_approved")
            for name, field in (("acceptance.json", "evidence_sha256"), ("gate-policy.json", "policy_sha256")):
                if promotion.get(field) != hashlib.sha256((root / name).read_bytes()).hexdigest():
                    raise ValueError("promotion_evidence_mismatch")
            # Recheck the archived reports/evidence. A marker alone cannot authorize scheduling.
            from .release import evaluate_gate
            decision = evaluate_gate(json.loads((root / "report.json").read_text(encoding="utf-8")),
                                     json.loads((root / "acceptance.json").read_text(encoding="utf-8")),
                                     json.loads((root / "gate-policy.json").read_text(encoding="utf-8")), manifest["version"])
            if decision["decision"] != "approved":
                raise ValueError("model_acceptance_expired_or_invalid")
            self.distribution_calibrated = True
            evidence = json.loads((root / "acceptance.json").read_text(encoding="utf-8"))
            policy = json.loads((root / "gate-policy.json").read_text(encoding="utf-8"))
            self.acceptance_expires_at = evidence["created_at"] + policy["evidence_max_age_seconds"]
        elif require_promoted:
            raise ValueError("model_not_approved")
        for filename, digest in manifest["sha256"].items():
            if hashlib.sha256((root / filename).read_bytes()).hexdigest() != digest:
                raise ValueError("model_checksum_mismatch")
        if manifest.get("feature_sha256") and manifest["feature_sha256"] != hashlib.sha256((root / "features.json").read_bytes()).hexdigest():
            raise ValueError("feature_checksum_mismatch")
        if manifest["dependencies"]["xgboost"].split(".")[0] != xgb.__version__.split(".")[0]:
            raise ValueError("xgboost_major_version_mismatch")
        self.spec = FeatureSpec(**json.loads((root / "features.json").read_text(encoding="utf-8")))
        embedding_path = root / "embedding-manifest.json"
        if embedding_path.exists():
            embedding = json.loads(embedding_path.read_text(encoding="utf-8"))
            if (embedding.get("version") != self.spec.embedding_version or
                    embedding.get("dim") != self.spec.embedding_dim or
                    embedding.get("model") != self.spec.embedding_model or
                    embedding.get("revision") != self.spec.embedding_revision or
                    embedding.get("kind") != self.spec.embedding_kind):
                raise ValueError("embedding_manifest_mismatch")
        self.action_families = ActionFamilies(**json.loads((root / "action-families.json").read_text(encoding="utf-8"))) if (root / "action-families.json").exists() else None
        if self.action_families and (self.action_families.embedding_kind != self.spec.embedding_kind or self.action_families.embedding_dim != self.spec.embedding_dim or self.action_families.embedding_revision != self.spec.embedding_revision):
            raise ValueError("action_embedding_backend_mismatch")
        if self.spec.version not in (3, 4):
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

    # AFT 返回 mu/sigma，输入不支持时返回状态原因。
    def _aft_prediction(self, mu, sigma, unknown_tool=False):
        from .distribution import LogNormal
        try:
            LogNormal(mu, sigma)
            p50, p90 = math.exp(mu), math.exp(mu+sigma*1.2815515655446004)
            if p50 <= 0 or not math.isfinite(p90):
                raise ValueError("nonfinite_result")
        except (ValueError, OverflowError):
            return None, "nonfinite_result"
        return {"mu": mu, "sigma": sigma, "model_version": self.version,
                "feature_version": self.spec.version, "feature_mode": self.spec.feature_mode,
                "distribution_kind": "lognormal", "time_unit": "seconds",
                "p50_seconds": p50, "p90_seconds": p90,
                "distribution_calibrated": self.distribution_calibrated,
                "unknown_tool": unknown_tool}, "ready"

    def predict(self, call):
        if self.distribution_calibrated and time.time() > self.acceptance_expires_at:
            return None, "model_acceptance_expired"
        try:
            call = self.action_families.enrich(call) if self.action_families else call
            row = self.spec.transform(call)
        except UnsupportedInput:
            return None, "unsupported_input"
        if call["name"] not in self.spec.tools and self.spec.feature_mode == "legacy_semantic":
            action = call.get("action_family")
            if not action or action not in self.spec.tools:
                return None, "unknown_tool"
        action = call.get("action_family", call["name"])
        if self.prediction_mode == "family_prior" and action in self.family_priors:
            mu = float(self.family_priors[action]["mu"])
            sigma = float(self.family_priors[action]["sigma"])
            return self._aft_prediction(mu, sigma)
        mu = float(self.booster.predict(xgb.DMatrix(row.reshape(1, -1)), output_margin=True, iteration_range=(0, self.best_iteration + 1))[0]) - float(np.log(self.time_scale))
        if not np.isfinite(mu):
            return None, "nonfinite_result"
        return self._aft_prediction(mu, self.sigma, call["name"] not in self.spec.tools)


# 在 staging 写入并刷新 AFT 模型、特征、清单与报告，再原子重命名为不可变候选版本。
def publish(root, booster, spec, sigma, best_iteration, report, action_families=None, prediction_mode=None, family_priors=None, curve_svg=None, artifacts=None):
    """Write an immutable candidate. Training never updates a serving pointer."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    model_bytes = bytes(booster.save_raw(raw_format="ubj"))
    version = hashlib.sha256(model_bytes + canonical({"report": report, "features": spec.to_dict(), "sigma": sigma,
                                                    "best_iteration": best_iteration, "dependencies": versions()}).encode()).hexdigest()[:16]
    target = root / version
    if target.exists():
        try:
            if Model(target).version != version:
                raise ValueError("candidate_version_mismatch")
        except (OSError, ValueError, KeyError, TypeError):
            # Retraining can reproduce the same content address. Preserve a
            # damaged package for diagnosis instead of silently reusing it.
            target.rename(root / f".invalid-{version}-{time.time_ns()}")
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
