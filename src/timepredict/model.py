# 文件作用：AFT 模型加载/预测入口与不可变候选打包。
import hashlib
import json
import math
import os
import platform
import re
import tempfile
import time
from pathlib import Path

import numpy as np
import scipy
import xgboost as xgb

from .features import STATS, FeatureSpec, UnsupportedInput, canonical
from .actions import ActionFamilies


def _read_json_object(path):
    def invalid_constant(value):
        raise ValueError("nonfinite_json: " + value)
    def finite_float(value):
        number = float(value)
        if not math.isfinite(number):
            invalid_constant(value)
        return number
    value = json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=invalid_constant, parse_float=finite_float)
    if not isinstance(value, dict):
        raise ValueError("json_object_required: " + str(path))
    return value


def _flat_name(name):
    # Bundles contain flat files; reject both Windows and POSIX path syntax.
    return (isinstance(name, str) and name not in ("", ".", "..") and
            not any(c in name for c in '/\\:<>"|?*') and
            not any(ord(c) < 32 for c in name) and not name.endswith((".", " ")))


def _positive_number(value):
    try:
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0
    except OverflowError:
        return False


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
            name = pointer.read_text(encoding="utf-8").strip()
            if not _flat_name(name) or (root/name).resolve().parent != root.resolve():
                raise ValueError("invalid_current_pointer")
            root = root / name
        manifest = _read_json_object(root / "manifest.json")
        if manifest.get("model_type") == "xgboost_direct_log_time":
            raise ValueError("direct_quantile_package_not_supported; retrain with survival:AFT")
        hashes = manifest.get("sha256")
        required = {"model.ubj", "features.json", "report.json"}
        if manifest.get("package_version") == 3:
            required.update(("embedding-manifest.json", "dataset-manifest.json"))
        # Optional behavior-changing files cannot be loaded outside the manifest.
        required.update(name for name in ("embedding-manifest.json", "action-families.json") if (root/name).exists())
        if not isinstance(hashes, dict) or not required <= hashes.keys():
            raise ValueError("incomplete_package_checksums")
        if len({name.casefold() for name in hashes}) != len(hashes):
            raise ValueError("ambiguous_package_filenames")
        for filename, digest in hashes.items():
            if (not _flat_name(filename) or filename in {"manifest.json", "promotion.json", "acceptance.json", "gate-policy.json", "CURRENT"} or
                    (root/filename).resolve().parent != root.resolve() or not isinstance(digest, str) or
                    not re.fullmatch(r"[0-9a-f]{64}", digest)):
                raise ValueError("invalid_package_checksum_entry")
            if hashlib.sha256((root / filename).read_bytes()).hexdigest() != digest:
                raise ValueError("model_checksum_mismatch")
        report = _read_json_object(root / "report.json")
        self.distribution_calibrated = False
        self.acceptance_expires_at = 0.0
        if (root / "promotion.json").exists():
            promotion = _read_json_object(root / "promotion.json")
            if promotion.get("decision") != "approved" or promotion.get("manifest_sha256") != hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest():
                raise ValueError("model_not_approved")
            for name, field in (("acceptance.json", "evidence_sha256"), ("gate-policy.json", "policy_sha256")):
                if promotion.get(field) != hashlib.sha256((root / name).read_bytes()).hexdigest():
                    raise ValueError("promotion_evidence_mismatch")
            # Recheck the archived reports/evidence. A marker alone cannot authorize scheduling.
            from .release import evaluate_gate
            evidence = _read_json_object(root / "acceptance.json")
            policy = _read_json_object(root / "gate-policy.json")
            decision = evaluate_gate(report, evidence, policy, manifest["version"])
            if decision["decision"] != "approved":
                raise ValueError("model_acceptance_expired_or_invalid")
            self.distribution_calibrated = True
            self.acceptance_expires_at = evidence["created_at"] + policy["evidence_max_age_seconds"]
        elif require_promoted:
            raise ValueError("model_not_approved")
        if manifest.get("feature_sha256") and manifest["feature_sha256"] != hashlib.sha256((root / "features.json").read_bytes()).hexdigest():
            raise ValueError("feature_checksum_mismatch")
        if manifest["dependencies"]["xgboost"].split(".")[0] != xgb.__version__.split(".")[0]:
            raise ValueError("xgboost_major_version_mismatch")
        self.spec = FeatureSpec(**_read_json_object(root / "features.json"))
        embedding_path = root / "embedding-manifest.json"
        if embedding_path.exists():
            embedding = _read_json_object(embedding_path)
            if (embedding.get("version") != self.spec.embedding_version or
                    embedding.get("dim") != self.spec.embedding_dim or
                    embedding.get("model") != self.spec.embedding_model or
                    embedding.get("revision") != self.spec.embedding_revision or
                    embedding.get("kind") != self.spec.embedding_kind):
                raise ValueError("embedding_manifest_mismatch")
        self.action_families = ActionFamilies(**_read_json_object(root / "action-families.json")) if (root / "action-families.json").exists() else None
        if self.action_families and (self.action_families.embedding_kind != self.spec.embedding_kind or self.action_families.embedding_dim != self.spec.embedding_dim or self.action_families.embedding_revision != self.spec.embedding_revision):
            raise ValueError("action_embedding_backend_mismatch")
        if self.spec.version not in (3, 4):
            raise ValueError("feature_version_mismatch")
        self.booster = xgb.Booster()
        self.booster.load_model(root / "model.ubj")
        self.sigma, self.version = manifest["sigma"], manifest["version"]
        self.prediction_mode = manifest.get("prediction_mode", "xgboost_aft")
        self.family_priors = manifest.get("family_priors", {})
        self.best_iteration = manifest["best_iteration"]
        if (not _positive_number(self.sigma) or not _positive_number(manifest.get("time_scale", 1.0)) or
                not _flat_name(self.version) or
                type(self.best_iteration) is not int or not 0 <= self.best_iteration < self.booster.num_boosted_rounds() or
                self.prediction_mode not in ("xgboost_aft", "family_prior")):
            raise ValueError("invalid_prediction_contract")
        self.time_scale = float(manifest.get("time_scale", 1.0))
        objective = json.loads(self.booster.save_config())["learner"]["objective"]
        aft = objective.get("aft_loss_param", {})
        if (objective.get("name") != "survival:aft" or aft.get("aft_loss_distribution") != "normal" or
                not math.isclose(float(aft.get("aft_loss_distribution_scale", 0)), self.sigma, rel_tol=1e-6)):
            raise ValueError("aft_objective_contract_mismatch")
        if self.booster.attr("best_iteration") is not None and self.best_iteration != self.booster.best_iteration:
            raise ValueError("best_iteration_contract_mismatch")
        for field, actual in (("time_scale", self.time_scale), ("prediction_mode", self.prediction_mode),
                              ("family_priors", self.family_priors), ("feature_mode", self.spec.feature_mode),
                              ("embedding_kind", self.spec.embedding_kind), ("embedding_dim", self.spec.embedding_dim)):
            if field in report and report[field] != actual:
                raise ValueError("report_contract_mismatch: " + field)
        if self.prediction_mode == "xgboost_aft" and self.family_priors:
            raise ValueError("unexpected_family_priors")
        if self.spec.feature_mode not in ("structured", "semantic", "legacy_semantic") or (self.spec.feature_mode == "structured" and (self.spec.embedding_dim != 0 or self.spec.embedding_kind != "none")):
            raise ValueError("invalid_feature_mode_contract")
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
    reserved = {"model.ubj", "features.json", "embedding-manifest.json", "report.json", "action-families.json",
                "manifest.json", "promotion.json", "gate-policy.json", "acceptance.json", "current", "training_curve.svg"}
    for name in artifacts or {}:
        if not _flat_name(name) or name.casefold() in reserved:
            raise ValueError("invalid_artifact_name")
    if len({name.casefold() for name in artifacts or {}}) != len(artifacts or {}):
        raise ValueError("ambiguous_artifact_names")
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    model_bytes = bytes(booster.save_raw(raw_format="ubj"))
    version = hashlib.sha256(model_bytes + canonical({"report": report, "features": spec.to_dict(), "sigma": sigma,
                                                    "best_iteration": best_iteration, "dependencies": versions(),
                                                    "action_families": action_families.to_dict() if action_families else None,
                                                    "prediction_mode": prediction_mode or "xgboost_aft", "family_priors": family_priors or {},
                                                    "curve_svg": curve_svg, "artifacts": artifacts or {}}).encode()).hexdigest()[:16]
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
        Model(stage)
        for filename in files + ["manifest.json"]:
            _fsync_file(stage / filename)
        _fsync_directory(stage)
        os.replace(stage, target)
        _fsync_directory(root)
    return version
