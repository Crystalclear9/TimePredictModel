# 文件作用：AFT 生产发布门禁：训练报告、服务器证据及显式策略都通过才切换 CURRENT。
"""Explicit quality gate and atomic promotion; no bypass flag."""
import hashlib
import json
import math
import os
import shutil
import tempfile
import time
from pathlib import Path

from .features import canonical
from .model import Model


# 计算文件 SHA-256，用于绑定发布包和证据。
def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# 尝试刷新发布目录的元数据。
def _fsync_directory(path):
    try:
        fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        pass


# 累计来源、校准、样本支持、基线、证据时效和服务器性能检查失败原因。
def evaluate_gate(report, evidence, policy, version):
    failures = []
    # 保留所有未满足项，不在第一项失败时终止评估。
    def require(test, label):
        if not test:
            failures.append(label)
    # 检查有限数值，并排除布尔值。
    def finite(value):
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
    positive = ("min_supported_rows", "min_effective_samples", "evidence_max_age_seconds",
                "max_entry_hook_p99_seconds", "max_prediction_available_p99_seconds")
    nonnegative = ("max_window_calibration_error", "max_quantile_coverage_error", "max_brier_regression", "max_throughput_drop")
    for field in (*positive, *nonnegative):
        value = policy.get(field)
        if not finite(value) or value < 0 or (field in positive and value == 0):
            failures.append("invalid gate policy: "+field)
    for field in ("horizons", "remaining_ages"):
        values = policy.get(field)
        if not isinstance(values, list) or not values or any(not finite(v) or v <= 0 for v in values):
            failures.append("invalid gate policy: "+field)
    if failures:
        return {"decision": "rejected", "failures": failures, "model_version": version}
    # 检查 IPCW 支持/有效样本量及预测概率与实际概率的校准误差。
    def metric(value, label, tolerance, target=None):
        require(isinstance(value, dict), label+": missing")
        if not isinstance(value, dict):
            return
        require(value.get("status") == "ok", label+": unsupported")
        require(value.get("supported_rows", 0) >= policy["min_supported_rows"], label+": insufficient samples")
        require(value.get("effective_sample_size", 0) >= policy["min_effective_samples"], label+": insufficient effective samples")
        actual = value.get("weighted_actual")
        predicted = target if target is not None else value.get("weighted_prediction")
        require(finite(actual) and finite(predicted), label+": invalid calibration")
        if finite(actual) and finite(predicted):
            require(abs(actual-predicted) <= tolerance, label+": calibration error")
    require(report.get("data_kind") == "observed", "training provenance must be observed")
    require(report.get("embedding_kind") == "sentence_transformers", "production release requires Sentence Transformer embeddings")
    for split in ("validation", "test"):
        result = report.get(split, {})
        require(finite(result.get("nll")), split+": invalid likelihood")
        for horizon in policy["horizons"]:
            metric(result.get("windows", {}).get(str(float(horizon))), f"{split}/H={horizon}", policy["max_window_calibration_error"])
        for q in (0.5, 0.9):
            metric(result.get("quantile_coverage", {}).get(str(q)), f"{split}/q={q}", policy["max_quantile_coverage_error"], q)
        for age in policy.get("remaining_ages", []):
            for horizon in policy["horizons"]:
                metric(result.get("remaining_windows", {}).get(str(float(age)), {}).get(str(float(horizon))),
                       f"{split}/a={age}/H={horizon}", policy["max_window_calibration_error"])
    if policy.get("require_tool_holdout"):
        require(finite(report.get("unknown_tool_test", {}).get("nll")), "tool holdout experiment missing")
    for horizon in policy["horizons"]:
        name = str(float(horizon))
        measured = report.get("test", {}).get("windows", {}).get(name, {}).get("ipcw_brier")
        baseline = report.get("baselines", {}).get("per_tool_empirical_windows", {}).get(name, {}).get("ipcw_brier")
        require(finite(measured) and finite(baseline), "Brier comparison unavailable at H="+name)
        if finite(measured) and finite(baseline):
            require(measured <= baseline+policy["max_brier_regression"], "Brier regression at H="+name)
    require(evidence.get("kind") == "server_acceptance", "target-server acceptance evidence required")
    require(evidence.get("model_version") == version, "benchmark model version mismatch")
    created = evidence.get("created_at")
    require(finite(created) and 0 <= time.time()-created <= policy["evidence_max_age_seconds"], "acceptance evidence expired or invalid")
    for check in ("sglang_nonstream", "sglang_stream", "cross_ingress_roundtrip", "clock_bounds", "worker_failover", "prediction_observed"):
        require(evidence.get("checks", {}).get(check) is True, "server check missing: "+check)
    measured = evidence.get("metrics", {})
    for key, limit in (("throughput_drop_fraction", policy["max_throughput_drop"]),
                       ("entry_hook_p99_seconds", policy["max_entry_hook_p99_seconds"]),
                       ("prediction_available_p99_seconds", policy["max_prediction_available_p99_seconds"])):
        require(finite(measured.get(key)) and measured[key] <= limit and (key == "throughput_drop_fraction" or measured[key] >= 0), key+": exceeds limit or unavailable")
    require(measured.get("request_errors") == 0, "benchmark request errors")
    require(measured.get("dropped_events") == 0, "event drops in benchmark")
    return {"decision": "rejected" if failures else "approved", "failures": failures, "model_version": version}


# 门禁通过后复制候选及证据，验证正式包，再原子替换 CURRENT 文本指针。
def promote(candidate, serving, evidence_file, policy_file):
    candidate, serving = Path(candidate).resolve(), Path(serving).resolve()
    model = Model(candidate)
    report = json.loads((candidate / "report.json").read_text())
    policy = json.loads(Path(policy_file).read_text())
    evidence = json.loads(Path(evidence_file).read_text())
    decision = evaluate_gate(report, evidence, policy, model.version)
    if decision["decision"] != "approved":
        raise ValueError(canonical(decision))
    decision.update(manifest_sha256=sha(candidate / "manifest.json"), evidence_sha256=sha(evidence_file),
                    policy_sha256=sha(policy_file), approved_at=time.time())
    serving.mkdir(parents=True, exist_ok=True)
    release_id = model.version + "-" + hashlib.sha256(canonical(decision).encode()).hexdigest()[:12]
    stage = Path(tempfile.mkdtemp(prefix=".staging-", dir=serving))
    manifest = json.loads((candidate / "manifest.json").read_text())
    for name in (*manifest["sha256"], "manifest.json"):
        shutil.copyfile(candidate / name, stage / name)
    (stage / "promotion.json").write_text(canonical(decision), encoding="utf-8")
    shutil.copyfile(policy_file, stage / "gate-policy.json")
    shutil.copyfile(evidence_file, stage / "acceptance.json")
    Model(stage, require_promoted=True)
    os.replace(stage, serving / release_id)
    fd, temp = tempfile.mkstemp(prefix=".current-", dir=serving)
    with os.fdopen(fd, "w") as handle:
        handle.write(release_id+"\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, serving / "CURRENT")
    _fsync_directory(serving)
    return decision
