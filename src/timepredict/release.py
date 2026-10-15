# 文件作用：AFT 生产发布门禁：训练报告、服务器证据及显式策略都通过才切换 CURRENT。
"""Explicit quality gate and atomic promotion; no bypass flag."""
import hashlib
import math
import os
import shutil
import tempfile
import time
from pathlib import Path

from .features import canonical
from .model import Model, _fsync_directory, _fsync_file, _read_json_object


# 计算文件 SHA-256，用于绑定发布包和证据。
def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class PromotionCommitUncertain(OSError):
    """The pointer was replaced, but its directory durability was not confirmed."""


# 累计来源、校准、样本支持、基线、证据时效和服务器性能检查失败原因。
def evaluate_gate(report, evidence, policy, version):
    failures = []
    if not all(isinstance(value, dict) for value in (report, evidence, policy)):
        return {"decision": "rejected", "failures": ["report, evidence and policy must be objects"], "model_version": version}
    # 保留所有未满足项，不在第一项失败时终止评估。
    def require(test, label):
        if not test:
            failures.append(label)
    # 检查有限数值，并排除布尔值。
    def finite(value):
        try:
            return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
        except OverflowError:
            return False
    def section(value, *keys):
        for key in keys:
            value = value.get(key) if isinstance(value, dict) else None
        return value if isinstance(value, dict) else {}
    def probability(value):
        return finite(value) and 0 <= value <= 1
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
        rows, ess = value.get("supported_rows"), value.get("effective_sample_size")
        valid_rows = type(rows) is int and rows >= policy["min_supported_rows"]
        require(valid_rows, label+": insufficient or invalid samples")
        require(valid_rows and finite(ess) and policy["min_effective_samples"] <= ess <= rows*(1+1e-12),
                label+": insufficient or invalid effective samples")
        actual = value.get("weighted_actual")
        predicted = target if target is not None else value.get("weighted_prediction")
        require(probability(actual) and probability(predicted), label+": invalid calibration")
        brier = value.get("ipcw_brier")
        require(finite(brier) and brier >= 0, label+": invalid Brier score")
        if probability(actual) and probability(predicted):
            require(abs(actual-predicted) <= tolerance, label+": calibration error")
    require(report.get("data_kind") == "observed", "training provenance must be observed")
    require(report.get("label_provenance_checked") is True and
            isinstance(report.get("label_kinds"), dict) and
            set(report["label_kinds"]) == {"sglang_return_observed"}, "formal label evidence must be validated")
    require(report.get("prediction_mode") == "xgboost_aft", "release requires xgboost_aft predictions")
    require(report.get("feature_mode") == "structured" or report.get("embedding_kind") == "sentence_transformers",
            "semantic release requires real Sentence Transformer embeddings")
    if "require_tool_holdout" in policy and type(policy["require_tool_holdout"]) is not bool:
        require(False, "require_tool_holdout must be boolean")
    splits = ["validation", "test"]
    unknown_scopes = []
    if policy.get("require_tool_holdout") is True:
        splits.append("unknown_tool_test")
        limit = policy.get("max_unknown_tool_nll")
        require(finite(limit), "unknown tool NLL limit required")
        require(finite(section(report, "unknown_tool_test").get("nll")) and finite(limit)
                and report["unknown_tool_test"]["nll"] <= limit, "unknown tool NLL exceeds limit")
        tools = report.get("holdout_tools")
        require(isinstance(tools, list) and bool(tools) and all(isinstance(t, str) and t for t in tools),
                "held-out tool names required")
        if isinstance(tools, list) and all(isinstance(t, str) and t for t in tools):
            unknown_scopes = [("unknown_tool/"+name, section(report, "unknown_tool_metrics", name)) for name in tools]
    for split, result in [(name, section(report, name)) for name in splits]+unknown_scopes:
        require(finite(result.get("nll")), split+": invalid likelihood")
        if split.startswith("unknown_tool/"):
            require(finite(result.get("nll")) and finite(policy.get("max_unknown_tool_nll"))
                    and result["nll"] <= policy["max_unknown_tool_nll"], split+": NLL exceeds limit")
        for horizon in policy["horizons"]:
            metric(section(result, "windows").get(str(float(horizon))), f"{split}/H={horizon}", policy["max_window_calibration_error"])
        for q in (0.5, 0.9):
            metric(section(result, "quantile_coverage").get(str(q)), f"{split}/q={q}", policy["max_quantile_coverage_error"], q)
        for age in policy.get("remaining_ages", []):
            for horizon in policy["horizons"]:
                metric(section(result, "remaining_windows", str(float(age))).get(str(float(horizon))),
                       f"{split}/a={age}/H={horizon}", policy["max_window_calibration_error"])
    for horizon in policy["horizons"]:
        name = str(float(horizon))
        measured = section(report, "test", "windows", name).get("ipcw_brier")
        baseline = section(report, "baselines", "per_tool_empirical_windows", name).get("ipcw_brier")
        require(finite(measured) and measured >= 0 and finite(baseline) and baseline >= 0, "Brier comparison unavailable at H="+name)
        if finite(measured) and measured >= 0 and finite(baseline) and baseline >= 0:
            require(measured <= baseline+policy["max_brier_regression"], "Brier regression at H="+name)
    require(evidence.get("kind") == "server_acceptance", "target-server acceptance evidence required")
    require(evidence.get("model_version") == version, "benchmark model version mismatch")
    created = evidence.get("created_at")
    require(finite(created) and 0 <= time.time()-created <= policy["evidence_max_age_seconds"], "acceptance evidence expired or invalid")
    for check in ("sglang_nonstream", "sglang_stream", "cross_ingress_roundtrip", "clock_bounds", "worker_failover", "prediction_observed"):
        require(section(evidence, "checks").get(check) is True, "server check missing: "+check)
    measured = section(evidence, "metrics")
    for key, limit in (("throughput_drop_fraction", policy["max_throughput_drop"]),
                       ("entry_hook_p99_seconds", policy["max_entry_hook_p99_seconds"]),
                       ("prediction_available_p99_seconds", policy["max_prediction_available_p99_seconds"])):
        require(finite(measured.get(key)) and measured[key] <= limit and (key == "throughput_drop_fraction" or measured[key] >= 0), key+": exceeds limit or unavailable")
    require(type(measured.get("request_errors")) is int and measured["request_errors"] == 0, "benchmark request errors or invalid count")
    require(type(measured.get("dropped_events")) is int and measured["dropped_events"] == 0, "event drops or invalid count in benchmark")
    return {"decision": "rejected" if failures else "approved", "failures": failures, "model_version": version}


# 门禁通过后复制候选及证据，验证正式包，再原子替换 CURRENT 文本指针。
def promote(candidate, serving, evidence_file, policy_file, *, before_commit=None):
    candidate, serving = Path(candidate).resolve(), Path(serving).resolve()
    model = Model(candidate)
    report = _read_json_object(candidate / "report.json")
    policy = _read_json_object(policy_file)
    evidence = _read_json_object(evidence_file)
    decision = evaluate_gate(report, evidence, policy, model.version)
    if decision["decision"] != "approved":
        raise ValueError(canonical(decision))
    decision.update(manifest_sha256=sha(candidate / "manifest.json"), evidence_sha256=sha(evidence_file),
                    policy_sha256=sha(policy_file), approved_at=time.time())
    serving.mkdir(parents=True, exist_ok=True)
    release_id = model.version + "-" + hashlib.sha256(canonical(decision).encode()).hexdigest()[:12]
    stage = Path(tempfile.mkdtemp(prefix=".staging-", dir=serving))
    manifest = _read_json_object(candidate / "manifest.json")
    for name in (*manifest["sha256"], "manifest.json"):
        shutil.copyfile(candidate / name, stage / name)
    (stage / "promotion.json").write_text(canonical(decision), encoding="utf-8")
    shutil.copyfile(policy_file, stage / "gate-policy.json")
    shutil.copyfile(evidence_file, stage / "acceptance.json")
    Model(stage, require_promoted=True)
    for path in stage.iterdir():
        if path.is_file():
            _fsync_file(path)
    _fsync_directory(stage)
    os.replace(stage, serving / release_id)
    _fsync_directory(serving)
    fd, temp = tempfile.mkstemp(prefix=".current-", dir=serving)
    with os.fdopen(fd, "w") as handle:
        handle.write(release_id+"\n")
        handle.flush()
        os.fsync(handle.fileno())
    try:
        if before_commit is not None:
            before_commit()
        os.replace(temp, serving / "CURRENT")
    finally:
        if os.path.exists(temp):
            os.unlink(temp)
    try:
        _fsync_directory(serving)
    except OSError as exc:
        raise PromotionCommitUncertain("CURRENT was replaced; publication durability is unconfirmed") from exc
    return decision
