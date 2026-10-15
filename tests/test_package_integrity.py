"""Damaged bundles and malformed gate metrics must fail closed."""
import copy
import hashlib
import json
import shutil
import time

import numpy as np
import pytest
import xgboost as xgb

from timepredict.model import Model, publish
from timepredict.release import evaluate_gate
from timepredict.training import train


@pytest.fixture(scope="module")
def package(tmp_path_factory):
    call = {"name": "search", "arguments": {"n": 2}}
    rows = [{"generation": f"g{i}", "workflow": f"w{i}", "workflow_start": float(i),
             "workflow_end": float(i)+.1, "call": call, "lower": 2.+i%3, "upper": 2.+i%3}
            for i in range(12)]
    root = tmp_path_factory.mktemp("integrity")
    version, _ = train(rows, root, threads=1, rounds=2, sigmas=(.5,), data_kind="synthetic")
    return root/version


def cloned(package, tmp_path):
    root = tmp_path/"candidate"
    shutil.copytree(package, root)
    return root, json.loads((root/"manifest.json").read_text(encoding="utf-8"))


def write_manifest(root, manifest):
    (root/"manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


@pytest.mark.parametrize("filename", ["model.ubj", "features.json", "report.json", "embedding-manifest.json", "dataset-manifest.json"])
def test_required_package_files_cannot_be_omitted_from_hashes(package, tmp_path, filename):
    root, manifest = cloned(package, tmp_path)
    del manifest["sha256"][filename]
    write_manifest(root, manifest)
    with pytest.raises(ValueError):
        Model(root)


@pytest.mark.parametrize("field,value", [("sigma", 0), ("sigma", .7), ("time_scale", 0),
    ("time_scale", 1.), ("best_iteration", -1), ("best_iteration", 100),
    ("best_iteration", True), ("prediction_mode", "typo"), ("version", "../outside"), ("time_scale", None)])
def test_prediction_parameters_must_match_trained_contract(package, tmp_path, field, value):
    root, manifest = cloned(package, tmp_path)
    manifest[field] = value
    write_manifest(root, manifest)
    with pytest.raises(ValueError):
        Model(root)


def test_regression_booster_cannot_masquerade_as_aft(package, tmp_path):
    root, manifest = cloned(package, tmp_path)
    n = Model(package).booster.num_features()
    booster = xgb.train({"objective": "reg:squarederror", "nthread": 1},
                        xgb.DMatrix(np.zeros((2, n)), label=[1., 2.]), num_boost_round=2)
    booster.save_model(root/"model.ubj")
    manifest["sha256"]["model.ubj"] = hashlib.sha256((root/"model.ubj").read_bytes()).hexdigest()
    write_manifest(root, manifest)
    with pytest.raises(ValueError):
        Model(root)


def test_unlisted_action_adapter_cannot_change_prediction(package, tmp_path):
    root, _ = cloned(package, tmp_path)
    (root/"action-families.json").write_text(json.dumps({"representatives": [], "embedding_dim": 0, "embedding_kind": "none"}), encoding="utf-8")
    with pytest.raises(ValueError):
        Model(root)


def test_current_cannot_escape_serving_directory(package, tmp_path):
    serving = tmp_path/"serving"
    serving.mkdir()
    (serving/"CURRENT").write_text(str(package), encoding="utf-8")
    with pytest.raises(ValueError):
        Model(serving)


def test_artifact_changes_produce_different_immutable_versions(package, tmp_path):
    loaded = Model(package)
    report = json.loads((package/"report.json").read_text(encoding="utf-8"))
    common = (tmp_path, loaded.booster, loaded.spec, loaded.sigma, loaded.best_iteration, report)
    a = publish(*common, artifacts={"experiment.txt": "first"})
    b = publish(*common, artifacts={"experiment.txt": "second"})
    assert a != b
    assert (tmp_path/b/"experiment.txt").read_text(encoding="utf-8") == "second"


@pytest.mark.parametrize("name", ["../outside.txt", "..\\outside.txt", "model.ubj", "manifest.json", "promotion.json", "MODEL.UBJ"])
def test_artifacts_cannot_escape_or_overwrite_model_contract(package, tmp_path, name):
    loaded = Model(package)
    report = json.loads((package/"report.json").read_text(encoding="utf-8"))
    with pytest.raises(ValueError):
        publish(tmp_path, loaded.booster, loaded.spec, loaded.sigma, loaded.best_iteration, report,
                artifacts={name: "invalid"})
    assert not (tmp_path/"outside.txt").exists()


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "1e999"])
def test_nonfinite_json_is_rejected_even_in_report_metadata(package, tmp_path, constant):
    root, manifest = cloned(package, tmp_path)
    path = root/"report.json"
    text = path.read_text(encoding="utf-8")
    path.write_text(text[:-1]+', "invalid_metadata": '+constant+'}', encoding="utf-8")
    manifest["sha256"]["report.json"] = hashlib.sha256(path.read_bytes()).hexdigest()
    write_manifest(root, manifest)
    with pytest.raises(ValueError, match="nonfinite_json"):
        Model(root)


def gate_inputs():
    # Explicit synthetic gate fixture: it is not target-server evidence.
    def metric(p):
        return {"status": "ok", "supported_rows": 10, "effective_sample_size": 10.,
                "weighted_actual": p, "weighted_prediction": p, "ipcw_brier": .1}
    split = {"nll": 1., "windows": {"2.0": metric(.5)},
             "quantile_coverage": {"0.5": metric(.5), "0.9": metric(.9)},
             "remaining_windows": {"1.0": {"2.0": metric(.5)}}}
    report = {"data_kind": "observed", "label_provenance_checked": True,
              "label_kinds": {"sglang_return_observed": 20}, "feature_mode": "structured",
              "prediction_mode": "xgboost_aft", "validation": copy.deepcopy(split), "test": copy.deepcopy(split),
              "baselines": {"per_tool_empirical_windows": {"2.0": {"ipcw_brier": .1}}}}
    policy = {"min_supported_rows": 5, "min_effective_samples": 5., "horizons": [2.], "remaining_ages": [1.],
              "max_window_calibration_error": .1, "max_quantile_coverage_error": .1,
              "max_brier_regression": .1, "max_entry_hook_p99_seconds": 1., "max_throughput_drop": .1,
              "max_prediction_available_p99_seconds": 1., "evidence_max_age_seconds": 3600}
    evidence = {"kind": "server_acceptance", "fixture_only": True, "created_at": time.time(), "model_version": "fixture",
                "checks": {k: True for k in ("sglang_nonstream", "sglang_stream", "cross_ingress_roundtrip",
                           "clock_bounds", "worker_failover", "prediction_observed")},
                "metrics": {"throughput_drop_fraction": 0., "entry_hook_p99_seconds": .001,
                            "prediction_available_p99_seconds": .01, "request_errors": 0, "dropped_events": 0}}
    assert evaluate_gate(report, evidence, policy, "fixture")["decision"] == "approved"
    return report, evidence, policy


@pytest.mark.parametrize("field,value", [("supported_rows", float("inf")), ("supported_rows", "10"),
    ("supported_rows", 10.5), ("effective_sample_size", float("inf")), ("effective_sample_size", 100.),
    ("weighted_actual", 2.), ("weighted_prediction", 2.), ("ipcw_brier", -.1)])
def test_malformed_metrics_cannot_pass_gate(field, value):
    report, evidence, policy = gate_inputs()
    metric = report["test"]["windows"]["2.0"]
    metric[field] = value
    if field in ("weighted_actual", "weighted_prediction"):
        metric["weighted_actual"] = metric["weighted_prediction"] = value
    assert evaluate_gate(report, evidence, policy, "fixture")["decision"] == "rejected"


@pytest.mark.parametrize("field", ["request_errors", "dropped_events"])
def test_boolean_counts_are_not_valid_acceptance_measurements(field):
    report, evidence, policy = gate_inputs()
    evidence["metrics"][field] = False
    assert evaluate_gate(report, evidence, policy, "fixture")["decision"] == "rejected"


@pytest.mark.parametrize("field", ["validation", "baselines", "label_kinds"])
def test_malformed_report_sections_are_rejected_without_crashing(field):
    report, evidence, policy = gate_inputs()
    report[field] = None
    assert evaluate_gate(report, evidence, policy, "fixture")["decision"] == "rejected"
