# 文件作用：精确标签的直接 log(T) 回归：P50 使用平方误差，P90 使用独立分位数目标；不提供完整生存分布。
"""Direct log-time XGBoost regression for observed replay labels.

This is deliberately separate from the AFT service: quantile regressors do
not define a complete survival distribution and must not be used by the
remaining-time API without a calibrated distribution layer.
"""
import json
import math
import hashlib
import os
import shutil
import tempfile
import uuid
from pathlib import Path

import numpy as np
import xgboost as xgb

from .actions import ActionFamilies
from .dataset import validate_rows
from .features import FeatureSpec
from .training import split_rows
from .semantic import SemanticEncoder


# 转换冻结特征，并以 log(lower) 作为精确回归标签。
def _dmatrix(rows, spec):
    x = np.stack([spec.transform(r["call"]) for r in rows])
    y = np.log(np.asarray([r["lower"] for r in rows], dtype=np.float32))
    return xgb.DMatrix(x, label=y)


# 把对数预测还原为秒，统计相对误差不超过 10%/20%/50% 的命中率。
def _accuracy(rows, pred):
    target = np.asarray([r["lower"] for r in rows], dtype=float)
    rel = np.abs(np.exp(pred) - target) / target
    return {str(t): float(np.mean(rel <= t)) for t in (0.1, 0.2, 0.5)}


# 计算给定分位数在 log(T) 空间的非对称 pinball loss。
def _pinball(rows, pred, q):
    y = np.log(np.asarray([r["lower"] for r in rows], dtype=float))
    e = y - pred
    return float(np.mean(np.maximum(q * e, (q - 1) * e)))


# 校验精确标签与真实编码器，按工作流划分，训练/校准两个模型并原子保存诊断包。
def train_regression(rows, output, threads=4, rounds=500,
                     allowed_label_kind="observed_replay", data_kind="synthetic_replay"):
    if os.environ.get("TIMEPREDICT_TEST_ENCODER") == "1":
        raise ValueError("test_encoder_forbidden_for_regression_training")
    if not 1 <= rounds <= 500 or threads < 1:
        raise ValueError("rounds must be 1..500 and threads positive")
    encoder = SemanticEncoder()
    if encoder.backend != "sentence_transformers":
        raise ValueError("real_encoder_required")
    rows = validate_rows(rows)
    kinds = {r.get("label_kind", "observed") for r in rows}
    if kinds != {allowed_label_kind}:
        raise ValueError("regression_label_kind_mismatch")
    if any(not math.isfinite(r["upper"]) for r in rows):
        raise ValueError("point_training_requires_exact_labels; use AFT for censored data")
    for row in rows:
        if allowed_label_kind == "observed_replay":
            b = row.get("boundaries", {})
            stamps = [b.get(k) for k in ("t_out", "t_tool_start", "t_tool_end", "t_return")]
            if any(not isinstance(x, (int, float)) or not math.isfinite(x) for x in stamps) or stamps != sorted(stamps):
                raise ValueError("invalid_replay_boundaries")
            if not math.isclose(stamps[-1] - stamps[0], row["lower"], rel_tol=1e-9):
                raise ValueError("label_boundary_mismatch")
    groups = split_rows(rows)
    families = ActionFamilies.fit(r["call"] for r in groups["train"])
    for group in groups.values():
        for row in group:
            row["call"] = families.enrich(row["call"])
    spec = FeatureSpec.fit((r["call"] for r in groups["train"]), feature_mode="legacy_semantic")
    train_data, valid_data = _dmatrix(groups["train"], spec), _dmatrix(groups["validation"], spec)
    boosters, curves, scores = {}, {}, {}
    # P50 is the point predictor.  P90 is an independent quantile model.
    # Keeping these objectives explicit prevents an AFT or quantile model from
    # silently becoming the serving median.
    for q in (0.5, 0.9):
        params = ({"objective": "reg:squarederror"} if q == 0.5 else
                  {"objective": "reg:quantileerror", "quantile_alpha": q})
        params.update({
                  "tree_method": "hist", "max_depth": 4, "eta": 0.05,
                  "seed": 42, "nthread": threads, "device": "cpu"})
        evaluation = {}
        booster = xgb.train(params, train_data, num_boost_round=rounds,
                            evals=[(train_data, "train"), (valid_data, "validation")],
                            evals_result=evaluation, early_stopping_rounds=30,
                            verbose_eval=False)
        boosters[str(q)] = booster
        curves[str(q)] = evaluation
        scores[str(q)] = {"best_iteration": booster.best_iteration,
                          "validation_metric": float(booster.best_score),
                          "objective": params["objective"]}
    # Calibrate each conditional log-quantile on validation residuals.  This
    # preserves the tree models while correcting finite-sample coverage.
    validation_matrix = _dmatrix(groups["validation"], spec)
    validation_log_y = np.log(np.asarray([r["lower"] for r in groups["validation"]], dtype=float))
    calibration_offsets = {}
    for q in (0.5, 0.9):
        raw = boosters[str(q)].predict(validation_matrix,
                                        iteration_range=(0, boosters[str(q)].best_iteration + 1))
        residuals = validation_log_y - raw
        try:
            calibration_offsets[str(q)] = float(np.quantile(residuals, q, method="higher"))
        except TypeError:
            calibration_offsets[str(q)] = float(np.quantile(residuals, q, interpolation="higher"))
    # 使用选定轮次及验证残差偏移预测；P90 按现有实现与原始 P50 比较取最大。
    def predict(group, q):
        data = _dmatrix(group, spec)
        model = boosters[str(q)]
        values = model.predict(data, iteration_range=(0, model.best_iteration + 1)) + calibration_offsets[str(q)]
        if q == .9:
            median = boosters["0.5"]
            values = np.maximum(values, median.predict(data, iteration_range=(0, median.best_iteration + 1)))
        return values
    report = {"model_type": "xgboost_direct_log_time", "label_kind": allowed_label_kind,
              "split_counts": {k: len(v) for k, v in groups.items()}, "quantile_trials": scores,
              "dataset": {"rows": len(rows), "workflows": len({r["workflow"] for r in rows})},
              "production_releasable": False, "data_kind": data_kind, "acceptance": {},
              "embedding": encoder_contract(),
              "split_workflows": {k: len({r['workflow'] for r in v}) for k, v in groups.items()},
              "selection_protocol": "validation pinball; early stopping 30; test once at selected iterations",
              "distribution": "not_available_from_two_quantiles; AFT remains separate",
              "calibration_status": "quantile_only",
              "calibration_reason": "validation residual calibration does not define a survival distribution",
              "calibration_offsets_log_seconds": calibration_offsets}
    report["metrics"] = {}
    for name, group in groups.items():
        if name == "quarantine":
            continue
        p50_pred, p90_pred = predict(group, 0.5), predict(group, 0.9)
        report["metrics"][name] = {"rows": len(group), "p50_accuracy": _accuracy(group, p50_pred),
                                   "p90_accuracy": _accuracy(group, p90_pred),
                                   "p50_log_mae": float(np.mean(np.abs(p50_pred - np.log([r["lower"] for r in group])))),
                                   "p90_pinball": _pinball(group, p90_pred, .9),
                                   "p50_pinball": _pinball(group, p50_pred, .5),
                                   "coverage": {"0.5": float(np.mean(np.log([r['lower'] for r in group]) <= p50_pred)),
                                                "0.9": float(np.mean(np.log([r['lower'] for r in group]) <= p90_pred))}}
        report["metrics"][name]["p90_quantile_diagnostics"] = {
            "coverage": report["metrics"][name]["coverage"]["0.9"],
            "target_coverage": 0.9,
            "coverage_error": report["metrics"][name]["coverage"]["0.9"] - 0.9,
            "underprediction_rate": float(np.mean(np.log([r["lower"] for r in group]) > p90_pred)),
            "interpretation": "P90 is evaluated by coverage and pinball loss; relative-error accuracy is diagnostic only"
        }
    test_acc = report["metrics"]["test"]["p50_accuracy"]["0.1"]
    baseline = float(np.median([r["lower"] for r in groups["train"]]))
    report["metrics"]["test"]["global_median_accuracy"] = _accuracy(groups["test"], np.full(len(groups["test"]), math.log(baseline)))
    by_tool = {name: float(np.median([r['lower'] for r in groups['train'] if r['call']['name'] == name]))
               for name in {r['call']['name'] for r in groups['train']}}
    tool_baseline = _accuracy(groups['test'], np.log([by_tool.get(r['call']['name'], baseline) for r in groups['test']]))
    report['metrics']['test']['per_tool_median_accuracy'] = tool_baseline
    report['square_error_diagnostic'] = {'objective': 'reg:squarederror', 'serving_p50': True}
    test_pred = predict(groups['test'], .5)
    report['by_tool'] = {}
    for name in sorted({r['call']['name'] for r in groups['test']}):
        ix = [i for i, r in enumerate(groups['test']) if r['call']['name'] == name]
        report['by_tool'][name] = {'rows': len(ix), 'accuracy': _accuracy([groups['test'][i] for i in ix], test_pred[ix])}
    report["acceptance"] = {"threshold": .60, "p50_accuracy_10pct": test_acc,
                            "passed": bool(test_acc >= .60 and test_acc > tool_baseline['0.1'] and test_acc > report["metrics"]["test"]["global_median_accuracy"]["0.1"])}
    curves = {}
    for q, model in boosters.items():
        curves[q] = {}
        for split in ("train", "validation"):
            group = groups[split]; data = _dmatrix(group, spec); target = np.asarray([r["lower"] for r in group])
            vals = []
            for iteration in range(1, model.num_boosted_rounds() + 1):
                pred = model.predict(data, iteration_range=(0, iteration))
                vals.append({str(t): float(np.mean(np.abs(np.exp(pred) - target) / target <= t)) for t in (.1, .2, .5)})
            curves[q][split] = vals
    report["accuracy_curves"] = curves
    output = Path(output).absolute()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not output.is_symlink():
        raise ValueError("output must be absent or an existing managed symlink")
    staging = Path(tempfile.mkdtemp(prefix="." + output.name + "-staging-", dir=output.parent))
    for q, model in boosters.items():
        model.save_model(staging / f"p{int(float(q)*100)}-model.ubj")
    (staging / "features.json").write_text(json.dumps(spec.to_dict(), indent=2), encoding="utf-8")
    (staging / "action-families.json").write_text(json.dumps(families.to_dict(), indent=2), encoding="utf-8")
    (staging / "embedding-manifest.json").write_text(json.dumps(report["embedding"], indent=2), encoding="utf-8")
    report["feature_sha256"] = hashlib.sha256((staging / "features.json").read_bytes()).hexdigest()
    (staging / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    manifest = {"package_version": 2, "model_type": report["model_type"], "label_kind": report["label_kind"],
                "data_kind": data_kind, "production_releasable": False,
                "feature_version": spec.version, "xgboost_version": xgb.__version__,
                "acceptance": report["acceptance"], "best_iterations": scores,
                "embedding": report['embedding'], "feature_sha256": report['feature_sha256'],
                "calibration_status": report["calibration_status"],
                "calibration_offsets_log_seconds": calibration_offsets}
    from .training import dataset_digest
    (staging / 'dataset-manifest.json').write_text(json.dumps({
        'label_kind': allowed_label_kind, 'normalized_sha256': dataset_digest(rows),
        'split_generations': {k: [r['generation'] for r in v] for k, v in groups.items()},
        'production_releasable': False, 'split_counts': report['split_counts']}, indent=2))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
        for axis, tolerance in zip(axes, (.1, .2, .5)):
            for q, color in (("0.5", "#2563eb"), ("0.9", "#dc2626")):
                for split, style in (("train", "-"), ("validation", "--")):
                    vals = [x[str(tolerance)] * 100 for x in curves[q][split]]
                    axis.plot(range(1, len(vals)+1), vals, color=color, linestyle=style, label=f"P{int(float(q)*100)} {split}")
                test = report['metrics']['test'][f'p{int(float(q)*100)}_accuracy'][str(tolerance)]
                axis.scatter([scores[q]['best_iteration'] + 1], [100 * test], color=color, marker='D')
            axis.set_title(f"relative error ≤ {tolerance:.0%}"); axis.set_xlabel("boosting round"); axis.set_ylim(0, 100); axis.grid(alpha=.2)
        axes[0].set_ylabel("accuracy (%)"); axes[0].legend(fontsize=7)
        fig.suptitle(f"Measured synthetic replay (not production); n={len(rows)}, test n={len(groups['test'])}")
        fig.tight_layout(); fig.savefig(staging / "accuracy_curve.png", dpi=160); plt.close(fig)
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot([0, 1], [0, 1], '--', color='gray')
        ax.plot([.5, .9], list(report['metrics']['test']['coverage'].values()), 'o-')
        ax.set(xlabel='Nominal quantile', ylabel='Observed test coverage', title='Replay quantile coverage', xlim=(0, 1), ylim=(0, 1))
        fig.tight_layout(); fig.savefig(staging / 'calibration_curve.png', dpi=160); plt.close(fig)
    except Exception:
        shutil.rmtree(staging)
        raise
    try:
        manifest['sha256'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in staging.iterdir() if p.is_file()}
        (staging / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        for path in staging.iterdir():
            if path.is_file():
                with path.open('ab') as handle:
                    os.fsync(handle.fileno())
        restored = RegressionModel(staging)
        expected = np.exp([predict(groups['test'][:1], .5)[0], predict(groups['test'][:1], .9)[0]])
        if not np.allclose(restored.quantiles(groups['test'][0]['call']), expected):
            raise ValueError('reload_prediction_mismatch')
        final = output.parent / ('.' + output.name + '-version-' + uuid.uuid4().hex)
        old = output.resolve() if output.is_symlink() else None
        staging.rename(final)
        pointer = output.parent / ('.' + output.name + '-pointer-' + uuid.uuid4().hex)
        try:
            try:
                pointer.symlink_to(final.name, target_is_directory=True)
                os.replace(pointer, output)
            except OSError:
                # Developer mode is commonly disabled on Windows. Linux uses
                # the symlink pointer; local Windows validation falls back to
                # an atomic directory rename and still verifies the package.
                if output.exists():
                    raise
                os.replace(final, output)
        finally:
            pointer.unlink(missing_ok=True)
        from .model import _fsync_directory
        _fsync_directory(output.parent)
        if old and old.parent == output.parent and old.name.startswith('.' + output.name + '-version-'):
            shutil.rmtree(old)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return report


# 仅从本地缓存读取固定 revision 的编码器，记录配置/词表/权重摘要。
def encoder_contract():
    from huggingface_hub import snapshot_download
    encoder = SemanticEncoder()
    if encoder.backend != 'sentence_transformers':
        raise ValueError('real_encoder_required')
    root = Path(snapshot_download(encoder.model_name, revision=encoder.revision,
                                 cache_dir=encoder.cache_folder, local_files_only=True))
    files = {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted(root.rglob('*')) if p.is_file() and p.suffix in ('.json', '.txt', '.safetensors', '.bin')}
    if not any(k.endswith(('.safetensors', '.bin')) for k in files):
        raise ValueError('encoder_weights_missing')
    return {**encoder.to_dict(), 'files_sha256': files}


# 独立恢复双模型包并验证文件及编码器契约。
class RegressionModel:
    # 校验必需文件、SHA-256、XGBoost 主版本与编码器，再载入两个 booster。
    def __init__(self, directory):
        root = Path(directory).resolve()
        self.manifest = json.loads((root / 'manifest.json').read_text(encoding="utf-8"))
        m = self.manifest
        if m.get('package_version') != 2 or m.get('model_type') != 'xgboost_direct_log_time':
            raise ValueError('regression_package_incompatible')
        required = {'p50-model.ubj', 'p90-model.ubj', 'features.json', 'action-families.json', 'embedding-manifest.json', 'report.json', 'dataset-manifest.json'}
        if not required <= m.get('sha256', {}).keys():
            raise ValueError('package_files_missing')
        for name, digest in m['sha256'].items():
            if Path(name).name != name or hashlib.sha256((root / name).read_bytes()).hexdigest() != digest:
                raise ValueError('model_checksum_mismatch')
        if m['xgboost_version'].split('.')[0] != xgb.__version__.split('.')[0]:
            raise ValueError('xgboost_major_version_mismatch')
        if m['embedding'] != encoder_contract():
            raise ValueError('encoder_contract_mismatch')
        self.spec = FeatureSpec(**json.loads((root / 'features.json').read_text(encoding="utf-8")))
        package_embedding = json.loads((root / 'embedding-manifest.json').read_text(encoding="utf-8"))
        if package_embedding != m['embedding']:
            raise ValueError('embedding_manifest_mismatch')
        self.families = ActionFamilies(**json.loads((root / 'action-families.json').read_text(encoding="utf-8")))
        self.boosters = {}
        self.calibration_offsets = {str(k): float(v) for k, v in m.get('calibration_offsets_log_seconds', {}).items()}
        for q in (.5, .9):
            b = xgb.Booster()
            b.load_model(root / f'p{int(q*100)}-model.ubj')
            self.boosters[q] = b

    # 匹配动作族并转换特征；加校准偏移、指数还原，最终保证返回 P90≥P50。
    def quantiles(self, call):
        call = self.families.enrich(call)
        data = xgb.DMatrix(self.spec.transform(call)[None, :])
        logs = [self.boosters[q].predict(data, iteration_range=(0, self.manifest['best_iterations'][str(q)]['best_iteration']+1))[0] + self.calibration_offsets.get(str(q), 0.0) for q in (.5, .9)]
        return np.exp([logs[0], max(logs)])


# 独立加载模型并核对标签类型；仅评估精确样本，报告排除的删失数量。
def evaluate_model(directory, rows):
    model = RegressionModel(directory)
    rows = validate_rows(rows)
    if {r.get('label_kind') for r in rows} != {model.manifest['label_kind']}:
        raise ValueError('evaluation_label_mismatch')
    exact = [r for r in rows if r['lower'] == r['upper']]
    if not exact:
        return {'status': 'not_applicable', 'excluded_censored': len(rows)}
    pred = np.asarray([model.quantiles(r['call']) for r in exact])
    return {'rows': len(exact), 'excluded_censored': len(rows)-len(exact),
            'p50_accuracy': _accuracy(exact, np.log(pred[:, 0])),
            'p90_accuracy': _accuracy(exact, np.log(pred[:, 1])), 'production_releasable': False}
