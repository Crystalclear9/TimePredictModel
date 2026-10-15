# 文件作用：旧 AFT 模型的逐轮准确率诊断；要求使用原训练数据，不适用于直接回归包。
"""Reconstruct iteration accuracy for a saved model on its original cohort."""
import argparse
import json
from pathlib import Path

import numpy as np

from .dataset import validate_rows
from .model import Model
from .training import dataset_digest, matrix, split_rows


# 校验数据摘要，重建时间划分，绘制训练/验证曲线和选定轮次的测试点。
def render(rows, model_dir, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model = Model(model_dir)
    report = json.loads((Path(model_dir) / "report.json").read_text(encoding="utf-8"))
    rows = validate_rows(rows)
    if model.action_families:
        for row in rows:
            row["call"] = model.action_families.enrich(row["call"])
    if dataset_digest(rows) != report["data_digest"]:
        raise ValueError("accuracy cohort differs from the original training dataset")
    groups = split_rows(rows)
    results = {"definition": "abs(exp(mu)-T)/T <= tolerance", "point_prediction": "P50",
               "limitations": "Existing proxy tool-duration model; original importer erased non-JSON arguments. Not SGLang return-time accuracy.",
               "model_version": model.version, "best_round": model.best_iteration + 1,
               "selection": "Original validation NLL; test data never selects the round",
               "tolerances": [0.1, 0.2, 0.5], "splits": {}}
    for name in ("train", "validation", "test"):
        cohort = groups[name]
        if any(not np.isfinite(r["upper"]) for r in cohort):
            raise ValueError("exact-label accuracy is unsupported on censored cohorts")
        data = matrix(cohort, model.spec)
        target = np.array([r["lower"] for r in cohort])
        curves = {str(t): [] for t in results["tolerances"]}
        for iteration in range(1, model.booster.num_boosted_rounds() + 1):
            if model.prediction_mode == "family_prior":
                prediction = np.exp(np.array([model.family_priors[r["call"]["action_family"]]["mu"] for r in cohort]))
            else:
                prediction = np.exp(model.booster.predict(data, output_margin=True, iteration_range=(0, iteration)))
            error = np.abs(prediction - target) / target
            for tolerance in results["tolerances"]:
                curves[str(tolerance)].append(float(np.mean(error <= tolerance)))
        results["splits"][name] = {"rows": len(cohort), "accuracy": curves,
                                     "best_round_accuracy": {t: v[model.best_iteration] for t, v in curves.items()}}
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    for axis, tolerance in zip(axes, results["tolerances"]):
        for name, color in (("train", "#2563eb"), ("validation", "#db2777")):
            values = results["splits"][name]["accuracy"][str(tolerance)]
            axis.plot(np.arange(1, len(values)+1), np.array(values)*100, label=name, color=color)
        test = results["splits"]["test"]["best_round_accuracy"][str(tolerance)]*100
        axis.scatter([results["best_round"]], [test], marker="D", color="#15803d", label=f"test at selected round: {test:.1f}%")
        axis.axvline(results["best_round"], color="#777", linestyle="--", linewidth=1)
        axis.set(title=f"Relative error <= {tolerance:.0%}", xlabel="Boosting round", ylim=(0, 100))
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    axes[0].set_ylabel("Accuracy / hit rate (%)")
    fig.suptitle("AgentTrace proxy duration: original model diagnostic (arguments omitted)")
    fig.tight_layout()
    fig.savefig(output / "accuracy_curve.png", dpi=160)
    plt.close(fig)
    (output / "accuracy-report.json").write_text(json.dumps(results, indent=2, allow_nan=False), encoding="utf-8")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    import pyarrow.parquet as pq
    from .datasets import _agenttrace
    rows = [sample for batch in pq.ParquetFile(args.parquet).iter_batches(batch_size=64)
            for trace in batch.to_pylist() for sample in _agenttrace(trace, "agenttrace")]
    result = render(rows, args.model, args.output)
    print(json.dumps({k: v["best_round_accuracy"] for k, v in result["splits"].items()}))
