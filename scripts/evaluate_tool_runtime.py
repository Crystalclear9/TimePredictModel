"""Evaluate recorded Agent tool runtimes with the existing structured AFT pipeline."""

import argparse
import ast
import copy
import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from timepredict.model import Model
from timepredict.training import matrix, split_rows, train


def read_jsonl(path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


def code_features(code):
    """Static workload hints only; never execute user code or inspect its output."""
    features = {"lines": len(code.splitlines()), "bytes": len(code.encode("utf-8"))}
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return {**features, "syntax_valid": False}
    features["syntax_valid"] = True
    counts, calls, imports, numbers = Counter(), Counter(), Counter(), {}
    for node in ast.walk(tree):
        counts[type(node).__name__] += 1
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                imports[
                    (node.module or "") if isinstance(node, ast.ImportFrom) else alias.name
                ] += 1
        if isinstance(node, ast.Call):
            name = ast.unparse(node.func)
            calls[name] += 1
            for kw in node.keywords:
                if kw.arg and not any(x in kw.arg.lower() for x in ("seed", "random_state")):
                    if isinstance(kw.value, ast.Constant) and type(kw.value.value) in (
                        int,
                        float,
                        bool,
                    ):
                        numbers[name + "." + kw.arg] = kw.value.value
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
            for target in node.targets:
                if isinstance(target, ast.Name) and not any(
                    x in target.id.lower() for x in ("seed", "random_state")
                ):
                    if type(node.value.value) in (int, float, bool):
                        numbers[target.id] = node.value.value
    features.update(
        {"nodes": dict(counts), "calls": dict(calls), "imports": dict(imports), "settings": numbers}
    )
    return features


class ExecutionContext:
    """Replay state in order; current and future results never enter the current call."""

    def __init__(self):
        self.experiment_code = {}
        self.dataset_code = ""
        self.dataset = {}
        self.paper = None

    def arguments(self, record):
        arguments = copy.deepcopy(record["arguments"])
        name = record["tool_name"]
        context = {}
        if name == "read_arxiv_paper" and self.paper:
            context["arxiv_id"] = self.paper
        if name in (
            "inspect_selected_dataset",
            "write_and_run_dataset_code",
            "execute_experiment_code",
        ):
            context["dataset"] = copy.deepcopy(self.dataset)
        if name == "execute_experiment_code":
            code = self.experiment_code.get((record["agent"], record["phase"]), "")
            context["code_available"] = bool(code)
            context["experiment_code"] = code
            context["experiment_static"] = code_features(code)
            context["dataset_code"] = self.dataset_code
            context["dataset_static"] = code_features(self.dataset_code)
        if isinstance(arguments.get("code"), str):
            context["code_static"] = code_features(arguments["code"])
        if context:
            arguments["_execution_context"] = context
        return arguments

    def observe(self, record):
        # Only earlier completed calls may update state for a later prediction.
        result = record.get("result") or {}
        if not record.get("success") or not isinstance(result, dict):
            return
        name, arguments = record["tool_name"], record["arguments"]
        key = (record["agent"], record["phase"])
        if name == "write_experiment_code":
            self.experiment_code[key] = arguments["code"]
        elif name == "edit_experiment_code":
            code = self.experiment_code.get(key)
            if code is not None:
                for edit in arguments["edits"]:
                    if edit["old_text"] not in code:
                        raise ValueError("Cannot reconstruct a successful code edit")
                    code = code.replace(edit["old_text"], edit["new_text"])
                self.experiment_code[key] = code
        elif name == "write_and_run_dataset_code":
            self.dataset_code = arguments["code"]
        elif name == "search_arxiv":
            candidates = result.get("candidate_ids", [])
            self.paper = candidates[0] if candidates else None
        elif name == "search_huggingface_datasets":
            selected = result.get("selected_dataset") or {}
            self.dataset = {
                k: selected[k]
                for k in ("dataset_id", "revision", "config_name", "split")
                if k in selected
            }
            for candidate in result.get("results", []):
                if candidate.get("id") == selected.get("dataset_id"):
                    for field in (
                        "repo_bytes",
                        "viewer_parquet_bytes",
                        "estimated_material_bytes",
                        "train_element_size",
                        "test_element_size",
                    ):
                        value = candidate.get(field)
                        if type(value) in (int, float):
                            self.dataset[field] = value
        elif name == "inspect_selected_dataset":
            schema = result.get("dataset_schema") or {}
            if type(schema.get("row_count")) in (int, float):
                self.dataset["row_count"] = schema["row_count"]
            if isinstance(schema.get("columns"), dict):
                self.dataset["column_count"] = len(schema["columns"])


def extract_rows(root, contextual=False):
    rows, sources = [], []
    seen = set()
    for path in sorted(root.rglob("prompts.jsonl")):
        records = read_jsonl(path)
        executions = [r for r in records if r.get("event") == "tool_execution"]
        if not executions:
            continue
        schemas = {}
        for record in records:
            if record.get("event") == "llm_exchange":
                definitions = {
                    s["function"]["name"]: s["function"].get("parameters", {})
                    for s in record.get("tool_schemas", [])
                }
                for call in record.get("returned_tool_calls", []):
                    schemas[call["id"]] = definitions.get(call["name"], {})
        summaries = sorted((path.parent / "traces").glob("*/run_summary.json"))
        if len(summaries) != 1:
            raise ValueError(f"Expected one run summary for {path}")
        summary = json.loads(summaries[0].read_text(encoding="utf-8"))
        start = datetime.fromisoformat(summary["started_at"]).timestamp()
        end = datetime.fromisoformat(summary["ended_at"]).timestamp()
        context = ExecutionContext()
        previous_completion = None
        for record in executions:
            completion = datetime.fromisoformat(record["recorded_at"]).timestamp()
            if previous_completion is not None and completion < previous_completion:
                raise ValueError(f"Execution records out of order: {path}")
            previous_completion = completion
            identity = record["query_id"] + "/" + record["tool_call_id"]
            if identity in seen:
                raise ValueError(f"Duplicate call: {identity}")
            seen.add(identity)
            duration = float(record["duration_seconds"])
            if not np.isfinite(duration) or duration <= 0:
                raise ValueError(f"Invalid duration: {identity}")
            rows.append(
                {
                    "generation": identity,
                    "workflow": record["query_id"],
                    "workflow_start": start,
                    "workflow_end": end,
                    "call": {
                        "name": record["tool_name"],
                        "arguments": context.arguments(record)
                        if contextual
                        else record["arguments"],
                        "schema": schemas.get(record["tool_call_id"], {}),
                    },
                    "lower": duration,
                    "upper": duration,
                    "label_kind": "agent_tool_execution_seconds_local_replay",
                    "success": record["success"],
                    "source": str(path.relative_to(root)),
                }
            )
            context.observe(record)
        replay = summaries[0].with_name("local_replay_summary.json")
        replay_info = json.loads(replay.read_text(encoding="utf-8")) if replay.exists() else {}
        sources.append(
            {
                "path": str(path.relative_to(root)),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "calls": len(executions),
                "replay_latency_mode": replay_info.get("latency_mode"),
                "replay_latency_scale": replay_info.get("latency_scale"),
            }
        )
    return rows, sources


def metrics(actual, prediction):
    actual, prediction = np.asarray(actual), np.asarray(prediction)
    if len(actual) == 0:
        return {"count": 0}
    error = np.abs(prediction - actual)
    relative = error / actual
    return {
        "count": len(actual),
        "accuracy_10pct": float(np.mean(relative <= 0.1)),
        "accuracy_20pct": float(np.mean(relative <= 0.2)),
        "accuracy_50pct": float(np.mean(relative <= 0.5)),
        "mae_seconds": float(np.mean(error)),
        "rmse_seconds": float(np.sqrt(np.mean(error**2))),
        "median_relative_error": float(np.median(relative)),
        "wape": float(error.sum() / actual.sum()),
    }


def experiment_groups(rows, strategy, seed):
    if strategy == "temporal":
        return split_rows(rows)
    if strategy == "all":
        # Deliberate resubstitution: selection and scoring reuse training rows.
        return {"train": rows, "validation": rows, "test": rows, "quarantine": []}
    order = np.random.default_rng(seed).permutation(len(rows))
    first, second = int(len(rows) * 0.7), int(len(rows) * 0.85)
    return {
        "train": [rows[i] for i in order[:first]],
        "validation": [rows[i] for i in order[first:second]],
        "test": [rows[i] for i in order[second:]],
        "quarantine": [],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("dataset"))
    parser.add_argument("--output", type=Path, default=Path("outputs/tool-runtime"))
    parser.add_argument("--split", choices=("temporal", "random", "all"), default="temporal")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--exclude-tool",
        action="append",
        default=[],
        help="Exclude a tool before model fitting and evaluation; repeatable",
    )
    parser.add_argument(
        "--context",
        action="store_true",
        help="Reconstruct pre-execution state and static code features",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows, sources = extract_rows(args.dataset, contextual=args.context)
    input_calls = len(rows)
    # Keep the original split memberships for a paired before/after comparison.
    groups = experiment_groups(rows, args.split, args.seed)
    excluded_tools = set(args.exclude_tool)
    rows = [r for r in rows if r["call"]["name"] not in excluded_tools]
    groups = {
        k: [r for r in v if r["call"]["name"] not in excluded_tools] for k, v in groups.items()
    }
    with (args.output / "samples.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    print(
        "Extracted", len(rows), "calls; split:", {k: len(v) for k, v in groups.items()}, flush=True
    )
    # These are measured execution times from a local replay, not SGLang return labels.
    # Override only this offline experiment's grouping; production splitting is unchanged.
    with patch("timepredict.training.split_rows", return_value=groups):
        version, report = train(
            rows, args.output / "candidate", data_kind="proxy", feature_mode="structured"
        )
    model = Model(args.output / "candidate" / version)
    test = groups["test"]
    mu = model.booster.predict(
        matrix(test, model.spec),
        output_margin=True,
        iteration_range=(0, model.best_iteration + 1),
    ) - np.log(model.time_scale)
    prediction = np.exp(mu)
    actual = np.array([r["lower"] for r in test])
    train_times = defaultdict(list)
    for row in groups["train"]:
        train_times[row["call"]["name"]].append(row["lower"])
    global_median = np.median([r["lower"] for r in groups["train"]])
    baseline = np.array(
        [
            np.median(train_times[r["call"]["name"]])
            if r["call"]["name"] in train_times
            else global_median
            for r in test
        ]
    )
    detail = []
    for row, predicted, base in zip(test, prediction, baseline):
        detail.append(
            {
                "query": row["workflow"],
                "tool_call_id": row["generation"].split("/", 1)[1],
                "tool": row["call"]["name"],
                "success": row["success"],
                "actual_seconds": row["lower"],
                "predicted_p50_seconds": float(predicted),
                "baseline_seconds": float(base),
                "relative_error": float(abs(predicted - row["lower"]) / row["lower"]),
            }
        )
    with (args.output / "predictions.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(detail[0]))
        writer.writeheader()
        writer.writerows(detail)
    per_tool = {}
    for tool in sorted({r["call"]["name"] for r in test}):
        mask = np.array([r["call"]["name"] == tool for r in test])
        per_tool[tool] = metrics(actual[mask], prediction[mask])
    per_query = {}
    for query in sorted({r["workflow"] for r in test}):
        mask = np.array([r["workflow"] == query for r in test])
        per_query[query] = metrics(actual[mask], prediction[mask])
    summary = {
        "target": "Agent tool execution duration_seconds; inclusive of nested work; excludes LLM exchange",
        "accuracy_definition": "abs(predicted_P50 - actual) / actual <= tolerance",
        "data_scope": "One row per prompts.jsonl tool_execution; lower-level tool_trace not duplicated",
        "data_kind": "local replay with measured execution duration; proxy for production tool return time",
        "features": "Existing structured tool-name and argument/schema features; no result, success, duration, IDs or phase as inputs",
        "context_features": args.context,
        "excluded_tools": sorted(excluded_tools),
        "input_calls_before_exclusion": input_calls,
        "excluded_calls": input_calls - len(rows),
        "exclusion_policy": "Preserve original split memberships, then remove excluded tools before fitting features and models",
        "selection": (
            "Existing AFT defaults; sigma and stopping round selected on reused training NLL"
            if args.split == "all"
            else "Existing AFT defaults; sigma and stopping round selected only on validation NLL"
        ),
        "split_strategy": args.split,
        "split_seed": args.seed if args.split == "random" else None,
        "evaluation_kind": "training_data_resubstitution"
        if args.split == "all"
        else "held_out_calls",
        "query_overlap_allowed": args.split != "temporal",
        "split": {
            k: {"calls": len(v), "queries": sorted({r["workflow"] for r in v})}
            for k, v in groups.items()
        },
        "total_calls": len(rows),
        "tool_counts": dict(Counter(r["call"]["name"] for r in rows)),
        "failure_calls_included": sum(not r["success"] for r in rows),
        "candidate_path": str((args.output / "candidate" / version).resolve()),
        "best_round": model.best_iteration + 1,
        "sigma": model.sigma,
        "test": metrics(actual, prediction),
        "per_tool_median_baseline": metrics(actual, baseline),
        "test_duration_at_least_1ms": metrics(actual[actual >= 0.001], prediction[actual >= 0.001]),
        "test_duration_at_least_100ms": metrics(actual[actual >= 0.1], prediction[actual >= 0.1]),
        "test_success_only": metrics(
            actual[[r["success"] for r in test]], prediction[[r["success"] for r in test]]
        ),
        "per_tool": per_tool,
        "per_query": per_query,
        "sources": sources,
    }
    (args.output / "metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), layout="constrained")
    colors = {"train": "#64748b", "validation": "#ea580c", "test": "#2563eb"}
    evaluation_label = "Training data (reused)" if args.split == "all" else "Held-out test"
    for name in ["train"] if args.split == "all" else colors:
        values = report["accuracy_curves"][name]["curves"]["0.5_0.2"]
        axes[0, 0].plot(
            np.arange(1, len(values) + 1), np.array(values) * 100, label=name, color=colors[name]
        )
    axes[0, 0].axvline(
        model.best_iteration + 1,
        color="#222",
        linestyle="--",
        label="training-selected round" if args.split == "all" else "validation-selected round",
    )
    axes[0, 0].set(
        title="Accuracy by boosting round (relative error <= 20%)",
        xlabel="Boosting round",
        ylabel="Accuracy (%)",
        ylim=(0, 100),
    )
    axes[0, 0].legend(fontsize=8)
    relative = np.abs(prediction - actual) / actual
    base_relative = np.abs(baseline - actual) / actual
    tolerances = np.linspace(0, 1, 201)
    axes[0, 1].plot(
        tolerances * 100,
        [np.mean(relative <= t) * 100 for t in tolerances],
        label="AFT P50",
        color="#2563eb",
    )
    axes[0, 1].plot(
        tolerances * 100,
        [np.mean(base_relative <= t) * 100 for t in tolerances],
        label="Train per-tool median",
        color="#64748b",
    )
    axes[0, 1].scatter([20], [summary["test"]["accuracy_20pct"] * 100], color="#2563eb")
    axes[0, 1].set(
        title=f"{evaluation_label} accuracy vs tolerance",
        xlabel="Allowed relative error (%)",
        ylabel="Accuracy (%)",
        ylim=(0, 100),
    )
    axes[0, 1].legend()
    for query in per_query:
        mask = np.array([r["workflow"] == query for r in test])
        axes[1, 0].scatter(actual[mask], prediction[mask], alpha=0.65, s=22, label=query)
    limits = [min(actual.min(), prediction.min()) * 0.5, max(actual.max(), prediction.max()) * 2]
    axes[1, 0].plot(limits, limits, color="#555", linestyle="--")
    axes[1, 0].set(
        xscale="log",
        yscale="log",
        xlim=limits,
        ylim=limits,
        title=f"{evaluation_label} predictions vs actual duration",
        xlabel="Actual seconds (log)",
        ylabel="Predicted P50 seconds (log)",
    )
    if len(per_query) <= 4:
        axes[1, 0].legend()
    labels = [f"{tool} (n={value['count']})" for tool, value in per_tool.items()]
    axes[1, 1].barh(
        labels, [value["accuracy_20pct"] * 100 for value in per_tool.values()], color="#2563eb"
    )
    axes[1, 1].set(
        title=f"{evaluation_label} accuracy by tool (error <= 20%)",
        xlabel="Accuracy (%)",
        xlim=(0, 100),
    )
    axes[1, 1].tick_params(axis="y", labelsize=8)
    for ax in axes.flat:
        ax.grid(alpha=0.15)
    fig.suptitle(
        f"Tool runtime | {args.split} split | {len(test)} scored calls | +/-20% accuracy: {summary['test']['accuracy_20pct']:.2%}",
        fontsize=14,
    )
    fig.savefig(args.output / "accuracy.png", dpi=180)
    plt.close(fig)
    print(
        json.dumps(
            {
                k: v
                for k, v in summary.items()
                if k
                in (
                    "test",
                    "per_tool_median_baseline",
                    "test_duration_at_least_1ms",
                    "test_duration_at_least_100ms",
                    "best_round",
                    "sigma",
                )
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
