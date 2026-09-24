"""Run independent public proxy experiments without cross-label ranking."""
import json
from pathlib import Path

from .datasets import DATASETS, load_proxy_rows, load_semantic_rows
from .training import train


def run(output, cache_dir=None, threads=4, rounds=500, max_rows=512):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    reports = {}
    for name in ("agenttrace", "tracecommons", "betterwright"):
        try:
            rows = load_proxy_rows(name, cache_dir, max_rows=max_rows)
            candidate = output / (name + "-candidate")
            version, report = train(rows, candidate, threads=threads, rounds=rounds,
                                    data_kind="proxy")
            reports[name] = {"status": "completed", "source_dataset": name,
                             "label_kind": report.get("label_kinds", {}),
                             "rows": len(rows), "workflow_count": len({r["workflow"] for r in rows}),
                             "validation_nll": report.get("validation_model_nll"),
                             "test": report.get("point_accuracy"),
                             "production_releasable": False, "model_path": str(candidate / version)}
        except (ValueError, RuntimeError, OSError) as exc:
            reports[name] = {"status": "failed", "error": str(exc),
                             "production_releasable": False}
    try:
        semantic = load_semantic_rows("exgentic", cache_dir, max_rows=max_rows)
        reports["exgentic"] = {"status": "semantic_only", "rows": len(semantic),
                                "label_kind": "none", "no_valid_return_boundary": len(semantic),
                                "production_releasable": False}
    except (ValueError, RuntimeError, OSError) as exc:
        reports["exgentic"] = {"status": "semantic_only", "error": str(exc),
                                "production_releasable": False}
    summary = {"status": "completed_independent_reports",
               "selection": "none; proxy labels are not comparable", "reports": reports,
               "revisions": {key: value[1] for key, value in DATASETS.items()},
               "production_releasable": False}
    (output / "experiment-report.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return summary
