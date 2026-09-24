import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import xgboost as xgb
from scipy.optimize import minimize
from scipy.special import log_ndtr, ndtr, ndtri

from .clock import elapsed
from .dataset import validate_rows, describe, validate_label_provenance
from .features import FeatureSpec, UnsupportedInput, extract
from .model import publish, versions
from .state import empty_state, reduce_event, workflow_key
from .actions import ActionFamilies


def _curve_svg(series, title):
    """Small dependency-free SVG plot retained with the best candidate."""
    width, height, left, top, right, bottom = 900, 500, 70, 55, 25, 55
    points = [(name, [float(v) for v in values if v is not None and math.isfinite(float(v))]) for name, values in series.items() if values and any(v is not None for v in values)]
    length = max((len(v) for _, v in points), default=1)
    values = [x for _, v in points for x in v if math.isfinite(x)] or [0.0, 1.0]
    low, high = min(values), max(values)
    if high <= low:
        high = low + 1.0
    def path(values):
        coords = []
        for i, value in enumerate(values):
            x = left + (width-left-right) * (i / max(1, length-1))
            y = top + (height-top-bottom) * (high-value) / (high-low)
            coords.append(f"{x:.2f},{y:.2f}")
        return "M " + " L ".join(coords)
    colors = ("#2563eb", "#dc2626", "#16a34a", "#9333ea")
    lines = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
             '<rect width="100%" height="100%" fill="white"/>', f'<text x="{left}" y="30" font-size="20">{title}</text>',
             f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#111"/>',
             f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#111"/>']
    for index, (name, vals) in enumerate(points):
        color = colors[index % len(colors)]
        lines.append(f'<path d="{path(vals)}" fill="none" stroke="{color}" stroke-width="2"/>')
        lines.append(f'<text x="{left + index*170}" y="{height-18}" fill="{color}" font-size="14">{name}</text>')
    lines.append("</svg>")
    return "\n".join(lines)


def snapshot(events, cutoff, heartbeat_ttl=30):
    if not math.isfinite(cutoff) or heartbeat_ttl <= 0:
        raise ValueError("cutoff must be finite and heartbeat TTL positive")
    state = empty_state()
    excluded = Counter()
    for event in events:
        if event.get("observed_at", event["stamp"]["utc"]) <= cutoff and event["stamp"]["utc"] <= cutoff:
            state, _ = reduce_event(state, event, copy_state=False)
    rows = []
    for record in state["records"].values():
        wf = workflow_key(record)
        if not record["valid"]:
            excluded[record["status"]] += 1
            continue
        if record["status"] == "returned":
            lower = upper = record["duration"]
        elif record["status"] == "waiting":
            active_sources = [s for s in state["sources"].values() if not s.get("retired")]
            if any(s.get("bad") or s.get("lost") or cutoff-s["last"]["utc"] > heartbeat_ttl for s in active_sources):
                excluded["incomplete_observation"] += 1
                continue
            source = state["sources"].get(record["out"]["source"] + "/" + record["out"]["boot"], {})
            last = source.get("last", {}).get("utc", -math.inf)
            if source.get("bad") or source.get("lost") or source.get("retired") or cutoff - last > heartbeat_ttl or cutoff - state["heartbeats"].get(record["key"], -math.inf) > heartbeat_ttl:
                excluded["incomplete_observation"] += 1
                continue
            # Administrative cutoff uses UTC and requires a measured clock bound.
            stamps = [record["out"], source.get("last", {}), state.get("heartbeat_stamps", {}).get(record["key"], {})]
            if any(s.get("error") is None or not math.isfinite(s["error"]) or not 0 <= s["error"] <= 0.0025 or s.get("jumped") for s in stamps):
                excluded["clock_uncertain"] += 1
                continue
            lower, upper = cutoff - record["out"]["utc"], math.inf
        else:
            excluded[record["status"]] += 1
            continue
        if lower <= 0:
            excluded["nonpositive_interval"] += 1
            continue
        try:
            extract(record["call"])
        except UnsupportedInput:
            excluded["unsupported_input"] += 1
            continue
        rows.append({"generation": record["generation"], "workflow": wf, "workflow_start": state["workflows"][wf]["start"],
                     "workflow_end": state["workflows"][wf]["end"], "call": record["call"], "lower": lower, "upper": upper})
    return rows, dict(excluded)


def split_rows(rows):
    starts = sorted({r["workflow_start"] for r in rows})
    if len(starts) < 7:
        raise ValueError("need at least 7 distinct workflow start times for temporal validation")
    first, second = starts[int(len(starts) * 0.70)], starts[int(len(starts) * 0.85)]
    groups = {"train": [], "validation": [], "test": [], "quarantine": []}
    for row in rows:
        start, end = row["workflow_start"], row["workflow_end"]
        bucket = "train" if start < first else "validation" if start < second else "test"
        boundary = first if bucket == "train" else second
        if bucket != "test" and (end is None or end >= boundary):
            bucket = "quarantine"
        groups[bucket].append(row)
    if any(not groups[k] for k in ("train", "validation", "test")):
        raise ValueError("empty temporal split after crossing-workflow quarantine")
    return groups


def matrix(rows, spec, time_scale=1.0):
    data = xgb.DMatrix(np.stack([spec.transform(row["call"]) for row in rows]))
    data.set_float_info("label_lower_bound", np.array([r["lower"] * time_scale for r in rows]))
    data.set_float_info("label_upper_bound", np.array([r["upper"] * time_scale for r in rows]))
    return data


def nll(rows, mu, sigma):
    t = np.array([r["lower"] for r in rows])
    observed = np.array([math.isfinite(r["upper"]) for r in rows])
    z = (np.log(t) - mu) / sigma
    losses = np.where(observed, np.log(t) + math.log(sigma) + 0.5 * math.log(2 * math.pi) + z*z/2, -log_ndtr(-z))
    return float(np.mean(losses))


def km_curve(rows, censor=False):
    if not rows:
        raise ValueError("empty survival cohort")
    observed_times = np.asarray([r["lower"] for r in rows], dtype=np.float64)
    events = np.asarray([not math.isfinite(r["upper"]) if censor else math.isfinite(r["upper"]) for r in rows], dtype=np.float64)
    times, inverse, counts = np.unique(observed_times, return_inverse=True, return_counts=True)
    risks = len(rows) - np.r_[0, np.cumsum(counts[:-1])]
    deaths = np.bincount(inverse, weights=events, minlength=len(times))
    values = np.cumprod(1.0 - deaths / risks)
    def at(t, before=False):
        idx = np.searchsorted(times, t, side="left" if before else "right") - 1
        return float(1.0 if idx < 0 else values[idx])
    return at


def ipcw_score(rows, probabilities, thresholds, censor_curve=None):
    g = censor_curve or km_curve(rows, censor=True)
    total, weight, actual, predicted = 0., 0., 0., 0.
    supported, low_support, squared_weight = 0, 0, 0.
    for row, p, h in zip(rows, probabilities, thresholds):
        p, h = float(p), float(h)
        t, observed = row["lower"], math.isfinite(row["upper"])
        if observed and t <= h:
            denom, y = g(t, True), 1.
        elif t > h:
            denom, y = g(h), 0.
        else:
            continue
        if denom < 0.05:
            low_support += 1
            continue
        w = 1 / denom
        total += w * (y-p)**2
        weight += w
        squared_weight += w*w
        actual += w*y
        predicted += w*p
        supported += 1
    valid = supported > 0 and low_support == 0
    return {"status": "ok" if valid else "unsupported_tail" if low_support else "insufficient_support",
            "ipcw_brier": total / len(rows) if valid else None, "weighted_actual": actual/weight if valid else None,
            "weighted_prediction": predicted/weight if valid else None, "supported_rows": supported,
            "low_support_rows": low_support, "effective_sample_size": weight*weight/squared_weight if squared_weight else 0.,
            "censoring_assumption": "independent administrative censoring; G >= 0.05"}


def evaluate(rows, mu, sigma, horizons=(1., 2., 5., 10.)):
    g = km_curve(rows, censor=True)
    result = {"nll": nll(rows, mu, sigma), "rows": len(rows), "censored": sum(not math.isfinite(r["upper"]) for r in rows)}
    result["windows"] = {str(h): ipcw_score(rows, ndtr((math.log(h)-mu)/sigma), np.full(len(rows), h), g) for h in horizons}
    result["quantile_coverage"] = {str(q): ipcw_score(rows, np.full(len(rows), q), np.exp(mu + sigma*ndtri(q)), g) for q in (0.5, 0.9)}
    result["remaining_windows"] = {}
    for age in (1., 2., 5.):
        indexes = [i for i, r in enumerate(rows) if r["lower"] > age]
        if not indexes:
            continue
        residual = [{**rows[i], "lower": rows[i]["lower"]-age,
                     "upper": rows[i]["upper"]-age} for i in indexes]
        selected_mu = np.broadcast_to(mu, (len(rows),))[indexes]
        log_sa = log_ndtr((selected_mu-math.log(age))/sigma)
        residual_g = km_curve(residual, censor=True)
        result["remaining_windows"][str(age)] = {
            str(h): ipcw_score(residual, -np.expm1(log_ndtr((selected_mu-math.log(age+h))/sigma)-log_sa), [h]*len(residual), residual_g)
            for h in horizons}
    return result


def point_accuracy(rows, prediction, tolerances=(0.1, 0.2, 0.5)):
    if any(not math.isfinite(r["upper"]) for r in rows):
        return {str(t): None for t in tolerances}
    target = np.asarray([r["lower"] for r in rows], dtype=float)
    prediction = np.broadcast_to(np.asarray(prediction, dtype=float), target.shape)
    relative = np.abs(prediction - target) / target
    return {str(t): float(np.mean(relative <= t)) for t in tolerances}


def accuracy_curve(rows, booster, spec, best_iteration, sigma=0.5, time_scale=1.0):
    if any(not math.isfinite(row["upper"]) for row in rows):
        return {"status": "unsupported_censored_labels", "tolerances": [0.1, 0.2, 0.5]}
    data = matrix(rows, spec, time_scale)
    target = np.asarray([row["lower"] * time_scale for row in rows], dtype=np.float64)
    curves = {f"{q}_{t}": [] for q in (0.5, 0.9) for t in (0.1, 0.2, 0.5)}
    for iteration in range(1, booster.num_boosted_rounds() + 1):
        mu = booster.predict(data, output_margin=True, iteration_range=(0, iteration))
        for q in (0.5, 0.9):
            relative = np.abs(np.exp(mu + sigma * ndtri(q)) - target) / target
            for tolerance in (0.1, 0.2, 0.5):
                curves[f"{q}_{tolerance}"].append(float(np.mean(relative <= tolerance)))
    return {"status": "ok", "tolerances": [0.1, 0.2, 0.5], "best_round": best_iteration + 1,
            "curves": curves, "best_round_accuracy": {t: values[best_iteration] for t, values in curves.items()}}


def family_priors(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["call"].get("action_family", row["call"]["name"])].append(math.log(row["lower"]))
    return {name: {"mu": float(np.median(values)), "sigma": max(float(np.std(values)), 0.25), "rows": len(values)}
            for name, values in grouped.items()}


def baseline_reports(train, test):
    def fit(rows):
        logs = np.log([r["lower"] for r in rows])
        fit_result = minimize(lambda p: nll(rows, p[0], math.exp(p[1])), [float(np.mean(logs)), math.log(max(float(np.std(logs)), 0.2))], bounds=[(-30, 30), (-4, 3)])
        return float(fit_result.x[0]), math.exp(float(fit_result.x[1]))
    global_mu, global_sigma = fit(train)
    grouped = defaultdict(list)
    for row in train:
        grouped[row["call"]["name"]].append(row)
    empirical = {name: km_curve(values) for name, values in grouped.items()}
    global_km = km_curve(train)
    g = km_curve(test, censor=True)
    return {"global_log_normal": evaluate(test, np.full(len(test), global_mu), global_sigma),
            "global_empirical_windows": {str(h): ipcw_score(test, [1-global_km(h)]*len(test), [h]*len(test), g) for h in (1., 2., 5., 10.)},
            "per_tool_empirical_windows": {str(h): ipcw_score(test, [1-empirical.get(r["call"]["name"], global_km)(h) for r in test], [h]*len(test), g) for h in (1., 2., 5., 10.)}}


def train(rows, output, threads=4, rounds=500, sigmas=(0.25, 0.5, 1., 1.5, 2.), holdout_tools=(), data_kind="unspecified"):
    rows = validate_rows(rows)
    if threads <= 0 or rounds <= 0 or not sigmas or any(not math.isfinite(s) or s <= 0 for s in sigmas):
        raise ValueError("threads, rounds and sigma values must be positive")
    if data_kind not in ("observed", "proxy", "synthetic", "unspecified"):
        raise ValueError("invalid data_kind")
    label_kinds = {r.get("label_kind", "observed") for r in rows}
    if len(label_kinds) > 1:
        raise ValueError("mixed_label_kinds_rejected")
    # Provenance is checked before feature fitting.  In particular, a caller
    # may not manufacture formal observations by selecting data_kind=observed.
    validate_label_provenance(rows, data_kind=data_kind, production=False)
    groups = split_rows(rows)
    held = set(holdout_tools)
    if held:
        # Quarantine whole workflows that contain a held-out tool, not just their tool rows.
        workflows = {r["workflow"] for r in rows if r["call"].get("action_family", r["call"]["name"]) in held or r["call"]["name"] in held}
        unseen = [r for r in rows if r["workflow"] in workflows and (r["call"].get("action_family", r["call"]["name"]) in held or r["call"]["name"] in held)]
        for name in ("train", "validation", "test"):
            groups[name] = [r for r in groups[name] if r["workflow"] not in workflows]
        if any(not groups[k] for k in ("train", "validation", "test")):
            raise ValueError("held-out tool workflows leave an empty split")
    else:
        unseen = []
    for name in ("train", "validation"):
        if not any(math.isfinite(r["upper"]) for r in groups[name]):
            raise ValueError(name+" has no observed returns; collect complete observations before fitting")
    action_families = ActionFamilies.fit(r["call"] for r in groups["train"])
    for bucket in groups.values():
        for row in bucket:
            row["call"] = action_families.enrich(row["call"])
    spec = FeatureSpec.fit(r["call"] for r in groups["train"])
    # Public traces contain microsecond-scale spans. Train in microseconds to
    # keep XGBoost AFT away from denormal floating-point values, then shift the
    # returned margin back to seconds for all reports and serving.
    time_scale = 1e6
    train_data, validation = matrix(groups["train"], spec, time_scale), matrix(groups["validation"], spec, time_scale)
    candidates = []
    for sigma in sigmas:
        params = {"objective": "survival:aft", "eval_metric": "aft-nloglik", "aft_loss_distribution": "normal",
                  "aft_loss_distribution_scale": sigma, "tree_method": "hist", "max_depth": 4,
                  "eta": 0.05, "seed": 42, "nthread": threads, "device": "cpu"}
        evals_result = {}
        booster = xgb.train(params, train_data, num_boost_round=rounds, evals=[(train_data, "train"), (validation, "validation")],
                            evals_result=evals_result, early_stopping_rounds=30, verbose_eval=False)
        candidates.append((float(booster.best_score), sigma, booster, evals_result))
    _, sigma, booster, evals_result = min(candidates, key=lambda v: v[0])
    def predict(data):
        return booster.predict(matrix(data, spec), output_margin=True, iteration_range=(0, booster.best_iteration+1)) - math.log(time_scale)
    def assess(data):
        return evaluate(data, predict(data), sigma)
    tree_validation = assess(groups["validation"])
    report = {"split_counts": {k: len(v) for k,v in groups.items()}, "validation": tree_validation,
              "test": assess(groups["test"]), "baselines": baseline_reports(groups["train"], groups["test"]),
              "sigma_trials": [{"sigma": s, "validation_nll": score} for score,s,_,_ in candidates],
              "time_scale": time_scale,
              "training_curve": evals_result, "dependencies": versions(), "threads": threads, "seed": 42,
              "data_kind": data_kind, "label_kinds": dict(Counter(r.get("label_kind", "observed") for r in rows)),
              "data_digest": dataset_digest(rows), "dataset": describe(rows),
              "unknown_tool_test": assess(unseen) if unseen else {"status": "not_run", "reason": "pass --holdout-tool"},
              "prediction_mode": "xgboost_aft", "family_priors": {}, "embedding_kind": spec.embedding_kind,
              "embedding_model": spec.embedding_model, "embedding_dim": spec.embedding_dim,
              "dataset_revision": next((row.get("dataset_revision") for row in rows if row.get("dataset_revision")), None),
              "converter_version": next((row.get("converter_version") for row in rows if row.get("converter_version")), None),
              "validation_model_nll": tree_validation["nll"], "promotion": "candidate; explicit quality-gated promote required"}
    curve = _curve_svg({name: values.get("aft-nloglik", []) for name, values in evals_result.items()}, "AFT training and validation negative log-likelihood")
    test_mu = predict(groups["test"]); selected_sigma = sigma
    train_median = float(np.median([r["lower"] for r in groups["train"]]))
    report["point_accuracy"] = {"definition": "relative absolute error of exp(AFT margin) P50",
                                 "model": point_accuracy(groups["test"], np.exp(test_mu)),
                                 "global_median_baseline": point_accuracy(groups["test"], train_median),
                                 "baseline_median_seconds": train_median}
    report["accuracy_curves"] = {"train": accuracy_curve(groups["train"], booster, spec, booster.best_iteration, sigma, time_scale),
                                  "validation": accuracy_curve(groups["validation"], booster, spec, booster.best_iteration, sigma, time_scale),
                                  "test": accuracy_curve(groups["test"], booster, spec, booster.best_iteration, sigma, time_scale)}
    q_grid = np.arange(0.1, 1.0, 0.1)
    q_actual = []
    for q in q_grid:
        score = ipcw_score(groups["test"], np.full(len(groups["test"]), q), np.exp(test_mu + selected_sigma * ndtri(q)))
        q_actual.append(score["weighted_actual"] if score["weighted_actual"] is not None else None)
    q_pred = list(q_grid)
    horizons = (0.5, 1., 2., 5., 10.)
    w_pred = [float(ndtr((np.log(h)-test_mu)/selected_sigma).mean()) for h in horizons]
    w_actual = []
    for h, p in zip(horizons, w_pred):
        score = ipcw_score(groups["test"], np.full(len(groups["test"]), p), np.full(len(groups["test"]), h))
        w_actual.append(score.get("weighted_actual"))
    artifacts = {
        "dataset-manifest.json": json.dumps({"dataset": report["dataset"],
                                               "revision": report.get("dataset_revision"),
                                               "converter_version": report.get("converter_version"),
                                               "source_hash_kind": "normalized_rows_sha256",
                                               "data_digest": report["data_digest"],
                                               "label_kinds": report["label_kinds"],
                                               "data_kind": data_kind}, ensure_ascii=False, sort_keys=True),
        "quantile_coverage.svg": _curve_svg({"predicted": q_pred, "observed": q_actual}, "Quantile coverage (IPCW observed)"),
        "window_calibration.svg": _curve_svg({"predicted": w_pred, "observed": w_actual}, "Window return probability calibration"),
        "accuracy_curve.svg": _curve_svg({"train_20pct": report["accuracy_curves"]["train"].get("curves", {}).get("0.5_0.2", []),
                                           "validation_20pct": report["accuracy_curves"]["validation"].get("curves", {}).get("0.5_0.2", [])},
                                          "Relative accuracy (error <= 20%)"),
    }
    return publish(output, booster, spec, sigma, booster.best_iteration, report, action_families=action_families,
                   prediction_mode="xgboost_aft", family_priors={}, curve_svg=curve, artifacts=artifacts), report


def load_rows(path):
    rows = []
    with Path(path).open(encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line, parse_constant=lambda value: (_ for _ in ()).throw(ValueError("nonstandard JSON constant: "+value)))
                if not isinstance(row, dict):
                    raise ValueError("expected a JSON object")
                rows.append(row)
            except (ValueError, RecursionError) as exc:
                raise ValueError(f"{path}: line {line_number}: {exc}") from exc
    return rows


def dataset_digest(rows):
    from hashlib import sha256
    from .features import canonical
    digest = sha256()
    for row in rows:
        digest.update((canonical({**row, "upper": None if math.isinf(row["upper"]) else row["upper"]})+"\n").encode())
    return digest.hexdigest()


def inspect_dataset(rows):
    rows = validate_rows(rows)
    groups = split_rows(rows)
    summary = {"dataset": describe(rows), "sha256": dataset_digest(rows), "splits": {name: describe(group) for name, group in groups.items()}}
    summary["warnings"] = [name+" has no observed returns" for name in ("train", "validation", "test") if not summary["splits"][name]["observed"]]
    summary["trainable"] = all(summary["splits"][name]["observed"] > 0 for name in ("train", "validation"))
    return summary
