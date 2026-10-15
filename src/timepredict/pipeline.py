"""Batch retraining and like-for-like semantic experiments; never fabricate acceptance."""
import asyncio
import copy
import json
import math
import os
import tempfile
import time
import threading
import uuid
from pathlib import Path

from .model import Model, _flat_name, _read_json_object
from .clock import Clock
from .training import dataset_digest, snapshot, split_rows, train, training_signature


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".write-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, allow_nan=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def ablation(rows, output, *, data_kind, threads=4, rounds=500, holdout_tools=()):
    """The same rows, temporal/whole-tool splits, sigma search and boosting budget."""
    output = Path(output)
    result = {"data_digest": dataset_digest(rows), "rounds": rounds, "threads": threads,
              "holdout_tools": list(holdout_tools), "production_benefit_verified": False, "models": {}}
    for mode in ("structured", "semantic"):
        started = time.perf_counter()
        version, report = train(copy.deepcopy(rows), output/mode, threads=threads, rounds=rounds,
                                data_kind=data_kind, holdout_tools=holdout_tools, feature_mode=mode)
        training_seconds = time.perf_counter()-started
        model = Model(output/mode/version)
        calls = [r["call"] for r in split_rows(rows)["test"]]
        # Initialize the optional encoder before measuring; then every cold pass clears its text cache.
        model.spec.transform(calls[0])
        timings, uncached = [], []
        for call in calls:
            encoder = getattr(model.spec, "_encoder", None)
            if encoder:
                encoder.clear_cache()
            started = time.perf_counter()
            model.predict(call)
            uncached.append(time.perf_counter()-started)
            started = time.perf_counter()
            model.predict(call)
            timings.append(time.perf_counter()-started)
        result["models"][mode] = {"version": version, "training_seconds": training_seconds,
                                  "inference_mean_seconds": sum(timings)/len(timings),
                                  "inference_p99_seconds": sorted(timings)[int((len(timings)-1)*.99)],
                                  "uncached_inference_mean_seconds": sum(uncached)/len(uncached),
                                  "uncached_inference_p99_seconds": sorted(uncached)[int((len(uncached)-1)*.99)],
                                  "timing_scope": "feature transform + tree prediction, CPU; cold text cache then immediate repeated call",
                                  "validation": report["validation"], "test": report["test"],
                                  "unknown_tool_test": report["unknown_tool_test"], "split_counts": report["split_counts"]}
    atomic_json(output/"ablation.json", result)
    return result


async def retrain_once(args):
    """Run outside the online worker. A renewed Redis lease prevents overlapping trainers."""
    from redis.asyncio import Redis
    from .release import promote, PromotionCommitUncertain
    redis = Redis.from_url(args.redis_url, decode_responses=True)
    key, token = args.prefix+":training-lease", uuid.uuid4().hex
    if not await redis.set(key, token, nx=True, ex=120):
        await redis.aclose()
        return {"status": "skipped", "reason": "trainer_already_running"}
    async def renew():
        while True:
            await asyncio.sleep(30)
            ok = await redis.eval("if redis.call('get',KEYS[1]) == ARGV[1] then return redis.call('expire',KEYS[1],120) else return 0 end", 1, key, token)
            if not ok:
                raise RuntimeError("training lease lost")
    lease_task = asyncio.create_task(renew())
    publication_cancelled = threading.Event()
    loop = asyncio.get_running_loop()
    def publication_guard():
        # Cancelling to_thread does not stop its OS thread. Fence at the actual pointer commit.
        if publication_cancelled.is_set():
            raise RuntimeError("training publication cancelled")
        future = asyncio.run_coroutine_threadsafe(redis.get(key), loop)
        if future.result(timeout=5) != token or publication_cancelled.is_set():
            raise RuntimeError("training lease lost before publication commit")
    work_task = None
    output = Path(args.output)
    try:
        async def work():
            if any((args.serving_dir, args.evidence, args.policy)) and not all((args.serving_dir, args.evidence, args.policy)):
                raise ValueError("promotion requires --serving-dir, --evidence and --policy together")
            cutoff_clock = Clock()
            await asyncio.to_thread(cutoff_clock.refresh)
            cutoff_stamp = cutoff_clock.stamp()
            cutoff, cursor, events = cutoff_stamp["utc"], "-", []
            tail = await redis.xrevrange(args.prefix+":history", count=1)
            boundary = tail[0][0] if tail else "0-0"
            while True:
                batch = await redis.xrange(args.prefix+":history", min=cursor, max=boundary, count=1000)
                if not batch:
                    break
                events.extend(json.loads(fields["data"]) for _, fields in batch)
                cursor = "("+batch[-1][0]
            rows, excluded = snapshot(events, cutoff, cutoff_stamp=cutoff_stamp)
            formal = [r for r in rows if r["label_kind"] == "sglang_return_observed"]
            digest = dataset_digest(formal)
            previous_path = output/"last-run.json"
            previous, quarantined_receipt = {}, None
            if previous_path.exists():
                try:
                    previous = _read_json_object(previous_path)
                except ValueError:
                    quarantined_receipt = output/f".invalid-last-run-{time.time_ns()}.json"
                    previous_path.rename(quarantined_receipt)
            settings = {"rounds": args.rounds, "threads": args.threads, "feature_mode": args.feature_mode,
                        "holdout_tools": sorted(set(args.holdout_tool)),
                        "training_signature": training_signature(args.feature_mode)}
            result = {"cutoff": cutoff, "excluded": excluded, "rows": len(formal), "data_digest": digest,
                      "settings": settings}
            if quarantined_receipt is not None:
                result["quarantined_receipt"] = str(quarantined_receipt)
            if not formal:
                result.update(status="waiting_for_data", reason="no verified return samples")
                atomic_json(previous_path, result)
                return result
            try:
                cohorts = split_rows(formal)
            except ValueError as exc:
                result.update(status="waiting_for_data", reason=str(exc))
                atomic_json(previous_path, result)
                return result
            if any(not any(math.isfinite(r["upper"]) for r in cohorts[k]) for k in ("train", "validation")):
                result.update(status="waiting_for_data", reason="train/validation require completed returns")
                atomic_json(previous_path, result)
                return result
            reuse = (previous.get("data_digest") == digest and previous.get("settings") == settings
                     and _flat_name(previous.get("version")))
            if reuse:
                try:
                    cached = await asyncio.to_thread(Model, output/"candidates"/previous["version"])
                    if cached.version != previous["version"]:
                        raise ValueError("cached_candidate_version_mismatch")
                    report = _read_json_object(output/"candidates"/previous["version"]/"report.json")
                    expected = {"data_digest": digest, "training_signature": settings["training_signature"],
                                "rounds": args.rounds, "threads": args.threads, "feature_mode": args.feature_mode,
                                "holdout_tools": settings["holdout_tools"]}
                    if any(report.get(field) != value for field, value in expected.items()):
                        raise ValueError("cached_candidate_training_mismatch")
                except (OSError, ValueError, KeyError, TypeError):
                    reuse = False
            if reuse:
                version = previous["version"]
                result.update(status="unchanged", version=version)
            else:
                output.mkdir(parents=True, exist_ok=True)
                snapshot_path = output/f"snapshot-{time.time_ns()}.jsonl"
                with snapshot_path.open("w", encoding="utf-8") as handle:
                    for row in formal:
                        handle.write(json.dumps({**row, "upper": None if math.isinf(row["upper"]) else row["upper"]}, ensure_ascii=False, allow_nan=False)+"\n")
                version, _ = await asyncio.to_thread(train, formal, output/"candidates", threads=args.threads,
                                                    rounds=args.rounds, data_kind="observed",
                                                    feature_mode=args.feature_mode, holdout_tools=args.holdout_tool)
                result.update(status="candidate_trained", version=version, snapshot=str(snapshot_path))
            if args.serving_dir:
                if await redis.get(key) != token:
                    raise RuntimeError("training lease lost before publication")
                try:
                    result["promotion"] = await asyncio.to_thread(promote, output/"candidates"/version,
                                                                  args.serving_dir, args.evidence, args.policy,
                                                                  before_commit=publication_guard)
                    result["status"] = "promoted"
                except PromotionCommitUncertain as exc:
                    result.update(status="publication_outcome_uncertain", publication_error=str(exc))
                except (ValueError, OSError) as exc:
                    result.update(status="candidate_requires_acceptance", promotion_error=str(exc))
            atomic_json(previous_path, result)
            return result
        work_task = asyncio.create_task(work())
        done, _ = await asyncio.wait((lease_task, work_task), return_when=asyncio.FIRST_COMPLETED)
        if lease_task in done:
            await lease_task
        return await work_task
    finally:
        publication_cancelled.set()
        lease_task.cancel()
        if work_task and not work_task.done():
            work_task.cancel()
        await asyncio.gather(lease_task, *([work_task] if work_task else []), return_exceptions=True)
        try:
            await redis.eval("if redis.call('get',KEYS[1]) == ARGV[1] then return redis.call('del',KEYS[1]) else return 0 end", 1, key, token)
        finally:
            await redis.aclose()
