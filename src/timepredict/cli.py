# 文件作用：timepredict 命令行入口：解析子命令并按需导入采集、训练、评估、发布模块。
import argparse
import asyncio
import json
import logging
import math
import os
from pathlib import Path


# 注册各子命令与参数，校验基本边界，再分派到对应实现。
def main():
    parser = argparse.ArgumentParser(description="Server-oriented SGLang return-time predictor")
    sub = parser.add_subparsers(dest="command", required=True)
    serve = sub.add_parser("serve")
    serve.add_argument("--redis-url", default=os.environ.get("TIMEPREDICT_REDIS_URL", "redis://127.0.0.1:6379/0"))
    serve.add_argument("--prefix", default="timepredict")
    serve.add_argument("--models")
    serve.add_argument("--shadow-candidate", help="evaluate an unpromoted candidate; never issue actionable candidates")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8098)
    serve.add_argument("--window-h", type=float, required=True)
    serve.add_argument("--tokens-file", required=True, help="JSON mapping of internal API token to tenant namespace")
    export = sub.add_parser("export")
    export.add_argument("--redis-url", default=os.environ.get("TIMEPREDICT_REDIS_URL", "redis://127.0.0.1:6379/0"))
    export.add_argument("--prefix", default="timepredict")
    export.add_argument("--output", required=True)
    snap = sub.add_parser("snapshot")
    snap.add_argument("--events", required=True)
    snap.add_argument("--cutoff", type=float, required=True, help="UTC Unix seconds")
    snap.add_argument("--cutoff-clock-error", type=float, help="measured absolute UTC cutoff error in seconds; needed for right-censoring")
    snap.add_argument("--output", required=True)
    training = sub.add_parser("train")
    training.add_argument("--data", required=True)
    training.add_argument("--output", required=True)
    training.add_argument("--threads", type=int, default=4)
    training.add_argument("--rounds", type=int, default=500)
    training.add_argument("--holdout-tool", action="append", default=[])
    training.add_argument("--data-kind", choices=["observed", "proxy", "synthetic", "unspecified"], default="unspecified")
    training.add_argument("--feature-mode", choices=["structured", "semantic"], default="structured")
    ablation = sub.add_parser("ablation", help="compare structured and joint-semantic AFT with identical splits/budgets")
    ablation.add_argument("--data", required=True)
    ablation.add_argument("--output", required=True)
    ablation.add_argument("--data-kind", choices=["observed", "proxy", "synthetic"], required=True)
    ablation.add_argument("--threads", type=int, default=4)
    ablation.add_argument("--rounds", type=int, default=500)
    ablation.add_argument("--holdout-tool", action="append", default=[])
    retrain = sub.add_parser("retrain", help="export an immutable event cutoff, train and optionally quality-gate promotion")
    retrain.add_argument("--redis-url", default=os.environ.get("TIMEPREDICT_REDIS_URL", "redis://127.0.0.1:6379/0"))
    retrain.add_argument("--prefix", default="timepredict")
    retrain.add_argument("--output", required=True)
    retrain.add_argument("--serving-dir")
    retrain.add_argument("--evidence")
    retrain.add_argument("--policy")
    retrain.add_argument("--threads", type=int, default=4)
    retrain.add_argument("--rounds", type=int, default=500)
    retrain.add_argument("--feature-mode", choices=["structured", "semantic"], default="structured")
    retrain.add_argument("--holdout-tool", action="append", default=[])
    experiment = sub.add_parser("experiment", help="run pinned public proxy-label experiments")
    experiment.add_argument("--output", required=True)
    experiment.add_argument("--cache-dir")
    experiment.add_argument("--threads", type=int, default=4)
    experiment.add_argument("--rounds", type=int, default=500)
    experiment.add_argument("--max-rows", type=int, default=512)
    replay = sub.add_parser("replay-generate", help="generate measured controlled-tool replay data")
    replay.add_argument("--output", required=True)
    replay.add_argument("--rows", type=int, default=5000)
    replay.add_argument("--seed", type=int, default=42)
    replay.add_argument("--concurrency", type=int, default=4)
    regression = sub.add_parser("train-regression", help="train direct quantile regressors on replay labels")
    regression.add_argument("--data", required=True)
    regression.add_argument("--output", required=True)
    regression.add_argument("--threads", type=int, default=4)
    regression.add_argument("--rounds", type=int, default=500)
    regression.add_argument("--data-kind", choices=["synthetic_replay", "proxy_tracecommons"], default="synthetic_replay")
    evaluation = sub.add_parser("evaluate", help="independently reload and evaluate quantile package")
    evaluation.add_argument("--model", required=True)
    evaluation.add_argument("--data", required=True)
    cache_manifest = sub.add_parser("cache-manifest", help="write a revision/hash manifest for local dataset Parquet files")
    cache_manifest.add_argument("--cache-dir", required=True)
    cache_manifest.add_argument("--dataset", choices=["exgentic", "agenttrace", "tracecommons", "betterwright"], required=True)
    audit = sub.add_parser("audit", help="write implementation status audit")
    audit.add_argument("--output", required=True)
    audit.add_argument("--evidence")
    check = sub.add_parser("check-data", help="validate labels, workflow grouping and temporal split before training")
    check.add_argument("--data", required=True)
    release = sub.add_parser("promote")
    release.add_argument("--candidate", required=True)
    release.add_argument("--serving-dir", required=True)
    release.add_argument("--evidence", required=True)
    release.add_argument("--policy", required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    if args.command == "serve":
        import uvicorn
        from redis.asyncio import Redis
        from .clock import Clock
        from .service import Worker, create_app
        from .store import RedisStore
        if not math.isfinite(args.window_h) or args.window_h <= 0:
            parser.error("--window-h must be positive and finite")
        redis = Redis.from_url(args.redis_url, decode_responses=True, socket_timeout=5)
        store = RedisStore(redis, args.prefix)
        clock = Clock()
        if args.models and args.shadow_candidate:
            parser.error("choose --models or --shadow-candidate")
        worker = Worker(store, args.models or args.shadow_candidate, clock, args.window_h, shadow=bool(args.shadow_candidate))
        app = create_app(store, json.loads(Path(args.tokens_file).read_text(encoding="utf-8")), clock, worker)
        uvicorn.run(app, host=args.host, port=args.port)
    elif args.command == "export":
        from redis.asyncio import Redis
        # 分批导出 Redis history 流；使用排他游标避免重复读取上一批末尾。
        async def export_events():
            redis = Redis.from_url(args.redis_url, decode_responses=True)
            cursor = "-"
            with Path(args.output).open("w", encoding="utf-8") as handle:
                while True:
                    batch = await redis.xrange(args.prefix + ":history", min=cursor, count=1000)
                    if not batch:
                        break
                    for ident, fields in batch:
                        handle.write(fields["data"] + "\n")
                    cursor = "(" + batch[-1][0]
            await redis.aclose()
        asyncio.run(export_events())
    elif args.command == "snapshot":
        from .training import load_rows, snapshot
        cutoff_stamp = None
        if args.cutoff_clock_error is not None:
            if not math.isfinite(args.cutoff_clock_error) or not 0 <= args.cutoff_clock_error <= .0025:
                parser.error("cutoff clock error must be finite and in [0,0.0025] seconds")
            cutoff_stamp = {"utc": args.cutoff, "error": args.cutoff_clock_error, "jumped": False}
        rows, excluded = snapshot(load_rows(args.events), args.cutoff, cutoff_stamp=cutoff_stamp)
        with Path(args.output).open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps({**row, "upper": None if math.isinf(row["upper"]) else row["upper"]}, ensure_ascii=False, allow_nan=False) + "\n")
        print(json.dumps({"samples": len(rows), "excluded": excluded}))
    elif args.command == "train":
        from .training import load_rows, train
        if args.threads <= 0 or args.rounds <= 0:
            parser.error("threads and rounds must be positive")
        rows = load_rows(args.data)
        version, report = train(rows, args.output, args.threads, args.rounds, holdout_tools=args.holdout_tool, data_kind=args.data_kind, feature_mode=args.feature_mode)
        print(json.dumps({"version": version, "report": report}, ensure_ascii=False, allow_nan=False))
    elif args.command == "ablation":
        from .pipeline import ablation
        from .training import load_rows
        print(json.dumps(ablation(load_rows(args.data), args.output, data_kind=args.data_kind,
                                  threads=args.threads, rounds=args.rounds, holdout_tools=args.holdout_tool), ensure_ascii=False, allow_nan=False))
    elif args.command == "retrain":
        from .pipeline import retrain_once
        print(json.dumps(asyncio.run(retrain_once(args)), ensure_ascii=False, allow_nan=False))
    elif args.command == "check-data":
        from .training import inspect_dataset, load_rows
        result = inspect_dataset(load_rows(args.data))
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        if not result["trainable"]:
            raise SystemExit(1)
    elif args.command == "experiment":
        from .experiment import run
        if args.threads <= 0 or args.rounds <= 0:
            parser.error("threads and rounds must be positive")
        print(json.dumps(run(args.output, args.cache_dir, args.threads, args.rounds, args.max_rows), ensure_ascii=False, allow_nan=False))
    elif args.command == "replay-generate":
        from .replay import generate
        print(json.dumps(generate(args.output, args.rows, args.seed, args.concurrency), ensure_ascii=False, indent=2))
    elif args.command == "train-regression":
        from .regression import train_regression
        from .training import load_rows
        if args.threads <= 0 or args.rounds <= 0:
            parser.error("threads and rounds must be positive")
        label_kind = "observed_replay" if args.data_kind == "synthetic_replay" else "proxy_agent_tool_return"
        report = train_regression(load_rows(args.data), args.output, args.threads, args.rounds,
                                  allowed_label_kind=label_kind, data_kind=args.data_kind)
        print(json.dumps({k: report[k] for k in ('acceptance', 'metrics', 'split_counts')}, indent=2))
    elif args.command == "evaluate":
        from .regression import evaluate_model
        from .training import load_rows
        print(json.dumps(evaluate_model(args.model, load_rows(args.data)), indent=2))
    elif args.command == "cache-manifest":
        from .datasets import write_local_manifest
        print(json.dumps(write_local_manifest(args.cache_dir, args.dataset), ensure_ascii=False, indent=2))
    elif args.command == "audit":
        from .audit import run
        print(json.dumps(run(args.output, args.evidence), ensure_ascii=False, indent=2))
    elif args.command == "promote":
        from .release import promote
        print(json.dumps(promote(args.candidate, args.serving_dir, args.evidence, args.policy), ensure_ascii=False))


if __name__ == "__main__":
    main()
