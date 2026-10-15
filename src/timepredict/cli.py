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
    serve.add_argument(
        "--redis-url", default=os.environ.get("TIMEPREDICT_REDIS_URL", "redis://127.0.0.1:6379/0")
    )
    serve.add_argument("--prefix", default="timepredict")
    serve.add_argument("--models")
    serve.add_argument(
        "--shadow-candidate",
        help="evaluate an unpromoted candidate; never issue actionable candidates",
    )
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8098)
    serve.add_argument("--window-h", type=float, required=True)
    serve.add_argument(
        "--tokens-file",
        required=True,
        help="JSON mapping of internal API token to tenant namespace",
    )
    serve.add_argument(
        "--admin-tokens-file", help="separate JSON array of operator tokens; disabled by default"
    )
    serve.add_argument("--notice-limit", type=int, default=100000)
    serve.add_argument("--max-state-bytes", type=int, default=64 * 1024 * 1024)
    serve.add_argument("--deadletter-limit", type=int, default=10000)
    export = sub.add_parser("export")
    export.add_argument(
        "--redis-url", default=os.environ.get("TIMEPREDICT_REDIS_URL", "redis://127.0.0.1:6379/0")
    )
    export.add_argument("--prefix", default="timepredict")
    export.add_argument("--output", required=True)
    export.add_argument("--archive-dir")
    export.add_argument("--max-history-events", type=int, default=100000)
    export.add_argument("--max-history-bytes", type=int, default=128 * 1024 * 1024)
    snap = sub.add_parser("snapshot")
    snap.add_argument("--events", required=True)
    snap.add_argument("--cutoff", type=float, required=True, help="UTC Unix seconds")
    snap.add_argument(
        "--cutoff-clock-error",
        type=float,
        help="measured absolute UTC cutoff error in seconds; needed for right-censoring",
    )
    snap.add_argument("--output", required=True)
    training = sub.add_parser("train")
    training.add_argument("--data", required=True)
    training.add_argument("--output", required=True)
    training.add_argument("--threads", type=int, default=4)
    training.add_argument("--rounds", type=int, default=500)
    training.add_argument("--holdout-tool", action="append", default=[])
    training.add_argument(
        "--data-kind",
        choices=["observed", "proxy", "synthetic", "unspecified"],
        default="unspecified",
    )
    training.add_argument(
        "--feature-mode", choices=["structured", "semantic"], default="structured"
    )
    ablation = sub.add_parser(
        "ablation", help="compare structured and joint-semantic AFT with identical splits/budgets"
    )
    ablation.add_argument("--data", required=True)
    ablation.add_argument("--output", required=True)
    ablation.add_argument("--data-kind", choices=["observed", "proxy", "synthetic"], required=True)
    ablation.add_argument("--threads", type=int, default=4)
    ablation.add_argument("--rounds", type=int, default=500)
    ablation.add_argument("--holdout-tool", action="append", default=[])
    retrain = sub.add_parser(
        "retrain",
        help="export an immutable event cutoff, train and optionally quality-gate promotion",
    )
    retrain.add_argument(
        "--redis-url", default=os.environ.get("TIMEPREDICT_REDIS_URL", "redis://127.0.0.1:6379/0")
    )
    retrain.add_argument("--prefix", default="timepredict")
    retrain.add_argument("--output", required=True)
    retrain.add_argument("--serving-dir")
    retrain.add_argument("--evidence")
    retrain.add_argument("--policy")
    retrain.add_argument("--threads", type=int, default=4)
    retrain.add_argument("--rounds", type=int, default=500)
    retrain.add_argument("--feature-mode", choices=["structured", "semantic"], default="structured")
    retrain.add_argument("--holdout-tool", action="append", default=[])
    retrain.add_argument(
        "--archive-dir", help="verified archive directory if Redis history has been trimmed"
    )
    retrain.add_argument("--max-history-events", type=int, default=100000)
    retrain.add_argument("--max-history-bytes", type=int, default=128 * 1024 * 1024)
    archive = sub.add_parser(
        "archive-history", help="durably archive history before optional stream trimming"
    )
    archive.add_argument(
        "--redis-url", default=os.environ.get("TIMEPREDICT_REDIS_URL", "redis://127.0.0.1:6379/0")
    )
    archive.add_argument("--prefix", default="timepredict")
    archive.add_argument("--archive-dir", required=True)
    archive.add_argument("--older-than-days", type=float, default=7.0)
    archive.add_argument("--trim-history", action="store_true")
    archive.add_argument("--trim-events", action="store_true")
    check = sub.add_parser(
        "check-data", help="validate labels, workflow grouping and temporal split before training"
    )
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
        store = RedisStore(
            redis,
            args.prefix,
            notice_limit=args.notice_limit,
            max_state_bytes=args.max_state_bytes,
            deadletter_limit=args.deadletter_limit,
        )
        clock = Clock()
        if args.models and args.shadow_candidate:
            parser.error("choose --models or --shadow-candidate")
        worker = Worker(
            store,
            args.models or args.shadow_candidate,
            clock,
            args.window_h,
            shadow=bool(args.shadow_candidate),
        )
        admin_tokens = (
            json.loads(Path(args.admin_tokens_file).read_text(encoding="utf-8"))
            if args.admin_tokens_file
            else ()
        )
        app = create_app(
            store,
            json.loads(Path(args.tokens_file).read_text(encoding="utf-8")),
            clock,
            worker,
            admin_tokens=admin_tokens,
        )
        uvicorn.run(app, host=args.host, port=args.port)
    elif args.command == "archive-history":
        from redis.asyncio import Redis
        from .retention import archive_history

        async def archive_events():
            redis = Redis.from_url(args.redis_url, decode_responses=True, socket_timeout=5)
            try:
                return await archive_history(
                    redis,
                    args.prefix,
                    args.archive_dir,
                    older_than_days=args.older_than_days,
                    trim_history=args.trim_history,
                    trim_events=args.trim_events,
                )
            finally:
                await redis.aclose()

        print(json.dumps(asyncio.run(archive_events())))
    elif args.command == "export":
        from redis.asyncio import Redis

        # 分批导出 Redis history 流；使用排他游标避免重复读取上一批末尾。
        async def export_events():
            redis = Redis.from_url(args.redis_url, decode_responses=True)
            from .retention import read_history

            try:
                events = await read_history(
                    redis,
                    args.prefix,
                    archive_dir=args.archive_dir,
                    max_events=args.max_history_events,
                    max_bytes=args.max_history_bytes,
                )
                with Path(args.output).open("w", encoding="utf-8") as handle:
                    for event in events:
                        handle.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")
            finally:
                await redis.aclose()

        asyncio.run(export_events())
    elif args.command == "snapshot":
        from .training import load_rows, snapshot

        cutoff_stamp = None
        if args.cutoff_clock_error is not None:
            if (
                not math.isfinite(args.cutoff_clock_error)
                or not 0 <= args.cutoff_clock_error <= 0.0025
            ):
                parser.error("cutoff clock error must be finite and in [0,0.0025] seconds")
            cutoff_stamp = {"utc": args.cutoff, "error": args.cutoff_clock_error, "jumped": False}
        rows, excluded = snapshot(load_rows(args.events), args.cutoff, cutoff_stamp=cutoff_stamp)
        with Path(args.output).open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(
                    json.dumps(
                        {**row, "upper": None if math.isinf(row["upper"]) else row["upper"]},
                        ensure_ascii=False,
                        allow_nan=False,
                    )
                    + "\n"
                )
        print(json.dumps({"samples": len(rows), "excluded": excluded}))
    elif args.command == "train":
        from .training import load_rows, train

        if args.threads <= 0 or args.rounds <= 0:
            parser.error("threads and rounds must be positive")
        rows = load_rows(args.data)
        version, report = train(
            rows,
            args.output,
            args.threads,
            args.rounds,
            holdout_tools=args.holdout_tool,
            data_kind=args.data_kind,
            feature_mode=args.feature_mode,
        )
        print(
            json.dumps({"version": version, "report": report}, ensure_ascii=False, allow_nan=False)
        )
    elif args.command == "ablation":
        from .pipeline import ablation
        from .training import load_rows

        print(
            json.dumps(
                ablation(
                    load_rows(args.data),
                    args.output,
                    data_kind=args.data_kind,
                    threads=args.threads,
                    rounds=args.rounds,
                    holdout_tools=args.holdout_tool,
                ),
                ensure_ascii=False,
                allow_nan=False,
            )
        )
    elif args.command == "retrain":
        from .pipeline import retrain_once

        print(json.dumps(asyncio.run(retrain_once(args)), ensure_ascii=False, allow_nan=False))
    elif args.command == "check-data":
        from .training import inspect_dataset, load_rows

        result = inspect_dataset(load_rows(args.data))
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        if not result["trainable"]:
            raise SystemExit(1)
    elif args.command == "promote":
        from .release import promote

        print(
            json.dumps(
                promote(args.candidate, args.serving_dir, args.evidence, args.policy),
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    main()
