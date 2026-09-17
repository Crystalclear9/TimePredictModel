import argparse
import asyncio
import json
import logging
import math
import os
from pathlib import Path


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
    snap.add_argument("--output", required=True)
    training = sub.add_parser("train")
    training.add_argument("--data", required=True)
    training.add_argument("--output", required=True)
    training.add_argument("--threads", type=int, default=4)
    training.add_argument("--rounds", type=int, default=500)
    training.add_argument("--holdout-tool", action="append", default=[])
    training.add_argument("--data-kind", choices=["observed", "synthetic", "unspecified"], default="unspecified")
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
        app = create_app(store, json.loads(Path(args.tokens_file).read_text()), clock, worker)
        uvicorn.run(app, host=args.host, port=args.port)
    elif args.command == "export":
        from redis.asyncio import Redis
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
        rows, excluded = snapshot(load_rows(args.events), args.cutoff)
        with Path(args.output).open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps({**row, "upper": None if math.isinf(row["upper"]) else row["upper"]}, ensure_ascii=False, allow_nan=False) + "\n")
        print(json.dumps({"samples": len(rows), "excluded": excluded}))
    elif args.command == "train":
        from .training import load_rows, train
        if args.threads <= 0 or args.rounds <= 0:
            parser.error("threads and rounds must be positive")
        rows = load_rows(args.data)
        version, report = train(rows, args.output, args.threads, args.rounds, holdout_tools=args.holdout_tool, data_kind=args.data_kind)
        print(json.dumps({"version": version, "report": report}, ensure_ascii=False, allow_nan=False))
    elif args.command == "check-data":
        from .training import inspect_dataset, load_rows
        result = inspect_dataset(load_rows(args.data))
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        if not result["trainable"]:
            raise SystemExit(1)
    elif args.command == "promote":
        from .release import promote
        print(json.dumps(promote(args.candidate, args.serving_dir, args.evidence, args.policy), ensure_ascii=False))


if __name__ == "__main__":
    main()
