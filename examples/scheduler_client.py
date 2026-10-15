"""Attach this consumer to your existing C2 scheduler; it does not perform KV loading."""

import argparse
import asyncio
import json
import os

import httpx

from timepredict.client import PredictionClient


async def run(args):
    async with httpx.AsyncClient(timeout=1) as http:
        predictor = PredictionClient(http, args.predictor, os.environ["TIMEPREDICT_API_KEY"])
        while True:
            try:
                candidates = await predictor.refresh(args.horizon)
            except httpx.HTTPError:
                candidates = predictor.candidates()
            # Replace printing with the external scheduler's density/KV adapter.
            # Distribution.mu/sigma and age_seconds can calculate any residence horizon.
            print(json.dumps(candidates, ensure_ascii=False))
            await asyncio.sleep(args.interval)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictor", default="http://127.0.0.1:8098")
    parser.add_argument("--horizon", type=float, default=2.0)
    parser.add_argument("--interval", type=float, default=0.1)
    args = parser.parse_args()
    if args.horizon <= 0 or args.interval <= 0:
        parser.error("horizon and interval must be positive")
    asyncio.run(run(args))
