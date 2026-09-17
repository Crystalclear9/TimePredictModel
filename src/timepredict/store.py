import json
import time

from redis.exceptions import WatchError

from .state import apply_prediction, empty_state, reduce_event


class RedisStore:
    def __init__(self, redis, prefix="timepredict"):
        self.redis, self.prefix = redis, prefix
        self.state_key = prefix + ":state"
        self.events = prefix + ":events"
        self.history = prefix + ":history"
        self.notices = prefix + ":notices"
        self.required_lease = None
        self.metrics = {"transactions": 0, "cas_retries": 0, "state_bytes": 0, "last_transaction_seconds": 0.}

    async def read(self):
        raw = await self.redis.get(self.state_key)
        return json.loads(raw) if raw else empty_state()

    async def mutate(self, reducer, event=None):
        started = time.monotonic()
        while True:
            async with self.redis.pipeline(transaction=True) as pipe:
                try:
                    await pipe.watch(self.state_key)
                    if self.required_lease:
                        lease_key, token = self.required_lease
                        await pipe.watch(lease_key)
                        if await pipe.get(lease_key) != token:
                            raise RuntimeError("worker lease lost before transaction")
                    raw = await pipe.get(self.state_key)
                    state = json.loads(raw) if raw else empty_state()
                    event_seen = event is not None and event["event_id"] in state["seen"]
                    state, notices = reducer(state)
                    serialized = json.dumps(state, allow_nan=False)
                    pipe.multi()
                    pipe.set(self.state_key, serialized)
                    for notice in notices:
                        pipe.xadd(self.notices, {"data": json.dumps(notice)})
                    if event and not event_seen:
                        pipe.xadd(self.history, {"data": json.dumps(event)})
                    await pipe.execute()
                    self.metrics.update(transactions=self.metrics["transactions"]+1, state_bytes=len(serialized.encode()),
                                        last_transaction_seconds=time.monotonic()-started)
                    return state, notices
                except WatchError:
                    self.metrics["cas_retries"] += 1
                    continue

    async def apply(self, event):
        return await self.mutate(lambda state: reduce_event(state, event, copy_state=False), event)

    async def prediction(self, generation, prediction, status, available_at=None):
        def update(state):
            return state, apply_prediction(state, generation, prediction, status, available_at)
        return await self.mutate(update)
