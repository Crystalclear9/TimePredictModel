# 文件作用：Redis 存储：单 JSON 状态文档通过 WATCH/MULTI/EXEC 原子更新，并同步写通知及历史。
import json
import time

from redis.exceptions import ResponseError, WatchError

from .state import apply_prediction, empty_state, reduce_event


# 用 prefix 隔离状态、事件、历史、通知键以及 leader 租约约束。
class RedisStore:
    # 建立键名、租约前置条件和 CAS 指标。
    def __init__(
        self,
        redis,
        prefix="timepredict",
        *,
        notice_limit=100000,
        max_state_bytes=64 * 1024 * 1024,
        deadletter_limit=10000,
    ):
        self.redis, self.prefix = redis, prefix
        self.state_key = prefix + ":state"
        self.events = prefix + ":events"
        self.history = prefix + ":history"
        self.notices = prefix + ":notices"
        self.deadletters = prefix + ":deadletters"
        if any(
            type(n) is not int or n <= 0 for n in (notice_limit, max_state_bytes, deadletter_limit)
        ):
            raise ValueError("positive storage budgets required")
        self.notice_limit, self.max_state_bytes = notice_limit, max_state_bytes
        self.deadletter_limit = deadletter_limit
        self.required_lease = None
        self.metrics = {
            "transactions": 0,
            "cas_retries": 0,
            "state_bytes": 0,
            "last_transaction_seconds": 0.0,
        }

    # 读取共享状态；尚未初始化时返回空状态。
    async def read(self):
        raw = await self.redis.get(self.state_key)
        return json.loads(raw) if raw else empty_state()

    # 监视状态及可选 leader token；冲突重试，将状态/通知/首次事件历史在同一事务提交。
    async def mutate(self, reducer, event=None, deadletter=None):
        started = time.monotonic()
        while True:
            async with self.redis.pipeline(transaction=True) as pipe:
                try:
                    # EXEC does not roll back SET when a later XADD fails at
                    # runtime. Check stream types before queuing writes and
                    # WATCH them so a concurrent type change aborts the CAS.
                    streams = (
                        (self.history, self.notices, self.deadletters)
                        if deadletter
                        else (self.history, self.notices)
                    )
                    await pipe.watch(self.state_key, *streams)
                    for stream in streams:
                        kind = await pipe.type(stream)
                        if kind not in ("none", "stream", b"none", b"stream"):
                            raise ResponseError("stream_key_type_mismatch: " + stream)
                    if self.required_lease:
                        lease_key, token = self.required_lease
                        await pipe.watch(lease_key)
                        if await pipe.get(lease_key) != token:
                            raise RuntimeError("worker lease lost before transaction")
                    raw = await pipe.get(self.state_key)
                    state = json.loads(raw) if raw else empty_state()
                    event_seen = event is not None and event["event_id"] in state["seen"]
                    if (
                        deadletter
                        and not event_seen
                        and await pipe.xlen(self.deadletters) >= self.deadletter_limit
                    ):
                        raise RuntimeError(
                            "deadletter_capacity_exceeded; archive and resolve quarantined events"
                        )
                    state, notices = reducer(state)
                    serialized = json.dumps(state, allow_nan=False)
                    size = len(serialized.encode())
                    if size > self.max_state_bytes:
                        raise RuntimeError("state_capacity_exceeded")
                    pipe.multi()
                    pipe.set(self.state_key, serialized)
                    for notice in notices:
                        pipe.xadd(
                            self.notices,
                            {"data": json.dumps(notice)},
                            maxlen=self.notice_limit,
                            approximate=False,
                        )
                    if event and not event_seen:
                        pipe.xadd(self.history, {"data": json.dumps(event)})
                        if deadletter:
                            pipe.xadd(self.deadletters, {"data": json.dumps(deadletter)})
                    await pipe.execute()
                    self.metrics.update(
                        transactions=self.metrics["transactions"] + 1,
                        state_bytes=size,
                        last_transaction_seconds=time.monotonic() - started,
                    )
                    return state, notices
                except WatchError:
                    self.metrics["cas_retries"] += 1
                    continue

    # 以事件状态机作为 reducer，原子应用一个事件。
    async def apply(self, event):
        return await self.mutate(lambda state: reduce_event(state, event, copy_state=False), event)

    async def quarantine(self, stream_id, fields, stamp):
        event = {
            "kind": "transport_gap",
            "event_id": "quarantine:" + stream_id,
            "stamp": stamp,
            "gap_id": stream_id,
        }
        deadletter = {
            "stream_id": stream_id,
            "fields": fields,
            "reason": "invalid_event_envelope",
            "stamp": stamp,
        }
        return await self.mutate(
            lambda state: reduce_event(state, event, copy_state=False), event, deadletter
        )

    # 把预测写回包装为共享状态事务。
    async def prediction(self, generation, prediction, status, available_at=None):
        # 返回新状态和 apply_prediction 产生的通知。
        def update(state):
            return state, apply_prediction(state, generation, prediction, status, available_at)

        return await self.mutate(update)
