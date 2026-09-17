import asyncio
import logging
import json
import uuid
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


class Publisher:
    def __init__(self, redis, clock, stream="timepredict:events", capacity=4096):
        self.redis, self.clock, self.stream = redis, clock, stream
        self.queue = asyncio.Queue(capacity)
        self.seq = 0
        self.dropped = 0
        self.task = None
        self.heartbeat_task = None
        self.overheads = defaultdict(lambda: deque(maxlen=8192))

    def record_overhead(self, phase, seconds):
        self.overheads[phase].append(seconds)

    def telemetry(self):
        return {"queue_depth": self.queue.qsize(), "dropped": self.dropped,
                "hook_seconds": {phase: {"samples": len(values), "p99": sorted(values)[int((len(values)-1)*0.99)]} for phase, values in self.overheads.items() if values}}

    def emit(self, kind, payload=None, stamp=None):
        self.seq += 1
        event = {**(payload or {}), "kind": kind, "stamp": stamp or self.clock.stamp(), "seq": self.seq, "event_id": uuid.uuid4().hex}
        try:
            self.queue.put_nowait(event)
        except asyncio.QueueFull:
            self.dropped += 1
            logger.error("timepredict queue overflow; source sequence gap=%s", self.seq)

    def start(self):
        self.task = asyncio.create_task(self.run())
        self.heartbeat_task = asyncio.create_task(self.heartbeat())

    async def heartbeat(self):
        while True:
            await asyncio.to_thread(self.clock.refresh)
            self.emit("source_heartbeat", {"telemetry": self.telemetry()})
            await asyncio.sleep(5)

    async def run(self):
        while True:
            event = await self.queue.get()
            delay = 0.1
            while True:
                try:
                    await self.redis.xadd(self.stream, {"data": json.dumps(event, ensure_ascii=False, allow_nan=False)})
                    self.queue.task_done()
                    break
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.warning("timepredict Redis unavailable; retrying", exc_info=True)
                    await asyncio.sleep(delay)
                    delay = min(delay * 2, 5)

    async def stop(self):
        if self.heartbeat_task:
            self.heartbeat_task.cancel()
        try:
            await asyncio.wait_for(self.queue.join(), 3)
        except asyncio.TimeoutError:
            logger.error("timepredict shutdown with undelivered events; observation is incomplete")
        tasks = [t for t in (self.task, self.heartbeat_task) if t]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
