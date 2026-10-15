# 文件作用：入口异步事件发布：采集钩子只入本地有界队列，后台按序写 Redis Streams。
import asyncio
import logging
import json
import uuid
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


# 管理事件序号、来源心跳、队列、丢弃计数和钩子开销。
class Publisher:
    # 初始化容量默认 4096 的异步队列与滚动遥测窗口。
    def __init__(self, redis, clock, stream="timepredict:events", capacity=4096):
        self.redis, self.clock, self.stream = redis, clock, stream
        self.queue = asyncio.Queue(capacity)
        self.seq = 0
        self.dropped = 0
        self.task = None
        self.heartbeat_task = None
        self.overheads = defaultdict(lambda: deque(maxlen=8192))

    # 保留各采集阶段的耗时样本。
    def record_overhead(self, phase, seconds):
        self.overheads[phase].append(seconds)

    # 输出队列深度、累计丢弃及每个钩子的窗口 P99。
    def telemetry(self):
        return {"queue_depth": self.queue.qsize(), "dropped": self.dropped,
                "hook_seconds": {phase: {"samples": len(values), "p99": sorted(values)[int((len(values)-1)*0.99)]} for phase, values in self.overheads.items() if values}}

    # 非阻塞入队；先增加 seq，使队列溢出能在后续事件中表现为序号缺口。
    def emit(self, kind, payload=None, stamp=None):
        self.seq += 1
        event = {**(payload or {}), "kind": kind, "stamp": stamp or self.clock.stamp(), "seq": self.seq, "event_id": uuid.uuid4().hex}
        try:
            self.queue.put_nowait(event)
        except asyncio.QueueFull:
            self.dropped += 1
            logger.error("timepredict queue overflow; source sequence gap=%s", self.seq)

    # 启动事件发送和来源心跳两个后台任务。
    def start(self):
        self.task = asyncio.create_task(self.run())
        self.heartbeat_task = asyncio.create_task(self.heartbeat())

    # 刷新时钟并每五秒发送一次来源心跳和遥测。
    async def heartbeat(self):
        while True:
            await asyncio.to_thread(self.clock.refresh)
            self.emit("source_heartbeat", {"telemetry": self.telemetry()})
            await asyncio.sleep(5)

    # 逐个写入事件；Redis 失败时指数退避并重试当前事件，保持顺序。
    async def run(self):
        while True:
            event = await self.queue.get()
            try:
                serialized = json.dumps(event, ensure_ascii=False, allow_nan=False)
                serialized.encode("utf-8")
            except (TypeError, ValueError, UnicodeError):
                self.dropped += 1
                self.queue.task_done()
                logger.exception("timepredict non-JSON event dropped; following sequence exposes the gap")
                continue
            delay = 0.1
            while True:
                try:
                    await self.redis.xadd(self.stream, {"data": serialized})
                    self.queue.task_done()
                    break
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.warning("timepredict Redis unavailable; retrying", exc_info=True)
                    await asyncio.sleep(delay)
                    delay = min(delay * 2, 5)

    # 停止心跳并最多等待三秒排空队列，然后取消后台任务。
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
