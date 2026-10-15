# 文件作用：FastAPI 内部服务与单 leader worker：消费事件、调度 CPU 预测、刷新候选并提供租户隔离接口。
import asyncio
import contextlib
import json
import logging
import re
import secrets
import time
import uuid
import math
from typing import Literal
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, HTTPException, Query
from pydantic import BaseModel, Field
from redis.exceptions import ResponseError

from .clock import Clock, elapsed
from .distribution import LogNormal, UnreliableDistribution
from .model import Model
from .state import compact, RETRYABLE_PREDICTIONS
from .events import decode_event
from .freshness import observation_status, PROCESSOR_TTL
from .metrics import Histograms

logger = logging.getLogger(__name__)


# 过滤租户与过期观察；仅已校准分布计算条件剩余时间，其他预测保留分位数并降级 unknown。
def window_rows(state, stamp, horizon, tenant=None):
    rows = []
    for gen in state["pending"].values():
        record = state["records"][gen]
        if tenant is not None and record["tenant"] != tenant:
            continue
        row = {k: record[k] for k in ("key", "generation", "tenant", "workflow_id", "agent_id", "revision")}
        row.update(status=record["prediction_status"], in_window=False, distribution=record["prediction"])
        reason, ttl = observation_status(state, record, stamp)
        row["valid_for_seconds"] = ttl
        row["expires_at_utc"] = stamp["utc"] + ttl
        if reason:
            row.update(status="unknown", reason=reason)
            rows.append(row)
            continue
        try:
            age = elapsed(record["out"], stamp)
            row["age_seconds"] = age
            prediction = record["prediction"]
            if prediction and record["prediction_status"] == "ready":
                row["model_version"] = prediction["model_version"]
                row["p50_seconds"] = prediction.get("p50_seconds")
                row["p90_seconds"] = prediction.get("p90_seconds")
                if not prediction.get("distribution_calibrated", False) and not (prediction.get("shadow") and "mu" in prediction):
                    row.update(status="unknown", reason="distribution_not_calibrated", in_window=False)
                else:
                    row.update(LogNormal(prediction["mu"], prediction["sigma"]).summarize(age, horizon))
                    row["status"] = "ready"
                    row["in_window"] = row["remaining_p50_seconds"] <= horizon and not prediction.get("shadow", False)
                    row["shadow"] = prediction.get("shadow", False)
            elif row["status"] != "pending_prediction":
                row["reason"], row["status"] = row["status"], "unknown"
        except (ValueError, UnreliableDistribution) as exc:
            row.update(status="unknown", reason=str(exc))
        rows.append(row)
    return {"api_version": 1, "time_unit": "seconds", "snapshot_utc": stamp["utc"], "revision": state["revision"], "H_seconds": horizon, "records": rows}


# 在 Redis 租约保护下处理事件和状态；API 副本可独立读取共享状态。
class Worker:
    # 初始化周期、预测并发上限四、运行指标和任务集合。
    def __init__(self, store, model_dir=None, clock=None, horizon=2., refresh=0.1, heartbeat_ttl=30, shadow=False):
        self.store, self.model_dir, self.clock = store, model_dir, clock or Clock()
        self.horizon, self.refresh, self.heartbeat_ttl = horizon, refresh, heartbeat_ttl
        self.model = None
        self.shadow = shadow
        self.inflight = set()
        self.prediction_tasks = set()
        # CPU 预测最多四个并发；tick 另将已排队 generation 限制为 64 个。
        self.semaphore = asyncio.Semaphore(4)
        self.metrics = {"events": 0, "prediction_errors": 0, "predictions": 0, "redis_errors": 0}
        self.last_error = None
        self.consumer_error = None
        self.histograms = Histograms()

    # 每秒调用时钟刷新；Clock 内部进一步限制 chronyc 查询频率。
    async def maintain_clock(self):
        while True:
            await asyncio.to_thread(self.clock.refresh)
            await asyncio.sleep(1)

    # 持续运行 worker，异常后记录原因并两秒后重启；取消信号正常向上传递。
    async def supervise(self):
        while True:
            try:
                await self.run()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.last_error = type(exc).__name__
                logger.exception("worker restarting after failure")
                await asyncio.sleep(2)

    # 在线程执行一次 CPU 预测并条件写回；等待期间已返回/取消的轮次不再更新。
    async def predict_one(self, record):
        generation = record["generation"]
        started = time.monotonic()
        try:
            async with self.semaphore:
                self.histograms.observe("prediction_task_queue_seconds", time.monotonic()-started)
                compute_started = time.monotonic()
                if self.model:
                    model = self.model
                    for attempt in range(3):
                        try:
                            prediction, status = await asyncio.to_thread(model.predict, record["call"])
                            break
                        except (TimeoutError, ConnectionError):
                            self.metrics["prediction_errors"] += 1
                            if attempt == 2:
                                prediction, status = None, "prediction_dependency_unavailable"
                            else:
                                await asyncio.sleep(0.1 * 2**attempt)
                        except Exception:
                            self.metrics["prediction_errors"] += 1
                            logger.exception("prediction failed permanently")
                            prediction, status = None, "prediction_error"
                            break
                else:
                    prediction, status = None, "model_unavailable"
                self.histograms.observe("prediction_compute_seconds", time.monotonic()-compute_started)
                if prediction:
                    prediction["shadow"] = self.shadow
                _, applied = await self.store.prediction(generation, prediction, status, self.clock.stamp())
                if not applied:
                    return  # Returned/cancelled/replaced while CPU inference was running.
                completed = self.clock.stamp()
                try:
                    self.histograms.observe("prediction_available_seconds", elapsed(record["out"], completed))
                except ValueError:
                    self.metrics["latency_clock_unknown"] = self.metrics.get("latency_clock_unknown", 0)+1
                self.metrics["predictions"] += 1
                self.metrics["last_prediction_seconds"] = time.monotonic() - started
        except Exception:
            self.metrics["prediction_errors"] += 1
            logger.exception("prediction failed")
        finally:
            self.inflight.discard(generation)

    # 优先接管全部未确认消息，再读取新消息；状态/历史成功提交后才 XACK。
    async def consume(self):
        redis = self.store.redis
        if not self.store.required_lease:
            raise RuntimeError("event consumer requires an elected worker lease")
        try:
            await redis.xgroup_create(self.store.events, "predictor", id="0", mkstream=True)
        except ResponseError as exc:
            if "BUSYGROUP" not in str(exc):
                raise
        consumer = uuid.uuid4().hex
        while True:
            try:
                # The elected worker is the only authorized state writer.
                # Drain even freshly pending entries before reading newer IDs;
                # waiting for an idle timeout creates false source sequence gaps
                # during a fast handover. Every apply is fenced by the lease CAS.
                claimed = await redis.xautoclaim(self.store.events, "predictor", consumer, 0, "0-0", count=100)
                messages = claimed[1]
                if not messages:
                    batches = await redis.xreadgroup("predictor", consumer, {self.store.events: ">"}, count=100, block=500)
                    messages = [item for _, batch in batches for item in batch]
                for event_id, fields in messages:
                    try:
                        event = decode_event(fields.get("data"))
                    except ValueError:
                        await self.store.quarantine(event_id, fields, self.clock.stamp())
                        await redis.xack(self.store.events, "predictor", event_id)
                        self.metrics["quarantined_events"] = self.metrics.get("quarantined_events", 0)+1
                        continue
                    event["observed_at"] = time.time()
                    try:
                        self.histograms.observe("event_delivery_seconds", elapsed(event["stamp"], self.clock.stamp()))
                    except ValueError:
                        pass
                    await self.store.apply(event)
                    await redis.xack(self.store.events, "predictor", event_id)
                    self.metrics["events"] += 1
                    self.consumer_error = None
                await asyncio.sleep(0.001)
            except asyncio.CancelledError:
                raise
            except Exception:
                self.consumer_error = "event_consumer_failure"
                self.metrics["redis_errors"] += 1
                logger.exception("event consumer failure")
                await asyncio.sleep(1)

    # 周期加载模型、失效过期观察、调度未预测轮次，并发布版本化候选快照。
    async def tick(self):
        last_load = 0.
        last_compact = 0.
        last_refresh = 0.
        while True:
            try:
                if self.model_dir and time.monotonic() - last_load > 5:
                    last_load = time.monotonic()
                    try:
                        self.model = await asyncio.to_thread(Model, self.model_dir, require_promoted=not self.shadow)
                    except (OSError, ValueError):
                        logger.warning("no compatible model; retaining last good model")
                state = await self.store.read()
                now = self.clock.stamp()
                for source_id, source in state["sources"].items():
                    if not source.get("lost") and not source.get("retired") and now["utc"]-source["last"]["utc"] > self.heartbeat_ttl:
                        await self.store.apply({"kind": "source_lost", "event_id": uuid.uuid4().hex, "stamp": now, "lost_source": source_id, "last_seen": source["last"]["utc"]})
                if time.monotonic()-last_compact > 60:
                    await self.store.mutate(lambda s: compact(s, now["utc"]))
                    last_compact = time.monotonic()
                for gen in list(state["pending"].values()):
                    record = state["records"][gen]
                    source = state["sources"].get(record["out"]["source"] + "/" + record["out"]["boot"], {})
                    last_source = source.get("last", record["out"])["utc"]
                    last_agent = state["heartbeats"].get(record["key"], record["out"]["utc"])
                    if now["utc"] - min(last_source, last_agent) > self.heartbeat_ttl:
                        await self.store.apply({"kind": "expire", "event_id": uuid.uuid4().hex, "stamp": now,
                                                "tenant": record["tenant"], "workflow_id": record["workflow_id"], "agent_id": record["agent_id"], "generation": gen})
                    elif ((record["prediction_status"] == "pending_prediction" or
                           (record["prediction_status"] in RETRYABLE_PREDICTIONS and now["utc"] >= record.get("prediction_retry_at", 0)))
                          and gen not in self.inflight and len(self.inflight) < 64):
                        self.inflight.add(gen)
                        task = asyncio.create_task(self.predict_one(record))
                        self.prediction_tasks.add(task)
                        task.add_done_callback(self.prediction_tasks.discard)
                # Every refresh is a versioned snapshot; transitions and snapshots share one CAS domain.
                # 在同一 CAS 事务中更新处理器心跳和候选 revision/通知。
                def refresh(state):
                    state["processor"] = now
                    state["consumer_error"] = self.consumer_error
                    state["processor_model"] = {"version": getattr(self.model, "version", None),
                                                "calibrated": bool(self.model and getattr(self.model, "distribution_calibrated", False)),
                                                "expires_at": getattr(self.model, "acceptance_expires_at", 0.),
                                                "shadow": self.shadow}
                    view = window_rows(state, now, self.horizon)
                    notices = []
                    for row in view["records"]:
                        state["revision"] += 1
                        record = state["records"][row["generation"]]
                        row["revision"] = record["revision"] = state["revision"]
                        notices.append({"action": "candidate_update", "tenant": row["tenant"], "key": row["key"],
                                        "generation": row["generation"], "revision": row["revision"], "H_seconds": self.horizon,
                                        "snapshot_utc": now["utc"], "snapshot_stamp": now, "candidate": row})
                    return state, notices
                if time.monotonic()-last_refresh >= .5:
                    await self.store.mutate(refresh)
                    last_refresh = time.monotonic()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("refresh failed")
            await asyncio.sleep(self.refresh)

    # 同时维护时钟与选主工作，结束时取消时钟任务。
    async def run(self):
        clock_task = asyncio.create_task(self.maintain_clock())
        try:
            await self.run_elected()
        finally:
            clock_task.cancel()
            await asyncio.gather(clock_task, return_exceptions=True)

    # 获取十五秒 leader 租约；同时运行消费、刷新和续租，退出时取消任务并按 token 释放租约。
    async def run_elected(self):
        # A single elected worker preserves per-source ordering; API replicas remain stateless.
        lease = self.store.prefix + ":worker-lease"
        token = uuid.uuid4().hex
        while not await self.store.redis.set(lease, token, nx=True, px=15000):
            await asyncio.sleep(2)
        self.store.required_lease = (lease, token)
        self.last_error = None
        # 每三秒按 token 续租；token 不匹配表示 leader 身份失效。
        async def renew():
            while True:
                await asyncio.sleep(3)
                ok = await self.store.redis.eval("if redis.call('get',KEYS[1]) == ARGV[1] then return redis.call('pexpire',KEYS[1],15000) else return 0 end", 1, lease, token)
                if not ok:
                    raise RuntimeError("worker lease lost")
        tasks = [asyncio.create_task(self.consume()), asyncio.create_task(self.tick()), asyncio.create_task(renew())]
        try:
            await asyncio.gather(*tasks)
        finally:
            for task in tasks + list(self.prediction_tasks):
                task.cancel()
            await asyncio.gather(*tasks, *self.prediction_tasks, return_exceptions=True)
            self.store.required_lease = None
            with contextlib.suppress(Exception):
                await self.store.redis.eval("if redis.call('get',KEYS[1]) == ARGV[1] then return redis.call('del',KEYS[1]) else return 0 end", 1, lease, token)


# 约束生命周期请求体：工作流和 Agent ID 长度为 1..256。
class Lifecycle(BaseModel):
    workflow_id: str = Field(min_length=1, max_length=256)
    agent_id: str = Field(default="*", min_length=1, max_length=256)


class Retirement(BaseModel):
    source_key: str = Field(min_length=1, max_length=4096)
    expected_seq: int = Field(gt=0, strict=True)
    routing_fenced: Literal[True]
    reason: str = Field(min_length=1, max_length=1024)


class Recovery(BaseModel):
    gap_id: str = Field(min_length=1, max_length=256)
    observation_reestablished: Literal[True]
    reason: str = Field(min_length=1, max_length=1024)


# 创建认证 API 和生命周期任务；tokens 必须至少包含一个内部令牌到 tenant 的映射。
def create_app(store, tokens, clock=None, worker=None, *, admin_tokens=()):
    if (not isinstance(tokens, dict) or not tokens or any(
            not isinstance(token, str) or not token or not token.isascii()
            or not isinstance(namespace, str) or not namespace
            for token, namespace in tokens.items())):
        raise ValueError("authentication token mapping requires nonempty ASCII tokens and nonempty tenant strings")
    tokens = dict(tokens)  # Do not let later caller mutations change validated tenant bindings.
    if (not isinstance(admin_tokens, (list, tuple)) or any(not isinstance(t, str) or not t or not t.isascii() for t in admin_tokens)
            or any(t in tokens for t in admin_tokens)):
        raise ValueError("admin tokens must be separate nonempty ASCII credentials")
    admin_tokens = tuple(admin_tokens)
    clock = clock or Clock()

    # 随应用启动后台任务，随应用退出取消并等待任务结束。
    @asynccontextmanager
    async def lifespan(app):
        # 未配置 worker 时仍维护 API 的本地时钟状态。
        async def api_clock():
            while True:
                await asyncio.to_thread(clock.refresh)
                await asyncio.sleep(1)
        task = asyncio.create_task(worker.supervise() if worker else api_clock())
        try:
            yield
        finally:
            if task:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task

    app = FastAPI(title="Tool return prediction", lifespan=lifespan)

    # 常量时间比较 Bearer 令牌并返回绑定的租户命名空间。
    async def tenant(authorization: str = Header(default="")):
        if not authorization.startswith("Bearer ") or not authorization.isascii():
            raise HTTPException(401, "invalid credential")
        token = authorization.removeprefix("Bearer ")
        for expected, namespace in tokens.items():
            if secrets.compare_digest(token, expected):
                return namespace
        raise HTTPException(401, "invalid credential")

    async def admin(authorization: str = Header(default="")):
        if authorization.startswith("Bearer ") and authorization.isascii():
            if any(secrets.compare_digest(authorization[7:], t) for t in admin_tokens):
                return
        raise HTTPException(403, "operator credential required")

    async def administrative_event(payload):
        stamp = clock.stamp()
        error = stamp.get("error")
        if stamp.get("jumped") or error is None or not math.isfinite(error) or not 0 <= error <= .0025:
            raise HTTPException(503, "verified administrative clock required")
        event = {**payload, "stamp": stamp, "event_id": uuid.uuid4().hex}
        try:
            await store.redis.xadd(store.events, {"data": json.dumps(event, allow_nan=False)})
        except Exception as exc:
            raise HTTPException(503, "event transport unavailable") from exc
        return {"accepted": True, "event_id": event["event_id"]}

    async def administrative_state():
        try:
            return await store.read()
        except Exception as exc:
            raise HTTPException(503, "state unavailable") from exc

    @app.post("/internal/admin/retire-source", dependencies=[Depends(admin)])
    async def retire_source(body: Retirement):
        state = await administrative_state()
        source = state["sources"].get(body.source_key)
        if not source or source.get("seq") != body.expected_seq:
            raise HTTPException(409, "source sequence changed or unknown")
        return await administrative_event({"kind": "source_retired", "retired_source": body.source_key,
                                           **body.model_dump(exclude={"source_key"})})

    @app.post("/internal/admin/recover-transport", dependencies=[Depends(admin)])
    async def recover_transport(body: Recovery):
        if body.gap_id not in (await administrative_state()).get("transport_failures", {}):
            raise HTTPException(404, "unknown gap")
        return await administrative_event({"kind": "transport_recovered", **body.model_dump()})

    # 先读取通知游标，再读取快照，减少并发更新时消费端漏通知的风险。
    @app.get("/internal/contract")
    async def contract(namespace=Depends(tenant)):
        return {"api_version": 1, "time_unit": "seconds", "target": "server_ingress - server_tool_response_out",
                "association": "one outstanding single-tool round per tenant/workflow_id/agent_id",
                "distribution": {"kind": "lognormal", "parameters": ["mu", "sigma"], "condition": "T > age_seconds"},
                "window_rule": "remaining_p50_seconds <= H_seconds",
                "scheduler_inputs": ["remaining_p50_seconds", "remaining_p90_seconds", "prob_return_within_H",
                                     "expected_remaining_seconds", "expected_capped_remaining_seconds", "distribution", "age_seconds"],
                "lease": "subtract measured response transit and expire locally; revisions reject stale updates",
                "unknown": "never treat unknown or shadow rows as actionable",
                "responsibility": "external scheduler supplies density scoring, KV node mapping and loading"}

    @app.get("/internal/window")
    async def window(H_seconds: float = Query(gt=0, allow_inf_nan=False), namespace=Depends(tenant)):
        try:
            # Read cursor first: any concurrent changes will either be in the snapshot or replayed.
            last_notice = await store.redis.xrevrange(store.notices, count=1)
            state = await store.read()
        except Exception as exc:
            raise HTTPException(503, "state unavailable") from exc
        result = await asyncio.to_thread(window_rows, state, clock.stamp(), H_seconds, namespace)
        result["notice_cursor"] = last_notice[0][0] if last_notice else "0-0"
        return result

    # 验证 heartbeat/cancel/terminate 并写入事件流；accepted 只表示已入流。
    @app.post("/internal/lifecycle/{action}")
    async def lifecycle(action: str, body: Lifecycle, namespace=Depends(tenant)):
        if action not in ("heartbeat", "cancel", "terminate"):
            raise HTTPException(404, "unknown lifecycle action")
        if action in ("cancel", "heartbeat") and body.agent_id == "*":
            raise HTTPException(422, "cancel and heartbeat require agent_id")
        event = {"event_id": uuid.uuid4().hex, "kind": action, "stamp": clock.stamp(), "tenant": namespace, **body.model_dump()}
        try:
            await store.redis.xadd(store.events, {"data": json.dumps(event)})
        except Exception as exc:
            raise HTTPException(503, "event transport unavailable") from exc
        return {"accepted": True, "event_id": event["event_id"]}

    # 读取游标后的通知并过滤租户；历史候选租约过期时降级而不重新激活。
    @app.get("/internal/notices")
    async def notices(after: str = "0-0", limit: int = Query(default=100, ge=1, le=1000), namespace=Depends(tenant)):
        component = r"(?:0|[1-9][0-9]{0,19})"
        if (not re.fullmatch(component + "-" + component, after)
                or any(int(part) > 2**64-1 for part in after.split("-"))):
            raise HTTPException(422, "invalid stream cursor")
        try:
            messages = await store.redis.xrange(store.notices, min="(" + after, max="+", count=limit)
            if after != "0-0":
                first = await store.redis.xrange(store.notices, count=1)
                if first and tuple(map(int, after.split("-"))) < tuple(map(int, first[0][0].split("-"))):
                    return {"cursor": after, "events": [], "resync_required": True}
            # Read after the page: a later REMOVE may be outside this page but
            # its closed state must already fence the historical candidate.
            state = await store.read()
        except Exception as exc:
            raise HTTPException(503, "notification state unavailable") from exc
        now = clock.stamp()
        events = []
        for ident, fields in messages:
            item = json.loads(fields["data"])
            if item["tenant"] != namespace:
                continue
            candidate = item.get("candidate")
            if candidate:
                # Never replay an expired actionable lease from the history stream.
                try:
                    transit = elapsed(item["snapshot_stamp"], now)
                    left = candidate.get("valid_for_seconds", 0)-transit-0.005
                    failure = "expired_notice" if left <= 0 else None
                except (KeyError, ValueError) as exc:
                    left, failure = 0., "notice_clock_unverified" if isinstance(exc, KeyError) else str(exc)
                if not failure:
                    current = state["records"].get(item["generation"])
                    if (not current or current["key"] != item["key"] or current["tenant"] != namespace
                            or current["status"] != "waiting" or not current["valid"]
                            or state["pending"].get(item["key"]) != item["generation"]):
                        failure = "round_no_longer_pending"
                    elif current["prediction_status"] != "ready":
                        failure = "prediction_no_longer_ready"
                    else:
                        failure, current_ttl = observation_status(state, current, now)
                        left = min(left, current_ttl)
                if failure:
                    candidate.update(status="unknown", reason=failure, in_window=False, valid_for_seconds=0.)
                else:
                    candidate["valid_for_seconds"] = min(candidate.get("valid_for_seconds", 0), left)
                    # Notices may sit in the history stream before being fetched.
                    # Advance their age as well as TTL, so consumers do not condition on an old age.
                    candidate["age_seconds"] = candidate.get("age_seconds", 0.)+transit
                    prediction = candidate.get("distribution") or {}
                    if candidate.get("status") == "ready" and "mu" in prediction:
                        try:
                            candidate.update(LogNormal(prediction["mu"], prediction["sigma"]).summarize(
                                candidate["age_seconds"], item["H_seconds"]))
                            candidate["in_window"] = candidate["remaining_p50_seconds"] <= item["H_seconds"] and not prediction.get("shadow", False)
                        except (ValueError, UnreliableDistribution):
                            candidate.update(status="unknown", reason="unsupported_tail", in_window=False)
            events.append({"id": ident, **item})
        return {"cursor": messages[-1][0] if messages else after,
                "events": events}

    # 报告 Redis、处理器新鲜度、模型、时钟和指标；degraded 响应体仍使用 HTTP 200。
    @app.get("/internal/health")
    async def health(namespace=Depends(tenant)):
        try:
            await store.redis.ping()
        except Exception as exc:
            raise HTTPException(503, "Redis unavailable") from exc
        try:
            state = await store.read()
        except Exception as exc:
            raise HTTPException(503, "state unavailable") from exc
        now = clock.stamp()
        try:
            fresh = elapsed(state["processor"], now) <= PROCESSOR_TTL
        except (ValueError, KeyError):
            fresh = False
        model = state.get("processor_model", {})
        reasons = []
        if not fresh:
            reasons.append("processor_stale")
        if not model.get("version"):
            reasons.append("model_unavailable")
        elif not model.get("calibrated") or model.get("shadow"):
            reasons.append("model_not_actionable")
        elif now["utc"] > model.get("expires_at", 0.):
            reasons.append("model_acceptance_expired")
        if now.get("jumped") or now.get("error") is None or not 0 <= now["error"] <= .0025:
            reasons.append("clock_unverified")
        for source in state["sources"].values():
            if source.get("retired"):
                continue
            try:
                valid = not source.get("bad") and not source.get("lost") and elapsed(source["last"], now) < 30.
            except ValueError:
                valid = False
            if not valid:
                reasons.append("observer_unavailable")
                break
        if not any(not s.get("retired") for s in state["sources"].values()):
            reasons.append("no_observers")
        if any("failure_until" not in gap for gap in state.get("transport_failures", {}).values()):
            reasons.append("transport_gap")
        if state.get("consumer_error"):
            reasons.append("event_consumer_failure")
        return {"status": "degraded" if reasons else "ok", "ready": not reasons, "live": True,
                "readiness_reasons": reasons, "metrics": worker.metrics if worker else {}, "clock": now,
                "store_metrics": store.metrics,
                "processor": state.get("processor"),
                "latency_seconds": worker.histograms.snapshot() if worker else {},
                "worker_last_error": worker.last_error if worker else None,
                "sources": {key: {"bad": value.get("bad", False), "lost": value.get("lost", False),
                                  "retired": value.get("retired", False), "seq": value.get("seq"), "last": value["last"],
                                  "failure_until": value.get("failure_until"), "telemetry": value.get("telemetry", {})}
                            for key, value in state["sources"].items()},
                "transport_gaps": state.get("transport_failures", {}),
                "model_version": model.get("version")}

    @app.get("/internal/ready")
    async def ready(namespace=Depends(tenant)):
        result = await health(namespace)
        if not result["ready"]:
            raise HTTPException(503, {"ready": False, "reasons": result["readiness_reasons"]})
        return {"ready": True, "model_version": result["model_version"]}

    return app
