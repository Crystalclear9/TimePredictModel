"""Optional SGLang v0.5.19 hooks. No model inference happens here."""
import asyncio
import hashlib
import json
import logging
import os
import time
from functools import wraps

from redis.asyncio import Redis

from .clock import Clock
from .publisher import Publisher

logger = logging.getLogger(__name__)


def safe_hook(function):
    @wraps(function)
    def wrapped(value, raw_request):
        observer = raw_request.scope.get("state", {}).get("timepredict_observer")
        if not observer:
            return
        started = time.perf_counter()
        try:
            return function(value, raw_request)
        except Exception:
            logger.exception("timepredict capture failed; inference continues")
            capture = raw_request.scope.get("state", {}).get("timepredict_capture")
            if capture:
                capture.reason = "capture_error"
            observer.emit("gap")
        finally:
            if hasattr(observer, "record_overhead"):
                observer.record_overhead(function.__name__, time.perf_counter()-started)
    return wrapped


def latest_tool_ids(messages):
    ids = []
    for message in reversed(messages):
        role = message.get("role")
        if role == "tool":
            if message.get("tool_call_id"):
                ids.append(message["tool_call_id"])
        elif role == "assistant":
            declared = {t.get("id") for t in message.get("tool_calls") or []}
            return [i for i in ids if i in declared]
        else:
            return []
    return []


class Capture:
    def __init__(self, identity, tools, supported=True):
        self.identity, self.tools = identity, tools
        self.calls = {}
        self.reason = None if supported else "unsupported_choices"
        self.saw_tool = False
        self.completed = False

    def observe(self, data, stream):
        if "error" in data:
            self.reason = "stream_error"
        for choice in data.get("choices", []):
            if choice.get("index", 0) != 0:
                self.reason = "unsupported_choices"
            msg = choice.get("delta" if stream else "message") or {}
            for offset, tool in enumerate(msg.get("tool_calls") or []):
                self.saw_tool = True
                index = tool.get("index", offset)
                current = self.calls.setdefault(index, {"id": "", "name": "", "arguments": ""})
                fun = tool.get("function") or {}
                for field, value in (("id", tool.get("id")), ("name", fun.get("name")), ("arguments", fun.get("arguments"))):
                    if value:
                        if field in ("id", "name") and current[field] == value:
                            continue
                        combined = current[field] + value if stream else value
                        if len(combined.encode("utf-8")) > 256 * 1024:
                            self.reason = "unsupported_input"
                        elif self.reason != "unsupported_input":
                            current[field] = combined
            if choice.get("finish_reason") in ("length", "error", "abort"):
                self.reason = "incomplete_response"

    def payload(self):
        if not self.saw_tool:
            return None
        if len(self.calls) != 1:
            self.reason = "unsupported_multi_tool"
        call = next(iter(self.calls.values()))
        definition = self.tools.get(call["name"], {})
        call = {**call, "description": definition.get("description", ""), "schema": definition.get("parameters", {})}
        try:
            # Bound expensive feature validation to the CPU service; check completeness here.
            if not call["id"] or not call["name"] or not isinstance(json.loads(call["arguments"]), dict):
                self.reason = self.reason or "incomplete_arguments"
        except (ValueError, TypeError, RecursionError):
            self.reason = self.reason or "incomplete_arguments"
        return {**self.identity, "call": call, "reason": self.reason}


@safe_hook
def accepted(request, raw_request):
    """Called by the authenticated route after request validation."""
    state = raw_request.scope.get("state", {})
    observer = state.get("timepredict_observer")
    if not observer:
        return
    serving = raw_request.app.state.openai_serving_chat
    if not getattr(serving.tokenizer_manager.server_args, "api_key", None):
        return
    headers = raw_request.headers
    identity = {"workflow_id": headers.get("x-workflow-id"), "agent_id": headers.get("x-agent-id"), "request_id": headers.get("x-request-id")}
    if not all(identity.values()):
        return
    # Only authenticated deployments are enabled. Tenant comes from the validated credential.
    credential = headers.get("authorization", "")
    if not credential.startswith("Bearer "):
        return
    identity["tenant"] = hashlib.sha256(credential[7:].encode()).hexdigest()
    data = request.model_dump(exclude_none=True)
    tools = {t["function"]["name"]: t["function"] for t in data.get("tools", []) if t.get("type") == "function"}
    capture = Capture(identity, tools, data.get("n", 1) == 1 and data.get("parallel_tool_calls", False) is False)
    state["timepredict_capture"] = capture
    latest_ids = latest_tool_ids(data.get("messages", []))
    observer.emit("ingress", {**identity, "latest_tool_ids": latest_ids,
                                "tool_call_ids_present": bool(latest_ids)}, state["timepredict_received"])


@safe_hook
def observe_nonstream(response, raw_request):
    capture = raw_request.scope.get("state", {}).get("timepredict_capture")
    if capture and hasattr(response, "model_dump"):
        capture.observe(response.model_dump(exclude_none=True), False)


async def observe_stream(generator, raw_request):
    capture = raw_request.scope.get("state", {}).get("timepredict_capture")
    async for chunk in generator:
        started = time.perf_counter()
        if capture:
            # SGLang yields complete SSE messages here; socket framing is irrelevant.
            for line in chunk.splitlines():
                if line.startswith("data: ") and line[6:] != "[DONE]":
                    try:
                        capture.observe(json.loads(line[6:]), True)
                    except (ValueError, TypeError, AttributeError, KeyError):
                        capture.reason = "invalid_stream"
        observer = raw_request.scope.get("state", {}).get("timepredict_observer")
        if observer and hasattr(observer, "record_overhead"):
            observer.record_overhead("stream_chunk", time.perf_counter()-started)
        yield chunk


class ObservationMiddleware:
    def __init__(self, app, redis_url=None, publisher=None):
        self.app = app
        self.publisher = publisher
        self.redis_url = redis_url or os.environ.get("TIMEPREDICT_REDIS_URL")
        self.owned = publisher is None

    async def __call__(self, scope, receive, send):
        if scope["type"] == "lifespan":
            if self.redis_url and self.publisher is None:
                redis = Redis.from_url(self.redis_url, decode_responses=True, socket_timeout=2)
                self.publisher = Publisher(redis, Clock(), os.environ.get("TIMEPREDICT_PREFIX", "timepredict") + ":events")
                self.publisher.start()
            try:
                await self.app(scope, receive, send)
            finally:
                if self.publisher and self.owned:
                    await self.publisher.stop()
                    await self.publisher.redis.aclose()
            return
        if scope["type"] != "http" or scope["path"] != "/v1/chat/completions" or not self.publisher:
            await self.app(scope, receive, send)
            return
        state = scope.setdefault("state", {})
        state["timepredict_observer"] = self.publisher
        state["timepredict_received"] = self.publisher.clock.stamp()
        streaming, success = False, False

        async def wrapped_send(message):
            nonlocal streaming, success
            if message["type"] == "http.response.start":
                success = message["status"] == 200
                streaming = any(k.lower() == b"content-type" and b"text/event-stream" in v for k, v in message.get("headers", []))
            capture = state.get("timepredict_capture")
            terminal = success and capture and message["type"] == "http.response.body" and (
                b"data: [DONE]" in message.get("body", b"") if streaming else not message.get("more_body", False))
            stamp = self.publisher.clock.stamp() if terminal else None
            if terminal and capture and not state.get("timepredict_prepared"):
                prepared = capture.payload()
                if prepared:
                    self.publisher.emit("prepared", prepared, stamp)
                    state["timepredict_prepared"] = True
            try:
                await send(message)
            except BaseException:
                if capture and not capture.completed:
                    payload = capture.payload()
                    if payload:
                        self.publisher.emit("out", {**payload, "reason": "delivery_failed"}, stamp)
                    capture.completed = True
                raise
            if terminal and not capture.completed:
                started = time.perf_counter()
                payload = capture.payload()
                if payload:
                    self.publisher.emit("out", payload, stamp)
                capture.completed = True
                if hasattr(self.publisher, "record_overhead"):
                    self.publisher.record_overhead("out", time.perf_counter()-started)

        try:
            await self.app(scope, receive, wrapped_send)
        finally:
            capture = state.get("timepredict_capture")
            if capture and not capture.completed and capture.saw_tool:
                payload = capture.payload()
                self.publisher.emit("out", {**payload, "reason": "delivery_failed"})
                capture.completed = True


def install(app, server_args):
    if os.environ.get("TIMEPREDICT_REDIS_URL"):
        worker_count = getattr(server_args, "tokenizer_worker_num", None)
        if worker_count != 1:
            raise ValueError("timepredict requires one authenticated HTTP worker per replica; deploy multiple replicas for multiple ingress")
        app.add_middleware(ObservationMiddleware)
