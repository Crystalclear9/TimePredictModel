"""Exercise the runnable Agent example over mocked OpenAI HTTP responses."""

import asyncio
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("finish", ["length", "abort", "error", "tool_calls"])
def test_agent_executes_only_complete_responses_and_reports_lifecycle(monkeypatch, stream, finish):
    path = Path(__file__).resolve().parents[1] / "examples" / "agent_client.py"
    spec = importlib.util.spec_from_file_location("agent_example_fixture", path)
    agent = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(agent)
    tool = {
        "id": "c",
        "type": "function",
        "function": {"name": "search_documents", "arguments": '{"query":"safe fixture"}'},
    }
    lifecycle = []
    requests = []

    def handle(request):
        if "/internal/lifecycle/" in request.url.path:
            lifecycle.append(request.url.path.rsplit("/", 1)[-1])
            return httpx.Response(200, json={"accepted": True})
        requests.append(json.loads(request.content))
        if len(requests) > 1:
            assert requests[-1]["messages"][-1]["tool_call_id"] == "c"
            assert json.loads(requests[-1]["messages"][-1]["content"])["query"] == "safe fixture"
            if stream:
                body = 'data: {"choices":[{"delta":{"content":"complete"},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n'
                return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {"role": "assistant", "content": "complete"},
                            "finish_reason": "stop",
                        }
                    ]
                },
            )
        if stream:
            chunks = [
                {
                    "choices": [
                        {"delta": {"tool_calls": [{"index": 0, **tool}]}, "finish_reason": finish}
                    ]
                }
            ]
            body = (
                "".join("data: " + json.dumps(item) + "\n\n" for item in chunks)
                + "data: [DONE]\n\n"
            )
            return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {"role": "assistant", "tool_calls": [tool]},
                        "finish_reason": finish,
                    }
                ]
            },
        )

    client_type = httpx.AsyncClient
    monkeypatch.setattr(
        agent.httpx,
        "AsyncClient",
        lambda **kw: client_type(transport=httpx.MockTransport(handle), **kw),
    )
    monkeypatch.setenv("SGLANG_API_KEY", "fixture")
    monkeypatch.setenv("TIMEPREDICT_API_KEY", "fixture-internal")
    slept = []
    original_sleep = asyncio.sleep

    async def sleep(seconds):
        if seconds == 0.2:
            slept.append("tool executed")
        await original_sleep(0)

    monkeypatch.setattr(agent.asyncio, "sleep", sleep)
    args = SimpleNamespace(
        predictor="http://fixture", sglang="http://fixture", model="fixture", stream=stream
    )
    if finish == "tool_calls":
        asyncio.run(agent.main(args))
        assert slept == ["tool executed"] and len(requests) == 2
        assert "terminate" in lifecycle and "cancel" not in lifecycle
    else:
        with pytest.raises(RuntimeError, match="incomplete|truncated"):
            asyncio.run(agent.main(args))
        assert not slept and len(requests) == 1
        assert "cancel" in lifecycle and "terminate" in lifecycle
