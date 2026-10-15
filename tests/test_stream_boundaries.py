"""Only a real SSE terminal event may establish the server OUT boundary."""

import asyncio
import json

import pytest

from timepredict.sglang import Capture, ObservationMiddleware


@pytest.mark.parametrize("text", ["data: [DONE]", "prefix data: [DONE] suffix"])
def test_done_text_inside_tool_arguments_does_not_end_response(text):
    captured = []
    current_time = [1.0]

    class ClockFixture:
        def stamp(self):
            t = current_time[0]
            return {
                "utc": t,
                "mono": t,
                "source": "s",
                "boot": "b",
                "error": 0.001,
                "jumped": False,
            }

    class PublisherFixture:
        clock = ClockFixture()

        def emit(self, kind, payload, stamp=None):
            captured.append({"kind": kind, "payload": payload, "stamp": stamp})

    async def app(scope, receive, send):
        capture = Capture({"request_id": "r1"}, {})
        scope["state"]["timepredict_capture"] = capture
        chunk = {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "c",
                                "function": {
                                    "name": "search",
                                    "arguments": json.dumps({"query": text}),
                                },
                            }
                        ]
                    }
                }
            ]
        }
        capture.observe(chunk, True)
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [(b"content-type", b"text/event-stream")],
            }
        )
        await send(
            {
                "type": "http.response.body",
                "body": ("data: " + json.dumps(chunk) + "\n\n").encode(),
                "more_body": True,
            }
        )
        assert not captured, "tool arguments are data, not the terminal event"
        current_time[0] = 3.0
        await send({"type": "http.response.body", "body": b"data: [DONE]\n\n", "more_body": True})
        await send({"type": "http.response.body", "body": b"", "more_body": False})

    async def send(message):
        pass

    asyncio.run(
        ObservationMiddleware(app, publisher=PublisherFixture())(
            {"type": "http", "path": "/v1/chat/completions"}, None, send
        )
    )
    assert [e["kind"] for e in captured] == ["prepared", "out"]
    assert all(e["stamp"]["utc"] == 3.0 for e in captured)
