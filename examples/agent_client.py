# 文件作用：串行单工具 Agent 示例：向 patched SGLang 发请求，执行演示工具并续接，维护观察心跳。
"""OpenAI wire-compatible single-tool Agent; run against a patched server."""
import argparse
import asyncio
import contextlib
import json
import os
import uuid

import httpx


# 创建业务身份，最多执行五轮，支持流式/非流式；完成或异常时发送生命周期事件。
async def main(args):
    workflow, agent = uuid.uuid4().hex, "agent-1"
    key = os.environ["SGLANG_API_KEY"]
    lifecycle_key = os.environ.get("TIMEPREDICT_API_KEY", key)
    async with httpx.AsyncClient(timeout=120) as client:
        # 用内部令牌上报当前工作流/Agent 的 heartbeat、cancel 或 terminate。
        async def lifecycle(action):
            response = await client.post(args.predictor + "/internal/lifecycle/" + action,
                headers={"Authorization": "Bearer " + lifecycle_key}, json={"workflow_id": workflow, "agent_id": agent})
            response.raise_for_status()
        # 每五秒上报存活，覆盖工具等待阶段。
        async def heartbeat():
            while True:
                await lifecycle("heartbeat")
                await asyncio.sleep(5)
        await lifecycle("heartbeat")
        heartbeat_task = asyncio.create_task(heartbeat())
        messages = [{"role": "user", "content": "Use search_documents to find documents about KV cache."}]
        tools = [{"type": "function", "function": {"name": "search_documents", "description": "Search the local demonstration document index",
                    "parameters": {"type": "object", "properties": {"query": {"type": "string"}, "max_results": {"type": "integer"}}, "required": ["query"]}}}]
        try:
            for _ in range(5):
                if heartbeat_task.done():
                    heartbeat_task.result()
                request_id = uuid.uuid4().hex  # Preserve this value if your transport retries this request.
                headers = {"Authorization": "Bearer " + key, "X-Workflow-ID": workflow, "X-Agent-ID": agent, "X-Request-ID": request_id}
                payload = {"model": args.model, "messages": messages, "tools": tools, "parallel_tool_calls": False, "n": 1, "stream": args.stream}
                if args.stream:
                    calls, content, done = {}, "", False
                    async with client.stream("POST", args.sglang + "/v1/chat/completions", headers=headers, json=payload) as response:
                        response.raise_for_status()
                        async for line in response.aiter_lines():
                            if not line.startswith("data: "):
                                continue
                            if line[6:] == "[DONE]":
                                done = True
                                break
                            chunk = json.loads(line[6:])
                            if "error" in chunk:
                                raise RuntimeError(chunk["error"])
                            for choice in chunk.get("choices", []):
                                delta = choice.get("delta", {})
                                content += delta.get("content") or ""
                                for item in delta.get("tool_calls") or []:
                                    call = calls.setdefault(item["index"], {"id": "", "type": "function", "function": {"name": "", "arguments": ""}})
                                    if item.get("id"):
                                        call["id"] = item["id"]
                                    for field in ("name", "arguments"):
                                        call["function"][field] += item.get("function", {}).get(field) or ""
                    if not done:
                        raise RuntimeError("incomplete stream; tool must not execute")
                    message = {"role": "assistant", "content": content, "tool_calls": [calls[i] for i in sorted(calls)]}
                else:
                    response = await client.post(args.sglang + "/v1/chat/completions", headers=headers, json=payload)
                    response.raise_for_status()
                    message = response.json()["choices"][0]["message"]
                calls = message.get("tool_calls") or []
                if not calls:
                    print(message.get("content", ""))
                    break
                if len(calls) != 1:
                    raise RuntimeError("this example requires exactly one tool")
                messages.append(message)
                call = calls[0]
                if call["function"]["name"] != "search_documents":
                    raise RuntimeError("unknown tool")
                arguments = json.loads(call["function"]["arguments"])
                # Replace this harmless demo executor with your real tool implementation.
                await asyncio.sleep(0.2)
                result = {"query": arguments["query"], "documents": ["KV cache demonstration document"]}
                messages.append({"role": "tool", "tool_call_id": call["id"], "content": json.dumps(result)})
            await lifecycle("terminate")
        except BaseException:
            with contextlib.suppress(Exception):
                await lifecycle("cancel")
                await lifecycle("terminate")
            raise
        finally:
            heartbeat_task.cancel()
            await asyncio.gather(heartbeat_task, return_exceptions=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sglang", default="http://127.0.0.1:30000")
    parser.add_argument("--predictor", default="http://127.0.0.1:8098")
    parser.add_argument("--model", required=True)
    parser.add_argument("--stream", action="store_true")
    asyncio.run(main(parser.parse_args()))
