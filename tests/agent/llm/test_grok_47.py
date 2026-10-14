"""Grok 4.7 native Chat Completions serialization and tool-loop regression."""

import json
from typing import Any, cast

import httpx
import pytest
from openai import AsyncOpenAI

from kolega_code.llm.client import LLMClient
from kolega_code.llm.models import (
    ImageBlock,
    Message,
    MessageHistory,
    TextBlock,
    ToolCall,
    ToolDefinition,
    ToolParameter,
    ToolResult,
)
from kolega_code.llm.providers import openai as openai_provider
from kolega_code.llm.providers.openai import OpenAIProvider


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["generate", "stream"])
@pytest.mark.parametrize("effort", ["low", "medium", "high", "xhigh"])
async def test_grok_47_native_requests_and_tool_result_replay(
    monkeypatch: pytest.MonkeyPatch, method: str, effort: str
) -> None:
    requests: list[dict[str, Any]] = []
    headers: list[httpx.Headers] = []
    call = {
        "id": "call-release",
        "type": "function",
        "function": {"name": "lookup", "arguments": '{"model":"grok-4.7"}'},
    }

    def respond(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        requests.append(payload)
        headers.append(request.headers)
        tool_turn = len(requests) == 1
        if payload.get("stream"):
            delta = (
                {"role": "assistant", "tool_calls": [{"index": 0, **call}], "reasoning_content": "Look up the release."}
                if tool_turn
                else {"content": "READY"}
            )
            chunks = [
                {
                    "id": "chatcmpl-grok",
                    "object": "chat.completion.chunk",
                    "created": 0,
                    "model": "grok-4.7",
                    "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
                },
                {
                    "id": "chatcmpl-grok",
                    "object": "chat.completion.chunk",
                    "created": 0,
                    "model": "grok-4.7",
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls" if tool_turn else "stop"}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                },
            ]
            data = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks) + "data: [DONE]\n\n"
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=data)
        assistant = (
            {"role": "assistant", "content": None, "tool_calls": [call], "reasoning_content": "Look up the release."}
            if tool_turn
            else {
                "role": "assistant",
                "content": "READY",
            }
        )
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-grok",
                "object": "chat.completion",
                "created": 0,
                "model": "grok-4.7",
                "choices": [{"index": 0, "message": assistant, "finish_reason": "tool_calls" if tool_turn else "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            },
        )

    def sdk_client(**kwargs: Any) -> AsyncOpenAI:
        return AsyncOpenAI(**kwargs, http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)))

    monkeypatch.setattr(openai_provider, "AsyncOpenAI", sdk_client)
    client = LLMClient(provider="xai", api_key="test-key", max_retries=0)
    provider = cast(OpenAIProvider, client.provider)
    history = MessageHistory(
        [
            Message(
                role="user",
                content=[
                    TextBlock("Look up this model."),
                    ImageBlock(image_type="url", media_type="image/png", data="https://example.invalid/model.png"),
                ],
            )
        ]
    )
    tool = ToolDefinition(
        name="lookup",
        description="Look up a model.",
        parameters=[ToolParameter(name="model", type="string", description="Model ID.", required=True)],
    )

    async def turn() -> Message:
        kwargs = {
            "model": "grok-4.7",
            "thinking": effort,
            "temperature": 0.7,
            "max_completion_tokens": 1024,
            "tools": [tool],
        }
        if method == "generate":
            return await client.generate(history, **kwargs)
        stream = await cast(Any, client.stream(history, **kwargs))
        async with stream:
            async for _ in stream:
                pass
            return await stream.get_final_message()

    try:
        first = await turn()
        calls = [block for block in first.content if isinstance(block, ToolCall)]
        assert len(calls) == 1
        assert calls[0].id == "call-release"
        assert calls[0].input == {"model": "grok-4.7"}
        assert first.stop_reason == "tool_use"
        history.append(first)
        history.append(
            Message(role="user", content=[ToolResult("call-release", content="READY", name="lookup", is_error=False)])
        )
        second = await turn()
        assert second.get_text_content() == "READY"
        assert second.stop_reason == "end_turn"
        assert second.usage_metadata["total_tokens"] == 15
    finally:
        await provider.async_client.close()

    assert len(requests) == 2
    for request, request_headers in zip(requests, headers):
        assert request["model"] == "grok-4.7"
        assert request["reasoning_effort"] == effort
        assert request["temperature"] == 0.7
        assert request["max_tokens"] == 1024
        assert request["tools"][0]["function"]["name"] == "lookup"
        assert request_headers["x-grok-conv-id"] == provider._session_id
        if method == "stream":
            assert request["stream_options"] == {"include_usage": True}
    assert requests[0]["messages"][0]["content"][1] == {
        "type": "image_url",
        "image_url": {"url": "https://example.invalid/model.png"},
    }
    replay = requests[1]["messages"][-2]
    assert replay["reasoning_content"] == "Look up the release."
    replay_calls = replay["tool_calls"]
    assert len(replay_calls) == 1
    assert replay_calls[0]["id"] == "call-release"
    assert replay_calls[0]["type"] == "function"
    assert replay_calls[0]["function"]["name"] == "lookup"
    assert json.loads(replay_calls[0]["function"]["arguments"]) == {"model": "grok-4.7"}
    assert requests[1]["messages"][-1] == {"role": "tool", "tool_call_id": "call-release", "content": "READY"}
