from __future__ import annotations

from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager
from functools import partial
import inspect
import json
from typing import Any

import httpx
from openai import AsyncOpenAI
import pytest

from kolega_code.agent.conversation import adapt_history_for_provider
from kolega_code.llm.client import LLMClient
from kolega_code.llm.models import (
    Message,
    MessageHistory,
    TextBlock,
    ThinkingBlock,
    ToolCall,
    ToolDefinition,
    ToolResult,
)
from kolega_code.llm.providers import openai as transport
from kolega_code.llm.specs import MODEL_SPECS
from kolega_code.llm.specs.thinking import reasoning_replay_field

pytestmark = pytest.mark.usefixtures("isolated_cli_env")

MODEL = "synthetic-refuseless-reasoning-route"
FAKE_KEY = "abl_refuseless_replay_hermetic_test_not_a_real_key"


@pytest.fixture(autouse=True)
def synthetic_reasoning_route(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(MODEL_SPECS, ("refuseless", MODEL), {})


def _chunk(delta: dict[str, Any], *, finish_reason: str | None = None) -> dict[str, Any]:
    return {
        "id": "chatcmpl-hermetic-replay",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": MODEL,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }


def _sse(events: list[dict[str, Any]]) -> httpx.Response:
    usage = {
        **_chunk({}),
        "choices": [],
        "usage": {"prompt_tokens": 8, "completion_tokens": 4, "total_tokens": 12},
    }
    body = "".join(f"data: {json.dumps(event)}\n\n" for event in [*events, usage]) + "data: [DONE]\n\n"
    return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body)


def _completion(message: dict[str, Any], *, finish_reason: str = "stop") -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "id": "chatcmpl-hermetic-replay",
            "object": "chat.completion",
            "created": 0,
            "model": MODEL,
            "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
            "usage": {"prompt_tokens": 8, "completion_tokens": 4, "total_tokens": 12},
        },
    )


@asynccontextmanager
async def _client(
    monkeypatch: pytest.MonkeyPatch,
    handler: Callable[[httpx.Request], httpx.Response],
) -> AsyncGenerator[LLMClient]:
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler), trust_env=False) as http_client:
        monkeypatch.setattr(transport, "AsyncOpenAI", partial(AsyncOpenAI, http_client=http_client))
        yield LLMClient(provider="refuseless", api_key=FAKE_KEY, model=MODEL, max_retries=0)


async def _invoke(client: LLMClient, method: str, history: MessageHistory) -> Message:
    messages = MessageHistory(
        adapt_history_for_provider(
            history,
            target_provider="refuseless",
            target_model=MODEL,
            supports_vision=False,
        )
    )
    tools = [
        ToolDefinition(
            name="add_numbers",
            description="Add two integers.",
            parameters=[],
            input_schema={
                "type": "object",
                "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                "required": ["a", "b"],
            },
        )
    ]
    if method == "generate":
        return await client.generate(messages, model=MODEL, tools=tools)
    stream: Any = client.stream(messages, model=MODEL, tools=tools)
    if inspect.isawaitable(stream):
        stream = await stream
    async with stream:
        async for _ in stream:
            pass
        return await stream.get_final_message()


@pytest.mark.parametrize("emitted_field", ["reasoning_content", "reasoning"])
def test_refuseless_replays_captured_reasoning_in_native_assistant_field(emitted_field: str) -> None:
    assistant = Message(
        role="assistant",
        content=[
            ThinkingBlock(thinking="check the sum"),
            TextBlock("calling"),
            ToolCall(id="call_refuseless", name="add_numbers", input={"a": 37, "b": 58}),
        ],
        usage_metadata={"provider": "refuseless", "reasoning_field": emitted_field},
    )
    history = MessageHistory(
        [
            assistant,
            Message(
                "user", [ToolResult(tool_use_id="call_refuseless", name="add_numbers", content="95", is_error=False)]
            ),
        ]
    )

    messages = history.to_openai(provider="refuseless", model=MODEL)

    assert reasoning_replay_field("refuseless", MODEL) == "reasoning_content"
    assert messages[0]["reasoning_content"] == "check the sum"
    assert "reasoning" not in messages[0]
    assert messages[0]["content"] == [{"type": "text", "text": "calling"}]
    assert messages[0]["tool_calls"][0]["id"] == "call_refuseless"
    assert messages[1]["role"] == "tool"
    assert messages[1]["tool_call_id"] == "call_refuseless"
    assert messages[1]["content"] == "95"


def test_refuseless_does_not_replay_foreign_provider_reasoning() -> None:
    assistant = Message(
        role="assistant",
        content=[ThinkingBlock(thinking="foreign reasoning"), TextBlock("answer")],
        usage_metadata={"provider": "anthropic"},
    )

    message = MessageHistory([assistant]).to_openai(provider="refuseless", model=MODEL)[0]

    assert "reasoning_content" not in message
    assert message["content"] == [
        {"type": "text", "text": "*Thinking:*\nforeign reasoning"},
        {"type": "text", "text": "answer"},
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["generate", "stream"])
@pytest.mark.parametrize("visible_text", [None, "calling"])
async def test_refuseless_sdk_tool_round_trip_replays_native_reasoning(
    monkeypatch: pytest.MonkeyPatch, method: str, visible_text: str | None
) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        assert len(requests) <= 2
        if len(requests) == 1:
            if method == "stream":
                return _sse(
                    [
                        _chunk({"role": "assistant", "reasoning_content": "check "}),
                        _chunk({"reasoning_content": "the sum"}),
                        _chunk({"content": visible_text}),
                        _chunk(
                            {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "call_refuseless",
                                        "type": "function",
                                        "function": {"name": "add_numbers", "arguments": '{"a":37,'},
                                    }
                                ]
                            }
                        ),
                        _chunk({"tool_calls": [{"index": 0, "function": {"arguments": '"b":58}'}}]}),
                        _chunk({}, finish_reason="tool_calls"),
                    ]
                )
            return _completion(
                {
                    "role": "assistant",
                    "content": visible_text,
                    "reasoning_content": "check the sum",
                    "tool_calls": [
                        {
                            "id": "call_refuseless",
                            "type": "function",
                            "function": {"name": "add_numbers", "arguments": '{"a":37,"b":58}'},
                        }
                    ],
                },
                finish_reason="tool_calls",
            )
        if method == "stream":
            return _sse([_chunk({"role": "assistant", "content": "95"}), _chunk({}, finish_reason="stop")])
        return _completion({"role": "assistant", "content": "95"})

    history = MessageHistory([Message("user", [TextBlock("Add 37 and 58 using add_numbers.")])])
    async with _client(monkeypatch, handler) as client:
        first = await _invoke(client, method, history)
        assert first.get_text_content() == (visible_text or "")
        assert [block.thinking for block in first.content if isinstance(block, ThinkingBlock)] == ["check the sum"]
        assert first.usage_metadata["provider"] == "refuseless"
        assert first.usage_metadata["reasoning_field"] == "reasoning_content"
        assert first.stop_reason == "tool_use"
        calls = [block for block in first.content if isinstance(block, ToolCall)]
        assert len(calls) == 1
        call = calls[0]
        assert call.id == "call_refuseless"
        assert call.name == "add_numbers"
        assert call.input == {"a": 37, "b": 58}
        history.extend(
            [
                first,
                Message(
                    "user",
                    [
                        ToolResult(
                            tool_use_id=call.id,
                            name=call.name,
                            content="95",
                            is_error=False,
                            execution_id=call.execution_id,
                        )
                    ],
                ),
            ]
        )
        final = await _invoke(client, method, history)

    assert final.get_text_content() == "95"
    assert final.stop_reason == "end_turn"
    assert final.usage_metadata["provider"] == "refuseless"
    assert not any(isinstance(block, (ThinkingBlock, ToolCall)) for block in final.content)
    assert len(requests) == 2
    for request in requests:
        assert request.method == "POST"
        assert str(request.url) == "https://api.refuseless.com/v1/chat/completions"
        assert request.headers["authorization"] == f"Bearer {FAKE_KEY}"
        body = json.loads(request.content)
        assert body["model"] == MODEL
        assert body.get("stream", False) is (method == "stream")
    assert json.loads(requests[0].content)["messages"] == [
        {"role": "user", "content": [{"type": "text", "text": "Add 37 and 58 using add_numbers."}]}
    ]
    second_messages = json.loads(requests[1].content)["messages"]
    assert second_messages == [
        {"role": "user", "content": [{"type": "text", "text": "Add 37 and 58 using add_numbers."}]},
        {
            "role": "assistant",
            "content": [{"type": "text", "text": visible_text}] if visible_text else "",
            "reasoning_content": "check the sum",
            "tool_calls": [
                {
                    "id": "call_refuseless",
                    "type": "function",
                    "function": {"name": "add_numbers", "arguments": '{"a": 37, "b": 58}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_refuseless", "content": "95"},
    ]
    assert "reasoning" not in second_messages[1]
    assert "check the sum" not in json.dumps(second_messages[1]["content"])
    assert "*Thinking:*" not in json.dumps(second_messages)


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["generate", "stream"])
async def test_refuseless_sdk_adaptation_drops_foreign_reasoning_but_keeps_tool_exchange(
    monkeypatch: pytest.MonkeyPatch, method: str
) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if method == "stream":
            return _sse([_chunk({"role": "assistant", "content": "95"}), _chunk({}, finish_reason="stop")])
        return _completion({"role": "assistant", "content": "95"})

    thinking = ThinkingBlock(thinking="foreign private reasoning", signature="synthetic-foreign-signature")
    call = ToolCall(id="call_foreign", name="add_numbers", input={"a": 37, "b": 58})
    assistant = Message(
        "assistant",
        [thinking, TextBlock("calling"), call],
        usage_metadata={"provider": "anthropic", "reasoning_field": "reasoning_content"},
    )
    history = MessageHistory(
        [
            Message("user", [TextBlock("Add 37 and 58 using add_numbers.")]),
            assistant,
            Message(
                "user",
                [
                    ToolResult(
                        tool_use_id=call.id,
                        name=call.name,
                        content="95",
                        is_error=False,
                        execution_id=call.execution_id,
                    )
                ],
            ),
        ]
    )
    async with _client(monkeypatch, handler) as client:
        final = await _invoke(client, method, history)

    assert final.get_text_content() == "95"
    assert final.usage_metadata["provider"] == "refuseless"
    assert len(requests) == 1
    body = json.loads(requests[0].content)
    assert body["messages"] == [
        {"role": "user", "content": [{"type": "text", "text": "Add 37 and 58 using add_numbers."}]},
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "calling"}],
            "tool_calls": [
                {
                    "id": "call_foreign",
                    "type": "function",
                    "function": {"name": "add_numbers", "arguments": '{"a": 37, "b": 58}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_foreign", "content": "95"},
    ]
    assert all("reasoning_content" not in message and "reasoning" not in message for message in body["messages"])
    assert "foreign private reasoning" not in json.dumps(body)
    assert "synthetic-foreign-signature" not in json.dumps(body)
    assert "*Thinking:*" not in json.dumps(body)
    assert history[1] is assistant
    assert assistant.content[0] is thinking
    assert thinking.thinking == "foreign private reasoning"
    assert thinking.signature == "synthetic-foreign-signature"
    assert assistant.usage_metadata == {"provider": "anthropic", "reasoning_field": "reasoning_content"}
