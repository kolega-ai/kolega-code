"""Native Mistral regressions using independent JSON/SSE fixtures, never inference."""

from __future__ import annotations

import asyncio
import json
import threading
from collections.abc import AsyncIterator
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
from typing import Any, cast
from unittest.mock import AsyncMock

import httpx
import pytest
import pytest_asyncio

from kolega_code.llm.client import LLMClient
from kolega_code.llm.exceptions import (
    LLMAuthenticationError,
    LLMBillingError,
    LLMConnectionError,
    LLMContentPolicyViolationError,
    LLMContextWindowExceededError,
    LLMError,
    LLMInternalServerError,
    LLMInvalidRequestError,
    LLMNotFoundError,
    LLMPermissionDeniedError,
    LLMRateLimitError,
    LLMTimeout,
    LLMUnprocessableEntityError,
)
from kolega_code.llm.ledger import UsageLedger
from kolega_code.llm.models import (
    ImageBlock,
    Message,
    MessageChunk,
    MessageHistory,
    TextBlock,
    ThinkingBlock,
    ToolCall,
    ToolDefinition,
    ToolParameter,
    ToolResult,
)
from kolega_code.llm.providers.base import BaseLLMProvider
from kolega_code.llm.providers.mistral import MistralProvider
from kolega_code.llm.providers.models import GenerationParams, TokenCount

MODEL = "mistral-medium-3-5"
USAGE = {"prompt_tokens": 31, "completion_tokens": 12, "total_tokens": 43}


def _history(*messages: Message) -> MessageHistory:
    return MessageHistory(list(messages) or [Message(role="user", content=[TextBlock("Hello")])])


def _tool(name: str = "read_file") -> ToolDefinition:
    return ToolDefinition(
        name=name,
        description="Read a local file.",
        parameters=[ToolParameter(name="path", type="string", description="File path", required=True)],
    )


def _completion(
    content: Any = "Hello",
    *,
    finish: str = "stop",
    calls: list[dict[str, Any]] | None = None,
    usage: dict[str, int] | None = USAGE,
) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if calls is not None:
        message["tool_calls"] = calls
    result: dict[str, Any] = {
        "id": "cmpl-native-test",
        "object": "chat.completion",
        "created": 1791331200,
        "model": MODEL,
        "choices": [{"index": 0, "message": message, "finish_reason": finish}],
    }
    if usage is not None:
        result["usage"] = usage
    return result


def _call(call_id: str, name: str, arguments: str, index: int | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }
    if index is not None:
        result["index"] = index
    return result


def _event(
    delta: dict[str, Any] | None = None,
    *,
    finish: str | None = None,
    usage: dict[str, int] | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "id": "cmpl-native-stream",
        "object": "chat.completion.chunk",
        "created": 1791331200,
        "model": MODEL,
        "choices": [] if delta is None else [{"index": 0, "delta": delta, "finish_reason": finish}],
    }
    if usage is not None:
        result["usage"] = usage
    return result


def _sse(*events: dict[str, Any], done: bool = True) -> bytes:
    # CRLF, UTF-8, and arbitrary transport boundaries exercise the real SSE decoder.
    wire = b": keep-alive\r\n\r\n"
    for event in events:
        wire += ("data: " + json.dumps(event, ensure_ascii=False) + "\r\n\r\n").encode()
    return wire + (b"data: [DONE]\r\n\r\n" if done else b"")


class FragmentedSSE(httpx.AsyncByteStream):
    """A transport body, not a mock provider or an already-parsed event stream."""

    def __init__(self, data: bytes, *, pause_after: bool = False, error: Exception | None = None) -> None:
        self.data = data
        self.pause_after = pause_after
        self.error = error
        self.closed = False
        self.waiting = asyncio.Event()
        self.release = asyncio.Event()

    async def __aiter__(self) -> AsyncIterator[bytes]:
        widths = (1, 7, 2, 19, 3, 5)
        offset = 0
        step = 0
        while offset < len(self.data):
            width = widths[step % len(widths)]
            yield self.data[offset : offset + width]
            offset += width
            step += 1
        if self.error is not None:
            raise self.error
        if self.pause_after:
            self.waiting.set()
            await self.release.wait()

    async def aclose(self) -> None:
        self.closed = True


class MistralWire:
    def __init__(self, provider: MistralProvider) -> None:
        self.provider = provider
        self.requests: list[httpx.Request] = []
        self.bodies: list[dict[str, Any]] = []
        self.responses: list[httpx.Response | Exception] = []

    async def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        self.bodies.append(json.loads(await request.aread()))
        assert self.responses, "Unexpected HTTP request (including an unexpected retry)"
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    def completion(self, **kwargs: Any) -> None:
        self.responses.append(httpx.Response(200, json=_completion(**kwargs)))

    def stream(self, data: bytes, **kwargs: Any) -> FragmentedSSE:
        body = FragmentedSSE(data, **kwargs)
        self.responses.append(httpx.Response(200, headers={"Content-Type": "text/event-stream"}, stream=body))
        return body


@pytest_asyncio.fixture
async def wire(request: pytest.FixtureRequest) -> AsyncIterator[MistralWire]:
    provider = MistralProvider(api_key="mistral-unit-test-not-a-secret", max_retries=getattr(request, "param", 0))
    assert isinstance(provider, BaseLLMProvider)
    assert isinstance(provider.async_client, httpx.AsyncClient)
    original = provider.async_client
    # Preserve constructor configuration so URL/auth assertions also verify production defaults.
    result = MistralWire(provider)
    provider.async_client = httpx.AsyncClient(
        base_url=original.base_url,
        headers=original.headers,
        transport=httpx.MockTransport(result.handle),
    )
    await original.aclose()
    try:
        yield result
    finally:
        await provider.async_client.aclose()


async def _consume(provider: MistralProvider, **kwargs: Any) -> tuple[list[MessageChunk], Message]:
    wrapper = await provider.stream(_history(), model=MODEL, **kwargs)
    async with wrapper as stream:
        chunks = [chunk async for chunk in stream]
        message = await stream.get_final_message()
    assert all(isinstance(chunk, MessageChunk) for chunk in chunks)
    return chunks, message


def _assert_usage(message: Message) -> None:
    assert message.usage_metadata["provider"] == "mistral"
    assert message.usage_metadata["prompt_tokens"] == 31
    assert message.usage_metadata["completion_tokens"] == 12
    assert message.usage is not None
    assert message.usage.reported
    assert message.usage.provider == "mistral"
    assert (message.usage.input_tokens, message.usage.output_tokens, message.usage.total_tokens) == (31, 12, 43)
    assert message.usage.cache_read_input_tokens is None
    assert message.usage.cache_write_input_tokens is None
    assert message.usage.reasoning_output_tokens is None


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_native_request_schema_auth_and_parallel_tools(wire: MistralWire, streaming: bool) -> None:
    params = GenerationParams(
        temperature=0.25,
        max_completion_tokens=2048,
        thinking="high",
        tools=[_tool(), _tool("stat_file")],
    )
    system = Message(role="system", content=[TextBlock("Be brief", cache_checkpoint=True)])
    if streaming:
        body = wire.stream(_sse(_event({"content": "ok"}, finish="stop"), _event(usage=USAGE)))
        await _consume(wire.provider, params=params, system=system, parallel_tool_calls=True)
        assert body.closed
    else:
        wire.completion()
        await wire.provider.generate(_history(), system=system, params=params, model=MODEL, parallel_tool_calls=True)
    request = wire.requests[0]
    payload = wire.bodies[0]
    assert request.method == "POST"
    assert str(request.url) == "https://api.mistral.ai/v1/chat/completions"
    assert request.headers["authorization"] == "Bearer mistral-unit-test-not-a-secret"
    assert request.headers["content-type"].startswith("application/json")
    assert payload["model"] == MODEL
    assert payload["stream"] is streaming
    assert payload["max_tokens"] == 2048
    assert payload["temperature"] == 0.25
    assert payload["reasoning_effort"] == "high"
    assert payload["parallel_tool_calls"] is True
    assert payload["messages"][0]["role"] == "system"
    assert "Be brief" in json.dumps(payload["messages"][0])
    assert [tool["function"]["name"] for tool in payload["tools"]] == ["read_file", "stat_file"]
    assert payload["tools"][0]["function"]["parameters"]["required"] == ["path"]
    for field in (
        "stream_options",
        "max_completion_tokens",
        "max_output_tokens",
        "prompt_cache_retention",
        "cache_control",
        "cache_ttl",
        "reasoning_content",
    ):
        assert f'"{field}"' not in json.dumps(payload)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "content, expected", [("answer", "answer"), ([{"type": "text", "text": "answer"}], "answer"), (None, "")]
)
async def test_generation_string_chunk_list_and_null_content(wire: MistralWire, content: Any, expected: str) -> None:
    wire.completion(content=content)
    message = await wire.provider.generate(_history(), model=MODEL)
    assert isinstance(message, Message)
    assert message.role == "assistant"
    assert message.get_text_content() == expected
    assert message.stop_reason == "end_turn"
    _assert_usage(message)


@pytest.mark.asyncio
async def test_nested_thinking_signature_and_resumed_native_replay(wire: MistralWire) -> None:
    native = [
        {
            "type": "thinking",
            "thinking": [{"type": "text", "text": "First "}, {"type": "text", "text": "reason."}],
            "signature": "opaque-native-signature",
            "closed": True,
        },
        {"type": "text", "text": "The answer."},
    ]
    wire.completion(content=native)
    message = await wire.provider.generate(_history(), model=MODEL)
    assert isinstance(message.content, list)
    assert [block.type for block in message.content] == ["thinking", "text"]
    thinking = message.content[0]
    assert isinstance(thinking, ThinkingBlock)
    assert thinking.thinking == "First reason."
    assert thinking.signature == "opaque-native-signature"
    assert message.get_text_content() == "The answer."
    resumed = Message.from_dict(json.loads(json.dumps(message.to_dict())))
    wire.completion(content="Next answer")
    await wire.provider.generate(
        _history(Message(role="user", content="First question"), resumed, Message(role="user", content="Continue")),
        model="mistral-small-2603",
    )
    assistant = next(item for item in wire.bodies[-1]["messages"] if item["role"] == "assistant")
    assert isinstance(assistant["content"], list)
    replay = next(item for item in assistant["content"] if item["type"] == "thinking")
    assert "".join(chunk["text"] for chunk in replay["thinking"]) == "First reason."
    assert replay["signature"] == "opaque-native-signature"
    assert replay["closed"] is True
    assert "*Thinking:*" not in json.dumps(assistant)
    assert "reasoning_content" not in assistant


@pytest.mark.asyncio
async def test_foreign_reasoning_is_dropped_not_rendered_as_visible_text(wire: MistralWire) -> None:
    foreign = Message(
        role="assistant",
        content=[ThinkingBlock("private foreign reasoning", signature="foreign-signature"), TextBlock("Public answer")],
        usage_metadata={"provider": "anthropic"},
    )
    wire.completion()
    await wire.provider.generate(_history(foreign, Message(role="user", content="Continue")), model=MODEL)
    serialized = json.dumps(wire.bodies[0])
    assert "Public answer" in serialized
    assert "private foreign reasoning" not in serialized
    assert "foreign-signature" not in serialized
    assert "*Thinking:*" not in serialized


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "finish, expected",
    [
        ("stop", "end_turn"),
        ("tool_calls", "tool_use"),
        ("length", "max_tokens"),
        ("model_length", "max_tokens"),
    ],
)
async def test_finish_reasons_are_terminal(wire: MistralWire, finish: str, expected: str) -> None:
    wire.completion(finish=finish)
    generated = await wire.provider.generate(_history(), model=MODEL)
    assert generated.stop_reason == expected
    body = wire.stream(_sse(_event({"content": "answer"}, finish=finish)))
    _, streamed = await _consume(wire.provider)
    assert streamed.stop_reason == expected
    assert body.closed


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "finish, expected",
    [("error", LLMInternalServerError), ("content_filter", LLMContentPolicyViolationError)],
)
@pytest.mark.parametrize("streaming", [False, True])
async def test_unsuccessful_finish_reasons_raise(
    wire: MistralWire, finish: str, expected: type[LLMError], streaming: bool
) -> None:
    body = None
    if streaming:
        body = wire.stream(_sse(_event({"content": "partial"}, finish=finish)))
    else:
        wire.completion(finish=finish)
    with pytest.raises(expected) as caught:
        if streaming:
            await _consume(wire.provider)
        else:
            await wire.provider.generate(_history(), model=MODEL)
    assert caught.value.provider == "mistral"
    if body is not None:
        assert body.closed


@pytest.mark.asyncio
async def test_fragmented_sse_mixed_transition_and_usage_only_event(wire: MistralWire) -> None:
    body = wire.stream(
        _sse(
            _event({"role": "assistant", "content": None}),
            _event({"content": [{"type": "thinking", "thinking": [{"type": "text", "text": "Consider "}]}]}),
            _event(
                {
                    "content": [
                        {
                            "type": "thinking",
                            "thinking": [{"type": "text", "text": "café."}],
                            "signature": "stream-signature",
                            "closed": True,
                        },
                        {"type": "text", "text": "Voilà"},
                    ]
                }
            ),
            _event({"content": "!"}),
            _event({}, finish="stop"),
            _event(usage=USAGE),
        )
    )
    chunks, message = await _consume(wire.provider)
    assert "".join(chunk.thinking or "" for chunk in chunks) == "Consider café."
    assert "".join(chunk.text or "" for chunk in chunks if chunk.type == "text") == "Voilà!"
    assert message.get_text_content() == "Voilà!"
    assert isinstance(message.content, list)
    thinking = [block for block in message.content if isinstance(block, ThinkingBlock)]
    assert "".join(block.thinking for block in thinking) == "Consider café."
    assert any(block.signature == "stream-signature" for block in thinking)
    assert message.stop_reason == "end_turn"
    _assert_usage(message)
    assert body.closed


@pytest.mark.asyncio
async def test_parallel_fragmented_tool_calls_preserve_ids_and_execution_identity(wire: MistralWire) -> None:
    body = wire.stream(
        _sse(
            _event(
                {
                    "tool_calls": [
                        _call("provider-call-", "read_", '{"path":"', 0),
                        _call("another-", "stat_", '{"path":"b', 1),
                    ]
                }
            ),
            _event(
                {
                    "tool_calls": [
                        {"index": 1, "id": "long-id", "function": {"name": "file", "arguments": '.py"}'}},
                        {"index": 0, "id": "long-id", "function": {"name": "file", "arguments": 'a.py"}'}},
                    ]
                }
            ),
            _event({}, finish="tool_calls"),
            _event(usage=USAGE),
        )
    )
    chunks, final = await _consume(wire.provider, params=GenerationParams(tools=[_tool(), _tool("stat_file")]))
    assert [(call.id, call.name, call.input) for call in final.tool_calls] == [
        ("provider-call-long-id", "read_file", {"path": "a.py"}),
        ("another-long-id", "stat_file", {"path": "b.py"}),
    ]
    assert final.stop_reason == "tool_use"
    assert len({call.execution_id for call in final.tool_calls}) == 2
    assert all(call.execution_id and call.execution_id != call.id for call in final.tool_calls)
    emitted = [chunk.tool_call for chunk in chunks if chunk.tool_call is not None]
    assert {call.execution_id for call in emitted} <= {call.execution_id for call in final.tool_calls}
    assert all(call.input in ({"path": "a.py"}, {"path": "b.py"}) for call in emitted)
    assert body.closed


@pytest.mark.asyncio
async def test_tool_cycle_resume_and_repeated_wire_ids_are_response_scoped(wire: MistralWire) -> None:
    history = _history()
    seen_execution_ids: list[str] = []
    for path in ("a.py", "b.py"):
        wire.completion(
            content=None, finish="tool_calls", calls=[_call("reused-wire-id", "read_file", json.dumps({"path": path}))]
        )
        message = await wire.provider.generate(history, model=MODEL, params=GenerationParams(tools=[_tool()]))
        call = message.tool_calls[0]
        assert call.id == "reused-wire-id"
        assert call.input == {"path": path}
        seen_execution_ids.append(call.execution_id)
        resumed = Message.from_dict(json.loads(json.dumps(message.to_dict())))
        assert resumed.tool_calls[0].execution_id == call.execution_id
        history.append(resumed)
        history.append(
            Message(
                role="user",
                content=[
                    ToolResult(
                        tool_use_id=call.id,
                        execution_id=call.execution_id,
                        name=call.name,
                        content=f"contents of {path}",
                        is_error=False,
                    )
                ],
            )
        )
    assert len(set(seen_execution_ids)) == 2
    wire.completion(content="Both files read.")
    final = await wire.provider.generate(history, model=MODEL)
    assert final.get_text_content() == "Both files read."
    tool_messages = [item for item in wire.bodies[-1]["messages"] if item["role"] == "tool"]
    assert [item["tool_call_id"] for item in tool_messages] == ["reused-wire-id", "reused-wire-id"]
    assert "contents of a.py" in json.dumps(tool_messages[0])
    assert "contents of b.py" in json.dumps(tool_messages[1])
    assert "execution_id" not in json.dumps(wire.bodies[-1])


@pytest.mark.asyncio
async def test_out_of_order_parallel_results_and_orphan_repair(wire: MistralWire) -> None:
    calls = [ToolCall(id=ident, name="read_file", input={"path": ident}) for ident in ("one", "two")]
    assistant = Message(role="assistant", content=[*calls], tool_calls=calls, usage_metadata={"provider": "mistral"})
    results = Message(
        role="user",
        content=[
            ToolResult(tool_use_id="two", name="read_file", content="second", is_error=False),
            ToolResult(tool_use_id="one", name="read_file", content="first", is_error=False),
            ToolResult(tool_use_id="orphan", name="read_file", content="lost call result", is_error=False),
            TextBlock("Continue"),
        ],
    )
    wire.completion()
    await wire.provider.generate(_history(assistant, results), model=MODEL)
    messages = wire.bodies[0]["messages"]
    pending: set[str] = set()
    for item in messages:
        if item["role"] == "assistant":
            assert not pending, "Do not place a new assistant turn before pending tool results"
            pending.update(call["id"] for call in item.get("tool_calls", []))
        elif item["role"] == "tool":
            assert item["tool_call_id"] in pending, "Every tool result needs a preceding matching call"
            pending.remove(item["tool_call_id"])
        else:
            assert not pending, "Tool results must precede subsequent user text"
    assert not pending
    serialized = json.dumps(messages)
    assert "first" in serialized and "second" in serialized and "lost call result" in serialized
    assert "Continue" in serialized


@pytest.mark.asyncio
async def test_url_base64_images_and_image_bearing_tool_results(wire: MistralWire) -> None:
    url = ImageBlock(image_type="url", media_type="image/png", data="https://example.invalid/image.png")
    encoded = ImageBlock(image_type="base64", media_type="image/png", data="aW1hZ2U=")
    call = ToolCall(id="image-call", name="read_image", input={"path": "plot.png"})
    history = _history(
        Message(role="user", content=[TextBlock("Compare"), url, encoded]),
        Message(role="assistant", content=[call], tool_calls=[call], usage_metadata={"provider": "mistral"}),
        Message(
            role="user",
            content=[
                ToolResult(
                    tool_use_id=call.id,
                    name=call.name,
                    content=[TextBlock("Plot"), encoded],
                    is_error=False,
                    execution_id=call.execution_id,
                )
            ],
        ),
    )
    wire.completion()
    await wire.provider.generate(history, model=MODEL)
    messages = wire.bodies[0]["messages"]
    serialized = json.dumps(messages)
    assert "https://example.invalid/image.png" in serialized
    assert serialized.count("data:image/png;base64,aW1hZ2U=") == 2
    assert "image_url" in serialized
    assert any(item["role"] == "tool" and item["tool_call_id"] == call.id for item in messages)
    assert "Plot" in serialized


@pytest.mark.asyncio
async def test_freeform_tools_remain_json_wrapped(wire: MistralWire) -> None:
    definition = ToolDefinition(name="apply_patch", description="Apply patch", parameters=[], input_kind="freeform")
    raw = "*** Begin Patch\n*** Delete File: old.txt\n*** End Patch\n"
    call = ToolCall(id="patch-call", name="apply_patch", input=raw, input_kind="freeform")
    history = _history(
        Message(role="assistant", content=[call], tool_calls=[call], usage_metadata={"provider": "mistral"}),
        Message(
            role="user",
            content=[
                ToolResult(
                    tool_use_id=call.id, name=call.name, content="Success", is_error=False, input_kind="freeform"
                )
            ],
        ),
    )
    wire.completion()
    await wire.provider.generate(history, model=MODEL, params=GenerationParams(tools=[definition]))
    payload = wire.bodies[0]
    tool = payload["tools"][0]
    assert tool["type"] == "function"
    assert tool["function"]["parameters"]["properties"]["input"]["type"] == "string"
    assert tool["function"]["parameters"]["required"] == ["input"]
    arguments = next(item for item in payload["messages"] if item["role"] == "assistant")["tool_calls"][0]["function"][
        "arguments"
    ]
    assert json.loads(arguments) == {"input": raw}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status, detail, expected",
    [
        (401, "Invalid API key", LLMAuthenticationError),
        (403, "Model access forbidden", LLMPermissionDeniedError),
        (402, "Payment required", LLMBillingError),
        (400, "Insufficient credits for this request", LLMBillingError),
        (403, "Insufficient balance", LLMBillingError),
        (429, "Insufficient credits", LLMBillingError),
        (429, "Rate limit exceeded", LLMRateLimitError),
        (400, "Prompt plus max_tokens exceeds context length", LLMContextWindowExceededError),
        (400, "Invalid temperature", LLMInvalidRequestError),
        (404, "Model not found", LLMNotFoundError),
        (422, "Validation failed", LLMUnprocessableEntityError),
        (500, "Internal server error", LLMInternalServerError),
        (503, "Backend unavailable", LLMInternalServerError),
    ],
)
@pytest.mark.parametrize("streaming", [False, True])
async def test_status_mapping_without_implicit_retries(
    wire: MistralWire, status: int, detail: str, expected: type[LLMError], streaming: bool
) -> None:
    wire.responses.append(
        httpx.Response(status, json={"message": detail, "object": "error", "type": "invalid_request_error"})
    )
    with pytest.raises(expected) as caught:
        if streaming:
            await _consume(wire.provider)
        else:
            await wire.provider.generate(_history(), model=MODEL)
    assert caught.value.provider == "mistral"
    assert len(wire.requests) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_http_timeout_is_normalized(wire: MistralWire, streaming: bool) -> None:
    wire.responses.append(httpx.ReadTimeout("native test timeout"))
    with pytest.raises(LLMTimeout) as caught:
        if streaming:
            await _consume(wire.provider)
        else:
            await wire.provider.generate(_history(), model=MODEL)
    assert caught.value.provider == "mistral"
    assert len(wire.requests) == 1


@pytest.fixture
def controlled_retry_sleep(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    delays: list[float] = []

    async def sleep(delay: float, result: Any = None) -> Any:
        delays.append(delay)
        return result

    monkeypatch.setattr(asyncio, "sleep", sleep)
    return delays


@pytest.mark.asyncio
@pytest.mark.parametrize("wire", [1], indirect=True)
async def test_rate_limit_retries_once_honoring_retry_after(
    wire: MistralWire, controlled_retry_sleep: list[float]
) -> None:
    wire.responses.append(httpx.Response(429, headers={"Retry-After": "2"}, json={"message": "Rate limit exceeded"}))
    wire.completion(content="Recovered")
    message = await wire.provider.generate(_history(), model=MODEL)
    assert message.get_text_content() == "Recovered"
    assert len(wire.requests) == 2
    assert 2 in controlled_retry_sleep
    assert wire.bodies[0] == wire.bodies[1]


@pytest.mark.asyncio
@pytest.mark.parametrize("wire", [1], indirect=True)
async def test_retry_budget_is_bounded(wire: MistralWire, controlled_retry_sleep: list[float]) -> None:
    for _ in range(2):
        wire.responses.append(httpx.Response(500, json={"message": "Backend unavailable"}))
    with pytest.raises(LLMInternalServerError) as caught:
        await wire.provider.generate(_history(), model=MODEL)
    assert caught.value.provider == "mistral"
    assert len(wire.requests) == 2
    assert len(controlled_retry_sleep) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("wire", [2], indirect=True)
@pytest.mark.parametrize("status, expected", [(401, LLMAuthenticationError), (400, LLMInvalidRequestError)])
async def test_permanent_errors_are_not_retried(
    wire: MistralWire, controlled_retry_sleep: list[float], status: int, expected: type[LLMError]
) -> None:
    wire.responses.append(httpx.Response(status, json={"message": "Invalid request"}))
    with pytest.raises(expected) as caught:
        await wire.provider.generate(_history(), model=MODEL)
    assert caught.value.provider == "mistral"
    assert len(wire.requests) == 1
    assert not controlled_retry_sleep


@pytest.mark.asyncio
async def test_malformed_generation_json_is_normalized(wire: MistralWire) -> None:
    wire.responses.append(httpx.Response(200, content=b"{not JSON", headers={"Content-Type": "application/json"}))
    with pytest.raises(LLMError) as caught:
        await wire.provider.generate(_history(), model=MODEL)
    assert caught.value.provider == "mistral"
    assert len(wire.requests) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "data",
    [
        _sse(_event({"content": "partial"}), done=False),
        b"data: {broken JSON}\n\n",
        _sse({"error": {"message": "stream backend failed", "code": 500}}),
    ],
    ids=["premature-eof", "bad-json", "error-event"],
)
async def test_invalid_stream_is_an_error_and_closes_response(wire: MistralWire, data: bytes) -> None:
    body = wire.stream(data)
    with pytest.raises(LLMError) as caught:
        await _consume(wire.provider)
    assert caught.value.provider == "mistral"
    assert body.closed
    assert len(wire.requests) == 1


@pytest.mark.asyncio
async def test_partial_stream_transport_failure_does_not_retry(wire: MistralWire) -> None:
    # Even a provider configured to retry requests must never duplicate a consumed stream.
    wire.provider.max_retries = 3
    body = wire.stream(
        _sse(_event({"content": "already delivered"}), done=False),
        error=httpx.ReadTimeout("failed after output"),
    )
    with pytest.raises(LLMTimeout) as caught:
        await _consume(wire.provider)
    assert caught.value.provider == "mistral"
    assert body.closed
    assert len(wire.requests) == 1


@pytest.mark.asyncio
async def test_cancellation_closes_inflight_response(wire: MistralWire) -> None:
    body = wire.stream(_sse(_event({"content": "partial"}), done=False), pause_after=True)
    task = asyncio.create_task(_consume(wire.provider))
    try:
        await asyncio.wait_for(body.waiting.wait(), timeout=2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert body.closed
        assert len(wire.requests) == 1
    finally:
        body.release.set()
        if not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass


@pytest.mark.asyncio
async def test_early_context_exit_closes_response(wire: MistralWire) -> None:
    body = wire.stream(_sse(_event({"content": "first"}), _event({"content": "second"}, finish="stop")))
    wrapper = await wire.provider.stream(_history(), model=MODEL)
    async with wrapper as stream:
        async for chunk in stream:
            if chunk.type == "text":
                break
    assert body.closed


@pytest.mark.asyncio
async def test_missing_usage_is_not_fabricated(wire: MistralWire) -> None:
    wire.completion(usage=None)
    message = await wire.provider.generate(_history(), model=MODEL)
    assert message.usage is not None
    assert not message.usage.reported
    assert message.usage.input_tokens is None
    assert message.usage.output_tokens is None
    assert message.usage.total_tokens is None


@pytest.mark.asyncio
async def test_conservative_local_count_includes_system_tools_and_unicode(wire: MistralWire) -> None:
    short = await wire.provider.count_tokens(_history(), model=MODEL)
    long = await wire.provider.count_tokens(
        _history(Message(role="user", content="中文🙂" * 1000)),
        system=Message(role="system", content="Instructions " * 100),
        tools=[_tool()],
        model=MODEL,
    )
    assert isinstance(short, TokenCount) and isinstance(long, TokenCount)
    assert type(short.input_tokens) is int and short.input_tokens > 0
    assert long.input_tokens > short.input_tokens
    assert long.input_tokens >= 1000, "Unicode must not be undercounted by an ASCII-only character heuristic"
    assert not wire.requests, "Native Mistral has no remote token-count endpoint"
    assert callable(wire.provider.retry_decorator)


@pytest.mark.asyncio
async def test_custom_base_url_and_provider_name_are_respected() -> None:
    provider = MistralProvider(
        api_key="test-only",
        max_retries=0,
        requests_per_minute=None,
        tokens_per_minute=None,
        base_url="https://native-mistral.example.invalid/v1",
        provider_name="mistral",
    )
    original = provider.async_client
    wire = MistralWire(provider)
    provider.async_client = httpx.AsyncClient(
        base_url=original.base_url, headers=original.headers, transport=httpx.MockTransport(wire.handle)
    )
    await original.aclose()
    try:
        wire.completion()
        await provider.generate(_history(), model=MODEL)
        assert str(wire.requests[0].url) == "https://native-mistral.example.invalid/v1/chat/completions"
        assert provider.provider_name == "mistral"
    finally:
        await provider.async_client.aclose()


@pytest.mark.asyncio
async def test_sse_multiline_data_bom_comments_and_cr_only(wire: MistralWire) -> None:
    event = _event({"content": "你好 café 🙂"}, finish="stop")
    # The JSON object spans multiple SSE data fields, not multiple JSON values.
    encoded = json.dumps(event, ensure_ascii=False, indent=2)
    data = b"\xef\xbb\xbf: comment\r\r"
    data += (
        "event: message\rid: event-1\rretry: 1000\r" + "".join(f"data: {line}\r" for line in encoded.splitlines())
    ).encode()
    body = wire.stream(data + b"\rdata: [DONE]\r\r")
    chunks, final = await _consume(wire.provider)
    assert "".join(chunk.text or "" for chunk in chunks) == "你好 café 🙂"
    assert final.get_text_content() == "你好 café 🙂"
    assert body.closed


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "data",
    [
        b"data: \xff\n\n",
        b"not-sse\n\n",
        b"event: unexpected\ndata: {}\n\n",
        _sse(_event({"content": [{"type": "audio", "audio": "unsupported"}]}, finish="stop")),
        _sse(_event({"content": [{"type": "thinking", "thinking": [{"type": "image_url"}]}]}, finish="stop")),
        _sse(_event({"reasoning_content": "wrong wire field"}, finish="stop")),
        _sse(_event({"content": "answer"})),
        _sse(_event({"content": "answer"}, finish="unexpected")),
        b"data: [DONE]",
        b"data: [DONE]\n\n",
    ],
    ids=[
        "invalid-utf8",
        "invalid-sse",
        "unknown-event",
        "unknown-content",
        "unknown-nested-thinking",
        "foreign-delta-field",
        "missing-finish",
        "unknown-finish",
        "truncated-done",
        "premature-done",
    ],
)
async def test_strict_stream_parser_never_silently_drops_unknown_data(wire: MistralWire, data: bytes) -> None:
    body = wire.stream(data)
    with pytest.raises(LLMError) as caught:
        await _consume(wire.provider)
    assert caught.value.provider == "mistral"
    assert body.closed


@pytest.mark.asyncio
async def test_get_final_message_drains_and_remains_available_after_exit(wire: MistralWire) -> None:
    body = wire.stream(_sse(_event({"content": "complete"}, finish="stop"), _event(usage=USAGE)))
    wrapper = await wire.provider.stream(_history(), model=MODEL)
    async with wrapper:
        message = await wrapper.get_final_message()
    assert message is await wrapper.get_final_message()
    assert message.get_text_content() == "complete"
    _assert_usage(message)
    assert body.closed


@pytest.mark.asyncio
async def test_final_message_is_unavailable_after_early_exit(wire: MistralWire) -> None:
    body = wire.stream(_sse(_event({"content": "partial"}), _event({}, finish="stop")))
    wrapper = await wire.provider.stream(_history(), model=MODEL)
    async with wrapper:
        assert (await anext(wrapper)).text == "partial"
    with pytest.raises(LLMError):
        await wrapper.get_final_message()
    assert body.closed


@pytest.mark.asyncio
async def test_stream_waits_for_done_and_trailing_usage(wire: MistralWire) -> None:
    body = wire.stream(_sse(_event({"content": "answer"}, finish="stop")), pause_after=True)
    # Retain the finish event, but wait before sending DONE. A finish reason alone
    # must not finalize usage or produce a premature successful message.
    body.data = body.data.removesuffix(b"data: [DONE]\r\n\r\n")
    task = asyncio.create_task(_consume(wire.provider))
    try:
        await asyncio.wait_for(body.waiting.wait(), 2)
        assert not task.done()
        body.release.set()
        with pytest.raises(LLMError):
            await task
        assert body.closed
    finally:
        body.release.set()
        if not task.done():
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task


@pytest.mark.asyncio
@pytest.mark.parametrize("signature", [None, "", "signed"])
async def test_stream_exact_thinking_metadata_replay(wire: MistralWire, signature: str | None) -> None:
    native = {
        "type": "thinking",
        "thinking": [{"type": "text", "text": "A"}, {"type": "text", "text": "B"}],
        "signature": signature,
        "closed": False,
    }
    body = wire.stream(
        _sse(
            _event({"content": [native]}),
            _event({"content": [{"type": "thinking", "thinking": [], "closed": True}, {"type": "text", "text": "ok"}]}),
            _event({}, finish="stop"),
        )
    )
    _, message = await _consume(wire.provider)
    restored = Message.from_dict(json.loads(json.dumps(message.to_dict())))
    direct = restored.to_openai(provider="mistral", model=MODEL)
    assert direct["content"][0] == native
    wire.completion()
    await wire.provider.generate(_history(restored, Message("user", "Continue")), model=MODEL)
    assistant = next(item for item in wire.bodies[-1]["messages"] if item["role"] == "assistant")
    assert assistant["content"] == [
        native,
        {"type": "thinking", "thinking": [], "closed": True},
        {"type": "text", "text": "ok"},
    ]
    assert body.closed


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_freeform_response_is_unwrapped(wire: MistralWire, streaming: bool) -> None:
    tool = ToolDefinition(name="apply_patch", description="Apply a patch", parameters=[], input_kind="freeform")
    raw = "*** Begin Patch\n*** End Patch\n"
    arguments = json.dumps({"input": raw})
    params = GenerationParams(tools=[tool])
    if streaming:
        wire.stream(_sse(_event({"tool_calls": [_call("patch-id", "apply_patch", arguments, 0)]}, finish="tool_calls")))
        _, message = await _consume(wire.provider, params=params)
    else:
        wire.completion(content=None, finish="tool_calls", calls=[_call("patch-id", "apply_patch", arguments)])
        message = await wire.provider.generate(_history(), model=MODEL, params=params)
    assert message.tool_calls[0].input_kind == "freeform"
    assert message.tool_calls[0].input == raw


@pytest.mark.asyncio
async def test_direct_images_rejected_on_known_text_only_model(wire: MistralWire) -> None:
    history = _history(Message("user", [ImageBlock(image_type="base64", media_type="image/png", data="aW1hZ2U=")]))
    with pytest.raises(LLMInvalidRequestError, match="does not support vision"):
        await wire.provider.generate(history, model="codestral-2508")
    assert not wire.requests


@pytest.mark.asyncio
async def test_tool_result_images_adapted_for_known_text_only_model(wire: MistralWire) -> None:
    call = ToolCall(id="read-image", name="read_file", input={"path": "plot.png"})
    history = _history(
        Message("assistant", [call]),
        Message(
            "user",
            [
                ToolResult(
                    tool_use_id=call.id,
                    name=call.name,
                    execution_id=call.execution_id,
                    is_error=False,
                    content=[
                        TextBlock("Retain this text"),
                        ImageBlock(image_type="base64", media_type="image/png", data="aW1hZ2U="),
                    ],
                )
            ],
        ),
    )
    wire.completion()
    await wire.provider.generate(history, model="codestral-2508")
    serialized = json.dumps(wire.bodies[-1])
    assert "Retain this text" in serialized
    assert "image_url" not in serialized
    assert "aW1hZ2U=" not in serialized


@pytest.mark.asyncio
async def test_local_count_image_budget_does_not_tokenize_base64(wire: MistralWire) -> None:
    def history(data: str) -> MessageHistory:
        return _history(Message("user", [ImageBlock(image_type="base64", media_type="image/png", data=data)]))

    tiny = await wire.provider.count_tokens(history("abcd"), model=MODEL)
    large = await wire.provider.count_tokens(history("abcd" * 100_000), model=MODEL)
    text = await wire.provider.count_tokens(_history(), model=MODEL)
    assert tiny.input_tokens == large.input_tokens
    assert tiny.input_tokens >= text.input_tokens + 4000
    assert not wire.requests


@pytest.mark.asyncio
async def test_serialization_and_counting_are_offloaded(wire: MistralWire, monkeypatch: pytest.MonkeyPatch) -> None:
    main_thread = threading.get_ident()
    history = _history()
    original = history.to_openai
    seen: list[int] = []

    def serialized(*args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        seen.append(threading.get_ident())
        return original(*args, **kwargs)

    monkeypatch.setattr(history, "to_openai", serialized)
    await wire.provider.count_tokens(history, model=MODEL)
    wire.completion()
    await wire.provider.generate(history, model=MODEL)
    assert len(seen) == 2 and all(ident != main_thread for ident in seen)


@pytest.mark.asyncio
async def test_native_defaults_explicit_output_and_cache_override(wire: MistralWire) -> None:
    for kwargs in ({}, {"prompt_cache_key": "explicit-conversation", "max_tokens": 50000}, {}):
        wire.completion()
        await wire.provider.generate(_history(), model=MODEL, **kwargs)
    assert wire.bodies[0]["max_tokens"] == 32768
    assert wire.bodies[1]["max_tokens"] == 50000, "Reserved catalog output is not a hard native ceiling"
    assert wire.bodies[1]["prompt_cache_key"] == "explicit-conversation"
    assert wire.bodies[0]["prompt_cache_key"] == wire.bodies[2]["prompt_cache_key"]


@pytest.mark.asyncio
@pytest.mark.parametrize("wire", [1], indirect=True)
async def test_retry_after_http_date_and_stream_establishment_retry(
    wire: MistralWire, controlled_retry_sleep: list[float]
) -> None:
    retry_at = datetime.now(timezone.utc) + timedelta(seconds=10)
    wire.responses.append(
        httpx.Response(503, headers={"Retry-After": format_datetime(retry_at, usegmt=True)}, json={"message": "busy"})
    )
    body = wire.stream(_sse(_event({"content": "recovered"}, finish="stop")))
    _, message = await _consume(wire.provider)
    assert message.get_text_content() == "recovered"
    assert len(wire.requests) == 2
    assert len(controlled_retry_sleep) == 1 and 8 <= controlled_retry_sleep[0] <= 10
    assert body.closed


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_transport_connection_error_is_provider_labeled(wire: MistralWire, streaming: bool) -> None:
    wire.responses.append(httpx.ConnectError("test-only connection failure"))
    with pytest.raises(LLMConnectionError) as caught:
        if streaming:
            await _consume(wire.provider)
        else:
            await wire.provider.generate(_history(), model=MODEL)
    assert caught.value.provider == "mistral"
    assert len(wire.requests) == 1


@pytest.mark.asyncio
async def test_rate_limiter_and_streaming_timeout_contract(wire: MistralWire) -> None:
    wire.provider.rate_limiter.acquire = AsyncMock()
    wire.completion()
    await wire.provider.generate(_history(), model=MODEL)
    wire.stream(_sse(_event({"content": "ok"}, finish="stop")))
    await _consume(wire.provider)
    assert wire.provider.rate_limiter.acquire.await_count == 2
    timeout = wire.requests[-1].extensions["timeout"]
    assert timeout == {"connect": 10.0, "read": 300.0, "write": 30.0, "pool": 10.0}


@pytest.mark.asyncio
async def test_reported_usage_subtotals_only(wire: MistralWire) -> None:
    wire.completion(
        usage={
            **USAGE,
            "prompt_tokens_details": {"cached_tokens": 7},
            "completion_tokens_details": {"reasoning_tokens": 5},
        }
    )
    message = await wire.provider.generate(_history(), model=MODEL)
    assert message.usage is not None
    assert message.usage.cache_read_input_tokens == 7
    assert message.usage.reasoning_output_tokens == 5
    assert message.usage.cache_write_input_tokens is None


@pytest.mark.asyncio
async def test_client_stream_ledger_settlement_after_context_exit(wire: MistralWire) -> None:
    ledger = UsageLedger()
    client = LLMClient(provider="mistral", api_key="unit-test-only", max_retries=0, usage_ledger=ledger)
    original = client.provider
    assert isinstance(original, MistralProvider)
    await original.async_client.aclose()
    client.provider = wire.provider
    for text, usage in (("reported", USAGE), ("unreported", None)):
        events = [_event({"content": text}, finish="stop")]
        if usage is not None:
            events.append(_event(usage=usage))
        body = wire.stream(_sse(*events))
        wrapper = await cast(Any, client.stream(_history(), model=MODEL))
        async with wrapper as stream:
            chunks = [chunk async for chunk in stream]
        assert "".join(chunk.text or "" for chunk in chunks) == text
        message = await wrapper.get_final_message()
        assert message.usage.reported is (usage is not None)
        assert body.closed
    wire.responses.append(httpx.Response(500, json={"message": "backend down"}))
    with pytest.raises(LLMInternalServerError):
        await cast(Any, client.stream(_history(), model=MODEL))
    snapshot = ledger.snapshot()
    assert (snapshot.requests, snapshot.responses, snapshot.reported, snapshot.unreported, snapshot.failed) == (
        3,
        2,
        1,
        1,
        1,
    )
    assert snapshot.total_tokens == 43


@pytest.mark.asyncio
async def test_unknown_nonstream_content_is_an_explicit_error(wire: MistralWire) -> None:
    wire.completion(content=[{"type": "reference", "reference_ids": [1]}])
    with pytest.raises(LLMError, match="unsupported content chunk") as caught:
        await wire.provider.generate(_history(), model=MODEL)
    assert caught.value.provider == "mistral"


@pytest.mark.asyncio
@pytest.mark.parametrize("wire", [3], indirect=True)
@pytest.mark.parametrize(("zero_quota", "billing"), [(False, True), (True, False), (True, True)])
async def test_nonrecoverable_quota_or_billing_is_not_retried(
    wire: MistralWire, controlled_retry_sleep: list[float], zero_quota: bool, billing: bool
) -> None:
    headers = {"x-ratelimit-limit-req-minute": "0"} if zero_quota else {}
    detail = "Insufficient credits" if billing else "Rate limit exceeded"
    wire.responses.append(httpx.Response(429, headers=headers, json={"message": detail}))
    error_type = LLMBillingError if billing else LLMRateLimitError
    with pytest.raises(error_type) as caught:
        await wire.provider.generate(_history(), model=MODEL)
    assert caught.value.provider == "mistral"
    assert len(wire.requests) == 1 and not controlled_retry_sleep
    if zero_quota and not billing:
        assert "zero request quota" in str(caught.value)


@pytest.mark.asyncio
async def test_cancelled_nonstream_read_closes_response(wire: MistralWire) -> None:
    body = FragmentedSSE(json.dumps(_completion()).encode(), pause_after=True)
    wire.responses.append(httpx.Response(200, headers={"Content-Type": "application/json"}, stream=body))
    task = asyncio.create_task(wire.provider.generate(_history(), model=MODEL))
    try:
        await asyncio.wait_for(body.waiting.wait(), 2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert body.closed
    finally:
        body.release.set()
        if not task.done():
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
