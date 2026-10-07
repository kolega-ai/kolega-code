"""Client/accounting/instrumentation/context integration tests for native Mistral.

These tests stay network-free: they replace the provider's ``httpx.AsyncClient``
with ``httpx.MockTransport`` and assert the request/response contract at the
LLMClient layer instead of reusing provider implementation details.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any, cast
from unittest.mock import MagicMock

import httpx
import pytest
import pytest_asyncio

from kolega_code.llm.client import LLMClient
from kolega_code.llm.exceptions import LLMContextWindowExceededError, LLMError
from kolega_code.llm.instrumented_client import InstrumentedLLMClient, get_output_tokens
from kolega_code.llm.ledger import UsageLedger
from kolega_code.llm.models import Message, MessageHistory, TextBlock, ThinkingBlock, ToolDefinition, ToolParameter
from kolega_code.llm.providers.mistral import MistralProvider
from kolega_code.llm.usage import REASON_NOT_REPORTED

MODEL = "mistral-medium-3-5"
SMALL_MODEL = "mistral-small-2603"
USAGE = {"prompt_tokens": 41, "completion_tokens": 7, "total_tokens": 48}


def _messages(*items: Message) -> MessageHistory:
    return MessageHistory(list(items) or [Message("user", [TextBlock("Reply with ok.")])])


def _system(text: str = "You are concise.") -> Message:
    return Message("system", [TextBlock(text)])


def _tool() -> ToolDefinition:
    return ToolDefinition(
        name="lookup",
        description="Lookup a local fact.",
        parameters=[ToolParameter(name="key", type="string", description="Fact key", required=True)],
    )


def _completion(content: Any = "ok", *, usage: dict[str, int] | None = USAGE) -> dict[str, Any]:
    body: dict[str, Any] = {
        "id": "cmpl-client-contract",
        "object": "chat.completion",
        "created": 1791331200,
        "model": MODEL,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
    }
    if usage is not None:
        body["usage"] = usage
    return body


def _event(
    delta: dict[str, Any] | None = None, *, finish: str | None = None, usage: dict[str, int] | None = None
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "id": "cmpl-client-stream",
        "object": "chat.completion.chunk",
        "created": 1791331200,
        "model": MODEL,
        "choices": [] if delta is None else [{"index": 0, "delta": delta, "finish_reason": finish}],
    }
    if usage is not None:
        body["usage"] = usage
    return body


def _sse(*events: dict[str, Any]) -> bytes:
    return b"".join(("data: " + json.dumps(event) + "\n\n").encode() for event in events) + b"data: [DONE]\n\n"


class MistralHttp:
    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []
        self.bodies: list[dict[str, Any]] = []
        self.responses: list[httpx.Response | Exception] = []

    async def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        self.bodies.append(json.loads(await request.aread()))
        assert self.responses, "unexpected Mistral HTTP request"
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    def completion(self, content: Any = "ok", *, usage: dict[str, int] | None = USAGE) -> None:
        self.responses.append(httpx.Response(200, json=_completion(content, usage=usage)))

    def stream(self, *events: dict[str, Any]) -> None:
        self.responses.append(httpx.Response(200, headers={"Content-Type": "text/event-stream"}, content=_sse(*events)))


@pytest_asyncio.fixture
async def mistral_http() -> AsyncIterator[MistralHttp]:
    wire = MistralHttp()
    try:
        yield wire
    finally:
        pass


async def _install_transport(client: LLMClient, wire: MistralHttp) -> None:
    provider = client.provider
    assert isinstance(provider, MistralProvider)
    original = provider.async_client
    provider.async_client = httpx.AsyncClient(
        base_url=original.base_url,
        headers=original.headers,
        transport=httpx.MockTransport(wire.handle),
    )
    await original.aclose()


async def _close_transport(client: LLMClient) -> None:
    provider = client.provider
    if isinstance(provider, MistralProvider):
        await provider.async_client.aclose()


@pytest.mark.asyncio
async def test_llmclient_dispatches_to_native_mistral_provider(mistral_http: MistralHttp) -> None:
    client = LLMClient(provider="mistral", api_key="mistral-integration-test", model=MODEL, max_retries=0)
    await _install_transport(client, mistral_http)
    try:
        mistral_http.completion("dispatched")
        response = await client.generate(
            messages=_messages(),
            system=_system(),
            model=MODEL,
            temperature=0.2,
            max_completion_tokens=123,
            thinking="none",
        )
    finally:
        await _close_transport(client)

    assert response.get_text_content() == "dispatched"
    assert response.usage is not None and response.usage.provider == "mistral"
    payload = mistral_http.bodies[0]
    assert mistral_http.requests[0].headers["authorization"] == "Bearer mistral-integration-test"
    assert payload["model"] == MODEL
    assert payload["stream"] is False
    assert payload["temperature"] == 0.2
    assert payload["max_tokens"] == 123
    assert payload["reasoning_effort"] == "none"


@pytest.mark.asyncio
async def test_strict_context_budget_rejects_before_http_request_and_counts_full_request(
    mistral_http: MistralHttp,
) -> None:
    client = LLMClient(
        provider="mistral",
        api_key="mistral-integration-test",
        model=MODEL,
        max_retries=0,
        context_window_tokens=70,
        max_output_tokens=20,
    )
    await _install_transport(client, mistral_http)
    counted: dict[str, Any] = {}

    async def count_tokens(**kwargs: Any):
        counted.update(kwargs)
        return type("TokenCountLike", (), {"input_tokens": 51})()

    client.provider.count_tokens = count_tokens  # type: ignore[method-assign]
    try:
        with pytest.raises(LLMContextWindowExceededError):
            await client.generate(
                messages=_messages(
                    Message("user", [TextBlock("history")]),
                    Message("assistant", [TextBlock("prior")]),
                    Message("user", [TextBlock("now")]),
                ),
                system=_system("system prompt"),
                tools=[_tool()],
                model=MODEL,
                thinking="high",
            )
    finally:
        await _close_transport(client)

    assert mistral_http.requests == []
    assert counted["system"].get_text_content() == "system prompt"
    assert [m.role for m in counted["messages"]] == ["user", "assistant", "user"]
    assert counted["tools"][0].name == "lookup"
    assert counted["model"] == MODEL
    assert counted["thinking"] == "high"


@pytest.mark.parametrize(
    "usage, expected_reported, expected_total",
    [
        (USAGE, True, 48),
        (None, False, 0),
        ({"prompt_tokens": 9, "completion_tokens": 4}, True, 13),
        ({"prompt_tokens": 9, "completion_tokens": "bad", "total_tokens": 99}, False, 0),
    ],
    ids=["reported", "missing", "partial-total-reconstructed", "malformed"],
)
@pytest.mark.asyncio
async def test_usage_ledger_generate_settlement_variants(
    mistral_http: MistralHttp, usage: dict[str, int] | None, expected_reported: bool, expected_total: int
) -> None:
    ledger = UsageLedger()
    client = LLMClient(provider="mistral", api_key="mistral-integration-test", usage_ledger=ledger, max_retries=0)
    await _install_transport(client, mistral_http)
    try:
        mistral_http.completion("usage", usage=usage)
        message = await client.generate(messages=_messages(), model=MODEL)
    finally:
        await _close_transport(client)

    assert message.usage is not None
    assert message.usage.reported is expected_reported
    if usage is None:
        assert message.usage.unavailable_reason == REASON_NOT_REPORTED
    snap = ledger.snapshot()
    assert snap.requests == 1
    assert snap.failed == 0
    assert snap.reported == (1 if expected_reported else 0)
    assert snap.unreported == (0 if expected_reported else 1)
    assert snap.total_tokens == expected_total


@pytest.mark.asyncio
async def test_usage_ledger_stream_reported_missing_and_error(mistral_http: MistralHttp) -> None:
    ledger = UsageLedger()
    client = LLMClient(provider="mistral", api_key="mistral-integration-test", usage_ledger=ledger, max_retries=0)
    await _install_transport(client, mistral_http)
    try:
        mistral_http.stream(_event({"content": "a"}, finish="stop"), _event(usage=USAGE))
        cm1 = await cast(Any, client.stream(messages=_messages(), model=MODEL))
        async with cm1 as stream:
            assert "".join([chunk.text or "" async for chunk in stream]) == "a"
        await cm1.get_final_message()

        mistral_http.stream(_event({"content": "b"}, finish="stop"))
        cm2 = await cast(Any, client.stream(messages=_messages(), model=MODEL))
        async with cm2 as stream:
            async for _ in stream:
                pass
        await cm2.get_final_message()

        mistral_http.responses.append(httpx.Response(500, json={"message": "backend down"}))
        with pytest.raises(LLMError):
            await cast(Any, client.stream(messages=_messages(), model=MODEL))
    finally:
        await _close_transport(client)

    snap = ledger.snapshot()
    assert (snap.requests, snap.responses, snap.reported, snap.unreported, snap.failed) == (3, 2, 1, 1, 1)
    assert snap.total_tokens == 48
    assert snap.complete is False


def _mock_langfuse() -> MagicMock:
    langfuse = MagicMock()
    generation = MagicMock()
    trace = MagicMock()
    trace.start_generation = MagicMock(return_value=generation)
    langfuse.start_span = MagicMock(return_value=trace)
    return langfuse


@pytest.mark.asyncio
async def test_instrumented_mistral_counters_and_usage_recorder(mistral_http: MistralHttp) -> None:
    recorded: list[dict[str, Any]] = []
    langfuse = _mock_langfuse()
    client = InstrumentedLLMClient(
        provider="mistral",
        api_key="mistral-integration-test",
        model=MODEL,
        max_retries=0,
        langfuse_client=langfuse,
        usage_recorder=recorded.append,
        workspace_id="workspace",
        thread_id="thread",
        agent_type="agent",
    )
    await _install_transport(client, mistral_http)
    try:
        mistral_http.completion("instrumented", usage=USAGE)
        response = await client.generate(messages=_messages(), model=MODEL)
    finally:
        await _close_transport(client)

    assert get_output_tokens(response.usage_metadata, "mistral") == 7
    assert recorded and recorded[0]["provider"] == "mistral"
    assert recorded[0]["input_tokens"] == 41
    assert recorded[0]["output_tokens"] == 7
    assert recorded[0]["metadata"]["raw_usage"] == response.usage_metadata
    generation = langfuse.start_span.return_value.start_generation.return_value
    assert generation.update.call_args.kwargs["usage_details"] == {
        "input": 41,
        "output": 7,
        "total": 48,
        "cache_read_input_tokens": 0,
        "cache_creation_input_tokens": 0,
    }


@pytest.mark.asyncio
async def test_instrumented_mistral_stream_counters(mistral_http: MistralHttp) -> None:
    recorded: list[dict[str, Any]] = []
    langfuse = _mock_langfuse()
    client = InstrumentedLLMClient(
        provider="mistral",
        api_key="mistral-integration-test",
        max_retries=0,
        langfuse_client=langfuse,
        usage_recorder=recorded.append,
    )
    await _install_transport(client, mistral_http)
    try:
        mistral_http.stream(_event({"content": "streamed"}, finish="stop"), _event(usage=USAGE))
        cm = await cast(Any, client.stream(messages=_messages(), model=MODEL))
        async with cm as stream:
            assert "".join([chunk.text or "" async for chunk in stream]) == "streamed"
        final = await stream.get_final_message()
        assert final.usage is not None and final.usage.total_tokens == 48
    finally:
        await _close_transport(client)
    assert len(recorded) == 1
    assert (recorded[0]["input_tokens"], recorded[0]["output_tokens"]) == (41, 7)
    generation = langfuse.start_span.return_value.start_generation.return_value
    assert generation.update.call_args.kwargs["usage_details"]["total"] == 48


@pytest.mark.asyncio
async def test_native_history_roundtrip_preserves_provider_metadata_and_replays_native_thinking(
    mistral_http: MistralHttp,
) -> None:
    native_content = [
        {
            "type": "thinking",
            "thinking": [{"type": "text", "text": "kept native"}],
            "signature": "sig-123",
            "closed": True,
        },
        {"type": "text", "text": "visible"},
    ]
    client = LLMClient(provider="mistral", api_key="mistral-integration-test", max_retries=0)
    await _install_transport(client, mistral_http)
    try:
        mistral_http.completion(native_content)
        first = await client.generate(messages=_messages(), model=MODEL, thinking="high")
        thinking = next(block for block in cast(list[Any], first.content) if isinstance(block, ThinkingBlock))
        assert thinking.thinking == "kept native"
        assert thinking.signature == "sig-123"
        assert getattr(thinking, "provider_metadata", {}).get("closed") is True

        restored = Message.from_dict(json.loads(json.dumps(first.to_dict())))
        restored_thinking = next(
            block for block in cast(list[Any], restored.content) if isinstance(block, ThinkingBlock)
        )
        assert getattr(restored_thinking, "provider_metadata", {}) == getattr(thinking, "provider_metadata", {})

        mistral_http.completion("continued")
        await client.generate(
            messages=_messages(
                Message("user", [TextBlock("first")]),
                restored,
                Message("user", [TextBlock("continue")]),
            ),
            model=SMALL_MODEL,
        )
    finally:
        await _close_transport(client)

    replay = next(item for item in mistral_http.bodies[-1]["messages"] if item["role"] == "assistant")
    assert replay["content"][0] == {
        "type": "thinking",
        "thinking": [{"type": "text", "text": "kept native"}],
        "signature": "sig-123",
        "closed": True,
    }
    assert "*Thinking:*" not in json.dumps(replay)


@pytest.mark.asyncio
async def test_stable_prompt_cache_key_across_sequential_requests_from_same_history(
    mistral_http: MistralHttp,
) -> None:
    client = LLMClient(provider="mistral", api_key="mistral-integration-test", model=MODEL, max_retries=0)
    await _install_transport(client, mistral_http)
    history = _messages(Message("user", [TextBlock("same history")]))
    try:
        mistral_http.completion("one")
        await client.generate(messages=history, model=MODEL)
        mistral_http.completion("two")
        await client.generate(messages=history, model=MODEL)
    finally:
        await _close_transport(client)

    keys = [body.get("prompt_cache_key") for body in mistral_http.bodies]
    assert keys[0] == keys[1]
    assert isinstance(keys[0], str) and keys[0]
