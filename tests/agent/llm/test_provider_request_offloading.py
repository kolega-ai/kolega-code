"""Request preparation must not run on the event loop.

Building a request body is O(history): for an image-heavy conversation the SDK's
parameter transform blocks the loop for hundreds of milliseconds per request, once per
concurrently running sub-agent. The Anthropic streaming path can move all of that off
the loop because ``messages.stream()`` is a synchronous SDK call; everywhere else only
our own payload construction can be offloaded, because the SDK method is async.
"""

import json
import threading
from typing import Any, cast

import httpx
import pytest
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

from kolega_code.llm.models import Message, MessageHistory, TextBlock
from kolega_code.llm.providers.anthropic import AnthropicProvider
from kolega_code.llm.providers.openai import OpenAIProvider

_ANTHROPIC_MESSAGE = {
    "id": "msg_1",
    "type": "message",
    "role": "assistant",
    "model": "claude-opus-5",
    "content": [{"type": "text", "text": "ok"}],
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {"input_tokens": 10, "output_tokens": 2},
}

_ANTHROPIC_EVENTS = [
    ("message_start", {"type": "message_start", "message": {**_ANTHROPIC_MESSAGE, "content": []}}),
    (
        "content_block_start",
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
    ),
    (
        "content_block_delta",
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "ok"}},
    ),
    ("content_block_stop", {"type": "content_block_stop", "index": 0}),
    (
        "message_delta",
        {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 2}},
    ),
    ("message_stop", {"type": "message_stop"}),
]

_OPENAI_CHUNKS = [
    {
        "id": "chatcmpl-1",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "gpt-5.6",
        "choices": [{"index": 0, "delta": {"role": "assistant", "content": "ok"}, "finish_reason": None}],
    },
    {
        "id": "chatcmpl-1",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "gpt-5.6",
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    },
]


def _sse(events: list[tuple[str, dict[str, Any]]]) -> str:
    return "".join(f"event: {name}\ndata: {json.dumps(body)}\n\n" for name, body in events)


def _history() -> MessageHistory:
    return MessageHistory([Message(role="user", content=[TextBlock(text="hello")])])


def _record_conversion_thread(monkeypatch: pytest.MonkeyPatch, method: str) -> dict[str, int]:
    """Note which thread builds the wire payload."""
    seen: dict[str, int] = {}
    original = getattr(MessageHistory, method)

    def tracking(self, *args, **kwargs):
        seen["thread"] = threading.get_ident()
        return original(self, *args, **kwargs)

    monkeypatch.setattr(MessageHistory, method, tracking)
    return seen


@pytest.mark.asyncio
async def test_anthropic_stream_is_prepared_off_the_event_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    bodies: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(200, content=_sse(_ANTHROPIC_EVENTS), headers={"content-type": "text/event-stream"})

    provider = AnthropicProvider(api_key="fake-key", max_retries=0)
    provider.async_client = AsyncAnthropic(
        api_key="fake-key",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    seen = _record_conversion_thread(monkeypatch, "to_anthropic")
    history = _history()

    wrapper = cast(
        Any, await provider.stream(history, system=Message(role="system", content=[TextBlock(text="system")]))
    )
    async with wrapper:
        chunks = [chunk async for chunk in wrapper]

    assert seen["thread"] != threading.get_ident()
    # The whole request survived the thread hop: same body, same streamed content.
    assert [chunk.text for chunk in chunks if chunk.type == "text"] == ["ok"]
    assert bodies[0]["messages"] == [{"role": "user", "content": [{"type": "text", "text": "hello"}]}]
    assert bodies[0]["system"] == [{"type": "text", "text": "system"}]
    assert bodies[0]["stream"] is True


@pytest.mark.asyncio
async def test_anthropic_generate_builds_its_payload_off_the_event_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_ANTHROPIC_MESSAGE)

    provider = AnthropicProvider(api_key="fake-key", max_retries=0)
    provider.async_client = AsyncAnthropic(
        api_key="fake-key",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    seen = _record_conversion_thread(monkeypatch, "to_anthropic")

    message = await provider.generate(_history(), system=Message(role="system", content=[TextBlock(text="system")]))

    assert seen["thread"] != threading.get_ident()
    assert message.get_text_content() == "ok"


@pytest.mark.asyncio
async def test_openai_builds_its_payload_off_the_event_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    bodies: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        payload = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in _OPENAI_CHUNKS) + "data: [DONE]\n\n"
        return httpx.Response(200, content=payload, headers={"content-type": "text/event-stream"})

    provider = OpenAIProvider(api_key="sk-test", max_retries=0)
    provider.async_client = AsyncOpenAI(
        api_key="sk-test",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    seen = _record_conversion_thread(monkeypatch, "to_openai")

    wrapper = cast(Any, await provider.stream(_history(), model="gpt-5.6"))
    async with wrapper:
        chunks = [chunk async for chunk in wrapper]

    assert seen["thread"] != threading.get_ident()
    assert [chunk.text for chunk in chunks if chunk.type == "text"] == ["ok"]
    assert bodies[0]["messages"] == [{"role": "user", "content": [{"type": "text", "text": "hello"}]}]
