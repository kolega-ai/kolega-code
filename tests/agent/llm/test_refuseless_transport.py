from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager
from functools import partial
import inspect
import json
from typing import Any

import httpx
from openai import AsyncOpenAI
import pytest

from kolega_code.llm.client import LLMClient
from kolega_code.llm.exceptions import LLMContextWindowExceededError
from kolega_code.llm.models import Message, MessageHistory, TextBlock, ThinkingBlock, ToolCall
from kolega_code.llm.providers.models import GenerationParams
from kolega_code.llm.providers import openai as transport

pytestmark = pytest.mark.usefixtures("isolated_cli_env")

FAKE_KEY = "abl_refuseless_hermetic_test_not_a_real_key"
MODEL = "deepseek-v4.1-flash-abliterated"


def _chunk(delta: dict[str, Any], *, finish_reason: str | None = None) -> dict[str, Any]:
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": MODEL,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }


def _sse(events: list[dict[str, Any]]) -> httpx.Response:
    body = "".join(f"data: {json.dumps(event)}\n\n" for event in events) + "data: [DONE]\n\n"
    return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body)


def _response(request: httpx.Request) -> httpx.Response:
    body = json.loads(request.content)
    if body.get("stream"):
        return _sse([_chunk({"content": "ok"}), _chunk({}, finish_reason="stop")])
    return httpx.Response(
        200,
        json={
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 0,
            "model": body["model"],
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
        },
    )


@asynccontextmanager
async def _client(
    monkeypatch: pytest.MonkeyPatch,
    handler: Callable[[httpx.Request], httpx.Response],
    **kwargs: Any,
) -> AsyncGenerator[LLMClient]:
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        monkeypatch.setattr(transport, "AsyncOpenAI", partial(AsyncOpenAI, http_client=http_client))
        yield LLMClient(provider="refuseless", api_key=FAKE_KEY, model=MODEL, **kwargs)


async def _invoke(client: LLMClient, method: str, **kwargs: Any) -> Message:
    messages = MessageHistory([Message("user", [TextBlock("hello")])])
    if method == "generate":
        return await client.generate(messages, **kwargs)
    stream: Any = client.stream(messages, **kwargs)
    if inspect.isawaitable(stream):
        stream = await stream
    async with stream:
        async for _ in stream:
            pass
        return await stream.get_final_message()


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["generate", "stream"])
@pytest.mark.parametrize(
    "model",
    [MODEL, "deepseek-v4.1-flash-fast"],
)
@pytest.mark.parametrize("cap", [None, 4096, 96000])
async def test_refuseless_does_not_inherit_deepseek_cap(
    monkeypatch: pytest.MonkeyPatch, method: str, model: str, cap: int | None
) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return _response(request)

    params = GenerationParams(max_completion_tokens=cap)
    async with _client(monkeypatch, handler) as client:
        message = await _invoke(client, method, model=model, params=params)
    assert message.get_text_content() == "ok"
    assert message.usage_metadata["provider"] == "refuseless"
    assert params.max_completion_tokens == cap
    assert len(requests) == 1
    request = requests[0]
    assert request.url.path == "/v1/chat/completions"
    assert request.headers["authorization"] == f"Bearer {FAKE_KEY}"
    body = json.loads(request.content)
    assert body["model"] == model
    assert body.get("max_tokens") == cap
    if cap is None:
        assert "max_tokens" not in body
        assert "max_completion_tokens" not in body
    if method == "stream":
        assert body["stream_options"] == {"include_usage": True}


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["generate", "stream"])
@pytest.mark.parametrize(
    ("model", "cap"),
    [
        ("glm-5.3-flash-abliterated", 131072),
        ("deepseek-v4.1-flash-fast", 262144),
        ("minimax-m3.1-flash", 512000),
    ],
)
async def test_refuseless_published_output_limits_reach_chat_completions(
    monkeypatch: pytest.MonkeyPatch, method: str, model: str, cap: int
) -> None:
    requests: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return _response(request)

    async with _client(monkeypatch, handler) as client:
        await _invoke(client, method, model=model, max_completion_tokens=cap)
    assert len(requests) == 1
    assert requests[0]["model"] == model
    assert requests[0]["max_tokens"] == cap


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["generate", "stream"])
@pytest.mark.parametrize("requested,expected", [(None, 2048), (96000, 2048), (256, 256)])
async def test_refuseless_preserves_strict_run_and_smaller_caller_limits(
    monkeypatch: pytest.MonkeyPatch, method: str, requested: int | None, expected: int
) -> None:
    requests: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return _response(request)

    params = GenerationParams(max_completion_tokens=requested)
    async with _client(monkeypatch, handler, context_window_tokens=10000, max_output_tokens=2048) as client:
        await _invoke(client, method, model=MODEL, params=params, _precomputed_input_tokens=10)
    assert len(requests) == 1
    assert requests[0]["max_tokens"] == expected
    assert "_precomputed_input_tokens" not in requests[0]
    assert params.max_completion_tokens == requested


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["generate", "stream"])
async def test_refuseless_strict_input_budget_rejects_before_transport(
    monkeypatch: pytest.MonkeyPatch, method: str
) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return _response(request)

    async with _client(monkeypatch, handler, context_window_tokens=10000, max_output_tokens=2048) as client:
        with pytest.raises(LLMContextWindowExceededError):
            await _invoke(client, method, model=MODEL, _precomputed_input_tokens=7953)
    assert requests == []


@pytest.mark.parametrize("provider", ["openai", "fireworks", "ollama_cloud", "openrouter", "custom:test", "deepseek"])
@pytest.mark.parametrize("requested,expected", [(None, 64000), (96000, 64000), (1024, 1024)])
def test_other_providers_retain_name_based_deepseek_cap(provider: str, requested: int | None, expected: int) -> None:
    instance = object.__new__(transport.OpenAIProvider)
    instance.provider_name = provider
    params: dict[str, Any] = {"model": MODEL}
    if requested is not None:
        params["max_tokens"] = requested
    instance._apply_model_request_rules(params)
    assert params["max_tokens"] == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("reasoning_field", ["reasoning_content", "reasoning"])
async def test_refuseless_stream_preserves_reasoning_split_tools_and_usage(
    monkeypatch: pytest.MonkeyPatch, reasoning_field: str
) -> None:
    events = [
        _chunk({reasoning_field: "check"}),
        _chunk({"content": "calling"}),
        _chunk(
            {
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "call_refuseless",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": '{"city":'},
                    }
                ]
            }
        ),
        _chunk({"tool_calls": [{"index": 0, "function": {"arguments": '"Paris"}'}}]}),
        _chunk({}, finish_reason="tool_calls"),
        {
            **_chunk({}),
            "choices": [],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 20,
                "total_tokens": 120,
                "prompt_tokens_details": {"cached_tokens": 40},
                "completion_tokens_details": {"reasoning_tokens": 7},
            },
        },
    ]
    async with _client(monkeypatch, lambda _: _sse(events)) as client:
        message = await _invoke(client, "stream", model=MODEL, max_completion_tokens=512)
    assert message.get_text_content() == "calling"
    assert [block.thinking for block in message.content if isinstance(block, ThinkingBlock)] == ["check"]
    tools = [block for block in message.content if isinstance(block, ToolCall)]
    assert len(tools) == 1
    assert tools[0].name == "lookup"
    assert tools[0].input == {"city": "Paris"}
    assert message.stop_reason == "tool_use"
    assert message.usage_metadata["reasoning_field"] == reasoning_field
    assert message.usage is not None
    assert message.usage.provider == "refuseless"
    assert message.usage.input_tokens == 100
    assert message.usage.output_tokens == 20
    assert message.usage.total_tokens == 120
    assert message.usage.cache_read_input_tokens == 40
    assert message.usage.reasoning_output_tokens == 7
