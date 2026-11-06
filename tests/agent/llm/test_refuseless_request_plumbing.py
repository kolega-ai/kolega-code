from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from functools import partial
import inspect
import json
from typing import Any

import httpx
from openai import AsyncOpenAI
import pytest

from kolega_code.llm.client import LLMClient
from kolega_code.llm.exceptions import LLMInvalidRequestError
from kolega_code.llm.models import Message, MessageHistory, TextBlock
from kolega_code.llm.providers import openai as transport
from kolega_code.llm.providers.models import GenerationParams
from kolega_code.llm.specs import MODEL_SPECS, ThinkingEffortSpec, normalize_thinking_effort

pytestmark = pytest.mark.usefixtures("isolated_cli_env", "synthetic_request_catalog")

FAKE_KEY = "abl_FAKE_refuseless_request_plumbing_not_a_credential"
SYNTHETIC_REASONING_MODEL = "synthetic-refuseless-request-reasoning"
SYNTHETIC_DEEPSEEK_MODEL = "synthetic-refuseless-request-deepseek"
SYNTHETIC_EFFORT_LADDERS = {
    SYNTHETIC_REASONING_MODEL: ("low", "high", "max"),
    SYNTHETIC_DEEPSEEK_MODEL: ("off", "low", "high", "max"),
}


@pytest.fixture
def synthetic_request_catalog(monkeypatch: pytest.MonkeyPatch) -> None:
    for model, efforts in SYNTHETIC_EFFORT_LADDERS.items():
        for provider in ("refuseless", "fireworks", "ollama_cloud"):
            monkeypatch.setitem(
                MODEL_SPECS,
                (provider, model),
                {
                    "context_length": 200_000,
                    "max_completion_tokens": 100_000,
                    "input_budget": "window_minus_output",
                    "default_temperature": 1.0,
                    "supports_temperature": model == SYNTHETIC_REASONING_MODEL,
                    "thinking_effort": ThinkingEffortSpec(
                        options=efforts, default="low", mode="openai_reasoning_effort"
                    ),
                },
            )


def _response(body: dict[str, Any]) -> httpx.Response:
    envelope = {"id": "chatcmpl-synthetic", "created": 0, "model": body["model"]}
    if body.get("stream"):
        events = [
            {
                **envelope,
                "object": "chat.completion.chunk",
                "choices": [{"index": 0, "delta": {"content": "ok"}, "finish_reason": None}],
            },
            {
                **envelope,
                "object": "chat.completion.chunk",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            },
        ]
        content = "".join(f"data: {json.dumps(event)}\n\n" for event in events) + "data: [DONE]\n\n"
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=content)
    return httpx.Response(
        200,
        json={
            **envelope,
            "object": "chat.completion",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
        },
    )


@asynccontextmanager
async def _client(
    monkeypatch: pytest.MonkeyPatch,
    requests: list[httpx.Request],
    *,
    provider: str = "refuseless",
    **kwargs: Any,
) -> AsyncGenerator[LLMClient, None]:
    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return _response(json.loads(request.content))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        monkeypatch.setattr(transport, "AsyncOpenAI", partial(AsyncOpenAI, http_client=http_client))
        client = LLMClient(
            provider=provider, api_key=FAKE_KEY, model=SYNTHETIC_REASONING_MODEL, max_retries=0, **kwargs
        )
        assert isinstance(client.provider, transport.OpenAIProvider)
        assert client.provider.provider_name == provider
        try:
            yield client
        finally:
            await client.provider.async_client.close()


async def _invoke(client: LLMClient, method: str, model: str, **kwargs: Any) -> Message:
    history = MessageHistory([Message("user", [TextBlock("hello")])])
    if method == "generate":
        return await client.generate(history, model=model, **kwargs)
    stream: Any = client.stream(history, model=model, **kwargs)
    if inspect.isawaitable(stream):
        stream = await stream
    async with stream:
        async for _ in stream:
            pass
        return await stream.get_final_message()


def _request_body(requests: list[httpx.Request], method: str, model: str) -> dict[str, Any]:
    assert len(requests) == 1
    request = requests[0]
    assert request.method == "POST"
    assert request.url.path.endswith("/chat/completions")
    assert request.headers["authorization"] == f"Bearer {FAKE_KEY}"
    body = json.loads(request.content)
    assert body["model"] == model
    assert "max_completion_tokens" not in body
    assert "_precomputed_input_tokens" not in body
    if method == "stream":
        assert body["stream"] is True
        assert body["stream_options"] == {"include_usage": True}
    else:
        assert not body.get("stream")
        assert "stream_options" not in body
    return body


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["generate", "stream"])
@pytest.mark.parametrize("model", SYNTHETIC_EFFORT_LADDERS)
async def test_normalized_synthetic_default_low_reaches_refuseless_request(
    monkeypatch: pytest.MonkeyPatch, method: str, model: str
) -> None:
    requests: list[httpx.Request] = []
    effort = normalize_thinking_effort("refuseless", model, None)
    assert effort == "low"
    async with _client(monkeypatch, requests) as client:
        message = await _invoke(client, method, model, thinking=effort, max_completion_tokens=128)
    assert message.get_text_content() == "ok"
    assert message.usage_metadata["provider"] == "refuseless"
    assert str(requests[0].url) == "https://api.refuseless.com/v1/chat/completions"
    assert _request_body(requests, method, model)["reasoning_effort"] == "low"


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["generate", "stream"])
@pytest.mark.parametrize(
    "model,effort",
    [(model, effort) for model, efforts in SYNTHETIC_EFFORT_LADDERS.items() for effort in efforts],
)
async def test_every_synthetic_effort_including_native_off_is_flat_on_wire(
    monkeypatch: pytest.MonkeyPatch, method: str, model: str, effort: str
) -> None:
    requests: list[httpx.Request] = []
    params = GenerationParams(thinking=effort, max_completion_tokens=128)
    async with _client(monkeypatch, requests) as client:
        message = await _invoke(client, method, model, params=params)
    assert message.get_text_content() == "ok"
    body = _request_body(requests, method, model)
    assert body["reasoning_effort"] == effort
    assert not {"thinking", "reasoning", "output_config", "extra_body", "chat_template_kwargs"} & body.keys()
    assert params.thinking == effort


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["generate", "stream"])
@pytest.mark.parametrize(
    "model,effort",
    [(model, effort) for model in SYNTHETIC_EFFORT_LADDERS for effort in ("medium", "minimal", "xhigh", "none")]
    + [(SYNTHETIC_REASONING_MODEL, "off")],
)
async def test_absent_synthetic_efforts_and_none_alias_are_rejected_before_transport(
    monkeypatch: pytest.MonkeyPatch, method: str, model: str, effort: str
) -> None:
    requests: list[httpx.Request] = []
    assert effort not in SYNTHETIC_EFFORT_LADDERS[model]
    async with _client(monkeypatch, requests) as client:
        with pytest.raises(LLMInvalidRequestError, match="Unsupported thinking effort") as caught:
            await _invoke(client, method, model, thinking=effort, max_completion_tokens=128)
    assert caught.value.provider == "refuseless"
    assert requests == []


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["generate", "stream"])
@pytest.mark.parametrize("model", SYNTHETIC_EFFORT_LADDERS)
@pytest.mark.parametrize("temperature", [0.0, 0.37, 1.0])
async def test_synthetic_temperature_policy_controls_wire_field_without_mutating_params(
    monkeypatch: pytest.MonkeyPatch, method: str, model: str, temperature: float
) -> None:
    requests: list[httpx.Request] = []
    params = GenerationParams(temperature=temperature, thinking="high", max_completion_tokens=128)
    async with _client(monkeypatch, requests) as client:
        await _invoke(client, method, model, params=params)
    body = _request_body(requests, method, model)
    if model == SYNTHETIC_REASONING_MODEL:
        assert body["temperature"] == temperature
    else:
        assert "temperature" not in body
    assert body["reasoning_effort"] == "high"
    assert params.temperature == temperature


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["generate", "stream"])
@pytest.mark.parametrize("model", SYNTHETIC_EFFORT_LADDERS)
@pytest.mark.parametrize("cap", [128, 4096])
async def test_explicit_small_output_caps_survive_synthetic_catalog(
    monkeypatch: pytest.MonkeyPatch, method: str, model: str, cap: int
) -> None:
    requests: list[httpx.Request] = []
    params = GenerationParams(max_completion_tokens=cap)
    async with _client(monkeypatch, requests) as client:
        await _invoke(client, method, model, params=params)
    assert _request_body(requests, method, model)["max_tokens"] == cap
    assert params.max_completion_tokens == cap


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["generate", "stream"])
@pytest.mark.parametrize("provider", ["refuseless", "fireworks", "ollama_cloud"])
@pytest.mark.parametrize("cap", [None, 128, 90_000])
async def test_synthetic_deepseek_route_bypasses_legacy_clamp_only_on_refuseless(
    monkeypatch: pytest.MonkeyPatch, method: str, provider: str, cap: int | None
) -> None:
    requests: list[httpx.Request] = []
    params = GenerationParams(max_completion_tokens=cap)
    async with _client(monkeypatch, requests, provider=provider) as client:
        await _invoke(client, method, SYNTHETIC_DEEPSEEK_MODEL, params=params)
    body = _request_body(requests, method, SYNTHETIC_DEEPSEEK_MODEL)
    if provider == "refuseless":
        assert body.get("max_tokens") == cap
        if cap is None:
            assert "max_tokens" not in body
    else:
        assert body["max_tokens"] == (64_000 if cap is None else min(cap, 64_000))
    assert params.max_completion_tokens == cap


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["generate", "stream"])
@pytest.mark.parametrize("model", SYNTHETIC_EFFORT_LADDERS)
@pytest.mark.parametrize("requested,expected", [(None, 2048), (90_000, 2048), (128, 128)])
async def test_synthetic_strict_run_cap_and_smaller_caller_cap_are_preserved(
    monkeypatch: pytest.MonkeyPatch, method: str, model: str, requested: int | None, expected: int
) -> None:
    requests: list[httpx.Request] = []
    params = GenerationParams(thinking="high", max_completion_tokens=requested)
    async with _client(monkeypatch, requests, context_window_tokens=10_000, max_output_tokens=2048) as client:
        await _invoke(client, method, model, params=params, _precomputed_input_tokens=10_000 - expected)
    body = _request_body(requests, method, model)
    assert body["max_tokens"] == expected
    assert body["reasoning_effort"] == "high"
    assert params.max_completion_tokens == requested
