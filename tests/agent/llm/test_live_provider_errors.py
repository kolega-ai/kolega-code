import json
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, Mock

import httpx
import pytest
from openai import APIStatusError, AsyncOpenAI, BadRequestError, NotFoundError

from kolega_code.llm.client import LLMClient
from kolega_code.llm.exceptions import LLMAuthenticationError, LLMError, LLMInvalidRequestError, LLMNotFoundError
from kolega_code.llm.providers.openai import OpenAIProvider
from kolega_code.llm.providers.responses_common import ResponsesProviderBase

from . import test_deepseek_output_cap_live as output_cap
from . import test_deepseek_responses_live as deepseek_responses
from . import test_live_providers as smoke
from . import test_normalized_usage_live as usage
from .test_perplexity_quota_errors import QUOTA_MESSAGE, _FailingStream, _sdk_error


_CASES = [
    (401, {"error": {"message": QUOTA_MESSAGE}}, True),
    (401, {"error": {"code": "insufficient_quota"}}, True),
    (401, {"error": {"code": "invalid_api_key", "message": "Invalid API key"}}, False),
    (401, {"error": {"message": "Unauthorized"}}, False),
    (401, {"error": {"message": "Check your billing details"}}, False),
    (400, {"error": {"message": "Malformed request"}}, False),
    (404, {"error": {"message": "Model not found"}}, False),
    (418, {"error": {"message": "Unknown failure"}}, False),
]
_EXPECTED_ERRORS = {
    401: LLMAuthenticationError,
    400: LLMInvalidRequestError,
    404: LLMNotFoundError,
    418: LLMError,
}


@pytest.mark.asyncio
@pytest.mark.parametrize("suite", ["smoke", "generate_usage", "stream_usage"])
@pytest.mark.parametrize("status", [200, 400, 404])
async def test_together_live_fixture_uses_real_sdk_transport_without_hiding_request_errors(
    monkeypatch: pytest.MonkeyPatch, suite: str, status: int
) -> None:
    model = "openai/gpt-oss-120b"
    requests: list[dict[str, Any]] = []
    reported_usage = {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}

    def respond(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        requests.append(payload)
        if status != 200:
            return httpx.Response(status, json={"error": {"message": "Unable to access requested model"}})
        base = {"id": "offline-together", "created": 0, "model": model}
        if payload.get("stream"):
            chunks = [
                {
                    **base,
                    "object": "chat.completion.chunk",
                    "choices": [{"index": 0, "delta": {"reasoning": "Add two and two"}, "finish_reason": None}],
                },
                {
                    **base,
                    "object": "chat.completion.chunk",
                    "choices": [{"index": 0, "delta": {"content": "4"}, "finish_reason": "stop"}],
                },
                {**base, "object": "chat.completion.chunk", "choices": [], "usage": reported_usage},
            ]
            body = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks) + "data: [DONE]\n\n"
            return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})
        return httpx.Response(
            200,
            json={
                **base,
                "object": "chat.completion",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "4", "reasoning": "Add two and two"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": reported_usage,
            },
        )

    client = LLMClient(provider="together", api_key="test-placeholder", model=model)
    assert isinstance(client.provider, OpenAIProvider)
    client.provider.async_client = AsyncOpenAI(
        api_key="test-placeholder",
        base_url="https://example.test/v1",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)),
    )
    for module in (smoke, usage):
        monkeypatch.setattr(module, "_require_key", lambda _: "test-placeholder")
        monkeypatch.setattr(module, "LLMClient", lambda **_: client)
    probe = {
        "smoke": smoke.test_live_provider_generate,
        "generate_usage": usage.test_live_generate_attaches_reported_usage,
        "stream_usage": usage.test_live_stream_attaches_reported_usage_idempotently,
    }[suite]
    try:
        if status == 200:
            await probe("together", model)
        else:
            expected = (
                {400: BadRequestError, 404: NotFoundError}[status]
                if suite == "stream_usage"
                else _EXPECTED_ERRORS[status]
            )
            with pytest.raises(expected) as caught:
                await probe("together", model)
            assert type(caught.value) is expected
    finally:
        await client.provider.async_client.close()
    assert len(requests) == 1
    assert requests[0]["model"] == model
    assert requests[0]["temperature"] == 1.0
    assert requests[0]["max_tokens"] == 4096
    assert "reasoning_effort" not in requests[0]
    assert "thinking" not in requests[0]
    if suite == "stream_usage":
        assert requests[0]["stream_options"] == {"include_usage": True}


@pytest.mark.asyncio
@pytest.mark.parametrize("text", ["vision-ok", "vision-9k"], ids=["correct", "incorrect"])
async def test_deepseek_literal_probe_requires_exact_text(text: str) -> None:
    requests: list[dict[str, Any]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        identity = {"item_id": "msg_1", "output_index": 1, "content_index": 0}
        reasoning = {
            "type": "reasoning",
            "id": "reasoning_1",
            "summary": [],
            "content": [{"type": "reasoning_text", "text": "Return vision-ok, not vision-9k."}],
        }
        item = {
            "type": "message",
            "id": "msg_1",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": text, "annotations": []}],
        }
        events = [
            {"type": "response.output_item.done", "output_index": 0, "item": reasoning},
            {"type": "response.output_text.delta", **identity, "delta": text},
            {"type": "response.output_text.done", **identity, "text": text},
            {"type": "response.output_item.done", "output_index": 1, "item": item},
            {
                "type": "response.completed",
                "response": {
                    "id": "offline-deepseek-literal",
                    "object": "response",
                    "model": "deepseek-flash",
                    "status": "completed",
                    "output": [reasoning, item],
                    "usage": {"input_tokens": 10, "output_tokens": 4, "total_tokens": 14},
                },
            },
        ]
        body = "".join(
            f"event: {event['type']}\ndata: {json.dumps({**event, 'sequence_number': index})}\n\n"
            for index, event in enumerate(events)
        )
        return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})

    client = LLMClient(provider="deepseek", api_key="test-placeholder", model="deepseek-flash")
    assert isinstance(client.provider, ResponsesProviderBase)
    client.provider.async_client = AsyncOpenAI(
        api_key="test-placeholder",
        base_url="https://example.test/beta",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)),
    )
    try:
        if text == "vision-ok":
            await deepseek_responses.test_vision_generate_text_with_thinking(client)
        else:
            with pytest.raises(AssertionError, match="vision-9k.*vision-ok"):
                await deepseek_responses.test_vision_generate_text_with_thinking(client)
    finally:
        await client.provider.async_client.close()

    assert len(requests) == 1
    assert requests[0]["model"] == "deepseek-flash"
    assert requests[0]["stream"] is True
    assert requests[0]["reasoning"] == {"effort": "high", "summary": "auto"}
    assert requests[0]["instructions"] == "Follow the user's instruction exactly."
    assert requests[0]["input"] == [
        {"role": "user", "content": [{"type": "input_text", "text": "Reply with exactly: vision-ok"}]}
    ]
    assert "temperature" not in requests[0]


@pytest.mark.asyncio
@pytest.mark.parametrize("model", ["deepseek/deepseek-v4-pro", "deepseek/deepseek-v4-flash"])
async def test_chat_cap_probe_preserves_tiny_cap_through_the_gateway(
    monkeypatch: pytest.MonkeyPatch, model: str
) -> None:
    requests: list[dict[str, Any]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        chunk = {
            "id": "offline-tiny-cap",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": model,
            "choices": [{"index": 0, "delta": {"content": "The ocean"}, "finish_reason": "length"}],
        }
        return httpx.Response(
            200,
            text=f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n",
            headers={"content-type": "text/event-stream"},
        )

    client = LLMClient(provider="openrouter", api_key="test-placeholder", model=model)
    assert isinstance(client.provider, OpenAIProvider)
    client.provider.async_client = AsyncOpenAI(
        api_key="test-placeholder",
        base_url="https://example.test/v1",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)),
    )
    monkeypatch.setattr(output_cap, "_require_key", lambda _: "test-placeholder")
    monkeypatch.setattr(output_cap, "LLMClient", lambda **_: client)
    record = Mock()
    monkeypatch.setattr(output_cap, "_record", record)
    try:
        await output_cap.test_chat_tiny_cap_truncation_reporting("openrouter", model)
    finally:
        await client.provider.async_client.close()
    assert len(requests) == 1
    assert requests[0]["model"] == model
    assert requests[0]["max_tokens"] == 16
    assert requests[0]["stream"] is True
    assert record.call_args.kwargs["stop_reason"] == "max_tokens"


@pytest.mark.asyncio
@pytest.mark.parametrize("suite", ["smoke", "usage"])
@pytest.mark.parametrize("stage", ["create", "iterate"])
@pytest.mark.parametrize("status,body,quota", _CASES)
async def test_live_generate_guards_skip_only_normalized_quota(
    monkeypatch: pytest.MonkeyPatch, suite: str, stage: str, status: int, body: dict[str, Any], quota: bool
) -> None:
    provider = "perplexity_agent"
    model = dict(usage.PROVIDER_MODELS)[provider]
    client = LLMClient(provider=provider, api_key="test-placeholder", model=model)
    assert isinstance(client.provider, ResponsesProviderBase)
    error = _sdk_error(status, body)
    create = AsyncMock(side_effect=error) if stage == "create" else AsyncMock(return_value=_FailingStream(error))
    client.provider.async_client = cast(AsyncOpenAI, SimpleNamespace(responses=SimpleNamespace(create=create)))
    monkeypatch.setattr(usage, "_require_key", lambda _: "test-placeholder")
    monkeypatch.setattr(usage, "LLMClient", lambda **_: client)
    if quota:
        with pytest.raises(pytest.skip.Exception, match="quota"):
            if suite == "smoke":
                await smoke._live_generate(client, messages=smoke._messages(), model=model)
            else:
                await usage.test_live_generate_attaches_reported_usage(provider, model)
    else:
        with pytest.raises(_EXPECTED_ERRORS[status]) as caught:
            if suite == "smoke":
                await smoke._live_generate(client, messages=smoke._messages(), model=model)
            else:
                await usage.test_live_generate_attaches_reported_usage(provider, model)
        assert caught.value.__cause__ is error
        assert type(caught.value) is _EXPECTED_ERRORS[status]
    create.assert_awaited_once()


@pytest.mark.parametrize(
    "message,recognized",
    [
        ("The valid range of max_output_tokens is [1, 393216]", True),
        ("max_tokens must be at most 65536", True),
        ("max_completion_tokens must be between 1 and 65536", True),
        ("Model not available", False),
        ("Invalid max_tokens format", False),
        ("Invalid billing parameter", False),
        ("Unknown failure", False),
    ],
)
def test_cap_probe_recognizes_only_explicit_output_limit_rejections(message: str, recognized: bool) -> None:
    request = httpx.Request("POST", "https://example.test/responses")
    body = {"error": {"message": message}}
    bad_request = BadRequestError(message, response=httpx.Response(400, request=request), body=body)
    assert output_cap._is_output_cap_rejection(bad_request) is recognized
    for status in (401, 403, 404, 500):
        other = APIStatusError(message, response=httpx.Response(status, request=request), body=body)
        assert output_cap._is_output_cap_rejection(other) is False


@pytest.mark.asyncio
@pytest.mark.parametrize("suite", ["chat", "responses", "openrouter"])
@pytest.mark.parametrize(
    "status,message,cap_rejection",
    [
        (400, "max_output_tokens must be at most 65536", True),
        (400, "max_tokens must be between 1 and 65536", True),
        (400, "Malformed request", False),
        (401, "Unauthorized", False),
        (404, "Model not available", False),
        (418, "Unknown failure", False),
    ],
)
async def test_cap_probe_records_only_explicit_output_limit_rejections(
    monkeypatch: pytest.MonkeyPatch, suite: str, status: int, message: str, cap_rejection: bool
) -> None:
    error = _sdk_error(status, {"error": {"message": message}})
    if status == 400:
        error = BadRequestError(
            message,
            response=httpx.Response(400, request=httpx.Request("POST", "https://example.test/responses")),
            body={"error": {"message": message}},
        )
    monkeypatch.setattr(output_cap, "_require_key", lambda _: "test-placeholder")
    record = Mock()
    monkeypatch.setattr(output_cap, "_record", record)
    if suite == "chat":
        normalized = _EXPECTED_ERRORS[status](message)
        normalized.__cause__ = error
        create = AsyncMock(side_effect=normalized)
        client = SimpleNamespace(generate=create)
        monkeypatch.setattr(output_cap, "LLMClient", lambda **_: client)
        probe = output_cap.test_chat_overcap_request_outcome(*output_cap.CHAT_HOSTS[0])
        expected_error = normalized
    elif suite == "responses":
        model = output_cap.RESPONSES_MODELS[0]
        client = LLMClient(provider="deepseek", api_key="test-placeholder", model=model)
        assert isinstance(client.provider, ResponsesProviderBase)
        create = AsyncMock(side_effect=error)
        client.provider.async_client = cast(AsyncOpenAI, SimpleNamespace(responses=SimpleNamespace(create=create)))
        monkeypatch.setattr(output_cap, "LLMClient", lambda **_: client)
        probe = output_cap.test_responses_overcap_request_outcome(model)
        expected_error = error
    else:
        response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="4"), finish_reason="length")],
            usage=SimpleNamespace(completion_tokens=16),
        )
        create = AsyncMock(side_effect=[response, response, error])
        client = SimpleNamespace(
            provider=SimpleNamespace(
                async_client=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
            )
        )
        monkeypatch.setattr(output_cap, "LLMClient", lambda **_: client)
        probe = output_cap.test_openrouter_deepseek_accepts_clamped_cap(output_cap.OPENROUTER_SLUGS[0])
        expected_error = error
    if cap_rejection:
        await probe
        label = {"chat": "chat-overcap", "responses": "responses-overcap", "openrouter": "openrouter-overcap"}[suite]
        assert record.call_args.args == (label,)
        assert record.call_args.kwargs["outcome"] == "rejected"
    else:
        with pytest.raises(type(expected_error)) as caught:
            await probe
        assert caught.value is expected_error
        assert all(call.kwargs.get("outcome") != "rejected" for call in record.call_args_list)
    assert create.await_count == (3 if suite == "openrouter" else 1)


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["perplexity_agent", "deepseek"])
@pytest.mark.parametrize("stage", ["create", "iterate"])
@pytest.mark.parametrize("status,body,quota", _CASES)
async def test_usage_stream_guard_skips_only_explicit_perplexity_quota(
    monkeypatch: pytest.MonkeyPatch, provider: str, stage: str, status: int, body: dict[str, Any], quota: bool
) -> None:
    model = dict(usage.PROVIDER_MODELS)[provider]
    client = LLMClient(provider=provider, api_key="test-placeholder", model=model)
    assert isinstance(client.provider, ResponsesProviderBase)
    error = _sdk_error(status, body)
    create = AsyncMock(side_effect=error) if stage == "create" else AsyncMock(return_value=_FailingStream(error))
    client.provider.async_client = cast(AsyncOpenAI, SimpleNamespace(responses=SimpleNamespace(create=create)))
    monkeypatch.setattr(usage, "_require_key", lambda _: "test-placeholder")
    monkeypatch.setattr(usage, "LLMClient", lambda **_: client)
    if quota and provider == "perplexity_agent":
        with pytest.raises(pytest.skip.Exception, match="Perplexity quota"):
            await usage.test_live_stream_attaches_reported_usage_idempotently(provider, model)
    else:
        with pytest.raises(type(error)) as caught:
            await usage.test_live_stream_attaches_reported_usage_idempotently(provider, model)
        assert caught.value is error
    create.assert_awaited_once()
