import asyncio
from collections.abc import AsyncIterator
from functools import partial
import inspect
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx
from openai import APIError, APIStatusError, AsyncOpenAI
import pytest

from kolega_code.agent.baseagent import BaseAgent
from kolega_code.config import ModelProvider
from kolega_code.llm.client import LLMClient
from kolega_code.llm.exceptions import (
    LLMAuthenticationError,
    LLMBillingError,
    LLMContextWindowExceededError,
    LLMError,
    LLMInternalServerError,
    LLMInvalidRequestError,
    LLMNotFoundError,
    LLMPermissionDeniedError,
    LLMRateLimitError,
    billing_error_message,
    map_to_llm_error,
)
from kolega_code.llm.instrumented_client import MinimalLangfuseStreamWrapper
from kolega_code.llm.ledger import UsageLedger
from kolega_code.llm.models import Message, MessageHistory, TextBlock
from kolega_code.llm.providers import openai as transport

pytestmark = pytest.mark.usefixtures("isolated_cli_env")

FAKE_KEY = "abl_refuseless_hermetic_test_not_a_real_key"
MODEL = "deepseek-v4.1-flash-abliterated"
URL = "https://refuseless.invalid/v1/chat/completions"


def _error(body: Any, status: int | None = None, message: str = "request rejected") -> APIError:
    request = httpx.Request("POST", URL)
    if status is None:
        return APIError(message, request=request, body=body)
    return APIStatusError(message, response=httpx.Response(status, request=request), body=body)


@pytest.mark.parametrize(
    "body,status",
    [
        (None, 402),
        ("<html>Payment required</html>", 402),
        ({"error": {"code": "insufficient_credits"}}, 400),
        ({"error": {"type": "insufficient_credits"}}, 429),
        ({"code": "insufficient_credits"}, None),
        ({"type": "insufficient_credits"}, None),
        ({"error": {"code": "insufficient_credits"}}, None),
    ],
)
def test_refuseless_exact_credit_errors_are_billing(body: Any, status: int | None) -> None:
    error = map_to_llm_error(_error(body, status), "refuseless")
    assert isinstance(error, LLMBillingError)
    assert error.provider == "refuseless"
    message = billing_error_message(error, MODEL)
    assert f"Refuseless/{MODEL}" in message
    assert "Add credits" in message


@pytest.mark.parametrize("attribute", ["code", "type"])
def test_refuseless_exact_credit_sdk_attributes_are_billing(attribute: str) -> None:
    raw = _error(None)
    setattr(raw, attribute, "insufficient_credits")
    assert isinstance(map_to_llm_error(raw, "refuseless"), LLMBillingError)


@pytest.mark.parametrize(
    "body",
    [
        None,
        "insufficient_credits",
        {"message": "insufficient_credits"},
        {"error": "insufficient_credits"},
        {"error": {"message": "insufficient credits: add billing balance"}},
        {"code": "not_insufficient_credits"},
        {"code": "insufficient_credits_extra"},
        {"code": "INSUFFICIENT_CREDITS"},
        {"code": ["insufficient_credits"]},
        {"code": {"code": "insufficient_credits"}},
        {"error": {"type": "billing_error"}},
    ],
)
def test_refuseless_does_not_match_messages_or_loose_codes(body: Any) -> None:
    error = map_to_llm_error(_error(body, message="insufficient_credits: payment required billing"), "refuseless")
    assert type(error) is LLMError
    assert error.provider == "refuseless"


@pytest.mark.parametrize("provider", ["openai", "fireworks", "deepseek", "custom:refuseless"])
@pytest.mark.parametrize("status", [None, 402])
def test_refuseless_credit_mapping_is_provider_scoped(provider: str, status: int | None) -> None:
    raw = _error({"error": {"code": "insufficient_credits"}}, status)
    error = map_to_llm_error(raw, provider)
    assert type(error) is LLMError
    assert error.provider == provider


@pytest.mark.parametrize(
    "status,expected",
    [
        (400, LLMInvalidRequestError),
        (401, LLMAuthenticationError),
        (403, LLMPermissionDeniedError),
        (404, LLMNotFoundError),
        (429, LLMRateLimitError),
        (500, LLMInternalServerError),
        (502, LLMInternalServerError),
        (503, LLMInternalServerError),
        (504, LLMInternalServerError),
    ],
)
def test_refuseless_retains_generic_http_errors(status: int, expected: type[LLMError]) -> None:
    raw = _error({"error": {"code": "upstream_error", "message": "not enough credits mentioned in prose"}}, status)
    error = map_to_llm_error(raw, "refuseless")
    assert type(error) is expected
    assert error.provider == "refuseless"


def test_refuseless_cloudflare_html_403_is_permission_denial() -> None:
    error = map_to_llm_error(_error("<html>Cloudflare denied access</html>", 403), "refuseless")
    assert type(error) is LLMPermissionDeniedError


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["generate", "stream"])
async def test_real_sdk_observed_context_overflow_is_not_a_generic_bad_request(
    monkeypatch: pytest.MonkeyPatch, method: str
) -> None:
    fixture = Path(__file__).resolve().parents[2] / "fixtures" / "refuseless" / "context-overflow.json"
    body = json.loads(fixture.read_text())
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(400, json=body)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        monkeypatch.setattr(transport, "AsyncOpenAI", partial(AsyncOpenAI, http_client=http_client))
        client = LLMClient("refuseless", FAKE_KEY, max_retries=3)
        messages = MessageHistory([Message("user", [TextBlock("hello")])])
        with pytest.raises((LLMContextWindowExceededError, APIStatusError)) as caught:
            if method == "generate":
                await client.generate(messages, model="glm-5.3-flash-abliterated")
            else:
                result = client.stream(messages, model="glm-5.3-flash-abliterated")
                assert inspect.isawaitable(result)
                await result
        error = map_to_llm_error(caught.value, "refuseless")
        assert type(error) is LLMContextWindowExceededError
        assert error.provider == "refuseless"
    assert len(requests) == 1


class _SSEBody(httpx.AsyncByteStream):
    def __init__(self, events: list[dict[str, Any]]) -> None:
        self.events = events
        self.closed = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for event in self.events:
            yield f"data: {json.dumps(event)}\n\n".encode()
        yield b"data: [DONE]\n\n"

    async def aclose(self) -> None:
        self.closed = True


def _partial_chunk() -> dict[str, Any]:
    return {
        "id": "chatcmpl-partial",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": MODEL,
        "choices": [{"index": 0, "delta": {"content": "partial"}, "finish_reason": None}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
    }


@pytest.mark.asyncio
async def test_real_sdk_sse_credit_failure_preserves_partial_usage_and_is_terminal(
    monkeypatch: pytest.MonkeyPatch, base_agent: BaseAgent
) -> None:
    wire = _SSEBody([_partial_chunk(), {"error": {"code": "insufficient_credits", "message": "Account exhausted"}}])
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=wire)

    ledger = UsageLedger()
    generation, trace = MagicMock(), MagicMock()
    recorder = SimpleNamespace(_record_usage=AsyncMock())
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        monkeypatch.setattr(transport, "AsyncOpenAI", partial(AsyncOpenAI, http_client=http_client))
        client = LLMClient("refuseless", FAKE_KEY, max_retries=3, model=MODEL, usage_ledger=ledger)
        inner: Any = client.stream(MessageHistory([Message("user", [TextBlock("hello")])]), model=MODEL)
        assert inspect.isawaitable(inner)
        inner = await inner
        stream = MinimalLangfuseStreamWrapper(inner, generation, trace, recorder, MODEL)
        chunks = []
        with pytest.raises(APIError) as caught:
            async with stream:
                async for chunk in stream:
                    chunks.append(chunk)

        raw = caught.value
        assert getattr(raw, "status_code", None) is None
        assert raw.body == {"code": "insufficient_credits", "message": "Account exhausted"}
        assert raw.code == "insufficient_credits"
        assert [chunk.text for chunk in chunks] == ["partial"]
        assert wire.closed
        assert inner.openai_stream.response.is_closed
        assert not inner.get_usage_observation().complete
        assert len(requests) == 1
        snapshot = ledger.snapshot()
        assert snapshot.requests == 1
        assert snapshot.failed == 1
        assert snapshot.responses == 0
        assert snapshot.input_tokens == 10
        assert snapshot.output_tokens == 2
        assert snapshot.total_tokens == 12
        assert generation.update.call_args.kwargs["level"] == "ERROR"
        assert generation.update.call_args.kwargs["status_message"] != "Success"
        assert recorder._record_usage.await_args.kwargs["success"] is False

    base_agent.primary_model_config.provider = ModelProvider.REFUSELESS
    base_agent.primary_model_config.model = MODEL
    status = AsyncMock()
    monkeypatch.setattr(base_agent, "emitter", SimpleNamespace(llm_error=AsyncMock(), llm_status=status))
    warning, sleep = AsyncMock(), AsyncMock()
    monkeypatch.setattr(base_agent, "log_warning", warning)
    monkeypatch.setattr(asyncio, "sleep", sleep)
    with pytest.raises(LLMBillingError) as mapped:
        await base_agent.handle_llm_error(raw)
    assert mapped.value.provider == "refuseless"
    assert base_agent._consecutive_llm_retries == 0
    warning.assert_not_awaited()
    sleep.assert_not_awaited()
    assert status.await_args is not None
    assert status.await_args.args[0] == "error"


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["generate", "stream"])
async def test_real_sdk_http_402_is_not_retried(monkeypatch: pytest.MonkeyPatch, method: str) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(402, json={"error": {"code": "insufficient_credits", "message": "Add credits"}})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        monkeypatch.setattr(transport, "AsyncOpenAI", partial(AsyncOpenAI, http_client=http_client))
        client = LLMClient("refuseless", FAKE_KEY, max_retries=3)
        messages = MessageHistory([Message("user", [TextBlock("hello")])])
        with pytest.raises((LLMBillingError, APIStatusError)) as caught:
            if method == "generate":
                await client.generate(messages, model=MODEL)
            else:
                result = client.stream(messages, model=MODEL)
                assert inspect.isawaitable(result)
                await result
        error = map_to_llm_error(caught.value, "refuseless")
        assert isinstance(error, LLMBillingError)
        assert error.provider == "refuseless"
    assert len(requests) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("exit_mode", ["unconsumed", "break", "cancel"])
async def test_real_sdk_stream_closes_on_early_exit(monkeypatch: pytest.MonkeyPatch, exit_mode: str) -> None:
    wire = _SSEBody([_partial_chunk(), _partial_chunk()])

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=wire)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        monkeypatch.setattr(transport, "AsyncOpenAI", partial(AsyncOpenAI, http_client=http_client))
        provider = transport.OpenAIProvider(
            FAKE_KEY, provider_name="refuseless", base_url="https://refuseless.invalid/v1"
        )
        stream: Any = await provider.stream(MessageHistory([]), model=MODEL)

        async def consume() -> None:
            async with stream:
                assert not wire.closed
                if exit_mode == "unconsumed":
                    return
                async for _ in stream:
                    if exit_mode == "cancel":
                        raise asyncio.CancelledError
                    break

        if exit_mode == "cancel":
            with pytest.raises(asyncio.CancelledError):
                await consume()
        else:
            await consume()
        assert wire.closed
        assert stream.openai_stream.response.is_closed
