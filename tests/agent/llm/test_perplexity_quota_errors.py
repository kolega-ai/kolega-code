from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock

import httpx
import pytest
from openai import APIError, APIStatusError, AsyncOpenAI, AuthenticationError, RateLimitError

from kolega_code.llm.client import LLMClient
from kolega_code.llm.exceptions import (
    LLMBillingError,
    LLMAuthenticationError,
    LLMError,
    LLMInvalidRequestError,
    LLMNotFoundError,
    LLMPermissionDeniedError,
    LLMRateLimitError,
    is_perplexity_quota_error,
    map_openai_errors,
    map_to_llm_error,
)
from kolega_code.llm.models import Message, MessageHistory, TextBlock
from kolega_code.llm.providers.perplexity_responses import PerplexityResponsesProvider

PROVIDER = "perplexity_agent"
MODEL = "perplexity/sonar"
QUOTA_MESSAGE = "You have exceeded your API quota. Please check your plan and billing details."


def _sdk_error(status: int | None, body: Any, message: str = "Request failed") -> APIError:
    request = httpx.Request("POST", "https://api.perplexity.ai/v1/responses")
    if status is None:
        return APIError(message, request=request, body=body)
    error_class = {401: AuthenticationError, 429: RateLimitError}.get(status, APIStatusError)
    return error_class(message, response=httpx.Response(status, request=request), body=body)


@pytest.mark.parametrize(
    ("status", "body", "message"),
    [
        (401, {"error": {"message": QUOTA_MESSAGE}}, "Unauthorized"),
        (401, {"error": {"code": "insufficient_quota"}}, "Unauthorized"),
        (429, {"error": {"type": "insufficient_quota"}}, "Request failed"),
        (402, {"code": "insufficient_credits"}, "Request failed"),
        (None, {"error": {"code": "quota_exceeded"}}, "Request failed"),
        (None, {"type": "quota_exhausted"}, "Request failed"),
        (401, {"error": {"message": "Insufficient credits. Please add credits."}}, "Unauthorized"),
        (401, {"message": "Your account has run out of credits."}, "Unauthorized"),
        (401, {"error": "You have exceeded your current quota."}, "Unauthorized"),
        (401, "You exceeded your current quota, please check your plan and billing details.", "Unauthorized"),
        (401, None, QUOTA_MESSAGE),
        (401, {"error": {"message": "  QUOTA EXCEEDED. Add credits.  "}}, "Unauthorized"),
    ],
)
def test_explicit_perplexity_quota_maps_to_billing(status: int | None, body: Any, message: str) -> None:
    error = _sdk_error(status, body, message)
    assert is_perplexity_quota_error(error, PROVIDER)
    for mapped in (map_openai_errors(error, PROVIDER), map_to_llm_error(error, PROVIDER)):
        assert type(mapped) is LLMBillingError
        assert mapped.provider == PROVIDER
        assert str(error) in str(mapped)


NEGATIVE_CASES = [
    (401, {"error": {"code": "invalid_api_key", "message": "Invalid API key."}}, LLMAuthenticationError),
    (401, {"error": {"message": "Unauthorized"}}, LLMAuthenticationError),
    (401, {"error": {"message": "Check your billing details."}}, LLMAuthenticationError),
    (401, {"error": {"message": "Check your API quota and billing details."}}, LLMAuthenticationError),
    (401, "Unauthorized", LLMAuthenticationError),
    (400, {"error": {"message": "Missing required max_output_tokens"}}, LLMInvalidRequestError),
    (400, {"error": {"message": "Unknown item type 'search_results'"}}, LLMInvalidRequestError),
    (400, {"error": {"message": "Invalid billing parameter"}}, LLMInvalidRequestError),
    (400, {"error": {"message": "Invalid 'input': contains 'quota exceeded'"}}, LLMInvalidRequestError),
    (400, {"input": QUOTA_MESSAGE, "error": {"message": "Malformed input"}}, LLMInvalidRequestError),
    (403, {"error": {"message": "This model requires a paid plan."}}, LLMPermissionDeniedError),
    (404, {"error": {"message": "Model not found"}}, LLMNotFoundError),
    (429, {"error": {"code": "rate_limit_exceeded", "message": "Too many requests"}}, LLMRateLimitError),
    (418, {"error": {"code": "unknown_error", "message": "Unknown failure"}}, LLMError),
    (None, {"error": {"message": "Unknown streaming error"}}, LLMError),
    (None, {"error": ["insufficient_quota"]}, LLMError),
]


@pytest.mark.parametrize(("status", "body", "expected"), NEGATIVE_CASES)
def test_non_quota_errors_keep_their_mapping(status: int | None, body: Any, expected: type[LLMError]) -> None:
    error = _sdk_error(status, body)
    assert not is_perplexity_quota_error(error, PROVIDER)
    mapped = map_openai_errors(error, PROVIDER)
    assert type(mapped) is expected
    assert mapped.provider == PROVIDER


@pytest.mark.parametrize("provider", [None, "openai", "deepseek", "openrouter", "cloudflare_workers_ai"])
@pytest.mark.parametrize("body", [{"error": {"code": "insufficient_quota"}}, {"error": {"message": QUOTA_MESSAGE}}])
def test_other_providers_authentication_is_unchanged(provider: str | None, body: dict[str, Any]) -> None:
    error = _sdk_error(401, body)
    assert not is_perplexity_quota_error(error, provider)
    mapped = map_openai_errors(error, provider)
    assert type(mapped) is LLMAuthenticationError
    assert mapped.provider == (provider or "openai")


@pytest.mark.parametrize(
    "error", [ValueError(QUOTA_MESSAGE), RuntimeError(QUOTA_MESSAGE), LLMBillingError(QUOTA_MESSAGE)]
)
def test_non_sdk_errors_do_not_qualify(error: Exception) -> None:
    assert not is_perplexity_quota_error(error, PROVIDER)


def test_sdk_error_attributes_without_a_body() -> None:
    error = _sdk_error(None, None)
    error.code = "insufficient_quota"
    assert is_perplexity_quota_error(error, PROVIDER)
    assert type(map_openai_errors(error, PROVIDER)) is LLMBillingError


class _FailingStream:
    def __init__(self, error: Exception) -> None:
        self.error = error
        self.closed = False

    def __aiter__(self) -> AsyncIterator[Any]:
        return self

    async def __anext__(self) -> Any:
        raise self.error

    async def aclose(self) -> None:
        self.closed = True


def _fake_transport(client: LLMClient, error: Exception, stage: str) -> tuple[AsyncMock, _FailingStream]:
    assert isinstance(client.provider, PerplexityResponsesProvider)
    stream = _FailingStream(error)
    create = AsyncMock(side_effect=error) if stage == "create" else AsyncMock(return_value=stream)
    client.provider.async_client = cast(AsyncOpenAI, SimpleNamespace(responses=SimpleNamespace(create=create)))
    return create, stream


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["create", "iterate"])
@pytest.mark.parametrize(
    ("status", "body", "expected"),
    [
        (401, {"error": {"message": QUOTA_MESSAGE}}, LLMBillingError),
        (None, {"error": {"code": "insufficient_quota"}}, LLMBillingError),
        *NEGATIVE_CASES,
    ],
)
async def test_normalized_generate_with_fake_transport(
    stage: str, status: int | None, body: Any, expected: type[LLMError]
) -> None:
    client = LLMClient(provider=PROVIDER, api_key="pplx-offline-fake", model=MODEL)
    error = _sdk_error(status, body)
    create, stream = _fake_transport(client, error, stage)
    history = MessageHistory([Message(role="user", content=[TextBlock(text="Hello")])])
    with pytest.raises(expected) as caught:
        await client.generate(history)
    assert type(caught.value) is expected
    assert caught.value.provider == PROVIDER
    assert caught.value.__cause__ is error
    create.assert_awaited_once()
    if stage == "iterate":
        assert stream.closed


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error", [ValueError(QUOTA_MESSAGE), AssertionError("Missing search_results"), RuntimeError("bug")]
)
async def test_live_probe_does_not_hide_non_sdk_failures(monkeypatch: pytest.MonkeyPatch, error: Exception) -> None:
    from tests.agent.llm import test_perplexity_live as live

    monkeypatch.setattr(live, "SKIP_IN_CI", False)
    monkeypatch.setattr(live, "_probe_server_tool_search_results_and_replay", AsyncMock(side_effect=error))
    with pytest.raises(type(error)) as caught:
        await live.test_live_server_tool_search_results_captured_and_replayable()
    assert caught.value is error


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["create", "iterate"])
@pytest.mark.parametrize(
    ("status", "body", "expected"),
    [
        (401, {"error": {"message": QUOTA_MESSAGE}}, LLMBillingError),
        (None, {"error": {"code": "insufficient_quota"}}, LLMBillingError),
        *NEGATIVE_CASES,
    ],
)
async def test_live_probe_skips_only_explicit_quota_with_fake_transport(
    monkeypatch: pytest.MonkeyPatch, stage: str, status: int | None, body: Any, expected: type[LLMError]
) -> None:
    from tests.agent.llm import test_perplexity_live as live

    client = LLMClient(provider=PROVIDER, api_key="pplx-offline-fake", model=MODEL)
    error = _sdk_error(status, body)
    create, stream = _fake_transport(client, error, stage)
    monkeypatch.setattr(live, "SKIP_IN_CI", False)
    monkeypatch.setattr(live, "_provider", lambda: client.provider)
    expected_error = pytest.skip.Exception if expected is LLMBillingError else APIError
    with pytest.raises(expected_error) as caught:
        await live.test_live_server_tool_search_results_captured_and_replayable()
    if expected is LLMBillingError:
        assert "Perplexity quota exhausted" in str(caught.value)
    else:
        assert caught.value is error
    create.assert_awaited_once()
    if stage == "iterate":
        assert stream.closed
