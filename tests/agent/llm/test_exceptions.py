"""
Tests for the LLM exception classes and mapping functions.
"""

import pytest
from anthropic import AnthropicError
from google.genai.errors import APIError as GoogleAPIError
from openai import OpenAIError

from kolega_code.config import ModelProvider
from kolega_code.llm.exceptions import (
    LLMBillingError,
    LLMAuthenticationError,
    LLMBadRequestError,
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
    LLMUnsupportedParamsError,
    llm_error_message,
    map_to_llm_error,
    map_anthropic_errors,
    map_google_errors,
    map_openai_errors,
)


# Assuming OpenAIError, GoogleAPIError, AnthropicError can be imported or mocked
# For simplicity, we'll mock them here.
class MockOpenAIError(OpenAIError):
    def __init__(self, message: str, status_code: int, code: str | None = None):
        super().__init__(message)
        self.status_code = status_code
        self.code = code


class MockGoogleAPIError(GoogleAPIError):
    status: int | None  # overrides GoogleAPIError.status (str) with the int code used by tests

    def __init__(self, message: str, status: int | None = None):
        # Bypass GoogleAPIError's complex constructor; emulate a status-bearing error.
        Exception.__init__(self, message)
        self.status = status


class MockAnthropicError(AnthropicError):
    def __init__(self, message: str, status_code: int):
        super().__init__(message)
        self.status_code = status_code


# Test basic exception instantiation
@pytest.mark.parametrize(
    "exception_class",
    [
        LLMError,
        LLMBadRequestError,
        LLMUnsupportedParamsError,
        LLMContextWindowExceededError,
        LLMContentPolicyViolationError,
        LLMBillingError,
        LLMInvalidRequestError,
        LLMAuthenticationError,
        LLMPermissionDeniedError,
        LLMNotFoundError,
        LLMTimeout,
        LLMUnprocessableEntityError,
        LLMRateLimitError,
        LLMInternalServerError,
    ],
)
def test_llm_exception_instantiation(exception_class):
    """Test that each LLM exception can be instantiated."""
    message = "Test error message"
    provider = "test_provider"
    error = exception_class(message, provider=provider)
    assert isinstance(error, LLMError)  # Check inheritance
    assert isinstance(error, Exception)
    assert str(error) == message
    assert error.provider == provider


# Test OpenAI error mapping
@pytest.mark.parametrize(
    "status_code, expected_exception",
    [
        (400, LLMInvalidRequestError),
        (401, LLMAuthenticationError),
        (403, LLMPermissionDeniedError),
        (404, LLMNotFoundError),
        (422, LLMUnprocessableEntityError),
        (429, LLMRateLimitError),
        (500, LLMInternalServerError),
        (999, LLMError),  # Test default case
    ],
)
def test_map_openai_errors(status_code, expected_exception):
    """Test the mapping of OpenAI error status codes to LLM exceptions."""
    original_error = MockOpenAIError("OpenAI test error", status_code=status_code)
    mapped_error = map_openai_errors(original_error)
    assert isinstance(mapped_error, expected_exception)
    # Check provider is set correctly - should always be OPENAI
    assert mapped_error.provider == ModelProvider.OPENAI.value
    assert "OpenAI APIError:" in str(mapped_error)


def test_map_openai_errors_no_status_code():
    """Test mapping OpenAI errors without a status code."""
    original_error = OpenAIError("Generic OpenAI error")  # error without status_code
    mapped_error = map_openai_errors(original_error)
    assert isinstance(mapped_error, LLMError)
    assert not isinstance(
        mapped_error,
        (
            LLMInvalidRequestError,
            LLMAuthenticationError,
            LLMPermissionDeniedError,
            LLMNotFoundError,
            LLMUnprocessableEntityError,
            LLMRateLimitError,
            LLMInternalServerError,
        ),
    )  # Should be the base LLMError
    assert mapped_error.provider == ModelProvider.OPENAI.value
    assert "OpenAI APIError:" in str(mapped_error)


def test_map_openai_context_length_exceeded():
    """A 400 with code="context_length_exceeded" maps to LLMContextWindowExceededError."""
    original_error = MockOpenAIError(
        "Error code: 400 - {'error': {'code': 'context_length_exceeded', ...}}",
        status_code=400,
        code="context_length_exceeded",
    )
    mapped_error = map_openai_errors(original_error)
    assert isinstance(mapped_error, LLMContextWindowExceededError)
    assert mapped_error.provider == ModelProvider.OPENAI.value


# Test Google error mapping
@pytest.mark.parametrize(
    "status, expected_exception",
    [
        (400, LLMInvalidRequestError),
        (403, LLMPermissionDeniedError),
        (429, LLMRateLimitError),
        (500, LLMInternalServerError),
        (999, LLMError),  # Test default case
    ],
)
def test_map_google_errors(status, expected_exception):
    """Test the mapping of Google error statuses to LLM exceptions."""
    original_error = MockGoogleAPIError("Google test error", status=status)
    mapped_error = map_google_errors(original_error)
    assert isinstance(mapped_error, expected_exception)
    assert mapped_error.provider == ModelProvider.GOOGLE.value
    if status in [400, 403, 429, 500]:
        assert "GoogleAPIError:" in str(mapped_error)
    else:
        assert "Google APIError:" in str(mapped_error)  # Note the subtle difference in the default message


def test_map_google_errors_no_status():
    """Test mapping Google errors without a status attribute."""
    original_error = MockGoogleAPIError("Generic Google error")  # error without a matching status
    mapped_error = map_google_errors(original_error)
    assert isinstance(mapped_error, LLMError)
    assert not isinstance(
        mapped_error, (LLMInvalidRequestError, LLMPermissionDeniedError, LLMRateLimitError, LLMInternalServerError)
    )  # Should be the base LLMError
    assert mapped_error.provider == ModelProvider.GOOGLE.value
    assert "Google APIError:" in str(mapped_error)


# Test Anthropic error mapping
@pytest.mark.parametrize(
    "status_code, expected_exception",
    [
        (400, LLMInvalidRequestError),
        (401, LLMAuthenticationError),
        (402, LLMBillingError),
        (403, LLMPermissionDeniedError),
        (404, LLMNotFoundError),
        (413, LLMContextWindowExceededError),
        (429, LLMRateLimitError),
        (500, LLMInternalServerError),
        (529, LLMInternalServerError),
        (999, LLMError),  # Test default case
    ],
)
def test_map_anthropic_errors(status_code, expected_exception):
    """Test the mapping of Anthropic error status codes to LLM exceptions."""
    original_error = MockAnthropicError("Anthropic test error", status_code=status_code)
    mapped_error = map_anthropic_errors(original_error)
    assert isinstance(mapped_error, expected_exception)
    assert mapped_error.provider == ModelProvider.ANTHROPIC.value
    assert "AnthropicError:" in str(mapped_error)


def test_map_anthropic_errors_no_status_code():
    """Test mapping Anthropic errors without a status code."""
    original_error = AnthropicError("Generic Anthropic error")  # error without status_code
    mapped_error = map_anthropic_errors(original_error)
    assert isinstance(mapped_error, LLMError)
    assert not isinstance(
        mapped_error,
        (
            LLMInvalidRequestError,
            LLMAuthenticationError,
            LLMBillingError,
            LLMPermissionDeniedError,
            LLMNotFoundError,
            LLMContextWindowExceededError,
            LLMRateLimitError,
            LLMInternalServerError,
        ),
    )  # Should be the base LLMError
    assert mapped_error.provider == ModelProvider.ANTHROPIC.value
    assert "AnthropicError:" in str(mapped_error)


def test_map_anthropic_api_status_error_invalid_request():
    """Ensure Anthropic APIStatusError with invalid_request_error maps to LLMContentPolicyViolationError when message indicates content filtering."""
    import httpx
    from anthropic import APIStatusError

    # Build a minimal httpx.Response to satisfy APIStatusError constructor
    response = httpx.Response(status_code=400, request=httpx.Request("POST", "https://api.anthropic.com/v1/messages"))
    body = {
        "type": "error",
        "error": {
            "details": None,
            "type": "invalid_request_error",
            "message": "Output blocked by content filtering policy",
        },
    }

    err = APIStatusError("invalid request", response=response, body=body)

    from kolega_code.llm.exceptions import map_anthropic_errors, LLMContentPolicyViolationError

    mapped = map_anthropic_errors(err)
    assert isinstance(mapped, LLMContentPolicyViolationError)
    assert mapped.provider == ModelProvider.ANTHROPIC.value
    assert "AnthropicError:" in str(mapped)


def test_map_anthropic_api_status_error_token_limit():
    import httpx
    from anthropic import APIStatusError

    response = httpx.Response(status_code=400, request=httpx.Request("POST", "https://api.anthropic.com/v1/messages"))
    body = {
        "type": "error",
        "error": {
            "type": "invalid_request_error",
            "message": "Invalid request: Your request exceeded model token limit: 262144 (requested: 1348145)",
        },
    }

    err = APIStatusError("invalid request", response=response, body=body)

    mapped = map_anthropic_errors(err)
    assert isinstance(mapped, LLMContextWindowExceededError)
    assert mapped.provider == ModelProvider.ANTHROPIC.value


def test_map_deepseek_anthropic_api_status_error_insufficient_balance():
    import httpx
    from anthropic import APIStatusError

    response = httpx.Response(
        status_code=402,
        request=httpx.Request("POST", "https://api.deepseek.com/anthropic/v1/messages"),
    )
    body = {
        "error": {
            "message": "Insufficient Balance",
            "type": "unknown_error",
            "param": None,
            "code": "invalid_request_error",
        },
    }

    err = APIStatusError("payment required", response=response, body=body)

    mapped = map_to_llm_error(err, provider=ModelProvider.DEEPSEEK.value)
    assert isinstance(mapped, LLMBillingError)
    assert mapped.provider == ModelProvider.DEEPSEEK.value
    assert "AnthropicError:" in str(mapped)


def test_llm_error_message_surfaces_provider_400_detail() -> None:
    error = LLMInvalidRequestError(
        "OpenAI APIError: Error code: 400 - {'error': {'message': \"Invalid 'tools': tool name "
        "'mcp__docs__get.file' does not match '^[a-zA-Z0-9_-]{1,64}$'\"}}",
        provider=ModelProvider.OPENAI.value,
    )
    message = llm_error_message(error, model="gpt-x")
    assert message.startswith("OpenAI/gpt-x could not process this request: ")
    detail = message.split("could not process this request: ", 1)[1]
    assert "OpenAI APIError:" not in detail
    assert "Error code: 400" in detail
    assert "does not match '^[a-zA-Z0-9_-]{1,64}$'" in detail


def test_llm_error_message_collapses_and_truncates_long_provider_detail() -> None:
    error = LLMInvalidRequestError(
        "OpenAI APIError: Error code: 400 - " + "word " * 500,
        provider=ModelProvider.OPENAI.value,
    )
    message = llm_error_message(error, model="gpt-x")
    detail = message.split("could not process this request: ", 1)[1]
    assert detail.endswith("…")
    assert len(detail) <= 301
    assert "\n" not in detail


def test_llm_error_message_falls_back_to_generic_copy_without_detail() -> None:
    error = LLMInvalidRequestError("", provider=ModelProvider.OPENAI.value)
    message = llm_error_message(error, model="gpt-x")
    assert message == ("OpenAI/gpt-x could not process this request. Check the selected provider/model and try again.")


def test_llm_error_message_other_branches_unchanged() -> None:
    billing = llm_error_message(LLMBillingError("OpenAI APIError: no credits", provider=ModelProvider.OPENAI.value))
    assert "insufficient balance" in billing
    context = llm_error_message(
        LLMContextWindowExceededError("OpenAI APIError: context", provider=ModelProvider.OPENAI.value)
    )
    assert "context became too large" in context


def test_map_to_llm_error_threads_provider_through_openai_mapping() -> None:
    """Compatible providers riding the OpenAI SDK keep their own provider label."""
    original_error = MockOpenAIError("no access", status_code=403)
    mapped_error = map_to_llm_error(original_error, "perplexity_agent")
    assert isinstance(mapped_error, LLMPermissionDeniedError)
    assert mapped_error.provider == "perplexity_agent"
    assert llm_error_message(mapped_error, model="perplexity/sonar").startswith("Perplexity/perplexity/sonar")


def test_map_to_llm_error_threads_provider_through_google_mapping() -> None:
    original_error = MockGoogleAPIError("quota", status=429)
    mapped_error = map_to_llm_error(original_error, "google")
    assert isinstance(mapped_error, LLMRateLimitError)
    assert mapped_error.provider == "google"


def test_map_openai_errors_defaults_to_openai_without_provider() -> None:
    original_error = MockOpenAIError("nope", status_code=401)
    assert map_openai_errors(original_error).provider == ModelProvider.OPENAI.value
