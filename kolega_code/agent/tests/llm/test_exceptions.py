"""
Tests for the LLM exception classes and mapping functions.
"""

import pytest


# Assuming OpenAIError, GoogleAPIError, AnthropicError can be imported or mocked
# For simplicity, we'll mock them here.
class MockOpenAIError(Exception):
    def __init__(self, message: str, status_code: int):
        super().__init__(message)
        self.status_code = status_code


class MockGoogleAPIError(Exception):
    def __init__(self, message: str, status: int):
        super().__init__(message)
        self.status = status


class MockAnthropicError(Exception):
    def __init__(self, message: str, status_code: int):
        super().__init__(message)
        self.status_code = status_code


from kolega_code.config import ModelProvider
from kolega_code.llm.exceptions import (
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
    map_anthropic_errors,
    map_google_errors,
    map_openai_errors,
)


# Test basic exception instantiation
@pytest.mark.parametrize(
    "exception_class",
    [
        LLMError,
        LLMBadRequestError,
        LLMUnsupportedParamsError,
        LLMContextWindowExceededError,
        LLMContentPolicyViolationError,
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
    original_error = Exception("Generic OpenAI error")  # Mock error without status_code
    # Ensure the base class or a simple Exception can be handled if needed
    # Re-mocking OpenAIError as a simple Exception for this case
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
    original_error = Exception("Generic Google error")  # Mock error without status
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
    original_error = Exception("Generic Anthropic error")  # Mock error without status_code
    mapped_error = map_anthropic_errors(original_error)
    assert isinstance(mapped_error, LLMError)
    assert not isinstance(
        mapped_error,
        (
            LLMInvalidRequestError,
            LLMAuthenticationError,
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
