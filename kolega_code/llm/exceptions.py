"""
This module defines custom exception classes for handling errors related to
Large Language Model (LLM) interactions across different providers. It provides a
base `LLMError` class and specific subclasses for common error scenarios.
Additionally, it includes utility functions to map provider-specific errors
(OpenAI, Google, Anthropic) to these standardized exceptions.

Error Mapping:

| Provider  | Status Code | Mapped LLMError                |
|-----------|-------------|--------------------------------|
| OpenAI    | 400         | `LLMInvalidRequestError`       |
| OpenAI    | 401         | `LLMAuthenticationError`       |
| OpenAI    | 403         | `LLMPermissionDeniedError`     |
| OpenAI    | 404         | `LLMNotFoundError`             |
| OpenAI    | 422         | `LLMUnprocessableEntityError`  |
| OpenAI    | 429         | `LLMRateLimitError`            |
| OpenAI    | 500         | `LLMInternalServerError`       |
| OpenAI    | Other       | `LLMError`                     |
| Google    | 400         | `LLMInvalidRequestError`       |
| Google    | 403         | `LLMPermissionDeniedError`     |
| Google    | 429         | `LLMRateLimitError`            |
| Google    | 500         | `LLMInternalServerError`       |
| Google    | Other       | `LLMError`                     |
| Anthropic | 400         | `LLMInvalidRequestError`       |
| Anthropic | 401         | `LLMAuthenticationError`       |
| Anthropic | 403         | `LLMPermissionDeniedError`     |
| Anthropic | 404         | `LLMNotFoundError`             |
| Anthropic | 402         | `LLMBillingError`              |
| Anthropic | 413         | `LLMContextWindowExceededError`|
| Anthropic | 429         | `LLMRateLimitError`            |
| Anthropic | 500         | `LLMInternalServerError`       |
| Anthropic | 529         | `LLMInternalServerError`       |
| Anthropic | Other       | `LLMError`                     |

"""

import asyncio

from anthropic import (
    AnthropicError,
    APIStatusError as AnthropicAPIStatusError,
    InternalServerError as AnthropicInternalServerError,
)
from google.genai.errors import APIError as GoogleAPIError
from openai import OpenAIError

try:
    import aiohttp
except ImportError:
    aiohttp = None  # Handle case where aiohttp not installed

# Handle httpx being optional. If installed, we will map certain httpx errors.
try:
    import httpx
except ImportError:
    httpx = None

from ..config import ModelProvider


class LLMError(Exception):
    """Base exception class for all LLM-related errors."""

    def __init__(self, message: str, model: str = None, provider: str = None):
        super().__init__(message)
        self.provider = provider


class LLMBadRequestError(LLMError):
    """Raised when the request to the LLM service is malformed or invalid."""


class LLMUnsupportedParamsError(LLMError):
    """Raised when unsupported parameters are provided to the LLM service."""


class LLMContextWindowExceededError(LLMError):
    """Raised when the input exceeds the model's maximum context window size."""


class LLMContentPolicyViolationError(LLMError):
    """Raised when the request violates the LLM provider's content policy."""


class LLMInvalidRequestError(LLMError):
    """Raised when the request is invalid for reasons other than malformed data."""


class LLMAuthenticationError(LLMError):
    """Raised when authentication with the LLM service fails."""


class LLMPermissionDeniedError(LLMError):
    """Raised when the authenticated user lacks permission for the requested operation."""


class LLMNotFoundError(LLMError):
    """Raised when the requested resource is not found."""


class LLMTimeout(LLMError):
    """Raised when the request to the LLM service times out."""


class LLMUnprocessableEntityError(LLMError):
    """Raised when the request is well-formed but cannot be processed."""


class LLMRateLimitError(LLMError):
    """Raised when the rate limit for the LLM service is exceeded."""


class LLMBillingError(LLMError):
    """Raised when the LLM provider rejects a request due to billing or credits."""


class LLMInternalServerError(LLMError):
    """Raised when the LLM service encounters an internal error."""


_BILLING_ERROR_PHRASES = (
    "insufficient balance",
    "insufficient credit",
    "insufficient credits",
    "payment required",
    "billing",
)


def _provider_display_name(provider: str | None) -> str:
    provider_names = {
        ModelProvider.ANTHROPIC.value: "Anthropic",
        ModelProvider.OPENAI.value: "OpenAI",
        ModelProvider.GOOGLE.value: "Google",
        ModelProvider.GROQ.value: "Groq",
        ModelProvider.TOGETHER.value: "Together",
        ModelProvider.FIREWORKS.value: "Fireworks",
        ModelProvider.XAI.value: "xAI",
        ModelProvider.DASHSCOPE.value: "DashScope",
        ModelProvider.MOONSHOT.value: "Moonshot",
        ModelProvider.DEEPSEEK.value: "DeepSeek",
        ModelProvider.LLAMA.value: "Llama",
    }
    if not provider:
        return "The selected provider"
    return provider_names.get(provider, provider.replace("_", " ").replace("-", " ").title())


def billing_error_message(error: LLMBillingError, model: str | None = None) -> str:
    """Return a concise user-facing message for provider billing failures."""
    provider = error.provider
    provider_name = _provider_display_name(provider)
    model_label = f"/{model}" if model else ""
    provider_model = f"{provider_name}{model_label}"

    return (
        f"{provider_model} could not run this request because {provider_name} reported insufficient balance. "
        f"Add credits to your {provider_name} account or switch to another provider/model in Settings or with /model."
    )


def llm_error_message(error: LLMError, model: str | None = None) -> str:
    """Return concise user-facing copy for terminal LLM failures."""
    if isinstance(error, LLMBillingError):
        return billing_error_message(error, model=model)

    provider_name = _provider_display_name(error.provider)
    model_label = f"/{model}" if model else ""
    provider_model = f"{provider_name}{model_label}"

    if isinstance(error, LLMContextWindowExceededError):
        return (
            "The conversation context became too large for the model. "
            "Oversized tool output is trimmed automatically; please retry the message."
        )

    if isinstance(error, LLMInternalServerError):
        return "There is high traffic on our LLM provider right now. Please try again in a few seconds."

    if isinstance(error, LLMAuthenticationError):
        return f"{provider_model} could not authenticate. Check the API key in Settings or your environment."

    if isinstance(error, LLMPermissionDeniedError):
        return f"{provider_model} rejected this request because the API key does not have access."

    if isinstance(error, LLMNotFoundError):
        return f"{provider_model} was not found. Check the selected provider/model in Settings or with /model."

    if isinstance(error, LLMTimeout):
        return f"{provider_model} timed out while processing this request. Please try again."

    if isinstance(error, LLMContentPolicyViolationError):
        return f"{provider_model} blocked this request due to the provider's content policy."

    if isinstance(
        error,
        (
            LLMBadRequestError,
            LLMUnsupportedParamsError,
            LLMInvalidRequestError,
            LLMUnprocessableEntityError,
        ),
    ):
        return f"{provider_model} could not process this request. Check the selected provider/model and try again."

    return f"{provider_model} returned an error: {error}"


def _is_billing_status(error: Exception) -> bool:
    return getattr(error, "status_code", None) == 402 or getattr(error, "status", None) == 402


def _is_billing_message(message: str) -> bool:
    message_lower = message.lower()
    return any(phrase in message_lower for phrase in _BILLING_ERROR_PHRASES)


def _anthropic_body_error_message(error: Exception) -> str:
    body = getattr(error, "body", None)
    if not isinstance(body, dict):
        return ""
    error_info = body.get("error")
    if not isinstance(error_info, dict):
        return ""
    return str(error_info.get("message") or "").strip()


def map_openai_errors(error: OpenAIError) -> LLMError:
    if hasattr(error, "status_code"):
        if error.status_code == 400:
            return LLMInvalidRequestError(message=f"OpenAI APIError: {str(error)}", provider=ModelProvider.OPENAI.value)
        elif error.status_code == 401:
            return LLMAuthenticationError(message=f"OpenAI APIError: {str(error)}", provider=ModelProvider.OPENAI.value)
        elif error.status_code == 403:
            return LLMPermissionDeniedError(
                message=f"OpenAI APIError: {str(error)}", provider=ModelProvider.OPENAI.value
            )
        elif error.status_code == 404:
            return LLMNotFoundError(message=f"OpenAI APIError: {str(error)}", provider=ModelProvider.OPENAI.value)
        elif error.status_code == 422:
            return LLMUnprocessableEntityError(
                message=f"OpenAI APIError: {str(error)}", provider=ModelProvider.OPENAI.value
            )
        elif error.status_code == 429:
            return LLMRateLimitError(message=f"OpenAI APIError: {str(error)}", provider=ModelProvider.OPENAI.value)
        elif error.status_code == 500:
            return LLMInternalServerError(message=f"OpenAI APIError: {str(error)}", provider=ModelProvider.OPENAI.value)

    return LLMError(message=f"OpenAI APIError: {str(error)}", provider=ModelProvider.OPENAI.value)


def map_google_errors(error: GoogleAPIError) -> LLMError:
    if hasattr(error, "status"):

        if error.status == 400:
            return LLMInvalidRequestError(message=f"GoogleAPIError: {str(error)}", provider=ModelProvider.GOOGLE.value)
        elif error.status == 403:
            return LLMPermissionDeniedError(
                message=f"GoogleAPIError: {str(error)}", provider=ModelProvider.GOOGLE.value
            )
        elif error.status == 429:
            return LLMRateLimitError(message=f"GoogleAPIError: {str(error)}", provider=ModelProvider.GOOGLE.value)
        elif error.status == 500:
            return LLMInternalServerError(message=f"GoogleAPIError: {str(error)}", provider=ModelProvider.GOOGLE.value)

    return LLMError(message=f"Google APIError: {str(error)}", provider=ModelProvider.GOOGLE.value)


def map_anthropic_errors(error: AnthropicError, provider: str | None = None) -> LLMError:
    provider = provider or ModelProvider.ANTHROPIC.value
    context_window_phrases = (
        "exceeded model token limit",
        "context window",
        "maximum context length",
        "prompt is too long",
    )

    if (
        _is_billing_status(error)
        or _is_billing_message(_anthropic_body_error_message(error))
        or _is_billing_message(str(error))
    ):
        return LLMBillingError(message=f"AnthropicError: {str(error)}", provider=provider)

    if type(error) is AnthropicAPIStatusError:
        try:
            error_data = error.body
            if isinstance(error_data, dict) and "error" in error_data:
                error_info = error_data["error"]
                error_type = error_info.get("type")
                error_message = (error_info.get("message") or "").strip()
                error_message_lower = error_message.lower()

                # Keep internal/server overload handling by type
                if error_type in ["overloaded_error", "api_error"]:
                    return LLMInternalServerError(
                        message=f"AnthropicError: {str(error)}", provider=provider
                    )

                if any(phrase in error_message_lower for phrase in context_window_phrases):
                    return LLMContextWindowExceededError(
                        message=f"AnthropicError: {str(error)}", provider=provider
                    )

                # Special case: content filtering block should be mapped to content policy violation
                if (
                    error_type == "invalid_request_error"
                    and error_message == "Output blocked by content filtering policy"
                ):
                    return LLMContentPolicyViolationError(
                        message=f"AnthropicError: {str(error)}", provider=provider
                    )

        except Exception:
            pass

    if type(error) is AnthropicInternalServerError:
        return LLMInternalServerError(message=f"AnthropicError: {str(error)}", provider=provider)

    if hasattr(error, "status_code"):
        if error.status_code == 400:
            error_text = str(error).lower()
            if any(phrase in error_text for phrase in context_window_phrases):
                return LLMContextWindowExceededError(
                    message=f"AnthropicError: {str(error)}", provider=provider
                )
            return LLMInvalidRequestError(
                message=f"AnthropicError: {str(error)}", provider=provider
            )
        elif error.status_code == 401:
            return LLMAuthenticationError(
                message=f"AnthropicError: {str(error)}", provider=provider
            )
        elif error.status_code == 403:
            return LLMPermissionDeniedError(
                message=f"AnthropicError: {str(error)}", provider=provider
            )
        elif error.status_code == 404:
            return LLMNotFoundError(message=f"AnthropicError: {str(error)}", provider=provider)
        elif error.status_code == 413:
            return LLMContextWindowExceededError(
                message=f"AnthropicError: {str(error)}", provider=provider
            )
        if error.status_code == 429:
            return LLMRateLimitError(message=f"AnthropicError: {str(error)}", provider=provider)
        if error.status_code == 500:
            return LLMInternalServerError(
                message=f"AnthropicError: {str(error)}", provider=provider
            )
        if error.status_code == 529:
            return LLMInternalServerError(
                message=f"AnthropicError: {str(error)}", provider=provider
            )

    return LLMError(message=f"AnthropicError: {str(error)}", provider=provider)


def map_to_llm_error(error: Exception, provider: str = None) -> LLMError:
    """Map any exception to a standardized LLM error.

    This function provides a single point of control for converting any exception
    type into an appropriate LLMError subclass. It ensures that only LLMError
    exceptions escape from the LLM client layer.

    Args:
        error: The exception to map
        provider: Optional provider name for context

    Returns:
        An appropriate LLMError subclass instance
    """
    # If already an LLM error, return as-is
    if isinstance(error, LLMError):
        return error

    # Map provider-specific errors using existing functions
    if isinstance(error, OpenAIError):
        return map_openai_errors(error)
    elif isinstance(error, GoogleAPIError):
        return map_google_errors(error)
    elif isinstance(error, AnthropicError):
        return map_anthropic_errors(error, provider=provider)

    # Map common Python exceptions
    if isinstance(error, ValueError):
        return LLMInvalidRequestError(message=f"Invalid parameter: {str(error)}", provider=provider)
    elif isinstance(error, (TimeoutError, asyncio.TimeoutError)):
        return LLMTimeout(message=f"Request timeout: {str(error)}", provider=provider)
    elif isinstance(error, ConnectionError):
        return LLMInternalServerError(message=f"Connection error: {str(error)}", provider=provider)
    elif aiohttp and isinstance(error, aiohttp.ClientError):
        return LLMInternalServerError(message=f"HTTP client error: {str(error)}", provider=provider)
    elif httpx and isinstance(error, httpx.RemoteProtocolError):
        return LLMInternalServerError(message=f"HTTPX protocol error: {str(error)}", provider=provider)
    elif isinstance(error, KeyError):
        return LLMInvalidRequestError(message=f"Missing required parameter: {str(error)}", provider=provider)
    elif isinstance(error, TypeError):
        return LLMInvalidRequestError(message=f"Invalid parameter type: {str(error)}", provider=provider)
    elif isinstance(error, RuntimeError):
        return LLMInternalServerError(message=f"Runtime error: {str(error)}", provider=provider)

    # Default fallback for any other exception
    return LLMError(message=f"Unexpected error ({type(error).__name__}): {str(error)}", provider=provider)
