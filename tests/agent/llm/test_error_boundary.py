"""Test suite for LLM client error boundary functionality.

This module tests that the LLM client correctly maps all exceptions to LLMError
subclasses, ensuring no raw exceptions escape from the client layer.
"""

import os
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from anthropic import AnthropicError
from openai import OpenAIError

from kolega_code.llm.client import LLMClient
from kolega_code.llm.exceptions import (
    LLMBillingError,
    LLMAuthenticationError,
    LLMConnectionError,
    LLMContextWindowExceededError,
    LLMError,
    LLMInternalServerError,
    LLMInvalidRequestError,
    LLMRateLimitError,
    LLMTimeout,
    map_to_llm_error,
)
from kolega_code.llm.models import Message, MessageHistory

# Check if running in CI environment
SKIP_IN_CI = bool(os.getenv("CI")) or bool(os.getenv("GITLAB_CI"))


class TestErrorMapping:
    """Test the comprehensive error mapping function."""

    def test_llm_error_passes_through(self):
        """Test that LLMError instances pass through unchanged."""
        original_error = LLMInvalidRequestError("test error", provider="test")
        mapped_error = map_to_llm_error(original_error)
        assert mapped_error is original_error

    def test_value_error_mapping(self):
        """Test that ValueError maps to LLMInvalidRequestError."""
        error = ValueError("Invalid value")
        mapped = map_to_llm_error(error, "test_provider")
        assert isinstance(mapped, LLMInvalidRequestError)
        assert "Invalid parameter" in str(mapped)
        assert mapped.provider == "test_provider"

    def test_timeout_error_mapping(self):
        """Test that timeout errors map to LLMTimeout."""
        # Test standard TimeoutError
        error = TimeoutError("Request timed out")
        mapped = map_to_llm_error(error, "test_provider")
        assert isinstance(mapped, LLMTimeout)
        assert "Request timeout" in str(mapped)

        # Test asyncio.TimeoutError
        error = asyncio.TimeoutError("Async timeout")
        mapped = map_to_llm_error(error, "test_provider")
        assert isinstance(mapped, LLMTimeout)

    def test_connection_error_mapping(self):
        """Test that ConnectionError maps to LLMConnectionError (a transport failure, not a 5xx)."""
        error = ConnectionError("Connection failed")
        mapped = map_to_llm_error(error, "test_provider")
        assert isinstance(mapped, LLMConnectionError)
        assert not isinstance(mapped, LLMInternalServerError)
        assert "Connection error" in str(mapped)

    def test_key_error_mapping(self):
        """Test that KeyError maps to LLMInvalidRequestError."""
        error = KeyError("missing_param")
        mapped = map_to_llm_error(error, "test_provider")
        assert isinstance(mapped, LLMInvalidRequestError)
        assert "Missing required parameter" in str(mapped)

    def test_type_error_mapping(self):
        """Test that TypeError maps to LLMInvalidRequestError."""
        error = TypeError("Wrong type")
        mapped = map_to_llm_error(error, "test_provider")
        assert isinstance(mapped, LLMInvalidRequestError)
        assert "Invalid parameter type" in str(mapped)

    def test_runtime_error_mapping(self):
        """Test that RuntimeError maps to LLMInternalServerError."""
        error = RuntimeError("Runtime issue")
        mapped = map_to_llm_error(error, "test_provider")
        assert isinstance(mapped, LLMInternalServerError)
        assert "Runtime error" in str(mapped)

    def test_generic_exception_mapping(self):
        """Test that generic exceptions map to base LLMError."""

        class CustomException(Exception):
            pass

        error = CustomException("Custom error")
        mapped = map_to_llm_error(error, "test_provider")
        assert isinstance(mapped, LLMError)
        assert not isinstance(mapped, (LLMInvalidRequestError, LLMInternalServerError, LLMTimeout))
        assert "Unexpected error (CustomException)" in str(mapped)
        assert mapped.provider == "test_provider"

    def test_httpx_remote_protocol_error_mapping(self):
        """httpx.RemoteProtocolError (a TransportError) maps to the retryable LLMConnectionError."""
        try:
            import httpx  # type: ignore
        except Exception:
            pytest.skip("httpx not installed in test environment")

        error = httpx.RemoteProtocolError(
            "peer closed connection without sending complete message body (incomplete chunked read)"
        )
        mapped = map_to_llm_error(error, "anthropic")
        assert isinstance(mapped, LLMConnectionError)
        assert "HTTPX transport error" in str(mapped)

    def test_provider_error_mapping(self):
        """Test that provider-specific errors are mapped correctly."""

        # Create proper mock OpenAI error that inherits from OpenAIError
        class MockOpenAIError(OpenAIError):
            def __init__(self, status_code):
                self.status_code = status_code
                super().__init__("Rate limit exceeded")

        openai_error = MockOpenAIError(429)
        mapped = map_to_llm_error(openai_error)
        assert isinstance(mapped, LLMRateLimitError)

        # Create proper mock Anthropic error
        class MockAnthropicError(AnthropicError):
            def __init__(self, status_code):
                self.status_code = status_code
                super().__init__("Invalid API key")

        anthropic_error = MockAnthropicError(401)
        mapped = map_to_llm_error(anthropic_error)
        assert isinstance(mapped, LLMAuthenticationError)

        deepseek_error = MockAnthropicError(402)
        mapped = map_to_llm_error(deepseek_error, provider="deepseek")
        assert isinstance(mapped, LLMBillingError)
        assert mapped.provider == "deepseek"


class TestLLMClientErrorBoundary:
    """Test that the LLM client correctly implements the error boundary."""

    def test_initialization_error_handling(self):
        """Test that initialization errors are properly mapped."""
        # Test unsupported provider
        with pytest.raises(LLMInvalidRequestError) as exc_info:
            LLMClient(provider="unsupported_provider", api_key="test_key")
        assert "Invalid parameter: Unsupported provider" in str(exc_info.value)
        assert exc_info.value.provider == "unsupported_provider"

    @pytest.mark.asyncio
    async def test_generate_error_handling(self):
        """Test that generate method errors are properly mapped."""
        client = LLMClient(provider="openai", api_key="test_key")

        # Mock the provider to raise various exceptions
        messages = MessageHistory([Message(role="user", content="Test")])

        # Test ValueError
        client.provider.generate = AsyncMock(side_effect=ValueError("Invalid parameter"))
        with pytest.raises(LLMInvalidRequestError) as exc_info:
            await client.generate(messages)
        assert "Invalid parameter" in str(exc_info.value)
        assert exc_info.value.provider == "openai"

        # Test ConnectionError (a transport failure → retryable LLMConnectionError, not a 5xx)
        client.provider.generate = AsyncMock(side_effect=ConnectionError("Connection lost"))
        with pytest.raises(LLMConnectionError) as exc_info:
            await client.generate(messages)
        assert "Connection error" in str(exc_info.value)

        # Test that LLMError passes through
        original_error = LLMRateLimitError("Rate limit hit", provider="openai")
        client.provider.generate = AsyncMock(side_effect=original_error)
        with pytest.raises(LLMRateLimitError) as exc_info:
            await client.generate(messages)
        assert exc_info.value is original_error

    @pytest.mark.asyncio
    async def test_count_tokens_error_handling(self):
        """Test that count_tokens method errors are properly mapped."""
        client = LLMClient(provider="anthropic", api_key="test_key")
        messages = MessageHistory([Message(role="user", content="Test")])

        # Test KeyError
        client.provider.count_tokens = AsyncMock(side_effect=KeyError("model"))
        with pytest.raises(LLMInvalidRequestError) as exc_info:
            await client.count_tokens(messages)
        assert "Missing required parameter" in str(exc_info.value)
        assert exc_info.value.provider == "anthropic"

        # Test TimeoutError
        client.provider.count_tokens = AsyncMock(side_effect=asyncio.TimeoutError())
        with pytest.raises(LLMTimeout) as exc_info:
            await client.count_tokens(messages)
        assert "Request timeout" in str(exc_info.value)

    def test_stream_error_handling(self):
        """Test that stream method errors are properly mapped."""
        client = LLMClient(provider="google", api_key="test_key")
        messages = MessageHistory([Message(role="user", content="Test")])

        # Test RuntimeError
        client.provider.stream = MagicMock(side_effect=RuntimeError("Stream failed"))
        with pytest.raises(LLMInternalServerError) as exc_info:
            client.stream(messages)
        assert "Runtime error" in str(exc_info.value)
        assert exc_info.value.provider == "google"

        # Test TypeError
        client.provider.stream = MagicMock(side_effect=TypeError("Invalid type"))
        with pytest.raises(LLMInvalidRequestError) as exc_info:
            client.stream(messages)
        assert "Invalid parameter type" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_provider_specific_error_mapping(self):
        """Test that provider-specific errors are correctly mapped through the client."""
        # Test OpenAI error mapping
        client = LLMClient(provider="openai", api_key="test_key")
        messages = MessageHistory([Message(role="user", content="Test")])

        # Create proper mock OpenAI error
        class MockOpenAIError(OpenAIError):
            def __init__(self, status_code, message="Invalid API key"):
                self.status_code = status_code
                super().__init__(message)

        openai_error = MockOpenAIError(401)

        client.provider.generate = AsyncMock(side_effect=openai_error)
        with pytest.raises(LLMAuthenticationError) as exc_info:
            await client.generate(messages)
        assert "OpenAI APIError" in str(exc_info.value)

        # Test Anthropic error mapping
        client = LLMClient(provider="anthropic", api_key="test_key")

        # Create proper mock Anthropic error
        class MockAnthropicError(AnthropicError):
            def __init__(self, status_code, message="Message too long"):
                self.status_code = status_code
                super().__init__(message)

        anthropic_error = MockAnthropicError(413)

        client.provider.generate = AsyncMock(side_effect=anthropic_error)
        with pytest.raises(LLMContextWindowExceededError) as exc_info:  # Should be mapped to context window error
            await client.generate(messages)
        assert "AnthropicError" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_no_raw_exceptions_escape(self):
        """Comprehensive test ensuring no raw exceptions escape the client."""
        client = LLMClient(provider="openai", api_key="test_key")
        messages = MessageHistory([Message(role="user", content="Test")])

        # List of various exception types to test
        test_exceptions = [
            ValueError("test"),
            KeyError("test"),
            TypeError("test"),
            RuntimeError("test"),
            ConnectionError("test"),
            TimeoutError("test"),
            asyncio.TimeoutError("test"),
            AttributeError("test"),
            IndexError("test"),
            ZeroDivisionError("test"),
            Exception("generic exception"),
        ]

        for original_exception in test_exceptions:
            # Test generate method
            client.provider.generate = AsyncMock(side_effect=original_exception)
            with pytest.raises(LLMError) as exc_info:
                await client.generate(messages)
            # Verify it's an LLMError subclass, not the original exception
            assert isinstance(exc_info.value, LLMError)
            assert type(exc_info.value) is not type(original_exception)

            # Test count_tokens method
            client.provider.count_tokens = AsyncMock(side_effect=original_exception)
            with pytest.raises(LLMError) as exc_info:
                await client.count_tokens(messages)
            assert isinstance(exc_info.value, LLMError)

            # Test stream method
            client.provider.stream = MagicMock(side_effect=original_exception)
            with pytest.raises(LLMError) as exc_info:
                client.stream(messages)
            assert isinstance(exc_info.value, LLMError)


class TestProviderInitialization:
    """Test error handling during provider initialization."""

    @pytest.mark.skipif(SKIP_IN_CI, reason="Skipping slow test in CI environment")
    def test_all_supported_providers_initialize(self):
        """Test that all supported providers can be initialized without error."""
        supported_providers = ["anthropic", "openai", "google", "together", "groq", "fireworks", "llama", "xai"]

        for provider in supported_providers:
            try:
                client = LLMClient(provider=provider, api_key="test_key")
                assert client.provider_name == provider
                assert client.provider is not None
            except Exception as e:
                # If any exception occurs, it should be an LLMError
                assert isinstance(e, LLMError), f"Provider {provider} raised non-LLMError: {type(e)}"

    def test_provider_initialization_with_error(self):
        """Test that provider initialization errors are properly wrapped."""
        # The api-key `openai` provider now uses the Responses provider, so mock
        # that class to raise during initialization.
        with patch("kolega_code.llm.client.OpenAIResponsesProvider") as MockProvider:
            MockProvider.side_effect = RuntimeError("Provider init failed")

            with pytest.raises(LLMInternalServerError) as exc_info:
                LLMClient(provider="openai", api_key="test_key")
            assert "Runtime error: Provider init failed" in str(exc_info.value)
