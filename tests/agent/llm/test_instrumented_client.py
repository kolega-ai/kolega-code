# ruff: noqa: F401,F811,E402
"""
Tests for the InstrumentedLLMClient class.
"""

import os
import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock

from kolega_code.llm.models import Message, MessageHistory, TextBlock
from kolega_code.llm.instrumented_client import (
    InstrumentedLLMClient,
    MinimalLangfuseStreamWrapper,
)

# Check if running in CI environment
SKIP_IN_CI = bool(os.getenv("CI")) or bool(os.getenv("GITLAB_CI"))


class TestInstrumentedLLMClient:
    """Test the InstrumentedLLMClient class."""

    @pytest.fixture
    def mock_langfuse(self):
        """Create mock Langfuse client with generation tracking (v3 API)."""
        langfuse = MagicMock()

        # Create a mock generation that tracks calls
        generation = MagicMock()
        generation.update = MagicMock()
        generation.end = MagicMock()

        # Create a mock trace/span that returns the generation
        trace = MagicMock()
        trace.update_trace = MagicMock()
        trace.update = MagicMock()
        trace.end = MagicMock()
        trace.start_generation = MagicMock(return_value=generation)

        # Make langfuse.start_span() return the trace
        langfuse.start_span = MagicMock(return_value=trace)

        return langfuse, generation

    @pytest.fixture
    def instrumented_client(self, mock_langfuse):
        """Create an instrumented LLM client with mocked Langfuse."""
        langfuse, _ = mock_langfuse
        return InstrumentedLLMClient(
            provider="anthropic",
            api_key="test-key",
            langfuse_client=langfuse,
            workspace_id="workspace-123",
            thread_id="thread-456",
            agent_type="test-agent",
            environment="test",
            user_id="user-789",
            user_email="test@example.com",
        )

    def test_init_with_langfuse(self, mock_langfuse):
        """Test initialization with Langfuse client."""
        langfuse, _ = mock_langfuse
        client = InstrumentedLLMClient(
            provider="anthropic",
            api_key="test-key",
            langfuse_client=langfuse,
            workspace_id="workspace-123",
            thread_id="thread-456",
            agent_type="test-agent",
            environment="production",
            user_id="user-789",
            user_email="test@example.com",
        )

        assert client.langfuse == langfuse
        assert client.workspace_id == "workspace-123"
        assert client.thread_id == "thread-456"
        assert client.agent_type == "test-agent"
        assert client.environment == "production"
        assert client.user_id == "user-789"
        assert client.user_email == "test@example.com"

    def test_init_without_langfuse(self):
        """Test initialization without Langfuse client."""
        client = InstrumentedLLMClient(
            provider="anthropic",
            api_key="test-key",
        )

        assert client.langfuse is None
        assert client.workspace_id is None
        assert client.thread_id is None
        assert client.agent_type is None
        assert client.environment == "development"  # default from os.environ

    def test_create_generation_metadata(self, instrumented_client):
        """Test metadata creation for Langfuse generation."""
        metadata = instrumented_client._create_generation_metadata(
            custom_field="value",
            another_field=123,
        )

        assert metadata["provider"] == "anthropic"
        assert metadata["workspace_id"] == "workspace-123"
        assert metadata["thread_id"] == "thread-456"
        assert metadata["agent_type"] == "test-agent"
        assert metadata["environment"] == "test"
        assert metadata["user_id"] == "user-789"
        assert metadata["user_email"] == "test@example.com"
        assert metadata["custom_field"] == "value"
        assert metadata["another_field"] == 123
        assert "timestamp" in metadata

    def test_extract_usage_details_anthropic(self, instrumented_client):
        """Test extraction of usage details from Anthropic response."""
        # Mock Message with usage_metadata
        mock_response = Mock()
        mock_response.usage_metadata = {
            "provider": "anthropic",
            "input_tokens": 100,
            "output_tokens": 50,
            "cache_read_input_tokens": 25,
            "cache_creation_input_tokens": 5,
            "cache_write_input_tokens": 5,  # Add this field that the code expects
        }

        usage = instrumented_client._extract_usage_details(mock_response)

        assert usage["input_tokens"] == 100
        assert usage["output_tokens"] == 50
        assert usage["cache_read_input_tokens"] == 25
        assert usage["cache_write_input_tokens"] == 5

    def test_extract_usage_details_openai(self, instrumented_client):
        """Test extraction of usage details from OpenAI response."""
        # Mock Message with usage_metadata
        mock_response = Mock()
        mock_response.usage_metadata = {
            "provider": "openai",
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
        }

        usage = instrumented_client._extract_usage_details(mock_response)

        assert usage["prompt_tokens"] == 100
        assert usage["completion_tokens"] == 50
        assert usage["total_tokens"] == 150

    def test_extract_usage_details_google(self, instrumented_client):
        """Test extraction of usage details from Google response."""
        # Mock Message with usage_metadata
        mock_response = Mock()
        mock_response.usage_metadata = {
            "provider": "google",
            "prompt_token_count": 100,
            "candidates_token_count": 50,
            "total_token_count": 150,
        }

        usage = instrumented_client._extract_usage_details(mock_response)

        assert usage["prompt_token_count"] == 100
        assert usage["candidates_token_count"] == 50
        assert usage["total_token_count"] == 150

    def test_normalize_usage_data_deepseek(self, instrumented_client):
        usage = instrumented_client._normalize_usage_data(
            {
                "provider": "deepseek",
                "input_tokens": 100,
                "output_tokens": 50,
                "cache_read_input_tokens": 25,
                "cache_write_input_tokens": 5,
            },
            model="deepseek-v4-pro",
        )

        assert usage["provider"] == "deepseek"
        assert usage["model"] == "deepseek-v4-pro"
        assert usage["input_tokens"] == 100
        assert usage["output_tokens"] == 50
        assert usage["cache_read_input_tokens"] == 25
        assert usage["cache_write_input_tokens"] == 5

    def test_extract_usage_details_no_metadata(self, instrumented_client):
        """Test extraction returns empty dict when no metadata."""
        # Mock Message without usage_metadata
        mock_response = Mock(spec=[])

        usage = instrumented_client._extract_usage_details(mock_response)
        assert usage == {}

        # Mock Message with empty usage_metadata
        mock_response = Mock()
        mock_response.usage_metadata = {}

        usage = instrumented_client._extract_usage_details(mock_response)
        assert usage == {}

    @pytest.mark.asyncio
    async def test_generate_with_langfuse(self, instrumented_client, mock_langfuse):
        """Test generate method with Langfuse tracing."""
        langfuse, generation = mock_langfuse
        trace = langfuse.start_span.return_value

        # Mock the parent generate method
        mock_response = Mock()
        mock_response.to_dict = Mock(return_value={"content": "test response"})
        mock_response.usage_metadata = {
            "provider": "anthropic",
            "input_tokens": 10,
            "output_tokens": 5,
        }

        # Use patch on the parent class method
        with patch("kolega_code.llm.client.LLMClient.generate", AsyncMock(return_value=mock_response)):
            messages = MessageHistory([Message(role="user", content=[TextBlock(text="Hello")])])

            await instrumented_client.generate(
                messages=messages, model="claude-3-opus", temperature=0.5, max_completion_tokens=100
            )

            # Verify Langfuse trace was called correctly
            langfuse.start_span.assert_called_once()
            span_args = langfuse.start_span.call_args

            assert span_args.kwargs["name"] == "test-agent-llm-call"

            # Verify trace attributes were updated
            trace.update_trace.assert_called_once()
            trace_update_args = trace.update_trace.call_args
            assert trace_update_args.kwargs["user_id"] == "user-789"  # Now uses actual user_id
            assert trace_update_args.kwargs["session_id"] == "workspace-123/thread-456"
            assert "test" in trace_update_args.kwargs["tags"]
            assert "user:user-789" in trace_update_args.kwargs["tags"]

            # Verify generation was called on the trace
            trace.start_generation.assert_called_once()
            gen_args = trace.start_generation.call_args
            assert gen_args.kwargs["name"] == "test-agent-llm-generation"
            assert gen_args.kwargs["model"] == "claude-3-opus"
            assert gen_args.kwargs["model_parameters"]["temperature"] == 0.5

            # Verify generation.update and end were called with success
            generation.update.assert_called_once()
            update_args = generation.update.call_args
            assert update_args.kwargs["level"] == "DEFAULT"
            assert update_args.kwargs["status_message"] == "Success"
            assert update_args.kwargs["usage_details"]["input"] == 10
            assert update_args.kwargs["usage_details"]["output"] == 5
            generation.end.assert_called_once()
            trace.end.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_with_user_tracking(self):
        """Test generate method with user tracking information."""
        mock_langfuse = MagicMock()
        generation = MagicMock()
        trace = MagicMock()
        trace.start_generation = MagicMock(return_value=generation)
        mock_langfuse.start_span = MagicMock(return_value=trace)

        # Create client with user information
        client = InstrumentedLLMClient(
            provider="anthropic",
            api_key="test-key",
            langfuse_client=mock_langfuse,
            workspace_id="workspace-123",
            thread_id="thread-456",
            agent_type="test-agent",
            environment="test",
            user_id="user-789",
            user_email="test@example.com",
        )

        # Mock the parent generate method
        mock_response = Mock()
        mock_response.to_dict = Mock(return_value={"content": "test response"})
        mock_response.usage_metadata = {
            "provider": "anthropic",
            "input_tokens": 10,
            "output_tokens": 5,
        }

        # Use patch on the parent class method
        with patch("kolega_code.llm.client.LLMClient.generate", AsyncMock(return_value=mock_response)):
            messages = MessageHistory([Message(role="user", content=[TextBlock(text="Hello")])])

            await client.generate(messages=messages, model="claude-3-opus", temperature=0.5, max_completion_tokens=100)

            # Verify trace attributes include user information
            trace.update_trace.assert_called_once()
            trace_update_args = trace.update_trace.call_args
            assert trace_update_args.kwargs["user_id"] == "user-789"  # Uses actual user_id, not workspace
            assert trace_update_args.kwargs["session_id"] == "workspace-123/thread-456"  # No email in session name
            assert "user:user-789" in trace_update_args.kwargs["tags"]

            # Verify metadata includes user information
            span_args = mock_langfuse.start_span.call_args
            metadata = span_args.kwargs["metadata"]
            assert metadata["user_id"] == "user-789"
            assert metadata["user_email"] == "test@example.com"

    @pytest.mark.asyncio
    async def test_generate_without_langfuse(self):
        """Test generate falls back to parent when no Langfuse."""
        client = InstrumentedLLMClient(
            provider="anthropic",
            api_key="test-key",
            langfuse_client=None,
        )

        # Mock parent generate
        mock_response = Mock()
        with patch("kolega_code.llm.client.LLMClient.generate", AsyncMock(return_value=mock_response)):
            messages = MessageHistory([Message(role="user", content=[TextBlock(text="Hello")])])

            result = await client.generate(messages=messages)
            assert result == mock_response

    @pytest.mark.asyncio
    async def test_error_handling(self, instrumented_client, mock_langfuse):
        """Test error handling in generate method."""
        langfuse, generation = mock_langfuse
        trace = langfuse.start_span.return_value

        error_msg = "API Error"
        with patch("kolega_code.llm.client.LLMClient.generate", AsyncMock(side_effect=Exception(error_msg))):
            messages = MessageHistory([Message(role="user", content=[TextBlock(text="Hello")])])

            with pytest.raises(Exception) as exc_info:
                await instrumented_client.generate(messages=messages)

            assert str(exc_info.value) == error_msg

            # Verify generation.update was called with error
            generation.update.assert_called_once()
            update_args = generation.update.call_args
            assert update_args.kwargs["level"] == "ERROR"
            assert update_args.kwargs["status_message"] == error_msg

            # Verify generation.end and trace.end were still called
            generation.end.assert_called_once()
            trace.update.assert_called_once()
            trace.end.assert_called_once()

    @pytest.mark.asyncio
    async def test_stream_with_langfuse(self, instrumented_client, mock_langfuse):
        """Test stream method with Langfuse tracing."""
        langfuse, generation = mock_langfuse
        trace = langfuse.start_span.return_value

        # Mock stream context manager
        mock_stream = AsyncMock()
        mock_stream.__aenter__ = AsyncMock(return_value=mock_stream)
        mock_stream.__aexit__ = AsyncMock(return_value=None)

        with patch("kolega_code.llm.client.LLMClient.stream", MagicMock(return_value=mock_stream)):
            messages = MessageHistory([Message(role="user", content=[TextBlock(text="Hello")])])

            # stream() now returns a coroutine, so we need to await it
            result_coro = instrumented_client.stream(messages=messages, model="claude-3-opus")

            # The coroutine should create langfuse tracing when awaited
            result = await result_coro

            # Verify Langfuse trace was called
            langfuse.start_span.assert_called_once()
            span_args = langfuse.start_span.call_args
            assert span_args.kwargs["name"] == "test-agent-llm-stream"

            # Verify trace attributes were updated
            trace.update_trace.assert_called_once()
            trace_update_args = trace.update_trace.call_args
            assert "streaming" in trace_update_args.kwargs["tags"]

            # Verify generation was called on the trace
            trace.start_generation.assert_called_once()
            gen_args = trace.start_generation.call_args
            assert gen_args.kwargs["name"] == "test-agent-llm-stream-generation"
            assert gen_args.kwargs["model"] == "claude-3-opus"

            # Should return an instrumented wrapper
            assert isinstance(result, MinimalLangfuseStreamWrapper)
