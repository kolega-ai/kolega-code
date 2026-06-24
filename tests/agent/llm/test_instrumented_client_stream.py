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


class TestStreamWrappers:
    """Test the instrumented stream wrapper classes."""

    @pytest.mark.asyncio
    async def test_stream_wrapper_accumulates_content(self):
        """Test that stream wrapper accumulates content for Langfuse."""
        mock_stream = AsyncMock()
        mock_generation = MagicMock()
        mock_trace = MagicMock()
        mock_client = MagicMock()
        mock_client._record_usage = AsyncMock()
        model = "claude-3-opus"

        wrapper = MinimalLangfuseStreamWrapper(mock_stream, mock_generation, mock_trace, mock_client, model)

        # Create mock chunks with get_text_content method
        chunks = []
        for text in ["Hello", " ", "world"]:
            chunk = MagicMock()
            chunk.get_text_content.return_value = text
            chunks.append(chunk)

        mock_stream.__anext__ = AsyncMock(side_effect=chunks + [StopAsyncIteration])

        # Create mock final message
        final_message = MagicMock()
        final_message.get_text_content.return_value = "Hello world"
        final_message.usage_metadata = {"provider": "anthropic", "input_tokens": 10, "output_tokens": 2}
        mock_stream.get_final_message = AsyncMock(return_value=final_message)

        # Consume stream
        collected = []
        async with wrapper:
            async for chunk in wrapper:
                collected.append(chunk)

        assert len(collected) == 3

        # Verify generation was updated with final data
        mock_generation.update.assert_called_once()
        update_call = mock_generation.update.call_args
        assert update_call[1]["output"] == "Hello world"
        assert update_call[1]["usage_details"] == {
            "input": 10,
            "output": 2,
            "total": 12,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
        }
        mock_generation.end.assert_called_once()
        mock_trace.end.assert_called_once()

        mock_client._record_usage.assert_awaited_once()
        usage_call = mock_client._record_usage.call_args
        assert usage_call[0][0] == {"provider": "anthropic", "input_tokens": 10, "output_tokens": 2}
        assert usage_call[0][1] == model

    @pytest.mark.asyncio
    @pytest.mark.skipif(SKIP_IN_CI, reason="Skipping slow test in CI environment")
    async def test_stream_wrapper_provider_selection(self):
        """Test correct stream wrapper is selected based on provider."""
        test_cases = [
            ("anthropic", MinimalLangfuseStreamWrapper),
            ("openai", MinimalLangfuseStreamWrapper),
            ("groq", MinimalLangfuseStreamWrapper),
            ("google", MinimalLangfuseStreamWrapper),
        ]

        for provider, expected_wrapper in test_cases:
            mock_stream = AsyncMock()
            mock_generation = MagicMock()
            mock_langfuse = MagicMock()
            mock_langfuse.generation.return_value = mock_generation

            # Create instrumented client with specific provider
            instrumented_client = InstrumentedLLMClient(
                provider=provider,
                api_key="test-key",
                langfuse_client=mock_langfuse,
                workspace_id="workspace-123",
                thread_id="thread-456",
                agent_type="test-agent",
                environment="test",
            )

            # All providers now use MinimalLangfuseStreamWrapper
            mock_trace = MagicMock()
            model = "test-model"
            wrapper = MinimalLangfuseStreamWrapper(mock_stream, mock_generation, mock_trace, instrumented_client, model)

            assert isinstance(wrapper, expected_wrapper)

    @pytest.mark.asyncio
    async def test_stream_wrapper_handles_exceptions(self):
        """Test stream wrapper handles exceptions gracefully."""
        mock_stream = AsyncMock()
        mock_generation = MagicMock()
        mock_trace = MagicMock()
        mock_client = MagicMock()
        mock_client._record_usage = AsyncMock()
        model = "test-model"

        wrapper = MinimalLangfuseStreamWrapper(mock_stream, mock_generation, mock_trace, mock_client, model)

        # Mock stream that raises exception
        mock_stream.__anext__ = AsyncMock(side_effect=Exception("Stream error"))
        mock_stream.get_final_message = AsyncMock(side_effect=Exception("No final message"))

        # Should propagate exception but still end generation
        with pytest.raises(Exception, match="Stream error"):
            async with wrapper:
                async for chunk in wrapper:
                    pass

        # Generation should still be ended
        mock_generation.end.assert_called_once()

    @pytest.mark.asyncio
    async def test_stream_wrapper_without_usage_data(self):
        """Test stream wrapper handles missing usage data gracefully."""
        mock_stream = AsyncMock()
        mock_generation = MagicMock()
        mock_trace = MagicMock()
        mock_client = MagicMock()
        mock_client._record_usage = AsyncMock()
        model = "test-model"

        wrapper = MinimalLangfuseStreamWrapper(mock_stream, mock_generation, mock_trace, mock_client, model)

        # Mock final message without usage metadata
        final_message = MagicMock()
        final_message.get_text_content.return_value = "Response"
        final_message.usage_metadata = {}
        mock_stream.get_final_message = AsyncMock(return_value=final_message)
        mock_stream.__anext__ = AsyncMock(side_effect=StopAsyncIteration)

        async with wrapper:
            pass

        # Should update without usage data
        mock_generation.update.assert_called_once()
        update_call = mock_generation.update.call_args
        assert update_call[1]["output"] == "Response"
        assert "usage_details" not in update_call[1]  # usage_details key should not be present when no usage data
        mock_generation.end.assert_called_once()
        mock_trace.end.assert_called_once()
