"""Test suite for the think_hard tool with streaming implementation."""

import pytest
from unittest.mock import AsyncMock, Mock, patch

from kolega_code.agent.config import AgentConfig, ModelConfig, ModelProvider, RateLimitConfig
from kolega_code.agent.connection_manager import AgentConnectionManager
from kolega_code.agent.llm.models import Message, TextBlock, ThinkingBlock
from kolega_code.agent.tool_backend.think_hard_tool import ThinkHardTool


class MockStreamWrapper:
    """Mock stream wrapper that simulates the AnthropicStreamWrapper behavior."""

    def __init__(self, final_message: Message):
        self.final_message = final_message
        self._entered = False
        self.chunks = []  # No chunks to iterate over
        self.chunk_index = 0

    async def __aenter__(self):
        self._entered = True
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self._entered = False
        return None

    def __aiter__(self):
        """Make the stream async iterable."""
        return self

    async def __anext__(self):
        """Return chunks for async iteration."""
        if self.chunk_index >= len(self.chunks):
            raise StopAsyncIteration
        chunk = self.chunks[self.chunk_index]
        self.chunk_index += 1
        return chunk

    async def get_final_message(self) -> Message:
        """Return the final message after streaming completes."""
        if not self._entered:
            raise RuntimeError("Must use 'async with' before getting final message")
        return self.final_message


class MockStreamChunk:
    def __init__(self, thinking: str = "", text: str = ""):
        self.thinking = thinking
        self.text = text


@pytest.fixture
def mock_config():
    """Create a mock agent configuration."""
    return AgentConfig(
        anthropic_api_key="test-key",
        openai_api_key="test-openai-key",  # Required for edit_model_config
        thinking_config=ModelConfig(
            provider=ModelProvider.ANTHROPIC,
            model="claude-3-7-sonnet-20250131",
            rate_limits=RateLimitConfig(requests_per_minute=10, tokens_per_minute=100000, max_retries=3),
            thinking_tokens=5000,
        ),
    )


@pytest.fixture
def mock_connection_manager():
    """Create a mock connection manager."""
    return AsyncMock(spec=AgentConnectionManager)


@pytest.fixture
def mock_caller():
    """Create a mock caller (base agent)."""
    mock = Mock()
    mock.agent_name = "test_agent"
    mock.user_id = "user-123"
    mock.user_email = "user@example.com"
    return mock


@pytest.fixture
def think_hard_tool(mock_config, mock_connection_manager, mock_caller):
    """Create a ThinkHardTool instance with mocked dependencies."""
    tool = ThinkHardTool(
        project_path="/test/path",
        workspace_id="test_workspace",
        thread_id="test_thread",
        connection_manager=mock_connection_manager,
        config=mock_config,
        caller=mock_caller,
    )

    # Mock the log methods
    tool.log_info = AsyncMock()
    tool.log_error = AsyncMock()

    # Mock the streaming update method
    tool.send_streaming_update = AsyncMock()

    return tool


@pytest.mark.asyncio
async def test_think_hard_streaming_with_thinking_and_text(think_hard_tool, mock_connection_manager):
    """Test think_hard with both thinking and text content using streaming."""

    # Create a mock final message with both thinking and text blocks
    final_message = Message(
        role="assistant",
        content=[
            ThinkingBlock(thinking="This is deep thinking about the problem..."),
            ThinkingBlock(thinking="Additional thinking process..."),
            TextBlock(text="This is the final analysis."),
            TextBlock(text="Additional insights."),
        ],
    )

    # Create mock stream wrapper
    mock_stream = MockStreamWrapper(final_message)

    # Mock the LLMClient and its stream method
    with patch("kolega_code.agent.tool_backend.think_hard_tool.LLMClient") as mock_llm_class:
        with patch("kolega_code.agent.tool_backend.think_hard_tool.get_model_specs") as mock_get_specs:
            # Mock model specs
            mock_get_specs.return_value = {"max_completion_tokens": 8192}

            mock_llm_instance = mock_llm_class.return_value

            # stream method returns a coroutine that returns the mock stream wrapper
            async def stream_coroutine(*args, **kwargs):
                return mock_stream

            mock_llm_instance.stream = stream_coroutine

            # Call think_hard
            result = await think_hard_tool.think_hard("Test problem statement")

            # Verify the LLMClient was created with correct parameters
            mock_llm_class.assert_called_once_with(
                provider="anthropic",
                api_key="test-key",
                max_retries=3,
                requests_per_minute=10,
                tokens_per_minute=100000,
            )

            # Verify stream was called (we can't use assert_called_once with a regular function)
            # The test passing indicates stream was called successfully

            # Verify the result format is correct
            expected_result = (
                "# Extended Thinking Process\n\n"
                "This is deep thinking about the problem...\n"
                "Additional thinking process...\n\n"
                "# Final Analysis\n\n"
                "This is the final analysis.\n"
                "Additional insights."
            )
            assert result == expected_result

            # Verify logging
            think_hard_tool.log_info.assert_called_once()
            assert "Thinking hard about: Test problem statement" in think_hard_tool.log_info.call_args[0][0]


@pytest.mark.asyncio
async def test_think_hard_streaming_updates_use_append_mode_for_live_deltas(think_hard_tool):
    """Test think_hard marks live deltas as append mode and final content as replacement."""
    think_hard_tool.caller.current_tool_call_id = "tool-1"

    final_message = Message(
        role="assistant",
        content=[
            ThinkingBlock(thinking="T" * 60),
            TextBlock(text="A" * 60),
        ],
    )
    mock_stream = MockStreamWrapper(final_message)
    mock_stream.chunks = [
        MockStreamChunk(thinking="T" * 60),
        MockStreamChunk(text="A" * 60),
    ]

    with patch("kolega_code.agent.tool_backend.think_hard_tool.LLMClient") as mock_llm_class:
        with patch("kolega_code.agent.tool_backend.think_hard_tool.get_model_specs") as mock_get_specs:
            mock_get_specs.return_value = {"max_completion_tokens": 8192}
            mock_llm_instance = mock_llm_class.return_value

            async def stream_coroutine(*args, **kwargs):
                return mock_stream

            mock_llm_instance.stream = stream_coroutine

            await think_hard_tool.think_hard("Test problem statement")

    calls = think_hard_tool.send_streaming_update.await_args_list
    incomplete_calls = [call for call in calls if call.kwargs.get("is_complete") is False]

    assert incomplete_calls
    assert all(call.kwargs["stream_mode"] == "append" for call in incomplete_calls)
    assert calls[-1].kwargs["is_complete"] is True
    assert calls[-1].kwargs["stream_mode"] == "replace"


@pytest.mark.asyncio
async def test_think_hard_streaming_only_text(think_hard_tool, mock_connection_manager):
    """Test think_hard with only text content (no thinking blocks)."""

    # Create a mock final message with only text blocks
    final_message = Message(role="assistant", content=[TextBlock(text="Direct response without extended thinking.")])

    # Create mock stream wrapper
    mock_stream = MockStreamWrapper(final_message)

    with patch("kolega_code.agent.tool_backend.think_hard_tool.LLMClient") as mock_llm_class:
        with patch("kolega_code.agent.tool_backend.think_hard_tool.get_model_specs") as mock_get_specs:
            # Mock model specs
            mock_get_specs.return_value = {"max_completion_tokens": 8192}

            mock_llm_instance = mock_llm_class.return_value

            # stream method returns a coroutine that returns the mock stream wrapper
            async def stream_coroutine(*args, **kwargs):
                return mock_stream

            mock_llm_instance.stream = stream_coroutine

            # Call think_hard
            result = await think_hard_tool.think_hard("Simple question")

            # Verify the result format (no thinking section)
            expected_result = "# Final Analysis\n\n" "Direct response without extended thinking."
            assert result == expected_result


@pytest.mark.asyncio
async def test_think_hard_streaming_error_handling(think_hard_tool, mock_connection_manager):
    """Test error handling during streaming."""

    # Mock the LLMClient to raise an exception
    with patch("kolega_code.agent.tool_backend.think_hard_tool.LLMClient") as mock_llm_class:
        with patch("kolega_code.agent.tool_backend.think_hard_tool.get_model_specs") as mock_get_specs:
            # Mock model specs
            mock_get_specs.return_value = {"max_completion_tokens": 8192}

            mock_llm_instance = mock_llm_class.return_value

            # stream method raises an exception
            async def stream_coroutine(*args, **kwargs):
                raise Exception("API Error: Streaming failed")

            mock_llm_instance.stream = stream_coroutine

            # Call think_hard
            result = await think_hard_tool.think_hard("Test problem")

            # Verify error handling
            assert result == "Error during extended thinking: API Error: Streaming failed"
            think_hard_tool.log_error.assert_called_once()
            assert "API Error" in think_hard_tool.log_error.call_args[0][0]


@pytest.mark.asyncio
async def test_think_hard_stream_context_manager_error(think_hard_tool, mock_connection_manager):
    """Test error handling when stream context manager fails."""

    class FailingStreamWrapper:
        async def __aenter__(self):
            raise Exception("Stream initialization failed")

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            return None

    failing_stream = FailingStreamWrapper()

    with patch("kolega_code.agent.tool_backend.think_hard_tool.LLMClient") as mock_llm_class:
        with patch("kolega_code.agent.tool_backend.think_hard_tool.get_model_specs") as mock_get_specs:
            # Mock model specs
            mock_get_specs.return_value = {"max_completion_tokens": 8192}

            mock_llm_instance = mock_llm_class.return_value

            # stream method returns a coroutine that returns the failing stream wrapper
            async def stream_coroutine(*args, **kwargs):
                return failing_stream

            mock_llm_instance.stream = stream_coroutine

            # Call think_hard
            result = await think_hard_tool.think_hard("Test problem")

            # Verify error handling
            assert result == "Error during extended thinking: Stream initialization failed"
            think_hard_tool.log_error.assert_called_once()


@pytest.mark.asyncio
async def test_think_hard_mixed_content_blocks(think_hard_tool, mock_connection_manager):
    """Test think_hard with mixed content blocks including tool calls (should be ignored)."""

    from kolega_code.agent.llm.models import ToolCall

    # Create a mock final message with mixed content types
    final_message = Message(
        role="assistant",
        content=[
            ThinkingBlock(thinking="Analyzing the problem..."),
            TextBlock(text="Here's my analysis:"),
            ToolCall(id="tool_1", name="some_tool", input={"arg": "value"}),  # Should be ignored
            TextBlock(text="Conclusion based on analysis."),
        ],
    )

    # Create mock stream wrapper
    mock_stream = MockStreamWrapper(final_message)

    with patch("kolega_code.agent.tool_backend.think_hard_tool.LLMClient") as mock_llm_class:
        with patch("kolega_code.agent.tool_backend.think_hard_tool.get_model_specs") as mock_get_specs:
            # Mock model specs
            mock_get_specs.return_value = {"max_completion_tokens": 8192}

            mock_llm_instance = mock_llm_class.return_value

            # stream method returns a coroutine that returns the mock stream wrapper
            async def stream_coroutine(*args, **kwargs):
                return mock_stream

            mock_llm_instance.stream = stream_coroutine

            # Call think_hard
            result = await think_hard_tool.think_hard("Complex problem")

            # Verify only thinking and text blocks are included
            expected_result = (
                "# Extended Thinking Process\n\n"
                "Analyzing the problem...\n\n"
                "# Final Analysis\n\n"
                "Here's my analysis:\n"
                "Conclusion based on analysis."
            )
            assert result == expected_result


@pytest.mark.asyncio
async def test_think_hard_empty_response(think_hard_tool, mock_connection_manager):
    """Test think_hard with empty response content."""

    # Create a mock final message with empty content
    final_message = Message(role="assistant", content=[])

    # Create mock stream wrapper
    mock_stream = MockStreamWrapper(final_message)

    with patch("kolega_code.agent.tool_backend.think_hard_tool.LLMClient") as mock_llm_class:
        with patch("kolega_code.agent.tool_backend.think_hard_tool.get_model_specs") as mock_get_specs:
            # Mock model specs
            mock_get_specs.return_value = {"max_completion_tokens": 8192}

            mock_llm_instance = mock_llm_class.return_value

            # stream method returns a coroutine that returns the mock stream wrapper
            async def stream_coroutine(*args, **kwargs):
                return mock_stream

            mock_llm_instance.stream = stream_coroutine

            # Call think_hard
            result = await think_hard_tool.think_hard("Empty response test")

            # Verify the result handles empty content gracefully
            expected_result = "# Final Analysis\n\n"
            assert result == expected_result


@pytest.mark.asyncio
async def test_think_hard_large_thinking_content(think_hard_tool, mock_connection_manager):
    """Test think_hard with very large thinking content (simulating long operations)."""

    # Create a large thinking content
    large_thinking = "\n".join([f"Thinking step {i}: " + "x" * 100 for i in range(100)])

    # Create a mock final message with large thinking content
    final_message = Message(
        role="assistant",
        content=[ThinkingBlock(thinking=large_thinking), TextBlock(text="Final conclusion after extensive thinking.")],
    )

    # Create mock stream wrapper
    mock_stream = MockStreamWrapper(final_message)

    with patch("kolega_code.agent.tool_backend.think_hard_tool.LLMClient") as mock_llm_class:
        with patch("kolega_code.agent.tool_backend.think_hard_tool.get_model_specs") as mock_get_specs:
            # Mock model specs
            mock_get_specs.return_value = {"max_completion_tokens": 8192}

            mock_llm_instance = mock_llm_class.return_value

            # stream method returns a coroutine that returns the mock stream wrapper
            async def stream_coroutine(*args, **kwargs):
                return mock_stream

            mock_llm_instance.stream = stream_coroutine

            # Call think_hard
            result = await think_hard_tool.think_hard("Complex problem requiring extensive thinking")

            # Verify the result contains the large thinking content
            assert "# Extended Thinking Process\n\n" in result
            assert large_thinking in result
            assert "# Final Analysis\n\n" in result
            assert "Final conclusion after extensive thinking." in result


@pytest.mark.asyncio
async def test_think_hard_model_specs_usage(think_hard_tool, mock_connection_manager):
    """Test that model specs are correctly retrieved and used."""

    final_message = Message(role="assistant", content=[TextBlock(text="Response")])

    mock_stream = MockStreamWrapper(final_message)

    with patch("kolega_code.agent.tool_backend.think_hard_tool.LLMClient") as mock_llm_class:
        with patch("kolega_code.agent.tool_backend.think_hard_tool.get_model_specs") as mock_get_specs:
            # Mock model specs
            mock_get_specs.return_value = {"max_completion_tokens": 8192}

            mock_llm_instance = mock_llm_class.return_value

            # stream method returns a coroutine that returns the mock stream wrapper
            async def stream_coroutine(*args, **kwargs):
                return mock_stream

            mock_llm_instance.stream = stream_coroutine

            # Call think_hard
            await think_hard_tool.think_hard("Test")

            # Verify get_model_specs was called correctly
            mock_get_specs.assert_called_once_with(ModelProvider.ANTHROPIC, "claude-3-7-sonnet-20250131")

            # The test passing indicates stream was called with correct parameters
