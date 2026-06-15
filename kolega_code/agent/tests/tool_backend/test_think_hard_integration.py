"""Integration tests for the think_hard tool."""

import pytest
from unittest.mock import AsyncMock, Mock, patch
from pathlib import Path

from kolega_code.config import AgentConfig, ModelConfig, ModelProvider, RateLimitConfig
from kolega_code.events import AgentConnectionManager
from kolega_code.agent.tool_backend.think_hard_tool import ThinkHardTool


@pytest.fixture
def mock_config():
    """Create a mock agent configuration."""
    return AgentConfig(
        anthropic_api_key="test-key",
        thinking_config=ModelConfig(
            provider=ModelProvider.ANTHROPIC,
            model="claude-3-7-sonnet-20250131",
            rate_limits=RateLimitConfig(requests_per_minute=10, tokens_per_minute=100000, max_retries=3),
            thinking_effort="high",
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
    return tool


@pytest.mark.asyncio
async def test_think_hard_tool_initialization(think_hard_tool):
    """Test that the think_hard tool initializes correctly."""
    assert think_hard_tool.project_path == Path("/test/path")
    assert think_hard_tool.workspace_id == "test_workspace"
    assert think_hard_tool.thread_id == "test_thread"
    assert think_hard_tool.config.thinking_config.thinking_effort == "high"
    assert think_hard_tool.config.thinking_config.provider == ModelProvider.ANTHROPIC


@pytest.mark.asyncio
async def test_think_hard_method_exists(think_hard_tool):
    """Test that the think_hard method exists and is callable."""
    assert hasattr(think_hard_tool, "think_hard")
    assert callable(think_hard_tool.think_hard)


@pytest.mark.asyncio
async def test_think_hard_returns_string(think_hard_tool):
    """Test that think_hard returns a string response."""
    # Mock the logging methods
    think_hard_tool.log_info = AsyncMock()
    think_hard_tool.log_error = AsyncMock()

    # This would require actual API key to test fully, so we'll mock the LLMClient
    with patch("kolega_code.agent.tool_backend.think_hard_tool.LLMClient") as mock_llm_class:
        with patch("kolega_code.agent.tool_backend.think_hard_tool.get_model_specs") as mock_get_specs:
            # Mock model specs
            mock_get_specs.return_value = {"max_completion_tokens": 8192}

            mock_llm_instance = mock_llm_class.return_value

            # Create a simple mock stream that returns without actual API call
            class SimpleStreamMock:
                def __init__(self):
                    self.chunks = []  # No chunks to iterate over
                    self.chunk_index = 0

                async def __aenter__(self):
                    return self

                async def __aexit__(self, exc_type, exc_val, exc_tb):
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

                async def get_final_message(self):
                    from kolega_code.llm.models import Message, TextBlock

                    return Message(role="assistant", content=[TextBlock(text="Test response")])

            # stream method returns a coroutine that returns the mock stream
            async def stream_coroutine(*args, **kwargs):
                return SimpleStreamMock()

            mock_llm_instance.stream = stream_coroutine

            # Call think_hard
            result = await think_hard_tool.think_hard("Test problem")

            # Verify the result is a string
            assert isinstance(result, str)
            assert "# Final Analysis" in result
            assert "Test response" in result


@pytest.mark.asyncio
async def test_think_hard_logging(think_hard_tool):
    """Test that think_hard logs appropriate messages."""
    # Mock the logging methods
    think_hard_tool.log_info = AsyncMock()
    think_hard_tool.log_error = AsyncMock()

    with patch("kolega_code.agent.tool_backend.think_hard_tool.LLMClient") as mock_llm_class:
        with patch("kolega_code.agent.tool_backend.think_hard_tool.get_model_specs") as mock_get_specs:
            # Mock model specs
            mock_get_specs.return_value = {"max_completion_tokens": 8192}

            mock_llm_instance = mock_llm_class.return_value

            # Create a simple mock stream
            class SimpleStreamMock:
                def __init__(self):
                    self.chunks = []  # No chunks to iterate over
                    self.chunk_index = 0

                async def __aenter__(self):
                    return self

                async def __aexit__(self, exc_type, exc_val, exc_tb):
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

                async def get_final_message(self):
                    from kolega_code.llm.models import Message, TextBlock

                    return Message(role="assistant", content=[TextBlock(text="Response")])

            # stream method returns a coroutine that returns the mock stream
            async def stream_coroutine(*args, **kwargs):
                return SimpleStreamMock()

            mock_llm_instance.stream = stream_coroutine

            problem = "This is a very long problem statement that should be truncated in the log message"
            await think_hard_tool.think_hard(problem)

            # Verify logging was called
            think_hard_tool.log_info.assert_called_once()
            log_call = think_hard_tool.log_info.call_args[0][0]
            assert "Thinking hard about:" in log_call
            assert problem[:100] in log_call
