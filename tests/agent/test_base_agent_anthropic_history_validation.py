# ruff: noqa: F401,F811,E402
import os
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from dotenv import load_dotenv

from kolega_code.agent.baseagent import BaseAgent
from kolega_code.agent.errors import MaxAgentIterationsExceeded
from kolega_code.config import AgentConfig, ModelConfig, ModelProvider, RateLimitConfig
from kolega_code.events import AgentConnectionManager
from kolega_code.llm.exceptions import (
    LLMBillingError,
    LLMAuthenticationError,
    LLMContextWindowExceededError,
    LLMInternalServerError,
    LLMRateLimitError,
)
from kolega_code.llm.models import (
    ImageBlock,
    Message,
    MessageHistory,
    RedactedThinkingBlock,
    TextBlock,
    ThinkingBlock,
    ToolCall,
    ToolResult,
)

from .compaction_helpers import FakeLLM

# Load environment variables
load_dotenv()


class TestBaseAgent:
    def test_is_history_valid_for_anthropic_valid_history(self, base_agent):
        """Test validation with a valid history containing tool calls and results."""
        valid_history = [
            Message(role="user", content=[TextBlock(text="Test message")]),
            Message(role="assistant", content=[TextBlock(text="Response")]),
            Message(role="assistant", content=[ToolCall(id="tool1", name="test_tool", input={})]),
            Message(
                role="user",
                content=[ToolResult(tool_use_id="tool1", name="test_tool", content="Success", is_error=False)],
            ),
        ]

        assert base_agent._is_history_valid_for_anthropic(valid_history) is True

    def test_is_history_valid_for_anthropic_valid_history_no_tools(self, base_agent):
        """Test validation with a valid history containing no tool calls."""
        valid_history = [
            Message(role="user", content=[TextBlock(text="Test message")]),
            Message(role="assistant", content=[TextBlock(text="Response")]),
            Message(role="user", content=[TextBlock(text="Another message")]),
            Message(role="assistant", content=[TextBlock(text="Another response")]),
        ]

        assert base_agent._is_history_valid_for_anthropic(valid_history) is True

    def test_is_history_valid_for_anthropic_missing_tool_result(self, base_agent):
        """Test validation fails when tool call has no corresponding result."""
        invalid_history = [
            Message(role="user", content=[TextBlock(text="Test message")]),
            Message(role="assistant", content=[ToolCall(id="tool1", name="test_tool", input={})]),
            # Missing tool result message
        ]

        assert base_agent._is_history_valid_for_anthropic(invalid_history) is False

    def test_is_history_valid_for_anthropic_incomplete_tool_results(self, base_agent):
        """Test validation fails when some tool calls don't have results."""
        invalid_history = [
            Message(role="user", content=[TextBlock(text="Test message")]),
            Message(
                role="assistant",
                content=[
                    ToolCall(id="tool1", name="test_tool1", input={}),
                    ToolCall(id="tool2", name="test_tool2", input={}),
                ],
            ),
            Message(
                role="user",
                content=[
                    ToolResult(tool_use_id="tool1", name="test_tool1", content="Success", is_error=False)
                    # Missing tool2 result
                ],
            ),
        ]

        assert base_agent._is_history_valid_for_anthropic(invalid_history) is False

    def test_is_history_valid_for_anthropic_wrong_role_sequence(self, base_agent):
        """Test validation fails when tool call is followed by non-user message."""
        invalid_history = [
            Message(role="user", content=[TextBlock(text="Test message")]),
            Message(role="assistant", content=[ToolCall(id="tool1", name="test_tool", input={})]),
            Message(role="assistant", content=[TextBlock(text="Another assistant message")]),  # Should be user
        ]

        assert base_agent._is_history_valid_for_anthropic(invalid_history) is False

    def test_is_history_valid_for_anthropic_empty_history(self, base_agent):
        """Test validation passes for empty history."""
        assert base_agent._is_history_valid_for_anthropic([]) is True

    def test_is_history_valid_for_anthropic_uses_self_history(self, base_agent):
        """Test validation uses self.history when no messages parameter provided."""
        base_agent.history = MessageHistory(
            [
                Message(role="assistant", content=[ToolCall(id="tool1", name="test_tool", input={})]),
                # Missing tool result
            ]
        )

        assert base_agent._is_history_valid_for_anthropic() is False
