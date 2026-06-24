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
    def test_dump_message_history_empty(self, base_agent):
        """Test dumping an empty message history."""
        base_agent.history = MessageHistory()
        dumped_history = base_agent.dump_message_history()
        assert dumped_history == []

    def test_dump_message_history_populated(self, base_agent):
        """Test dumping a history with various message types using custom to_dict."""
        original_history = MessageHistory(
            [
                Message(role="user", content=[TextBlock(text="Hello")]),
                Message(role="assistant", content=[TextBlock(text="Hi there!")]),
                Message(role="assistant", content=[ToolCall(id="tool1", name="read_file", input={"path": "a.txt"})]),
                Message(
                    role="user",
                    content=[ToolResult(tool_use_id="tool1", name="read_file", content="File content", is_error=False)],
                ),
            ]
        )
        base_agent.history = original_history
        dumped_history = base_agent.dump_message_history()

        assert len(dumped_history) == 4
        assert isinstance(dumped_history[0], dict)
        assert dumped_history[0]["role"] == "user"
        assert isinstance(dumped_history[0]["content"], list)
        assert dumped_history[0]["content"][0]["type"] == "text"
        assert dumped_history[0]["content"][0]["text"] == "Hello"
        assert dumped_history[0]["content"][0]["cache_checkpoint"] is False  # Verify default

        assert isinstance(dumped_history[1], dict)
        assert dumped_history[1]["role"] == "assistant"
        assert dumped_history[1]["content"][0]["type"] == "text"
        assert dumped_history[1]["content"][0]["text"] == "Hi there!"

        assert isinstance(dumped_history[2], dict)
        assert dumped_history[2]["role"] == "assistant"
        assert dumped_history[2]["content"][0]["type"] == "tool_call"
        assert dumped_history[2]["content"][0]["id"] == "tool1"
        assert dumped_history[2]["content"][0]["name"] == "read_file"
        assert dumped_history[2]["content"][0]["input"] == {"path": "a.txt"}

        assert isinstance(dumped_history[3], dict)
        assert dumped_history[3]["role"] == "user"  # Role for ToolResult message
        assert dumped_history[3]["content"][0]["type"] == "tool_result"
        assert dumped_history[3]["content"][0]["tool_use_id"] == "tool1"
        assert dumped_history[3]["content"][0]["content"] == "File content"
        assert dumped_history[3]["content"][0]["name"] == "read_file"
        assert dumped_history[3]["content"][0]["is_error"] is False

        # Check against the actual to_dict output for exact structure validation
        expected_dump = [msg.to_dict() for msg in original_history]
        assert dumped_history == expected_dump

    def test_restore_message_history_empty(self, base_agent):
        """Test restoring an empty message history using custom from_dict."""
        serialized_history = []
        base_agent.restore_message_history(serialized_history)
        assert isinstance(base_agent.history, MessageHistory)
        assert len(base_agent.history) == 0

    def test_restore_message_history_populated(self, base_agent):
        """Test restoring a history with various message types using custom from_dict."""
        # Use the structure produced by to_dict
        serialized_history = [
            {
                "role": "user",
                "content": [{"type": "text", "text": "Another query", "cache_checkpoint": False}],
                "stop_reason": None,
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_call",
                        "id": "tool2",
                        "name": "list_dir",
                        "input": {"path": "/tmp"},
                        "cache_checkpoint": False,
                    }
                ],
                "stop_reason": "tool_use",
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "tool2",
                        "content": "[file1, file2]",
                        "name": "list_dir",
                        "is_error": False,
                        "cache_checkpoint": False,
                    }
                ],
                "stop_reason": None,
            },
        ]

        base_agent.restore_message_history(serialized_history)

        assert isinstance(base_agent.history, MessageHistory)
        assert len(base_agent.history) == 3

        # Validate first message
        msg1 = base_agent.history[0]
        assert isinstance(msg1, Message)
        assert msg1.role == "user"
        assert isinstance(msg1.content[0], TextBlock)
        assert msg1.content[0].text == "Another query"
        assert msg1.stop_reason is None

        # Validate second message (ToolCall)
        msg2 = base_agent.history[1]
        assert isinstance(msg2, Message)
        assert msg2.role == "assistant"
        assert isinstance(msg2.content[0], ToolCall)
        assert msg2.content[0].id == "tool2"
        assert msg2.content[0].name == "list_dir"
        assert msg2.content[0].input == {"path": "/tmp"}
        assert msg2.stop_reason == "tool_use"
        # Check tool_calls attribute is populated correctly
        assert len(msg2.tool_calls) == 1
        assert msg2.tool_calls[0] == msg2.content[0]

        # Validate third message (ToolResult)
        msg3 = base_agent.history[2]
        assert isinstance(msg3, Message)
        assert msg3.role == "user"
        assert isinstance(msg3.content[0], ToolResult)
        assert msg3.content[0].tool_use_id == "tool2"
        assert msg3.content[0].content == "[file1, file2]"
        assert msg3.content[0].name == "list_dir"
        assert msg3.content[0].is_error is False
        assert msg3.stop_reason is None

    def test_restore_message_history_sanitizes_oversized_tool_results(self, base_agent):
        oversized_content = "x" * 100_001
        serialized_history = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "read_entire_file_24",
                        "content": oversized_content,
                        "name": "read_entire_file",
                        "is_error": False,
                        "cache_checkpoint": False,
                    }
                ],
                "stop_reason": None,
            }
        ]

        base_agent.restore_message_history(serialized_history)

        result = base_agent.history[0].content[0]
        assert isinstance(result, ToolResult)
        assert result.tool_use_id == "read_entire_file_24"
        assert result.name == "read_entire_file"
        assert result.is_error is False
        assert len(result.content) < 500
        assert "Tool result omitted from history" in result.content

    def test_dump_restore_cycle(self, base_agent):
        """Test that dumping and then restoring results in the original history using custom methods."""
        original_history = MessageHistory(
            [
                Message(role="user", content=[TextBlock(text="Cycle Test")]),
                Message(
                    role="assistant",
                    content=[
                        ThinkingBlock(thinking="reasoning", signature="provider-signature"),
                        RedactedThinkingBlock(data="encrypted-redacted-reasoning"),
                        TextBlock(text="Acknowledged."),
                    ],
                ),
                Message(role="assistant", content=[ToolCall(id="tool3", name="dummy_tool", input={})]),
                Message(
                    role="user",
                    content=[ToolResult(tool_use_id="tool3", name="dummy_tool", content="Success", is_error=False)],
                ),
            ]
        )
        base_agent.history = original_history

        # Dump the history
        dumped_data = base_agent.dump_message_history()

        # Restore the history
        base_agent.restore_message_history(dumped_data)

        # Assert the restored history matches the original content structure
        # We need a more nuanced comparison since direct object comparison might fail
        # due to new object instances, even if structurally identical.
        assert len(base_agent.history) == len(original_history)
        for restored_msg, original_msg in zip(base_agent.history, original_history):
            # Use the to_dict method for comparing structure
            assert restored_msg.to_dict() == original_msg.to_dict()
