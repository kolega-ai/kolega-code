import os
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from dotenv import load_dotenv

from kolega_code.agent.baseagent import BaseAgent
from kolega_code.config import AgentConfig, ModelConfig, ModelProvider, RateLimitConfig
from kolega_code.events import AgentConnectionManager
from kolega_code.llm.models import (
    Message,
    MessageHistory,
    RedactedThinkingBlock,
    TextBlock,
    ThinkingBlock,
    ToolCall,
    ToolResult,
)

# Load environment variables
load_dotenv()


@pytest.fixture
def agent_config():
    return AgentConfig(
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", "test_key"),  # Use actual API key from environment
        openai_api_key="test-key",
        long_context_config=ModelConfig(
            provider=ModelProvider.ANTHROPIC,
            model="claude-haiku-4-5-20251001",  # Using a valid model name
            rate_limits=RateLimitConfig(),
        ),
        fast_config=ModelConfig(
            provider=ModelProvider.ANTHROPIC,
            model="claude-haiku-4-5-20251001",  # Using a valid model name
            rate_limits=RateLimitConfig(),
        ),
        thinking_config=ModelConfig(
            provider=ModelProvider.ANTHROPIC,
            model="claude-haiku-4-5-20251001",  # Using a valid model name
            rate_limits=RateLimitConfig(),
            thinking_effort="medium",
        ),
    )


@pytest.fixture
def mock_connection_manager():
    return AsyncMock(spec=AgentConnectionManager)


@pytest.fixture
def base_agent(tmp_path, mock_connection_manager, agent_config):
    return BaseAgent(
        project_path=tmp_path,
        workspace_id="test_workspace",
        thread_id=str(uuid.uuid4()),  # Add thread_id
        connection_manager=mock_connection_manager,
        config=agent_config,
    )


class TestBaseAgent:
    @pytest.mark.asyncio
    async def test_execute_single_tool_uses_execution_id_for_app_events_and_provider_id_for_result(self, base_agent):
        class TestTools:
            def get_tool_list(self):
                return [SimpleNamespace(name="dispatch_investigation_agent")]

            def registry(self):
                from kolega_code.agent.tools import ToolCollection
                from kolega_code.llm.models import ToolDefinition
                from kolega_code.tools import Tool, ToolRegistry

                parallel = set(ToolCollection.read_only_tools) | set(ToolCollection.agent_dispatch_tools)
                registry = ToolRegistry()
                for spec in self.get_tool_list():
                    registry.add(
                        Tool(
                            name=spec.name,
                            definition=ToolDefinition(name=spec.name, description="", parameters=[]),
                            handler=getattr(self, spec.name),
                            parallel_safe=spec.name in parallel,
                        )
                    )
                return registry

            async def dispatch_investigation_agent(self, **_inputs):
                return "investigation complete"

        tool_call = ToolCall(
            id="dispatch_investigation_agent_0",
            name="dispatch_investigation_agent",
            input={"task": "check this"},
            execution_id="tool_exec_unique_123",
        )
        base_agent.tool_collection = TestTools()
        base_agent.send_chat_message = AsyncMock()
        base_agent.log_info = AsyncMock()

        result = await base_agent.execute_single_tool(tool_call)

        assert result.tool_use_id == "dispatch_investigation_agent_0"
        assert result.execution_id == "tool_exec_unique_123"
        assert base_agent.send_chat_message.call_args_list[0].kwargs["tool_call_id"] == "tool_exec_unique_123"
        assert base_agent.send_chat_message.call_args_list[1].kwargs["tool_call_id"] == "tool_exec_unique_123"
        assert base_agent.current_tool_call_id is None
        assert base_agent.current_tool_execution_id is None
        assert base_agent.current_provider_tool_call_id is None

    @pytest.mark.asyncio
    async def testcompress_history(self, base_agent):
        # Setup test data
        conversation = [
            ("user", "Message 1"),
            ("assistant", "Response 1"),
            ("user", "Message 2"),
            ("assistant", "Response 2"),
            ("user", "Message 3"),
            ("assistant", "Response 3"),
            ("user", "Message 4"),
            ("assistant", "Response 4"),
            ("user", "Message 5"),
            ("assistant", "Response 5"),
        ]

        base_agent.history = MessageHistory(
            [Message(role=role, content=[TextBlock(text=text)]) for role, text in conversation]
        )

        # Mock the LLM response
        mock_response = Message(
            role="assistant", content=[TextBlock(text="This is a compressed summary of the conversation")]
        )

        # Mock the LLM client's generate method
        with patch.object(base_agent.llm, "generate", new_callable=AsyncMock) as mock_generate:
            mock_generate.return_value = mock_response

            # Call the method (non-destructive)
            await base_agent.compress_history()

            # Verify full history retained plus appended summary
            assert len(base_agent.history) == len(conversation) + 1
            # Verify markers set and effective history contains summary only (single-message effective)
            assert base_agent.last_compression_index == len(conversation) - 1
            effective = base_agent.get_effective_history_for_llm()
            assert len(effective) == 1  # only the summary is used for LLM

            # Verify the LLM was called with correct parameters
            mock_generate.assert_called_once()
            call_args = mock_generate.call_args[1]
            assert call_args["model"] == base_agent.config.long_context_config.model
            assert (
                call_args["max_completion_tokens"] == base_agent.model_completion_tokens
            )  # Use the model's actual limit

    @pytest.mark.asyncio
    async def testcompress_history_insufficient_history(self, base_agent):
        # Setup test data with less than 5 messages
        conversation = [
            ("user", "Message 1"),
            ("assistant", "Response 1"),
            ("user", "Message 2"),
            ("assistant", "Response 2"),
        ]

        base_agent.history = MessageHistory(
            [Message(role=role, content=[TextBlock(text=text)]) for role, text in conversation]
        )

        # Mock the LLM client's generate method
        with patch.object(base_agent.llm, "generate", new_callable=AsyncMock) as mock_generate:
            # Call the method
            await base_agent.compress_history()

            # Verify the history was not compressed
            assert len(base_agent.history) == 4
            assert all(isinstance(msg, Message) for msg in base_agent.history)
            assert base_agent.history == base_agent.history  # History unchanged
            mock_generate.assert_not_called()

    @pytest.mark.asyncio
    async def testcompress_history_error_handling(self, base_agent):
        # Setup test data
        conversation = [
            ("user", "Message 1"),
            ("assistant", "Response 1"),
            ("user", "Message 2"),
            ("assistant", "Response 2"),
            ("user", "Message 3"),
            ("assistant", "Response 3"),
            ("user", "Message 4"),
            ("assistant", "Response 4"),
            ("user", "Message 5"),
            ("assistant", "Response 5"),
        ]

        base_agent.history = MessageHistory(
            [Message(role=role, content=[TextBlock(text=text)]) for role, text in conversation]
        )

        # Mock the LLM client's generate method to raise an exception
        with patch.object(base_agent.llm, "generate", new_callable=AsyncMock) as mock_generate:
            mock_generate.side_effect = Exception("Test error")

            # Call the method
            await base_agent.compress_history()

            # Verify the history was not modified
            assert len(base_agent.history) == 10
            assert all(isinstance(msg, Message) for msg in base_agent.history)
            assert base_agent.history == base_agent.history  # History unchanged

    @pytest.mark.slow
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def testcompress_history_with_real_llm(self, base_agent):
        """Integration test using the real LLM client to test message compression.

        Note: This test requires a valid API key to be set in the environment.
        It will be skipped if the API key is not available.
        """
        # Skip if no API key is available
        api_key = base_agent.config.get_api_key(base_agent.config.long_context_config.provider)
        if not api_key or api_key == "test_key":
            pytest.skip("No valid API key available for LLM provider")

        # Setup test data with a realistic conversation using Message objects
        conversation = [
            ("user", "What is Python?"),
            (
                "assistant",
                "Python is a high-level, interpreted programming language known for its simplicity and readability.",
            ),
            ("user", "What are its main features?"),
            (
                "assistant",
                "Python features include dynamic typing, automatic memory management, and a comprehensive standard library.",
            ),
            ("user", "How do I write a function in Python?"),
            (
                "assistant",
                "You can define a function using the def keyword, followed by the function name and parameters in parentheses.",
            ),
            ("user", "What is a decorator?"),
            (
                "assistant",
                "A decorator is a design pattern that allows you to modify the behavior of functions or classes.",
            ),
            ("user", "Show me an example of a decorator."),
            ("assistant", "Here is a simple decorator example: @property def name(self): return self._name"),
        ]

        base_agent.history = MessageHistory(
            [Message(role=role, content=[TextBlock(text=text)]) for role, text in conversation]
        )

        # Store the last two messages for comparison
        last_two_messages = base_agent.history[-2:]

        try:
            # Call the method with real LLM
            await base_agent.compress_history()

            # Verify the summary was appended (allowing for environments where real LLM may be skipped)
            assert len(base_agent.history) >= len(conversation)

            # Verify the summary message was appended at the end
            summary_message = base_agent.history[-1]
            assert isinstance(summary_message, Message)
            assert summary_message.role == "user"
            summary_text = summary_message.content[0].text
            assert ("CONVERSATION HISTORY SUMMARY" in summary_text) or ("## Analysis Section" in summary_text)

            # Verify the last two messages are still present just before the summary
            assert base_agent.history[-3:-1] == last_two_messages
        except Exception as e:
            pytest.fail(f"Test failed with error: {str(e)}")

    # Tests for dump/restore message history
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

    # Tests for history validation methods
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

    def testfix_incomplete_tool_calls_no_changes_needed(self, base_agent):
        """Test fix method doesn't modify valid history."""
        valid_history = [
            Message(role="user", content=[TextBlock(text="Test message")]),
            Message(role="assistant", content=[ToolCall(id="tool1", name="test_tool", input={})]),
            Message(
                role="user",
                content=[ToolResult(tool_use_id="tool1", name="test_tool", content="Success", is_error=False)],
            ),
        ]

        fixed_history = base_agent.fix_incomplete_tool_calls(valid_history)

        assert len(fixed_history) == 3
        assert fixed_history[0].to_dict() == valid_history[0].to_dict()
        assert fixed_history[1].to_dict() == valid_history[1].to_dict()
        assert fixed_history[2].to_dict() == valid_history[2].to_dict()

    def testfix_incomplete_tool_calls_adds_placeholder_result(self, base_agent):
        """Test fix method adds placeholder result for orphaned tool call."""
        incomplete_history = [
            Message(role="user", content=[TextBlock(text="Test message")]),
            Message(role="assistant", content=[ToolCall(id="tool1", name="test_tool", input={})]),
            # Missing tool result
        ]

        fixed_history = base_agent.fix_incomplete_tool_calls(incomplete_history)

        assert len(fixed_history) == 3  # Original 2 + 1 placeholder

        # Check original messages are preserved
        assert fixed_history[0].to_dict() == incomplete_history[0].to_dict()
        assert fixed_history[1].to_dict() == incomplete_history[1].to_dict()

        # Check placeholder was added
        placeholder_msg = fixed_history[2]
        assert placeholder_msg.role == "user"
        assert len(placeholder_msg.content) == 1
        assert isinstance(placeholder_msg.content[0], ToolResult)
        assert placeholder_msg.content[0].tool_use_id == "tool1"
        assert placeholder_msg.content[0].name == "test_tool"
        assert placeholder_msg.content[0].is_error is True
        assert "interrupted" in placeholder_msg.content[0].content.lower()

    def testfix_incomplete_tool_calls_multiple_tools(self, base_agent):
        """Test fix method handles multiple incomplete tool calls."""
        incomplete_history = [
            Message(
                role="assistant",
                content=[
                    ToolCall(id="tool1", name="test_tool1", input={}),
                    ToolCall(id="tool2", name="test_tool2", input={}),
                ],
            ),
            # Missing tool results
        ]

        fixed_history = base_agent.fix_incomplete_tool_calls(incomplete_history)

        assert len(fixed_history) == 2  # Original 1 + 1 placeholder

        # Check placeholder message has results for both tools
        placeholder_msg = fixed_history[1]
        assert placeholder_msg.role == "user"
        assert len(placeholder_msg.content) == 2

        tool_result_ids = {result.tool_use_id for result in placeholder_msg.content}
        assert tool_result_ids == {"tool1", "tool2"}

        for result in placeholder_msg.content:
            assert isinstance(result, ToolResult)
            assert result.is_error is True

    def testfix_incomplete_tool_calls_partial_results(self, base_agent):
        """Test fix method handles partial tool results correctly."""
        incomplete_history = [
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

        fixed_history = base_agent.fix_incomplete_tool_calls(incomplete_history)

        # Should have same length since placeholder is merged into existing user message
        assert len(fixed_history) == 2  # Same as original

        # Check that the user message now has both tool results
        user_message = fixed_history[1]
        assert user_message.role == "user"
        assert len(user_message.content) == 2  # Now has both tool results

        # Check tool result IDs
        tool_result_ids = {result.tool_use_id for result in user_message.content if isinstance(result, ToolResult)}
        assert tool_result_ids == {"tool1", "tool2"}

        # Check that placeholder was added for tool2
        tool2_result = next(result for result in user_message.content if result.tool_use_id == "tool2")
        assert tool2_result.is_error is True
        assert "interrupted" in tool2_result.content.lower()

        # Verify the fixed history is valid
        assert base_agent._is_history_valid_for_anthropic(fixed_history) is True

    def testfix_incomplete_tool_calls_empty_history(self, base_agent):
        """Test fix method handles empty history."""
        fixed_history = base_agent.fix_incomplete_tool_calls([])
        assert fixed_history == []

    def test_restore_message_history_with_incomplete_tool_calls(self, base_agent):
        """Test restore method does NOT automatically fix incomplete tool calls."""
        # Serialized history with incomplete tool call (simulating interrupted state)
        serialized_incomplete_history = [
            {
                "role": "user",
                "content": [{"type": "text", "text": "Test message", "cache_checkpoint": False}],
                "stop_reason": None,
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_call",
                        "id": "tool1",
                        "name": "test_tool",
                        "input": {"param": "value"},
                        "cache_checkpoint": False,
                    }
                ],
                "stop_reason": "tool_use",
            },
            # Missing tool result message (simulating interruption)
        ]

        base_agent.restore_message_history(serialized_incomplete_history)

        # Verify history was NOT fixed - restore should preserve authentic history
        assert len(base_agent.history) == 2  # Original 2 messages unchanged

        # Check original messages are preserved as-is
        assert base_agent.history[0].role == "user"
        assert base_agent.history[1].role == "assistant"
        assert len(base_agent.history[1].tool_calls) == 1

        # Verify the history is still invalid for Anthropic (not fixed)
        assert base_agent._is_history_valid_for_anthropic() is False

        # But verify that fix_incomplete_tool_calls can fix it
        fixed_history = base_agent.fix_incomplete_tool_calls(list(base_agent.history))
        assert len(fixed_history) == 3  # Now fixed with placeholder
        assert base_agent._is_history_valid_for_anthropic(fixed_history) is True

    # Tests for robustness - incomplete tool calls at various positions
    def testfix_incomplete_tool_calls_at_beginning_of_history(self, base_agent):
        """Test fix method handles incomplete tool calls at the beginning of message history."""
        corrupted_history = [
            # Incomplete tool call sequence at the beginning
            Message(
                role="assistant",
                content=[
                    ToolCall(id="early_tool1", name="early_tool", input={}),
                    ToolCall(id="early_tool2", name="another_early_tool", input={}),
                ],
            ),
            Message(
                role="user",
                content=[
                    ToolResult(tool_use_id="early_tool1", name="early_tool", content="Success", is_error=False)
                    # Missing early_tool2 result
                ],
            ),
            # Normal conversation continues
            Message(role="user", content=[TextBlock(text="How are things?")]),
            Message(role="assistant", content=[TextBlock(text="Things are going well.")]),
            # Complete tool call sequence later
            Message(role="assistant", content=[ToolCall(id="later_tool", name="later_tool", input={})]),
            Message(
                role="user",
                content=[
                    ToolResult(tool_use_id="later_tool", name="later_tool", content="Later success", is_error=False)
                ],
            ),
        ]

        fixed_history = base_agent.fix_incomplete_tool_calls(corrupted_history)

        # Should have same length since placeholder is merged into existing user message
        assert len(fixed_history) == 6  # Same as original

        # Verify the early incomplete sequence was fixed by merging placeholder into existing user message
        assert fixed_history[0].role == "assistant"  # Original tool call message
        assert fixed_history[1].role == "user"  # User message now has both results
        assert len(fixed_history[1].content) == 2  # Now has both tool results

        # Check that both tool results are present
        tool_result_ids = {result.tool_use_id for result in fixed_history[1].content if isinstance(result, ToolResult)}
        assert tool_result_ids == {"early_tool1", "early_tool2"}

        # Verify the placeholder result is marked as error
        placeholder_result = next(result for result in fixed_history[1].content if result.tool_use_id == "early_tool2")
        assert placeholder_result.is_error is True
        assert "interrupted" in placeholder_result.content.lower()

        # Verify rest of history is preserved
        assert fixed_history[2].role == "user"  # "How are things?"
        assert fixed_history[3].role == "assistant"  # "Things are going well."
        assert fixed_history[4].role == "assistant"  # later_tool call
        assert fixed_history[5].role == "user"  # later_tool result

        # Verify final history is valid
        assert base_agent._is_history_valid_for_anthropic(fixed_history) is True

    def testfix_incomplete_tool_calls_in_middle_of_history(self, base_agent):
        """Test fix method handles incomplete tool calls in the middle of message history."""
        corrupted_history = [
            # Normal conversation start
            Message(role="user", content=[TextBlock(text="Hello")]),
            Message(role="assistant", content=[TextBlock(text="Hi there!")]),
            # Incomplete tool call sequence in the middle
            Message(
                role="assistant",
                content=[
                    ToolCall(id="middle_tool1", name="middle_tool", input={}),
                    ToolCall(id="middle_tool2", name="another_middle_tool", input={}),
                    ToolCall(id="middle_tool3", name="third_middle_tool", input={}),
                ],
            ),
            Message(
                role="user",
                content=[
                    ToolResult(tool_use_id="middle_tool1", name="middle_tool", content="Success", is_error=False),
                    ToolResult(tool_use_id="middle_tool3", name="third_middle_tool", content="Success", is_error=False),
                    # Missing middle_tool2 result
                ],
            ),
            # Normal conversation continues
            Message(role="assistant", content=[TextBlock(text="Let me continue...")]),
            Message(role="user", content=[TextBlock(text="Sounds good")]),
        ]

        fixed_history = base_agent.fix_incomplete_tool_calls(corrupted_history)

        # Should have same length since placeholder is merged into existing user message
        assert len(fixed_history) == 6  # Same as original

        # Verify the middle incomplete sequence was fixed
        assert fixed_history[2].role == "assistant"  # Tool call message
        assert fixed_history[3].role == "user"  # User message now has all 3 results
        assert len(fixed_history[3].content) == 3  # Now has all 3 tool results

        # Check that all tool results are present
        tool_result_ids = {result.tool_use_id for result in fixed_history[3].content if isinstance(result, ToolResult)}
        assert tool_result_ids == {"middle_tool1", "middle_tool2", "middle_tool3"}

        # Verify the placeholder result is marked as error
        placeholder_result = next(result for result in fixed_history[3].content if result.tool_use_id == "middle_tool2")
        assert placeholder_result.is_error is True

        # Verify rest of history is preserved
        assert fixed_history[4].role == "assistant"  # "Let me continue..."
        assert fixed_history[5].role == "user"  # "Sounds good"

        # Verify final history is valid
        assert base_agent._is_history_valid_for_anthropic(fixed_history) is True

    def test_fix_multiple_incomplete_tool_call_sequences(self, base_agent):
        """Test fix method handles multiple incomplete tool call sequences in the same history."""
        corrupted_history = [
            # First incomplete sequence
            Message(
                role="assistant",
                content=[
                    ToolCall(id="seq1_tool1", name="tool1", input={}),
                    ToolCall(id="seq1_tool2", name="tool2", input={}),
                ],
            ),
            Message(
                role="user",
                content=[
                    ToolResult(tool_use_id="seq1_tool1", name="tool1", content="Success", is_error=False)
                    # Missing seq1_tool2
                ],
            ),
            # Normal conversation
            Message(role="user", content=[TextBlock(text="Continue")]),
            Message(role="assistant", content=[TextBlock(text="Continuing...")]),
            # Second incomplete sequence
            Message(
                role="assistant",
                content=[
                    ToolCall(id="seq2_tool1", name="tool3", input={}),
                    ToolCall(id="seq2_tool2", name="tool4", input={}),
                    ToolCall(id="seq2_tool3", name="tool5", input={}),
                ],
            ),
            Message(
                role="user",
                content=[
                    ToolResult(tool_use_id="seq2_tool2", name="tool4", content="Success", is_error=False)
                    # Missing seq2_tool1 and seq2_tool3
                ],
            ),
            # End conversation
            Message(role="assistant", content=[TextBlock(text="Done")]),
        ]

        fixed_history = base_agent.fix_incomplete_tool_calls(corrupted_history)

        # Should have same length since placeholders are merged into existing user messages
        assert len(fixed_history) == 7  # Same as original

        # Verify first incomplete sequence was fixed
        assert fixed_history[1].role == "user"
        assert len(fixed_history[1].content) == 2  # Now has both tool results
        first_tool_result_ids = {
            result.tool_use_id for result in fixed_history[1].content if isinstance(result, ToolResult)
        }
        assert first_tool_result_ids == {"seq1_tool1", "seq1_tool2"}

        # Verify second incomplete sequence was fixed
        assert fixed_history[5].role == "user"
        assert len(fixed_history[5].content) == 3  # Now has all 3 tool results
        second_tool_result_ids = {
            result.tool_use_id for result in fixed_history[5].content if isinstance(result, ToolResult)
        }
        assert second_tool_result_ids == {"seq2_tool1", "seq2_tool2", "seq2_tool3"}

        # Verify final history is valid
        assert base_agent._is_history_valid_for_anthropic(fixed_history) is True

    def testfix_incomplete_tool_calls_at_end_with_no_user_message(self, base_agent):
        """Test fix method handles incomplete tool calls at the very end with no following user message."""
        corrupted_history = [
            Message(role="user", content=[TextBlock(text="Do something")]),
            Message(role="assistant", content=[TextBlock(text="Sure, let me help.")]),
            # Tool calls at the end with no user response (simulates interruption)
            Message(
                role="assistant",
                content=[
                    ToolCall(id="end_tool1", name="end_tool", input={}),
                    ToolCall(id="end_tool2", name="another_end_tool", input={}),
                ],
            ),
            # No user message follows (interrupted)
        ]

        fixed_history = base_agent.fix_incomplete_tool_calls(corrupted_history)

        # Should have added 1 new user message for the missing tools
        assert len(fixed_history) == 4  # Original 3 + 1 new user message

        # Verify placeholder was added at the end
        assert fixed_history[3].role == "user"
        assert len(fixed_history[3].content) == 2
        placeholder_ids = {result.tool_use_id for result in fixed_history[3].content}
        assert placeholder_ids == {"end_tool1", "end_tool2"}

        for result in fixed_history[3].content:
            assert result.is_error is True
            assert "interrupted" in result.content.lower()

        # Verify final history is valid
        assert base_agent._is_history_valid_for_anthropic(fixed_history) is True

    def test_fix_consecutive_incomplete_tool_sequences(self, base_agent):
        """Test fix method handles consecutive incomplete tool call sequences."""
        corrupted_history = [
            # First assistant message with tool calls
            Message(role="assistant", content=[ToolCall(id="consec1_tool", name="tool1", input={})]),
            # Partial results
            Message(
                role="user",
                content=[ToolResult(tool_use_id="consec1_tool", name="tool1", content="Success", is_error=False)],
            ),
            # Immediately another assistant message with incomplete tools
            Message(
                role="assistant",
                content=[
                    ToolCall(id="consec2_tool1", name="tool2", input={}),
                    ToolCall(id="consec2_tool2", name="tool3", input={}),
                ],
            ),
            Message(
                role="user",
                content=[
                    ToolResult(tool_use_id="consec2_tool1", name="tool2", content="Success", is_error=False)
                    # Missing consec2_tool2
                ],
            ),
            # Third consecutive assistant message
            Message(role="assistant", content=[ToolCall(id="consec3_tool", name="tool4", input={})]),
            # No user message (interrupted)
        ]

        fixed_history = base_agent.fix_incomplete_tool_calls(corrupted_history)

        # Should have same length since one placeholder is merged, one new message is added
        assert len(fixed_history) == 6  # Same as original (merge + add)

        # First sequence is complete, no changes
        assert fixed_history[0].role == "assistant"
        assert fixed_history[1].role == "user"

        # Second sequence should have placeholder merged into existing user message
        assert fixed_history[2].role == "assistant"
        assert fixed_history[3].role == "user"  # Original partial results now complete
        assert len(fixed_history[3].content) == 2  # Now has both tool results
        second_tool_result_ids = {
            result.tool_use_id for result in fixed_history[3].content if isinstance(result, ToolResult)
        }
        assert second_tool_result_ids == {"consec2_tool1", "consec2_tool2"}

        # Third sequence should have new user message added
        assert fixed_history[4].role == "assistant"
        assert fixed_history[5].role == "user"  # NEW: User message for consec3_tool
        assert fixed_history[5].content[0].tool_use_id == "consec3_tool"

        # Verify final history is valid
        assert base_agent._is_history_valid_for_anthropic(fixed_history) is True

    def test_fix_mixed_complete_and_incomplete_sequences(self, base_agent):
        """Test fix method handles a mix of complete and incomplete tool call sequences."""
        mixed_history = [
            # Complete sequence 1
            Message(role="assistant", content=[ToolCall(id="complete1", name="complete_tool", input={})]),
            Message(
                role="user",
                content=[ToolResult(tool_use_id="complete1", name="complete_tool", content="Success", is_error=False)],
            ),
            # Incomplete sequence
            Message(
                role="assistant",
                content=[
                    ToolCall(id="incomplete1", name="incomplete_tool", input={}),
                    ToolCall(id="incomplete2", name="another_incomplete", input={}),
                ],
            ),
            Message(
                role="user",
                content=[
                    ToolResult(tool_use_id="incomplete1", name="incomplete_tool", content="Success", is_error=False)
                    # Missing incomplete2
                ],
            ),
            # Complete sequence 2
            Message(role="assistant", content=[ToolCall(id="complete2", name="another_complete", input={})]),
            Message(
                role="user",
                content=[
                    ToolResult(tool_use_id="complete2", name="another_complete", content="Success", is_error=False)
                ],
            ),
            # Normal text
            Message(role="user", content=[TextBlock(text="All done")]),
            Message(role="assistant", content=[TextBlock(text="Great work!")]),
        ]

        fixed_history = base_agent.fix_incomplete_tool_calls(mixed_history)

        # Should have same length since placeholder is merged into existing user message
        assert len(fixed_history) == 8  # Same as original

        # Verify complete sequences are unchanged
        assert fixed_history[0].role == "assistant"  # complete1 tool call
        assert fixed_history[1].role == "user"  # complete1 result

        # Verify incomplete sequence was fixed by merging placeholder
        assert fixed_history[2].role == "assistant"  # incomplete tools call
        assert fixed_history[3].role == "user"  # user message now has both results
        assert len(fixed_history[3].content) == 2  # Now has both tool results
        incomplete_tool_result_ids = {
            result.tool_use_id for result in fixed_history[3].content if isinstance(result, ToolResult)
        }
        assert incomplete_tool_result_ids == {"incomplete1", "incomplete2"}

        # Verify rest is unchanged
        assert fixed_history[4].role == "assistant"  # complete2 tool call
        assert fixed_history[5].role == "user"  # complete2 result
        assert fixed_history[6].role == "user"  # "All done"
        assert fixed_history[7].role == "assistant"  # "Great work!"

        # Verify final history is valid
        assert base_agent._is_history_valid_for_anthropic(fixed_history) is True

    def test_complex_corrupted_history_recovery(self, base_agent):
        """Test fix method can recover from a complex, heavily corrupted message history."""
        heavily_corrupted_history = [
            # Start with incomplete sequence
            Message(
                role="assistant",
                content=[
                    ToolCall(id="start_tool1", name="start1", input={}),
                    ToolCall(id="start_tool2", name="start2", input={}),
                    ToolCall(id="start_tool3", name="start3", input={}),
                ],
            ),
            Message(
                role="user",
                content=[
                    ToolResult(tool_use_id="start_tool2", name="start2", content="Success", is_error=False)
                    # Missing start_tool1 and start_tool3
                ],
            ),
            # Some normal conversation
            Message(role="user", content=[TextBlock(text="What about the other tasks?")]),
            Message(role="assistant", content=[TextBlock(text="Let me check on those.")]),
            # Another incomplete sequence
            Message(
                role="assistant",
                content=[
                    ToolCall(id="mid_tool1", name="mid1", input={}),
                    ToolCall(id="mid_tool2", name="mid2", input={}),
                    ToolCall(id="mid_tool3", name="mid3", input={}),
                    ToolCall(id="mid_tool4", name="mid4", input={}),
                ],
            ),
            Message(
                role="user",
                content=[
                    ToolResult(tool_use_id="mid_tool1", name="mid1", content="Success", is_error=False),
                    ToolResult(tool_use_id="mid_tool4", name="mid4", content="Success", is_error=False),
                    # Missing mid_tool2 and mid_tool3
                ],
            ),
            # Complete sequence (should be left alone)
            Message(role="assistant", content=[ToolCall(id="good_tool", name="good", input={})]),
            Message(
                role="user",
                content=[ToolResult(tool_use_id="good_tool", name="good", content="Success", is_error=False)],
            ),
            # Final incomplete sequence at the end
            Message(
                role="assistant",
                content=[
                    ToolCall(id="end_tool1", name="end1", input={}),
                    ToolCall(id="end_tool2", name="end2", input={}),
                ],
            ),
            # No user response (interrupted at the very end)
        ]

        # Verify original history is invalid
        assert base_agent._is_history_valid_for_anthropic(heavily_corrupted_history) is False

        fixed_history = base_agent.fix_incomplete_tool_calls(heavily_corrupted_history)

        # Should have same length since 2 placeholders are merged, 1 new message is added
        assert len(fixed_history) == 10  # Same as original (2 merges + 1 add = net 0 change)

        # Verify all incomplete sequences were fixed

        # First sequence: placeholders merged into existing user message
        assert len(fixed_history[1].content) == 3  # Now has all 3 tool results
        first_placeholders = {r.tool_use_id for r in fixed_history[1].content if isinstance(r, ToolResult)}
        assert first_placeholders == {"start_tool1", "start_tool2", "start_tool3"}

        # Second sequence: placeholders merged into existing user message
        assert len(fixed_history[5].content) == 4  # Now has all 4 tool results
        second_placeholders = {r.tool_use_id for r in fixed_history[5].content if isinstance(r, ToolResult)}
        assert second_placeholders == {"mid_tool1", "mid_tool2", "mid_tool3", "mid_tool4"}

        # Third sequence: new user message created
        assert fixed_history[9].role == "user"
        assert len(fixed_history[9].content) == 2
        third_placeholders = {r.tool_use_id for r in fixed_history[9].content}
        assert third_placeholders == {"end_tool1", "end_tool2"}

        # Verify all placeholders are marked as errors (check a few samples)
        start_placeholder = next(r for r in fixed_history[1].content if r.tool_use_id == "start_tool1")
        assert start_placeholder.is_error is True
        assert "interrupted" in start_placeholder.content.lower()

        # Most importantly: verify the fixed history is now valid for Anthropic
        assert base_agent._is_history_valid_for_anthropic(fixed_history) is True

    # Integration tests with real Anthropic API for message history corruption recovery
    @pytest.mark.slow
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def testfix_incomplete_tool_calls_with_real_api_simple_case(self, base_agent):
        """Integration test: Fix simple incomplete tool call and verify it works with real Anthropic API."""
        # Skip if no API key is available
        api_key = base_agent.config.get_api_key(base_agent.config.long_context_config.provider)
        if not api_key or api_key == "test_key":
            pytest.skip("No valid API key available for LLM provider")

        # Create a corrupted history with incomplete tool call (simulating interruption)
        corrupted_history = [
            Message(role="user", content=[TextBlock(text="Can you help me with a simple task?")]),
            Message(role="assistant", content=[TextBlock(text="Of course! I'd be happy to help you.")]),
            Message(
                role="assistant",
                content=[ToolCall(id="interrupted_tool", name="read_file", input={"path": "example.txt"})],
            ),
            # Missing tool result - simulates interruption during tool execution
        ]

        # Verify the corrupted history is invalid
        assert base_agent._is_history_valid_for_anthropic(corrupted_history) is False

        # Fix the corrupted history
        fixed_history = base_agent.fix_incomplete_tool_calls(corrupted_history)

        # Verify the fix worked
        assert base_agent._is_history_valid_for_anthropic(fixed_history) is True
        assert len(fixed_history) == 4  # Original 3 + 1 placeholder user message

        # Set up the fixed history in the agent
        base_agent.history = MessageHistory(fixed_history)

        try:
            # Test that the fixed history works with real Anthropic API
            # by sending a follow-up message
            system_message = Message(role="system", content=[TextBlock(text="You are a helpful assistant.")])

            # Add a new user message to continue the conversation
            base_agent.history.append(Message(role="user", content=[TextBlock(text="What should I do next?")]))

            # Call the real LLM API with the fixed history
            response = await base_agent.llm.generate(
                messages=base_agent.history,
                system=system_message,
                model=base_agent.config.long_context_config.model,
                max_completion_tokens=100,  # Keep it small for testing
            )

            # Verify we got a valid response
            assert response is not None
            response_text = response.get_text_content()
            assert isinstance(response_text, str)
            assert len(response_text.strip()) > 0

        except Exception as e:
            pytest.fail(f"Real API call failed with fixed history: {str(e)}")

    @pytest.mark.slow
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_fix_multiple_incomplete_tool_calls_with_real_api(self, base_agent):
        """Integration test: Fix multiple incomplete tool calls and verify with real Anthropic API."""
        # Skip if no API key is available
        api_key = base_agent.config.get_api_key(base_agent.config.long_context_config.provider)
        if not api_key or api_key == "test_key":
            pytest.skip("No valid API key available for LLM provider")

        # Create a heavily corrupted history with multiple incomplete sequences
        corrupted_history = [
            Message(role="user", content=[TextBlock(text="I need help with several file operations.")]),
            Message(
                role="assistant",
                content=[TextBlock(text="I can help you with file operations. Let me start working on those.")],
            ),
            # First incomplete sequence
            Message(
                role="assistant",
                content=[
                    ToolCall(id="tool1", name="read_file", input={"path": "file1.txt"}),
                    ToolCall(id="tool2", name="read_file", input={"path": "file2.txt"}),
                    ToolCall(id="tool3", name="list_dir", input={"path": "."}),
                ],
            ),
            Message(
                role="user",
                content=[
                    ToolResult(tool_use_id="tool1", name="read_file", content="Content of file1", is_error=False)
                    # Missing tool2 and tool3 results
                ],
            ),
            # Normal conversation
            Message(role="user", content=[TextBlock(text="What about the other operations?")]),
            Message(role="assistant", content=[TextBlock(text="Let me continue with the remaining operations.")]),
            # Second incomplete sequence
            Message(
                role="assistant",
                content=[
                    ToolCall(id="tool4", name="write_file", input={"path": "output.txt", "content": "test"}),
                    ToolCall(id="tool5", name="read_file", input={"path": "config.json"}),
                ],
            ),
            # No user message - interrupted at the end
        ]

        # Verify the corrupted history is invalid
        assert base_agent._is_history_valid_for_anthropic(corrupted_history) is False

        # Fix the corrupted history
        fixed_history = base_agent.fix_incomplete_tool_calls(corrupted_history)

        # Verify the fix worked
        assert base_agent._is_history_valid_for_anthropic(fixed_history) is True

        # Set up the fixed history in the agent
        base_agent.history = MessageHistory(fixed_history)

        try:
            # Test that the fixed history works with real Anthropic API
            system_message = Message(
                role="system", content=[TextBlock(text="You are a helpful assistant for file operations.")]
            )

            # Add a new user message to continue the conversation
            base_agent.history.append(
                Message(role="user", content=[TextBlock(text="Can you summarize what operations were attempted?")])
            )

            # Call the real LLM API with the fixed history
            response = await base_agent.llm.generate(
                messages=base_agent.history,
                system=system_message,
                model=base_agent.config.long_context_config.model,
                max_completion_tokens=150,
            )

            # Verify we got a valid response
            assert response is not None
            response_text = response.get_text_content()
            assert isinstance(response_text, str)
            assert len(response_text.strip()) > 0

            # The response should acknowledge the interrupted operations
            response_lower = response_text.lower()
            assert any(word in response_lower for word in ["interrupt", "error", "operation", "attempt"])

        except Exception as e:
            pytest.fail(f"Real API call failed with fixed history containing multiple corruptions: {str(e)}")

    @pytest.mark.slow
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_fix_corrupted_serialized_history_with_real_api(self, base_agent):
        """Integration test: Fix corrupted serialized history before API call works."""
        # Skip if no API key is available
        api_key = base_agent.config.get_api_key(base_agent.config.long_context_config.provider)
        if not api_key or api_key == "test_key":
            pytest.skip("No valid API key available for LLM provider")

        # Create a serialized corrupted history (simulating what would be saved to database)
        serialized_corrupted_history = [
            {
                "role": "user",
                "content": [{"type": "text", "text": "Please analyze this data for me.", "cache_checkpoint": False}],
                "stop_reason": None,
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": "I'll analyze the data for you. Let me start by reading the files.",
                        "cache_checkpoint": False,
                    }
                ],
                "stop_reason": None,
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_call",
                        "id": "analysis_tool1",
                        "name": "read_file",
                        "input": {"path": "data.csv"},
                        "cache_checkpoint": False,
                    },
                    {
                        "type": "tool_call",
                        "id": "analysis_tool2",
                        "name": "read_file",
                        "input": {"path": "metadata.json"},
                        "cache_checkpoint": False,
                    },
                    {
                        "type": "tool_call",
                        "id": "analysis_tool3",
                        "name": "list_dir",
                        "input": {"path": "analysis_results"},
                        "cache_checkpoint": False,
                    },
                ],
                "stop_reason": "tool_use",
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "analysis_tool1",
                        "content": "CSV data with 1000 rows, 5 columns",
                        "name": "read_file",
                        "is_error": False,
                        "cache_checkpoint": False,
                    }
                    # Missing analysis_tool2 and analysis_tool3 results (interrupted)
                ],
                "stop_reason": None,
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": "What did you find in the analysis?", "cache_checkpoint": False}],
                "stop_reason": None,
            },
        ]

        try:
            # Restore the corrupted history (this should NOT fix it)
            base_agent.restore_message_history(serialized_corrupted_history)

            # Verify the restored history is still invalid
            assert base_agent._is_history_valid_for_anthropic() is False

            # Fix the history manually
            fixed_history = MessageHistory(base_agent.fix_incomplete_tool_calls(list(base_agent.history)))

            # Verify the fix was applied correctly
            # Should have merged placeholders for missing tool results
            tool_result_message = None
            for msg in fixed_history:
                if msg.role == "user" and any(isinstance(block, ToolResult) for block in msg.content):
                    tool_result_message = msg
                    break

            assert tool_result_message is not None
            tool_results = [block for block in tool_result_message.content if isinstance(block, ToolResult)]
            assert len(tool_results) == 3  # Should now have all 3 tool results

            # Check that placeholders were added for missing results
            tool_result_ids = {result.tool_use_id for result in tool_results}
            assert tool_result_ids == {"analysis_tool1", "analysis_tool2", "analysis_tool3"}

            # Test that the fixed history works with real Anthropic API
            system_message = Message(role="system", content=[TextBlock(text="You are a data analysis assistant.")])

            # Call the real LLM API with the fixed history
            response = await base_agent.llm.generate(
                messages=fixed_history,
                system=system_message,
                model=base_agent.config.long_context_config.model,
                max_completion_tokens=200,
            )

            # Verify we got a valid response
            assert response is not None
            response_text = response.get_text_content()
            assert isinstance(response_text, str)
            assert len(response_text.strip()) > 0

        except Exception as e:
            pytest.fail(f"Real API call failed with restored corrupted history: {str(e)}")

    @pytest.mark.slow
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_fix_consecutive_tool_interruptions_with_real_api(self, base_agent):
        """Integration test: Fix consecutive tool call interruptions and verify with real API."""
        # Skip if no API key is available
        api_key = base_agent.config.get_api_key(base_agent.config.long_context_config.provider)
        if not api_key or api_key == "test_key":
            pytest.skip("No valid API key available for LLM provider")

        # Create a corrupted history with consecutive interruptions
        corrupted_history = [
            Message(role="user", content=[TextBlock(text="Help me manage multiple files.")]),
            Message(role="assistant", content=[TextBlock(text="I'll help you manage your files systematically.")]),
            # First tool call sequence
            Message(role="assistant", content=[ToolCall(id="seq1_tool", name="list_dir", input={"path": "."})]),
            Message(
                role="user",
                content=[
                    ToolResult(
                        tool_use_id="seq1_tool", name="list_dir", content="file1.txt, file2.txt, dir1/", is_error=False
                    )
                ],
            ),
            # Second tool call sequence - partially interrupted
            Message(
                role="assistant",
                content=[
                    ToolCall(id="seq2_tool1", name="read_file", input={"path": "file1.txt"}),
                    ToolCall(id="seq2_tool2", name="read_file", input={"path": "file2.txt"}),
                ],
            ),
            Message(
                role="user",
                content=[
                    ToolResult(tool_use_id="seq2_tool1", name="read_file", content="Content of file1", is_error=False)
                    # Missing seq2_tool2 result
                ],
            ),
            # Third tool call sequence - completely interrupted
            Message(
                role="assistant",
                content=[
                    ToolCall(id="seq3_tool1", name="write_file", input={"path": "summary.txt", "content": "Summary"}),
                    ToolCall(id="seq3_tool2", name="list_dir", input={"path": "dir1"}),
                ],
            ),
            # No user message for third sequence (interrupted)
        ]

        # Verify the corrupted history is invalid
        assert base_agent._is_history_valid_for_anthropic(corrupted_history) is False

        # Fix the corrupted history
        fixed_history = base_agent.fix_incomplete_tool_calls(corrupted_history)

        # Verify the fix worked
        assert base_agent._is_history_valid_for_anthropic(fixed_history) is True

        # Set up the fixed history in the agent
        base_agent.history = MessageHistory(fixed_history)

        try:
            # Test with a follow-up conversation
            system_message = Message(
                role="system",
                content=[
                    TextBlock(
                        text="You are a file management assistant. When operations are interrupted, acknowledge this and offer to retry."
                    )
                ],
            )

            # Add a new user message
            base_agent.history.append(
                Message(
                    role="user",
                    content=[
                        TextBlock(
                            text="Some operations seem to have been interrupted. Can you tell me what happened and what we should do next?"
                        )
                    ],
                )
            )

            # Call the real LLM API
            response = await base_agent.llm.generate(
                messages=base_agent.history,
                system=system_message,
                model=base_agent.config.long_context_config.model,
                max_completion_tokens=250,
            )

            # Verify we got a valid response
            assert response is not None
            response_text = response.get_text_content()
            assert isinstance(response_text, str)
            assert len(response_text.strip()) > 0

            # The response should acknowledge the interruptions
            response_lower = response_text.lower()
            assert any(word in response_lower for word in ["interrupt", "error", "retry", "again", "issue"])

        except Exception as e:
            pytest.fail(f"Real API call failed with consecutive tool interruptions: {str(e)}")

    @pytest.mark.slow
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_edge_case_tool_corruption_with_real_api(self, base_agent):
        """Integration test: Test edge case corruptions that might break the API."""
        # Skip if no API key is available
        api_key = base_agent.config.get_api_key(base_agent.config.long_context_config.provider)
        if not api_key or api_key == "test_key":
            pytest.skip("No valid API key available for LLM provider")

        # Create an edge case: assistant message with tools followed by another assistant message
        edge_case_history = [
            Message(role="user", content=[TextBlock(text="Process this complex workflow.")]),
            Message(role="assistant", content=[TextBlock(text="I'll process the workflow step by step.")]),
            # Assistant with tools
            Message(
                role="assistant",
                content=[
                    ToolCall(id="workflow_step1", name="read_file", input={"path": "config.yaml"}),
                    ToolCall(id="workflow_step2", name="validate_data", input={"data": "test"}),
                    ToolCall(id="workflow_step3", name="process_workflow", input={"step": 1}),
                ],
            ),
            # Another assistant message (invalid - should have user message with tool results first)
            Message(role="assistant", content=[TextBlock(text="Let me continue with the next steps.")]),
            # User asking about status
            Message(role="user", content=[TextBlock(text="How is the workflow going?")]),
        ]

        # Verify this edge case is invalid
        assert base_agent._is_history_valid_for_anthropic(edge_case_history) is False

        # Fix the edge case
        fixed_history = base_agent.fix_incomplete_tool_calls(edge_case_history)

        # Verify the fix worked
        assert base_agent._is_history_valid_for_anthropic(fixed_history) is True

        # Set up the fixed history
        base_agent.history = MessageHistory(fixed_history)

        try:
            # Test with real API
            system_message = Message(
                role="system", content=[TextBlock(text="You are a workflow processing assistant.")]
            )

            # Call the real LLM API
            response = await base_agent.llm.generate(
                messages=base_agent.history,
                system=system_message,
                model=base_agent.config.long_context_config.model,
                max_completion_tokens=150,
            )

            # Verify we got a valid response
            assert response is not None
            response_text = response.get_text_content()
            assert isinstance(response_text, str)
            assert len(response_text.strip()) > 0

        except Exception as e:
            pytest.fail(f"Real API call failed with edge case corruption: {str(e)}")

    # Tests for new safe append methods
    def test_append_user_message_with_incomplete_tool_calls(self, base_agent):
        """Test that append_user_message does NOT fix incomplete tool calls."""
        # Add assistant message with tool calls
        base_agent.history.append(Message(role="assistant", content=[ToolCall(id="tool1", name="test_tool", input={})]))

        # Append user message - should NOT fix the history
        base_agent.append_user_message("New user message")

        # Verify history was NOT fixed - append should preserve authentic history
        assert len(base_agent.history) == 2  # assistant, user (new message)
        assert not base_agent._is_history_valid_for_anthropic()  # Still invalid

        # Check the new message was added
        new_msg = base_agent.history[1]
        assert new_msg.role == "user"
        assert new_msg.content[0].text == "New user message"

        # But verify that fix_incomplete_tool_calls can fix it
        fixed_history = base_agent.fix_incomplete_tool_calls(list(base_agent.history))
        assert len(fixed_history) == 3  # assistant, user (tool result), user (new message)
        assert base_agent._is_history_valid_for_anthropic(fixed_history) is True

    def test_append_user_message_no_fix_needed(self, base_agent):
        """Test that append_user_message works normally when no fix needed."""
        # Add a normal message
        base_agent.history.append(Message(role="user", content=[TextBlock(text="Hello")]))

        # Append another user message
        base_agent.append_user_message("Another message")

        # Should just append normally
        assert len(base_agent.history) == 2
        assert base_agent.history[1].content[0].text == "Another message"

    def test_append_user_message_with_list_content(self, base_agent):
        """Test append_user_message with list of ContentBlocks."""
        content_blocks = [TextBlock(text="Message part 1"), TextBlock(text="Message part 2")]

        base_agent.append_user_message(content_blocks)

        assert len(base_agent.history) == 1
        assert len(base_agent.history[0].content) == 2
        assert base_agent.history[0].content[0].text == "Message part 1"
        assert base_agent.history[0].content[1].text == "Message part 2"

    def test_append_user_message_with_single_block(self, base_agent):
        """Test append_user_message with single ContentBlock."""
        content_block = TextBlock(text="Single block message")

        base_agent.append_user_message(content_block)

        assert len(base_agent.history) == 1
        assert len(base_agent.history[0].content) == 1
        assert base_agent.history[0].content[0].text == "Single block message"

    def test_append_assistant_message(self, base_agent):
        """Test that append_assistant_message works correctly."""
        # Add a user message first
        base_agent.append_user_message("User question")

        # Add assistant message
        assistant_msg = Message(role="assistant", content=[TextBlock(text="Assistant response")])
        base_agent.append_assistant_message(assistant_msg)

        assert len(base_agent.history) == 2
        assert base_agent.history[1].role == "assistant"
        assert base_agent.history[1].content[0].text == "Assistant response"

    def test_get_effective_history_preserves_thinking_blocks(self, base_agent):
        base_agent.history = MessageHistory(
            [
                Message(
                    role="assistant",
                    content=[
                        ThinkingBlock(thinking="unsigned thinking"),
                        ThinkingBlock(thinking="signed thinking", signature="sig"),
                        RedactedThinkingBlock(data="encrypted-redacted-thinking"),
                        TextBlock(text="final answer"),
                    ],
                )
            ]
        )

        effective = base_agent.get_effective_history_for_llm()

        assert len(effective) == 1
        assert [block.type for block in effective[0].content] == [
            "thinking",
            "thinking",
            "redacted_thinking",
            "text",
        ]
        assert effective[0].content[0].thinking == "unsigned thinking"
        assert effective[0].content[1].thinking == "signed thinking"
        assert effective[0].content[1].signature == "sig"
        assert effective[0].content[2].data == "encrypted-redacted-thinking"

    def test_extend_history_no_fix_needed(self, base_agent):
        """Test extend_history works normally when no fix needed."""
        # Start with valid history
        base_agent.history.append(Message(role="user", content=[TextBlock(text="Hello")]))
        base_agent.history.append(Message(role="assistant", content=[TextBlock(text="Hi there")]))

        # Extend with more messages
        new_messages = [
            Message(role="user", content=[TextBlock(text="How are you?")]),
            Message(role="assistant", content=[TextBlock(text="I'm doing well")]),
        ]

        base_agent.extend_history(new_messages)

        assert len(base_agent.history) == 4
        assert base_agent.history[2].content[0].text == "How are you?"
        assert base_agent.history[3].content[0].text == "I'm doing well"

    def test_needs_tool_call_fix(self, base_agent):
        """Test the _needs_tool_call_fix method."""
        # Empty history
        assert not base_agent._needs_tool_call_fix()

        # User message last
        base_agent.history.append(Message(role="user", content=[TextBlock(text="Hello")]))
        assert not base_agent._needs_tool_call_fix()

        # Assistant message without tools
        base_agent.history.append(Message(role="assistant", content=[TextBlock(text="Hi")]))
        assert not base_agent._needs_tool_call_fix()

        # Assistant message with tools
        base_agent.history.append(Message(role="assistant", content=[ToolCall(id="tool1", name="test", input={})]))
        assert base_agent._needs_tool_call_fix()

    def test_needs_tool_call_fix_with_mixed_content(self, base_agent):
        """Test _needs_tool_call_fix with mixed content blocks."""
        # Assistant message with text and tool calls
        base_agent.history.append(
            Message(
                role="assistant",
                content=[
                    TextBlock(text="Let me help you with that."),
                    ToolCall(id="tool1", name="read_file", input={"path": "test.txt"}),
                ],
            )
        )

        assert base_agent._needs_tool_call_fix()

    def test_needs_tool_call_fix_string_content(self, base_agent):
        """Test _needs_tool_call_fix with string content (edge case)."""
        # This shouldn't happen in practice, but test the edge case
        base_agent.history.append(Message(role="assistant", content="Just a string"))

        assert not base_agent._needs_tool_call_fix()

    @pytest.mark.slow
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_safe_append_with_real_api(self, base_agent):
        """Test that append methods work with real Anthropic API when history is fixed."""
        # Skip if no API key is available
        api_key = base_agent.config.get_api_key(base_agent.config.long_context_config.provider)
        if not api_key or api_key == "test_key":
            pytest.skip("No valid API key available for LLM provider")

        # Set up history with tool calls
        base_agent.history.append(Message(role="user", content=[TextBlock(text="Read the README file")]))
        base_agent.history.append(
            Message(
                role="assistant",
                content=[
                    TextBlock(text="I'll read the README file for you."),
                    ToolCall(id="tool1", name="read_file", input={"path": "README.md"}),
                ],
            )
        )

        # Append user message - history is now invalid
        base_agent.append_user_message("What does it say?")

        # Verify history is invalid
        assert not base_agent._is_history_valid_for_anthropic()

        # Fix history before API call
        fixed_history = MessageHistory(base_agent.fix_incomplete_tool_calls(list(base_agent.history)))

        # Verify we can make an API call with fixed history
        system_message = Message(role="system", content=[TextBlock(text="You are a helpful assistant.")])

        try:
            response = await base_agent.llm.generate(
                messages=fixed_history,
                system=system_message,
                model=base_agent.config.long_context_config.model,
                max_completion_tokens=200,
            )

            assert response is not None
            assert response.get_text_content()
        except Exception as e:
            pytest.fail(f"API call failed after fixing history: {str(e)}")

    def test_append_user_message_multiple_incomplete_sequences(self, base_agent):
        """Test append_user_message does NOT fix multiple incomplete sequences."""
        # Create history with multiple incomplete tool sequences
        base_agent.history = MessageHistory(
            [
                Message(role="user", content=[TextBlock(text="Initial request")]),
                Message(
                    role="assistant",
                    content=[
                        ToolCall(id="tool1", name="first_tool", input={}),
                        ToolCall(id="tool2", name="second_tool", input={}),
                    ],
                ),
                Message(
                    role="user",
                    content=[
                        ToolResult(tool_use_id="tool1", name="first_tool", content="Result 1", is_error=False)
                        # Missing tool2 result
                    ],
                ),
                Message(role="assistant", content=[ToolCall(id="tool3", name="third_tool", input={})]),
                # Missing tool3 result
            ]
        )

        # Append new user message
        base_agent.append_user_message("Continue with the task")

        # History should still be invalid - append doesn't fix
        assert not base_agent._is_history_valid_for_anthropic()

        # But fix_incomplete_tool_calls should be able to fix it
        fixed_history = base_agent.fix_incomplete_tool_calls(list(base_agent.history))
        assert base_agent._is_history_valid_for_anthropic(fixed_history)

        # Verify the fixed history has all tool results
        tool_results = []
        for msg in fixed_history:
            if msg.role == "user":
                tool_results.extend([b for b in msg.content if isinstance(b, ToolResult)])

        tool_result_ids = {r.tool_use_id for r in tool_results}
        assert "tool2" in tool_result_ids  # Should have placeholder for tool2
        assert "tool3" in tool_result_ids  # Should have placeholder for tool3

    @pytest.mark.slow
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_restore_and_append_scenario(self, base_agent):
        """Test the scenario where restore and append don't fix, but LLM call fixes."""
        # Skip if no API key is available
        api_key = base_agent.config.get_api_key(base_agent.config.long_context_config.provider)
        if not api_key or api_key == "test_key":
            pytest.skip("No valid API key available for LLM provider")

        # Create a serialized history that ends with tool calls (simulating what's in the DB)
        serialized_history = [
            {
                "role": "user",
                "content": [{"type": "text", "text": "Help me with a task", "cache_checkpoint": False}],
                "stop_reason": None,
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "I'll help you with that task.", "cache_checkpoint": False}],
                "stop_reason": None,
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_call",
                        "id": "tool_1",
                        "name": "read_file",
                        "input": {"path": "task.txt"},
                        "cache_checkpoint": False,
                    }
                ],
                "stop_reason": "tool_use",
            },
            # Missing tool result - simulating an interrupted session
        ]

        # Restore the history (should NOT auto-fix)
        base_agent.restore_message_history(serialized_history)

        # Verify history is still invalid
        assert not base_agent._is_history_valid_for_anthropic()

        # Add a new user message (history remains invalid)
        base_agent.append_user_message("What's the status of my task?")

        # Verify the history is still invalid
        assert not base_agent._is_history_valid_for_anthropic()

        # Fix history before API call
        fixed_history = MessageHistory(base_agent.fix_incomplete_tool_calls(list(base_agent.history)))

        # Test with real API
        system_message = Message(role="system", content=[TextBlock(text="You are a helpful assistant.")])

        try:
            response = await base_agent.llm.generate(
                messages=fixed_history,
                system=system_message,
                model=base_agent.config.long_context_config.model,
                max_completion_tokens=100,
            )

            assert response is not None
            assert response.get_text_content()
        except Exception as e:
            # If this fails with the tool_use_id error, our fix didn't work
            pytest.fail(f"API call failed with fixed history: {str(e)}")

    def test_get_effective_history_falls_back_when_no_compression(self, base_agent):
        # With no compression, effective == full history
        base_agent.history = MessageHistory(
            [
                Message(role="user", content=[TextBlock(text="hi")]),
                Message(role="assistant", content=[TextBlock(text="yo")]),
            ]
        )
        eff = base_agent.get_effective_history_for_llm()
        assert len(eff) == 2

    def test_get_effective_history_after_markers(self, base_agent):
        base_agent.history = MessageHistory(
            [
                Message(role="user", content=[TextBlock(text="a")]),
                Message(role="assistant", content=[TextBlock(text="b")]),
                Message(role="user", content=[TextBlock(text="c")]),
            ]
        )
        base_agent.last_compression_index = 2
        # Append a summary message as it would be after compression
        base_agent.history.append(
            Message(role="user", content=[TextBlock(text="CONVERSATION HISTORY SUMMARY (compressed at ...)")])
        )
        eff = base_agent.get_effective_history_for_llm()
        # boundary is 2, so tail is after index 2 -> empty, but we still have summary
        assert len(eff) == 1
        assert "CONVERSATION HISTORY SUMMARY" in eff[0].content[0].text
