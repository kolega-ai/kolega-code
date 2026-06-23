#!/usr/bin/env python3
"""
Test duplicate tool result prevention logic.
"""

import pytest
from unittest.mock import Mock, patch
from kolega_code.agent.baseagent import BaseAgent
from kolega_code.llm.models import Message, TextBlock, ToolCall, ToolResult
from kolega_code.config import AgentConfig, ModelConfig, ModelProvider, RateLimitConfig


class TestDuplicateToolResultPrevention:
    """Test that duplicate tool results are properly handled."""

    @pytest.fixture
    def base_agent(self):
        """Create a base agent instance for testing."""
        config = AgentConfig(
            anthropic_api_key="test-key",
            openai_api_key="test-key",
            long_context_config=ModelConfig(
                provider=ModelProvider.ANTHROPIC, model="test-model", rate_limits=RateLimitConfig()
            ),
            fast_config=ModelConfig(
                provider=ModelProvider.ANTHROPIC, model="test-model", rate_limits=RateLimitConfig()
            ),
            thinking_config=ModelConfig(
                provider=ModelProvider.ANTHROPIC,
                model="test-model",
                rate_limits=RateLimitConfig(),
                thinking_effort="medium",
            ),
        )

        with patch("kolega_code.agent.baseagent.AgentConnectionManager"), patch(
            "kolega_code.agent.baseagent.get_model_specs"
        ) as mock_get_model_specs, patch("kolega_code.agent.context.LocalTerminalManager"), patch(
            "kolega_code.agent.context.PlaywrightBrowserManager"
        ), patch(
            "kolega_code.agent.context.LLMClient"
        ), patch(
            "kolega_code.agent.baseagent.ToolCollection"
        ), patch(
            "kolega_code.agent.context.LocalFileSystem"
        ) as mock_filesystem_class:

            # Mock get_model_specs to return reasonable values
            mock_get_model_specs.return_value = {"context_length": 100000, "max_completion_tokens": 4096}

            # Create mock filesystem instance
            mock_filesystem = Mock()
            mock_filesystem.exists.return_value = True
            mock_filesystem.is_dir.return_value = True
            mock_filesystem_class.return_value = mock_filesystem

            agent = BaseAgent(
                project_path="/test/path",
                workspace_id="test-workspace",
                thread_id="test-thread",
                connection_manager=Mock(),
                config=config,
            )
            agent.llm = Mock()
            return agent

    def test_replace_dummy_result_with_real_result(self, base_agent):
        """Test that dummy 'Operation was interrupted' results are replaced with real results."""
        # Set up history with tool call
        base_agent.history.extend(
            [
                Message(role="user", content=[TextBlock(text="Do something")]),
                Message(role="assistant", content=[ToolCall(id="tool_123", name="test_tool", input={})]),
            ]
        )

        # The append_user_message will trigger fix_incomplete_tool_calls
        # which adds a dummy result, then the real result should replace it
        real_result = ToolResult(
            tool_use_id="tool_123", name="test_tool", content="Real tool execution result", is_error=False
        )

        base_agent.append_user_message([real_result])

        # Check that we have exactly one user message with the real result
        user_messages = [msg for msg in base_agent.history if msg.role == "user"]
        assert len(user_messages) == 2  # Original user message + tool result message

        # Check the tool result message
        tool_result_msg = user_messages[-1]
        assert len(tool_result_msg.content) == 1
        assert tool_result_msg.content[0].tool_use_id == "tool_123"
        assert tool_result_msg.content[0].content == "Real tool execution result"
        assert tool_result_msg.content[0].is_error is False

    def test_multiple_tool_calls_with_partial_real_results(self, base_agent):
        """Test handling multiple tool calls where some have real results and others are interrupted."""
        # Set up history with multiple tool calls
        base_agent.history.extend(
            [
                Message(role="user", content=[TextBlock(text="Do multiple things")]),
                Message(
                    role="assistant",
                    content=[
                        ToolCall(id="tool_1", name="tool1", input={}),
                        ToolCall(id="tool_2", name="tool2", input={}),
                        ToolCall(id="tool_3", name="tool3", input={}),
                    ],
                ),
            ]
        )

        # Append real results for tool_1 and tool_3, but tool_2 was interrupted
        real_results = [
            ToolResult(tool_use_id="tool_1", name="tool1", content="Tool 1 completed successfully", is_error=False),
            ToolResult(tool_use_id="tool_3", name="tool3", content="Tool 3 completed successfully", is_error=False),
        ]

        base_agent.append_user_message(real_results)

        # History is now invalid - append doesn't fix
        assert not base_agent._is_history_valid_for_anthropic()

        # Check that we have only the provided results
        user_messages = [msg for msg in base_agent.history if msg.role == "user"]
        tool_result_msg = user_messages[-1]

        # Should have only 2 results (what was provided)
        assert len(tool_result_msg.content) == 2

        # But when we fix the history, it should have all 3
        fixed_history = base_agent.fix_incomplete_tool_calls(list(base_agent.history))

        # Find the tool result message in fixed history
        fixed_user_messages = [msg for msg in fixed_history if msg.role == "user"]
        fixed_tool_result_msg = fixed_user_messages[-1]

        # Should have 3 results total: 2 real + 1 dummy for tool_2
        assert len(fixed_tool_result_msg.content) == 3

        # Check each tool result
        tool_results_by_id = {r.tool_use_id: r for r in fixed_tool_result_msg.content}

        # tool_1 should have real result
        assert tool_results_by_id["tool_1"].content == "Tool 1 completed successfully"
        assert tool_results_by_id["tool_1"].is_error is False

        # tool_2 should have dummy result
        assert "Operation was interrupted" in tool_results_by_id["tool_2"].content
        assert tool_results_by_id["tool_2"].is_error is True

        # tool_3 should have real result
        assert tool_results_by_id["tool_3"].content == "Tool 3 completed successfully"
        assert tool_results_by_id["tool_3"].is_error is False

    def test_immediate_real_result_replaces_dummy_same_operation(self, base_agent):
        """Test that real results replace dummies when appended in the same operation that creates the dummy."""
        # Set up history with tool call
        base_agent.history.extend(
            [
                Message(role="user", content=[TextBlock(text="Do something")]),
                Message(role="assistant", content=[ToolCall(id="immediate_tool", name="test_tool", input={})]),
            ]
        )

        # This simulates the actual flow where:
        # 1. Assistant message with tool call exists
        # 2. Tool execution completes (possibly after brief interruption)
        # 3. append_user_message is called with the real result
        # 4. _needs_tool_call_fix() returns True, so dummy is created
        # 5. But then the real result replaces the dummy

        real_result = ToolResult(
            tool_use_id="immediate_tool", name="test_tool", content="Real execution result", is_error=False
        )

        # This single append will:
        # 1. Detect incomplete tool calls and add dummy
        # 2. Replace the dummy with the real result
        base_agent.append_user_message([real_result])

        # Verify only the real result exists, not the dummy
        tool_results = []
        for msg in base_agent.history:
            if msg.role == "user" and isinstance(msg.content, list):
                for block in msg.content:
                    if isinstance(block, ToolResult):
                        tool_results.append(block)

        assert len(tool_results) == 1
        assert tool_results[0].tool_use_id == "immediate_tool"
        assert tool_results[0].content == "Real execution result"
        assert tool_results[0].is_error is False
        # Ensure it's not the dummy
        assert "Operation was interrupted" not in tool_results[0].content

    def test_delayed_real_result_replaces_dummy(self, base_agent):
        """Test that a delayed real result replaces a previously added dummy result."""
        # Set up history with tool call
        base_agent.history.extend(
            [
                Message(role="user", content=[TextBlock(text="Do something")]),
                Message(role="assistant", content=[ToolCall(id="delayed_tool", name="slow_tool", input={})]),
            ]
        )

        # Append a text message - history remains invalid
        base_agent.append_user_message([TextBlock(text="Status check")])

        # Verify history is invalid (no dummy was created)
        assert not base_agent._is_history_valid_for_anthropic()

        # Remove the status check message to prepare for the real tool result
        base_agent.history = base_agent.history[:-1]

        # Manually fix the history to simulate what would happen before sending to LLM
        fixed_history = base_agent.fix_incomplete_tool_calls(list(base_agent.history))

        # Verify dummy was created in the fixed history
        tool_results = []
        for msg in fixed_history:
            if msg.role == "user" and isinstance(msg.content, list):
                for block in msg.content:
                    if isinstance(block, ToolResult):
                        tool_results.append(block)

        assert len(tool_results) == 1
        assert tool_results[0].tool_use_id == "delayed_tool"
        assert "Operation was interrupted" in tool_results[0].content
        assert tool_results[0].is_error is True

        # Now replace the history with the fixed version to simulate a real scenario
        base_agent.history = fixed_history

        # Add another assistant message (simulating continued conversation)
        base_agent.history.append(Message(role="assistant", content=[TextBlock(text="Let me continue...")]))

        # The real result arrives late
        real_result = ToolResult(
            tool_use_id="delayed_tool", name="slow_tool", content="Finally completed!", is_error=False
        )

        base_agent.append_user_message([real_result])

        # Our implementation correctly replaces the dummy with the real result
        # even across different messages, ensuring only one result per tool_use_id
        final_tool_results = []
        for msg in base_agent.history:
            if msg.role == "user" and isinstance(msg.content, list):
                for block in msg.content:
                    if isinstance(block, ToolResult) and block.tool_use_id == "delayed_tool":
                        final_tool_results.append(block)

        # We expect only one result - the real one replaced the dummy
        assert len(final_tool_results) == 1
        assert final_tool_results[0].content == "Finally completed!"
        assert final_tool_results[0].is_error is False

    def test_no_duplicate_when_all_results_provided_immediately(self, base_agent):
        """Test that no duplicates are created when all results are provided immediately."""
        # Set up history with tool calls
        base_agent.history.extend(
            [
                Message(role="user", content=[TextBlock(text="Do something")]),
                Message(
                    role="assistant",
                    content=[
                        ToolCall(id="immediate_1", name="tool1", input={}),
                        ToolCall(id="immediate_2", name="tool2", input={}),
                    ],
                ),
            ]
        )

        # Append all results immediately
        results = [
            ToolResult(tool_use_id="immediate_1", name="tool1", content="Result 1", is_error=False),
            ToolResult(tool_use_id="immediate_2", name="tool2", content="Result 2", is_error=False),
        ]

        base_agent.append_user_message(results)

        # Check that we have exactly the expected results with no duplicates
        tool_results = []
        for msg in base_agent.history:
            if msg.role == "user" and isinstance(msg.content, list):
                for block in msg.content:
                    if isinstance(block, ToolResult):
                        tool_results.append(block)

        assert len(tool_results) == 2
        tool_ids = {r.tool_use_id for r in tool_results}
        assert tool_ids == {"immediate_1", "immediate_2"}

    def test_real_error_result_not_replaced(self, base_agent):
        """Test that real error results are replaced by success results."""
        # Set up history with tool call
        base_agent.history.extend(
            [
                Message(role="user", content=[TextBlock(text="Do something")]),
                Message(role="assistant", content=[ToolCall(id="error_tool", name="failing_tool", input={})]),
            ]
        )

        # Append a real error result (not a dummy)
        real_error = ToolResult(
            tool_use_id="error_tool",
            name="failing_tool",
            content="FileNotFoundError: The file does not exist",
            is_error=True,
        )

        base_agent.append_user_message([real_error])

        # Verify we have the error
        tool_results = []
        for msg in base_agent.history:
            if msg.role == "user" and isinstance(msg.content, list):
                for block in msg.content:
                    if isinstance(block, ToolResult) and block.tool_use_id == "error_tool":
                        tool_results.append(block)

        assert len(tool_results) == 1
        assert tool_results[0].is_error is True

        # Add an assistant response
        base_agent.history.append(Message(role="assistant", content=[TextBlock(text="Let me try again")]))

        # Try to append a success result for the same tool
        success_result = ToolResult(
            tool_use_id="error_tool", name="failing_tool", content="Success after retry", is_error=False
        )

        base_agent.append_user_message([success_result])

        # Should have only the success result - real error was replaced by success
        final_tool_results = []
        for msg in base_agent.history:
            if msg.role == "user" and isinstance(msg.content, list):
                for block in msg.content:
                    if isinstance(block, ToolResult) and block.tool_use_id == "error_tool":
                        final_tool_results.append(block)

        # We expect only one result - the success replaced the error
        assert len(final_tool_results) == 1
        assert final_tool_results[0].content == "Success after retry"
        assert final_tool_results[0].is_error is False

    def test_duplicate_success_results_prevented(self, base_agent):
        """Test that duplicate success results for the same tool ID are prevented."""
        # Set up history with tool call
        base_agent.history.extend(
            [
                Message(role="user", content=[TextBlock(text="Do something")]),
                Message(role="assistant", content=[ToolCall(id="success_tool", name="test_tool", input={})]),
            ]
        )

        # Append first success result
        first_success = ToolResult(
            tool_use_id="success_tool", name="test_tool", content="First successful execution", is_error=False
        )

        base_agent.append_user_message([first_success])

        # Add assistant response
        base_agent.history.append(Message(role="assistant", content=[TextBlock(text="Continuing...")]))

        # Try to append another success result for the same tool
        second_success = ToolResult(
            tool_use_id="success_tool", name="test_tool", content="Second successful execution", is_error=False
        )

        base_agent.append_user_message([second_success])

        # Should have only one result - duplicates are prevented
        tool_results = []
        for msg in base_agent.history:
            if msg.role == "user" and isinstance(msg.content, list):
                for block in msg.content:
                    if isinstance(block, ToolResult) and block.tool_use_id == "success_tool":
                        tool_results.append(block)

        # We expect only one result - the first one is kept
        assert len(tool_results) == 1
        assert tool_results[0].content == "First successful execution"
        assert tool_results[0].is_error is False

    def test_cross_message_tool_results_during_restoration(self, base_agent):
        """Test that tool results found in non-adjacent messages are handled correctly during restoration."""
        # Create a scenario where tool result is not in the immediately following message
        messages = [
            Message(
                role="assistant",
                content=[
                    TextBlock(text="I'll check that file."),
                    ToolCall(id="toolu_test123", name="read_file", input={"path": "test.py"}),
                ],
            ),
            # This message is between the tool call and its result
            Message(role="user", content=[TextBlock(text="Please hurry up!")]),
            # Tool result appears here instead of immediately after tool call
            Message(
                role="user",
                content=[
                    ToolResult(
                        tool_use_id="toolu_test123",
                        content="File contents: print('hello')",
                        name="read_file",
                        is_error=False,
                    )
                ],
            ),
        ]

        # Test fix_incomplete_tool_calls
        fixed_messages = base_agent.fix_incomplete_tool_calls(messages)

        # Should have 3 messages: assistant with tool call, user with tool result, user with text
        assert len(fixed_messages) == 3

        # First message should be the assistant message
        assert fixed_messages[0].role == "assistant"
        assert any(isinstance(block, ToolCall) for block in fixed_messages[0].content)

        # Second message should have the tool result (moved to correct position)
        assert fixed_messages[1].role == "user"
        tool_results = [block for block in fixed_messages[1].content if isinstance(block, ToolResult)]
        assert len(tool_results) == 1
        assert tool_results[0].tool_use_id == "toolu_test123"
        assert tool_results[0].content == "File contents: print('hello')"

        # Third message should be the user text message
        assert fixed_messages[2].role == "user"
        assert fixed_messages[2].content[0].text == "Please hurry up!"

        # Verify no duplicate tool results
        all_tool_results = []
        for msg in fixed_messages:
            if msg.role == "user" and isinstance(msg.content, list):
                all_tool_results.extend([block for block in msg.content if isinstance(block, ToolResult)])

        # Should only have one tool result total
        assert len(all_tool_results) == 1

    def test_multiple_tool_calls_with_scattered_results(self, base_agent):
        """Test handling multiple tool calls where results are scattered across messages."""
        messages = [
            Message(
                role="assistant",
                content=[
                    TextBlock(text="I'll check both files."),
                    ToolCall(id="toolu_001", name="read_file", input={"path": "file1.py"}),
                    ToolCall(id="toolu_002", name="read_file", input={"path": "file2.py"}),
                ],
            ),
            # Only one result in the next message
            Message(
                role="user",
                content=[
                    ToolResult(tool_use_id="toolu_001", content="File 1 contents", name="read_file", is_error=False)
                ],
            ),
            # Other user activity
            Message(role="user", content=[TextBlock(text="Also check file2")]),
            # Second result appears later
            Message(
                role="user",
                content=[
                    ToolResult(tool_use_id="toolu_002", content="File 2 contents", name="read_file", is_error=False)
                ],
            ),
        ]

        fixed_messages = base_agent.fix_incomplete_tool_calls(messages)

        # First message: assistant with tool calls
        assert fixed_messages[0].role == "assistant"

        # Second message: should have BOTH tool results
        assert fixed_messages[1].role == "user"
        tool_results = [block for block in fixed_messages[1].content if isinstance(block, ToolResult)]
        assert len(tool_results) == 2

        result_ids = {r.tool_use_id for r in tool_results}
        assert result_ids == {"toolu_001", "toolu_002"}

        # Remaining messages
        remaining_messages = fixed_messages[2:]
        for msg in remaining_messages:
            if msg.role == "user" and isinstance(msg.content, list):
                # No tool results should remain in other messages
                tool_results_in_msg = [block for block in msg.content if isinstance(block, ToolResult)]
                assert len(tool_results_in_msg) == 0

    def test_no_dummy_creation_when_providing_all_results(self, base_agent, monkeypatch):
        """Test that dummy results are not created when all tool results are provided immediately."""
        # Track calls to fix_incomplete_tool_calls
        fix_calls = []
        original_fix = base_agent.fix_incomplete_tool_calls

        def mock_fix(messages):
            fix_calls.append(True)
            return original_fix(messages)

        monkeypatch.setattr(base_agent, "fix_incomplete_tool_calls", mock_fix)

        # Set up history with tool calls
        base_agent.history.extend(
            [
                Message(role="user", content=[TextBlock(text="Do something")]),
                Message(
                    role="assistant",
                    content=[
                        ToolCall(id="tool_1", name="tool1", input={}),
                        ToolCall(id="tool_2", name="tool2", input={}),
                    ],
                ),
            ]
        )

        # Append all results immediately - this should NOT trigger fix_incomplete_tool_calls
        results = [
            ToolResult(tool_use_id="tool_1", name="tool1", content="Result 1", is_error=False),
            ToolResult(tool_use_id="tool_2", name="tool2", content="Result 2", is_error=False),
        ]

        base_agent.append_user_message(results)

        # Verify fix_incomplete_tool_calls was NOT called
        assert len(fix_calls) == 0, "fix_incomplete_tool_calls should not be called when all results are provided"

        # Verify the results were added correctly
        tool_results = []
        for msg in base_agent.history:
            if msg.role == "user" and isinstance(msg.content, list):
                for block in msg.content:
                    if isinstance(block, ToolResult):
                        tool_results.append(block)

        assert len(tool_results) == 2
        tool_ids = {r.tool_use_id for r in tool_results}
        assert tool_ids == {"tool_1", "tool_2"}

    # Note: When partial tool results are provided, the current implementation still creates
    # dummy results for ALL missing tool calls first, then immediately replaces the ones we
    # have results for. This results in log messages like:
    # - "Adding placeholder result for missing tool call: X"
    # - "Replaced tool result for tool_use_id: X"
    # This is an acceptable trade-off for the simplicity of the implementation, and it only
    # happens when some (but not all) tool results are provided, which is an edge case.


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
