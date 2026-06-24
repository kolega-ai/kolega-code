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
