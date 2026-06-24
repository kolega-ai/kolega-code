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
        base_agent.system_prompt = Message(role="system", content=[TextBlock(text="sys")])
        base_agent.tool_collection = MagicMock()
        base_agent.tool_collection.get_tool_list = MagicMock(return_value=[])
        base_agent.model_context_length = 1000

        fake = FakeLLM(token_script=[300], summary_text="This is a compressed summary of the conversation")
        base_agent.llm = fake

        result = await base_agent.compress_history()

        assert result.ok is True
        # Non-destructive: full history retained; the summary is stored out-of-band.
        assert len(base_agent.history) == len(conversation)
        assert base_agent.conversation.summary is not None
        assert base_agent.last_compression_index is not None
        # Effective history keeps recent turns verbatim after the summary (fewer than the full log).
        effective = base_agent.get_effective_history_for_llm()
        assert len(effective) < len(conversation)
        assert base_agent.history[-1] in list(effective)  # most recent turn kept verbatim

        fake.stream.assert_awaited_once()
        call_args = fake.stream.await_args.kwargs
        assert call_args["model"] == base_agent.primary_model_config.model
        # Summary budget is capped small (we are not aiming for a gigantic summary).
        assert call_args["max_completion_tokens"] <= 2048

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
        base_agent.system_prompt = Message(role="system", content=[TextBlock(text="sys")])
        base_agent.tool_collection = MagicMock()
        base_agent.tool_collection.get_tool_list = MagicMock(return_value=[])
        base_agent.model_context_length = 1000

        fake = FakeLLM(token_script=[100])
        base_agent.llm = fake

        result = await base_agent.compress_history()

        # Too few messages to compress: nothing recorded, generate never called.
        assert result.ok is False
        assert result.reason == "too_few"
        assert len(base_agent.history) == 4
        assert base_agent.conversation.summary is None
        fake.stream.assert_not_called()

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
        base_agent.system_prompt = Message(role="system", content=[TextBlock(text="sys")])
        base_agent.tool_collection = MagicMock()
        base_agent.tool_collection.get_tool_list = MagicMock(return_value=[])
        base_agent.model_context_length = 1000

        fake = FakeLLM(token_script=[100])
        fake.stream = AsyncMock(side_effect=Exception("Test error"))
        base_agent.llm = fake

        result = await base_agent.compress_history()

        # The failure is surfaced (not swallowed as success) and history is untouched.
        assert result.ok is False
        assert result.reason == "llm_error"
        assert len(base_agent.history) == 10
        assert base_agent.conversation.summary is None

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

        base_agent.system_prompt = Message(role="system", content=[TextBlock(text="You are a helpful coding agent.")])
        base_agent.tool_collection = MagicMock()
        base_agent.tool_collection.get_tool_list = MagicMock(return_value=[])

        # The most recent original message should survive compaction verbatim.
        last_message = base_agent.history[-1]

        try:
            result = await base_agent.compress_history()

            assert result.ok is True
            # Non-destructive: the full log is retained and the summary is recorded out-of-band.
            assert len(base_agent.history) == len(conversation)
            summary = base_agent.conversation.summary
            assert summary is not None
            assert summary.get_text_content().strip()

            # The recent tail is kept verbatim in the effective view.
            effective = list(base_agent.get_effective_history_for_llm())
            assert last_message in effective
        except Exception as e:
            pytest.fail(f"Test failed with error: {str(e)}")
