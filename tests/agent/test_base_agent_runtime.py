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
    def test_default_max_iterations_is_uncapped(self, base_agent):
        assert base_agent.max_iterations is None

    @pytest.mark.parametrize("max_iterations", [0, -1])
    def test_invalid_max_iterations_rejected(self, tmp_path, mock_connection_manager, agent_config, max_iterations):
        with pytest.raises(ValueError, match="max_iterations must be a positive integer or None"):
            BaseAgent(
                project_path=tmp_path,
                workspace_id="test_workspace",
                thread_id=str(uuid.uuid4()),
                connection_manager=mock_connection_manager,
                config=agent_config,
                max_iterations=max_iterations,
            )

    @pytest.mark.asyncio
    async def test_max_iterations_raises_for_runaway_tool_loop(self, tmp_path, mock_connection_manager, agent_config):
        agent = BaseAgent(
            project_path=tmp_path,
            workspace_id="test_workspace",
            thread_id=str(uuid.uuid4()),
            connection_manager=mock_connection_manager,
            config=agent_config,
            max_iterations=2,
        )
        tool_call = ToolCall(id="tool_1", name="read_file", input={})
        looping_message = Message(
            role="assistant",
            content=[tool_call],
            stop_reason="tool_use",
            tool_calls=[tool_call],
        )
        agent.system_prompt = Message(role="system", content=[TextBlock(text="sys")])
        agent.tool_collection = MagicMock()
        agent.tool_collection.get_tool_list = MagicMock(return_value=[])
        agent.llm = FakeLLM(token_script=[100], final_message=looping_message)
        agent.process_tool_calls = AsyncMock(
            return_value=[ToolResult(tool_use_id="tool_1", name="read_file", content="ok", is_error=False)]
        )
        agent.log_info = AsyncMock()
        agent.log_error = AsyncMock()

        with pytest.raises(MaxAgentIterationsExceeded, match="max_iterations=2"):
            async for _chunk in agent.process_message_stream("loop"):
                pass

        assert agent.process_tool_calls.await_count == 2

    @pytest.mark.asyncio
    async def test_terminal_turn_completes_under_max_iterations(self, base_agent):
        base_agent.max_iterations = 1
        base_agent.system_prompt = Message(role="system", content=[TextBlock(text="sys")])
        base_agent.tool_collection = MagicMock()
        base_agent.tool_collection.get_tool_list = MagicMock(return_value=[])
        base_agent.llm = FakeLLM(
            token_script=[100],
            final_message=Message(role="assistant", content=[TextBlock(text="done")], stop_reason="end_turn"),
        )
        base_agent.log_info = AsyncMock()

        chunks = [chunk async for chunk in base_agent.process_message_stream("finish")]

        assert chunks[-1]["complete"] is True
        assert base_agent.history[-1].stop_reason == "end_turn"

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "error,provider,model,expected_message",
        [
            (
                LLMBillingError("DeepSeek APIError: Insufficient Balance", provider=ModelProvider.DEEPSEEK.value),
                ModelProvider.DEEPSEEK,
                "deepseek-v4-pro",
                "DeepSeek/deepseek-v4-pro could not run this request",
            ),
            (
                LLMContextWindowExceededError("context too large", provider=ModelProvider.ANTHROPIC.value),
                ModelProvider.ANTHROPIC,
                "claude-haiku-4-5-20251001",
                "The conversation context became too large for the model",
            ),
            (
                LLMAuthenticationError("invalid key", provider=ModelProvider.ANTHROPIC.value),
                ModelProvider.ANTHROPIC,
                "claude-haiku-4-5-20251001",
                "Anthropic/claude-haiku-4-5-20251001 could not authenticate",
            ),
        ],
    )
    async def test_handle_llm_error_emits_status_and_reraises(
        self,
        base_agent,
        mock_connection_manager,
        error,
        provider,
        model,
        expected_message,
    ):
        base_agent.config.long_context_config.provider = provider
        base_agent.config.long_context_config.model = model

        with pytest.raises(type(error)):
            await base_agent.handle_llm_error(error)

        event = mock_connection_manager.broadcast_event.await_args.args[0]
        assert event.event_type == "llm_status_update"
        assert event.content["status"] == "error"
        assert expected_message in event.content["message"]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "make_error",
        [
            lambda: LLMRateLimitError("429 rate limited", provider=ModelProvider.ANTHROPIC.value),
            lambda: LLMInternalServerError("provider overloaded", provider=ModelProvider.ANTHROPIC.value),
        ],
    )
    async def test_handle_llm_error_retries_transient_then_caps(self, base_agent, make_error):
        """Rate-limit and overload/5xx errors back off and retry up to loop_max_retries
        consecutive attempts, then surface cleanly."""
        cap = base_agent.primary_model_config.rate_limits.loop_max_retries
        assert cap >= 1

        sleeps: list[float] = []

        async def fake_sleep(delay):
            sleeps.append(delay)

        with patch("kolega_code.agent.baseagent.asyncio.sleep", side_effect=fake_sleep):
            # Under the cap: returns without raising (the turn loop will re-issue).
            for attempt in range(cap):
                await base_agent.handle_llm_error(make_error())
                assert base_agent._consecutive_llm_retries == attempt + 1
            # Exceeding the cap re-raises the mapped error.
            with pytest.raises((LLMRateLimitError, LLMInternalServerError)):
                await base_agent.handle_llm_error(make_error())

        assert len(sleeps) == cap
        assert all(s >= 0 for s in sleeps)

    @pytest.mark.asyncio
    async def test_handle_llm_error_honors_retry_after(self, base_agent):
        """A retry-after header on the raw exception is used (capped) for the wait."""
        raw = SimpleNamespace(response=SimpleNamespace(headers={"retry-after": "7"}))
        # Make it look like a rate-limit so map_to_llm_error classifies it as retryable.
        raw.status_code = 429

        sleeps: list[float] = []

        async def fake_sleep(delay):
            sleeps.append(delay)

        with (
            patch("kolega_code.agent.baseagent.asyncio.sleep", side_effect=fake_sleep),
            patch(
                "kolega_code.agent.baseagent.map_to_llm_error",
                return_value=LLMRateLimitError("429", provider=ModelProvider.ANTHROPIC.value),
            ),
        ):
            await base_agent.handle_llm_error(raw)

        assert sleeps == [7.0]

    def test_parse_retry_after_forms(self):
        seconds = SimpleNamespace(response=SimpleNamespace(headers={"retry-after": "12"}))
        assert BaseAgent._parse_retry_after(seconds) == 12.0
        assert BaseAgent._parse_retry_after(Exception("no header")) is None
