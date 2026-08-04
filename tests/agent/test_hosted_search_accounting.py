"""Hosted web-search residual accounting: the gauge must read billed-true.

Hosted search content is injected into the model context SERVER-side (restored
on replay of web_search_call items, billed as input) and never reaches the
client, so tiktoken-over-history undercounts. BaseAgent tracks the residual
(billed − counted) as a side-channel: search-bearing responses bump a
provisional per-call constant (their own billing telescopes internal rounds),
clean responses replace it with the measured residual, compaction resets it.
Live counterpart: the gauge-invariant test in
tests/agent/llm/test_hosted_web_search_live.py.
"""

import uuid
from unittest.mock import AsyncMock, Mock

import pytest

from kolega_code.agent.coder import CoderAgent
from kolega_code.agent.prompt_provider import AgentMode
from kolega_code.config import AgentConfig
from kolega_code.events import AgentConnectionManager
from kolega_code.llm.models import ContentBlock, Message, TextBlock, WebSearchCallBlock
from kolega_code.llm.providers.models import TokenCount


@pytest.fixture
def agent(tmp_path):
    manager = Mock(spec=AgentConnectionManager)
    manager.workspace_id = "test_workspace"
    manager.send_message = AsyncMock()
    config = Mock(spec=AgentConfig)
    config.long_context_config = Mock()
    config.long_context_config.provider = "anthropic"
    config.long_context_config.model = "claude-sonnet-4-5-20250929"
    config.openai_api_key = "test_key"
    config.anthropic_api_key = "test_key"
    config.browser_use_headless = True
    config.agent_models = {}
    config.web_search_mode = "auto"
    config.model_config_for_agent.return_value = config.long_context_config
    return CoderAgent(
        project_path=str(tmp_path),
        workspace_id="test_workspace",
        thread_id=str(uuid.uuid4()),
        connection_manager=manager,
        config=config,
        agent_mode=AgentMode.CLI,
    )


def _search_message(n_calls=1, billed=None):
    content: list[ContentBlock] = [
        WebSearchCallBlock(item_id=f"ws_{i}", status="completed", action={"type": "search", "queries": ["q"]})
        for i in range(n_calls)
    ]
    usage = {"prompt_tokens": billed} if billed is not None else {}
    return Message(role="assistant", content=content, usage_metadata=usage)


def _clean_message(billed):
    return Message(role="assistant", content=[TextBlock(text="ok")], usage_metadata={"prompt_tokens": billed})


class TestResidualUpdates:
    def test_search_bearing_response_adds_provisional_per_call(self, agent):
        agent._last_raw_context_count = 1000
        agent._update_hosted_search_residual(_search_message(n_calls=2, billed=50_000))
        assert agent._hosted_injected_tokens == 2 * CoderAgent.HOSTED_SEARCH_PROVISIONAL_TOKENS

    def test_clean_response_replaces_estimate_with_measurement(self, agent):
        agent._hosted_injected_tokens = 12_000
        agent._last_raw_context_count = 1_000
        agent._update_hosted_search_residual(_clean_message(billed=3_800))
        assert agent._hosted_injected_tokens == 2_800

    def test_clean_response_without_any_search_history_is_a_noop(self, agent):
        # Tokenizer drift (billed > counted) must not leak into the gauge for
        # sessions that never searched.
        agent._last_raw_context_count = 1_000
        agent._update_hosted_search_residual(_clean_message(billed=1_400))
        assert agent._hosted_injected_tokens == 0

    def test_recovers_after_compaction_when_search_blocks_survive(self, agent):
        # Post-compaction the residual is 0 but surviving web_search_call blocks
        # keep the server restoring content — the next clean response re-measures.
        agent.history.append(_search_message(n_calls=1))
        agent._hosted_injected_tokens = 0
        agent._last_raw_context_count = 1_000
        agent._update_hosted_search_residual(_clean_message(billed=4_000))
        assert agent._hosted_injected_tokens == 3_000

    def test_negative_residual_clamps_to_zero(self, agent):
        agent._hosted_injected_tokens = 5_000
        agent._last_raw_context_count = 2_000
        agent._update_hosted_search_residual(_clean_message(billed=1_500))
        assert agent._hosted_injected_tokens == 0

    def test_missing_usage_keeps_last_value(self, agent):
        agent._hosted_injected_tokens = 5_000
        agent._last_raw_context_count = 2_000
        agent._update_hosted_search_residual(
            Message(role="assistant", content=[TextBlock(text="ok")], usage_metadata={})
        )
        assert agent._hosted_injected_tokens == 5_000


class TestGaugeApplication:
    @pytest.mark.asyncio
    async def test_count_current_context_adds_positive_residual(self, agent, monkeypatch):
        monkeypatch.setattr(agent.llm, "count_tokens", AsyncMock(return_value=TokenCount(input_tokens=10_000)))
        agent._hosted_injected_tokens = 2_500
        token_count = await agent.count_current_context()
        assert token_count.input_tokens == 12_500
        # The raw (pre-residual) count is stashed for the next residual update.
        assert agent._last_raw_context_count == 10_000

    @pytest.mark.asyncio
    async def test_count_current_context_untouched_without_residual(self, agent, monkeypatch):
        monkeypatch.setattr(agent.llm, "count_tokens", AsyncMock(return_value=TokenCount(input_tokens=10_000)))
        token_count = await agent.count_current_context()
        assert token_count.input_tokens == 10_000

    @pytest.mark.asyncio
    async def test_compress_history_resets_the_residual(self, agent, monkeypatch):
        from kolega_code.agent.compression import CompactionResult

        agent._hosted_injected_tokens = 9_999
        monkeypatch.setattr(agent.llm, "count_tokens", AsyncMock(return_value=TokenCount(input_tokens=100)))
        monkeypatch.setattr(
            agent.compressor,
            "summarize",
            AsyncMock(return_value=CompactionResult(ok=True, reason="ok", summarized_messages=0)),
        )
        await agent.compress_history()
        assert agent._hosted_injected_tokens == 0
