"""The usable input budget is the window minus the requested output allowance."""

import uuid
from unittest.mock import AsyncMock, Mock

import pytest

from kolega_code.agent.coder import CoderAgent
from kolega_code.agent.prompt_provider import AgentMode
from kolega_code.config import AgentConfig
from kolega_code.events import AgentConnectionManager
from kolega_code.llm.specs import get_model_specs


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


def test_max_input_budget_is_window_minus_output(agent):
    specs = get_model_specs("anthropic", "claude-sonnet-4-5-20250929")
    assert agent.model_max_input_tokens == specs["context_length"] - specs["max_completion_tokens"]


@pytest.mark.asyncio
async def test_gauge_denominates_on_input_budget(agent, monkeypatch):
    from kolega_code.llm.providers.models import TokenCount

    monkeypatch.setattr(agent.llm, "count_tokens", AsyncMock(return_value=TokenCount(input_tokens=10_000)))
    emitted = {}

    async def capture(**kwargs):
        emitted.update(kwargs)

    monkeypatch.setattr(agent.emitter, "context_update", capture)
    await agent.count_current_context()
    assert emitted["model_context_length"] == agent.model_max_input_tokens


def test_resolver_covers_all_three_conventions():
    from kolega_code.llm.specs import resolve_max_input_tokens

    shared = {"context_length": 1000, "max_completion_tokens": 400, "input_budget": "window_minus_output"}
    separate = {"context_length": 1000, "max_completion_tokens": 400, "input_budget": "separate_output_limit"}
    shares = {"context_length": 1000, "max_completion_tokens": 1000, "input_budget": "output_shares_window"}
    assert resolve_max_input_tokens(shared) == 600
    assert resolve_max_input_tokens(separate) == 1000
    assert resolve_max_input_tokens(shares) == 1000


def test_catalog_declares_input_budget_everywhere():
    from kolega_code.llm.specs.accessors import INPUT_BUDGET_CONVENTIONS, MODEL_SPECS, WILDCARD_MODEL_SPECS

    for key, specs in list(MODEL_SPECS.items()) + list(WILDCARD_MODEL_SPECS.items()):
        assert specs.get("input_budget") in INPUT_BUDGET_CONVENTIONS, key
