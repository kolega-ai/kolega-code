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
    config.context_window_tokens = None
    config.max_output_tokens = None
    config.long_context_config = Mock()
    config.long_context_config.provider = "anthropic"
    config.long_context_config.model = "claude-sonnet-4-5-20250929"
    config.openai_api_key = "test_key"
    config.anthropic_api_key = "test_key"
    config.browser_use_headless = True
    config.agent_models = {}
    config.web_search_mode = "auto"
    config.model_config_for_agent.return_value = config.long_context_config
    config.llm_client_kwargs.return_value = {}
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


@pytest.mark.parametrize(
    "window,output,primary_supports_maximum,valid",
    [
        (872000, 128000, True, True),
        (872001, 128000, True, False),
        (872000, 128001, True, False),
        (872000, 128000, False, False),
    ],
)
def test_strict_budget_uses_actual_primary_model(
    tmp_path, monkeypatch: pytest.MonkeyPatch, window: int, output: int, primary_supports_maximum: bool, valid: bool
) -> None:
    primary_specs = {
        "context_length": 400000,
        "max_completion_tokens": 128000,
        "input_budget": "window_minus_output",
    }
    if primary_supports_maximum:
        primary_specs["max_context_length"] = 872000
    headline_specs = {**primary_specs, "max_context_length": 872000}
    lookups = []

    def get_specs(provider, model):
        lookups.append(model)
        return primary_specs if model == "actual-primary" else headline_specs

    monkeypatch.setattr("kolega_code.agent.baseagent.get_model_specs", get_specs)
    manager = Mock(spec=AgentConnectionManager)
    manager.workspace_id = "test_workspace"
    manager.send_message = AsyncMock()
    config = Mock(spec=AgentConfig)
    config.context_window_tokens = window
    config.max_output_tokens = output
    config.long_context_config = Mock(provider="anthropic", model="headline")
    primary = Mock(provider="anthropic", model="actual-primary")
    config.model_config_for_agent.return_value = primary
    config.openai_api_key = "test_key"
    config.anthropic_api_key = "test_key"
    config.browser_use_headless = True
    config.agent_models = {}
    config.web_search_mode = "auto"
    config.llm_client_kwargs.return_value = {}

    def construct():
        return CoderAgent(
            project_path=str(tmp_path),
            workspace_id="test_workspace",
            thread_id=str(uuid.uuid4()),
            connection_manager=manager,
            config=config,
            agent_mode=AgentMode.CLI,
        )

    if valid:
        result = construct()
        assert result.primary_model_config is primary
        assert result.strict_context_budget is True
        assert result.model_context_length == 872000
        assert result.model_completion_tokens == 128000
        assert result.model_max_input_tokens == 744000
        assert not result.compressor.over_budget(706800, result.model_max_input_tokens)
        assert result.compressor.over_budget(706801, result.model_max_input_tokens)
        assert primary_specs["context_length"] == 400000
    else:
        with pytest.raises(ValueError, match="catalogued .* (length|maximum)"):
            construct()
    assert lookups == ["actual-primary"]


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


@pytest.mark.asyncio
async def test_count_passes_thinking_config(agent, monkeypatch):
    from unittest.mock import AsyncMock

    from kolega_code.llm.providers.models import TokenCount

    counter = AsyncMock(return_value=TokenCount(input_tokens=10))
    monkeypatch.setattr(agent.llm, "count_tokens", counter)
    await agent.count_current_context()
    # Count-before-sample must select the same renderer configuration as
    # sampling on thinking-sensitive providers (native Tinker).
    assert "thinking" in counter.call_args.kwargs
