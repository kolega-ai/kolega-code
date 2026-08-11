"""Tool inventory checks for shared agent classes."""

import asyncio
import uuid
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest

from kolega_code.agent.browseragent import BrowserAgent
from kolega_code.agent.coder import CoderAgent
from kolega_code.config import AgentConfig, ModelConfig, ModelProvider
from kolega_code.events import AgentConnectionManager
from kolega_code.agent.generalagent import GeneralAgent
from kolega_code.agent.investigationagent import InvestigationAgent
from kolega_code.agent.planningagent import PlanningAgent
from kolega_code.agent.prompt_provider import AgentMode, PromptProvider
from kolega_code.agent.tools import ToolCollection, ToolExtension
from kolega_code.llm.specs import MODEL_SPECS
from tests.agent._terminal_manager_stub import SandboxCarryingTerminalManager


INTERNAL_TOOL_NAMES = {"registry", "has_tool", "call", "cleanup", "initialize"}


def assert_internal_tools_not_exposed(tool_names):
    assert INTERNAL_TOOL_NAMES.isdisjoint(tool_names)


def dispatch_agent_types(tool_collection):
    """agent_type enum offered by the collection's dispatch_agent tool."""
    definition = next(tool for tool in tool_collection.get_tool_list() if tool.name == "dispatch_agent")
    return definition.input_schema["properties"]["agent_type"]["enum"]


@pytest.fixture
def mock_connection_manager():
    """Create a mock connection manager."""
    manager = Mock(spec=AgentConnectionManager)
    manager.workspace_id = "test_workspace"
    manager.send_message = AsyncMock()
    return manager


@pytest.fixture
def agent_config():
    """Create a mock agent configuration."""
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
    config.model_config_for_agent.return_value = config.long_context_config
    return config


@pytest.fixture
def project_path(tmp_path):
    """Create a temporary project path."""
    return str(tmp_path)


def hosted_prompt_provider(project_path):
    template_dir = Path(project_path) / "prompt_templates"
    agents_dir = template_dir / "system" / "agents"
    agents_dir.mkdir(parents=True, exist_ok=True)
    (agents_dir / "coder_code.md.j2").write_text("Private hosted test prompt.", encoding="utf-8")
    return PromptProvider(template_dirs=[template_dir])


def test_browser_agent_tools(project_path, mock_connection_manager, agent_config):
    """BrowserAgent exposes only browser tools."""
    agent = BrowserAgent(
        project_path=project_path,
        workspace_id="test_workspace",
        thread_id=str(uuid.uuid4()),
        connection_manager=mock_connection_manager,
        config=agent_config,
    )

    assert agent.tool_collection is not None
    tools = agent.tool_collection.get_tool_list()
    tool_names = [tool.name for tool in tools]

    expected_tools = [
        "browser_click",
        "browser_close",
        "browser_console_messages",
        "browser_drag",
        "browser_drop",
        "browser_evaluate",
        "browser_file_upload",
        "browser_fill_form",
        "browser_find",
        "browser_handle_dialog",
        "browser_hover",
        "browser_navigate",
        "browser_navigate_back",
        "browser_network_request",
        "browser_network_requests",
        "browser_press_key",
        "browser_resize",
        "browser_scroll",
        "browser_select_option",
        "browser_snapshot",
        "browser_tabs",
        "browser_take_screenshot",
        "browser_type",
        "browser_wait_for",
        "read_image",
    ]

    assert len(tools) == len(expected_tools)
    assert set(tool_names) == set(expected_tools)


@pytest.mark.parametrize("use_override", [False, True], ids=["inherited", "explicit"])
def test_browser_agent_rejects_nonvision_model_before_initialization(tmp_path, mock_connection_manager, use_override):
    long_context = ModelConfig(
        provider=ModelProvider.ANTHROPIC,
        model="claude-sonnet-4-5-20250929",
    )
    agent_models = {}
    if use_override:
        agent_models["browser"] = ModelConfig(provider=ModelProvider.DEEPSEEK, model="deepseek-v4-pro")
    else:
        long_context = ModelConfig(provider=ModelProvider.DEEPSEEK, model="deepseek-v4-pro")
    config = AgentConfig(
        anthropic_api_key="anthropic-key",
        deepseek_api_key="deepseek-key",
        long_context_config=long_context,
        agent_models=agent_models,
    )

    missing_project = tmp_path / "not-created"
    with pytest.raises(
        ValueError,
        match=r"BrowserAgent requires a vision-capable model.*deepseek/deepseek-v4-pro",
    ):
        BrowserAgent(
            project_path=missing_project,
            workspace_id="test_workspace",
            thread_id=str(uuid.uuid4()),
            connection_manager=mock_connection_manager,
            config=config,
        )


def test_browser_agent_treats_missing_vision_metadata_as_unsupported(tmp_path, mock_connection_manager, monkeypatch):
    model = "test-model-without-vision-metadata"
    monkeypatch.setitem(
        MODEL_SPECS,
        (ModelProvider.ANTHROPIC.value, model),
        {"context_length": 1_000, "max_completion_tokens": 100},
    )
    config = AgentConfig(
        anthropic_api_key="anthropic-key",
        agent_models={"browser": ModelConfig(provider=ModelProvider.ANTHROPIC, model=model)},
    )

    with pytest.raises(ValueError, match=r"does not support image input"):
        BrowserAgent(
            project_path=tmp_path,
            workspace_id="test_workspace",
            thread_id=str(uuid.uuid4()),
            connection_manager=mock_connection_manager,
            config=config,
        )


def test_investigation_agent_tools(project_path, mock_connection_manager, agent_config):
    """InvestigationAgent exposes read-only investigation tools."""
    agent = InvestigationAgent(
        project_path=project_path,
        workspace_id="test_workspace",
        thread_id=str(uuid.uuid4()),
        connection_manager=mock_connection_manager,
        config=agent_config,
    )

    assert agent.tool_collection is not None
    tools = agent.tool_collection.get_tool_list()
    tool_names = [tool.name for tool in tools]

    expected_tools = [
        "exec_command",
        "write_stdin",
        "kill_command",
        "list_sessions",
        "eval",
        "lsp",
        "read",
        "read_image",
        "web_fetch",
        "web_search",
    ]

    assert len(tools) == len(expected_tools)
    assert set(tool_names) == set(expected_tools)
    # File-edit tools remain unavailable to the read-only investigation agent.
    assert "edit" not in tool_names
    assert "multi_edit" not in tool_names
    assert "write" not in tool_names
    assert "lsp_edit" not in tool_names


def test_cli_coder_agent_does_not_expose_manifest_build_tools(project_path, mock_connection_manager, agent_config):
    """CLI CoderAgent does not expose platform-only manifest build tools."""
    agent = CoderAgent(
        project_path=project_path,
        workspace_id="test_workspace",
        thread_id=str(uuid.uuid4()),
        connection_manager=mock_connection_manager,
        config=agent_config,
        agent_mode=AgentMode.CLI,
    )

    tool_names = {tool.name for tool in agent.tool_collection.get_tool_list()}

    assert "build_backend" not in tool_names
    assert "build_frontend" not in tool_names


def test_non_cli_coder_agent_keeps_manifest_build_tools(project_path, mock_connection_manager, agent_config):
    """Non-CLI CoderAgent keeps manifest build tools for platform use."""
    agent = CoderAgent(
        project_path=project_path,
        workspace_id="test_workspace",
        thread_id=str(uuid.uuid4()),
        connection_manager=mock_connection_manager,
        config=agent_config,
        agent_mode=AgentMode.CODE,
        prompt_provider=hosted_prompt_provider(project_path),
    )

    tool_names = {tool.name for tool in agent.tool_collection.get_tool_list()}

    assert "build_backend" in tool_names
    assert "build_frontend" in tool_names


def test_coder_agent_dispatches_general_but_not_coding_agents(project_path, mock_connection_manager, agent_config):
    """CoderAgent can dispatch general sub-agents but still not coding agents."""
    agent = CoderAgent(
        project_path=project_path,
        workspace_id="test_workspace",
        thread_id=str(uuid.uuid4()),
        connection_manager=mock_connection_manager,
        config=agent_config,
        agent_mode=AgentMode.CLI,
    )

    tool_names = {tool.name for tool in agent.tool_collection.get_tool_list()}

    assert_internal_tools_not_exposed(tool_names)
    assert "dispatch_agent" in tool_names
    agent_types = dispatch_agent_types(agent.tool_collection)
    assert "general" in agent_types
    assert "investigation" in agent_types
    assert "coding" not in agent_types
    assert "list_subagent_models" in tool_names
    assert not tool_names.intersection(ToolCollection.browser_tools)


MEMORY_TOOL_NAMES = {"read_memory", "list_memory", "write_memory", "edit_memory", "delete_memory"}


@pytest.mark.parametrize("agent_cls", [CoderAgent, PlanningAgent])
def test_memory_enabled_false_gates_tools_and_prompt_for_top_level_agents(
    agent_cls, project_path, mock_connection_manager, agent_config
):
    """memory_enabled=False nulls the manager, removing memory tools AND the memory-policy
    prompt section — for whichever top-level agent the mode selects, not just the coder."""

    def build(*, memory_enabled):
        return agent_cls(
            project_path=project_path,
            workspace_id="test_workspace",
            thread_id=str(uuid.uuid4()),
            connection_manager=mock_connection_manager,
            config=agent_config,
            agent_mode=AgentMode.CLI,
            memory_enabled=memory_enabled,
        )

    enabled = build(memory_enabled=True)
    disabled = build(memory_enabled=False)

    # "Off" is one representation: a present-but-disabled manager, never None.
    assert enabled.memory_manager is not None and enabled.memory_manager.enabled
    assert disabled.memory_manager is not None and not disabled.memory_manager.enabled

    # Both surfaces follow the manager: tools...
    enabled_tools = {tool.name for tool in enabled.tool_collection.get_tool_list()}
    disabled_tools = {tool.name for tool in disabled.tool_collection.get_tool_list()}
    assert MEMORY_TOOL_NAMES & enabled_tools  # at least the read tools when enabled
    assert not (MEMORY_TOOL_NAMES & disabled_tools)

    # ...and the memory-policy section injected into the system prompt.
    assert enabled.build_prompt_context().memory_policy != ""
    assert disabled.build_prompt_context().memory_policy == ""


def test_browser_agent_type_gated_on_browser_agent_model_vision(project_path, mock_connection_manager, agent_config):
    """The "browser" agent_type is offered iff the model *resolved for the browser-agent
    role* is vision-capable — independent of the main coding model."""
    main = agent_config.long_context_config  # claude-sonnet — vision-capable

    vision_browser = Mock(provider="anthropic", model="claude-sonnet-4-5-20250929")
    blind_browser = Mock(provider="deepseek", model="deepseek-v4-flash")

    def agent_types_with_browser_model(browser_model):
        agent_config.model_config_for_agent.side_effect = lambda name: (
            browser_model if name == "browser-agent" else main
        )
        agent = CoderAgent(
            project_path=project_path,
            workspace_id="test_workspace",
            thread_id=str(uuid.uuid4()),
            connection_manager=mock_connection_manager,
            config=agent_config,
            agent_mode=AgentMode.CLI,
        )
        return dispatch_agent_types(agent.tool_collection)

    # Vision-capable browser-agent model -> offered even if we later flip the main model.
    assert "browser" in agent_types_with_browser_model(vision_browser)
    # Vision-less browser-agent model -> not offered, so the model can't waste a turn on it.
    assert "browser" not in agent_types_with_browser_model(blind_browser)


def test_sub_agent_coder_cannot_dispatch_general_agent(project_path, mock_connection_manager, agent_config):
    """A dispatched CoderAgent must not fan out into further general sub-agents."""
    agent = CoderAgent(
        project_path=project_path,
        workspace_id="test_workspace",
        thread_id=str(uuid.uuid4()),
        connection_manager=mock_connection_manager,
        config=agent_config,
        agent_mode=AgentMode.CLI,
        sub_agent=True,
    )

    agent_types = dispatch_agent_types(agent.tool_collection)

    assert "general" not in agent_types
    assert "investigation" in agent_types


def test_general_agent_tool_inventory(project_path, mock_connection_manager, agent_config):
    """GeneralAgent has the full toolset but cannot dispatch sub-agents."""
    agent = GeneralAgent(
        project_path=project_path,
        workspace_id="test_workspace",
        thread_id=str(uuid.uuid4()),
        connection_manager=mock_connection_manager,
        config=agent_config,
    )

    assert agent.tool_collection is not None
    tool_names = {tool.name for tool in agent.tool_collection.get_tool_list()}

    assert_internal_tools_not_exposed(tool_names)
    # Full read/write/terminal access
    assert "read" in tool_names
    assert "edit" in tool_names
    assert "multi_edit" in tool_names
    assert "write" in tool_names
    assert "lsp_edit" in tool_names
    assert "exec_command" in tool_names
    assert not tool_names.intersection(ToolCollection.browser_tools)
    # Recursion guard: no dispatch tools at all
    assert not any(name.startswith("dispatch_") for name in tool_names)
    assert "list_subagent_models" not in tool_names


def test_cli_general_agent_excludes_manifest_build_tools(project_path, mock_connection_manager, agent_config):
    """GeneralAgent inherits the CLI-mode exclusion of platform build tools."""
    agent = GeneralAgent(
        project_path=project_path,
        workspace_id="test_workspace",
        thread_id=str(uuid.uuid4()),
        connection_manager=mock_connection_manager,
        config=agent_config,
        agent_mode=AgentMode.CLI,
    )

    assert agent.tool_collection is not None
    tool_names = {tool.name for tool in agent.tool_collection.get_tool_list()}

    assert "build_backend" not in tool_names
    assert "build_frontend" not in tool_names


def test_planning_agent_exposes_read_only_and_planning_tools(project_path, mock_connection_manager, agent_config):
    """PlanningAgent cannot edit files and can capture a final plan."""
    agent = PlanningAgent(
        project_path=project_path,
        workspace_id="test_workspace",
        thread_id=str(uuid.uuid4()),
        connection_manager=mock_connection_manager,
        config=agent_config,
        agent_mode=AgentMode.CLI,
    )

    tool_names = {tool.name for tool in agent.tool_collection.get_tool_list()}
    expected_planning_tools = {"write_plan"}

    assert expected_planning_tools.issubset(tool_names)
    assert "get_task_list" not in tool_names
    assert "update_task_list" not in tool_names
    assert "edit" not in tool_names
    assert "multi_edit" not in tool_names
    assert "write" not in tool_names
    # Planning agent can run investigative shell commands but cannot edit files.
    assert {"exec_command", "write_stdin", "kill_command", "list_sessions"} <= tool_names
    # ``planning_tools`` is checked before the read_only filter in
    # _should_include_tool, so declaring a tool in that group is a deliberate
    # bypass of this contract, not a label. Anything added to it must be
    # justified against this subset.
    assert tool_names - expected_planning_tools <= (
        set(agent.tool_collection.read_only_tools)
        | set(agent.tool_collection.command_tools)
        | {"write_memory", "edit_memory", "delete_memory"}
    )


def _session_control_tool_extensions():
    """Extensions shaped like the TUI's worktree/goal control pair."""

    async def switch_worktree(path: str) -> str:
        """Switch the active workspace."""
        return path

    async def set_goal(condition: str) -> str:
        """Set an autonomous goal."""
        return condition

    return [
        ToolExtension(
            name="cli-worktree-control",
            tools={"switch_worktree": switch_worktree},
            tool_descriptions={"switch_worktree": "Switch the active workspace."},
            tool_schemas={
                "switch_worktree": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                }
            },
            tool_groups={"cli_worktree_tools": ["switch_worktree"]},
            propagate_to_sub_agents=False,
            exclusive_tools=frozenset({"switch_worktree"}),
        ),
        ToolExtension(
            name="cli-goal-control",
            tools={"set_goal": set_goal},
            tool_descriptions={"set_goal": "Set an autonomous goal."},
            tool_schemas={
                "set_goal": {
                    "type": "object",
                    "properties": {"condition": {"type": "string"}},
                    "required": ["condition"],
                }
            },
            tool_groups={"cli_goal_tools": ["set_goal"]},
            propagate_to_sub_agents=False,
        ),
    ]


def test_planning_agent_cannot_reach_session_control_tools(project_path, mock_connection_manager, agent_config):
    """Switching the workspace or starting a goal loop is not a planning action.

    Neither tool is declared in ``planning_tools``, so the read_only filter drops
    both from the registry. ``call``/``has_tool`` dispatch through the registry,
    so the bound callbacks are unreachable even if the model names them.
    """
    agent = PlanningAgent(
        project_path=project_path,
        workspace_id="test_workspace",
        thread_id=str(uuid.uuid4()),
        connection_manager=mock_connection_manager,
        config=agent_config,
        agent_mode=AgentMode.CLI,
        tool_extensions=_session_control_tool_extensions(),
    )

    tool_names = {tool.name for tool in agent.tool_collection.get_tool_list()}

    assert "switch_worktree" not in tool_names
    assert "set_goal" not in tool_names
    assert agent.tool_collection.has_tool("switch_worktree") is False
    assert agent.tool_collection.has_tool("set_goal") is False


def test_coder_agent_keeps_session_control_tools(project_path, mock_connection_manager, agent_config):
    """The same extensions stay available in build mode."""
    agent = CoderAgent(
        project_path=project_path,
        workspace_id="test_workspace",
        thread_id=str(uuid.uuid4()),
        connection_manager=mock_connection_manager,
        config=agent_config,
        agent_mode=AgentMode.CLI,
        tool_extensions=_session_control_tool_extensions(),
    )

    tool_names = {tool.name for tool in agent.tool_collection.get_tool_list()}

    assert {"switch_worktree", "set_goal"} <= tool_names
    assert agent.tool_collection.has_tool("switch_worktree") is True
    assert agent.tool_collection.has_tool("set_goal") is True


def test_exec_command_exposes_optional_background_param(project_path, mock_connection_manager, agent_config):
    """The model-facing exec_command schema includes the optional background flag."""
    agent = CoderAgent(
        project_path=project_path,
        workspace_id="test_workspace",
        thread_id=str(uuid.uuid4()),
        connection_manager=mock_connection_manager,
        config=agent_config,
        agent_mode=AgentMode.CLI,
    )

    exec_tool = next(tool for tool in agent.tool_collection.get_tool_list() if tool.name == "exec_command")
    schema = exec_tool._object_schema()

    assert "background" in schema["properties"]
    assert schema["properties"]["background"]["type"] == "boolean"
    assert "background" not in schema["required"]
    assert schema["properties"]["background"]["description"]


def test_eval_tool_schema_carries_the_kernel_contract(project_path, mock_connection_manager, agent_config):
    """The eval tool's declared definition teaches the model the feature."""
    agent = CoderAgent(
        project_path=project_path,
        workspace_id="test_workspace",
        thread_id=str(uuid.uuid4()),
        connection_manager=mock_connection_manager,
        config=agent_config,
        agent_mode=AgentMode.CLI,
    )

    eval_tool = next(tool for tool in agent.tool_collection.get_tool_list() if tool.name == "eval")
    description = eval_tool.description

    # The contract: persistence, the prelude API, the bridge, and cell workflow.
    assert "persistent" in description
    assert "tool.<name>" in description
    assert "list_tools()" in description
    assert "pip_install" in description
    assert "reset" in description
    assert "parallel" in description
    # The read()-vs-bridge guidance: raw bytes via read(), tool formats via tool.*.
    assert "read()/write()" in description
    assert "model-facing format" in description

    schema = eval_tool._object_schema()
    # `code` is the payload and stays required; `language` defaults to "py" (the
    # dominant case), so omitting it is a valid call rather than a hard error.
    assert "code" in schema["required"]
    assert "language" not in schema["required"]
    assert "title" not in schema["required"]
    assert schema["properties"]["timeout"]["type"] == "number"
    assert schema["properties"]["reset"]["type"] == "boolean"


def test_eval_tool_hidden_when_disabled(project_path, mock_connection_manager, agent_config):
    """AgentConfig.eval_enabled=False removes the eval tool from the registry."""
    agent_config.eval_enabled = False
    agent = CoderAgent(
        project_path=project_path,
        workspace_id="test_workspace",
        thread_id=str(uuid.uuid4()),
        connection_manager=mock_connection_manager,
        config=agent_config,
        agent_mode=AgentMode.CLI,
    )

    tool_names = {tool.name for tool in agent.tool_collection.get_tool_list()}
    assert "eval" not in tool_names


def test_lsp_tools_hidden_when_disabled(project_path, mock_connection_manager, agent_config):
    """AgentConfig.lsp.enabled=False removes lsp, lsp_edit, and resolve from the registry."""
    from kolega_code.services.lsp import LspConfig

    agent_config.lsp = LspConfig(enabled=False)
    agent = CoderAgent(
        project_path=project_path,
        workspace_id="test_workspace",
        thread_id=str(uuid.uuid4()),
        connection_manager=mock_connection_manager,
        config=agent_config,
        agent_mode=AgentMode.CLI,
    )

    tool_names = {tool.name for tool in agent.tool_collection.get_tool_list()}
    assert "lsp" not in tool_names
    assert "lsp_edit" not in tool_names
    # resolve applies/discards pending lsp_edit(apply=false) previews, so it is
    # LSP-coupled: no pending action can exist while LSP is off.
    assert "resolve" not in tool_names


def test_dispatch_tools_hidden_when_subagents_disabled(project_path, mock_connection_manager, agent_config):
    """AgentConfig.subagents_enabled=False removes dispatch_agent — and the
    informational list_subagent_models — from the registry."""
    agent_config.subagents_enabled = False
    agent = CoderAgent(
        project_path=project_path,
        workspace_id="test_workspace",
        thread_id=str(uuid.uuid4()),
        connection_manager=mock_connection_manager,
        config=agent_config,
        agent_mode=AgentMode.CLI,
    )

    tool_names = {tool.name for tool in agent.tool_collection.get_tool_list()}
    assert "dispatch_agent" not in tool_names
    assert "list_subagent_models" not in tool_names
    # The registry is the dispatch boundary: a hidden tool is unreachable.
    assert agent.tool_collection.has_tool("dispatch_agent") is False


def test_subagents_disabled_keeps_list_subagent_models_for_gigacode(
    project_path, mock_connection_manager, agent_config
):
    """Gigacode workflow authoring keeps the model-catalog tool: the subagents
    gate clears dispatch candidates but not the gigacode delegation branch."""
    agent_config.subagents_enabled = False
    agent = CoderAgent(
        project_path=project_path,
        workspace_id="test_workspace",
        thread_id=str(uuid.uuid4()),
        connection_manager=mock_connection_manager,
        config=agent_config,
        agent_mode=AgentMode.CLI,
    )
    agent.apply_gigacode(True)

    tool_names = {tool.name for tool in agent.tool_collection.get_tool_list()}
    assert "dispatch_agent" not in tool_names
    assert "run_workflow" in tool_names
    assert "list_subagent_models" in tool_names


def test_subagents_gate_beats_planning_custom_group_whitelist(tmp_path, mock_connection_manager, agent_config):
    """PlanningAgent's custom_agent_tools group is a whitelist checked before the
    read-only filter, so the subagents gate must run before the group checks:
    with a plan-mode custom agent registered, dispatch_agent is offered by
    default and removed host-wide when subagents_enabled=False."""
    from kolega_code.agent.custom_agents import discover_custom_agents

    agents_dir = tmp_path / ".kolega" / "agents"
    agents_dir.mkdir(parents=True)
    (agents_dir / "planner.md").write_text(
        "---\nname: planner\ndescription: Produces focused plans\nmode: plan\n---\n\nFollow the custom instructions.\n",
        encoding="utf-8",
    )
    catalog = discover_custom_agents(tmp_path, tmp_path / "state").for_mode("plan")

    def build():
        return PlanningAgent(
            project_path=tmp_path,
            workspace_id="test_workspace",
            thread_id=str(uuid.uuid4()),
            connection_manager=mock_connection_manager,
            config=agent_config,
            agent_mode=AgentMode.CLI,
            custom_agent_catalog=catalog,
        )

    assert "dispatch_agent" in {tool.name for tool in build().tool_collection.get_tool_list()}

    agent_config.subagents_enabled = False
    disabled = build()
    assert "dispatch_agent" not in {tool.name for tool in disabled.tool_collection.get_tool_list()}
    assert disabled.tool_collection.has_tool("dispatch_agent") is False


def _coder(project_path, mock_connection_manager, agent_config):
    return CoderAgent(
        project_path=project_path,
        workspace_id="test_workspace",
        thread_id=str(uuid.uuid4()),
        connection_manager=mock_connection_manager,
        config=agent_config,
        agent_mode=AgentMode.CLI,
    )


def _with_hosted_support(monkeypatch):
    """Mark the fixture model hosted-web-search-capable in the catalog."""
    from kolega_code.llm.specs import MODEL_SPECS

    key = ("anthropic", "claude-sonnet-4-5-20250929")
    monkeypatch.setitem(MODEL_SPECS, key, {**MODEL_SPECS[key], "supports_hosted_web_search": True})


def test_web_tools_present_when_auto_without_hosted_support(project_path, mock_connection_manager, agent_config):
    """auto on a model without hosted search keeps the client web tools (today's set)."""
    agent_config.web_search_mode = "auto"
    agent = _coder(project_path, mock_connection_manager, agent_config)
    tool_names = {tool.name for tool in agent.tool_collection.get_tool_list()}
    assert {"web_search", "web_fetch"} <= tool_names
    assert agent.hosted_web_search_active is False


def test_web_tools_hidden_when_mode_off(project_path, mock_connection_manager, agent_config):
    """off removes both client web tools and never requests the hosted tool."""
    agent_config.web_search_mode = "off"
    agent = _coder(project_path, mock_connection_manager, agent_config)
    tool_names = {tool.name for tool in agent.tool_collection.get_tool_list()}
    assert "web_search" not in tool_names
    assert "web_fetch" not in tool_names
    assert agent.hosted_web_search_active is False


def test_web_tools_hidden_when_hosted_supported_on_auto(
    project_path, mock_connection_manager, agent_config, monkeypatch
):
    """auto on a hosted-capable model swaps the client tools for the hosted one."""
    _with_hosted_support(monkeypatch)
    agent_config.web_search_mode = "auto"
    agent = _coder(project_path, mock_connection_manager, agent_config)
    tool_names = {tool.name for tool in agent.tool_collection.get_tool_list()}
    assert "web_search" not in tool_names
    assert "web_fetch" not in tool_names
    assert agent.hosted_web_search_active is True


def test_mode_hosted_unsupported_falls_back_to_client(project_path, mock_connection_manager, agent_config):
    """hosted on a model without support degrades to the client tools, not to no-web."""
    agent_config.web_search_mode = "hosted"
    agent = _coder(project_path, mock_connection_manager, agent_config)
    tool_names = {tool.name for tool in agent.tool_collection.get_tool_list()}
    assert {"web_search", "web_fetch"} <= tool_names
    assert agent.hosted_web_search_active is False


def test_mode_client_wins_even_when_hosted_supported(project_path, mock_connection_manager, agent_config, monkeypatch):
    """client pins today's behavior regardless of model capability."""
    _with_hosted_support(monkeypatch)
    agent_config.web_search_mode = "client"
    agent = _coder(project_path, mock_connection_manager, agent_config)
    tool_names = {tool.name for tool in agent.tool_collection.get_tool_list()}
    assert {"web_search", "web_fetch"} <= tool_names
    assert agent.hosted_web_search_active is False


def test_shared_tool_names_are_well_formed(project_path, mock_connection_manager, agent_config):
    """Shared agent tool definitions have valid names and descriptions."""
    agents = [
        BrowserAgent(
            project_path=project_path,
            workspace_id="test_workspace",
            thread_id=str(uuid.uuid4()),
            connection_manager=mock_connection_manager,
            config=agent_config,
        ),
        InvestigationAgent(
            project_path=project_path,
            workspace_id="test_workspace",
            thread_id=str(uuid.uuid4()),
            connection_manager=mock_connection_manager,
            config=agent_config,
        ),
    ]

    for agent in agents:
        for tool in agent.tool_collection.get_tool_list():
            assert tool.name.replace("_", "").isalnum()
            assert tool.name.islower() or tool.name.replace("_", "").isalnum()
            assert tool.description


def test_get_host_absent_without_sandbox_terminal_manager(project_path, mock_connection_manager, agent_config):
    """get_host is a sandbox-only tool: the default local CLI coder does not get it."""
    agent = _coder(project_path, mock_connection_manager, agent_config)
    tool_names = {tool.name for tool in agent.tool_collection.get_tool_list()}
    assert "get_host" not in tool_names
    assert agent.tool_collection.has_tool("get_host") is False


def test_get_host_present_with_sandbox_terminal_manager(project_path, mock_connection_manager, agent_config):
    """get_host appears when the injected terminal manager carries a ``sandbox``."""
    agent = CoderAgent(
        project_path=project_path,
        workspace_id="test_workspace",
        thread_id=str(uuid.uuid4()),
        connection_manager=mock_connection_manager,
        config=agent_config,
        agent_mode=AgentMode.CLI,
        terminal_manager=SandboxCarryingTerminalManager(),
    )

    tool_names = {tool.name for tool in agent.tool_collection.get_tool_list()}
    assert "get_host" in tool_names
    assert agent.tool_collection.has_tool("get_host") is True

    # The registered tool resolves the host from the sandbox, not localhost.
    host = asyncio.run(agent.tool_collection.call("get_host", port=8000))
    assert host == "stub-sandbox:8000"


def test_wire_descriptions_document_parameters_only_in_schema(project_path, mock_connection_manager, agent_config):
    """CLI coder definitions never document the same parameter twice.

    Declared descriptions carry no dangling Args section: parameters are
    documented in the schema. Tools whose schema leaves a property undocumented
    (enum-only, e.g. browser button) keep an Args block in the description —
    that is the only place those parameters are documented, and the declared
    artifacts preserve it deliberately.
    """
    agent = _coder(project_path, mock_connection_manager, agent_config)
    definitions = {definition.name: definition for definition in agent.tool_collection.get_tool_list()}
    assert "exec_command" in definitions

    for name, definition in definitions.items():
        assert not any(line.strip() == "Args:" for line in definition.description.splitlines()), name

    # The same parameters are fully documented in the declared schema.
    schema = definitions["exec_command"].to_anthropic()["input_schema"]
    assert schema["properties"]["command"]["description"]
    assert schema["properties"]["yield_time_ms"]["description"]

    # dispatch_agent's explicit schema describes every property, so its Args
    # block is stripped too (the type catalog stays).
    dispatch = definitions["dispatch_agent"]
    assert not any(line.strip() == "Args:" for line in dispatch.description.splitlines())
    assert "Available agent_type values:" in dispatch.description


def test_browser_schema_without_property_descriptions_keeps_args_block(
    project_path, mock_connection_manager, agent_config
):
    """Enum-only schema properties keep the Args block as their documentation."""
    agent = BrowserAgent(
        project_path=project_path,
        workspace_id="test_workspace",
        thread_id=str(uuid.uuid4()),
        connection_manager=mock_connection_manager,
        config=agent_config,
    )
    assert agent.tool_collection is not None
    definitions = {definition.name: definition for definition in agent.tool_collection.get_tool_list()}

    # browser_click's `button` and `modifiers` are enum/items-only in the
    # schema, so its Args block must survive; browser_navigate documents every
    # property and strips it.
    assert any(line.strip() == "Args:" for line in definitions["browser_click"].description.splitlines())
    assert "Mouse button: left, right, or middle." in definitions["browser_click"].description
    assert not any(line.strip() == "Args:" for line in definitions["browser_navigate"].description.splitlines())
