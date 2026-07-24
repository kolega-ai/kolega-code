"""Tool inventory checks for shared agent classes."""

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
from kolega_code.agent.tools import ToolCollection
from kolega_code.llm.specs import MODEL_SPECS


INTERNAL_TOOL_NAMES = {"registry", "has_tool", "call", "cleanup", "initialize"}


def assert_internal_tools_not_exposed(tool_names):
    assert INTERNAL_TOOL_NAMES.isdisjoint(tool_names)


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
        "find_files_by_pattern",
        "list_directory",
        "lsp",
        "read_entire_file",
        "read_file_section",
        "read_image",
        "search_codebase",
        "sleep",
        "think_hard",
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


def test_coder_agent_exposes_dispatch_general_agent(project_path, mock_connection_manager, agent_config):
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
    assert "dispatch_general_agent" in tool_names
    assert "dispatch_investigation_agent" in tool_names
    assert "dispatch_coding_agent" not in tool_names
    assert "list_subagent_models" in tool_names
    assert not tool_names.intersection(ToolCollection.browser_tools)


def test_sub_agent_coder_cannot_dispatch_general_agent(project_path, mock_connection_manager, agent_config):
    """A dispatched CoderAgent must not fan out into further sub-agents."""
    agent = CoderAgent(
        project_path=project_path,
        workspace_id="test_workspace",
        thread_id=str(uuid.uuid4()),
        connection_manager=mock_connection_manager,
        config=agent_config,
        agent_mode=AgentMode.CLI,
        sub_agent=True,
    )

    tool_names = {tool.name for tool in agent.tool_collection.get_tool_list()}

    assert "dispatch_general_agent" not in tool_names


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
    assert "read_entire_file" in tool_names
    assert "search_codebase" in tool_names
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
    assert tool_names - expected_planning_tools <= (
        set(agent.tool_collection.read_only_tools)
        | set(agent.tool_collection.command_tools)
        | {"write_memory", "edit_memory", "delete_memory"}
    )


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
    params = {param.name: param for param in exec_tool.parameters}

    assert "background" in params
    assert params["background"].type == "boolean"
    assert params["background"].required is False
    assert params["background"].description


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
