"""Tool inventory checks for shared agent classes."""

import uuid
from unittest.mock import AsyncMock, Mock

import pytest

from kolega_code.agent.browseragent import BrowserAgent
from kolega_code.agent.coder import CoderAgent
from kolega_code.agent.config import AgentConfig
from kolega_code.agent.connection_manager import AgentConnectionManager
from kolega_code.agent.investigationagent import InvestigationAgent
from kolega_code.agent.planningagent import PlanningAgent
from kolega_code.agent.prompt_provider import AgentMode


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
    return config


@pytest.fixture
def project_path(tmp_path):
    """Create a temporary project path."""
    return str(tmp_path)


def test_browser_agent_tools(project_path, mock_connection_manager, agent_config):
    """BrowserAgent exposes only browser tools."""
    agent = BrowserAgent(
        project_path=project_path,
        workspace_id="test_workspace",
        thread_id=str(uuid.uuid4()),
        connection_manager=mock_connection_manager,
        config=agent_config,
    )

    tools = agent.tool_collection.get_tool_list()
    tool_names = [tool.name for tool in tools]

    expected_tools = [
        "close_browser",
        "get_browser_console_logs",
        "get_browser_interactive_elements",
        "interact_with_browser",
        "launch_browser",
        "list_browsers",
        "set_browser_select_value",
        "take_browser_screenshot",
    ]

    assert len(tools) == len(expected_tools)
    assert set(tool_names) == set(expected_tools)


def test_investigation_agent_tools(project_path, mock_connection_manager, agent_config):
    """InvestigationAgent exposes read-only investigation tools."""
    agent = InvestigationAgent(
        project_path=project_path,
        workspace_id="test_workspace",
        thread_id=str(uuid.uuid4()),
        connection_manager=mock_connection_manager,
        config=agent_config,
    )

    tools = agent.tool_collection.get_tool_list()
    tool_names = [tool.name for tool in tools]

    expected_tools = [
        "find_files_by_pattern",
        "list_directory",
        "read_entire_file",
        "read_file_section",
        "search_codebase",
        "sleep",
        "think_hard",
        "web_fetch",
    ]

    assert len(tools) == len(expected_tools)
    assert set(tool_names) == set(expected_tools)


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
    )

    tool_names = {tool.name for tool in agent.tool_collection.get_tool_list()}

    assert "build_backend" in tool_names
    assert "build_frontend" in tool_names


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
    assert "create_file" not in tool_names
    assert "replace_entire_file" not in tool_names
    assert "apply_patch" not in tool_names
    assert "run_command_tracked" not in tool_names
    assert tool_names - expected_planning_tools <= set(agent.tool_collection.read_only_tools)


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
