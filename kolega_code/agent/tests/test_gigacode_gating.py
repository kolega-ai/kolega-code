"""Tool-gating checks for gigacode's run_workflow."""

import uuid
from unittest.mock import AsyncMock, Mock

import pytest

from kolega_code.agent.coder import CoderAgent
from kolega_code.agent.generalagent import GeneralAgent
from kolega_code.agent.planningagent import PlanningAgent
from kolega_code.agent.prompt_provider import AgentMode, PromptExtension
from kolega_code.agent.tool_backend.agent_tool import AgentTool
from kolega_code.agent.tools import ToolExtension
from kolega_code.config import AgentConfig
from kolega_code.events import AgentConnectionManager


@pytest.fixture
def mock_connection_manager():
    manager = Mock(spec=AgentConnectionManager)
    manager.workspace_id = "test_workspace"
    manager.send_message = AsyncMock()
    return manager


@pytest.fixture
def agent_config():
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
    return str(tmp_path)


def _coder(project_path, manager, config, *, sub_agent=False):
    return CoderAgent(
        project_path=project_path,
        workspace_id="test_workspace",
        thread_id=str(uuid.uuid4()),
        connection_manager=manager,
        config=config,
        agent_mode=AgentMode.CLI,
        sub_agent=sub_agent,
    )


def test_run_workflow_absent_by_default(project_path, mock_connection_manager, agent_config):
    """A top-level coder without gigacode enabled does not expose run_workflow."""
    agent = _coder(project_path, mock_connection_manager, agent_config)
    names = {tool.name for tool in agent.tool_collection.get_tool_list()}
    assert "run_workflow" not in names


def test_run_workflow_appears_when_enabled(project_path, mock_connection_manager, agent_config):
    """Enabling gigacode exposes run_workflow with its explicit input schema."""
    agent = _coder(project_path, mock_connection_manager, agent_config)
    agent.gigacode_enabled = True

    tools = agent.tool_collection.get_tool_list()
    by_name = {tool.name: tool for tool in tools}
    assert "run_workflow" in by_name

    # The explicit schema must be applied (args is free-form, no `type`).
    schema = by_name["run_workflow"].input_schema
    assert schema is not None
    assert "args" in schema["properties"]
    assert "type" not in schema["properties"]["args"]


def test_sub_agent_never_gets_run_workflow(project_path, mock_connection_manager, agent_config):
    """A dispatched (sub) coder must not get run_workflow even with the flag set."""
    agent = _coder(project_path, mock_connection_manager, agent_config, sub_agent=True)
    agent.gigacode_enabled = True
    names = {tool.name for tool in agent.tool_collection.get_tool_list()}
    assert "run_workflow" not in names


def test_general_agent_never_gets_run_workflow(project_path, mock_connection_manager, agent_config):
    """GeneralAgent (always a sub-agent) never exposes run_workflow."""
    agent = GeneralAgent(
        project_path=project_path,
        workspace_id="test_workspace",
        thread_id=str(uuid.uuid4()),
        connection_manager=mock_connection_manager,
        config=agent_config,
    )
    agent.gigacode_enabled = True
    names = {tool.name for tool in agent.tool_collection.get_tool_list()}
    assert "run_workflow" not in names


def test_planning_agent_gets_run_workflow_when_enabled(project_path, mock_connection_manager, agent_config):
    """A top-level planning agent (plan mode) can orchestrate; it is read-only so its
    workflow sub-agents will be forced read-only at dispatch."""
    agent = PlanningAgent(
        project_path=project_path,
        workspace_id="test_workspace",
        thread_id=str(uuid.uuid4()),
        connection_manager=mock_connection_manager,
        config=agent_config,
        agent_mode=AgentMode.CLI,
    )
    agent.gigacode_enabled = True
    names = {tool.name for tool in agent.tool_collection.get_tool_list()}
    assert "run_workflow" in names
    # The orchestrator is read-only, which the dispatch adapter uses to force
    # read-only sub-agents.
    assert agent.tool_collection.read_only is True


def test_sub_agent_extensions_filter_drops_non_propagating():
    """The AgentTool filter keeps only extensions marked to propagate to sub-agents."""
    keep_tool = ToolExtension(name="keep", tools={})
    drop_tool = ToolExtension(name="drop", tools={}, propagate_to_sub_agents=False)
    keep_prompt = PromptExtension(id="keep", title="k", markdown="m")
    drop_prompt = PromptExtension(id="drop", title="d", markdown="m", propagate_to_sub_agents=False)

    assert AgentTool._sub_agent_extensions([keep_tool, drop_tool]) == [keep_tool]
    assert AgentTool._sub_agent_extensions([keep_prompt, drop_prompt]) == [keep_prompt]
    assert AgentTool._sub_agent_extensions(None) is None
    assert AgentTool._sub_agent_extensions([]) == []


def test_workflow_sub_agent_does_not_inherit_task_list(project_path, mock_connection_manager, agent_config):
    """A sub-agent constructed by AgentTool does not inherit a non-propagating
    (task-list) extension carried by its caller, even though the caller has it."""

    async def get_task_list() -> str:
        return ""

    async def update_task_list(task_list_markdown: str) -> str:
        return ""

    task_list_ext = ToolExtension(
        name="cli-shared-task-list",
        tools={"get_task_list": get_task_list, "update_task_list": update_task_list},
        tool_groups={"planning_tools": ["get_task_list", "update_task_list"]},
        propagate_to_sub_agents=False,
    )

    caller = CoderAgent(
        project_path=project_path,
        workspace_id="test_workspace",
        thread_id=str(uuid.uuid4()),
        connection_manager=mock_connection_manager,
        config=agent_config,
        agent_mode=AgentMode.CLI,
        tool_extensions=[task_list_ext],
    )
    # The top-level caller does expose the task-list tools.
    caller_tools = {tool.name for tool in caller.tool_collection.get_tool_list()}
    assert {"get_task_list", "update_task_list"} <= caller_tools

    agent_tool = AgentTool(
        project_path,
        "test_workspace",
        str(uuid.uuid4()),
        mock_connection_manager,
        agent_config,
        caller,
        None,
    )
    sub_agent = agent_tool._construct_workflow_sub_agent(GeneralAgent, None, [])
    sub_tools = {tool.name for tool in sub_agent.tool_collection.get_tool_list()}
    assert "get_task_list" not in sub_tools
    assert "update_task_list" not in sub_tools


def test_apply_gigacode_toggles_flag(project_path, mock_connection_manager, agent_config):
    """apply_gigacode flips the gate and is reflected in the registry."""
    agent = _coder(project_path, mock_connection_manager, agent_config)

    agent.apply_gigacode(True, None)
    assert agent.gigacode_enabled is True
    assert "run_workflow" in {tool.name for tool in agent.tool_collection.get_tool_list()}

    agent.apply_gigacode(False, None)
    assert agent.gigacode_enabled is False
    assert "run_workflow" not in {tool.name for tool in agent.tool_collection.get_tool_list()}
