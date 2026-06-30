import base64
import uuid
from unittest.mock import AsyncMock, Mock

import pytest

from kolega_code.config import AgentConfig, ModelConfig, ModelProvider, RateLimitConfig
from kolega_code.events import AgentConnectionManager
from kolega_code.llm.models import Message, TextBlock, ToolCall
from kolega_code.llm.providers.models import TokenCount
from kolega_code.agent.planningagent import PlanningAgent
from kolega_code.agent.prompt_provider import AgentMode
from kolega_code.agent.tools import ToolExtension


def _deepseek_config() -> AgentConfig:
    model_config = ModelConfig(
        provider=ModelProvider.DEEPSEEK,
        model="deepseek-v4-pro",
        rate_limits=RateLimitConfig(),
    )
    return AgentConfig(
        deepseek_api_key="test-key",
        long_context_config=model_config,
        fast_config=model_config,
        thinking_config=model_config,
    )


def _image_attachment() -> dict:
    return {
        "type": "image",
        "media_type": "image/png",
        "data": base64.b64encode(b"fake-image-data").decode("utf-8"),
        "filename": "test-image.png",
    }


class _EmptyStream:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    def __aiter__(self):
        return self

    async def __anext__(self):
        raise StopAsyncIteration

    async def get_final_message(self):
        return Message("assistant", [TextBlock("done")], stop_reason="end_turn")


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
    config.long_context_config.thinking_effort = None
    config.openai_api_key = "test_key"
    config.anthropic_api_key = "test_key"
    config.browser_use_headless = True
    config.agent_models = {}
    config.model_config_for_agent.return_value = config.long_context_config
    return config


@pytest.mark.asyncio
async def test_planning_agent_uses_host_task_list_extension(tmp_path, mock_connection_manager, agent_config):
    task_list = ""

    async def update_task_list(task_list_markdown: str) -> str:
        nonlocal task_list
        task_list = task_list_markdown.strip()
        return "Task list updated."

    async def get_task_list() -> str:
        return task_list or "No task list has been set."

    agent = PlanningAgent(
        project_path=tmp_path,
        workspace_id="test_workspace",
        thread_id=str(uuid.uuid4()),
        connection_manager=mock_connection_manager,
        config=agent_config,
        agent_mode=AgentMode.CLI,
        tool_extensions=[
            ToolExtension(
                name="host-task-list",
                tools={"update_task_list": update_task_list, "get_task_list": get_task_list},
                tool_groups={"planning_tools": ["update_task_list", "get_task_list"]},
            )
        ],
    )

    result = await agent.tool_collection.update_task_list("- [ ] inspect CLI\n- [x] choose tool shape")
    tool_names = {tool.name for tool in agent.tool_collection.get_tool_list()}

    assert result == "Task list updated."
    assert task_list == "- [ ] inspect CLI\n- [x] choose tool shape"
    assert await agent.tool_collection.get_task_list() == task_list
    assert {"write_plan", "get_task_list", "update_task_list"}.issubset(tool_names)
    assert "edit" not in tool_names
    assert "multi_edit" not in tool_names
    assert "write" not in tool_names


def test_planning_agent_only_exposes_write_plan_without_host_task_tools(
    tmp_path, mock_connection_manager, agent_config
):
    agent = PlanningAgent(
        project_path=tmp_path,
        workspace_id="test_workspace",
        thread_id=str(uuid.uuid4()),
        connection_manager=mock_connection_manager,
        config=agent_config,
        agent_mode=AgentMode.CLI,
    )

    tool_names = {tool.name for tool in agent.tool_collection.get_tool_list()}

    assert "write_plan" in tool_names
    assert "update_task_list" not in tool_names
    assert "get_task_list" not in tool_names


@pytest.mark.asyncio
async def test_planning_agent_rejects_unavailable_file_edit_tool(tmp_path, mock_connection_manager, agent_config):
    target = tmp_path / "notes.txt"
    target.write_text("original\n", encoding="utf-8")
    agent = PlanningAgent(
        project_path=tmp_path,
        workspace_id="test_workspace",
        thread_id=str(uuid.uuid4()),
        connection_manager=mock_connection_manager,
        config=agent_config,
        agent_mode=AgentMode.CLI,
    )

    result = await agent.execute_single_tool(
        ToolCall(
            id="tool-call-1",
            name="write",
            input={"path": "notes.txt", "content": "mutated\n"},
        )
    )

    assert result.is_error is True
    assert result.content == "Tool 'write' is not available in this mode."
    assert target.read_text(encoding="utf-8") == "original\n"


@pytest.mark.asyncio
async def test_planning_agent_write_plan_is_consumable(tmp_path, mock_connection_manager, agent_config):
    agent = PlanningAgent(
        project_path=tmp_path,
        workspace_id="test_workspace",
        thread_id=str(uuid.uuid4()),
        connection_manager=mock_connection_manager,
        config=agent_config,
        agent_mode=AgentMode.CLI,
    )

    result = await agent.tool_collection.write_plan("# Plan\n\nImplement planning mode.")

    assert result == "Plan captured."
    assert agent.consume_completed_plan() == "# Plan\n\nImplement planning mode."
    assert agent.consume_completed_plan() is None


@pytest.mark.asyncio
async def test_planning_agent_rejects_deepseek_image_without_llm_call(tmp_path, mock_connection_manager):
    agent = PlanningAgent(
        project_path=tmp_path,
        workspace_id="test_workspace",
        thread_id=str(uuid.uuid4()),
        connection_manager=mock_connection_manager,
        config=_deepseek_config(),
        agent_mode=AgentMode.CLI,
    )
    agent.llm = Mock()

    chunks = [chunk async for chunk in agent.process_message_stream("Plan from this screenshot", [_image_attachment()])]

    assert len(chunks) == 1
    assert chunks[0]["type"] == "response"
    assert "does not support image input" in chunks[0]["content"]
    assert "deepseek-v4-pro" in chunks[0]["content"]
    assert chunks[0]["complete"] is True
    assert agent.history == []
    agent.llm.stream.assert_not_called()


@pytest.mark.asyncio
async def test_planning_agent_does_not_print_context_token_counts(tmp_path, mock_connection_manager, capsys):
    agent = PlanningAgent(
        project_path=tmp_path,
        workspace_id="test_workspace",
        thread_id=str(uuid.uuid4()),
        connection_manager=mock_connection_manager,
        config=_deepseek_config(),
        agent_mode=AgentMode.CLI,
    )
    agent.count_current_context = AsyncMock(return_value=TokenCount(input_tokens=42))
    agent.llm = Mock()
    agent.llm.stream = AsyncMock(return_value=_EmptyStream())

    chunks = [chunk async for chunk in agent.process_message_stream("hello")]

    assert chunks[-1]["complete"] is True
    assert capsys.readouterr().out == ""


def test_planning_agent_exposes_command_tools_but_not_file_edits(tmp_path, mock_connection_manager, agent_config):
    """PlanningAgent can run investigative shell commands but cannot edit files."""
    agent = PlanningAgent(
        project_path=tmp_path,
        workspace_id="test_workspace",
        thread_id=str(uuid.uuid4()),
        connection_manager=mock_connection_manager,
        config=agent_config,
        agent_mode=AgentMode.CLI,
    )

    tool_names = {tool.name for tool in agent.tool_collection.get_tool_list()}

    assert {"exec_command", "write_stdin", "kill_command", "list_sessions"} <= tool_names
    assert "edit" not in tool_names
    assert "multi_edit" not in tool_names
    assert "write" not in tool_names
    # read_only stays True so gigacode workflow sub-agents are still forced read-only.
    assert agent.tool_collection.read_only is True
