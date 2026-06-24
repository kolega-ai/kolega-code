# ruff: noqa: F401,F811,E402
from pathlib import Path
from unittest.mock import AsyncMock, Mock
import uuid

import pytest

from kolega_code.agent.baseagent import BaseAgent
from kolega_code.config import AgentConfig, ModelConfig, ModelProvider, RateLimitConfig
from kolega_code.events import AgentConnectionManager
from kolega_code.agent.tool_backend.memory_tool import MemoryTool
from kolega_code.agent.tools import ToolCollection, ToolDefinition, ToolCollectionConfig

@pytest.fixture
def mock_connection_manager() -> AsyncMock:
    return AsyncMock()

@pytest.fixture
def project_path(tmp_path: Path) -> Path:
    return tmp_path

@pytest.fixture
def agent_config() -> AgentConfig:
    return AgentConfig(
        anthropic_api_key="test_key",
        openai_api_key="test-key",
        long_context_config=ModelConfig(
            provider=ModelProvider.ANTHROPIC, model="test-model", rate_limits=RateLimitConfig()
        ),
        fast_config=ModelConfig(provider=ModelProvider.ANTHROPIC, model="test-model", rate_limits=RateLimitConfig()),
        thinking_config=ModelConfig(
            provider=ModelProvider.ANTHROPIC,
            model="test-model",
            rate_limits=RateLimitConfig(),
            thinking_effort="medium",
        ),
    )

@pytest.fixture
def mock_base_agent() -> Mock:
    mock = Mock()
    mock.agent_name = "test_agent"
    # Default: non-vision mock so the read_image tool gate excludes it.
    mock.supports_vision = False
    return mock

@pytest.fixture
def tool_collection(
    project_path: Path,
    mock_connection_manager: AgentConnectionManager,
    agent_config: AgentConfig,
    mock_base_agent: BaseAgent,
) -> ToolCollection:
    # Create a ToolCollection with mocked tools
    collection = ToolCollection(
        project_path, "test_workspace", str(uuid.uuid4()), mock_connection_manager, agent_config, mock_base_agent
    )

    # Mock all tool methods
    collection.think_hard_tool.think_hard = AsyncMock()
    collection.search_and_replace_tool.search_and_replace = AsyncMock()
    collection.list_directory_tool.list_directory = AsyncMock()
    collection.terminal_tool.execute_terminal_command = AsyncMock()
    collection.read_file_tool.read_entire_file = AsyncMock()
    collection.read_file_tool.read_file_section = AsyncMock()
    collection.create_file_tool.create_file = AsyncMock()
    collection.replace_entire_file_tool.replace_entire_file = AsyncMock()
    collection.replace_lines_tool.replace_lines = AsyncMock()
    collection.memory_tool.read_memory = AsyncMock()
    collection.memory_tool.write_memory = AsyncMock()
    collection.search_codebase_tool.search_codebase = AsyncMock()
    collection.glob_tool.find_files_by_pattern = AsyncMock()
    collection.web_fetch_tool.web_fetch = AsyncMock()
    collection.terminal_tool.write_stdin = AsyncMock()

    return collection


@pytest.mark.asyncio
class TestToolCollection:
    async def test_tool_collection_extension_tools(
        self,
        project_path: Path,
        mock_connection_manager: AgentConnectionManager,
        agent_config: AgentConfig,
        mock_base_agent: BaseAgent,
    ) -> None:
        """Test that host-provided extension tools can be included by group."""
        from kolega_code.agent.tools import ToolExtension

        async def custom_status() -> str:
            """Return custom host status."""
            return "ok"

        extension = ToolExtension(
            name="test-extension",
            tools={"custom_status": custom_status},
            tool_groups={"host_tools": ["custom_status"]},
        )
        config = ToolCollectionConfig(custom_tool_groups=["host_tools"], restrict_to_tool_groups=True)
        tool_collection = ToolCollection(
            project_path,
            "test_workspace",
            str(uuid.uuid4()),
            mock_connection_manager,
            agent_config,
            mock_base_agent,
            tool_config=config,
            tool_extensions=[extension],
        )

        tool_list = tool_collection.get_tool_list()
        tool_names = [tool.name for tool in tool_list]

        assert tool_names == ["custom_status"]

    async def test_tool_extension_explicit_schema_injected(
        self,
        project_path: Path,
        mock_connection_manager: AgentConnectionManager,
        agent_config: AgentConfig,
        mock_base_agent: BaseAgent,
    ) -> None:
        """An explicit tool_schemas entry overrides the introspected schema on the built tool."""
        from kolega_code.agent.tools import ToolExtension

        async def ask_things(questions: list) -> str:
            """Ask things."""
            return "{}"

        schema = {
            "type": "object",
            "properties": {"questions": {"type": "array", "items": {"type": "object"}}},
            "required": ["questions"],
        }
        extension = ToolExtension(
            name="schema-extension",
            tools={"ask_things": ask_things},
            tool_schemas={"ask_things": schema},
            tool_groups={"host_tools": ["ask_things"]},
        )
        config = ToolCollectionConfig(custom_tool_groups=["host_tools"], restrict_to_tool_groups=True)
        tool_collection = ToolCollection(
            project_path,
            "test_workspace",
            str(uuid.uuid4()),
            mock_connection_manager,
            agent_config,
            mock_base_agent,
            tool_config=config,
            tool_extensions=[extension],
        )

        definition = next(d for d in tool_collection.get_tool_list() if d.name == "ask_things")
        assert definition.input_schema == schema
        assert definition.to_anthropic()["input_schema"] == schema

