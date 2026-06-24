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
    async def test_get_tool_list(self, tool_collection: AsyncMock) -> None:
        tool_list = tool_collection.get_tool_list()
        assert isinstance(tool_list, list)
        assert len(tool_list) > 0

        # Check that each tool has required fields
        for tool in tool_list:
            assert isinstance(tool, ToolDefinition)
            assert hasattr(tool, "name")
            assert hasattr(tool, "description")
            assert hasattr(tool, "parameters")
            assert isinstance(tool.parameters, list)
            for param in tool.parameters:
                assert hasattr(param, "name")
                assert hasattr(param, "type")
                assert hasattr(param, "description")
                assert hasattr(param, "required")

        # Check that excluded tools are not in the list
        excluded_tools = tool_collection.tool_exclusions
        tool_names = [tool.name for tool in tool_list]
        assert "exec_command" in tool_names
        assert "write_stdin" in tool_names
        for excluded_tool in excluded_tools:
            assert excluded_tool not in tool_names

    async def test_tool_collection_config_read_only(
        self,
        project_path: Path,
        mock_connection_manager: AgentConnectionManager,
        agent_config: AgentConfig,
        mock_base_agent: BaseAgent,
    ) -> None:
        """Test that read_only configuration properly filters tools."""
        config = ToolCollectionConfig(read_only=True)
        tool_collection = ToolCollection(
            project_path,
            "test_workspace",
            str(uuid.uuid4()),
            mock_connection_manager,
            agent_config,
            mock_base_agent,
            tool_config=config,
        )

        tool_list = tool_collection.get_tool_list()
        tool_names = [tool.name for tool in tool_list]

        # Should only include read-only tools
        for tool_name in tool_names:
            assert tool_name in ToolCollection.read_only_tools

        # Should not include write tools
        write_tools = ["create_file", "replace_entire_file"]
        for write_tool in write_tools:
            assert write_tool not in tool_names

    async def test_tool_collection_config_browser_only(
        self,
        project_path: Path,
        mock_connection_manager: AgentConnectionManager,
        agent_config: AgentConfig,
        mock_base_agent: BaseAgent,
    ) -> None:
        """Test that browser_only configuration properly filters tools."""
        config = ToolCollectionConfig(browser_only=True)
        tool_collection = ToolCollection(
            project_path,
            "test_workspace",
            str(uuid.uuid4()),
            mock_connection_manager,
            agent_config,
            mock_base_agent,
            tool_config=config,
        )

        tool_list = tool_collection.get_tool_list()
        tool_names = [tool.name for tool in tool_list]

        # Should only include browser tools
        for tool_name in tool_names:
            assert tool_name in ToolCollection.browser_tools

        # Should not include file tools
        file_tools = ["read_entire_file", "create_file", "list_directory"]
        for file_tool in file_tools:
            assert file_tool not in tool_names

    async def test_tool_collection_config_mixed_options(
        self,
        project_path: Path,
        mock_connection_manager: AgentConnectionManager,
        agent_config: AgentConfig,
        mock_base_agent: BaseAgent,
    ) -> None:
        """Test combinations of configuration options work correctly."""
        config = ToolCollectionConfig(include_agent_dispatch_tools=True, tool_exclusions=["think_hard"])
        tool_collection = ToolCollection(
            project_path,
            "test_workspace",
            str(uuid.uuid4()),
            mock_connection_manager,
            agent_config,
            mock_base_agent,
            tool_config=config,
        )

        tool_list = tool_collection.get_tool_list()
        tool_names = [tool.name for tool in tool_list]

        # Should exclude explicitly excluded tools
        assert "think_hard" not in tool_names

        # Should include investigation tools
        assert "dispatch_investigation_agent" in tool_names
        assert "dispatch_browser_agent" in tool_names

    async def test_backward_compatibility_legacy_parameters(
        self,
        project_path: Path,
        mock_connection_manager: AgentConnectionManager,
        agent_config: AgentConfig,
        mock_base_agent: BaseAgent,
    ) -> None:
        """Test that legacy read_only and browser_only parameters still work."""
        # Test legacy read_only parameter
        tool_collection_ro = ToolCollection(
            project_path,
            "test_workspace",
            str(uuid.uuid4()),
            mock_connection_manager,
            agent_config,
            mock_base_agent,
            read_only=True,
        )

        tool_list_ro = tool_collection_ro.get_tool_list()
        tool_names_ro = [tool.name for tool in tool_list_ro]

        for tool_name in tool_names_ro:
            assert tool_name in ToolCollection.read_only_tools

        # Test legacy browser_only parameter
        tool_collection_browser = ToolCollection(
            project_path,
            "test_workspace",
            str(uuid.uuid4()),
            mock_connection_manager,
            agent_config,
            mock_base_agent,
            browser_only=True,
        )

        tool_list_browser = tool_collection_browser.get_tool_list()
        tool_names_browser = [tool.name for tool in tool_list_browser]

        for tool_name in tool_names_browser:
            assert tool_name in ToolCollection.browser_tools

