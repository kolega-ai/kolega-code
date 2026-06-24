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
    async def test_initialization_with_valid_path(
        self,
        project_path: Path,
        mock_connection_manager: AgentConnectionManager,
        agent_config: AgentConfig,
        mock_base_agent: BaseAgent,
    ) -> None:
        tool_collection = ToolCollection(
            project_path, "test_workspace", str(uuid.uuid4()), mock_connection_manager, agent_config, mock_base_agent
        )
        assert tool_collection.project_path == project_path
        assert tool_collection.workspace_id == "test_workspace"
        assert tool_collection.connection_manager == mock_connection_manager

    async def test_initialization_with_string_path(
        self,
        tmp_path: Path,
        mock_connection_manager: AgentConnectionManager,
        agent_config: AgentConfig,
        mock_base_agent: BaseAgent,
    ) -> None:
        tool_collection = ToolCollection(
            str(tmp_path), "test_workspace", str(uuid.uuid4()), mock_connection_manager, agent_config, mock_base_agent
        )
        assert tool_collection.project_path == tmp_path

    async def test_initialization_with_nonexistent_path(
        self, mock_connection_manager: AgentConnectionManager, agent_config: AgentConfig, mock_base_agent: BaseAgent
    ) -> None:
        # Local filesystems validate their root eagerly
        with pytest.raises(ValueError) as exc_info:
            ToolCollection(
                "/nonexistent/path",
                "test_workspace",
                str(uuid.uuid4()),
                mock_connection_manager,
                agent_config,
                mock_base_agent,
            )
        assert "Project path does not exist" in str(exc_info.value)

    async def test_initialization_with_file_path(
        self,
        tmp_path: Path,
        mock_connection_manager: AgentConnectionManager,
        agent_config: AgentConfig,
        mock_base_agent: BaseAgent,
    ) -> None:
        file_path = tmp_path / "test.txt"
        file_path.touch()
        # Local filesystems validate their root eagerly
        with pytest.raises(ValueError) as exc_info:
            ToolCollection(
                file_path, "test_workspace", str(uuid.uuid4()), mock_connection_manager, agent_config, mock_base_agent
            )
        assert "Project path is not a directory" in str(exc_info.value)

    async def test_initialization_with_nonexistent_path_sandbox_filesystem(
        self, mock_connection_manager: AgentConnectionManager, agent_config: AgentConfig, mock_base_agent: BaseAgent
    ) -> None:
        """Sandbox filesystems don't validate their root locally (validate_root is a no-op)."""
        sandbox_fs = Mock()
        sandbox_fs.validate_root = Mock(return_value=None)
        tool_collection = ToolCollection(
            "/nonexistent/path",
            "test_workspace",
            str(uuid.uuid4()),
            mock_connection_manager,
            agent_config,
            mock_base_agent,
            filesystem=sandbox_fs,
        )
        sandbox_fs.validate_root.assert_called_once()
        assert tool_collection.workspace_id == "test_workspace"
        assert str(tool_collection.project_path) == "/nonexistent/path"

    async def test_initialization_with_file_path_sandbox_filesystem(
        self,
        tmp_path: Path,
        mock_connection_manager: AgentConnectionManager,
        agent_config: AgentConfig,
        mock_base_agent: BaseAgent,
    ) -> None:
        """Sandbox filesystems accept any project path; the sandbox provisions the root."""
        file_path = tmp_path / "test.txt"
        file_path.touch()

        sandbox_fs = Mock()
        sandbox_fs.validate_root = Mock(return_value=None)
        tool_collection = ToolCollection(
            file_path,
            "test_workspace",
            str(uuid.uuid4()),
            mock_connection_manager,
            agent_config,
            mock_base_agent,
            filesystem=sandbox_fs,
        )
        sandbox_fs.validate_root.assert_called_once()
        assert tool_collection.workspace_id == "test_workspace"
        assert tool_collection.project_path == file_path
