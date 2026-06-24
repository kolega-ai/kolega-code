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
    async def test_read_memory(self, tool_collection: AsyncMock) -> None:
        expected_response = "Memory content"
        tool_collection.memory_tool.read_memory.return_value = expected_response

        result = await tool_collection.read_memory()
        assert result == expected_response
        tool_collection.memory_tool.read_memory.assert_called_once()

    async def test_write_memory(self, tool_collection: AsyncMock) -> None:
        memory_content = "New memory"
        expected_response = "Success"
        tool_collection.memory_tool.write_memory.return_value = expected_response

        result = await tool_collection.write_memory(memory_content)
        assert result == expected_response
        tool_collection.memory_tool.write_memory.assert_called_once_with(memory_content)

    async def test_memory_tool_reads_agent_memory_file(
        self,
        project_path: Path,
        mock_connection_manager: AgentConnectionManager,
        agent_config: AgentConfig,
        mock_base_agent: BaseAgent,
    ) -> None:
        (project_path / "AGENT_MEMORY.md").write_text("Memory content", encoding="utf-8")
        memory_tool = MemoryTool(
            project_path,
            "test_workspace",
            str(uuid.uuid4()),
            mock_connection_manager,
            agent_config,
            mock_base_agent,
        )
        memory_tool.log_info = AsyncMock()

        result = await memory_tool.read_memory()

        assert result == "Memory content"
        memory_tool.log_info.assert_called_once_with(
            "Successfully read memory file AGENT_MEMORY.md", sender="test_agent"
        )

    async def test_memory_tool_write_creates_agent_memory_file(
        self,
        project_path: Path,
        mock_connection_manager: AgentConnectionManager,
        agent_config: AgentConfig,
        mock_base_agent: BaseAgent,
    ) -> None:
        memory_tool = MemoryTool(
            project_path,
            "test_workspace",
            str(uuid.uuid4()),
            mock_connection_manager,
            agent_config,
            mock_base_agent,
        )
        memory_tool.log_info = AsyncMock()

        result = await memory_tool.write_memory("New memory")

        assert result == "Created memory file AGENT_MEMORY.md and added new memory"
        assert (project_path / "AGENT_MEMORY.md").read_text(encoding="utf-8") == "# Agent Memory\n\n- New memory\n"

    async def test_memory_tool_write_appends_to_agent_memory_file(
        self,
        project_path: Path,
        mock_connection_manager: AgentConnectionManager,
        agent_config: AgentConfig,
        mock_base_agent: BaseAgent,
    ) -> None:
        (project_path / "AGENT_MEMORY.md").write_text("# Agent Memory\n\n- Existing memory\n", encoding="utf-8")
        memory_tool = MemoryTool(
            project_path,
            "test_workspace",
            str(uuid.uuid4()),
            mock_connection_manager,
            agent_config,
            mock_base_agent,
        )
        memory_tool.log_info = AsyncMock()

        result = await memory_tool.write_memory("New memory")

        assert result == "Successfully added new memory to AGENT_MEMORY.md"
        assert (project_path / "AGENT_MEMORY.md").read_text(
            encoding="utf-8"
        ) == "# Agent Memory\n\n- Existing memory\n- New memory\n"

    async def test_memory_tool_read_missing_agent_memory_file(
        self,
        project_path: Path,
        mock_connection_manager: AgentConnectionManager,
        agent_config: AgentConfig,
        mock_base_agent: BaseAgent,
    ) -> None:
        memory_tool = MemoryTool(
            project_path,
            "test_workspace",
            str(uuid.uuid4()),
            mock_connection_manager,
            agent_config,
            mock_base_agent,
        )
        memory_tool.log_error = AsyncMock()

        with pytest.raises(FileNotFoundError, match="AGENT_MEMORY.md"):
            await memory_tool.read_memory()
