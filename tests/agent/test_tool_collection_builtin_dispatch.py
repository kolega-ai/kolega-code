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
    async def test_think_hard(self, tool_collection: AsyncMock) -> None:
        problem = "Test problem"
        expected_response = "Test response"
        tool_collection.think_hard_tool.think_hard.return_value = expected_response

        result = await tool_collection.think_hard(problem)
        assert result == expected_response
        tool_collection.think_hard_tool.think_hard.assert_called_once_with(problem)

    async def test_search_and_replace(self, tool_collection: AsyncMock) -> None:
        path = "test.txt"
        block = "<<<<<<< SEARCH\nold\n======\nnew\n>>>>>>> REPLACE"
        expected_response = "Updated content"
        tool_collection.search_and_replace_tool.search_and_replace.return_value = expected_response

        result = await tool_collection.search_and_replace(path, block)
        assert result == expected_response
        tool_collection.search_and_replace_tool.search_and_replace.assert_called_once_with(path, block)

    async def test_list_directory(self, tool_collection: AsyncMock) -> None:
        path = "test_dir"
        expected_response = "Directory listing"
        tool_collection.list_directory_tool.list_directory.return_value = expected_response

        result = await tool_collection.list_directory(path)
        assert result == expected_response
        tool_collection.list_directory_tool.list_directory.assert_called_once_with(path)

    async def test_execute_terminal_command(self, tool_collection: AsyncMock) -> None:
        command = "ls -la"
        expected_response = "Command output"
        tool_collection.terminal_tool.execute_terminal_command.return_value = expected_response

        result = await tool_collection.execute_terminal_command(command)
        assert result == expected_response
        tool_collection.terminal_tool.execute_terminal_command.assert_called_once_with(command)

    async def test_write_stdin(self, tool_collection: AsyncMock) -> None:
        expected_response = '{"status": "running"}'
        tool_collection.terminal_tool.write_stdin.return_value = expected_response

        result = await tool_collection.write_stdin("s_1", "Ada\n")

        assert result == expected_response
        tool_collection.terminal_tool.write_stdin.assert_called_once_with(
            "s_1", "Ada\n", yield_time_ms=10000, max_output_tokens=10000
        )

    async def test_read_entire_file(self, tool_collection: AsyncMock) -> None:
        path = "test.txt"
        expected_response = "File content"
        tool_collection.read_file_tool.read_entire_file.return_value = expected_response

        result = await tool_collection.read_entire_file(path)
        assert result == expected_response
        tool_collection.read_file_tool.read_entire_file.assert_called_once_with(path)

    async def test_read_file_section(self, tool_collection: AsyncMock) -> None:
        path = "test.txt"
        start_line = 1
        end_line = 10
        expected_response = "File section"
        tool_collection.read_file_tool.read_file_section.return_value = expected_response

        result = await tool_collection.read_file_section(path, start_line, end_line)
        assert result == expected_response
        tool_collection.read_file_tool.read_file_section.assert_called_once_with(path, start_line, end_line)

    async def test_create_file(self, tool_collection: AsyncMock) -> None:
        path = "test.txt"
        content = "New file content"
        expected_response = "Created file content"
        tool_collection.create_file_tool.create_file.return_value = expected_response

        result = await tool_collection.create_file(path, content)
        assert result == expected_response
        tool_collection.create_file_tool.create_file.assert_called_once_with(path, content)

    async def test_replace_entire_file(self, tool_collection: AsyncMock) -> None:
        path = "test.txt"
        content = "New content"
        expected_response = "Updated content"
        tool_collection.replace_entire_file_tool.replace_entire_file.return_value = expected_response

        result = await tool_collection.replace_entire_file(path, content)
        assert result == expected_response
        tool_collection.replace_entire_file_tool.replace_entire_file.assert_called_once_with(path, content)

    async def test_replace_lines(self, tool_collection: AsyncMock) -> None:
        path = "test.txt"
        start_line = 1
        end_line = 5
        new_content = "New lines"
        expected_response = "Updated content"
        tool_collection.replace_lines_tool.replace_lines.return_value = expected_response

        result = await tool_collection.replace_lines(path, start_line, end_line, new_content)
        assert result == expected_response
        tool_collection.replace_lines_tool.replace_lines.assert_called_once_with(
            path, start_line, end_line, new_content
        )

    async def test_search_codebase(self, tool_collection: AsyncMock) -> None:
        pattern = "test"
        file_pattern = "*.py"
        case_sensitive = True
        expected_response = "Search results"
        tool_collection.search_codebase_tool.search_codebase.return_value = expected_response

        result = await tool_collection.search_codebase(pattern, file_pattern, case_sensitive)
        assert result == expected_response
        tool_collection.search_codebase_tool.search_codebase.assert_called_once_with(
            pattern, file_pattern=file_pattern, case_sensitive=case_sensitive, literal=False
        )

    async def test_web_fetch(self, tool_collection: AsyncMock) -> None:
        url = "https://example.com"
        instruction = "Summarize this page"
        expected_response = "Summary"
        tool_collection.web_fetch_tool.web_fetch.return_value = expected_response

        result = await tool_collection.web_fetch(url, instruction)

        assert result == expected_response
        tool_collection.web_fetch_tool.web_fetch.assert_called_once_with(url, instruction)

    @pytest.mark.asyncio
    async def test_find_files_by_pattern(self, tool_collection: AsyncMock) -> None:
        pattern = "*.py"
        include_directories = True
        show_details = False
        expected_response = "File list"
        tool_collection.glob_tool.find_files_by_pattern.return_value = expected_response

        result = await tool_collection.find_files_by_pattern(pattern, include_directories, show_details)
        assert result == expected_response
        tool_collection.glob_tool.find_files_by_pattern.assert_called_once_with(
            pattern, include_directories=include_directories, show_details=show_details
        )

    async def test_read_image_tool_is_registered(self) -> None:
        """read_image is in read_only_tools and has a ToolCollection wrapper."""
        assert "read_image" in ToolCollection.read_only_tools
        assert hasattr(ToolCollection, "read_image")
