from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch
import uuid

import pytest

from kolega_code.agent.baseagent import BaseAgent
from kolega_code.config import AgentConfig, ModelConfig, ModelProvider, RateLimitConfig
from kolega_code.events import AgentConnectionManager
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
            provider=ModelProvider.ANTHROPIC, model="test-model", rate_limits=RateLimitConfig(), thinking_tokens=1024
        ),
    )


@pytest.fixture
def mock_base_agent() -> Mock:
    mock = Mock()
    mock.agent_name = "test_agent"
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
    collection.apply_edit_tool.edit_file = AsyncMock()
    collection.search_and_replace_tool.search_and_replace = AsyncMock()
    collection.list_directory_tool.list_directory = AsyncMock()
    collection.terminal_tool.execute_terminal_command = AsyncMock()
    collection.read_file_tool.read_entire_file = AsyncMock()
    collection.read_file_tool.read_file_section = AsyncMock()
    collection.create_file_tool.create_file = AsyncMock()
    collection.replace_entire_file_tool.replace_entire_file = AsyncMock()
    collection.replace_lines_tool.replace_lines = AsyncMock()
    collection.apply_patch_tool.apply_patch = AsyncMock()
    collection.memory_tool.read_memory = AsyncMock()
    collection.memory_tool.write_memory = AsyncMock()
    collection.search_codebase_tool.search_codebase = AsyncMock()
    collection.glob_tool.find_files_by_pattern = AsyncMock()
    collection.web_fetch_tool.web_fetch = AsyncMock()
    collection.terminal_tool.send_terminal_input = AsyncMock()

    return collection


@pytest.mark.asyncio
class TestToolCollection:
    """Test cases for the ToolCollection class"""

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
            file_path, "test_workspace", str(uuid.uuid4()), mock_connection_manager, agent_config, mock_base_agent,
            filesystem=sandbox_fs,
        )
        sandbox_fs.validate_root.assert_called_once()
        assert tool_collection.workspace_id == "test_workspace"
        assert tool_collection.project_path == file_path

    async def test_think_hard(self, tool_collection: AsyncMock) -> None:
        problem = "Test problem"
        expected_response = "Test response"
        tool_collection.think_hard_tool.think_hard.return_value = expected_response

        result = await tool_collection.think_hard(problem)
        assert result == expected_response
        tool_collection.think_hard_tool.think_hard.assert_called_once_with(problem)

    async def test_edit_file(self, tool_collection: AsyncMock) -> None:
        relative_path = "test.txt"
        instructions = "instructions"
        code_edit = "test content"
        expected_response = "Updated content"
        tool_collection.apply_edit_tool.edit_file.return_value = expected_response

        result = await tool_collection.edit_file(relative_path, instructions, code_edit)
        assert result == expected_response
        tool_collection.apply_edit_tool.edit_file.assert_called_once_with(relative_path, instructions, code_edit)

    async def test_search_and_replace(self, tool_collection: AsyncMock) -> None:
        relative_path = "test.txt"
        block = "<<<<<<< SEARCH\nold\n======\nnew\n>>>>>>> REPLACE"
        expected_response = "Updated content"
        tool_collection.search_and_replace_tool.search_and_replace.return_value = expected_response

        result = await tool_collection.search_and_replace(relative_path, block)
        assert result == expected_response
        tool_collection.search_and_replace_tool.search_and_replace.assert_called_once_with(relative_path, block)

    async def test_list_directory(self, tool_collection: AsyncMock) -> None:
        relative_path = "test_dir"
        expected_response = "Directory listing"
        tool_collection.list_directory_tool.list_directory.return_value = expected_response

        result = await tool_collection.list_directory(relative_path)
        assert result == expected_response
        tool_collection.list_directory_tool.list_directory.assert_called_once_with(relative_path)

    async def test_execute_terminal_command(self, tool_collection: AsyncMock) -> None:
        command = "ls -la"
        expected_response = "Command output"
        tool_collection.terminal_tool.execute_terminal_command.return_value = expected_response

        result = await tool_collection.execute_terminal_command(command)
        assert result == expected_response
        tool_collection.terminal_tool.execute_terminal_command.assert_called_once_with(command)

    async def test_send_terminal_input(self, tool_collection: AsyncMock) -> None:
        expected_response = "Sent input"
        tool_collection.terminal_tool.send_terminal_input.return_value = expected_response

        result = await tool_collection.send_terminal_input("terminal_1", "Ada", submit=True, command_id="cmd_1")

        assert result == expected_response
        tool_collection.terminal_tool.send_terminal_input.assert_called_once_with(
            "terminal_1", "Ada", submit=True, command_id="cmd_1"
        )

    async def test_read_entire_file(self, tool_collection: AsyncMock) -> None:
        relative_path = "test.txt"
        expected_response = "File content"
        tool_collection.read_file_tool.read_entire_file.return_value = expected_response

        result = await tool_collection.read_entire_file(relative_path)
        assert result == expected_response
        tool_collection.read_file_tool.read_entire_file.assert_called_once_with(relative_path)

    async def test_read_file_section(self, tool_collection: AsyncMock) -> None:
        relative_path = "test.txt"
        start_line = 1
        end_line = 10
        expected_response = "File section"
        tool_collection.read_file_tool.read_file_section.return_value = expected_response

        result = await tool_collection.read_file_section(relative_path, start_line, end_line)
        assert result == expected_response
        tool_collection.read_file_tool.read_file_section.assert_called_once_with(relative_path, start_line, end_line)

    async def test_create_file(self, tool_collection: AsyncMock) -> None:
        relative_path = "test.txt"
        content = "New file content"
        expected_response = "Created file content"
        tool_collection.create_file_tool.create_file.return_value = expected_response

        result = await tool_collection.create_file(relative_path, content)
        assert result == expected_response
        tool_collection.create_file_tool.create_file.assert_called_once_with(relative_path, content)

    async def test_replace_entire_file(self, tool_collection: AsyncMock) -> None:
        relative_path = "test.txt"
        content = "New content"
        expected_response = "Updated content"
        tool_collection.replace_entire_file_tool.replace_entire_file.return_value = expected_response

        result = await tool_collection.replace_entire_file(relative_path, content)
        assert result == expected_response
        tool_collection.replace_entire_file_tool.replace_entire_file.assert_called_once_with(relative_path, content)

    async def test_replace_lines(self, tool_collection: AsyncMock) -> None:
        relative_path = "test.txt"
        start_line = 1
        end_line = 5
        new_content = "New lines"
        expected_response = "Updated content"
        tool_collection.replace_lines_tool.replace_lines.return_value = expected_response

        result = await tool_collection.replace_lines(relative_path, start_line, end_line, new_content)
        assert result == expected_response
        tool_collection.replace_lines_tool.replace_lines.assert_called_once_with(
            relative_path, start_line, end_line, new_content
        )

    async def test_apply_patch(self, tool_collection: AsyncMock) -> None:
        patch_content = "diff content"
        expected_response = "Patched content"
        tool_collection.apply_patch_tool.apply_patch.return_value = expected_response

        result = await tool_collection.apply_patch(patch_content)
        assert result == expected_response
        tool_collection.apply_patch_tool.apply_patch.assert_called_once_with(patch_content)

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

    async def test_search_codebase(self, tool_collection: AsyncMock) -> None:
        pattern = "test"
        file_pattern = "*.py"
        case_sensitive = True
        expected_response = "Search results"
        tool_collection.search_codebase_tool.search_codebase.return_value = expected_response

        result = await tool_collection.search_codebase(pattern, file_pattern, case_sensitive)
        assert result == expected_response
        tool_collection.search_codebase_tool.search_codebase.assert_called_once_with(
            pattern, file_pattern=file_pattern, case_sensitive=case_sensitive, literal=True
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
        assert "send_terminal_input" in tool_names
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
        write_tools = ["create_file", "replace_entire_file", "edit_file"]
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
