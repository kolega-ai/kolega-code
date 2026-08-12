# ruff: noqa: F401,F811,E402
from pathlib import Path
from unittest.mock import AsyncMock, Mock
import uuid

import pytest

from kolega_code.agent.baseagent import BaseAgent
from kolega_code.config import AgentConfig, EditProtocol, ModelConfig, ModelProvider, RateLimitConfig
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
        edit_protocol=EditProtocol.SEARCH_REPLACE,
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
    collection.edit_tool.edit = AsyncMock()
    collection.edit_tool.multi_edit = AsyncMock()
    collection.edit_tool.write = AsyncMock()
    collection.terminal_tool.execute_terminal_command = AsyncMock()
    collection.read_file_tool.read = AsyncMock()
    collection.memory_tool.read_memory = AsyncMock()
    collection.memory_tool.write_memory = AsyncMock()
    collection.web_fetch_tool.web_fetch = AsyncMock()
    collection.terminal_tool.write_stdin = AsyncMock()

    return collection


@pytest.mark.asyncio
class TestToolCollection:
    async def test_edit(self, tool_collection: AsyncMock) -> None:
        path = "test.txt"
        block = "<<<<<<< SEARCH\nold\n======\nnew\n>>>>>>> REPLACE"
        expected_response = "Updated content"
        tool_collection.edit_tool.edit.return_value = expected_response

        result = await tool_collection.edit(path, block)
        assert result == expected_response
        tool_collection.edit_tool.edit.assert_called_once_with(path, block)

    async def test_multi_edit(self, tool_collection: AsyncMock) -> None:
        path = "test.txt"
        blocks = "<<<<<<< SEARCH\nold\n======\nnew\n>>>>>>> REPLACE"
        expected_response = "Updated content"
        tool_collection.edit_tool.multi_edit.return_value = expected_response

        result = await tool_collection.multi_edit(path, blocks)
        assert result == expected_response
        tool_collection.edit_tool.multi_edit.assert_called_once_with(path, blocks)

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

    async def test_read(self, tool_collection: AsyncMock) -> None:
        path = "test.txt"
        expected_response = "File content"
        tool_collection.read_file_tool.read.return_value = expected_response

        result = await tool_collection.read(path)
        assert result == expected_response
        tool_collection.read_file_tool.read.assert_called_once_with(
            file_path=path, offset=1, limit=None, line_formatter=None
        )

    async def test_read_section(self, tool_collection: AsyncMock) -> None:
        path = "test.txt"
        offset = 1
        limit = 10
        expected_response = "File section"
        tool_collection.read_file_tool.read.return_value = expected_response

        result = await tool_collection.read(path, offset=offset, limit=limit)
        assert result == expected_response
        tool_collection.read_file_tool.read.assert_called_once_with(
            file_path=path, offset=offset, limit=limit, line_formatter=None
        )

    async def test_write(self, tool_collection: AsyncMock) -> None:
        path = "test.txt"
        content = "New file content"
        expected_response = "Wrote file content"
        tool_collection.edit_tool.write.return_value = expected_response

        result = await tool_collection.write(path, content)
        assert result == expected_response
        tool_collection.edit_tool.write.assert_called_once_with(path, content)

    async def test_web_fetch(self, tool_collection: AsyncMock) -> None:
        url = "https://example.com"
        instruction = "Summarize this page"
        expected_response = "Summary"
        tool_collection.web_fetch_tool.web_fetch.return_value = expected_response

        result = await tool_collection.web_fetch(url, instruction)

        assert result == expected_response
        tool_collection.web_fetch_tool.web_fetch.assert_called_once_with(url, instruction)

    @pytest.mark.asyncio
    async def test_read_image_tool_is_registered(self) -> None:
        """read_image is in read_only_tools and has a ToolCollection wrapper."""
        assert "read_image" in ToolCollection.read_only_tools
        assert hasattr(ToolCollection, "read_image")


@pytest.mark.asyncio
class TestScratchpadPathExpansion:
    """File-tool path arguments expand $KOLEGA_SCRATCHPAD at the dispatch choke point."""

    @pytest.fixture
    def real_collection(
        self, project_path: Path, mock_connection_manager: AsyncMock, agent_config: AgentConfig, mock_base_agent: Mock
    ) -> ToolCollection:
        # Real (unmocked) handlers: the expansion must reach the actual tools.
        # A real tool-call id keeps the snapshot-service records serializable.
        mock_base_agent.current_tool_execution_id = "test-call-id"
        return ToolCollection(
            project_path, "test_workspace", str(uuid.uuid4()), mock_connection_manager, agent_config, mock_base_agent
        )

    async def test_write_expands_reference_and_never_creates_literal_dir(
        self, real_collection: ToolCollection, project_path: Path, mock_base_agent: Mock, tmp_path: Path
    ) -> None:
        scratchpad = tmp_path / "scratchpad"
        scratchpad.mkdir()
        mock_base_agent.scratchpad_dir = scratchpad

        result = await real_collection.call("write", path="$KOLEGA_SCRATCHPAD/notes.txt", content="hello")

        assert str(scratchpad / "notes.txt") in result
        assert (scratchpad / "notes.txt").read_text() == "hello"
        # The literal $KOLEGA_SCRATCHPAD directory must never appear in the workspace.
        assert not (project_path / "$KOLEGA_SCRATCHPAD").exists()

    async def test_write_expands_braced_reference(
        self, real_collection: ToolCollection, mock_base_agent: Mock, tmp_path: Path
    ) -> None:
        scratchpad = tmp_path / "scratchpad"
        scratchpad.mkdir()
        mock_base_agent.scratchpad_dir = scratchpad

        result = await real_collection.call("write", path="${KOLEGA_SCRATCHPAD}/notes.txt", content="hi")

        assert str(scratchpad / "notes.txt") in result
        assert (scratchpad / "notes.txt").read_text() == "hi"

    async def test_read_after_write_round_trips_through_reference(
        self, real_collection: ToolCollection, mock_base_agent: Mock, tmp_path: Path
    ) -> None:
        scratchpad = tmp_path / "scratchpad"
        scratchpad.mkdir()
        mock_base_agent.scratchpad_dir = scratchpad

        await real_collection.call("write", path="$KOLEGA_SCRATCHPAD/code.py", content="value = 1\n")
        read_result = await real_collection.call("read", file_path="$KOLEGA_SCRATCHPAD/code.py")

        assert str(scratchpad / "code.py") in read_result
        assert "value = 1" in read_result

    async def test_edit_expands_reference(
        self, real_collection: ToolCollection, mock_base_agent: Mock, tmp_path: Path
    ) -> None:
        scratchpad = tmp_path / "scratchpad"
        scratchpad.mkdir()
        mock_base_agent.scratchpad_dir = scratchpad
        (scratchpad / "code.py").write_text("value = 1\n")

        block = "<<<<<<< SEARCH\nvalue = 1\n=======\nvalue = 2\n>>>>>>> REPLACE"
        result = await real_collection.call("edit", path="$KOLEGA_SCRATCHPAD/code.py", block=block)

        assert str(scratchpad / "code.py") in result
        assert (scratchpad / "code.py").read_text() == "value = 2\n"

    async def test_without_scratchpad_leaves_path_unchanged(self, tool_collection: AsyncMock) -> None:
        tool_collection.caller.scratchpad_dir = None
        tool_collection.edit_tool.write.return_value = "Wrote $KOLEGA_SCRATCHPAD/x.txt"

        result = await tool_collection.call("write", path="$KOLEGA_SCRATCHPAD/x.txt", content="hi")

        assert result == "Wrote $KOLEGA_SCRATCHPAD/x.txt"
        tool_collection.edit_tool.write.assert_called_once_with("$KOLEGA_SCRATCHPAD/x.txt", "hi")
