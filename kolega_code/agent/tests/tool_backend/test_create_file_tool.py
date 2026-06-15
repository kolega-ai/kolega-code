from unittest.mock import AsyncMock, Mock, patch

import pytest
import uuid

from kolega_code.config import AgentConfig, ModelConfig, ModelProvider, RateLimitConfig
from kolega_code.agent.tool_backend.create_file_tool import CreateFileTool


@pytest.fixture
def mock_connection_manager():
    return AsyncMock()


@pytest.fixture
def project_path(tmp_path):
    return tmp_path


@pytest.fixture
def agent_config():
    return AgentConfig(
        anthropic_api_key="test_key",
        openai_api_key="test-key",
        long_context_config=ModelConfig(
            provider=ModelProvider.ANTHROPIC, model="test-model", rate_limits=RateLimitConfig()
        ),
        fast_config=ModelConfig(provider=ModelProvider.ANTHROPIC, model="test-model", rate_limits=RateLimitConfig()),
        thinking_config=ModelConfig(
            provider=ModelProvider.ANTHROPIC, model="test-model", rate_limits=RateLimitConfig(), thinking_effort="medium"
        ),
    )


@pytest.fixture
def mock_base_agent():
    mock = Mock()
    mock.agent_name = "test_agent"
    return mock


@pytest.fixture
def create_file_tool(project_path, mock_connection_manager, agent_config, mock_base_agent):
    return CreateFileTool(
        project_path, "test_workspace", str(uuid.uuid4()), mock_connection_manager, agent_config, mock_base_agent
    )


class TestCreateFileTool:
    @pytest.mark.asyncio
    async def test_create_file_success(self, create_file_tool, project_path):
        result = await create_file_tool.create_file("test.txt", "Hello World")

        assert "File created successfully" in result
        assert (project_path / "test.txt").exists()
        assert (project_path / "test.txt").read_text() == "Hello World"

    @pytest.mark.asyncio
    async def test_create_file_in_subdirectory(self, create_file_tool, project_path):
        result = await create_file_tool.create_file("subdir/test.txt", "Hello World")

        assert "File created successfully" in result
        assert (project_path / "subdir" / "test.txt").exists()
        assert (project_path / "subdir" / "test.txt").read_text() == "Hello World"

    @pytest.mark.asyncio
    async def test_create_file_already_exists(self, create_file_tool, project_path):
        # Create the file first
        (project_path / "test.txt").write_text("Original content")

        result = await create_file_tool.create_file("test.txt", "New content")

        assert "File already exists" in result
        assert (project_path / "test.txt").read_text() == "Original content"

    @pytest.mark.asyncio
    async def test_create_file_parent_directory_does_not_exist(self, create_file_tool, project_path):
        result = await create_file_tool.create_file("nonexistent/test.txt", "Hello World")

        assert "File created successfully" in result
        assert (project_path / "nonexistent" / "test.txt").exists()
        assert (project_path / "nonexistent" / "test.txt").read_text() == "Hello World"

    @pytest.mark.asyncio
    async def test_create_file_permission_error(self, create_file_tool, project_path):
        with patch("pathlib.Path.write_text", side_effect=PermissionError):
            result = await create_file_tool.create_file("test.txt", "Hello World")

            assert "Permission denied" in result
            assert not (project_path / "test.txt").exists()

    @pytest.mark.asyncio
    async def test_create_file_general_error(self, create_file_tool, project_path):
        with patch("pathlib.Path.write_text", side_effect=Exception("General error")):
            result = await create_file_tool.create_file("test.txt", "Hello World")

            assert "Error creating file" in result
            assert not (project_path / "test.txt").exists()

    @pytest.mark.asyncio
    async def test_create_file_with_empty_content(self, create_file_tool, project_path):
        result = await create_file_tool.create_file("test.txt", "")

        assert "File created successfully" in result
        assert (project_path / "test.txt").exists()
        assert (project_path / "test.txt").read_text() == ""

    @pytest.mark.asyncio
    async def test_create_file_with_multiline_content(self, create_file_tool, project_path):
        content = "Line 1\nLine 2\nLine 3"
        result = await create_file_tool.create_file("test.txt", content)

        assert "File created successfully" in result
        assert (project_path / "test.txt").exists()
        assert (project_path / "test.txt").read_text() == content
