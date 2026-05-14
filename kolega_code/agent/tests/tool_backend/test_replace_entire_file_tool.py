from unittest.mock import AsyncMock, Mock, patch

import pytest
import uuid

from kolega_code.agent.config import AgentConfig, ModelConfig, ModelProvider, RateLimitConfig
from kolega_code.agent.tool_backend.replace_entire_file_tool import ReplaceEntireFileTool


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
            provider=ModelProvider.ANTHROPIC, model="test-model", rate_limits=RateLimitConfig(), thinking_tokens=1024
        ),
    )


@pytest.fixture
def mock_base_agent():
    mock = Mock()
    mock.agent_name = "test_agent"
    return mock


@pytest.fixture
def replace_entire_file_tool(project_path, mock_connection_manager, agent_config, mock_base_agent):
    return ReplaceEntireFileTool(
        project_path, "test_workspace", str(uuid.uuid4()), mock_connection_manager, agent_config, mock_base_agent
    )


@pytest.fixture
def sample_file(project_path):
    file_path = project_path / "test.txt"
    file_path.write_text("Original content\nLine 2\nLine 3")
    return file_path


@pytest.mark.asyncio
class TestReplaceEntireFileTool:
    async def test_replace_entire_file_success(self, replace_entire_file_tool, sample_file):
        new_content = "New content\nLine 2\nLine 3"
        result = await replace_entire_file_tool.replace_entire_file("test.txt", new_content)

        assert result == "# test.txt has been replaced."
        assert sample_file.read_text() == new_content

    async def test_replace_entire_file_with_empty_content(self, replace_entire_file_tool, sample_file):
        result = await replace_entire_file_tool.replace_entire_file("test.txt", "")

        assert result == "# test.txt has been replaced."
        assert sample_file.read_text() == ""

    async def test_replace_entire_file_with_multiline_content(self, replace_entire_file_tool, sample_file):
        new_content = "Line 1\n\nLine 3\n\nLine 5"
        result = await replace_entire_file_tool.replace_entire_file("test.txt", new_content)

        assert result == "# test.txt has been replaced."
        assert sample_file.read_text() == new_content

    async def test_replace_entire_file_file_not_found(self, replace_entire_file_tool):
        with pytest.raises(FileNotFoundError) as exc_info:
            await replace_entire_file_tool.replace_entire_file("nonexistent.txt", "Content")
        assert str(exc_info.value) == "File not found: nonexistent.txt"

    @patch("pathlib.Path.write_text")
    async def test_replace_entire_file_permission_error(self, mock_write_text, replace_entire_file_tool, sample_file):
        mock_write_text.side_effect = PermissionError("Permission denied")

        with pytest.raises(PermissionError) as exc_info:
            await replace_entire_file_tool.replace_entire_file("test.txt", "Content")
        assert str(exc_info.value) == "Permission denied"

    @patch("pathlib.Path.write_text")
    async def test_replace_entire_file_general_error(self, mock_write_text, replace_entire_file_tool, sample_file):
        mock_write_text.side_effect = Exception("Unexpected error")

        with pytest.raises(Exception) as exc_info:
            await replace_entire_file_tool.replace_entire_file("test.txt", "Content")
        assert str(exc_info.value) == "Unexpected error"

    async def test_replace_entire_file_preserve_newline(self, replace_entire_file_tool, project_path):
        # Create a file with a trailing newline
        file_path = project_path / "newline.txt"
        file_path.write_text("Line 1\nLine 2\n")

        new_content = "New Line 1\nNew Line 2\n"
        result = await replace_entire_file_tool.replace_entire_file("newline.txt", new_content)

        assert result == "# newline.txt has been replaced."
        assert file_path.read_text() == new_content

    async def test_replace_entire_file_with_special_characters(self, replace_entire_file_tool, sample_file):
        new_content = "Special chars: !@#$%^&*()\nUnicode: 🚀\nTabs:\t\t\t\n"
        result = await replace_entire_file_tool.replace_entire_file("test.txt", new_content)

        assert result == "# test.txt has been replaced."
        assert sample_file.read_text() == new_content
