from unittest.mock import AsyncMock, Mock, patch

import pytest
import uuid

from kolega_code.agent.config import AgentConfig, ModelConfig, ModelProvider, RateLimitConfig
from kolega_code.agent.tool_backend.replace_lines_tool import ReplaceLinesTool


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
def replace_lines_tool(project_path, mock_connection_manager, agent_config, mock_base_agent):
    return ReplaceLinesTool(
        project_path, "test_workspace", str(uuid.uuid4()), mock_connection_manager, agent_config, mock_base_agent
    )


@pytest.fixture
def sample_file(project_path):
    file_path = project_path / "test.txt"
    file_path.write_text("Line 1\nLine 2\nLine 3\nLine 4\nLine 5")
    return file_path


@pytest.mark.asyncio
class TestReplaceLinesTool:
    async def test_replace_lines_success(self, replace_lines_tool, sample_file):
        new_content = "New Line 2\nNew Line 3"
        result = await replace_lines_tool.replace_lines("test.txt", 2, 3, new_content)

        expected_result = "Replaced lines 2-3 in file test.txt\n\nReplaced:\n```\nLine 2\nLine 3\n```\nwith:\n```\nNew Line 2\nNew Line 3\n```"
        assert result == expected_result
        assert sample_file.read_text() == "Line 1\nNew Line 2\nNew Line 3\nLine 4\nLine 5"

    async def test_replace_lines_single_line(self, replace_lines_tool, sample_file):
        new_content = "New Line 1"
        result = await replace_lines_tool.replace_lines("test.txt", 1, 1, new_content)

        expected_result = (
            "Replaced lines 1-1 in file test.txt\n\nReplaced:\n```\nLine 1\n```\nwith:\n```\nNew Line 1\n```"
        )
        assert result == expected_result
        assert sample_file.read_text() == "New Line 1\nLine 2\nLine 3\nLine 4\nLine 5"

    async def test_replace_lines_at_end(self, replace_lines_tool, sample_file):
        new_content = "New Line 5"
        result = await replace_lines_tool.replace_lines("test.txt", 5, 5, new_content)

        # Use direct string comparison to avoid newline issues
        # Note: No trailing newline for the last line of the file
        expected_result = "Replaced lines 5-5 in file test.txt\n\nReplaced:\n```\nLine 5```\nwith:\n```\nNew Line 5```"
        assert result == expected_result
        assert sample_file.read_text() == "Line 1\nLine 2\nLine 3\nLine 4\nNew Line 5"

    async def test_replace_lines_file_not_found(self, replace_lines_tool):
        with pytest.raises(FileNotFoundError) as exc_info:
            await replace_lines_tool.replace_lines("nonexistent.txt", 1, 1, "Content")
        assert str(exc_info.value) == "File not found: nonexistent.txt"

    async def test_replace_lines_invalid_start_line(self, replace_lines_tool, sample_file):
        with pytest.raises(ValueError) as exc_info:
            await replace_lines_tool.replace_lines("test.txt", 0, 1, "Content")
        assert str(exc_info.value) == "Invalid start_line: 0. Line numbers must be 1-indexed."

    async def test_replace_lines_invalid_end_line(self, replace_lines_tool, sample_file):
        with pytest.raises(ValueError) as exc_info:
            await replace_lines_tool.replace_lines("test.txt", 3, 2, "Content")
        assert (
            str(exc_info.value) == "Invalid line range: end_line (2) must be greater than or equal to start_line (3)."
        )

    async def test_replace_lines_start_line_exceeds_file_length(self, replace_lines_tool, sample_file):
        with pytest.raises(ValueError) as exc_info:
            await replace_lines_tool.replace_lines("test.txt", 6, 6, "Content")
        assert str(exc_info.value) == "Invalid start_line: 6. File only has 5 lines."

    @patch("pathlib.Path.write_text")
    async def test_replace_lines_permission_error(self, mock_write_text, replace_lines_tool, sample_file):
        mock_write_text.side_effect = PermissionError("Permission denied")

        with pytest.raises(PermissionError) as exc_info:
            await replace_lines_tool.replace_lines("test.txt", 1, 1, "Content")
        assert str(exc_info.value) == "Permission denied"

    async def test_replace_lines_with_empty_content(self, replace_lines_tool, sample_file):
        result = await replace_lines_tool.replace_lines("test.txt", 2, 2, "")

        expected_result = "Replaced lines 2-2 in file test.txt\n\nReplaced:\n```\nLine 2\n```\nwith:\n```\n\n```"
        assert result == expected_result
        assert sample_file.read_text() == "Line 1\n\nLine 3\nLine 4\nLine 5"

    async def test_replace_lines_preserve_newline(self, replace_lines_tool, project_path):
        # Create a file with a trailing newline
        file_path = project_path / "newline.txt"
        file_path.write_text("Line 1\nLine 2\n")

        new_content = "New Line 1\nNew Line 2"
        result = await replace_lines_tool.replace_lines("newline.txt", 1, 2, new_content)

        expected_result = "Replaced lines 1-2 in file newline.txt\n\nReplaced:\n```\nLine 1\nLine 2\n```\nwith:\n```\nNew Line 1\nNew Line 2\n```"
        assert result == expected_result
        assert file_path.read_text() == "New Line 1\nNew Line 2\n"

    async def test_replace_lines_with_multiple_newlines(self, replace_lines_tool, sample_file):
        new_content = "New Line 2\n\nNew Line 4"
        result = await replace_lines_tool.replace_lines("test.txt", 2, 3, new_content)

        expected_result = "Replaced lines 2-3 in file test.txt\n\nReplaced:\n```\nLine 2\nLine 3\n```\nwith:\n```\nNew Line 2\n\nNew Line 4\n```"
        assert result == expected_result
        assert sample_file.read_text() == "Line 1\nNew Line 2\n\nNew Line 4\nLine 4\nLine 5"
