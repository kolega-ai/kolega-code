from unittest.mock import AsyncMock, Mock

import pytest
import uuid

from kolega_code.config import AgentConfig, ModelConfig, ModelProvider, RateLimitConfig
from kolega_code.agent.tool_backend.read_file_tool import ReadFileTool


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
            provider=ModelProvider.ANTHROPIC,
            model="test-model",
            rate_limits=RateLimitConfig(),
            thinking_effort="medium",
        ),
    )


@pytest.fixture
def mock_base_agent():
    return Mock()


@pytest.fixture
def read_file_tool(project_path, mock_connection_manager, agent_config, mock_base_agent):
    return ReadFileTool(
        project_path, "test_workspace", str(uuid.uuid4()), mock_connection_manager, agent_config, mock_base_agent
    )


@pytest.fixture
def sample_file(project_path):
    file_path = project_path / "test.txt"
    file_path.write_text("Line 1\nLine 2\nLine 3\nLine 4\nLine 5")
    return file_path


@pytest.mark.asyncio
class TestReadFileTool:
    async def test_read_entire_file(self, read_file_tool, sample_file):
        content = await read_file_tool.read_entire_file("test.txt")
        expected = "# test.txt\n\n```\nLine 1\nLine 2\nLine 3\nLine 4\nLine 5\n```"
        assert content == expected

    async def test_read_entire_file_not_found(self, read_file_tool):
        with pytest.raises(FileNotFoundError) as exc_info:
            await read_file_tool.read_entire_file("nonexistent.txt")
        assert str(exc_info.value) == "File not found: nonexistent.txt"

    async def test_read_file_section(self, read_file_tool, sample_file):
        content = await read_file_tool.read_file_section("test.txt", 2, 4)
        expected = "# test.txt (lines 2-4)\n\n```\nLine 2\nLine 3\nLine 4\n\n```"
        assert content == expected

    async def test_read_file_section_single_line(self, read_file_tool, sample_file):
        content = await read_file_tool.read_file_section("test.txt", 1, 1)
        expected = "# test.txt (lines 1-1)\n\n```\nLine 1\n\n```"
        assert content == expected

    async def test_read_file_section_not_found(self, read_file_tool):
        with pytest.raises(FileNotFoundError) as exc_info:
            await read_file_tool.read_file_section("nonexistent.txt", 1, 1)
        assert str(exc_info.value) == "File not found: nonexistent.txt"

    async def test_read_file_section_invalid_start_line(self, read_file_tool, sample_file):
        with pytest.raises(ValueError) as exc_info:
            await read_file_tool.read_file_section("test.txt", 0, 1)
        assert str(exc_info.value) == "Start line must be at least 1, got 0"

    async def test_read_file_section_invalid_end_line(self, read_file_tool, sample_file):
        with pytest.raises(ValueError) as exc_info:
            await read_file_tool.read_file_section("test.txt", 3, 2)
        assert str(exc_info.value) == "End line (2) must be greater than or equal to start line (3)"

    async def test_read_file_section_start_line_exceeds_file_length(self, read_file_tool, sample_file):
        with pytest.raises(ValueError) as exc_info:
            await read_file_tool.read_file_section("test.txt", 6, 6)
        assert str(exc_info.value) == "Start line 6 exceeds file length 5"

    async def test_read_file_section_end_line_exceeds_file_length(self, read_file_tool, sample_file):
        content = await read_file_tool.read_file_section("test.txt", 4, 10)
        expected = "# test.txt (lines 4-5)\n\n```\nLine 4\nLine 5\n```"
        assert content == expected

    async def test_read_entire_file_truncation(self, read_file_tool, project_path):
        """Test that files over 2000 lines are truncated with a warning."""
        # Create a large file with 2500 lines
        large_file_path = project_path / "large_file.txt"
        lines = [f"Line {i}\n" for i in range(1, 2501)]
        large_file_path.write_text("".join(lines))

        content = await read_file_tool.read_entire_file("large_file.txt")

        # Check that the response indicates truncation
        assert "# large_file.txt (TRUNCATED)" in content
        assert "⚠️ File truncated: Showing first 2000 of 2500 lines" in content
        assert "To read specific sections, use `read_file_section`" in content

        # Verify that only 2000 lines are included
        # Count the actual lines in the code block
        code_block_start = content.find("```\n") + 4
        code_block_end = content.rfind("\n```")
        code_content = content[code_block_start:code_block_end]
        actual_lines = code_content.strip().split("\n")
        assert len(actual_lines) == 2000
        assert actual_lines[0] == "Line 1"
        assert actual_lines[-1] == "Line 2000"

    async def test_read_entire_file_char_truncation(self, read_file_tool, project_path):
        large_file_path = project_path / "large_one_line.html"
        large_file_path.write_text("a" * 100_050)

        content = await read_file_tool.read_entire_file("large_one_line.html")

        assert "# large_one_line.html (TRUNCATED)" in content
        assert "File truncated by size: Showing first 100,000 of 100,050 characters" in content
        code_content = content.split("```\n", 1)[1].rsplit("\n```", 1)[0]
        assert len(code_content) == 100_000

    async def test_read_file_section_char_truncation(self, read_file_tool, project_path):
        large_file_path = project_path / "large_section.txt"
        large_file_path.write_text(("a" * 30_000 + "\n") * 5)

        content = await read_file_tool.read_file_section("large_section.txt", 1, 5)

        assert "# large_section.txt (lines 1-5) (TRUNCATED)" in content
        assert "File truncated by size" in content
        code_content = content.split("```\n", 1)[1].rsplit("\n```", 1)[0]
        assert len(code_content) == 100_000

    async def test_read_entire_file_exactly_at_limit(self, read_file_tool, project_path):
        """Test that files with exactly 2000 lines are not truncated."""
        # Create a file with exactly 2000 lines
        exact_limit_file_path = project_path / "exact_limit_file.txt"
        lines = [f"Line {i}\n" for i in range(1, 2001)]
        exact_limit_file_path.write_text("".join(lines))

        content = await read_file_tool.read_entire_file("exact_limit_file.txt")

        # Check that the response does NOT indicate truncation
        assert "# exact_limit_file.txt\n\n```" in content
        assert "(TRUNCATED)" not in content
        assert "⚠️ File truncated" not in content

    async def test_read_entire_file_below_limit(self, read_file_tool, project_path):
        """Test that files with fewer than 2000 lines are not truncated."""
        # Create a file with 1999 lines
        below_limit_file_path = project_path / "below_limit_file.txt"
        lines = [f"Line {i}\n" for i in range(1, 2000)]
        below_limit_file_path.write_text("".join(lines))

        content = await read_file_tool.read_entire_file("below_limit_file.txt")

        # Check that the response does NOT indicate truncation
        assert "# below_limit_file.txt\n\n```" in content
        assert "(TRUNCATED)" not in content
        assert "⚠️ File truncated" not in content
