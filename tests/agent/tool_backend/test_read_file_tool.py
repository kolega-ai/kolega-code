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
    async def test_read_whole_file(self, read_file_tool, sample_file):
        content = await read_file_tool.read("test.txt")
        expected = "# test.txt\n\n```\nLine 1\nLine 2\nLine 3\nLine 4\nLine 5\n```"
        assert content == expected

    async def test_read_not_found(self, read_file_tool):
        with pytest.raises(FileNotFoundError) as exc_info:
            await read_file_tool.read("nonexistent.txt")
        assert str(exc_info.value) == "File not found: nonexistent.txt"

    async def test_read_section(self, read_file_tool, sample_file):
        content = await read_file_tool.read("test.txt", offset=2, limit=3)
        expected = "# test.txt (lines 2-4)\n\n```\nLine 2\nLine 3\nLine 4\n\n```"
        assert content == expected

    async def test_read_section_single_line(self, read_file_tool, sample_file):
        content = await read_file_tool.read("test.txt", offset=1, limit=1)
        expected = "# test.txt (lines 1-1)\n\n```\nLine 1\n\n```"
        assert content == expected

    async def test_read_section_not_found(self, read_file_tool):
        with pytest.raises(FileNotFoundError) as exc_info:
            await read_file_tool.read("nonexistent.txt", offset=1, limit=1)
        assert str(exc_info.value) == "File not found: nonexistent.txt"

    async def test_read_invalid_offset(self, read_file_tool, sample_file):
        with pytest.raises(ValueError) as exc_info:
            await read_file_tool.read("test.txt", offset=0)
        assert str(exc_info.value) == "Offset must be at least 1, got 0"

    async def test_read_invalid_limit(self, read_file_tool, sample_file):
        with pytest.raises(ValueError) as exc_info:
            await read_file_tool.read("test.txt", limit=0)
        assert str(exc_info.value) == "Limit must be at least 1, got 0"

        with pytest.raises(ValueError) as exc_info:
            await read_file_tool.read("test.txt", limit=-3)
        assert str(exc_info.value) == "Limit must be at least 1, got -3"

    async def test_read_offset_exceeds_file_length(self, read_file_tool, sample_file):
        with pytest.raises(ValueError) as exc_info:
            await read_file_tool.read("test.txt", offset=6)
        assert str(exc_info.value) == "Offset 6 exceeds file length 5"

    async def test_read_limit_exceeds_file_length(self, read_file_tool, sample_file):
        content = await read_file_tool.read("test.txt", offset=4, limit=100)
        expected = "# test.txt (lines 4-5)\n\n```\nLine 4\nLine 5\n```"
        assert content == expected

    async def test_read_offset_without_limit_reads_to_end(self, read_file_tool, sample_file):
        content = await read_file_tool.read("test.txt", offset=3)
        expected = "# test.txt (lines 3-5)\n\n```\nLine 3\nLine 4\nLine 5\n```"
        assert content == expected

    async def test_read_line_truncation(self, read_file_tool, project_path):
        """Test that files over 2000 lines are truncated with an actionable notice."""
        # Create a large file with 2500 lines
        large_file_path = project_path / "large_file.txt"
        lines = [f"Line {i}\n" for i in range(1, 2501)]
        large_file_path.write_text("".join(lines))

        content = await read_file_tool.read("large_file.txt")

        # Check that the response indicates truncation
        assert "# large_file.txt (lines 1-2000) (TRUNCATED)" in content
        assert "[Showing lines 1-2000 of 2500 (2000-line limit). Use offset=2001 to continue.]" in content

        # Verify that only 2000 lines are included
        # Count the actual lines in the code block
        code_block_start = content.find("```\n") + 4
        code_block_end = content.rfind("\n```")
        code_content = content[code_block_start:code_block_end]
        actual_lines = code_content.strip().split("\n")
        assert len(actual_lines) == 2000
        assert actual_lines[0] == "Line 1"
        assert actual_lines[-1] == "Line 2000"

    async def test_read_explicit_limit_is_capped_at_2000_lines(self, read_file_tool, project_path):
        large_file_path = project_path / "large_file.txt"
        lines = [f"Line {i}\n" for i in range(1, 2501)]
        large_file_path.write_text("".join(lines))

        content = await read_file_tool.read("large_file.txt", limit=5000)

        assert "(2000-line limit)" in content
        code_block_start = content.find("```\n") + 4
        code_block_end = content.rfind("\n```")
        code_content = content[code_block_start:code_block_end]
        assert len(code_content.strip().split("\n")) == 2000

    async def test_read_byte_truncation(self, read_file_tool, project_path):
        large_file_path = project_path / "large_section.txt"
        # 5 lines of 30_001 bytes each: line 1 fits the 50KB budget, adding
        # line 2 would exceed it, so the excerpt snaps to line boundaries.
        large_file_path.write_text(("a" * 30_000 + "\n") * 5)

        content = await read_file_tool.read("large_section.txt")

        assert "# large_section.txt (lines 1-1) (TRUNCATED)" in content
        assert "[Showing lines 1-1 of 5 (50KB limit). Use offset=2 to continue.]" in content
        code_content = content.split("```\n", 1)[1].rsplit("\n```", 1)[0]
        assert code_content == "a" * 30_000 + "\n"

    async def test_read_byte_budget_counts_utf8_bytes_not_characters(self, read_file_tool, project_path):
        large_file_path = project_path / "wide.txt"
        # Two lines of 20_000 two-byte characters: 40_000 bytes each, 80_000
        # total. A character budget would show both; the 50KB byte budget
        # shows only the first.
        large_file_path.write_text(("é" * 20_000 + "\n") * 2)

        content = await read_file_tool.read("wide.txt")

        assert "# wide.txt (lines 1-1) (TRUNCATED)" in content
        code_content = content.split("```\n", 1)[1].rsplit("\n```", 1)[0]
        assert code_content == "é" * 20_000 + "\n"

    async def test_read_giant_line_cannot_be_displayed(self, read_file_tool, project_path):
        """A single line larger than the byte budget gets an explicit notice."""
        large_file_path = project_path / "large_one_line.html"
        large_file_path.write_text("a" * 100_050)

        content = await read_file_tool.read("large_one_line.html")

        assert "# large_one_line.html (TRUNCATED)" in content
        assert "Line 1 is 100,050 bytes, exceeding the 50KB output budget, so it cannot be displayed." in content
        assert "Use rg or exec_command for targeted extraction" in content
        assert "a partial excerpt would not contain the answer" in content
        code_content = content.split("```\n", 1)[1].rsplit("\n```", 1)[0]
        assert code_content == ""

    async def test_read_giant_line_mid_file_stops_excerpt(self, read_file_tool, project_path):
        """A giant line after some readable lines stops the excerpt at the line boundary."""
        large_file_path = project_path / "mixed.txt"
        large_file_path.write_text("small\n" + "x" * 80_000 + "\n" + "tail\n")

        content = await read_file_tool.read("mixed.txt")

        assert "# mixed.txt (lines 1-1) (TRUNCATED)" in content
        assert (
            "Showing lines 1-1 of 3: line 2 is 80,001 bytes, exceeding the 50KB output budget, so it is "
            "omitted." in content
        )
        assert "Use rg or exec_command for targeted extraction" in content
        assert "the excerpt may not contain the answer" in content
        code_content = content.split("```\n", 1)[1].rsplit("\n```", 1)[0]
        assert code_content == "small\n"

    async def test_read_giant_line_can_be_skipped_with_offset(self, read_file_tool, project_path):
        """Reading past a giant line works: only lines in the requested window are scanned."""
        large_file_path = project_path / "mixed.txt"
        large_file_path.write_text("small\n" + "x" * 80_000 + "\n" + "tail\n")

        content = await read_file_tool.read("mixed.txt", offset=3, limit=10)

        assert "# mixed.txt (lines 3-3)\n\n```\ntail\n\n```" == content

    async def test_read_exactly_at_line_limit(self, read_file_tool, project_path):
        """Test that files with exactly 2000 lines are not truncated."""
        # Create a file with exactly 2000 lines
        exact_limit_file_path = project_path / "exact_limit_file.txt"
        lines = [f"Line {i}\n" for i in range(1, 2001)]
        exact_limit_file_path.write_text("".join(lines))

        content = await read_file_tool.read("exact_limit_file.txt")

        # Check that the response does NOT indicate truncation
        assert "# exact_limit_file.txt\n\n```" in content
        assert "(TRUNCATED)" not in content

    async def test_read_below_line_limit(self, read_file_tool, project_path):
        """Test that files with fewer than 2000 lines are not truncated."""
        # Create a file with 1999 lines
        below_limit_file_path = project_path / "below_limit_file.txt"
        lines = [f"Line {i}\n" for i in range(1, 2000)]
        below_limit_file_path.write_text("".join(lines))

        content = await read_file_tool.read("below_limit_file.txt")

        # Check that the response does NOT indicate truncation
        assert "# below_limit_file.txt\n\n```" in content
        assert "(TRUNCATED)" not in content

    async def test_read_line_exactly_fits_byte_budget(self, read_file_tool, project_path):
        """A single line exactly at the 50KB budget is displayed in full."""
        exact_file_path = project_path / "exact_size.txt"
        exact_file_path.write_text("a" * (50 * 1024))

        content = await read_file_tool.read("exact_size.txt")

        assert "# exact_size.txt\n\n```" in content
        assert "(TRUNCATED)" not in content
        code_content = content.split("```\n", 1)[1].rsplit("\n```", 1)[0]
        assert code_content == "a" * (50 * 1024)
