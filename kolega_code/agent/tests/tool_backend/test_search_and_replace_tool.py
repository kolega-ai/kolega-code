from unittest.mock import AsyncMock, Mock, patch

import pytest
import uuid

from kolega_code.agent.tool_backend.search_and_replace_tool import SearchAndReplaceTool


@pytest.fixture
def project_path(tmp_path):
    return tmp_path


@pytest.fixture
def sample_file(project_path):
    file_path = project_path / "test.txt"
    file_path.write_text("Line 1\nLine 2\nLine 3\nLine 4\nLine 5")
    return file_path


@pytest.fixture
def mock_connection_manager():
    return AsyncMock()


@pytest.fixture
def agent_config():
    return {}


@pytest.fixture
def mock_base_agent():
    mock = Mock()
    mock.agent_name = "test_agent"
    return mock


@pytest.fixture
def search_and_replace_tool(project_path, mock_connection_manager, agent_config, mock_base_agent):
    return SearchAndReplaceTool(
        project_path, "test_workspace", str(uuid.uuid4()), mock_connection_manager, agent_config, mock_base_agent
    )


@pytest.mark.asyncio
class TestSearchAndReplaceTool:
    async def test_search_and_replace_success(self, search_and_replace_tool, sample_file):
        blocks = """<<<<<<< SEARCH
Line 2
Line 3
=======
New Line 2
New Line 3
>>>>>>> REPLACE"""

        result = await search_and_replace_tool.search_and_replace("test.txt", blocks)

        expected_result = (
            "Search and replace in file test.txt\n\n"
            "Replaced:\n"
            "```\nLine 2\nLine 3\n```\n"
            "with:\n"
            "```\nNew Line 2\nNew Line 3\n```"
        )
        assert result == expected_result
        assert sample_file.read_text() == "Line 1\nNew Line 2\nNew Line 3\nLine 4\nLine 5"

    async def test_search_and_replace_no_match(self, search_and_replace_tool, sample_file):
        blocks = """<<<<<<< SEARCH
Line X
Line Y
=======
New Line X
New Line Y
>>>>>>> REPLACE"""

        with pytest.raises(ValueError) as exc_info:
            await search_and_replace_tool.search_and_replace("test.txt", blocks)

        assert "does not match any content in the file" in str(exc_info.value)

    async def test_search_and_replace_multiple_matches(self, search_and_replace_tool, project_path):
        file_path = project_path / "duplicate.txt"
        file_path.write_text("Line A\nLine B\nLine C\nLine A\nLine B\nLine C")

        blocks = """<<<<<<< SEARCH
Line A
Line B
=======
New Line A
New Line B
>>>>>>> REPLACE"""

        with pytest.raises(ValueError) as exc_info:
            await search_and_replace_tool.search_and_replace("duplicate.txt", blocks)

        assert "matched 2 occurrences" in str(exc_info.value)

    async def test_search_and_replace_no_blocks(self, search_and_replace_tool, sample_file):
        with pytest.raises(ValueError) as exc_info:
            await search_and_replace_tool.search_and_replace("test.txt", "Invalid blocks format")

        assert "No valid search and replace blocks found" in str(exc_info.value)

    async def test_search_and_replace_multiple_blocks(self, search_and_replace_tool, sample_file):
        blocks = """<<<<<<< SEARCH
Line 1
=======
New Line 1
>>>>>>> REPLACE
<<<<<<< SEARCH
Line 2
=======
New Line 2
>>>>>>> REPLACE"""

        with pytest.raises(ValueError) as exc_info:
            await search_and_replace_tool.search_and_replace("test.txt", blocks)

        assert "Multiple search and replace blocks provided" in str(exc_info.value)

    async def test_search_and_replace_empty_search(self, search_and_replace_tool, sample_file):
        blocks = """<<<<<<< SEARCH

=======
New Content
>>>>>>> REPLACE"""

        with pytest.raises(ValueError) as exc_info:
            await search_and_replace_tool.search_and_replace("test.txt", blocks)

        assert "Empty search block" in str(exc_info.value)

    async def test_search_and_replace_file_not_found(self, search_and_replace_tool):
        blocks = """<<<<<<< SEARCH
Line 1
=======
New Line 1
>>>>>>> REPLACE"""

        with pytest.raises(FileNotFoundError) as exc_info:
            await search_and_replace_tool.search_and_replace("nonexistent.txt", blocks)

        assert "File not found: nonexistent.txt" in str(exc_info.value)

    async def test_search_and_replace_no_changes(self, search_and_replace_tool, sample_file):
        blocks = """<<<<<<< SEARCH
Line 2
Line 3
=======
Line 2
Line 3
>>>>>>> REPLACE"""

        result = await search_and_replace_tool.search_and_replace("test.txt", blocks)

        assert "No changes made" in result
        assert sample_file.read_text() == "Line 1\nLine 2\nLine 3\nLine 4\nLine 5"

    async def test_search_and_replace_permission_error(self, search_and_replace_tool, sample_file):
        blocks = """<<<<<<< SEARCH
Line 2
Line 3
=======
New Line 2
New Line 3
>>>>>>> REPLACE"""

        # Create a test that we can't write to
        with patch("pathlib.Path.open", side_effect=PermissionError("Permission denied when writing to file")):
            with pytest.raises(PermissionError) as exc_info:
                await search_and_replace_tool.search_and_replace("test.txt", blocks)

            assert "Permission denied" in str(exc_info.value)
