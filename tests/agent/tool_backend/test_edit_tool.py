from unittest.mock import AsyncMock, Mock, patch

import pytest
import uuid

from kolega_code.config import AgentConfig, ModelConfig, ModelProvider, RateLimitConfig
from kolega_code.agent.tool_backend.edit_tool import EditTool


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
    mock = Mock()
    mock.agent_name = "test_agent"
    mock.sub_agent = False
    mock.current_tool_execution_id = "test-call-id"
    return mock


@pytest.fixture
def edit_tool(project_path, mock_connection_manager, agent_config, mock_base_agent):
    return EditTool(
        project_path, "test_workspace", str(uuid.uuid4()), mock_connection_manager, agent_config, mock_base_agent
    )


class MemoryFileSystem:
    def __init__(self, content: str):
        self.content = content

    def exists(self, path: str) -> bool:
        return path == "crlf.txt"

    def read_text(self, path: str) -> str:
        return self.content

    def write_text(self, path: str, content: str) -> None:
        self.content = content

    def get_parent(self, path: str) -> str:
        return "."

    def create_directory(self, path: str) -> None:
        return None


def block(search: str, replace: str) -> str:
    return f"<<<<<<< SEARCH\n{search}\n=======\n{replace}\n>>>>>>> REPLACE"


@pytest.mark.asyncio
class TestEditTool:
    async def test_edit_success_exact_match(self, edit_tool, project_path):
        file_path = project_path / "test.txt"
        file_path.write_text("Line 1\nLine 2\nLine 3\nLine 4\nLine 5")

        result = await edit_tool.edit("test.txt", block("Line 2\nLine 3", "New Line 2\nNew Line 3"))

        assert result == "Edited test.txt"
        assert file_path.read_text() == "Line 1\nNew Line 2\nNew Line 3\nLine 4\nLine 5"

    async def test_edit_success_line_strip_fallback(self, edit_tool, project_path):
        file_path = project_path / "indent.py"
        file_path.write_text("def f():\n\tvalue = 1   \n\treturn value\n")

        result = await edit_tool.edit(
            "indent.py", block("    value = 1\n    return value", "    value = 2\n    return value")
        )

        assert result == "Edited indent.py"
        assert file_path.read_text() == "def f():\n    value = 2\n    return value\n"

    async def test_edit_success_line_ending_fallback(
        self, project_path, mock_connection_manager, agent_config, mock_base_agent
    ):
        filesystem = MemoryFileSystem("prefix\r\nsuffix\r\n")
        edit_tool = EditTool(
            project_path,
            "test_workspace",
            str(uuid.uuid4()),
            mock_connection_manager,
            agent_config,
            mock_base_agent,
            filesystem,
        )

        result = await edit_tool.edit("crlf.txt", block("fix\nsuf", "XX"))

        assert result == "Edited crlf.txt"
        assert filesystem.content == "preXXfix\r\n"

    async def test_edit_success_unicode_punctuation_fallback(self, edit_tool, project_path):
        file_path = project_path / "quotes.txt"
        file_path.write_text('print("hello")\nmessage = "Ada said \u201chello\u201d"\n')

        result = await edit_tool.edit(
            "quotes.txt",
            block('message = "Ada said "hello""', 'message = "Ada said goodbye"'),
        )

        assert result == "Edited quotes.txt"
        assert file_path.read_text() == 'print("hello")\nmessage = "Ada said goodbye"\n'

    async def test_edit_no_match(self, edit_tool, project_path):
        (project_path / "test.txt").write_text("Line 1\nLine 2\n")

        with pytest.raises(ValueError) as exc_info:
            await edit_tool.edit("test.txt", block("Line X", "New Line X"))

        assert "does not match any content in the file" in str(exc_info.value)

    async def test_edit_multiple_matches(self, edit_tool, project_path):
        (project_path / "duplicate.txt").write_text("Line A\nLine B\nLine A\nLine B\n")

        with pytest.raises(ValueError) as exc_info:
            await edit_tool.edit("duplicate.txt", block("Line A\nLine B", "New Line A\nNew Line B"))

        assert "matched 2 occurrences" in str(exc_info.value)

    async def test_edit_no_blocks(self, edit_tool, project_path):
        (project_path / "test.txt").write_text("Line 1\n")

        with pytest.raises(ValueError) as exc_info:
            await edit_tool.edit("test.txt", "Invalid blocks format")

        assert "No valid search and replace blocks found" in str(exc_info.value)

    async def test_edit_rejects_multiple_blocks(self, edit_tool, project_path):
        (project_path / "test.txt").write_text("Line 1\nLine 2\n")
        blocks = block("Line 1", "New Line 1") + "\n" + block("Line 2", "New Line 2")

        with pytest.raises(ValueError) as exc_info:
            await edit_tool.edit("test.txt", blocks)

        assert "Use multi_edit" in str(exc_info.value)

    async def test_edit_empty_search(self, edit_tool, project_path):
        (project_path / "test.txt").write_text("Line 1\n")

        with pytest.raises(ValueError) as exc_info:
            await edit_tool.edit("test.txt", block("", "New Content"))

        assert "Empty search block" in str(exc_info.value)

    async def test_edit_file_not_found(self, edit_tool):
        with pytest.raises(FileNotFoundError) as exc_info:
            await edit_tool.edit("nonexistent.txt", block("Line 1", "New Line 1"))

        assert "File not found: nonexistent.txt" in str(exc_info.value)

    async def test_edit_no_changes(self, edit_tool, project_path):
        file_path = project_path / "test.txt"
        file_path.write_text("Line 1\nLine 2\n")

        result = await edit_tool.edit("test.txt", block("Line 2", "Line 2"))

        assert "No changes made" in result
        assert file_path.read_text() == "Line 1\nLine 2\n"

    async def test_edit_permission_error(self, edit_tool, project_path):
        (project_path / "test.txt").write_text("Line 1\nLine 2\n")

        with patch("pathlib.Path.open", side_effect=PermissionError("Permission denied when writing to file")):
            with pytest.raises(PermissionError) as exc_info:
                await edit_tool.edit("test.txt", block("Line 2", "New Line 2"))

        assert "Permission denied" in str(exc_info.value)

    async def test_multi_edit_applies_non_overlapping_blocks(self, edit_tool, project_path):
        file_path = project_path / "test.txt"
        file_path.write_text("alpha\nbeta\ngamma\n")
        blocks = block("alpha", "ALPHA") + "\n" + block("gamma", "GAMMA")

        result = await edit_tool.multi_edit("test.txt", blocks)

        assert result == "Edited test.txt with 2 replacements"
        assert file_path.read_text() == "ALPHA\nbeta\nGAMMA\n"

    async def test_multi_edit_matches_against_original_snapshot(self, edit_tool, project_path):
        file_path = project_path / "test.txt"
        file_path.write_text("alpha\nbeta\n")
        blocks = block("alpha", "beta") + "\n" + block("beta", "gamma")

        result = await edit_tool.multi_edit("test.txt", blocks)

        assert result == "Edited test.txt with 2 replacements"
        assert file_path.read_text() == "beta\ngamma\n"

    async def test_multi_edit_applies_replacements_in_reverse_offset_order(self, edit_tool, project_path):
        file_path = project_path / "test.txt"
        file_path.write_text("aa\nbb\ncc\n")
        blocks = block("aa", "AAAA") + "\n" + block("cc", "C")

        result = await edit_tool.multi_edit("test.txt", blocks)

        assert result == "Edited test.txt with 2 replacements"
        assert file_path.read_text() == "AAAA\nbb\nC\n"

    async def test_multi_edit_fails_without_writing_when_any_block_fails(self, edit_tool, project_path):
        file_path = project_path / "test.txt"
        file_path.write_text("alpha\nbeta\n")
        blocks = block("alpha", "ALPHA") + "\n" + block("missing", "MISSING")

        with pytest.raises(ValueError):
            await edit_tool.multi_edit("test.txt", blocks)

        assert file_path.read_text() == "alpha\nbeta\n"

    async def test_multi_edit_fails_without_writing_when_blocks_overlap(self, edit_tool, project_path):
        file_path = project_path / "test.txt"
        file_path.write_text("alpha beta gamma\n")
        blocks = block("alpha beta", "one") + "\n" + block("beta gamma", "two")

        with pytest.raises(ValueError) as exc_info:
            await edit_tool.multi_edit("test.txt", blocks)

        assert "overlaps" in str(exc_info.value)
        assert file_path.read_text() == "alpha beta gamma\n"


@pytest.mark.asyncio
class TestWriteTool:
    async def test_write_creates_file(self, edit_tool, project_path):
        result = await edit_tool.write("test.txt", "Hello World")

        assert result == "Wrote test.txt"
        assert (project_path / "test.txt").read_text() == "Hello World"

    async def test_write_creates_missing_parent_directories(self, edit_tool, project_path):
        result = await edit_tool.write("subdir/test.txt", "Hello World")

        assert result == "Wrote subdir/test.txt"
        assert (project_path / "subdir" / "test.txt").read_text() == "Hello World"

    async def test_write_overwrites_existing_file(self, edit_tool, project_path):
        file_path = project_path / "test.txt"
        file_path.write_text("Original content")

        result = await edit_tool.write("test.txt", "New content")

        assert result == "Wrote test.txt"
        assert file_path.read_text() == "New content"

    async def test_write_handles_empty_content(self, edit_tool, project_path):
        result = await edit_tool.write("test.txt", "")

        assert result == "Wrote test.txt"
        assert (project_path / "test.txt").read_text() == ""

    async def test_write_create_emits_head_preview(self, edit_tool, mock_connection_manager):
        result = await edit_tool.write("m.py", "import os\nx = 1\n")

        assert result == "Wrote m.py"
        events = [call.args[0] for call in mock_connection_manager.broadcast_event.await_args_list]
        previews = [event for event in events if event.event_type == "file_edit_preview"]
        assert len(previews) == 1
        content = previews[0].content
        assert content["tool_name"] == "write"
        assert content["tool_call_id"] == "test-call-id"
        assert content["kind"] == "head"
        assert content["path"] == "m.py"

    async def test_write_overwrite_emits_diff_preview(self, edit_tool, mock_connection_manager, project_path):
        (project_path / "m.py").write_text("x = 1\n")

        result = await edit_tool.write("m.py", "x = 2\n")

        assert result == "Wrote m.py"
        events = [call.args[0] for call in mock_connection_manager.broadcast_event.await_args_list]
        previews = [event for event in events if event.event_type == "file_edit_preview"]
        assert len(previews) == 1
        content = previews[0].content
        assert content["tool_name"] == "write"
        assert content["kind"] == "diff"
        assert content["path"] == "m.py"

    async def test_write_permission_error(self, edit_tool, project_path):
        with patch("pathlib.Path.write_text", side_effect=PermissionError("Permission denied")):
            with pytest.raises(PermissionError) as exc_info:
                await edit_tool.write("test.txt", "Hello World")

        assert "Permission denied" in str(exc_info.value)
        assert not (project_path / "test.txt").exists()
