from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest
import uuid

from kolega_code.config import AgentConfig, ModelConfig, ModelProvider, RateLimitConfig
from kolega_code.agent.tool_backend.edit_tool import EditTool
from kolega_code.sandbox.filesystem import SandboxFileSystem
from kolega_code.services.file_system import LocalFileSystem


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


class MemoryFileSystem(LocalFileSystem):
    """CRLF-preserving in-memory filesystem (mimics sandbox/E2B reads)."""

    def __init__(self, content: str):
        self.content = content

    def exists(self, path: str) -> bool:
        return path == "crlf.txt"

    def read_text(self, path: str) -> str:
        return self.content

    def read_bytes(self, path: str) -> bytes:
        return self.content.encode("utf-8")

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

    async def test_legacy_edit_multi_edit_and_write_support_external_paths(
        self,
        project_path: Path,
        mock_connection_manager: AsyncMock,
        agent_config: AgentConfig,
        mock_base_agent: Mock,
    ) -> None:
        project = project_path / "project"
        outside = project_path / "outside"
        project.mkdir()
        outside.mkdir()
        tool = EditTool(
            project,
            "test_workspace",
            str(uuid.uuid4()),
            mock_connection_manager,
            agent_config,
            mock_base_agent,
        )
        absolute_edit = outside / "absolute-edit.txt"
        relative_multi_edit = outside / "relative-multi-edit.txt"
        absolute_edit.write_text("before\n")
        relative_multi_edit.write_text("alpha\nbeta\n")

        assert await tool.edit(str(absolute_edit), block("before", "after")) == f"Edited {absolute_edit}"
        assert (
            await tool.multi_edit(
                "../outside/relative-multi-edit.txt",
                block("alpha", "ALPHA") + "\n" + block("beta", "BETA"),
            )
            == "Edited ../outside/relative-multi-edit.txt with 2 replacements"
        )
        assert await tool.write("../outside/relative-write.txt", "relative\n") == "Wrote ../outside/relative-write.txt"
        absolute_write = outside / "absolute-write.txt"
        assert await tool.write(str(absolute_write), "absolute\n") == f"Wrote {absolute_write}"

        assert absolute_edit.read_text() == "after\n"
        assert relative_multi_edit.read_text() == "ALPHA\nBETA\n"
        assert (outside / "relative-write.txt").read_text() == "relative\n"
        assert absolute_write.read_text() == "absolute\n"

    # --- line-ending handling: parse tolerance (the reported failure) ---

    async def test_edit_parses_block_with_crlf_markers(self, edit_tool, project_path):
        """CRLF at the SEARCH/=======/REPLACE marker lines parses and applies."""
        file_path = project_path / "src.py"
        file_path.write_bytes(b"def foo():\r\n    return 1\r\n\r\ndef bar():\r\n    return 2\r\n")
        crlf_block = (
            "<<<<<<< SEARCH\r\ndef foo():\r\n    return 1\r\n=======\r\ndef foo():\r\n    return 99\r\n>>>>>>> REPLACE"
        )

        result = await edit_tool.edit("src.py", crlf_block)

        assert result == "Edited src.py"
        after = file_path.read_bytes()
        assert b"def foo():\r\n    return 99\r\n" in after
        # unchanged lines keep CRLF; no bare LF introduced
        assert b"\n" not in after.replace(b"\r\n", b"")

    async def test_multi_edit_parses_block_with_crlf_markers(self, edit_tool, project_path):
        """multi_edit tolerates CRLF marker lines across multiple blocks."""
        file_path = project_path / "src.py"
        file_path.write_bytes(b"alpha\r\nbeta\r\ngamma\r\n")
        crlf_blocks = (
            "<<<<<<< SEARCH\r\nalpha\r\n=======\r\nALPHA\r\n>>>>>>> REPLACE\r\n"
            "<<<<<<< SEARCH\r\ngamma\r\n=======\r\nGAMMA\r\n>>>>>>> REPLACE"
        )

        result = await edit_tool.multi_edit("src.py", crlf_blocks)

        assert result == "Edited src.py with 2 replacements"
        assert file_path.read_bytes() == b"ALPHA\r\nbeta\r\nGAMMA\r\n"

    async def test_edit_crlf_markers_lf_content_parses(self, edit_tool, project_path):
        """CRLF marker lines with LF content parses and edits an LF file."""
        file_path = project_path / "src.py"
        file_path.write_text("def foo():\n    return 1\n\ndef bar():\n    return 2\n")
        crlf_markers_lf_content = (
            "<<<<<<< SEARCH\r\ndef foo():\n    return 1\n=======\r\ndef foo():\n    return 99\n>>>>>>> REPLACE"
        )

        result = await edit_tool.edit("src.py", crlf_markers_lf_content)

        assert result == "Edited src.py"
        assert file_path.read_text() == "def foo():\n    return 99\n\ndef bar():\n    return 2\n"

    # --- line-ending handling: preservation ---

    async def test_edit_preserves_crlf_line_endings(self, edit_tool, project_path):
        """Editing a CRLF file preserves CRLF on unchanged and changed lines."""
        file_path = project_path / "src.py"
        file_path.write_bytes(b"def foo():\r\n    return 1\r\n\r\ndef bar():\r\n    return 2\r\n")

        result = await edit_tool.edit("src.py", block("def foo():\n    return 1", "def foo():\n    return 99"))

        assert result == "Edited src.py"
        after = file_path.read_bytes()
        assert b"def foo():\r\n    return 99\r\n" in after
        assert b"def bar():\r\n    return 2\r\n" in after
        assert b"\n" not in after.replace(b"\r\n", b"")

    async def test_edit_preserves_lf_line_endings(self, edit_tool, project_path):
        """Editing an LF file introduces no carriage returns."""
        file_path = project_path / "lf.py"
        file_path.write_bytes(b"def foo():\n    return 1\n\ndef bar():\n    return 2\n")

        result = await edit_tool.edit("lf.py", block("def foo():\n    return 1", "def foo():\n    return 99"))

        assert result == "Edited lf.py"
        after = file_path.read_bytes()
        assert b"\r" not in after
        assert after == b"def foo():\n    return 99\n\ndef bar():\n    return 2\n"

    async def test_edit_normalizes_mixed_line_endings_to_dominant(self, edit_tool, project_path):
        """A file with mixed endings is normalized to the dominant ending after edit."""
        file_path = project_path / "mix.py"
        # two CRLF lines, one bare-LF line -> dominant is CRLF
        file_path.write_bytes(b"line one\r\nline two\nline three\r\n")

        result = await edit_tool.edit("mix.py", block("line one", "LINE ONE"))

        assert result == "Edited mix.py"
        after = file_path.read_bytes()
        assert after == b"LINE ONE\r\nline two\r\nline three\r\n"
        assert b"\n" not in after.replace(b"\r\n", b"")

    async def test_edit_crlf_content_no_mixed_endings(
        self, project_path, mock_connection_manager, agent_config, mock_base_agent
    ):
        """A CRLF-preserving (sandbox-like) filesystem edit yields no mixed endings."""
        filesystem = MemoryFileSystem("def foo():\r\n    return 1\r\n\r\ndef bar():\r\n    return 2\r\n")
        tool = EditTool(
            project_path,
            "test_workspace",
            str(uuid.uuid4()),
            mock_connection_manager,
            agent_config,
            mock_base_agent,
            filesystem,
        )

        result = await tool.edit("crlf.txt", block("def foo():\n    return 1", "def foo():\n    return 99"))

        assert result == "Edited crlf.txt"
        assert "\n" not in filesystem.content.replace("\r\n", "")
        assert filesystem.content == "def foo():\r\n    return 99\r\n\r\ndef bar():\r\n    return 2\r\n"

    async def test_edit_no_change_does_not_write_crlf_file(self, edit_tool, project_path):
        """A no-op edit on a CRLF file leaves disk bytes unchanged (no write, no LE flip)."""
        file_path = project_path / "nc.py"
        original = b"foo\r\nbar\r\n"
        file_path.write_bytes(original)

        result = await edit_tool.edit("nc.py", block("foo", "foo"))

        assert "No changes made" in result
        assert file_path.read_bytes() == original


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

    async def test_write_overwrite_preserves_dominant_line_endings(self, edit_tool, project_path):
        """write overwrite adopts the file's dominant line ending; new files stay LF."""
        crlf_file = project_path / "crlf.py"
        crlf_file.write_bytes(b"old\r\ncontent\r\n")

        result = await edit_tool.write("crlf.py", "new\ncontent\n")

        assert result == "Wrote crlf.py"
        after = crlf_file.read_bytes()
        assert after == b"new\r\ncontent\r\n"
        assert b"\n" not in after.replace(b"\r\n", b"")

        # a brand-new file keeps the LF content the caller provided
        result = await edit_tool.write("new.py", "brand\nnew\n")
        assert result == "Wrote new.py"
        assert (project_path / "new.py").read_bytes() == b"brand\nnew\n"

    async def test_write_permission_error(self, edit_tool, project_path):
        with patch("pathlib.Path.write_text", side_effect=PermissionError("Permission denied")):
            with pytest.raises(PermissionError) as exc_info:
                await edit_tool.write("test.txt", "Hello World")

        assert "Permission denied" in str(exc_info.value)
        assert not (project_path / "test.txt").exists()


@pytest.mark.asyncio
async def test_apply_patch_external_paths_use_sandbox_root_semantics(
    project_path: Path,
    mock_connection_manager: AsyncMock,
    agent_config: AgentConfig,
    mock_base_agent: Mock,
) -> None:
    sandbox = Mock()

    def run(command: str) -> Mock:
        result = Mock()
        result.exit_code = 1 if command.startswith("test -") else 0
        result.stdout = ""
        result.stderr = ""
        return result

    sandbox.commands.run.side_effect = run
    sandbox.files.write = Mock()
    filesystem = SandboxFileSystem(sandbox, "/sandbox/project")
    tool = EditTool(
        project_path,
        "test_workspace",
        str(uuid.uuid4()),
        mock_connection_manager,
        agent_config,
        mock_base_agent,
        filesystem,
    )
    patch_text = (
        "*** Begin Patch\n"
        "*** Add File: /external/absolute.txt\n"
        "+absolute\n"
        "*** Add File: ../../../outside/repeated-parent.txt\n"
        "+relative\n"
        "*** End Patch\n"
    )

    result = await tool.apply_patch(patch_text)

    assert "A /external/absolute.txt" in result
    assert "A ../../../outside/repeated-parent.txt" in result
    sandbox.files.write.assert_any_call("/external/absolute.txt", "absolute\n")
    sandbox.files.write.assert_any_call(
        "/sandbox/project/../../../outside/repeated-parent.txt",
        "relative\n",
    )


def _make_mock_lsp_manager(*, auto_diagnostics_on_edit: bool, diagnostics=None) -> Mock:
    """Build a mock LspManager with the attributes _maybe_append_lsp_diagnostics() inspects."""
    manager = Mock()
    manager.enabled = True
    manager._config = Mock()
    manager._config.auto_diagnostics_on_edit = auto_diagnostics_on_edit
    manager._initialized = True
    manager.server_for_path = Mock(return_value="pyright")
    manager.get_fresh_diagnostics = AsyncMock(return_value=diagnostics or [])
    return manager


@pytest.mark.asyncio
class TestEditToolLspDiagnostics:
    """Tests for LSP diagnostics behavior controlled by auto_diagnostics_on_edit."""

    @staticmethod
    def _make_edit_tool(project_path, mock_connection_manager, agent_config, mock_base_agent, lsp_manager):
        return EditTool(
            project_path,
            "test_workspace",
            str(uuid.uuid4()),
            mock_connection_manager,
            agent_config,
            mock_base_agent,
            lsp_manager=lsp_manager,
        )

    async def test_edit_skips_diagnostics_when_auto_diagnostics_disabled(
        self, project_path, mock_connection_manager, agent_config, mock_base_agent
    ):
        """When auto_diagnostics_on_edit=False, edit() does NOT append LSP diagnostics."""
        lsp_manager = _make_mock_lsp_manager(auto_diagnostics_on_edit=False)
        tool = self._make_edit_tool(project_path, mock_connection_manager, agent_config, mock_base_agent, lsp_manager)

        file_path = project_path / "test.py"
        file_path.write_text("x = 1\n")

        result = await tool.edit("test.py", block("x = 1", "x = 2"))

        assert result == "Edited test.py"
        assert "LSP diagnostics" not in result
        # The diagnostics path is short-circuited before any LSP queries happen.
        lsp_manager.get_fresh_diagnostics.assert_not_called()

    async def test_edit_queries_diagnostics_when_auto_diagnostics_enabled(
        self, project_path, mock_connection_manager, agent_config, mock_base_agent
    ):
        """When auto_diagnostics_on_edit=True (default), edit() queries the LSP manager."""
        lsp_manager = _make_mock_lsp_manager(auto_diagnostics_on_edit=True)
        tool = self._make_edit_tool(project_path, mock_connection_manager, agent_config, mock_base_agent, lsp_manager)

        file_path = project_path / "test.py"
        file_path.write_text("x = 1\n")

        result = await tool.edit("test.py", block("x = 1", "x = 2"))

        assert result == "Edited test.py"
        # Mock returns empty diagnostics, so no diagnostics block is appended.
        assert "LSP diagnostics" not in result
        # But the manager WAS queried for fresh diagnostics.
        lsp_manager.get_fresh_diagnostics.assert_awaited_once_with("test.py")
