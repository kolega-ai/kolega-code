import asyncio
from unittest.mock import AsyncMock, Mock

import pytest
import uuid

from kolega_code.config import AgentConfig, ModelConfig, ModelProvider, RateLimitConfig
from kolega_code.agent.tool_backend.glob_tool import GlobSearchResult, GlobTool
from kolega_code.services.workspace_scan import ScanOutcome, ScannedPath


@pytest.fixture
def mock_connection_manager():
    mock = AsyncMock()
    return mock


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
    return mock


@pytest.fixture
def glob_tool(project_path, mock_connection_manager, agent_config, mock_base_agent):
    return GlobTool(
        project_path, "test_workspace", str(uuid.uuid4()), mock_connection_manager, agent_config, mock_base_agent
    )


@pytest.fixture
def sample_files(project_path):
    # Create a directory structure with various files
    (project_path / "src").mkdir()
    (project_path / "tests").mkdir()
    (project_path / "docs").mkdir()

    # Create Python files
    (project_path / "src" / "main.py").write_text('def main():\n    print("Hello World")\n')
    (project_path / "src" / "utils.py").write_text('def helper():\n    return "Helper function"\n')
    (project_path / "tests" / "test_main.py").write_text("def test_main():\n    assert True\n")

    # Create other file types
    (project_path / "docs" / "README.md").write_text("# Project Documentation")
    (project_path / "docs" / "CHANGELOG.md").write_text("# Changelog")

    # Create a binary file
    (project_path / "src" / "data.bin").write_bytes(b"\x00\x01\x02\x03")

    # Create a file in excluded directory
    (project_path / ".git").mkdir()
    (project_path / ".git" / "config").write_text("git config content")

    # Create a large file
    (project_path / "src" / "large.txt").write_text("x" * (11 * 1024 * 1024))

    return project_path


class TestGlobTool:
    @pytest.mark.asyncio
    async def test_find_files_by_pattern_basic(self, glob_tool, sample_files):
        result = await glob_tool.find_files_by_pattern("**/*.py")

        assert "# Files Matching '**/*.py'" in result
        assert "Found 3 matching items" in result
        assert "**main.py**" in result
        assert "**utils.py**" in result
        assert "**test_main.py**" in result

    @pytest.mark.asyncio
    async def test_find_files_by_pattern_recursive(self, glob_tool, sample_files):
        result = await glob_tool.find_files_by_pattern("**/*.md")

        assert "# Files Matching '**/*.md'" in result
        assert "Found 2 matching items" in result
        assert "**README.md**" in result
        assert "**CHANGELOG.md**" in result

    @pytest.mark.asyncio
    async def test_find_files_by_pattern_no_matches(self, glob_tool, sample_files):
        result = await glob_tool.find_files_by_pattern("*.nonexistent")

        assert "No files found matching pattern: '*.nonexistent'" in result

    @pytest.mark.asyncio
    async def test_find_files_by_pattern_include_directories(self, glob_tool, sample_files):
        result = await glob_tool.find_files_by_pattern("*", include_directories=True)

        assert "📁 Directory" in result
        assert "**src**" in result
        assert "**tests**" in result
        assert "**docs**" in result

    @pytest.mark.asyncio
    async def test_find_files_by_pattern_exclude_directories(self, glob_tool, sample_files):
        result = await glob_tool.find_files_by_pattern("**/*", include_directories=False)

        assert "📁 Directory" not in result
        assert "**main.py**" in result
        assert "**utils.py**" in result
        assert "**test_main.py**" in result

    @pytest.mark.asyncio
    async def test_find_files_by_pattern_without_details(self, glob_tool, sample_files):
        result = await glob_tool.find_files_by_pattern("**/*.py", show_details=False)

        assert "**main.py**" in result
        assert "**utils.py**" in result
        assert "**test_main.py**" in result
        assert "Size:" not in result
        assert "Modified:" not in result

    @pytest.mark.asyncio
    async def test_find_files_by_pattern_excludes_binary_files(self, glob_tool, sample_files):
        result = await glob_tool.find_files_by_pattern("**/*")

        assert "**data.bin**" not in result

    @pytest.mark.asyncio
    async def test_find_files_by_pattern_excludes_git_files(self, glob_tool, sample_files):
        result = await glob_tool.find_files_by_pattern("**/*")

        assert ".git/config" not in result

    @pytest.mark.asyncio
    async def test_find_files_by_pattern_excludes_large_files(self, glob_tool, sample_files):
        result = await glob_tool.find_files_by_pattern("**/*")

        assert "**large.txt**" not in result

    @pytest.mark.asyncio
    async def test_find_files_by_pattern_result_limit(self, glob_tool, sample_files):
        # Create many files
        for i in range(150):
            (sample_files / f"file_{i}.txt").write_text("test")

        result = await glob_tool.find_files_by_pattern("*.txt")

        assert "Found at least 129 matching items" in result
        assert "showing first 128" in result

    @pytest.mark.asyncio
    async def test_find_files_by_pattern_grouped_by_directory(self, glob_tool, sample_files):
        result = await glob_tool.find_files_by_pattern("**/*.py")

        assert "# Files Matching '**/*.py'" in result
        assert "## src/" in result
        assert "## tests/" in result
        assert "**main.py**" in result
        assert "**utils.py**" in result
        assert "**test_main.py**" in result

    @pytest.mark.asyncio
    async def test_find_files_by_pattern_in_directory(self, glob_tool, sample_files):
        result = await glob_tool.find_files_by_pattern("src/*.py")

        assert "**main.py**" in result
        assert "**utils.py**" in result
        assert "**test_main.py**" not in result

    @pytest.mark.asyncio
    async def test_find_files_by_pattern_with_special_characters(self, glob_tool, sample_files):
        # Create a file with special characters
        special_file = sample_files / "src" / "special@#$%.txt"
        special_file.write_text("test")

        result = await glob_tool.find_files_by_pattern("**/*.txt")

        assert "**special@#$%.txt**" in result

    @pytest.mark.asyncio
    async def test_find_files_by_pattern_leading_slash(self, glob_tool, sample_files):
        result = await glob_tool.find_files_by_pattern("/src/*.py")

        assert "**main.py**" in result
        assert "**utils.py**" in result

    @pytest.mark.asyncio
    async def test_local_search_does_not_use_unbounded_filesystem_glob(self, glob_tool, sample_files, monkeypatch):
        monkeypatch.setattr(
            glob_tool.filesystem,
            "glob",
            Mock(side_effect=AssertionError("unbounded glob should not be called")),
        )

        result = await glob_tool.find_files_by_pattern("**/*.py")

        assert "**main.py**" in result

    @pytest.mark.asyncio
    async def test_incomplete_empty_search_does_not_claim_no_files(self, glob_tool, monkeypatch):
        async def incomplete(*args, **kwargs):
            return GlobSearchResult([], 0, False, "deadline", visited_entries=100, elapsed_seconds=5.0)

        monkeypatch.setattr(glob_tool, "_search_files_local", incomplete)
        result = await glob_tool.find_files_by_pattern("**/missing.txt")

        assert "No files found" not in result
        assert "Incomplete search" in result
        assert "deadline" in result

    @pytest.mark.asyncio
    async def test_no_find_falls_back_to_off_thread_scanner(self, glob_tool, monkeypatch):
        async def fallback_scan(*args, **kwargs):
            return ScanOutcome(paths=[ScannedPath("main.py", False, 4, 1)], visited_entries=1)

        monkeypatch.setattr("kolega_code.agent.tool_backend.glob_tool.shutil.which", lambda name: None)
        monkeypatch.setattr("kolega_code.agent.tool_backend.glob_tool.scan_workspace", fallback_scan)

        result = await glob_tool.find_files_by_pattern("**/*.py")

        assert "**main.py**" in result

    @pytest.mark.asyncio
    async def test_cancelling_native_find_terminates_child(self, glob_tool, monkeypatch):
        class BlockingStdout:
            async def readuntil(self, separator):
                await asyncio.Event().wait()

        class BlockingProcess:
            def __init__(self):
                self.stdout = BlockingStdout()
                self.returncode = None
                self.terminated = False

            def terminate(self):
                self.terminated = True
                self.returncode = -15

            def kill(self):
                self.returncode = -9

            async def wait(self):
                return self.returncode

        proc = BlockingProcess()

        async def create_process(*args, **kwargs):
            return proc

        monkeypatch.setattr(
            "kolega_code.agent.tool_backend.glob_tool.asyncio.create_subprocess_exec",
            create_process,
        )
        task = asyncio.create_task(glob_tool._search_files_local_process("**/*.py", True, 128))
        await asyncio.sleep(0)
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task
        assert proc.terminated is True
