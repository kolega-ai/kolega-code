from pathlib import Path
from typing import Any, List

import pytest
import uuid
from unittest.mock import AsyncMock, Mock

from kolega_code.config import AgentConfig, ModelConfig, ModelProvider, RateLimitConfig
from kolega_code.agent.tool_backend.glob_tool import GlobTool
from kolega_code.services.file_system import LocalFileSystem
from kolega_code.sandbox.filesystem import SandboxFileSystem


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
def mock_connection_manager():
    return AsyncMock()


@pytest.fixture
def sample_project(tmp_path: Path):
    # Directories
    (tmp_path / "src").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "docs").mkdir()

    # Files
    (tmp_path / "src" / "main.py").write_text("print('Hello World')\n")
    (tmp_path / "src" / "utils.py").write_text("def helper():\n    return 1\n")
    (tmp_path / "tests" / "test_main.py").write_text("def test_main():\n    assert True\n")
    (tmp_path / "docs" / "README.md").write_text("# Readme")

    # Excluded dir content
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text("git")

    # Binary and large files
    (tmp_path / "src" / "data.bin").write_bytes(b"\x00\x01\x02")
    (tmp_path / "src" / "large.txt").write_text("x" * (11 * 1024 * 1024))

    return tmp_path


def _parse_embedded_var(script: str, var_name: str, default: str = "") -> str:
    prefix = f"{var_name}="
    for line in script.splitlines():
        if line.strip().startswith(prefix):
            return line.split("=", 1)[1].strip().strip("'\"")
    return default


def _glob_from_root(root: Path, pattern: str) -> List[str]:
    # Use Python glob relative to root, return relative paths
    matches = list(root.glob(pattern)) if "**" not in pattern else list(root.rglob(pattern.split("**/", 1)[1]))
    # For generality, fall back to glob.glob with recursive
    if not matches:
        matches = list(root.glob(pattern))
    rels: List[str] = []
    for p in matches:
        try:
            rel = str(p.relative_to(root))
        except Exception:
            rel = str(p)
        rels.append(rel)
    # Deduplicate and sort
    return sorted(dict.fromkeys(rels).keys())


def _make_mock_sandbox_for_glob(tmp_path: Path) -> Any:
    sandbox = Mock()
    sandbox.files = Mock()
    sandbox.commands = Mock()

    exclude_dirs = GlobTool.EXCLUDE_DIRS
    bin_exts = GlobTool.BINARY_EXTENSIONS

    def run_side_effect(cmd: str, **kwargs):
        result = Mock()
        result.exit_code = 0
        result.stderr = ""
        result.stdout = ""

        if "bash -O globstar" in cmd:
            pattern = _parse_embedded_var(cmd, "pattern", "")
            include_dirs_flag = _parse_embedded_var(cmd, "include_dirs", "0")
            include_dirs = include_dirs_flag == "1"

            # Compute matches using Python glob
            matches = _glob_from_root(tmp_path, pattern)

            # Apply excludes and build rows
            rows: List[str] = []
            total = 0
            for rel in matches:
                parts = Path(rel).parts
                if any(part in exclude_dirs for part in parts):
                    continue

                full = tmp_path / rel
                if full.is_file():
                    # exclude by extension and size
                    if full.suffix.lower() in bin_exts:
                        continue
                    try:
                        size = full.stat().st_size
                        if size > GlobTool.MAX_FILE_SIZE_BYTES:
                            continue
                        mtime = int(full.stat().st_mtime)
                    except Exception:
                        continue
                    total += 1
                    if len(rows) < GlobTool.MAX_RESULTS:
                        rows.append(f"{rel}\tf\t{size}\t{mtime}")
                elif full.is_dir() and include_dirs:
                    try:
                        mtime = int(full.stat().st_mtime)
                    except Exception:
                        mtime = 0
                    total += 1
                    if len(rows) < GlobTool.MAX_RESULTS:
                        rows.append(f"{rel}\td\t0\t{mtime}")

            # Compose stdout
            out = "\n".join(rows) + f"\n__TOTAL__ {total}\n"
            result.stdout = out

        return result

    from unittest.mock import AsyncMock as _AsyncMock

    sandbox.commands.run = _AsyncMock(side_effect=run_side_effect)
    sandbox.files.read = lambda p: (tmp_path / p).read_text() if (tmp_path / p).exists() else ""

    return sandbox


@pytest.mark.asyncio
async def test_sandbox_fast_path_detection(sample_project, mock_connection_manager, agent_config, mock_base_agent):
    mock_sandbox = _make_mock_sandbox_for_glob(sample_project)
    sandbox_fs = SandboxFileSystem(mock_sandbox, str(sample_project))

    tool = GlobTool(
        sample_project,
        "test_workspace",
        str(uuid.uuid4()),
        mock_connection_manager,
        agent_config,
        mock_base_agent,
        filesystem=sandbox_fs,
    )

    result = await tool.find_files_by_pattern("**/*.py")
    # Ensure bash path invoked
    assert any("bash -O globstar" in str(call) for call in mock_sandbox.commands.run.call_args_list)
    assert "**main.py**" in result
    assert "**utils.py**" in result


@pytest.mark.asyncio
async def test_glob_tool_local_vs_sandbox_parity(
    sample_project, mock_connection_manager, agent_config, mock_base_agent
):
    # Local tool
    local_tool = GlobTool(
        sample_project,
        "test_workspace",
        str(uuid.uuid4()),
        mock_connection_manager,
        agent_config,
        mock_base_agent,
        filesystem=LocalFileSystem(root_path=sample_project),
    )

    # Sandbox tool with mock sandbox
    mock_sandbox = _make_mock_sandbox_for_glob(sample_project)
    sandbox_fs = SandboxFileSystem(mock_sandbox, str(sample_project))
    sandbox_tool = GlobTool(
        sample_project,
        "test_workspace",
        str(uuid.uuid4()),
        mock_connection_manager,
        agent_config,
        mock_base_agent,
        filesystem=sandbox_fs,
    )

    patterns = ["**/*.py", "src/*.py", "**/*.md", "*.md"]
    for pat in patterns:
        local_res = await local_tool.find_files_by_pattern(pat)
        sandbox_res = await sandbox_tool.find_files_by_pattern(pat)

        # Normalize outcome parity
        local_has = "# Files Matching" in local_res
        sandbox_has = "# Files Matching" in sandbox_res
        assert local_has == sandbox_has, (
            f"Mismatch in results presence for pattern {pat}:\nlocal={local_res}\nsandbox={sandbox_res}"
        )

        # Compare presence of key filenames
        for fname in ["main.py", "utils.py", "test_main.py", "README.md"]:
            assert (fname in local_res) == (fname in sandbox_res), f"Mismatch for {fname} with pattern {pat}"

    # Include directories
    local_res_dirs = await local_tool.find_files_by_pattern("*", include_directories=True)
    sandbox_res_dirs = await sandbox_tool.find_files_by_pattern("*", include_directories=True)
    for d in ["src", "tests", "docs"]:
        assert (d in local_res_dirs) == (d in sandbox_res_dirs), f"Directory presence mismatch for {d}"
