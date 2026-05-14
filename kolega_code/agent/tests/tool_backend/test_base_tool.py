from pathlib import Path
from unittest.mock import Mock, patch
import uuid

import pytest

from kolega_code.agent.config import AgentConfig, ModelConfig, ModelProvider, RateLimitConfig
from kolega_code.agent.tool_backend.base_tool import BaseTool


@pytest.fixture
def mock_connection_manager():
    return Mock()


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
    return Mock()


@pytest.fixture
def base_tool(project_path, mock_connection_manager, agent_config, mock_base_agent):
    return BaseTool(
        project_path, "test_workspace", str(uuid.uuid4()), mock_connection_manager, agent_config, mock_base_agent
    )


class TestBaseTool:
    def test_initialization(self, base_tool, project_path):
        assert base_tool.workspace_id == "test_workspace"
        assert base_tool.project_path == project_path
        assert base_tool.connection_manager is not None
        assert base_tool.caller is not None

    def test_initialization_with_string_path(self, mock_connection_manager, agent_config, mock_base_agent):
        tool = BaseTool(
            "/test/path", "test_workspace", str(uuid.uuid4()), mock_connection_manager, agent_config, mock_base_agent
        )
        assert isinstance(tool.project_path, Path)
        assert str(tool.project_path) == "/test/path"
        assert tool.caller is not None

    @pytest.mark.parametrize(
        "extension,expected",
        [
            (".pyc", True),
            (".jpg", True),
            (".txt", False),
            (".py", False),
            (".md", False),
        ],
    )
    def test_is_binary_file_by_extension(self, base_tool, tmp_path, extension, expected):
        test_file = tmp_path / f"test{extension}"
        test_file.touch()
        assert base_tool._is_binary_file(test_file) == expected

    def test_is_binary_file_by_content(self, base_tool, tmp_path):
        # Create a binary file
        binary_file = tmp_path / "test.bin"
        with open(binary_file, "wb") as f:
            f.write(b"\x00\x01\x02\x03")
        assert base_tool._is_binary_file(binary_file) is True

        # Create a text file
        text_file = tmp_path / "test.txt"
        with open(text_file, "w") as f:
            f.write("Hello, World!")
        assert base_tool._is_binary_file(text_file) is False

    @pytest.mark.parametrize(
        "directory,expected",
        [
            (".git", True),
            ("__pycache__", True),
            ("src", False),
            ("tests", False),
        ],
    )
    def test_should_exclude_file_by_directory(self, base_tool, tmp_path, directory, expected):
        test_file = tmp_path / directory / "test.txt"
        test_file.parent.mkdir(exist_ok=True)
        test_file.touch()
        assert base_tool._should_exclude_file(test_file) == expected

    def test_should_exclude_large_file(self, base_tool, tmp_path):
        test_file = tmp_path / "large.txt"
        with open(test_file, "w") as f:
            f.write("x" * (11 * 1024 * 1024))  # 11MB file
        assert base_tool._should_exclude_file(test_file) is True

    @patch("pathspec.PathSpec.from_lines")
    def test_is_gitignored(self, mock_pathspec, base_tool, tmp_path):
        # Setup mock gitignore
        gitignore_path = tmp_path / ".gitignore"
        gitignore_path.write_text("*.pyc\n__pycache__\n")

        # Create a mock pathspec that matches certain patterns
        mock_spec = Mock()
        mock_spec.match_file.side_effect = lambda path: path.endswith(".pyc")
        mock_pathspec.return_value = mock_spec

        # Test ignored file
        ignored_file = tmp_path / "test.pyc"
        ignored_file.touch()
        assert base_tool._is_gitignored(ignored_file) is True

        # Test non-ignored file
        non_ignored_file = tmp_path / "test.py"
        non_ignored_file.touch()
        assert base_tool._is_gitignored(non_ignored_file) is False

    def test_load_gitignore_patterns_no_file(self, base_tool, tmp_path):
        base_tool._load_gitignore_patterns()
        assert base_tool._gitignore_spec is None

    @patch("pathspec.PathSpec.from_lines")
    def test_load_gitignore_patterns_with_file(self, mock_pathspec, base_tool, tmp_path):
        gitignore_path = tmp_path / ".gitignore"
        gitignore_path.write_text("*.pyc\n__pycache__\n")

        mock_spec = Mock()
        mock_pathspec.return_value = mock_spec

        base_tool._load_gitignore_patterns()
        assert base_tool._gitignore_spec is not None
        mock_pathspec.assert_called_once()
