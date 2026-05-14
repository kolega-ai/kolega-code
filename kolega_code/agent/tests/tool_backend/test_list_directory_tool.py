"""Unit tests for the ListDirectoryTool with local filesystem."""

import pytest
import pytest_asyncio
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, AsyncMock
from datetime import datetime

from kolega_code.agent.services.file_system import LocalFileSystem
from kolega_code.agent.tool_backend.list_directory_tool import ListDirectoryTool
from kolega_code.agent.baseagent import BaseAgent


class TestListDirectoryTool:
    """Unit tests for list directory tool with local filesystem."""

    @pytest_asyncio.fixture
    async def test_directory(self):
        """Create a temporary directory with test files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test structure
            os.makedirs(os.path.join(tmpdir, ".git", "objects"))
            os.makedirs(os.path.join(tmpdir, "src", "components"))
            os.makedirs(os.path.join(tmpdir, "docs"))
            os.makedirs(os.path.join(tmpdir, "tests"))

            # Create files
            Path(os.path.join(tmpdir, "main.py")).write_text('print("Hello")')
            Path(os.path.join(tmpdir, "README.md")).write_text("# Project")
            Path(os.path.join(tmpdir, "package.json")).write_text('{"name": "test"}')
            Path(os.path.join(tmpdir, "Dockerfile")).write_text("FROM python:3.9")
            Path(os.path.join(tmpdir, ".gitignore")).write_text("*.pyc\n__pycache__/")
            Path(os.path.join(tmpdir, "requirements.txt")).write_text("pytest==7.0.0")

            # Create files in subdirectories
            Path(os.path.join(tmpdir, ".git", "config")).write_text("[core]\n")
            Path(os.path.join(tmpdir, "src", "app.py")).write_text("def main(): pass")
            Path(os.path.join(tmpdir, "src", "components", "button.py")).write_text("class Button: pass")
            Path(os.path.join(tmpdir, "docs", "api.md")).write_text("# API")
            Path(os.path.join(tmpdir, "tests", "test_main.py")).write_text("def test_main(): pass")

            # Create a larger file
            with open(os.path.join(tmpdir, "large.dat"), "wb") as f:
                f.write(b"0" * (1024 * 1024))  # 1MB file

            yield tmpdir

    @pytest.fixture
    def mock_agent(self):
        """Create a mock agent for the tool."""
        agent = Mock(spec=BaseAgent)
        agent.agent_name = "test-agent"
        agent.log_info = AsyncMock()
        agent.log_error = AsyncMock()
        return agent

    @pytest.mark.asyncio
    async def test_list_root_excludes_git(self, test_directory, mock_agent):
        """Test that .git directory is excluded from root listing."""
        # Create filesystem and tool
        filesystem = LocalFileSystem(root_path=test_directory)
        tool = ListDirectoryTool(
            project_path=test_directory,
            workspace_id="test",
            thread_id="test",
            connection_manager=Mock(),
            config=Mock(),
            caller=mock_agent,
            filesystem=filesystem,
        )

        # List root directory
        result = await tool.list_directory("")

        # Verify .git is not in the output
        assert "| 📁 | .git" not in result

        # Verify other directories are present
        assert "| 📁 | src" in result
        assert "| 📁 | docs" in result
        assert "| 📁 | tests" in result

        # Verify files are present
        assert "| 📄 | main.py" in result
        assert "| 📄 | README.md" in result
        assert "| 📄 | .gitignore" in result

    @pytest.mark.asyncio
    async def test_file_descriptions(self, test_directory, mock_agent):
        """Test that file descriptions are correct."""
        filesystem = LocalFileSystem(root_path=test_directory)
        tool = ListDirectoryTool(
            project_path=test_directory,
            workspace_id="test",
            thread_id="test",
            connection_manager=Mock(),
            config=Mock(),
            caller=mock_agent,
            filesystem=filesystem,
        )

        result = await tool.list_directory("")

        # Check file descriptions
        assert "Python Source |" in result  # main.py
        assert "Project Documentation |" in result  # README.md
        assert "Node.js Package |" in result  # package.json
        assert "Docker Definition |" in result  # Dockerfile
        assert "Git Ignore Rules |" in result  # .gitignore
        assert "Python Dependencies |" in result  # requirements.txt
        assert "DAT File |" in result  # large.dat

    @pytest.mark.asyncio
    async def test_file_sizes(self, test_directory, mock_agent):
        """Test that file sizes are displayed correctly."""
        filesystem = LocalFileSystem(root_path=test_directory)
        tool = ListDirectoryTool(
            project_path=test_directory,
            workspace_id="test",
            thread_id="test",
            connection_manager=Mock(),
            config=Mock(),
            caller=mock_agent,
            filesystem=filesystem,
        )

        result = await tool.list_directory("")

        # The large.dat file should show as 1.0 MB
        lines = result.split("\n")
        large_file_line = [line for line in lines if "large.dat" in line][0]
        assert "1.0 MB" in large_file_line

        # Small files should show in bytes
        main_py_line = [line for line in lines if "main.py" in line][0]
        assert " B |" in main_py_line

    @pytest.mark.asyncio
    async def test_directory_item_counts(self, test_directory, mock_agent):
        """Test that directory item counts are correct."""
        filesystem = LocalFileSystem(root_path=test_directory)
        tool = ListDirectoryTool(
            project_path=test_directory,
            workspace_id="test",
            thread_id="test",
            connection_manager=Mock(),
            config=Mock(),
            caller=mock_agent,
            filesystem=filesystem,
        )

        result = await tool.list_directory("")

        # Check directory item counts
        lines = result.split("\n")
        src_line = [line for line in lines if "| 📁 | src" in line][0]
        assert "2 items" in src_line  # app.py and components/

        docs_line = [line for line in lines if "| 📁 | docs" in line][0]
        assert "1 items" in docs_line  # api.md

        tests_line = [line for line in lines if "| 📁 | tests" in line][0]
        assert "1 items" in tests_line  # test_main.py

    @pytest.mark.asyncio
    async def test_subdirectory_listing(self, test_directory, mock_agent):
        """Test listing a subdirectory."""
        filesystem = LocalFileSystem(root_path=test_directory)
        tool = ListDirectoryTool(
            project_path=test_directory,
            workspace_id="test",
            thread_id="test",
            connection_manager=Mock(),
            config=Mock(),
            caller=mock_agent,
            filesystem=filesystem,
        )

        result = await tool.list_directory("src")

        # Verify header and navigation
        assert "# Directory: src" in result
        assert "📁 Root Directory" in result

        # Verify contents
        assert "| 📄 | app.py" in result
        assert "| 📁 | components" in result
        assert "Python Source |" in result

    @pytest.mark.asyncio
    async def test_nested_directory(self, test_directory, mock_agent):
        """Test listing a nested directory."""
        filesystem = LocalFileSystem(root_path=test_directory)
        tool = ListDirectoryTool(
            project_path=test_directory,
            workspace_id="test",
            thread_id="test",
            connection_manager=Mock(),
            config=Mock(),
            caller=mock_agent,
            filesystem=filesystem,
        )

        result = await tool.list_directory("src/components")

        # Verify header and navigation
        assert "# Directory: src/components" in result
        assert "📁 Parent Directory: src" in result

        # Verify contents
        assert "| 📄 | button.py" in result
        assert "Python Source |" in result

    @pytest.mark.asyncio
    async def test_nonexistent_directory(self, test_directory, mock_agent):
        """Test error handling for non-existent directory."""
        filesystem = LocalFileSystem(root_path=test_directory)
        tool = ListDirectoryTool(
            project_path=test_directory,
            workspace_id="test",
            thread_id="test",
            connection_manager=Mock(),
            config=Mock(),
            caller=mock_agent,
            filesystem=filesystem,
        )

        with pytest.raises(FileNotFoundError) as exc_info:
            await tool.list_directory("nonexistent")

        assert "Directory not found: nonexistent" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_list_file_instead_of_directory(self, test_directory, mock_agent):
        """Test error when trying to list a file."""
        filesystem = LocalFileSystem(root_path=test_directory)
        tool = ListDirectoryTool(
            project_path=test_directory,
            workspace_id="test",
            thread_id="test",
            connection_manager=Mock(),
            config=Mock(),
            caller=mock_agent,
            filesystem=filesystem,
        )

        with pytest.raises(NotADirectoryError) as exc_info:
            await tool.list_directory("main.py")

        assert "Not a directory: main.py" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_summary_counts(self, test_directory, mock_agent):
        """Test that summary counts are correct."""
        filesystem = LocalFileSystem(root_path=test_directory)
        tool = ListDirectoryTool(
            project_path=test_directory,
            workspace_id="test",
            thread_id="test",
            connection_manager=Mock(),
            config=Mock(),
            caller=mock_agent,
            filesystem=filesystem,
        )

        result = await tool.list_directory("")

        # Verify summary - should be 3 directories (not counting .git) and 7 files
        assert "**Summary:** 3 directories, 7 files" in result

    @pytest.mark.asyncio
    async def test_dates_shown(self, test_directory, mock_agent):
        """Test that modification dates are shown."""
        filesystem = LocalFileSystem(root_path=test_directory)
        tool = ListDirectoryTool(
            project_path=test_directory,
            workspace_id="test",
            thread_id="test",
            connection_manager=Mock(),
            config=Mock(),
            caller=mock_agent,
            filesystem=filesystem,
        )

        result = await tool.list_directory("")

        # Dates should be in YYYY-MM-DD HH:MM format
        current_year = datetime.now().year
        assert f"{current_year}-" in result
        assert "Unknown" not in result  # No unknown dates
