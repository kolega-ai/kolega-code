"""
Tests to ensure the optimized SearchCodebaseTool implementation
behaves identically to the original implementation.
"""

import asyncio
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import AsyncMock, Mock, patch

import pytest
import uuid

from kolega_code.agent.config import AgentConfig, ModelConfig, ModelProvider, RateLimitConfig
from kolega_code.agent.tool_backend.search_codebase_tool import SearchCodebaseTool
from kolega_code.agent.tool_backend.search_codebase_tool_original import SearchCodebaseToolOriginal
from kolega_code.agent.services.file_system import LocalFileSystem
from kolega_code.agent.services.sandbox.sandbox_filesystem import SandboxFileSystem


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
def mock_connection_manager():
    return AsyncMock()


@pytest.fixture
def complex_project_structure(tmp_path):
    """Create a complex project structure for thorough testing"""
    # Source code
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text(
        """
def main():
    print("Hello World")
    print("Starting application...")
    
class Application:
    def __init__(self):
        self.name = "MyApp"
        print(f"Initializing {self.name}")
    
    def run(self):
        print("Running application")
"""
    )

    (tmp_path / "src" / "utils.py").write_text(
        '''
import re

def parse_config(config_str):
    """Parse configuration string"""
    pattern = re.compile(r"(\\w+)=(\\w+)")
    matches = pattern.findall(config_str)
    return dict(matches)

def format_output(data):
    print(f"Formatted: {data}")
    return str(data)
'''
    )

    # Tests
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_main.py").write_text(
        """
def test_application():
    print("Testing application")
    assert True

def test_utils():
    print("Testing utilities")
    assert True
"""
    )

    # Documentation
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "README.md").write_text(
        """
# Project Documentation

This is a sample project for testing.
It contains print statements and various patterns.
"""
    )

    # Binary files
    (tmp_path / "assets").mkdir()
    (tmp_path / "assets" / "logo.png").write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR")
    (tmp_path / "assets" / "data.bin").write_bytes(b"\x00\x01\x02\x03\x04\x05")

    # Large file
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs" / "debug.log").write_text("DEBUG: " * 200000)  # ~1.2MB file

    # Files in excluded directories
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text("print('git config')")

    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "package").mkdir()
    (tmp_path / "node_modules" / "package" / "index.js").write_text("console.log('print from node_modules')")

    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "cache.pyc").write_bytes(b"\x00\x01print\x02\x03")

    # Edge cases
    (tmp_path / "edge_cases.py").write_text(
        """
# File with many matches on same line
print("test1"); print("test2"); print("test3")

# Unicode content
print("Hello 世界 🌍")

# Special regex characters
print("Special: [test] (pattern) {match}")
"""
    )

    # Deeply nested structure
    nested = tmp_path / "deep" / "nested" / "structure" / "code"
    nested.mkdir(parents=True)
    (nested / "module.py").write_text('print("Deeply nested")')

    # Create .gitignore file
    (tmp_path / ".gitignore").write_text(
        """
*.pyc
__pycache__/
node_modules/
.git/
*.log
"""
    )

    return tmp_path


@pytest.fixture
def simple_project(tmp_path):
    """Create a simple project for basic testing"""
    (tmp_path / "test.py").write_text('print("test")')
    (tmp_path / "main.py").write_text('def main():\n    return "hello"')
    return tmp_path


class TestSearchCodebaseToolParity:
    """Test that both implementations produce identical results"""

    async def _run_both_implementations(
        self,
        project_path: Path,
        pattern: str,
        file_pattern: str = "*",
        case_sensitive: bool = False,
        literal: bool = False,  # Default to False to match original tool behavior
        filesystem=None,
        mock_connection_manager=None,
        agent_config=None,
        mock_base_agent=None,
    ) -> Tuple[str, str, float, float]:
        """Run both implementations and return results with timing"""

        if not mock_connection_manager:
            mock_connection_manager = AsyncMock()

        if not agent_config:
            agent_config = AgentConfig(
                anthropic_api_key="test_key",
                openai_api_key="test-key",
                long_context_config=ModelConfig(
                    provider=ModelProvider.ANTHROPIC, model="test-model", rate_limits=RateLimitConfig()
                ),
                fast_config=ModelConfig(
                    provider=ModelProvider.ANTHROPIC, model="test-model", rate_limits=RateLimitConfig()
                ),
                thinking_config=ModelConfig(
                    provider=ModelProvider.ANTHROPIC,
                    model="test-model",
                    rate_limits=RateLimitConfig(),
                    thinking_tokens=1024,
                ),
            )

        if not mock_base_agent:
            mock_base_agent = Mock()
            mock_base_agent.agent_name = "test_agent"

        # Original implementation
        original_tool = SearchCodebaseToolOriginal(
            project_path,
            "test_workspace",
            str(uuid.uuid4()),
            mock_connection_manager,
            agent_config,
            mock_base_agent,
            filesystem=filesystem,
        )

        # Optimized implementation
        optimized_tool = SearchCodebaseTool(
            project_path,
            "test_workspace",
            str(uuid.uuid4()),
            mock_connection_manager,
            agent_config,
            mock_base_agent,
            filesystem=filesystem,
        )

        # Run original
        start_original = time.time()
        result_original = await original_tool.search_codebase(pattern, file_pattern, case_sensitive)
        time_original = time.time() - start_original

        # Run optimized
        start_optimized = time.time()
        result_optimized = await optimized_tool.search_codebase(pattern, file_pattern, case_sensitive, literal)
        time_optimized = time.time() - start_optimized

        return result_original, result_optimized, time_original, time_optimized

    def _normalize_results(self, result: str) -> Dict[str, Any]:
        """Parse and normalize results for comparison"""
        if "No matches found" in result:
            return {"no_matches": True}

        if "Error:" in result:
            return {"error": result}

        # Extract files and their match counts
        files_matches = {}
        lines = result.split("\n")

        current_file = None
        for line in lines:
            # File line pattern: - **filepath** (N matches)
            file_match = re.match(r"- \*\*(.+?)\*\* \((\d+) matches\)", line)
            if file_match:
                current_file = file_match.group(1).strip()
                files_matches[current_file] = {"match_count": int(file_match.group(2)), "lines": []}
            # Line match pattern: Line N: content
            elif current_file and line.strip().startswith("Line "):
                line_match = re.match(r"\s*Line (\d+): (.+)", line)
                if line_match:
                    files_matches[current_file]["lines"].append(
                        {"line_num": int(line_match.group(1)), "content": line_match.group(2).strip()}
                    )

        return {"no_matches": False, "files": files_matches, "reached_limit": "⚠️ **Note:**" in result}

    @pytest.mark.asyncio
    async def test_basic_search_parity(
        self, complex_project_structure, mock_connection_manager, agent_config, mock_base_agent
    ):
        """Test basic search functionality produces identical results"""
        patterns = [
            "print",
            "def",
            "class",
            "import",
            "assert",
            "return",
        ]

        for pattern in patterns:
            orig, opt, _, _ = await self._run_both_implementations(
                complex_project_structure,
                pattern,
                mock_connection_manager=mock_connection_manager,
                agent_config=agent_config,
                mock_base_agent=mock_base_agent,
            )

            orig_parsed = self._normalize_results(orig)
            opt_parsed = self._normalize_results(opt)

            assert (
                orig_parsed == opt_parsed
            ), f"Results differ for pattern '{pattern}'\nOriginal: {orig_parsed}\nOptimized: {opt_parsed}"

    @pytest.mark.asyncio
    async def test_case_sensitivity_parity(
        self, complex_project_structure, mock_connection_manager, agent_config, mock_base_agent
    ):
        """Test case sensitivity produces identical results"""
        test_cases = [
            ("PRINT", False),  # Case insensitive
            ("PRINT", True),  # Case sensitive
            ("Print", False),
            ("Print", True),
        ]

        for pattern, case_sensitive in test_cases:
            orig, opt, _, _ = await self._run_both_implementations(
                complex_project_structure,
                pattern,
                case_sensitive=case_sensitive,
                mock_connection_manager=mock_connection_manager,
                agent_config=agent_config,
                mock_base_agent=mock_base_agent,
            )

            orig_parsed = self._normalize_results(orig)
            opt_parsed = self._normalize_results(opt)

            assert (
                orig_parsed == opt_parsed
            ), f"Results differ for pattern '{pattern}' with case_sensitive={case_sensitive}"

    @pytest.mark.asyncio
    async def test_file_pattern_parity(
        self, complex_project_structure, mock_connection_manager, agent_config, mock_base_agent
    ):
        """Test file pattern filtering produces identical results"""
        test_cases = [
            ("print", "*.py"),
            ("print", "*.md"),
            ("print", "test_*.py"),
            ("def", "utils.py"),
            ("print", "edge_cases.py"),
        ]

        for pattern, file_pattern in test_cases:
            orig, opt, _, _ = await self._run_both_implementations(
                complex_project_structure,
                pattern,
                file_pattern=file_pattern,
                mock_connection_manager=mock_connection_manager,
                agent_config=agent_config,
                mock_base_agent=mock_base_agent,
            )

            orig_parsed = self._normalize_results(orig)
            opt_parsed = self._normalize_results(opt)

            assert (
                orig_parsed == opt_parsed
            ), f"Results differ for pattern '{pattern}' with file_pattern='{file_pattern}'"

    @pytest.mark.asyncio
    async def test_special_regex_patterns_parity(
        self, complex_project_structure, mock_connection_manager, agent_config, mock_base_agent
    ):
        """Test special regex patterns produce identical results"""
        patterns = [
            r"\[test\]",  # Escaped brackets
            r"print\(.+\)",  # Regex with groups
            r"^def",  # Start of line
            r"import.*re",  # Wildcard
            r"test\d",  # Digit pattern
        ]

        for pattern in patterns:
            orig, opt, _, _ = await self._run_both_implementations(
                complex_project_structure,
                pattern,
                mock_connection_manager=mock_connection_manager,
                agent_config=agent_config,
                mock_base_agent=mock_base_agent,
            )

            orig_parsed = self._normalize_results(orig)
            opt_parsed = self._normalize_results(opt)

            assert orig_parsed == opt_parsed, f"Results differ for regex pattern '{pattern}'"

    @pytest.mark.asyncio
    async def test_error_handling_parity(
        self, complex_project_structure, mock_connection_manager, agent_config, mock_base_agent
    ):
        """Test error handling produces identical results"""
        invalid_patterns = [
            "[",  # Invalid regex
            "(",  # Unclosed group
            "*",  # Invalid regex
        ]

        for pattern in invalid_patterns:
            orig, opt, _, _ = await self._run_both_implementations(
                complex_project_structure,
                pattern,
                mock_connection_manager=mock_connection_manager,
                agent_config=agent_config,
                mock_base_agent=mock_base_agent,
            )

            assert "Error: Invalid regular expression" in orig
            assert "Error: Invalid regular expression" in opt

    @pytest.mark.asyncio
    async def test_unicode_content_parity(self, tmp_path, mock_connection_manager, agent_config, mock_base_agent):
        """Test searching unicode content"""
        (tmp_path / "unicode.py").write_text('print("Hello 世界 🌍")', encoding="utf-8")

        orig, opt, _, _ = await self._run_both_implementations(
            tmp_path,
            "世界",
            mock_connection_manager=mock_connection_manager,
            agent_config=agent_config,
            mock_base_agent=mock_base_agent,
        )

        assert "unicode.py" in orig
        assert "unicode.py" in opt
        assert "世界" in orig
        assert "世界" in opt

    @pytest.mark.asyncio
    async def test_empty_directory_parity(self, tmp_path, mock_connection_manager, agent_config, mock_base_agent):
        """Test searching in empty directory"""
        orig, opt, _, _ = await self._run_both_implementations(
            tmp_path,
            "test",
            mock_connection_manager=mock_connection_manager,
            agent_config=agent_config,
            mock_base_agent=mock_base_agent,
        )

        assert "No matches found" in orig
        assert "No matches found" in opt

    @pytest.mark.asyncio
    async def test_binary_file_exclusion_parity(
        self, complex_project_structure, mock_connection_manager, agent_config, mock_base_agent
    ):
        """Test that binary files are excluded identically"""
        # Search for content that might be in binary files
        orig, opt, _, _ = await self._run_both_implementations(
            complex_project_structure,
            "PNG",
            mock_connection_manager=mock_connection_manager,
            agent_config=agent_config,
            mock_base_agent=mock_base_agent,
        )

        # Both should not find matches in binary files
        assert "logo.png" not in orig
        assert "logo.png" not in opt

    @pytest.mark.asyncio
    async def test_excluded_directories_parity(
        self, complex_project_structure, mock_connection_manager, agent_config, mock_base_agent
    ):
        """Test that excluded directories are handled identically"""
        # Search for content in excluded directories
        orig, opt, _, _ = await self._run_both_implementations(
            complex_project_structure,
            "git config",
            mock_connection_manager=mock_connection_manager,
            agent_config=agent_config,
            mock_base_agent=mock_base_agent,
        )

        # Both should not find matches in .git directory
        assert ".git/config" not in orig
        assert ".git/config" not in opt

    @pytest.mark.asyncio
    async def test_performance_improvement(
        self, complex_project_structure, mock_connection_manager, agent_config, mock_base_agent
    ):
        """Test that optimized version is at least as fast as original"""
        # Run multiple searches to get average times
        patterns = ["print", "def", "import", "class", "return"]

        total_time_original = 0
        total_time_optimized = 0

        for pattern in patterns:
            _, _, time_orig, time_opt = await self._run_both_implementations(
                complex_project_structure,
                pattern,
                mock_connection_manager=mock_connection_manager,
                agent_config=agent_config,
                mock_base_agent=mock_base_agent,
            )
            total_time_original += time_orig
            total_time_optimized += time_opt

        # Optimized should not be significantly slower (allow 20% margin for variance)
        assert (
            total_time_optimized <= total_time_original * 1.2
        ), f"Optimized ({total_time_optimized:.3f}s) is slower than original ({total_time_original:.3f}s)"

        print(f"\nPerformance: Original {total_time_original:.3f}s, Optimized {total_time_optimized:.3f}s")
        if total_time_optimized < total_time_original:
            print(f"Improvement: {total_time_original / total_time_optimized:.2f}x faster")


class TestSearchCodebaseToolSandboxParity:
    """Test parity in sandbox environment"""

    @pytest.fixture
    def mock_sandbox(self):
        """Create a mock sandbox that simulates E2B sandbox behavior"""
        sandbox = Mock()

        # Mock file operations
        sandbox.files = Mock()
        sandbox.commands = Mock()

        # Setup mock responses for various commands
        def mock_run_side_effect(cmd):
            result = Mock()
            result.exit_code = 0
            result.stderr = ""

            if "grep" in cmd and "-r" in cmd:
                # Mock grep command output in AWK-formatted style
                result.stdout = """- **src/main.py** (1 matches)
  Line 2: print("Hello World")

- **docs/README.md** (1 matches)
  Line 2: This project prints output."""
            elif "find" in cmd and "-printf" in cmd:
                # Mock find command output (for original implementation)
                result.stdout = """src/main.py\t512
src/utils.py\t256
tests/test_main.py\t128
docs/README.md\t1024"""
            elif "test -e" in cmd:
                result.exit_code = 0  # File exists
            elif "test -f" in cmd:
                result.exit_code = 0  # Is file
            elif "test -d" in cmd:
                result.exit_code = 1  # Not directory
            elif "file -b --mime" in cmd:
                result.stdout = "text/plain; charset=utf-8"
            else:
                result.stdout = ""

            return result

        # Make sandbox.commands.run a Mock with side_effect
        sandbox.commands.run = Mock(side_effect=mock_run_side_effect)

        # Mock file reading
        def mock_read(path):
            if "main.py" in path:
                return 'def main():\n    print("Hello World")'
            elif "utils.py" in path:
                return 'def helper():\n    return "Helper"'
            elif "test_main.py" in path:
                return "def test_main():\n    assert True"
            elif "README.md" in path:
                return "# README\nThis project prints output."
            return ""

        sandbox.files.read = mock_read

        return sandbox

    @pytest.mark.asyncio
    async def test_sandbox_filesystem_detection(
        self, mock_sandbox, tmp_path, mock_connection_manager, agent_config, mock_base_agent
    ):
        """Test that sandbox filesystem is properly detected and used"""
        # Create sandbox filesystem
        sandbox_fs = SandboxFileSystem(mock_sandbox, str(tmp_path))

        # Create tools with sandbox filesystem
        original_tool = SearchCodebaseToolOriginal(
            tmp_path,
            "test_workspace",
            str(uuid.uuid4()),
            mock_connection_manager,
            agent_config,
            mock_base_agent,
            filesystem=sandbox_fs,
        )

        optimized_tool = SearchCodebaseTool(
            tmp_path,
            "test_workspace",
            str(uuid.uuid4()),
            mock_connection_manager,
            agent_config,
            mock_base_agent,
            filesystem=sandbox_fs,
        )

        # Verify sandbox is detected in optimized version
        assert hasattr(optimized_tool.filesystem, "sandbox")

        # Run search
        result = await optimized_tool.search_codebase("print")

        # Verify grep command was called (optimized version uses grep in sandbox)
        assert any("grep" in str(call) for call in mock_sandbox.commands.run.call_args_list)


@pytest.mark.asyncio
class TestSearchCodebaseToolEdgeCases:
    """Test edge cases and boundary conditions"""

    async def test_multiple_matches_per_line(self, tmp_path, mock_connection_manager, agent_config, mock_base_agent):
        """Test files with multiple matches on the same line"""
        (tmp_path / "multi.py").write_text('print("test"); print("test"); print("test")')

        test = TestSearchCodebaseToolParity()
        orig, opt, _, _ = await test._run_both_implementations(
            tmp_path,
            "print",
            mock_connection_manager=mock_connection_manager,
            agent_config=agent_config,
            mock_base_agent=mock_base_agent,
        )

        # Both should find the same number of matches
        orig_parsed = test._normalize_results(orig)
        opt_parsed = test._normalize_results(opt)

        assert orig_parsed == opt_parsed
        assert "multi.py" in orig
        assert "(3 matches)" in orig

    async def test_very_long_lines(self, tmp_path, mock_connection_manager, agent_config, mock_base_agent):
        """Test files with very long lines"""
        long_line = "x" * 10000 + 'print("found")' + "y" * 10000
        (tmp_path / "long.py").write_text(long_line)

        test = TestSearchCodebaseToolParity()
        orig, opt, _, _ = await test._run_both_implementations(
            tmp_path,
            "print",
            mock_connection_manager=mock_connection_manager,
            agent_config=agent_config,
            mock_base_agent=mock_base_agent,
        )

        # Both should handle long lines similarly
        assert ("print" in orig) == ("print" in opt)
        assert ("long.py" in orig) == ("long.py" in opt)
        
        # Both should truncate long lines to 200 characters + "..."
        assert "x" * 10000 not in orig, "Original should truncate long lines"
        assert "x" * 10000 not in opt, "Optimized should truncate long lines"
        assert "..." in orig, "Original should add ellipsis to truncated lines"
        assert "..." in opt, "Optimized should add ellipsis to truncated lines"
        
        # Verify both implementations produce identical output
        orig_parsed = test._normalize_results(orig)
        opt_parsed = test._normalize_results(opt)
        assert orig_parsed == opt_parsed, "Long line handling should be identical"