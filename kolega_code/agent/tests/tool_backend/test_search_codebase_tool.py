from unittest.mock import AsyncMock, Mock

import pytest
import uuid

from kolega_code.config import AgentConfig, ModelConfig, ModelProvider, RateLimitConfig
from kolega_code.agent.tool_backend.search_codebase_tool import SearchCodebaseTool


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
            provider=ModelProvider.ANTHROPIC, model="test-model", rate_limits=RateLimitConfig(), thinking_effort="medium"
        ),
    )


@pytest.fixture
def mock_base_agent():
    mock = Mock()
    mock.agent_name = "test_agent"
    return mock


@pytest.fixture
def search_codebase_tool(project_path, mock_connection_manager, agent_config, mock_base_agent):
    return SearchCodebaseTool(
        project_path, "test_workspace", str(uuid.uuid4()), mock_connection_manager, agent_config, mock_base_agent
    )


@pytest.fixture
def sample_files(project_path):
    # Create a directory structure with various files
    (project_path / "src").mkdir()
    (project_path / "tests").mkdir()

    # Create Python files
    (project_path / "src" / "main.py").write_text('def main():\n    print("Hello World")\n')
    (project_path / "src" / "utils.py").write_text('def helper():\n    return "Helper function"\n')
    (project_path / "tests" / "test_main.py").write_text("def test_main():\n    assert True\n")

    # Create a binary file
    (project_path / "src" / "data.bin").write_bytes(b"\x00\x01\x02\x03")

    # Create a file in excluded directory
    (project_path / ".git").mkdir()
    (project_path / ".git" / "config").write_text("git config content")

    # Create a large file
    (project_path / "src" / "large.txt").write_text("x" * (11 * 1024 * 1024))


@pytest.mark.asyncio
class TestSearchCodebaseTool:
    async def test_search_codebase_basic(self, search_codebase_tool, sample_files):
        result = await search_codebase_tool.search_codebase("print")

        assert "Search Results for 'print'" in result
        assert "src/main.py" in result
        assert 'Line 2: print("Hello World")' in result

    async def test_search_codebase_case_insensitive(self, search_codebase_tool, sample_files):
        result = await search_codebase_tool.search_codebase("PRINT")

        assert "Search Results for 'PRINT'" in result
        assert "src/main.py" in result
        assert 'Line 2: print("Hello World")' in result

    async def test_search_codebase_case_sensitive(self, search_codebase_tool, sample_files):
        result = await search_codebase_tool.search_codebase("PRINT", case_sensitive=True)

        assert "No matches found for pattern 'PRINT'" in result

    async def test_search_codebase_file_pattern(self, search_codebase_tool, sample_files):
        result = await search_codebase_tool.search_codebase("def", file_pattern="*.py")

        assert "Search Results for 'def'" in result
        assert "src/main.py" in result
        assert "src/utils.py" in result
        assert "tests/test_main.py" in result

    async def test_search_codebase_no_matches(self, search_codebase_tool, sample_files):
        result = await search_codebase_tool.search_codebase("nonexistent_pattern")

        assert "No matches found for pattern 'nonexistent_pattern'" in result

    async def test_search_codebase_invalid_regex(self, search_codebase_tool, sample_files):
        result = await search_codebase_tool.search_codebase("[", literal=False)

        assert "Error: Invalid regular expression" in result

    async def test_search_codebase_excludes_binary_files(self, search_codebase_tool, sample_files):
        result = await search_codebase_tool.search_codebase("\x00")

        assert "No matches found for pattern" in result

    async def test_search_codebase_excludes_git_files(self, search_codebase_tool, sample_files):
        result = await search_codebase_tool.search_codebase("git config")

        assert "No matches found for pattern" in result

    async def test_search_codebase_excludes_large_files(self, search_codebase_tool, sample_files):
        result = await search_codebase_tool.search_codebase("x")

        assert "large.txt" not in result

    async def test_search_codebase_multiple_matches(self, search_codebase_tool, sample_files):
        result = await search_codebase_tool.search_codebase("def")

        assert "Search Results for 'def'" in result
        assert "src/main.py" in result
        assert "src/utils.py" in result
        assert "tests/test_main.py" in result
        assert "Line 1: def main():" in result
        assert "Line 1: def helper():" in result
        assert "Line 1: def test_main():" in result

    async def test_search_codebase_with_context(self, search_codebase_tool, project_path):
        # Create a file with multiple matches
        file_path = project_path / "test.py"
        file_path.write_text('def first():\n    print("First")\n\ndef second():\n    print("Second")\n')

        result = await search_codebase_tool.search_codebase("print")

        assert "test.py" in result
        assert 'Line 2: print("First")' in result
        assert 'Line 5: print("Second")' in result

    async def test_search_codebase_result_limit(self, search_codebase_tool, project_path):
        # Create a file with more than 5 matches (the per-file display limit)
        file_path = project_path / "test.py"
        file_path.write_text("\n".join(f'print("Line {i}")' for i in range(200)))

        result = await search_codebase_tool.search_codebase("print")

        # Check that we show 5 lines and then indicate there are more
        assert 'Line 1: print("Line 0")' in result
        assert 'Line 5: print("Line 4")' in result
        assert "... and 195 more matches" in result
        assert "(200 matches)" in result

    async def test_search_codebase_with_special_characters(self, search_codebase_tool, project_path):
        # Create a file with special characters
        file_path = project_path / "special.py"
        file_path.write_text('def special():\n    print("Special chars: !@#$%^&*()")\n')

        result = await search_codebase_tool.search_codebase("!@#")

        assert "special.py" in result
        assert 'Line 2: print("Special chars: !@#$%^&*()")' in result

    async def test_search_codebase_literal_mode(self, search_codebase_tool, project_path):
        """Test literal search mode with regex special characters"""
        # Create files with patterns that would be invalid regex
        file_path = project_path / "code.py"
        file_path.write_text("""
def process_array(arr):
    value = arr[0])  # Unbalanced parenthesis
    return value

def func():
    print("Testing [](){}")
""")

        # Test 1: Search for unbalanced parenthesis - should work in literal mode
        result = await search_codebase_tool.search_codebase("])", literal=True)
        assert "code.py" in result
        assert "Line 3: value = arr[0])" in result

        # Test 2: Same pattern should fail in regex mode
        result = await search_codebase_tool.search_codebase("])", literal=False)
        assert "Error: Invalid regular expression" in result

        # Test 3: Search for pattern with special chars in literal mode
        result = await search_codebase_tool.search_codebase("[](){}", literal=True)
        assert "code.py" in result
        assert 'Line 7: print("Testing [](){}")' in result

        # Test 4: Verify default is literal=True
        result = await search_codebase_tool.search_codebase("])")
        assert "code.py" in result
        assert "Line 3: value = arr[0])" in result

    async def test_search_codebase_long_line_truncation(self, search_codebase_tool, project_path):
        """Test that long lines are truncated to 200 characters"""
        # Create a file with a very long line (minified JSON style)
        long_line = '{"key":"' + "x" * 500 + '","match":"FINDME"}'
        file_path = project_path / "minified.json"
        file_path.write_text(long_line)

        result = await search_codebase_tool.search_codebase("FINDME")

        assert "minified.json" in result
        # The line should be truncated to 200 characters with "..."
        assert "..." in result
        # Verify the full long line is NOT in the output
        assert "x" * 500 not in result
        # The truncated line should be approximately 200 characters (plus "Line N: " prefix and "...")
        lines = result.split("\n")
        for line in lines:
            if "Line 1:" in line and "minified.json" not in line:
                # Extract just the content part after "Line N: "
                content_match = line.split("Line 1: ", 1)
                if len(content_match) > 1:
                    content = content_match[1]
                    # Should be exactly 200 chars + "..." = 203 chars
                    assert len(content) == 203, f"Expected 203 chars, got {len(content)}"
                    assert content.endswith("...")
                break

