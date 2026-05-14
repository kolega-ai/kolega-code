import datetime
import pytest
import uuid
from unittest.mock import AsyncMock, MagicMock, patch
from kolega_code.agent.tool_backend.browser_tool import BrowserTool
from kolega_code.agent.config import AgentConfig


class TestBrowserTool:
    """Test suite for BrowserTool console log filtering functionality."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock agent config."""
        return MagicMock(spec=AgentConfig)

    @pytest.fixture
    def mock_connection_manager(self):
        """Create a mock connection manager."""
        return AsyncMock()

    @pytest.fixture
    def mock_caller(self):
        """Create a mock caller."""
        caller = MagicMock()
        caller.agent_name = "test-agent"
        return caller

    @pytest.fixture
    def browser_tool(self, mock_config, mock_connection_manager, mock_caller):
        """Create a browser tool instance for testing."""
        return BrowserTool(
            project_path="/test/path",
            workspace_id="test-workspace",
            thread_id=str(uuid.uuid4()),
            connection_manager=mock_connection_manager,
            config=mock_config,
            caller=mock_caller,
        )

    @pytest.fixture
    def mock_console_logs_result(self):
        """Create mock console logs result from browser manager."""
        return {
            "console_logs": [
                {
                    "type": "error",
                    "text": "JavaScript error occurred",
                    "timestamp": datetime.datetime.now().isoformat(),
                    "location": {"url": "test.js", "lineNumber": 42, "columnNumber": 10},
                },
                {
                    "type": "warning",
                    "text": "Deprecated API usage",
                    "timestamp": datetime.datetime.now().isoformat(),
                    "location": None,
                },
            ],
            "total_logs_count": 10,
            "returned_count": 2,
            "filters_applied": {
                "max_logs": 50,
                "log_types": ["error", "warning", "assert"],
                "minutes_back": None,
                "max_chars": 8000,
            },
        }

    @pytest.mark.asyncio
    async def test_get_browser_console_logs_default_parameters(self, browser_tool, mock_console_logs_result):
        """Test get_browser_console_logs with default parameters."""
        browser_tool.browser_manager.get_browser_console_logs = AsyncMock(return_value=mock_console_logs_result)

        result = await browser_tool.get_browser_console_logs("test-browser-id")

        # Verify the browser manager was called with default parameters
        browser_tool.browser_manager.get_browser_console_logs.assert_called_once_with(
            "test-browser-id", max_logs=50, log_types=None, minutes_back=None, max_chars=8000
        )

        # Check that the result contains expected markdown formatting
        assert "## Console Logs" in result
        assert "**Showing 2 of 10 total logs**" in result
        assert "**Filtered by types:** error, warning, assert" in result
        assert "**Max logs:** 50" in result
        assert "| Type | Timestamp | Message | Location |" in result
        assert "| error |" in result
        assert "| warning |" in result
        assert "JavaScript error occurred" in result
        assert "Deprecated API usage" in result

    @pytest.mark.asyncio
    async def test_get_browser_console_logs_custom_parameters(self, browser_tool, mock_console_logs_result):
        """Test get_browser_console_logs with custom parameters."""
        # Update mock result to reflect custom parameters
        mock_console_logs_result["filters_applied"] = {
            "max_logs": 10,
            "log_types": ["error"],
            "minutes_back": 5,
            "max_chars": 1000,
        }

        browser_tool.browser_manager.get_browser_console_logs = AsyncMock(return_value=mock_console_logs_result)

        result = await browser_tool.get_browser_console_logs(
            "test-browser-id", max_logs=10, log_types=["error"], minutes_back=5, max_chars=1000
        )

        # Verify the browser manager was called with custom parameters
        browser_tool.browser_manager.get_browser_console_logs.assert_called_once_with(
            "test-browser-id", max_logs=10, log_types=["error"], minutes_back=5, max_chars=1000
        )

        # Check that the result reflects custom filtering
        assert "**Filtered by types:** error" in result
        assert "**Time window:** Last 5 minutes" in result
        assert "**Character limit:** 1000" in result
        assert "**Max logs:** 10" in result

    @pytest.mark.asyncio
    async def test_get_browser_console_logs_no_logs(self, browser_tool):
        """Test get_browser_console_logs when no logs are found."""
        mock_empty_result = {
            "console_logs": [],
            "total_logs_count": 0,
            "returned_count": 0,
            "filters_applied": {
                "max_logs": 50,
                "log_types": ["error", "warning", "assert"],
                "minutes_back": None,
                "max_chars": 8000,
            },
        }

        browser_tool.browser_manager.get_browser_console_logs = AsyncMock(return_value=mock_empty_result)

        result = await browser_tool.get_browser_console_logs("test-browser-id")

        assert result == "## Console Logs\n\nNo console logs found."

    @pytest.mark.asyncio
    async def test_get_browser_console_logs_location_formatting(self, browser_tool):
        """Test that console log locations are formatted correctly."""
        mock_result = {
            "console_logs": [
                {
                    "type": "error",
                    "text": "Error with location",
                    "timestamp": datetime.datetime.now().isoformat(),
                    "location": {"url": "test.js", "lineNumber": 42, "columnNumber": 10},
                },
                {
                    "type": "warning",
                    "text": "Warning without location",
                    "timestamp": datetime.datetime.now().isoformat(),
                    "location": None,
                },
            ],
            "total_logs_count": 2,
            "returned_count": 2,
            "filters_applied": {
                "max_logs": 50,
                "log_types": ["error", "warning", "assert"],
                "minutes_back": None,
                "max_chars": 8000,
            },
        }

        browser_tool.browser_manager.get_browser_console_logs = AsyncMock(return_value=mock_result)

        result = await browser_tool.get_browser_console_logs("test-browser-id")

        # Check location formatting
        assert "test.js:42:10" in result  # Formatted location
        assert "N/A" in result  # No location case

    @pytest.mark.asyncio
    async def test_get_browser_console_logs_escapes_pipe_characters(self, browser_tool):
        """Test that pipe characters in log messages are escaped for markdown tables."""
        mock_result = {
            "console_logs": [
                {
                    "type": "error",
                    "text": "Error with | pipe character",
                    "timestamp": datetime.datetime.now().isoformat(),
                    "location": None,
                },
            ],
            "total_logs_count": 1,
            "returned_count": 1,
            "filters_applied": {
                "max_logs": 50,
                "log_types": ["error", "warning", "assert"],
                "minutes_back": None,
                "max_chars": 8000,
            },
        }

        browser_tool.browser_manager.get_browser_console_logs = AsyncMock(return_value=mock_result)

        result = await browser_tool.get_browser_console_logs("test-browser-id")

        # Check that pipe character is escaped
        assert "Error with \\| pipe character" in result

    @pytest.mark.asyncio
    async def test_get_browser_content_with_console_log_filtering(self, browser_tool):
        """Test get_browser_content with console log filtering."""
        mock_content_result = {
            "current_url": "https://example.com",
            "title": "Test Page",
            "html": "<html><body>Test</body></html>",
            "console_logs": [
                {
                    "type": "error",
                    "text": "JavaScript error",
                    "timestamp": datetime.datetime.now().isoformat(),
                    "location": None,
                },
            ],
            "console_log_metadata": {
                "total_logs_count": 5,
                "returned_count": 1,
                "filters_applied": {"max_logs": 10, "log_types": ["error"], "minutes_back": None, "max_chars": 1000},
            },
        }

        browser_tool.browser_manager.get_browser_content = AsyncMock(return_value=mock_content_result)

        result = await browser_tool.get_browser_content(
            "test-browser-id", max_logs=10, log_types=["error"], max_chars=1000
        )

        # Verify the browser manager was called with filtering parameters
        browser_tool.browser_manager.get_browser_content.assert_called_once_with(
            "test-browser-id", max_logs=10, log_types=["error"], minutes_back=None, max_chars=1000
        )

        # Check that the result contains expected content
        assert "# Browser Content: Test Page" in result
        assert "**Current URL:** https://example.com" in result
        assert "## Console Logs" in result
        assert "**Showing 1 of 5 total logs**" in result
        assert "**Filtered by types:** error" in result
        assert "**Character limit:** 1000" in result
        assert "**Max logs:** 10" in result
        assert "## Page HTML" in result
        assert "<html><body>Test</body></html>" in result

    @pytest.mark.asyncio
    async def test_get_browser_content_no_console_logs(self, browser_tool):
        """Test get_browser_content when there are no console logs."""
        mock_content_result = {
            "current_url": "https://example.com",
            "title": "Test Page",
            "html": "<html><body>Test</body></html>",
            "console_logs": [],
            "console_log_metadata": {
                "total_logs_count": 0,
                "returned_count": 0,
                "filters_applied": {
                    "max_logs": 50,
                    "log_types": ["error", "warning", "assert"],
                    "minutes_back": None,
                    "max_chars": 8000,
                },
            },
        }

        browser_tool.browser_manager.get_browser_content = AsyncMock(return_value=mock_content_result)

        result = await browser_tool.get_browser_content("test-browser-id")

        # Should not contain console logs section when there are no logs
        assert "# Browser Content: Test Page" in result
        assert "**Current URL:** https://example.com" in result
        assert "## Console Logs" not in result
        assert "## Page HTML" in result

    @pytest.mark.asyncio
    async def test_get_browser_content_without_metadata(self, browser_tool):
        """Test get_browser_content when console_log_metadata is missing."""
        mock_content_result = {
            "current_url": "https://example.com",
            "title": "Test Page",
            "html": "<html><body>Test</body></html>",
            "console_logs": [
                {
                    "type": "error",
                    "text": "JavaScript error",
                    "timestamp": datetime.datetime.now().isoformat(),
                    "location": None,
                },
            ],
            # Missing console_log_metadata
        }

        browser_tool.browser_manager.get_browser_content = AsyncMock(return_value=mock_content_result)

        result = await browser_tool.get_browser_content("test-browser-id")

        # Should still work without metadata
        assert "# Browser Content: Test Page" in result
        assert "## Console Logs" in result
        assert "JavaScript error" in result
        # Should not contain metadata information
        assert "**Showing" not in result
        assert "**Filtered by types:**" not in result
