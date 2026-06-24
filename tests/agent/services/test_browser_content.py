# ruff: noqa: F401,F811,E402
import datetime
import os
import pytest
from unittest.mock import AsyncMock
from kolega_code.services.browser import PlaywrightBrowserManager

# Check if running in CI environment
SKIP_IN_CI = bool(os.getenv("CI")) or bool(os.getenv("GITLAB_CI"))


class TestPlaywrightBrowserManager:
    @pytest.fixture
    def browser_manager(self):
        """Create a browser manager instance for testing."""
        return PlaywrightBrowserManager()

    @pytest.fixture
    def mock_browser_info(self):
        """Create mock browser info with sample console logs."""
        now = datetime.datetime.now()
        console_logs = [
            {
                "type": "log",
                "text": "Regular log message 1",
                "timestamp": (now - datetime.timedelta(minutes=10)).isoformat(),
                "location": None,
            },
            {
                "type": "error",
                "text": "JavaScript error occurred",
                "timestamp": (now - datetime.timedelta(minutes=8)).isoformat(),
                "location": {"url": "test.js", "lineNumber": 42, "columnNumber": 10},
            },
            {
                "type": "warning",
                "text": "Deprecated API usage",
                "timestamp": (now - datetime.timedelta(minutes=6)).isoformat(),
                "location": None,
            },
            {
                "type": "log",
                "text": "Regular log message 2",
                "timestamp": (now - datetime.timedelta(minutes=4)).isoformat(),
                "location": None,
            },
            {
                "type": "assert",
                "text": "Assertion failed: condition not met",
                "timestamp": (now - datetime.timedelta(minutes=2)).isoformat(),
                "location": None,
            },
            {
                "type": "info",
                "text": "Information message",
                "timestamp": now.isoformat(),
                "location": None,
            },
        ]

        return {
            "type": "chromium",
            "url": "https://example.com",
            "console_logs": console_logs,
            "launched_at": now.isoformat(),
        }

    @pytest.mark.asyncio
    async def test_get_browser_content_with_filtered_logs(self, browser_manager, mock_browser_info):
        """Test that get_browser_content uses filtered console logs."""
        browser_id = "test-browser-id"

        # Mock the page object
        mock_page = AsyncMock()
        mock_page.url = "https://example.com"
        mock_page.title.return_value = "Test Page"
        mock_page.content.return_value = "<html><body>Test</body></html>"

        mock_browser_info["page"] = mock_page
        browser_manager.browsers[browser_id] = mock_browser_info

        result = await browser_manager.get_browser_content(browser_id, max_logs=2, log_types=["error"])

        assert "current_url" in result
        assert "title" in result
        assert "html" in result
        assert "console_logs" in result
        assert "console_log_metadata" in result

        metadata = result["console_log_metadata"]
        assert metadata["total_logs_count"] == 6
        assert metadata["returned_count"] == 1  # Only 1 error log
        assert metadata["filters_applied"]["log_types"] == ["error"]
        assert metadata["filters_applied"]["max_logs"] == 2
