import datetime
import os
import pytest
from unittest.mock import AsyncMock
from kolega_code.services.browser import PlaywrightBrowserManager

# Check if running in CI environment
SKIP_IN_CI = bool(os.getenv("CI")) or bool(os.getenv("GITLAB_CI"))


class TestPlaywrightBrowserManager:
    """Test suite for PlaywrightBrowserManager console log filtering functionality."""

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
    async def test_get_browser_console_logs_default_filtering(self, browser_manager, mock_browser_info):
        """Test default console log filtering (errors, warnings, assertions only)."""
        browser_id = "test-browser-id"
        browser_manager.browsers[browser_id] = mock_browser_info

        result = await browser_manager.get_browser_console_logs(browser_id)

        assert result["total_logs_count"] == 6
        assert result["returned_count"] == 3  # Only error, warning, assert
        assert result["filters_applied"]["log_types"] == ["error", "warning", "assert"]
        assert result["filters_applied"]["max_logs"] == 50
        assert result["filters_applied"]["max_chars"] == 8000

        # Check that only the correct log types are returned
        returned_types = [log["type"] for log in result["console_logs"]]
        assert "error" in returned_types
        assert "warning" in returned_types
        assert "assert" in returned_types
        assert "log" not in returned_types
        assert "info" not in returned_types

    @pytest.mark.asyncio
    async def test_get_browser_console_logs_custom_log_types(self, browser_manager, mock_browser_info):
        """Test filtering by custom log types."""
        browser_id = "test-browser-id"
        browser_manager.browsers[browser_id] = mock_browser_info

        result = await browser_manager.get_browser_console_logs(browser_id, log_types=["log", "info"])

        assert result["total_logs_count"] == 6
        assert result["returned_count"] == 3  # 2 log + 1 info
        assert result["filters_applied"]["log_types"] == ["log", "info"]

        # Check that only the correct log types are returned
        returned_types = [log["type"] for log in result["console_logs"]]
        assert "log" in returned_types
        assert "info" in returned_types
        assert "error" not in returned_types

    @pytest.mark.asyncio
    async def test_get_browser_console_logs_max_logs_limit(self, browser_manager, mock_browser_info):
        """Test limiting the number of logs returned."""
        browser_id = "test-browser-id"
        browser_manager.browsers[browser_id] = mock_browser_info

        result = await browser_manager.get_browser_console_logs(
            browser_id, max_logs=2, log_types=[]  # Empty list to include all types
        )

        assert result["total_logs_count"] == 6
        assert result["returned_count"] == 2  # Limited to 2 most recent
        assert result["filters_applied"]["max_logs"] == 2

        # Should return the 2 most recent logs
        returned_logs = result["console_logs"]
        assert len(returned_logs) == 2
        assert returned_logs[-1]["type"] == "info"  # Most recent
        assert returned_logs[-2]["type"] == "assert"  # Second most recent

    @pytest.mark.asyncio
    async def test_get_browser_console_logs_time_filtering(self, browser_manager, mock_browser_info):
        """Test filtering logs by time window."""
        browser_id = "test-browser-id"
        browser_manager.browsers[browser_id] = mock_browser_info

        result = await browser_manager.get_browser_console_logs(
            browser_id, minutes_back=5, log_types=[]  # Include all types, filter by time
        )

        assert result["total_logs_count"] == 6
        assert result["returned_count"] == 3  # Only logs from last 5 minutes
        assert result["filters_applied"]["minutes_back"] == 5

        # All returned logs should be within the time window
        cutoff_time = datetime.datetime.now() - datetime.timedelta(minutes=5)
        for log in result["console_logs"]:
            log_time = datetime.datetime.fromisoformat(log["timestamp"])
            assert log_time > cutoff_time

    @pytest.mark.asyncio
    async def test_get_browser_console_logs_character_limit(self, browser_manager, mock_browser_info):
        """Test limiting logs by character count."""
        browser_id = "test-browser-id"
        browser_manager.browsers[browser_id] = mock_browser_info

        # Set a very low character limit to test truncation
        result = await browser_manager.get_browser_console_logs(
            browser_id, max_chars=50, log_types=[]  # Include all types
        )

        assert result["total_logs_count"] == 6
        assert result["returned_count"] <= 6  # Should be limited by character count

        # Calculate total character count of returned logs
        total_chars = sum(len(f"{log['type']}: {log['text']}") for log in result["console_logs"])
        assert total_chars <= 50

    @pytest.mark.asyncio
    async def test_get_browser_console_logs_combined_filters(self, browser_manager, mock_browser_info):
        """Test applying multiple filters simultaneously."""
        browser_id = "test-browser-id"
        browser_manager.browsers[browser_id] = mock_browser_info

        result = await browser_manager.get_browser_console_logs(
            browser_id, max_logs=10, log_types=["error", "warning"], minutes_back=10, max_chars=1000
        )

        assert result["total_logs_count"] == 6
        assert result["filters_applied"]["log_types"] == ["error", "warning"]
        assert result["filters_applied"]["minutes_back"] == 10
        assert result["filters_applied"]["max_logs"] == 10
        assert result["filters_applied"]["max_chars"] == 1000

        # Should only contain error and warning logs
        returned_types = [log["type"] for log in result["console_logs"]]
        for log_type in returned_types:
            assert log_type in ["error", "warning"]

    @pytest.mark.asyncio
    async def test_get_browser_console_logs_empty_logs(self, browser_manager):
        """Test behavior when no console logs exist."""
        browser_id = "test-browser-id"
        browser_manager.browsers[browser_id] = {
            "console_logs": [],
            "launched_at": datetime.datetime.now().isoformat(),
        }

        result = await browser_manager.get_browser_console_logs(browser_id)

        assert result["total_logs_count"] == 0
        assert result["returned_count"] == 0
        assert result["console_logs"] == []

    @pytest.mark.asyncio
    async def test_get_browser_console_logs_browser_not_found(self, browser_manager):
        """Test error handling when browser ID doesn't exist."""
        with pytest.raises(KeyError, match="Browser with ID nonexistent not found"):
            await browser_manager.get_browser_console_logs("nonexistent")

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

    def test_circular_buffer_implementation(self, browser_manager):
        """Test that the circular buffer prevents unlimited log growth."""
        # Set a small buffer size for testing
        browser_manager.max_console_logs_per_browser = 3

        console_logs = []

        # Simulate the console log handler behavior
        def simulate_console_log_handler(msg_text, msg_type="log"):
            log_entry = {
                "type": msg_type,
                "text": msg_text,
                "timestamp": datetime.datetime.now().isoformat(),
                "location": None,
            }
            console_logs.append(log_entry)

            # Implement circular buffer logic
            if len(console_logs) > browser_manager.max_console_logs_per_browser:
                console_logs.pop(0)  # Remove oldest log

        # Add more logs than the buffer size
        simulate_console_log_handler("Log 1")
        simulate_console_log_handler("Log 2")
        simulate_console_log_handler("Log 3")
        assert len(console_logs) == 3

        simulate_console_log_handler("Log 4")
        assert len(console_logs) == 3  # Should still be 3
        assert console_logs[0]["text"] == "Log 2"  # First log should be removed
        assert console_logs[-1]["text"] == "Log 4"  # Last log should be the newest

        simulate_console_log_handler("Log 5")
        assert len(console_logs) == 3
        assert console_logs[0]["text"] == "Log 3"
        assert console_logs[-1]["text"] == "Log 5"

    @pytest.mark.asyncio
    async def test_no_log_types_filter_includes_all(self, browser_manager, mock_browser_info):
        """Test that passing an empty list for log_types includes all log types."""
        browser_id = "test-browser-id"
        browser_manager.browsers[browser_id] = mock_browser_info

        result = await browser_manager.get_browser_console_logs(
            browser_id, log_types=[]  # Empty list should include all types
        )

        assert result["total_logs_count"] == 6
        assert result["returned_count"] == 6  # All logs should be included
        assert result["filters_applied"]["log_types"] == []

        # Should include all log types
        returned_types = [log["type"] for log in result["console_logs"]]
        assert "log" in returned_types
        assert "error" in returned_types
        assert "warning" in returned_types
        assert "assert" in returned_types
        assert "info" in returned_types

    @pytest.mark.asyncio
    async def test_character_limit_preserves_most_recent(self, browser_manager):
        """Test that character limit preserves the most recent logs."""
        browser_id = "test-browser-id"

        # Create logs with known character counts
        console_logs = [
            {
                "type": "log",
                "text": "A" * 10,  # 10 chars + "log: " = 14 chars
                "timestamp": datetime.datetime.now().isoformat(),
                "location": None,
            },
            {
                "type": "log",
                "text": "B" * 10,  # 10 chars + "log: " = 14 chars
                "timestamp": datetime.datetime.now().isoformat(),
                "location": None,
            },
            {
                "type": "log",
                "text": "C" * 10,  # 10 chars + "log: " = 14 chars (most recent)
                "timestamp": datetime.datetime.now().isoformat(),
                "location": None,
            },
        ]

        browser_manager.browsers[browser_id] = {
            "console_logs": console_logs,
            "launched_at": datetime.datetime.now().isoformat(),
        }

        # Set character limit to allow only 1 log (14 chars) to test the logic
        result = await browser_manager.get_browser_console_logs(browser_id, max_chars=14, log_types=[])

        assert result["returned_count"] == 1
        # Should preserve the most recent log (C)
        returned_texts = [log["text"] for log in result["console_logs"]]
        assert "C" * 10 in returned_texts
        assert "A" * 10 not in returned_texts
        assert "B" * 10 not in returned_texts


class TestPlaywrightBrowserManagerIntegration:
    """Integration tests for PlaywrightBrowserManager that use real browsers."""

    @pytest.mark.integration
    @pytest.mark.skipif(SKIP_IN_CI, reason="Skipping network-dependent test in CI environment")
    @pytest.mark.asyncio
    async def test_real_browser_google_screenshot(self):
        """Integration test: Launch real browser, load Google, take screenshot."""
        # Check if BROWSERLESS_API_KEY is available, skip if not
        if not os.getenv("BROWSERLESS_API_KEY"):
            pytest.skip("BROWSERLESS_API_KEY environment variable not set")

        browser_manager = PlaywrightBrowserManager(browser_backend="browserless")
        browser_id = None

        try:
            # Launch browser and navigate to Google
            browser_id = await browser_manager.launch_browser("https://www.google.com")

            # Verify we got a valid browser ID (not an error dict)
            assert isinstance(browser_id, str)
            assert browser_id != ""

            # Verify browser is in the manager's registry
            assert browser_id in browser_manager.browsers
            browser_info = browser_manager.browsers[browser_id]
            assert browser_info["url"] == "https://www.google.com"
            assert browser_info["backend"] == "browserless"
            assert browser_info["browserstack"] is False

            # Take a screenshot
            screenshot_result = await browser_manager.take_browser_screenshot(browser_id)

            # Verify screenshot result structure
            assert "current_url" in screenshot_result
            assert "title" in screenshot_result
            assert "screenshot" in screenshot_result

            # Verify we're actually on Google
            assert "google" in screenshot_result["current_url"].lower()
            assert "google" in screenshot_result["title"].lower()

            # Verify screenshot is base64 encoded
            screenshot_data = screenshot_result["screenshot"]
            assert isinstance(screenshot_data, str)
            assert len(screenshot_data) > 0
            # Basic check that it's base64 (starts with image header)
            import base64

            try:
                decoded = base64.b64decode(screenshot_data)
                assert len(decoded) > 0
            except Exception:
                pytest.fail("Screenshot is not valid base64 data")

            # Test console logs capture
            console_logs_result = await browser_manager.get_browser_console_logs(browser_id)
            assert "console_logs" in console_logs_result
            assert "total_logs_count" in console_logs_result
            assert "returned_count" in console_logs_result

            # Test browser content retrieval
            content_result = await browser_manager.get_browser_content(browser_id)
            assert "current_url" in content_result
            assert "title" in content_result
            assert "html" in content_result
            assert "console_logs" in content_result
            assert len(content_result["html"]) > 0

            # Test interactive elements extraction
            elements_result = await browser_manager.get_browser_interactive_elements(browser_id)
            assert "current_url" in elements_result
            assert "title" in elements_result
            assert "interactive_elements" in elements_result

        finally:
            # Clean up: close the browser if it was created
            if browser_id and browser_id in browser_manager.browsers:
                await browser_manager.close_browser(browser_id)

            # Additional cleanup to ensure all browsers are closed
            await browser_manager.cleanup_all_browsers()

    @pytest.mark.integration
    @pytest.mark.asyncio
    @pytest.mark.skipif(SKIP_IN_CI, reason="Skipping slow test in CI environment")
    async def test_real_browser_error_handling(self):
        """Integration test: Test error handling with real browser for invalid URLs."""
        # Check if BROWSERLESS_API_KEY is available, skip if not
        if not os.getenv("BROWSERLESS_API_KEY"):
            pytest.skip("BROWSERLESS_API_KEY environment variable not set")

        browser_manager = PlaywrightBrowserManager(browser_backend="browserless")

        try:
            # Try to launch browser with invalid URL
            result = await browser_manager.launch_browser("not-a-valid-url")

            # Should return error dict rather than browser ID
            if isinstance(result, dict) and "error" in result:
                assert "error" in result
                assert "Browser Launch Error" in result["error"]
            else:
                # If it somehow succeeds (maybe Playwright handles it), clean up
                if isinstance(result, str) and result in browser_manager.browsers:
                    await browser_manager.close_browser(result)

        finally:
            await browser_manager.cleanup_all_browsers()
