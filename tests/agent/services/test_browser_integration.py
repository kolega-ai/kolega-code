# ruff: noqa: F401,F811,E402
import datetime
import os
import pytest
from unittest.mock import AsyncMock
from kolega_code.services.browser import PlaywrightBrowserManager

# Check if running in CI environment
SKIP_IN_CI = bool(os.getenv("CI")) or bool(os.getenv("GITLAB_CI"))


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
            if isinstance(browser_id, str) and browser_id in browser_manager.browsers:
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
