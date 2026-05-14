"""Test parity between local and BrowserStack browser managers."""

import datetime
import os
import pytest
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock
from dotenv import load_dotenv

from kolega_code.agent.services.browser import PlaywrightBrowserManager
from kolega_code.agent.services.sandbox.sandbox_browser import SandboxBrowserManager

# Load environment variables from .env file
load_dotenv()


class TestBrowserManagerParity:
    """Test suite to ensure identical behavior between local and BrowserStack browser managers."""

    @pytest.fixture
    def mock_env_vars(self, monkeypatch):
        """Mock environment variables for BrowserStack and Browserless."""
        monkeypatch.setenv("BROWSERSTACK_USERNAME", "test_user")
        monkeypatch.setenv("BROWSERSTACK_ACCESS_KEY", "test_key")
        monkeypatch.setenv("BROWSERLESS_API_KEY", "test_browserless_key")

    @pytest.fixture
    def local_browser_manager(self):
        """Create a local browser manager instance."""
        return PlaywrightBrowserManager(browser_backend="local")

    @pytest.fixture
    def browserstack_browser_manager(self, mock_env_vars):
        """Create a BrowserStack browser manager instance."""
        return PlaywrightBrowserManager(browser_backend="browserstack")

    @pytest.fixture
    def browserless_browser_manager(self, mock_env_vars):
        """Create a Browserless browser manager instance."""
        return PlaywrightBrowserManager(browser_backend="browserless")

    @pytest.fixture
    def sandbox_browser_manager(self, mock_env_vars):
        """Create a sandbox browser manager instance (uses Browserless)."""
        return SandboxBrowserManager()

    def test_initialization(self, local_browser_manager, browserstack_browser_manager, sandbox_browser_manager):
        """Test that all managers initialize with the same properties."""
        # Check common properties
        assert local_browser_manager.viewport == browserstack_browser_manager.viewport
        assert local_browser_manager.viewport == sandbox_browser_manager.viewport

        assert local_browser_manager.user_agent == browserstack_browser_manager.user_agent
        assert local_browser_manager.user_agent == sandbox_browser_manager.user_agent

        assert local_browser_manager.headless == browserstack_browser_manager.headless
        assert local_browser_manager.headless == sandbox_browser_manager.headless

        assert local_browser_manager.interaction_timeout == browserstack_browser_manager.interaction_timeout
        assert local_browser_manager.interaction_timeout == sandbox_browser_manager.interaction_timeout

        assert (
            local_browser_manager.max_console_logs_per_browser
            == browserstack_browser_manager.max_console_logs_per_browser
        )
        assert (
            local_browser_manager.max_console_logs_per_browser == sandbox_browser_manager.max_console_logs_per_browser
        )

        # Check backend-specific properties
        assert local_browser_manager.browser_backend == "local"
        assert browserstack_browser_manager.browser_backend == "browserstack"
        assert sandbox_browser_manager.browser_backend == "browserless"

    def test_browserstack_credentials_required(self):
        """Test that BrowserStack and Browserless managers require credentials."""
        # Clear environment variables
        os.environ.pop("BROWSERSTACK_USERNAME", None)
        os.environ.pop("BROWSERSTACK_ACCESS_KEY", None)
        os.environ.pop("BROWSERLESS_API_KEY", None)

        # Should raise ValueError without BrowserStack credentials
        with pytest.raises(ValueError, match="BrowserStack credentials not found"):
            PlaywrightBrowserManager(browser_backend="browserstack")

        # Should raise ValueError without Browserless credentials
        with pytest.raises(ValueError, match="Browserless API key not found"):
            PlaywrightBrowserManager(browser_backend="browserless")

        # SandboxBrowserManager uses Browserless, so should fail without Browserless credentials
        with pytest.raises(ValueError, match="Browserless API key not found"):
            SandboxBrowserManager()

    def test_backward_compatibility(self, mock_env_vars):
        """Test backward compatibility with use_browserstack parameter."""
        # Using use_browserstack=True should set browser_backend to "browserstack"
        manager = PlaywrightBrowserManager(use_browserstack=True)
        assert manager.browser_backend == "browserstack"

        # Using use_browserstack=False should keep default backend
        manager = PlaywrightBrowserManager(use_browserstack=False)
        assert manager.browser_backend == "local"

    @pytest.mark.asyncio
    async def test_launch_browser_interface(self, local_browser_manager, browserstack_browser_manager):
        """Test that launch_browser has the same interface for both managers."""
        # Mock the playwright async_playwright
        with patch("kolega_code.agent.services.browser.async_playwright") as mock_playwright:
            # Setup mock playwright - fix the async mock issue
            mock_pw_instance = MagicMock()
            mock_async_playwright = AsyncMock()
            mock_async_playwright.start.return_value = mock_pw_instance
            mock_playwright.return_value = mock_async_playwright

            # Mock browser and page
            mock_browser = AsyncMock()
            mock_context = AsyncMock()
            mock_page = AsyncMock()

            mock_browser.new_context.return_value = mock_context
            mock_context.new_page.return_value = mock_page
            mock_page.url = "https://example.com"
            mock_page.evaluate = AsyncMock()
            mock_page.goto = AsyncMock()
            mock_page.on = MagicMock()

            # For local browser
            mock_pw_instance.chromium = MagicMock()
            mock_pw_instance.chromium.launch = AsyncMock(return_value=mock_browser)

            # For BrowserStack browser - mock connect() instead of connect_over_cdp()
            mock_pw_instance.chromium.connect = AsyncMock(return_value=mock_browser)

            # Test local browser
            local_result = await local_browser_manager.launch_browser("https://example.com")
            assert local_result is not None
            assert isinstance(local_result, str)  # Should return browser ID

            # Test BrowserStack browser
            bs_result = await browserstack_browser_manager.launch_browser("https://example.com")
            assert bs_result is not None
            assert isinstance(bs_result, str)  # Should return browser ID

            # Verify different connection methods were used
            mock_pw_instance.chromium.launch.assert_called_once()
            mock_pw_instance.chromium.connect.assert_called_once()

    @pytest.mark.asyncio
    async def test_browser_info_structure(self, local_browser_manager, browserstack_browser_manager):
        """Test that browser info structure is identical for both managers."""
        # Mock the playwright async_playwright
        with patch("kolega_code.agent.services.browser.async_playwright") as mock_playwright:
            # Setup mock playwright - fix the async mock issue
            mock_pw_instance = MagicMock()
            mock_async_playwright = AsyncMock()
            mock_async_playwright.start.return_value = mock_pw_instance
            mock_playwright.return_value = mock_async_playwright

            # Mock browser and page
            mock_browser = AsyncMock()
            mock_context = AsyncMock()
            mock_page = AsyncMock()

            mock_browser.new_context.return_value = mock_context
            mock_context.new_page.return_value = mock_page
            mock_page.url = "https://example.com"
            mock_page.evaluate = AsyncMock()
            mock_page.goto = AsyncMock()
            mock_page.on = MagicMock()

            # For both browser types
            mock_pw_instance.chromium = MagicMock()
            mock_pw_instance.chromium.launch = AsyncMock(return_value=mock_browser)
            mock_pw_instance.chromium.connect = AsyncMock(return_value=mock_browser)

            # Launch browsers
            local_id = await local_browser_manager.launch_browser("https://example.com")
            bs_id = await browserstack_browser_manager.launch_browser("https://example.com")

            # Check browser info structure
            local_info = local_browser_manager.browsers[local_id]
            bs_info = browserstack_browser_manager.browsers[bs_id]

            # Both should have the same keys
            assert set(local_info.keys()) == set(bs_info.keys())

            # Check specific fields
            assert local_info["type"] == bs_info["type"] == "chromium"
            assert local_info["url"] == bs_info["url"] == "https://example.com"
            assert "playwright" in local_info and "playwright" in bs_info
            assert "browser" in local_info and "browser" in bs_info
            assert "context" in local_info and "context" in bs_info
            assert "page" in local_info and "page" in bs_info
            assert "console_logs" in local_info and "console_logs" in bs_info
            assert "network_requests" in local_info and "network_requests" in bs_info
            assert "launched_at" in local_info and "launched_at" in bs_info

            # BrowserStack flag should be different
            assert local_info["browserstack"] is False
            assert bs_info["browserstack"] is True

            # Backend field should be different
            assert local_info["backend"] == "local"
            assert bs_info["backend"] == "browserstack"

    @pytest.mark.asyncio
    async def test_console_log_handling(self, local_browser_manager, browserstack_browser_manager):
        """Test that console log handling is identical for both managers."""
        # Create mock browser info with console logs
        now = datetime.datetime.now()
        console_logs = [
            {
                "type": "error",
                "text": "Test error",
                "timestamp": now.isoformat(),
                "location": None,
            }
        ]

        browser_id = "test-browser-id"
        mock_browser_info = {
            "console_logs": console_logs,
            "launched_at": now.isoformat(),
        }

        # Add to both managers
        local_browser_manager.browsers[browser_id] = mock_browser_info.copy()
        browserstack_browser_manager.browsers[browser_id] = mock_browser_info.copy()

        # Test console log retrieval
        local_logs = await local_browser_manager.get_browser_console_logs(browser_id)
        bs_logs = await browserstack_browser_manager.get_browser_console_logs(browser_id)

        # Results should be identical
        assert local_logs == bs_logs
        assert local_logs["total_logs_count"] == 1
        assert local_logs["returned_count"] == 1
        assert local_logs["console_logs"][0]["type"] == "error"
        assert local_logs["console_logs"][0]["text"] == "Test error"

    @pytest.mark.asyncio
    async def test_error_handling(self, local_browser_manager, browserstack_browser_manager):
        """Test that error handling is consistent across both managers."""
        # Test browser not found error
        with pytest.raises(KeyError, match="Browser with ID nonexistent not found"):
            await local_browser_manager.get_browser_console_logs("nonexistent")

        with pytest.raises(KeyError, match="Browser with ID nonexistent not found"):
            await browserstack_browser_manager.get_browser_console_logs("nonexistent")

        # Test other methods with non-existent browser
        with pytest.raises(KeyError):
            await local_browser_manager.take_browser_screenshot("nonexistent")

        with pytest.raises(KeyError):
            await browserstack_browser_manager.take_browser_screenshot("nonexistent")

    @pytest.mark.asyncio
    async def test_sandbox_manager_inheritance(self, browserless_browser_manager, sandbox_browser_manager):
        """Test that SandboxBrowserManager behaves as a PlaywrightBrowserManager with Browserless backend."""
        # Check that they both use Browserless backend
        assert browserless_browser_manager.browser_backend == "browserless"
        assert sandbox_browser_manager.browser_backend == "browserless"

        # Both managers should have browserless credentials
        assert hasattr(browserless_browser_manager, "browserless_api_key")

        # Sandbox manager should have browserless credentials
        assert hasattr(sandbox_browser_manager, "browserless_api_key")

        # SandboxBrowserManager should have its additional sandbox attribute
        assert hasattr(sandbox_browser_manager, "sandbox")
        assert not hasattr(browserless_browser_manager, "sandbox")

    def test_cdp_url_generation(
        self, browserstack_browser_manager, browserless_browser_manager, sandbox_browser_manager
    ):
        """Test that managers generate correct CDP URLs for their respective backends."""
        # BrowserStack manager should generate BrowserStack URL
        bs_url = browserstack_browser_manager._get_browserstack_cdp_url()
        assert bs_url.startswith("wss://cdp.browserstack.com/playwright?caps=")
        assert "browserstack.username" in bs_url
        assert "browserstack.accessKey" in bs_url

        # Browserless manager should generate Browserless URL
        browserless_url = browserless_browser_manager._get_browserless_cdp_url()
        assert browserless_url.startswith("wss://production-sfo.browserless.io?token=")
        assert "timeout=" in browserless_url

        # Sandbox manager should generate Browserless URL
        sandbox_browserless_url = sandbox_browser_manager._get_browserless_cdp_url()
        assert sandbox_browserless_url.startswith("wss://production-sfo.browserless.io?token=")
        assert "timeout=" in sandbox_browserless_url

    @pytest.mark.asyncio
    async def test_list_browsers_parity(self, local_browser_manager, browserstack_browser_manager):
        """Test that list_browsers returns the same structure for both managers."""
        # Mock browser info
        browser_info = {
            "url": "https://example.com",
            "launched_at": datetime.datetime.now().isoformat(),
            "browserstack": False,
            "backend": "local",
        }

        browser_id = "test-id"
        local_browser_manager.browsers[browser_id] = {**browser_info, "browserstack": False, "backend": "local"}
        browserstack_browser_manager.browsers[browser_id] = {
            **browser_info,
            "browserstack": True,
            "backend": "browserstack",
        }

        local_list = await local_browser_manager.list_browsers()
        bs_list = await browserstack_browser_manager.list_browsers()

        # Check structure is the same
        assert set(local_list[browser_id].keys()) == set(bs_list[browser_id].keys())
        assert local_list[browser_id]["url"] == bs_list[browser_id]["url"]
        assert local_list[browser_id]["launched_at"] == bs_list[browser_id]["launched_at"]

        # Backend flags should be different
        assert local_list[browser_id]["browserstack"] is False
        assert bs_list[browser_id]["browserstack"] is True
        assert local_list[browser_id]["backend"] == "local"
        assert bs_list[browser_id]["backend"] == "browserstack"

    @pytest.mark.asyncio
    async def test_all_methods_available(
        self, local_browser_manager, browserstack_browser_manager, sandbox_browser_manager
    ):
        """Test that all browser manager methods are available on all implementations."""
        methods = [
            "launch_browser",
            "list_browsers",
            "get_browser_console_logs",
            "get_browser_interactive_elements",
            "get_browser_content",
            "take_browser_screenshot",
            "interact_with_browser",
            "set_select_value",
            "close_browser",
            "cleanup_all_browsers",
        ]

        for method in methods:
            assert hasattr(local_browser_manager, method)
            assert hasattr(browserstack_browser_manager, method)
            assert hasattr(sandbox_browser_manager, method)

            # All methods should be callable
            assert callable(getattr(local_browser_manager, method))
            assert callable(getattr(browserstack_browser_manager, method))
            assert callable(getattr(sandbox_browser_manager, method))
