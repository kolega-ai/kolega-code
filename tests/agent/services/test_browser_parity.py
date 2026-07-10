import urllib.parse
from unittest.mock import AsyncMock, MagicMock

import pytest

from kolega_code.sandbox.browser import SandboxBrowserManager
from kolega_code.services.browser import PlaywrightBrowserManager


@pytest.fixture
def browserless_env(monkeypatch):
    monkeypatch.setenv("BROWSERLESS_API_KEY", "test-token")


def test_unknown_backend_is_rejected():
    with pytest.raises(ValueError, match="Unknown browser backend"):
        PlaywrightBrowserManager("unknown")


def test_browserless_cloud_requires_auth(monkeypatch):
    monkeypatch.delenv("BROWSERLESS_API_KEY", raising=False)
    monkeypatch.delenv("BROWSERLESS_WS_ENDPOINT", raising=False)

    with pytest.raises(ValueError, match="Browserless credentials not found"):
        PlaywrightBrowserManager("browserless")


def test_browserless_self_hosted_endpoint_does_not_require_token(monkeypatch):
    monkeypatch.delenv("BROWSERLESS_API_KEY", raising=False)
    manager = PlaywrightBrowserManager("browserless", browserless_endpoint="ws://localhost:3000")

    assert manager._browserless_url() == "ws://localhost:3000"


def test_browserless_cloud_hostname_requires_domain_boundary(monkeypatch):
    monkeypatch.delenv("BROWSERLESS_API_KEY", raising=False)
    manager = PlaywrightBrowserManager("browserless", browserless_endpoint="wss://notbrowserless.io")

    assert manager._browserless_url() == "wss://notbrowserless.io"


def test_browserless_endpoint_preserves_query_and_adds_configured_values(browserless_env):
    manager = PlaywrightBrowserManager(
        "browserless",
        browserless_endpoint="wss://production-lon.browserless.io/chromium?stealth=true",
        browserless_timeout_ms=300000,
    )

    parsed = urllib.parse.urlsplit(manager._browserless_url())
    query = urllib.parse.parse_qs(parsed.query)
    assert parsed.hostname == "production-lon.browserless.io"
    assert query == {"stealth": ["true"], "token": ["test-token"], "timeout": ["300000"]}


def test_browserless_region_and_native_protocol(monkeypatch, browserless_env):
    monkeypatch.setenv("BROWSERLESS_REGION", "ams")
    manager = PlaywrightBrowserManager("browserless", browserless_protocol="playwright")

    assert manager.browserless_endpoint == "wss://production-ams.browserless.io/chromium/playwright"


def test_browserless_does_not_force_session_timeout(browserless_env):
    manager = PlaywrightBrowserManager("browserless")

    assert "timeout=" not in manager._browserless_url()


def test_browserless_token_is_redacted_from_connection_errors(browserless_env):
    manager = PlaywrightBrowserManager("browserless")

    message = manager._redact_connection_error(
        RuntimeError("failed wss://production-sfo.browserless.io?token=test-token&timeout=1")
    )

    assert "test-token" not in message
    assert "token=***" in message


@pytest.mark.asyncio
async def test_browserless_cdp_uses_connect_over_cdp(browserless_env):
    manager = PlaywrightBrowserManager("browserless")
    playwright = MagicMock()
    playwright.chromium.connect_over_cdp = AsyncMock(return_value="browser")

    result = await manager._connect(playwright)

    assert result == "browser"
    playwright.chromium.connect_over_cdp.assert_awaited_once()
    playwright.chromium.connect.assert_not_called()


@pytest.mark.asyncio
async def test_browserless_native_uses_playwright_connect(browserless_env):
    manager = PlaywrightBrowserManager("browserless", browserless_protocol="playwright")
    playwright = MagicMock()
    playwright.chromium.connect = AsyncMock(return_value="browser")

    result = await manager._connect(playwright)

    assert result == "browser"
    playwright.chromium.connect.assert_awaited_once()
    playwright.chromium.connect_over_cdp.assert_not_called()


def test_sandbox_manager_selects_browserless(browserless_env):
    manager = SandboxBrowserManager(sandbox=object())

    assert manager.browser_backend == "browserless"
    assert manager.sandbox is not None


def test_browserstack_capabilities_do_not_leak_into_browserless(monkeypatch):
    monkeypatch.setenv("BROWSERSTACK_USERNAME", "user")
    monkeypatch.setenv("BROWSERSTACK_ACCESS_KEY", "key")
    manager = PlaywrightBrowserManager("browserstack")

    assert manager._browserstack_url().startswith("wss://cdp.browserstack.com/playwright?caps=")
