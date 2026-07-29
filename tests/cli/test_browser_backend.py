from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

from kolega_code.browser_extension.installer import CHROME_EXTENSION_ID, NativeHostStatus
from kolega_code.browser_extension.manager import ChromeExtensionUnavailableError
from kolega_code.browser_extension.native_host import NativeHostConfigurationError
from kolega_code.cli.browser_backend import _configured_extension_origin, build_browser_manager
from kolega_code.services.browser import PlaywrightBrowserManager

_EXTENSION_ORIGIN = "chrome-extension://edihigldhbmimflgjkohkgnjefmhngdn/"


def test_configuration_requires_a_valid_native_host(tmp_path: Path) -> None:
    status = NativeHostStatus(
        manifest_path=tmp_path / "manifest.json",
        installed=True,
        valid=True,
        host_path=tmp_path / "host",
        extension_id=CHROME_EXTENSION_ID,
        detail="valid",
    )
    with (
        patch(
            "kolega_code.cli.browser_backend.configured_extension_origin",
            return_value=_EXTENSION_ORIGIN,
        ),
        patch("kolega_code.cli.browser_backend.native_host_status", return_value=status) as host_status,
    ):
        assert _configured_extension_origin(tmp_path) == _EXTENSION_ORIGIN

    host_status.assert_called_once_with(
        channel="production",
        extension_id=None,
        state_dir=tmp_path,
    )


def test_configured_chrome_target_is_advertised(tmp_path: Path) -> None:
    with patch(
        "kolega_code.cli.browser_backend._configured_extension_origin",
        return_value=_EXTENSION_ORIGIN,
    ):
        manager = build_browser_manager(tmp_path, "session-1")

        assert manager.browser_targets == ("playwright", "chrome")


def test_default_and_playwright_targets_resolve_to_default_manager(tmp_path: Path) -> None:
    manager = build_browser_manager(tmp_path, "session-1")

    assert manager.resolve_browser_target() is manager
    assert manager.resolve_browser_target("playwright") is manager


def test_chrome_target_is_created_lazily_and_cached(tmp_path: Path) -> None:
    chrome = Mock()
    with (
        patch(
            "kolega_code.cli.browser_backend._configured_extension_origin",
            return_value=_EXTENSION_ORIGIN,
        ) as configured_origin,
        patch(
            "kolega_code.cli.browser_backend.ChromeExtensionBrowserManager",
            return_value=chrome,
        ) as chrome_manager,
    ):
        manager = build_browser_manager(tmp_path, "session-1")

        chrome_manager.assert_not_called()
        assert manager.resolve_browser_target("chrome") is chrome
        assert manager.resolve_browser_target("chrome") is chrome

    configured_origin.assert_called_once_with(tmp_path)
    chrome_manager.assert_called_once_with(
        state_dir=tmp_path,
        kolega_session_id="session-1",
        extension_origin=_EXTENSION_ORIGIN,
    )


def test_unconfigured_chrome_target_is_not_advertised_or_fallback(tmp_path: Path) -> None:
    with patch(
        "kolega_code.cli.browser_backend._configured_extension_origin",
        side_effect=NativeHostConfigurationError("not configured"),
    ):
        manager = build_browser_manager(tmp_path, "session-1")

        assert manager.browser_targets == ("playwright",)
        with pytest.raises(ChromeExtensionUnavailableError, match=r"kolega-code browser install"):
            manager.resolve_browser_target("chrome")


@pytest.mark.parametrize(
    ("browser_visible", "expected_headless"),
    [(False, True), (True, False)],
)
def test_factory_sets_cli_fields_and_playwright_visibility(
    tmp_path: Path,
    browser_visible: bool,
    expected_headless: bool,
) -> None:
    manager = build_browser_manager(tmp_path, "session-1", browser_visible)

    assert isinstance(manager, PlaywrightBrowserManager)
    assert manager.state_dir == tmp_path
    assert manager.kolega_session_id == "session-1"
    assert manager.headless is expected_headless


@pytest.mark.asyncio
async def test_cleanup_attempts_chrome_and_playwright_cleanup(tmp_path: Path) -> None:
    chrome = Mock()
    chrome.cleanup_all_browsers = AsyncMock(side_effect=RuntimeError("chrome cleanup failed"))
    with (
        patch(
            "kolega_code.cli.browser_backend._configured_extension_origin",
            return_value=_EXTENSION_ORIGIN,
        ),
        patch(
            "kolega_code.cli.browser_backend.ChromeExtensionBrowserManager",
            return_value=chrome,
        ),
        patch.object(
            PlaywrightBrowserManager,
            "cleanup_all_browsers",
            new_callable=AsyncMock,
        ) as playwright_cleanup,
    ):
        manager = build_browser_manager(tmp_path, "session-1")
        manager.resolve_browser_target("chrome")

        with pytest.raises(RuntimeError, match="chrome cleanup failed"):
            await manager.cleanup_all_browsers()

    chrome.cleanup_all_browsers.assert_awaited_once_with()
    playwright_cleanup.assert_awaited_once_with()
