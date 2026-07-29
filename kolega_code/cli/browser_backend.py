"""Ordinary-host browser backend selection for the CLI."""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

from kolega_code.browser_extension.installer import (
    CHROME_EXTENSION_ID,
    configured_extension_origin,
    native_host_status,
)
from kolega_code.browser_extension.manager import ChromeExtensionBrowserManager, ChromeExtensionUnavailableError
from kolega_code.browser_extension.native_host import NativeHostConfigurationError
from kolega_code.browser_extension.runtime import UnsupportedRuntimeTransportError
from kolega_code.services.base import BrowserManager
from kolega_code.services.browser import PlaywrightBrowserManager


def _configured_extension_origin(state_dir: Path) -> str:
    try:
        origin = configured_extension_origin(state_dir=state_dir)
        extension_id = origin.removeprefix("chrome-extension://").removesuffix("/")
        channel = "production" if extension_id == CHROME_EXTENSION_ID else "dev"
        status = native_host_status(
            channel=channel,
            extension_id=None if channel == "production" else extension_id,
            state_dir=state_dir,
        )
    except UnsupportedRuntimeTransportError as exc:
        raise NativeHostConfigurationError(str(exc)) from None
    if not status.valid:
        raise NativeHostConfigurationError(status.detail)
    return origin


class OrdinaryHostBrowserManager(PlaywrightBrowserManager):
    """Keep Playwright as the default while exposing configured host backends."""

    browser_target: ClassVar[str] = "playwright"

    def __init__(
        self,
        *,
        state_dir: Path,
        kolega_session_id: str,
        browser_visible: bool = False,
    ) -> None:
        super().__init__()
        self.state_dir = state_dir
        self.kolega_session_id = kolega_session_id
        self.headless = not browser_visible
        self._chrome_manager: ChromeExtensionBrowserManager | None = None

    @property
    def browser_targets(self) -> tuple[str, ...]:
        try:
            _configured_extension_origin(self.state_dir)
        except NativeHostConfigurationError:
            return ("playwright",)
        return ("playwright", "chrome")

    def resolve_browser_target(self, browser_target: str | None = None) -> BrowserManager:
        if browser_target is None or browser_target == "playwright":
            return self
        if browser_target != "chrome":
            raise ValueError(f"Unknown browser target {browser_target!r}. Expected 'playwright' or 'chrome'.")
        if self._chrome_manager is not None:
            return self._chrome_manager
        try:
            extension_origin = _configured_extension_origin(self.state_dir)
        except NativeHostConfigurationError:
            raise ChromeExtensionUnavailableError(
                "Chrome browser integration is not configured. Run `kolega-code browser install` and retry."
            ) from None
        self._chrome_manager = ChromeExtensionBrowserManager(
            state_dir=self.state_dir,
            kolega_session_id=self.kolega_session_id,
            extension_origin=extension_origin,
        )
        return self._chrome_manager

    async def cleanup_all_browsers(self) -> None:
        try:
            if self._chrome_manager is not None:
                await self._chrome_manager.cleanup_all_browsers()
        finally:
            await super().cleanup_all_browsers()


def build_browser_manager(
    state_dir: Path,
    kolega_session_id: str,
    browser_visible: bool = False,
) -> OrdinaryHostBrowserManager:
    """Build the default ordinary-host Playwright manager."""
    return OrdinaryHostBrowserManager(
        state_dir=state_dir,
        kolega_session_id=kolega_session_id,
        browser_visible=browser_visible,
    )
