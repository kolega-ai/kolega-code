"""One headless Chromium per player-test module.

Every replay-player test used to enter ``async_playwright()`` and launch Chromium
itself. ``async_playwright()`` starts a Node driver process (~0.6 s) and a launch
adds ~0.15 s, so a module that renders the player a dozen times paid most of its
wall time starting browsers rather than playing sessions.

The driver and browser are per module here; every use still gets a fresh
``BrowserContext`` and page, which is the boundary Playwright isolates cookies,
storage, network state and console/network listeners across. A module-scoped
``asyncio`` fixture keeps the browser on the same event loop as the tests, so
these modules run their cases with ``loop_scope="module"``.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import pytest
import pytest_asyncio
from playwright.async_api import Browser, Page, ViewportSize, async_playwright

_BROWSER: Browser | None = None


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def shared_player_browser() -> AsyncGenerator[None, None]:
    """Start one Chromium for the module; skip the module when it is missing."""
    global _BROWSER
    driver = async_playwright()
    playwright = await driver.__aenter__()
    try:
        try:
            _BROWSER = await playwright.chromium.launch(headless=True)
        except Exception as exc:  # pragma: no cover - depends on the local install
            pytest.skip(f"chromium is not installed for playwright: {exc}")
        try:
            yield
        finally:
            await _BROWSER.close()
            _BROWSER = None
    finally:
        await driver.__aexit__(None, None, None)


@asynccontextmanager
async def player_page(*, viewport: ViewportSize | None = None) -> AsyncGenerator[Page, None]:
    """A fresh, isolated page on the module's browser.

    The page lives in its own context, so a test never inherits another test's
    cookies, storage, navigation history or listeners. The context is closed
    even when the test body fails.
    """
    assert _BROWSER is not None, "player pages need the shared_player_browser fixture"
    context = await _BROWSER.new_context(viewport=viewport) if viewport else await _BROWSER.new_context()
    try:
        yield await context.new_page()
    finally:
        await context.close()
