"""TUI feature tests operate on a ready app, not its first loading frame."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import pytest
from textual.pilot import Pilot

from kolega_code.cli.app import KolegaCodeApp


@pytest.fixture(autouse=True)
def ready_tui_run_test(monkeypatch: pytest.MonkeyPatch) -> None:
    """Retain the ready-app contract of existing feature tests.

    Production startup now runs *after* the first paint. Lifecycle tests call
    ``App.run_test(app, ...)`` directly to exercise that loading interval;
    ``app.run_test(...)`` waits for real initialization, never a guessed sleep.
    """
    run_test = KolegaCodeApp.run_test

    @asynccontextmanager
    async def ready(app: KolegaCodeApp, **kwargs: Any) -> AsyncIterator[Pilot]:
        async with run_test(app, **kwargs) as pilot:
            await asyncio.wait_for(app._startup_complete.wait(), timeout=10)
            await pilot.pause()
            yield pilot

    monkeypatch.setattr(KolegaCodeApp, "run_test", ready)
