"""Recent-session rows inside the startup card."""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Sequence

import pytest
from rich.cells import cell_len
from rich.text import Text
from textual import on
from textual.app import App, ComposeResult

from kolega_code.cli.tui.startup import (
    RecentSessionItem,
    RecentSessionRow,
    RecentSessionsWidget,
    StartupEntryWidget,
)
from kolega_code.cli.tui.state import ConversationEntry


async def _wait_for_layout(pilot, predicate, *, timeout: float = 6.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        await pilot.pause(0.02)
        if predicate():
            return
    raise AssertionError("recent-session layout did not settle")


class RecentSessionsHarness(App[None]):
    def __init__(self, factory: Callable[[], Sequence[RecentSessionItem]] | None, *, title: str = "Startup") -> None:
        super().__init__()
        self.requests: list[str] = []
        self.factory = factory
        self.entry = ConversationEntry(kind="startup", content="Project: /tmp/project\nSession: current")
        self.title = title

    def compose(self) -> ComposeResult:
        yield StartupEntryWidget(
            self.entry,
            lambda _entry, _width=None: self.title,
            lambda _entry, _width: Text("model · mode"),
            lambda entry: entry.content,
            recent_sessions_factory=self.factory,
        )

    @on(StartupEntryWidget.ResumeRequested)
    def _resume_requested(self, event: StartupEntryWidget.ResumeRequested) -> None:
        self.requests.append(event.session_id)


def _item(
    session_id: str,
    title: str,
    *,
    updated_at: str | None = None,
    locked: bool = False,
) -> RecentSessionItem:
    return RecentSessionItem(
        session_id=session_id,
        title=title,
        updated_at=updated_at or (datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat(),
        locked=locked,
    )


@pytest.mark.asyncio
async def test_recent_session_rows_click_tab_enter_and_locked_disabled(tmp_path: Path) -> None:
    app = RecentSessionsHarness(
        lambda: [
            _item("session-click", "clickable.py"),
            _item("session-keyboard", "keyboard.py"),
            _item("session-locked", "locked.py", locked=True),
        ]
    )

    async with app.run_test(size=(90, 30)) as pilot:
        await _wait_for_layout(pilot, lambda: bool(app.query(RecentSessionRow)))
        rows = list(app.query(RecentSessionRow))
        assert len(rows) == 3
        assert rows[0].display and rows[0].can_focus and not rows[0].disabled
        assert rows[1].display and rows[1].can_focus and not rows[1].disabled
        assert rows[2].display and not rows[2].can_focus and rows[2].disabled
        assert "in use" in rows[2].render_line(0).text

        assert await pilot.click(rows[0])
        await pilot.pause()
        assert app.requests == ["session-click"]

        for _ in range(12):
            await pilot.press("tab")
            await pilot.pause()
            if app.focused is rows[1]:
                break
        assert app.focused is rows[1]
        await pilot.press("enter")
        await pilot.pause()
        assert app.requests == ["session-click", "session-keyboard"]

        assert await pilot.click(rows[2])
        await pilot.pause()
        assert app.requests == ["session-click", "session-keyboard"]


@pytest.mark.asyncio
async def test_recent_sessions_refresh_shows_and_hides_section(tmp_path: Path) -> None:
    items: list[RecentSessionItem] = []
    calls = 0

    def factory() -> Sequence[RecentSessionItem]:
        nonlocal calls
        calls += 1
        return list(items)

    app = RecentSessionsHarness(factory)

    async with app.run_test(size=(90, 30)) as pilot:
        await _wait_for_layout(pilot, lambda: bool(app.query(RecentSessionsWidget)))
        section = app.query_one(RecentSessionsWidget)
        rows = list(app.query(RecentSessionRow))
        assert calls >= 1
        assert not section.display
        assert all(not row.display for row in rows)

        before = calls
        items[:] = [_item("session-visible", "visible.py")]
        app.query_one(StartupEntryWidget).refresh_content()
        await _wait_for_layout(pilot, lambda: section.display and rows[0].display)
        assert calls > before
        assert rows[0].render_line(0).text.strip()
        assert all(not row.display for row in rows[1:])

        before = calls
        items.clear()
        app.query_one(StartupEntryWidget).refresh_content()
        await _wait_for_layout(pilot, lambda: not section.display and all(not row.display for row in rows))
        assert calls > before


@pytest.mark.asyncio
@pytest.mark.parametrize("width", [28, 120])
async def test_recent_sessions_render_literal_single_line_without_link_metadata(width: int) -> None:
    raw_uuid = "123e4567-e89b-12d3-a456-426614174000"
    title = "[bold]literal[/bold]\n漢字-folder/\tvery-long-file-name-with-extension.py\x1b[31m"
    app = RecentSessionsHarness(lambda: [_item(raw_uuid, title, updated_at="not-a-timestamp")])

    async with app.run_test(size=(width, 24)) as pilot:
        await _wait_for_layout(pilot, lambda: bool(app.query(RecentSessionRow)))
        row = app.query_one(RecentSessionRow)
        await _wait_for_layout(pilot, lambda: row.display and row.size.width > 0)
        line = row.render_line(0)
        text = line.text
        assert row.size.height == 1
        assert cell_len(text) <= row.size.width
        assert "unknown" in text
        assert raw_uuid not in text
        assert "\n" not in text and "\t" not in text and "\x1b" not in text
        assert not any(segment.style and segment.style.link for segment in line)
        if width >= 120:
            assert "[bold]literal[/bold]" in text
            assert "漢字-folder/" in text
        else:
            assert "…" in text
