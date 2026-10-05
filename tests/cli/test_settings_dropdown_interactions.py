"""Exercise settings dropdowns through real keyboard and pointer events."""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path

import pytest
from textual import events
from textual.containers import VerticalScroll
from textual.pilot import Pilot
from textual.widgets import Select, Switch
from textual.widgets._select import SelectOverlay

from kolega_code.cli.tui.settings_screen import ConfirmDiscardSettingsScreen, SettingsScreen
from kolega_code.mcp.config import MCPConfigFile, MCPServerConfig, global_mcp_config_path, save_config_file

from .test_tui_settings_screens import _configured_app


async def _wait_for_layout(pilot: Pilot, predicate: Callable[[], bool], *, timeout: float = 6.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        await pilot.pause(0.02)
        if predicate():
            return
    raise AssertionError("Settings dropdown layout did not settle")


async def _position_select(pilot: Pilot, page: VerticalScroll, select: Select) -> None:
    """Leave room to scroll the parent in either direction before opening the menu."""
    await _wait_for_layout(pilot, lambda: select.region.height > 0 and page.max_scroll_y > 0)
    select.focus(scroll_visible=False)
    target_y = page.content_region.y + page.scrollable_content_region.height // 2
    target_scroll = max(1, min(page.max_scroll_y - 1, page.scroll_y + select.region.y - target_y))
    page.scroll_to(y=target_scroll, animate=False)
    await _wait_for_layout(
        pilot,
        lambda: (
            select.has_focus
            and page.scroll_y == target_scroll
            and page.content_region.y <= select.region.y
            and select.region.bottom <= page.content_region.bottom
        ),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(80, 24), (120, 40)])
@pytest.mark.parametrize("selector", ["#mcp_server_select", "#mcp_enabled_select"])
async def test_settings_dropdown_contains_wheel_at_both_boundaries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    isolated_cli_env: None,
    size: tuple[int, int],
    selector: str,
) -> None:
    app, store = _configured_app(tmp_path, monkeypatch)
    # A real inventory puts the server picker mid-page at both terminal sizes.
    # The fake agent never connects to these servers.
    save_config_file(
        global_mcp_config_path(store.root),
        MCPConfigFile(
            servers=[
                MCPServerConfig(id=f"example-{index:02}", url=f"https://example-{index:02}.invalid/mcp")
                for index in range(20)
            ]
        ),
    )
    async with app.run_test(size=size) as pilot:
        app.action_open_settings(category="mcp")
        await _wait_for_layout(pilot, lambda: isinstance(app.screen, SettingsScreen) and not app.screen._initializing)
        screen = app.screen
        assert isinstance(screen, SettingsScreen)
        page = screen.query_one("#settings_page_mcp", VerticalScroll)
        select = screen.query_one(selector, Select)
        await _position_select(pilot, page, select)
        page_offset = page.scroll_offset
        current_value = select.value
        await pilot.press("enter")
        overlay = select.query_one(SelectOverlay)
        await _wait_for_layout(pilot, lambda: select.expanded and overlay.has_focus and overlay.region.height > 0)
        menu_region = overlay.region
        assert page.scroll_offset == page_offset
        assert (overlay.max_scroll_y > 0) == (selector == "#mcp_server_select")
        pointer = (overlay.content_region.x + 1, overlay.content_region.y + 1)

        overlay.scroll_home(animate=False)
        await _wait_for_layout(pilot, lambda: overlay.scroll_y == 0)
        await pilot._post_mouse_events([events.MouseScrollUp], offset=pointer)
        assert page.scroll_offset == page_offset
        assert overlay.region == menu_region

        await pilot._post_mouse_events([events.MouseScrollDown], offset=pointer)
        if overlay.max_scroll_y:
            await _wait_for_layout(pilot, lambda: overlay.scroll_y > 0)
        assert page.scroll_offset == page_offset
        assert overlay.region == menu_region

        overlay.scroll_end(animate=False)
        await _wait_for_layout(pilot, lambda: overlay.scroll_y == overlay.max_scroll_y)
        for _ in range(3):
            await pilot._post_mouse_events([events.MouseScrollDown], offset=pointer)
            assert page.scroll_offset == page_offset
            assert overlay.region == menu_region
        assert select.value == current_value  # Wheel movement is not a committed choice.
        assert select.expanded

        # Containment is local to the dropdown, not a page-wide scroll lock.
        await pilot._post_mouse_events([events.MouseScrollDown], page, offset=(1, 1))
        await _wait_for_layout(pilot, lambda: page.scroll_offset != page_offset)


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(80, 24), (120, 40)])
@pytest.mark.parametrize("dirty", [False, True])
async def test_settings_escape_closes_dropdown_before_settings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    isolated_cli_env: None,
    size: tuple[int, int],
    dirty: bool,
) -> None:
    app, store = _configured_app(tmp_path, monkeypatch)
    saved = store.load().to_dict()
    async with app.run_test(size=size) as pilot:
        app.action_open_settings(category="tools")
        await _wait_for_layout(pilot, lambda: isinstance(app.screen, SettingsScreen) and not app.screen._initializing)
        screen = app.screen
        assert isinstance(screen, SettingsScreen)
        if dirty:
            tips = screen.query_one("#discovery_tips_switch", Switch)
            tips.value = not tips.value
            await _wait_for_layout(pilot, lambda: screen.dirty)
        page = screen.query_one("#settings_page_tools", VerticalScroll)
        select = screen.query_one("#skills_enabled_select", Select)
        await _position_select(pilot, page, select)
        page_offset = page.scroll_offset
        select_region = select.region
        value = select.value
        await pilot.press("enter")
        await _wait_for_layout(pilot, lambda: select.expanded and select.query_one(SelectOverlay).has_focus)
        assert page.scroll_offset == page_offset

        await pilot.press("escape")
        assert app.screen is screen
        await _wait_for_layout(pilot, lambda: not select.expanded and select.has_focus)
        assert page.scroll_offset == page_offset
        assert select.region == select_region
        assert select.value == value
        assert screen.dirty is dirty
        assert store.load().to_dict() == saved

        await pilot.press("escape")
        if dirty:
            await _wait_for_layout(pilot, lambda: isinstance(app.screen, ConfirmDiscardSettingsScreen))
            await pilot.press("escape")  # Keep Editing still works.
            await _wait_for_layout(pilot, lambda: app.screen is screen)
            assert screen.dirty
        else:
            await _wait_for_layout(pilot, lambda: app.screen is not screen)
        assert store.load().to_dict() == saved


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(80, 24), (120, 40)])
@pytest.mark.parametrize("method", ["keyboard", "mouse"])
async def test_settings_dropdown_selection_keeps_page_and_focus_stable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    isolated_cli_env: None,
    size: tuple[int, int],
    method: str,
) -> None:
    app, _ = _configured_app(tmp_path, monkeypatch)
    async with app.run_test(size=size) as pilot:
        app.action_open_settings(category="tools")
        await _wait_for_layout(pilot, lambda: isinstance(app.screen, SettingsScreen) and not app.screen._initializing)
        screen = app.screen
        assert isinstance(screen, SettingsScreen)
        page = screen.query_one("#settings_page_tools", VerticalScroll)
        select = screen.query_one("#skills_enabled_select", Select)
        assert select.value == "true"
        await _position_select(pilot, page, select)
        page_offset = page.scroll_offset
        select_region = select.region
        await pilot.press("enter")
        overlay = select.query_one(SelectOverlay)
        await _wait_for_layout(pilot, lambda: select.expanded and overlay.has_focus and overlay.region.height > 0)
        if method == "keyboard":
            await pilot.press("down", "enter")
        else:
            # Click the second option through hit-testing, including the border.
            await pilot.click(offset=(overlay.content_region.x + 1, overlay.content_region.y + 1))
        await _wait_for_layout(pilot, lambda: select.value == "false" and not select.expanded and select.has_focus)
        assert page.scroll_offset == page_offset
        assert select.region == select_region
        assert screen.dirty

        # Closed controls must not swallow ordinary page scrolling.
        await pilot._post_mouse_events([events.MouseScrollUp], offset=(select.region.x + 1, select.region.y + 1))
        await _wait_for_layout(pilot, lambda: page.scroll_y < page_offset.y)
