# ruff: noqa: F401,F811,E402
"""Scrollback-window behavior: bounded mounted widgets on large transcripts.

The transcript mounts a sliding window of its entries so Textual reflows stay
O(window). These tests also cover windowed restores, modal-cover deferral, and
the sub-agent inspector's expanding suffix.
"""

from pathlib import Path
import time
from typing import Any, cast

import pytest
from textual import events
from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.geometry import Size
from textual.screen import ModalScreen
from textual.widgets import Collapsible, Static

from kolega_code.cli import theme
from kolega_code.cli.tui import state as tui_state
from kolega_code.cli.tui.state import ConversationEntry
from kolega_code.cli.tui.sub_agent_screen import SubAgentRosterRow
from kolega_code.cli.tui.widgets import ConversationEntryWidget, JumpToBottomBar, TrajectoryScrollView

from ._app_test_utils import _build_sub_agent_test_app, _sub_agent_delta, _sub_agent_event, wait_for_layout

WINDOW_MAX = theme.TRANSCRIPT_WINDOW_MAX


def _window(app):
    window = app._transcript_window
    assert window is not None
    return window


TRIM_CHUNK = theme.TRANSCRIPT_WINDOW_TRIM_CHUNK
EXPAND_CHUNK = theme.TRANSCRIPT_WINDOW_EXPAND_CHUNK
INSPECTOR_MAX = theme.INSPECTOR_WINDOW_MAX


def _add_entries(app, count: int, *, start: int = 0, kind: str = "user") -> None:
    for index in range(start, start + count):
        app._add_conversation_entry(ConversationEntry(kind=kind, content=f"message {index}"))


async def _settle(pilot, rounds: int = 3) -> None:
    for _ in range(rounds):
        await pilot.pause()


async def _wait_for_layout(pilot, predicate, *, timeout: float = 6.0) -> None:
    """Poll until ``predicate()`` is truthy or the layout deadline expires."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        await pilot.pause(0.02)
        if predicate():
            return
    raise AssertionError(f"layout did not settle within {timeout}s")


@pytest.mark.asyncio
async def test_transcript_trims_oldest_entries_while_following_bottom(tmp_path: Path, monkeypatch) -> None:
    app = _build_sub_agent_test_app(tmp_path, monkeypatch)

    async with app.run_test() as pilot:
        await _settle(pilot)
        # Stage 1: at the cap (counting the startup entry), everything mounts.
        stage_one = WINDOW_MAX - 1
        _add_entries(app, stage_one)
        app._flush_conversation_render()
        await _settle(pilot)
        assert len(app._entry_widgets) == stage_one + 1

        # Stage 2: crossing max + trim chunk trims the oldest back to the cap.
        stage_two = TRIM_CHUNK + 10
        total = stage_one + stage_two + 1
        _add_entries(app, stage_two, start=stage_one)
        app._flush_conversation_render()
        await _settle(pilot)

        entries = app.conversation_entries
        assert len(entries) == total  # data model keeps everything
        assert len(app._entry_widgets) == WINDOW_MAX
        assert _window(app).mounted_start == total - WINDOW_MAX
        first_mounted = next(iter(app._entry_widgets))
        assert first_mounted == entries[total - WINDOW_MAX].entry_id
        assert entries[0].entry_id not in app._entry_widgets


@pytest.mark.asyncio
async def test_transcript_keeps_visible_history_when_output_arrives(tmp_path: Path, monkeypatch) -> None:
    app = _build_sub_agent_test_app(tmp_path, monkeypatch)

    async with app.run_test() as pilot:
        await wait_for_layout(pilot, app._startup_complete.is_set)
        stage_one = WINDOW_MAX - 1
        _add_entries(app, stage_one)
        app._flush_conversation_render()
        view = app._conversation
        await wait_for_layout(
            pilot, lambda: view.max_scroll_y > 20 and view.is_at_bottom() and not app._conversation_anchor_pending
        )
        view.scroll_to(y=20, animate=False, immediate=True)
        window = _window(app)
        before = list(window.widgets)
        anchor = next(widget for widget in window.widgets.values() if widget.virtual_region.bottom > 20)
        await wait_for_layout(pilot, lambda: anchor in view.screen._compositor.visible_widgets)
        offset = anchor.region.y - view.content_region.y

        # Both incremental updates and bursts beyond the usual rebuild guard
        # should leave the same mounted interval and screen position.
        for count in (1, 300):
            _add_entries(app, count, start=len(app.conversation_entries))
            app._flush_conversation_render()
            await wait_for_layout(pilot, lambda: view.has_newer_entries and not view.auto_follow_bottom)
            assert list(window.widgets) == before
            assert anchor.region.y - view.content_region.y == offset
            assert view.scroll_y == 20

        # Arrive at the mounted edge without paging on the preceding scroll,
        # as can happen when a swap or a coverage modal suppresses its watcher.
        bar = app.query_one("#jump_to_bottom", JumpToBottomBar)
        with monkeypatch.context() as setup:
            setup.setattr(app, "_maybe_expand_transcript_window_down", lambda: None)
            view.scroll_to(y=view.max_scroll_y, animate=False, immediate=True)
            await wait_for_layout(pilot, lambda: view.scroll_y == view.max_scroll_y and bar.display)
            assert view.has_newer_entries and not view.is_at_bottom()
        old_end = window.mounted_end
        await view._on_message(events.MouseScrollDown(view, 0, 0, 0, 0, 0, False, False, False))
        await wait_for_layout(
            pilot, lambda: window.mounted_end == old_end + EXPAND_CHUNK and view._scrollback_anchor is None
        )
        assert not view.auto_follow_bottom


@pytest.mark.asyncio
async def test_transcript_expand_on_scroll_up_preserves_position(tmp_path: Path, monkeypatch) -> None:
    app = _build_sub_agent_test_app(tmp_path, monkeypatch)

    async with app.run_test() as pilot:
        await _settle(pilot)
        _add_entries(app, 250)
        app._flush_conversation_render()
        await _settle(pilot)
        _add_entries(app, 200, start=250)
        app._flush_conversation_render()
        await _settle(pilot)
        assert len(app._entry_widgets) == WINDOW_MAX

        view = app._conversation
        entries = app.conversation_entries
        anchor_entry = entries[451 - WINDOW_MAX]  # first mounted before expansion
        old_virtual_height = view.virtual_size.height

        view.scroll_to(y=0, animate=False)
        # Wait for the expansion compensation to adjust the scroll position.
        await _wait_for_layout(pilot, lambda: view.scroll_y > 0)

        assert len(app._entry_widgets) == WINDOW_MAX + EXPAND_CHUNK
        assert next(iter(app._entry_widgets)) == entries[451 - WINDOW_MAX - EXPAND_CHUNK].entry_id
        # Scroll compensation: scroll_y grew by exactly the added content height,
        # so the previously-first entry is still at the top of the viewport.
        assert view.scroll_y == pytest.approx(view.virtual_size.height - old_virtual_height, abs=1)
        anchor_widget = app._entry_widgets[anchor_entry.entry_id]
        assert 0 <= anchor_widget.region.y - view.content_region.y <= 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("wheel_during_prepend", "start_near_zero"),
    [(False, False), (True, False), (True, True)],
    ids=["single-wheel", "queued-wheel", "queued-wheel-at-zero"],
)
async def test_transcript_prepend_never_displays_an_unanchored_frame(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, wheel_during_prepend: bool, start_near_zero: bool
) -> None:
    app = _build_sub_agent_test_app(tmp_path, monkeypatch)

    async with app.run_test(size=(90, 24)) as pilot:
        await wait_for_layout(pilot, app._startup_complete.is_set)
        for index in range(240):
            kind = "thinking" if index % 2 else "user"
            app._add_conversation_entry(
                ConversationEntry(kind=kind, content=f"Entry {index}: " + "wrapped history " * (3 + index % 7))
            )
        app._render_conversation()
        view = app._conversation
        window = _window(app)
        await wait_for_layout(pilot, lambda: view.is_at_bottom() and not app._conversation_anchor_pending)
        old_start = window.mounted_start
        step = int(app.scroll_sensitivity_y)
        view.scroll_to(y=view.max_scroll_y // 2, animate=False, immediate=True)
        bar = app.query_one("#jump_to_bottom", JumpToBottomBar)
        await wait_for_layout(pilot, lambda: bar.display and bar.region.height and view.region.bottom <= bar.region.y)
        start_y = 1 if start_near_zero else 2 * view.outer_size.height + 2 * step - 1

        def anchor_at_start() -> str | None:
            visible = view.screen._compositor.visible_widgets
            for entry_id, widget in window.widgets.items():
                if widget in visible:
                    row = visible[widget][0].y - view.content_region.y
                    if 1 <= row <= 8 and widget.virtual_region.y - row == start_y:
                        return entry_id
            return None

        with monkeypatch.context() as setup:
            # Put the reader one line from the old top without triggering paging
            # during setup. The first input will exhaust the old scroll range.
            if start_near_zero:
                setup.setattr(app, "_maybe_expand_transcript_window", lambda: None)
            view.scroll_to(y=start_y, animate=False, immediate=True)
            await wait_for_layout(pilot, lambda: anchor_at_start() is not None)
        anchor_id = anchor_at_start()
        assert anchor_id is not None

        def row() -> int | None:
            geometry = view.screen._compositor.visible_widgets.get(window.widgets[anchor_id])
            return None if geometry is None else geometry[0].y - view.content_region.y

        initial_row = row()
        assert initial_row is not None
        before = initial_row
        if not start_near_zero:
            # First wheel remains above the paging threshold.
            assert view._scroll_up_for_pointer(animate=False)
            before += step
            await wait_for_layout(pilot, lambda: row() == before)
        assert window.mounted_start == old_start
        frames = []
        original_display = app._display

        def display(screen, renderable) -> None:
            if renderable is not None and not app._batch_count:
                frames.append(row())
            original_display(screen, renderable)

        monkeypatch.setattr(app, "_display", display)

        async def wheel_up() -> None:
            # Dispatch synchronously with respect to layout, including Textual's
            # default handler, so the second wheel is definitely in flight.
            await view._on_message(events.MouseScrollUp(view, 0, 0, 0, 0, 0, False, False, False))

        # Crossing the threshold mounts a page. Send a second wheel before
        # yielding to the layout pass, while those widgets are still pending.
        await wheel_up()
        assert window.mounted_start < old_start
        if start_near_zero:
            assert view.scroll_y == 0
        after_crossing = before + min(step, start_y)
        if wheel_during_prepend:
            await wheel_up()
        intended = before + step + (step if wheel_during_prepend else 0)
        await wait_for_layout(
            pilot, lambda: view.scroll_y > view.outer_size.height and bool(frames) and row() == frames[-1]
        )

        assert frames[-1] == intended, frames
        assert set(frames) <= {before, after_crossing, intended}, frames


@pytest.mark.asyncio
async def test_transcript_prepend_bounds_each_batch_and_services_queued_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = _build_sub_agent_test_app(tmp_path, monkeypatch)

    async with app.run_test(size=(90, 24)) as pilot:
        await wait_for_layout(pilot, app._startup_complete.is_set)
        _add_entries(app, 240)
        app._render_conversation()
        view = app._conversation
        window = _window(app)
        await wait_for_layout(pilot, lambda: view.is_at_bottom() and not app._conversation_anchor_pending)

        built: list[str] = []
        batches: list[tuple[str, ...]] = []
        make_widget = window._make_widget
        mount = view.mount

        def record_build(entry: ConversationEntry):
            built.append(entry.entry_id)
            return make_widget(entry)

        def record_mount(*widgets, **kwargs):
            batches.append(tuple(widget.entry.entry_id for widget in widgets))
            result = mount(*widgets, **kwargs)
            # Queue real input on the view's message pump before yielding, not
            # a direct scroll offset write after the batch has settled.
            assert view.post_message(events.MouseScrollDown(view, 0, 0, 0, 0, 0, False, False, False))
            return result

        monkeypatch.setattr(window, "_make_widget", record_build)
        monkeypatch.setattr(view, "mount", record_mount)
        expected = [entry.entry_id for entry in app.conversation_entries]
        step = app.scroll_sensitivity_y
        view.scroll_to(y=2 * view.outer_size.height + 1, animate=False, immediate=True)

        while window.mounted_start:
            old_start = window.mounted_start
            # The first row survives an upward swap even when it evicts the
            # opposite edge and net virtual height is unchanged.
            retained = next(iter(window.widgets.values()))
            old_y = retained.virtual_region.y
            built.clear()
            batches.clear()
            for _ in range(int(view.scroll_y / step) + 1):
                await view._on_message(events.MouseScrollUp(view, 0, 0, 0, 0, 0, False, False, False))
                if window.mounted_start < old_start:
                    break
            assert window.mounted_start < old_start
            assert 1 <= len(built) <= 12  # the per-input work budget, independent of the configured chunk
            assert batches == [tuple(built)]
            assert built == expected[window.mounted_start : old_start]
            crossing_y = view.scroll_y

            def queued_wheel_applied() -> bool:
                distance = retained.virtual_region.y - old_y
                return (
                    distance > 0 and view.scroll_y == crossing_y + distance + step and view._scrollback_anchor is None
                )

            await wait_for_layout(pilot, queued_wheel_applied)
            assert list(window.widgets) == expected[window.mounted_start : window.mounted_end]

        assert list(window.widgets)[0] == expected[0]


@pytest.mark.asyncio
async def test_transcript_history_stays_bounded_in_both_directions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = _build_sub_agent_test_app(tmp_path, monkeypatch)

    async with app.run_test(size=(90, 26)) as pilot:
        await wait_for_layout(pilot, app._startup_complete.is_set)
        for index in range(480):
            app._add_conversation_entry(
                ConversationEntry(
                    kind="thinking" if index % 7 == 0 else "user",
                    content=f"Entry {index}: " + "wrapped history " * (3 + index % 9),
                )
            )
        app._render_conversation()
        view = app._conversation
        window = _window(app)
        await wait_for_layout(pilot, lambda: view.is_at_bottom() and not app._conversation_anchor_pending)
        expected = [entry.entry_id for entry in app.conversation_entries]
        initial_count = len(expected)
        disclosure_id = expected[-4]  # thinking entry 476
        disclosure = window.widgets[disclosure_id].query_one(Collapsible)
        disclosure.collapsed = False
        await wait_for_layout(pilot, lambda: disclosure.region.height > 3 and view.is_at_bottom())

        async def page(direction: int) -> None:
            old_start = window.mounted_start
            old_end = window.mounted_end
            target_y = 0 if direction < 0 else view.max_scroll_y
            anchor = next(widget for widget in window.widgets.values() if widget.virtual_region.bottom > target_y)
            old_offset = anchor.virtual_region.y - target_y
            view.scroll_to(y=target_y, animate=False, immediate=True)
            if direction > 0 and window.mounted_end == initial_count and len(expected) == initial_count:
                # The last page is mounted in the DOM but the old bottom is
                # still published. New output must not restore follow here.
                assert view._scrollback_anchor is not None
                incoming = ConversationEntry(kind="user", content="arrived in flight")
                app._add_conversation_entry(incoming)
                expected.append(incoming.entry_id)
                app._flush_conversation_render()
                assert view.has_newer_entries and not view.auto_follow_bottom
                assert incoming.entry_id not in window.widgets

            def settled() -> bool:
                end = window.mounted_start + len(window.widgets)
                moved = window.mounted_start < old_start if direction < 0 else end > old_end
                return (
                    moved
                    and list(view.children) == list(window.widgets.values())
                    and all(widget.is_mounted and widget.region.height > 0 for widget in window.widgets.values())
                    and view._scrollback_anchor is None
                )

            await wait_for_layout(pilot, settled)
            start = window.mounted_start
            end = start + len(window.widgets)
            assert len(view.children) <= 160
            assert list(window.widgets) == expected[start:end]
            assert start < old_end and end > old_start
            assert anchor in window.widgets.values()
            assert anchor.region.y - view.content_region.y == old_offset

        while window.mounted_start:
            await page(-1)
        assert disclosure_id not in window.widgets and not disclosure.is_attached
        while window.mounted_end < len(expected):
            await page(1)
        assert not window.widgets[disclosure_id].query_one(Collapsible).collapsed

        view.scroll_to(y=view.max_scroll_y, animate=False, immediate=True)
        await wait_for_layout(pilot, lambda: view.auto_follow_bottom and list(window.widgets) == expected[-WINDOW_MAX:])
        assert view.is_at_bottom()


@pytest.mark.asyncio
async def test_transcript_wheel_to_latest_hides_jump_bar(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    app = _build_sub_agent_test_app(tmp_path, monkeypatch)

    async with app.run_test(size=(90, 26)) as pilot:
        await wait_for_layout(pilot, app._startup_complete.is_set)
        _add_entries(app, 250)
        app._render_conversation()
        view = app._conversation
        window = _window(app)
        bar = app.query_one("#jump_to_bottom", JumpToBottomBar)
        newest = app.conversation_entries[-1]
        await wait_for_layout(
            pilot, lambda: view.max_scroll_y > 0 and view.is_at_bottom() and not app._conversation_anchor_pending
        )

        while not view.has_newer_entries:
            start = window.mounted_start
            view.scroll_to(y=0, animate=False, immediate=True)
            await wait_for_layout(pilot, lambda: window.mounted_start < start and view._scrollback_anchor is None)
        assert bar.display and not view.auto_follow_bottom
        assert newest.entry_id not in window.widgets

        # Enter the last page on a wheel tick, then deliver the rest of the
        # burst before layout publishes that page's scroll range.
        view.scroll_to(y=view.max_scroll_y - 2 * view.outer_size.height - 1, animate=False, immediate=True)
        assert view.has_newer_entries and bar.display
        await view._on_message(events.MouseScrollDown(view, 0, 0, 0, 0, 0, False, False, False))
        assert view._scrollback_anchor is not None
        assert newest.entry_id in window.widgets
        for _ in range(2 * view.outer_size.height + 2 * EXPAND_CHUNK):
            await view._on_message(events.MouseScrollDown(view, 0, 0, 0, 0, 0, False, False, False))

        def latest_is_visible() -> bool:
            row = window.widgets.get(app.conversation_entries[-1].entry_id)
            return row is not None and row.region.overlaps(view.content_region)

        def at_latest() -> bool:
            return view.is_at_bottom() and view.scroll_y == view.max_scroll_y and latest_is_visible()

        await wait_for_layout(pilot, at_latest)
        await wait_for_layout(pilot, lambda: not bar.display and view.auto_follow_bottom)

        _add_entries(app, 1, start=250)
        app._flush_conversation_render()
        await wait_for_layout(pilot, at_latest)
        assert view.auto_follow_bottom and not bar.display


@pytest.mark.asyncio
async def test_transcript_jump_to_bottom_refollows_and_trims(tmp_path: Path, monkeypatch) -> None:
    app = _build_sub_agent_test_app(tmp_path, monkeypatch)

    async with app.run_test() as pilot:
        await _settle(pilot)
        _add_entries(app, 250)
        app._flush_conversation_render()
        await _settle(pilot)
        _add_entries(app, 200, start=250)
        app._flush_conversation_render()
        await _settle(pilot)

        view = app._conversation
        window = _window(app)
        tail = [entry.entry_id for entry in app.conversation_entries[-WINDOW_MAX:]]
        for _ in range(5):
            before = window.mounted_start
            view.scroll_to(y=0, animate=False, immediate=True)
            await wait_for_layout(pilot, lambda: window.mounted_start < before and view._scrollback_anchor is None)
        assert view.has_newer_entries  # Jump from a window that does not contain the tail.

        bar = app.query_one("#jump_to_bottom", JumpToBottomBar)
        app.on_jump_to_bottom_bar_pressed(JumpToBottomBar.Pressed(bar))
        await wait_for_layout(
            pilot, lambda: view.auto_follow_bottom and view.is_at_bottom() and not app._conversation_anchor_pending
        )
        assert list(window.widgets) == tail

        _add_entries(app, 150, start=450)
        app._flush_conversation_render()
        await _settle(pilot)
        assert len(app._entry_widgets) == WINDOW_MAX
        assert bar.display is False


@pytest.mark.asyncio
async def test_restore_mounts_only_the_trailing_window(tmp_path: Path, monkeypatch) -> None:
    app = _build_sub_agent_test_app(tmp_path, monkeypatch)

    async with app.run_test() as pilot:
        await _settle(pilot)
        history = [
            {"role": "user" if index % 2 == 0 else "assistant", "content": f"turn {index}"} for index in range(700)
        ]
        app._restore_conversation_history(history)
        await _settle(pilot)

        total = len(app.conversation_entries)
        assert total == 701  # 700 + startup entry
        assert len(app._entry_widgets) == WINDOW_MAX
        assert _window(app).mounted_start == total - WINDOW_MAX


class _LateLayoutView:
    """Inspector-view double whose mounted height lags one refresh behind.

    Forwards every attribute to the real view, but serves the height recorded by
    :meth:`hide_next_growth` to the compensation read that follows the expansion:
    what a loaded event loop produces when the idle pass runs the compensation
    before layout publishes the rows just mounted.
    """

    def __init__(self, view: VerticalScroll) -> None:
        self._view = view
        self._stale_height: int | None = None
        self._fresh_reads = 0

    def __getattr__(self, name: str) -> Any:
        return getattr(self._view, name)

    @property
    def virtual_size(self) -> Size:
        size = self._view.virtual_size
        if self._stale_height is None:
            return size
        if self._fresh_reads > 0:
            self._fresh_reads -= 1
            return size
        stale, self._stale_height = self._stale_height, None
        return Size(size.width, stale)

    def hide_next_growth(self) -> None:
        """Report the pre-expansion height to the next compensation attempt."""
        self._stale_height = self._view.virtual_size.height
        self._fresh_reads = 1


@pytest.mark.asyncio
async def test_expansion_compensates_when_the_first_height_read_is_stale(tmp_path: Path, monkeypatch) -> None:
    app = _build_sub_agent_test_app(tmp_path, monkeypatch)

    async with app.run_test() as pilot:
        app._render_event(_sub_agent_event(uuid="s0", text="start"))
        activity = app._sub_agent_activities["agent-1"]
        for index in range(350):
            activity.steps.append(ConversationEntry(kind="user", content=f"step {index}", complete=True))
        app.action_open_sub_agent()
        screen = app._sub_agent_inspector
        assert screen is not None
        await wait_for_layout(pilot, lambda: len(screen._step_widgets) == INSPECTOR_MAX)
        view = screen.query_one("#inspector_trajectory", TrajectoryScrollView)
        await wait_for_layout(pilot, lambda: view.scroll_y > 0)
        window = screen._get_trajectory_window(view)
        late_layout = _LateLayoutView(view)
        monkeypatch.setattr(window, "_view", cast(VerticalScroll, late_layout))

        while window.mounted_start > 0:
            before = window.mounted_start
            late_layout.hide_next_growth()
            view.scroll_to(y=0, animate=False, immediate=True)
            # Without a compensating scroll the viewport stays on the top row,
            # where scrolling up again changes nothing and expansion stops for good.
            await wait_for_layout(pilot, lambda: window.mounted_start < before and view.scroll_y > 0)

        assert list(window.widgets) == [step.entry_id for step in activity.steps]


@pytest.mark.asyncio
async def test_mid_list_removal_rebuilds_the_window(tmp_path: Path, monkeypatch) -> None:
    app = _build_sub_agent_test_app(tmp_path, monkeypatch)

    async with app.run_test() as pilot:
        await _settle(pilot)
        _add_entries(app, 350)
        app._flush_conversation_render()
        await _settle(pilot)
        assert len(app._entry_widgets) == WINDOW_MAX

        # Queued-message restore removes entries from the middle/end of the list.
        del app.conversation_entries[5]
        del app.conversation_entries[10]
        app._invalidate_conversation()
        app._flush_conversation_render()
        await _settle(pilot)

        entries = app.conversation_entries
        assert len(entries) == 349
        assert len(app._entry_widgets) == WINDOW_MAX
        assert list(app._entry_widgets) == [entry.entry_id for entry in entries[-WINDOW_MAX:]]


class _BareModal(ModalScreen):
    def compose(self) -> ComposeResult:
        yield Static("cover")


@pytest.mark.asyncio
async def test_transcript_sync_defers_while_modal_covers_screen(tmp_path: Path, monkeypatch) -> None:
    app = _build_sub_agent_test_app(tmp_path, monkeypatch)

    async with app.run_test() as pilot:
        await _settle(pilot)
        _add_entries(app, 5)
        app._flush_conversation_render()
        await _settle(pilot)
        base_count = len(app._entry_widgets)

        app._push_fullscreen_modal(_BareModal())
        await _settle(pilot)
        assert app._modal_cover_active is True

        _add_entries(app, 3, start=5)
        app._flush_conversation_render()
        await _settle(pilot)

        assert len(app._entry_widgets) == base_count  # hidden screen untouched
        assert app._transcript_sync_pending is True

        window = _window(app)
        rebuild_count = 0
        sync_count = 0
        original_rebuild = window.rebuild
        original_sync = window.sync

        def count_rebuild(entries) -> None:
            nonlocal rebuild_count
            rebuild_count += 1
            original_rebuild(entries)

        def count_sync(entries, dirty_ids, *, follow_bottom: bool) -> None:
            nonlocal sync_count
            sync_count += 1
            original_sync(entries, dirty_ids, follow_bottom=follow_bottom)

        monkeypatch.setattr(window, "rebuild", count_rebuild)
        monkeypatch.setattr(window, "sync", count_sync)
        monkeypatch.setattr(app, "_render_coalesce_interval", lambda _entry: 60.0)

        # Leave a coalesced flush pending when the modal closes. The dismissal
        # callback must neutralize that timer and use one full rebuild to catch up.
        app._add_conversation_entry(ConversationEntry(kind="user", content="pending at dismiss"))
        assert app._render_pending is True
        app.screen.dismiss()
        await _wait_for_layout(pilot, lambda: not app._transcript_sync_pending)

        assert app._transcript_sync_pending is False
        assert app._render_pending is False
        assert len(app._entry_widgets) == base_count + 4
        assert rebuild_count == 1
        assert sync_count == 0

        # Simulate the stale timer firing after the post-dismiss rebuild.
        app._flush_conversation_render()
        assert rebuild_count == 1
        assert sync_count == 0


@pytest.mark.asyncio
async def test_modal_dismiss_promotes_an_unflushed_render_to_full_sync(tmp_path: Path, monkeypatch) -> None:
    app = _build_sub_agent_test_app(tmp_path, monkeypatch)

    async with app.run_test() as pilot:
        await _settle(pilot)
        # A coalesced flush scheduled before the modal opens fires while the modal
        # covers the transcript and sets the deferred-sync marker, so the
        # experiment would begin after a resync already happened. Drain the
        # pending render, then freeze the cadence so nothing flushes under the
        # modal.
        await _wait_for_layout(pilot, lambda: not app._render_pending)
        base_count = len(app._entry_widgets)
        monkeypatch.setattr(app, "_render_coalesce_interval", lambda _entry: 60.0)

        app._push_fullscreen_modal(_BareModal())
        await _settle(pilot)
        assert app._modal_cover_active is True

        app._add_conversation_entry(ConversationEntry(kind="user", content="last-moment update"))
        assert app._render_pending is True
        assert app._transcript_sync_pending is False

        app.screen.dismiss()
        await _wait_for_layout(pilot, lambda: len(app._entry_widgets) == base_count + 1)

        assert app._render_pending is False
        assert app._transcript_sync_pending is False


@pytest.mark.asyncio
async def test_sub_agent_inspector_defers_base_transcript_then_resyncs(tmp_path: Path, monkeypatch) -> None:
    app = _build_sub_agent_test_app(tmp_path, monkeypatch)

    async with app.run_test() as pilot:
        app._render_event(_sub_agent_event(uuid="r1", text="working"))
        app._flush_conversation_render()
        await _settle(pilot)
        base_count = len(app._entry_widgets)

        app.action_open_sub_agent()
        await _settle(pilot)
        assert app._modal_cover_active is True

        # Live activity while covered: the card mutates and a user entry arrives.
        app._render_event(_sub_agent_event(uuid="r2", text="still working"))
        app._add_conversation_entry(ConversationEntry(kind="user", content="while open"))
        app._flush_conversation_render()
        await _settle(pilot)

        assert len(app._entry_widgets) == base_count
        assert app._transcript_sync_pending is True

        screen = app._sub_agent_inspector
        assert screen is not None
        screen.action_close()
        await _wait_for_layout(pilot, lambda: not app._transcript_sync_pending)

        assert app._transcript_sync_pending is False
        assert len(app._entry_widgets) == base_count + 1


@pytest.mark.asyncio
async def test_inspector_trajectory_mounts_only_trailing_window(tmp_path: Path, monkeypatch) -> None:
    app = _build_sub_agent_test_app(tmp_path, monkeypatch)

    async with app.run_test() as pilot:
        app._render_event(_sub_agent_event(uuid="s0", text="start"))
        activity = app._sub_agent_activities["agent-1"]
        for index in range(400):
            activity.steps.append(ConversationEntry(kind="assistant", content=f"step {index}", complete=True))

        app.action_open_sub_agent()
        await _settle(pilot)
        screen = app._sub_agent_inspector
        assert screen is not None

        total_steps = len(activity.steps)
        assert total_steps > INSPECTOR_MAX
        assert len(screen._step_widgets) == INSPECTOR_MAX
        # The mounted window is the trajectory tail.
        assert list(screen._step_widgets) == [step.entry_id for step in activity.steps[-INSPECTOR_MAX:]]

        # Scrolling the trajectory to the top mounts older steps.
        view = screen.query_one("#inspector_trajectory", TrajectoryScrollView)
        view.scroll_to(y=0, animate=False)
        await _settle(pilot, rounds=5)

        assert len(screen._step_widgets) == INSPECTOR_MAX + theme.INSPECTOR_WINDOW_EXPAND_CHUNK
        assert (
            list(screen._step_widgets)[0]
            == activity.steps[-(INSPECTOR_MAX + theme.INSPECTOR_WINDOW_EXPAND_CHUNK)].entry_id
        )


@pytest.mark.asyncio
async def test_inspector_switching_agents_rebuilds_windowed(tmp_path: Path, monkeypatch) -> None:
    app = _build_sub_agent_test_app(tmp_path, monkeypatch)

    async with app.run_test() as pilot:
        app._render_event(_sub_agent_event(agent_id="a1", uuid="u1", text="one"))
        app._render_event(_sub_agent_event(agent_id="a2", parent_tool_call_id="tc-2", uuid="u2", text="two"))
        for key in ("a1", "a2"):
            activity = app._sub_agent_activities[key]
            for index in range(300):
                activity.steps.append(ConversationEntry(kind="assistant", content=f"{key} step {index}", complete=True))

        app.action_open_sub_agent(key="a1")
        await _settle(pilot)
        screen = app._sub_agent_inspector
        assert screen is not None
        assert len(screen._step_widgets) == INSPECTOR_MAX

        screen._selected_key = "a2"
        screen._select_changed()
        await _settle(pilot)

        activity_a2 = app._sub_agent_activities["a2"]
        assert len(screen._step_widgets) == INSPECTOR_MAX
        assert list(screen._step_widgets) == [step.entry_id for step in activity_a2.steps[-INSPECTOR_MAX:]]


@pytest.mark.asyncio
async def test_inspector_refreshes_only_dirty_steps_in_a_paged_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = _build_sub_agent_test_app(tmp_path, monkeypatch)

    async with app.run_test(size=(100, 35)) as pilot:
        app._render_event(_sub_agent_event(status="GENERATING"))
        activity = app._sub_agent_activities["agent-1"]
        for index in range(400):
            if index in (200, 201):
                app._render_event(_sub_agent_delta("start", uuid=f"reply-{index}", complete=False))
            else:
                activity.steps.append(ConversationEntry(kind="user", content=f"step {index}"))
        app.action_open_sub_agent()
        screen = app._sub_agent_inspector
        assert screen is not None
        await wait_for_layout(pilot, lambda: bool(screen.query(TrajectoryScrollView)))
        view = screen.query_one(TrajectoryScrollView)
        await wait_for_layout(
            pilot,
            lambda: (
                len(screen._step_widgets) == INSPECTOR_MAX and view.max_scroll_y > 0 and view.is_vertical_scroll_end
            ),
        )
        window = screen._trajectory_window
        assert window is not None
        view.scroll_to(y=0, animate=False, immediate=True)
        await wait_for_layout(
            pilot,
            lambda: (
                len(screen._step_widgets) == INSPECTOR_MAX + theme.INSPECTOR_WINDOW_EXPAND_CHUNK
                and not window._expand_pending
                and 0 < view.scroll_y < view.max_scroll_y
            ),
        )
        first = activity.stream_steps["reply-200"]
        second = activity.stream_steps["reply-201"]
        rows = dict(window.widgets)
        reader = next(
            row
            for row in rows.values()
            if row.region.y >= view.content_region.y and row.region.overlaps(view.content_region)
        )
        old_offset = reader.region.y - view.content_region.y
        refreshes: list[str] = []
        syncing = False
        sync_trajectory = screen._sync_trajectory
        refresh_content = ConversationEntryWidget.refresh_content

        def count_sync() -> None:
            nonlocal syncing
            syncing = True
            try:
                sync_trajectory()
            finally:
                syncing = False

        def count_refresh(row: ConversationEntryWidget) -> None:
            if syncing and row.entry.entry_id in rows:
                refreshes.append(row.entry.entry_id)
            refresh_content(row)

        monkeypatch.setattr(screen, "_sync_trajectory", count_sync)
        monkeypatch.setattr(ConversationEntryWidget, "refresh_content", count_refresh)
        app._render_event(_sub_agent_delta(" one", uuid="reply-200", complete=False))
        assert screen._flush_pending
        await wait_for_layout(pilot, lambda: not screen._flush_pending)
        assert len(refreshes) == 1
        assert refreshes == [first.entry_id]
        assert rows[first.entry_id] is window.widgets[first.entry_id]
        assert first.content == "start one"

        refreshes.clear()
        app._render_event(_sub_agent_event(status="GENERATING", total_tokens=100))
        assert screen._flush_pending
        await wait_for_layout(pilot, lambda: not screen._flush_pending)
        assert refreshes == []

        app._render_event(_sub_agent_delta(" two", uuid="reply-200"))
        app._render_event(_sub_agent_delta(" more", uuid="reply-201", complete=False))
        app._render_event(_sub_agent_delta(" done", uuid="reply-201"))
        app._render_event(_sub_agent_delta("new tail", uuid="new"))
        appended = activity.stream_steps["new"]
        await wait_for_layout(
            pilot,
            lambda: (
                not screen._flush_pending
                and appended.entry_id in window.widgets
                and window.widgets[appended.entry_id].size.height > 0
            ),
        )
        assert sorted(refreshes) == sorted([first.entry_id, second.entry_id])
        assert first.content == "start one two"
        assert second.content == "start more done"
        assert window.widgets[reader.entry.entry_id] is reader
        assert reader.region.y - view.content_region.y == old_offset

        view.scroll_end(animate=False, immediate=True)
        await wait_for_layout(pilot, lambda: view.is_vertical_scroll_end)
        app._render_event(_sub_agent_delta("followed tail", uuid="followed"))
        followed = activity.stream_steps["followed"]
        await wait_for_layout(
            pilot,
            lambda: (
                followed.entry_id in window.widgets
                and window.widgets[followed.entry_id].region.overlaps(view.content_region)
                and view.is_vertical_scroll_end
            ),
        )


@pytest.mark.asyncio
async def test_inspector_roster_skips_unchanged_rows(tmp_path: Path, monkeypatch) -> None:
    app = _build_sub_agent_test_app(tmp_path, monkeypatch)

    async with app.run_test() as pilot:
        app._render_event(_sub_agent_event(agent_id="a1", uuid="u1", text="running"))
        app._render_event(
            _sub_agent_event(agent_id="a2", parent_tool_call_id="tc-2", status="STOPPED", message="Completed")
        )

        app.action_open_sub_agent()
        await _settle(pilot)
        screen = app._sub_agent_inspector
        assert screen is not None
        screen._flush()  # settle: roster/header caches reflect current state

        updates: list[str] = []
        monkeypatch.setattr(SubAgentRosterRow, "update", lambda self, content: updates.append(self.key))
        screen._on_tick()

        # Only the still-running agent re-renders (spinner frame); the finished
        # agent's row is identical and skipped.
        assert set(updates) == {"a1"}


@pytest.mark.asyncio
async def test_turn_status_strip_skips_identical_content(tmp_path: Path, monkeypatch) -> None:
    app = _build_sub_agent_test_app(tmp_path, monkeypatch)

    async with app.run_test() as pilot:
        await _settle(pilot)
        strip = app._turn_status
        updates: list[tuple[str, bool]] = []
        monkeypatch.setattr(strip, "update", lambda content="", *, layout=True: updates.append((content, layout)))

        app._turn_final_text = "Done in 5s"
        app._turn_final_state = tui_state.TurnState.IDLE
        app._refresh_turn_status_strip()
        app._refresh_turn_status_strip()

        # First call updates (with layout=False); the identical second call is skipped.
        assert len(updates) == 1
        assert updates[0][1] is False
