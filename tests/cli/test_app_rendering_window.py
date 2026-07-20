# ruff: noqa: F401,F811,E402
"""Scrollback-window behavior: bounded mounted widgets on large transcripts.

The transcript and the sub-agent inspector mount only a trailing window of their
entries so Textual reflows stay O(window) no matter how long a session runs.
These tests cover trimming while following the bottom, history expansion on
scroll-up, windowed restores, modal-cover deferral, and the inspector's window.
"""

from pathlib import Path
import time

import pytest
from textual.app import ComposeResult
from textual.screen import ModalScreen
from textual.widgets import Static

from kolega_code.cli import theme
from kolega_code.cli.tui import state as tui_state
from kolega_code.cli.tui.state import ConversationEntry
from kolega_code.cli.tui.sub_agent_screen import SubAgentRosterRow
from kolega_code.cli.tui.widgets import JumpToBottomBar, TrajectoryScrollView

from ._app_test_utils import _build_sub_agent_test_app, _sub_agent_event

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
        # Stage 1: below the cap, everything mounts.
        _add_entries(app, 250)
        app._flush_conversation_render()
        await _settle(pilot)
        assert len(app._entry_widgets) == 251  # 250 + startup entry

        # Stage 2: crossing max + trim chunk trims the oldest back to the cap.
        _add_entries(app, 200, start=250)
        app._flush_conversation_render()
        await _settle(pilot)

        entries = app.conversation_entries
        assert len(entries) == 451  # data model keeps everything
        assert len(app._entry_widgets) == WINDOW_MAX
        assert _window(app).mounted_start == 451 - WINDOW_MAX
        first_mounted = next(iter(app._entry_widgets))
        assert first_mounted == entries[451 - WINDOW_MAX].entry_id
        assert entries[0].entry_id not in app._entry_widgets


@pytest.mark.asyncio
async def test_transcript_does_not_trim_while_scrolled_up(tmp_path: Path, monkeypatch) -> None:
    app = _build_sub_agent_test_app(tmp_path, monkeypatch)

    async with app.run_test() as pilot:
        await _settle(pilot)
        _add_entries(app, 250)
        app._flush_conversation_render()
        await _settle(pilot)

        view = app._conversation
        view.scroll_to(y=0, animate=False)
        await _settle(pilot)
        assert view.auto_follow_bottom is False

        _add_entries(app, 200, start=250)
        app._flush_conversation_render()
        await _settle(pilot)

        # Nothing trimmed: the user is reading history, so the window only grows.
        assert len(app._entry_widgets) == 451
        assert _window(app).mounted_start == 0


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
        view.scroll_to(y=0, animate=False)
        await _settle(pilot)
        expanded = len(app._entry_widgets)
        assert expanded > WINDOW_MAX  # expansion grew the window while reading

        bar = app.query_one("#jump_to_bottom", JumpToBottomBar)
        app.on_jump_to_bottom_bar_pressed(JumpToBottomBar.Pressed(bar))
        await _settle(pilot)
        assert view.auto_follow_bottom is True

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

        # Scrolling to the top repeatedly expands until the whole history is mounted.
        view = app._conversation
        for _ in range(10):
            if _window(app).mounted_start == 0:
                break
            view.scroll_to(y=0, animate=False)
            await _settle(pilot)

        assert _window(app).mounted_start == 0
        assert len(app._entry_widgets) == total


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
        base_count = len(app._entry_widgets)

        app._push_fullscreen_modal(_BareModal())
        await _settle(pilot)
        assert app._modal_cover_active is True

        # Keep the coalesced flush from firing before dismissal, so no earlier
        # modal-covered flush can set the deferred-sync marker.
        monkeypatch.setattr(app, "_render_coalesce_interval", lambda _entry: 60.0)
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
