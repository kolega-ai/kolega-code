"""The background strip belongs to the session, not a turn or transcript."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest
from rich.text import Text
from textual.screen import ModalScreen
from textual.widgets import Static
from textual.widgets._footer import FooterKey

from kolega_code.cli.tui.shortcut_bar import ContextFooter
from kolega_code.cli.tui.state import ConversationEntry, TurnState
from kolega_code.cli.tui.widgets import ChatComposer
from kolega_code.cli.session_journal import SessionJournalError
from kolega_code.session.background import BackgroundAgentManager

from ._app_test_utils import wait_for_turn_idle
from .test_app_background_jobs import BackgroundHarnessAgent, _build_background_app, _wait_for

pytestmark = [pytest.mark.asyncio, pytest.mark.usefixtures("isolated_cli_env")]


def _plain(app: Any) -> str:
    return Text.from_markup(app._turn_status_content(width=app._turn_status.content_size.width or None)).plain


async def _start_job(manager: BackgroundAgentManager, job_id: str, gate: asyncio.Event) -> None:
    started = asyncio.Event()

    async def child() -> str:
        started.set()
        await gate.wait()
        return f"Report for {job_id}"

    await manager.submit(job_id, child, agent_type="investigation", task=f"Investigate {job_id}")
    await asyncio.wait_for(started.wait(), 3)


async def test_idle_strip_survives_parent_turns_scroll_and_preserves_draft(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app, _, _ = _build_background_app(tmp_path, monkeypatch)
    gate = asyncio.Event()
    async with app.run_test(size=(110, 32)) as pilot:
        agent = app.agent
        assert isinstance(agent, BackgroundHarnessAgent)
        manager = app._background_jobs
        assert isinstance(manager, BackgroundAgentManager)
        await _start_job(manager, "persistent", gate)
        await _wait_for(pilot, lambda: "1 background agent" in _plain(app), description="idle agent strip")
        strip = app._turn_status
        assert strip.display
        assert app._turn_timer is None
        assert app._background_status_timer is not None
        assert "Esc" not in _plain(app)
        composer = app.query_one(ChatComposer)
        composer.load_text("keep this draft")
        composer.focus()
        for index in range(80):
            app._add_conversation_entry(ConversationEntry(kind="assistant", content=f"History {index}\n" * 3))
        view = app.query_one("#conversation")
        await _wait_for(pilot, lambda: view.max_scroll_y > 10, description="scrollable transcript")
        view.scroll_home(animate=False)
        await _wait_for(pilot, lambda: view.scroll_y == 0, description="manual scrollback")
        await pilot.pause(0.3)
        assert app._turn_status is strip
        assert strip.display and strip.region.height == 1
        assert strip.region.bottom <= composer.region.y
        assert app.focused is composer
        assert composer.text == "keep this draft"
        assert view.scroll_y == 0
        assert any(key.key == "ctrl+g" for key in app.query_one(ContextFooter).query(FooterKey))

        agent.turn_gate = asyncio.Event()
        app.agent_worker = app.run_worker(app._process_message("Follow up while child runs"), exit_on_error=False)
        await asyncio.wait_for(agent.turn_started.wait(), 3)
        await _wait_for(pilot, lambda: "Esc" in _plain(app), description="foreground and background strip")
        assert "1 background agent" in _plain(app)
        assert app._turn_timer is not None
        assert app._background_status_timer is None
        agent.turn_gate.set()
        await wait_for_turn_idle(app, pilot)
        await _wait_for(pilot, lambda: app._background_status_timer is not None, description="idle animation")
        assert "1 background agent" in _plain(app)
        assert "Done in" not in _plain(app)
        assert "Esc" not in _plain(app)
        gate.set()
        await asyncio.wait_for(agent.continued.wait(), 3)
        await wait_for_turn_idle(app, pilot)
        await _wait_for(pilot, lambda: not app._background_status().visible, description="consumed success")
        assert "Done in" in _plain(app)
        assert app._background_status_timer is None


async def test_only_background_text_is_clickable_and_inspector_keeps_live_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app, _, _ = _build_background_app(tmp_path, monkeypatch)
    gate = asyncio.Event()
    async with app.run_test(size=(120, 32)) as pilot:
        agent = app.agent
        assert isinstance(agent, BackgroundHarnessAgent)
        manager = app._background_jobs
        assert isinstance(manager, BackgroundAgentManager)
        await _start_job(manager, "clickable", gate)
        agent.turn_gate = asyncio.Event()
        app.agent_worker = app.run_worker(app._process_message("Foreground label"), exit_on_error=False)
        await asyncio.wait_for(agent.turn_started.wait(), 3)
        await _wait_for(
            pilot,
            lambda: "Ctrl+G" in _plain(app) and app._turn_status.content_size.width > 20,
            description="click target",
        )
        text = Text.from_markup(app._turn_status_content(width=app._turn_status.content_size.width))
        start = next(i for i in range(len(text)) if text.get_style_at_offset(app.console, i).meta.get("@click"))
        assert start > len("Foreground label")
        assert all(not text.get_style_at_offset(app.console, i).meta for i in range(start))
        await pilot.click(app._turn_status, offset=(2, 0))
        assert app._sub_agent_inspector is None
        await pilot.click(app._turn_status, offset=(start + 2, 0))
        await _wait_for(pilot, lambda: app._sub_agent_inspector is not None, description="clicked inspector")
        screen = app._sub_agent_inspector
        assert screen is not None
        assert screen._selected_key == "clickable"
        assert app._sub_agent_activities["clickable"].status == "running"
        agent.turn_gate.set()
        await wait_for_turn_idle(app, pilot)
        agent.continue_gate = asyncio.Event()
        gate.set()
        await asyncio.wait_for(agent.continuation_started.wait(), 3)
        assert app._sub_agent_inspector is screen
        await _wait_for(
            pilot,
            lambda: any(
                step.content == "Report for clickable" for step in app._sub_agent_activities["clickable"].steps
            ),
            description="retained report in live inspector",
        )
        assert app._sub_agent_activities["clickable"].status == "completed"
        assert sum(step.content == "Report for clickable" for step in app._sub_agent_activities["clickable"].steps) == 1
        agent.continue_gate.set()
        await wait_for_turn_idle(app, pilot)
        assert app._sub_agent_inspector is screen
        await pilot.press("escape")
        await _wait_for(pilot, lambda: app._sub_agent_inspector is None, description="close inspector")
        assert app.focused is app.query_one(ChatComposer)


async def test_pending_results_and_failure_acknowledgment_are_per_presented_completion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app, _, _ = _build_background_app(tmp_path, monkeypatch)
    async with app.run_test(size=(110, 32)) as pilot:
        manager = app._background_jobs
        assert isinstance(manager, BackgroundAgentManager)
        app._background_paused = True

        async def fail() -> str:
            raise RuntimeError("selected child failed")

        for key in ("failure-one", "failure-two"):
            await manager.submit(key, fail, agent_type="general", task=key)
        await manager.wait()
        await _wait_for(
            pilot,
            lambda: all(row["delivered"] for row in manager.snapshot()) and app._background_status().failed == 2,
            description="delivered failures retained",
        )
        assert app._background_status_timer is None
        assert "2 failed agents" in _plain(app)
        app.action_open_sub_agent("failure-one")
        screen = app._sub_agent_inspector
        assert screen is not None
        # Opening chrome must not acknowledge an unseen result.
        assert not app._background_acknowledged_failures
        first_id = manager.snapshot("failure-one")[0]["completion_id"]
        second_id = manager.snapshot("failure-two")[0]["completion_id"]
        await _wait_for(
            pilot, lambda: first_id in app._background_acknowledged_failures, description="selected result painted"
        )
        assert second_id not in app._background_acknowledged_failures
        assert app._background_status().failed == 1
        assert any("selected child failed" in step.content for step in app._sub_agent_activities["failure-one"].steps)
        app._acknowledge_background_failure("failure-two", completion_id=first_id)
        assert second_id not in app._background_acknowledged_failures
        screen._selected_key = "failure-two"
        screen._select_changed()
        await _wait_for(
            pilot, lambda: second_id in app._background_acknowledged_failures, description="second result painted"
        )
        assert not app._background_status().visible
        await pilot.press("escape")


@pytest.mark.parametrize("width", [1, 12, 24, 32, 40, 80])
@pytest.mark.parametrize("foreground", [False, True])
async def test_narrow_unicode_strip_remains_single_line_and_keeps_shortcut_when_it_fits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, width: int, foreground: bool
) -> None:
    app, _, _ = _build_background_app(tmp_path, monkeypatch)
    gate = asyncio.Event()
    async with app.run_test(size=(width, 30)) as pilot:
        app._set_sidebar_visible(False)
        manager = app._background_jobs
        assert isinstance(manager, BackgroundAgentManager)
        started = asyncio.Event()

        async def child() -> str:
            started.set()
            await gate.wait()
            return "done"

        await manager.submit(
            "unicode", child, agent_type="general", task="Indexing [bold]測試 e\u0301[/bold]\nsecond line " * 20
        )
        await asyncio.wait_for(started.wait(), 3)
        if foreground:
            app._begin_turn_progress("Foreground [bold]測試 e\u0301[/bold]\n" * 20)
        await _wait_for(pilot, lambda: app._turn_status.display, description="narrow strip")
        content = Text.from_markup(app._turn_status_content(width=width))
        assert content.cell_len <= width
        assert "\n" not in content.plain and "\r" not in content.plain
        if width >= 12:
            assert "Ctrl+G" in content.plain
        assert len(app._turn_status.children) == 0
        if foreground:
            app._finish_turn_progress("Finished", TurnState.IDLE)


async def test_idle_timer_is_cached_no_reflow_and_honors_reduced_motion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app, _, _ = _build_background_app(tmp_path, monkeypatch)
    gate = asyncio.Event()
    async with app.run_test(size=(100, 30)) as pilot:
        manager = app._background_jobs
        assert isinstance(manager, BackgroundAgentManager)
        await _start_job(manager, "cached", gate)
        await _wait_for(pilot, lambda: app._background_status_timer is not None, description="idle timer")
        strip = app._turn_status
        await _wait_for(
            pilot, lambda: strip.region.height == 1 and strip.content_size.width > 0, description="strip layout"
        )
        snapshot = Mock(wraps=manager.status_snapshot)
        update = Mock(wraps=strip.update)
        monkeypatch.setattr(manager, "status_snapshot", snapshot)
        monkeypatch.setattr(strip, "update", update)
        before = (strip.region, app.query_one(ChatComposer).region)
        for _ in range(5):
            app._refresh_turn_status_strip()
        assert snapshot.call_count == 0
        assert update.call_count > 0
        assert all(call.kwargs == {"layout": False} for call in update.call_args_list)
        await pilot.pause(0.3)
        assert (strip.region, app.query_one(ChatComposer).region) == before
        app.animation_level = "none"
        app._refresh_turn_status_strip()
        assert app._background_status_timer is None
        static = app._turn_status_content(width=90)
        app._refresh_turn_status_strip()
        assert app._turn_status_content(width=90) == static
        assert "1 background agent" in _plain(app)
        app.animation_level = "full"
        app._refresh_turn_status_strip()
        assert app._background_status_timer is not None
    assert app._background_status_timer is None
    assert manager._on_changed is None


async def test_queued_inspector_rows_follow_manager_transitions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app, _, _ = _build_background_app(tmp_path, monkeypatch)
    gate = asyncio.Event()
    async with app.run_test(size=(100, 30)) as pilot:
        old = app._background_jobs
        assert isinstance(old, BackgroundAgentManager)
        await old.close()
        manager = BackgroundAgentManager(max_running=1)
        app._background_jobs = manager
        await app._attach_background_jobs()
        await _start_job(manager, "running", gate)
        queued_started = asyncio.Event()
        queued_gate = asyncio.Event()

        async def queued() -> str:
            queued_started.set()
            await queued_gate.wait()
            return "queued report"

        await manager.submit("queued", queued, agent_type="general", task="Queued work")
        await _wait_for(pilot, lambda: app._background_status().queued == 1, description="queued count")
        assert "1 running" in _plain(app) and "1 queued" in _plain(app)
        app.action_open_sub_agent("queued")
        await _wait_for(pilot, lambda: app._sub_agent_inspector is not None, description="queued inspector")
        assert app._sub_agent_activities["queued"].status == "queued"
        gate.set()
        await asyncio.wait_for(queued_started.wait(), 3)
        await _wait_for(
            pilot, lambda: app._sub_agent_activities["queued"].status == "running", description="queued job running"
        )
        assert app._sub_agent_activities["queued"].current_action != "Queued"
        queued_gate.set()
        await manager.wait()
        await _wait_for(
            pilot, lambda: app._sub_agent_activities["queued"].status == "completed", description="queued job finished"
        )
        assert any(step.content == "queued report" for step in app._sub_agent_activities["queued"].steps)


async def test_deferred_refresh_reads_successor_and_ignores_retired_callbacks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app, _, _ = _build_background_app(tmp_path, monkeypatch)
    async with app.run_test(size=(100, 30)) as pilot:
        old = app._background_jobs
        assert isinstance(old, BackgroundAgentManager)
        callback = old._on_changed
        assert callback is not None
        callback()
        successor = BackgroundAgentManager()
        app._background_jobs = successor
        await app._attach_background_jobs()
        await _wait_for(pilot, lambda: not app._background_status_refresh_pending, description="successor refresh")
        assert app._background_status_owner is successor
        assert not app._background_status().visible
        callback()
        assert not app._background_status_refresh_pending
        await old.close()


@pytest.mark.parametrize("terminal", [False, True])
async def test_static_status_reformats_on_resize_without_polling_jobs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, terminal: bool
) -> None:
    app, _, _ = _build_background_app(tmp_path, monkeypatch)
    gate = asyncio.Event()
    async with app.run_test(size=(140, 32)) as pilot:
        app._set_sidebar_visible(False)
        app.animation_level = "none"
        app._background_paused = True
        manager = app._background_jobs
        assert isinstance(manager, BackgroundAgentManager)

        async def child() -> str:
            await gate.wait()
            raise RuntimeError("Retained resize failure")

        await manager.submit("resize", child, agent_type="general", task="A long background task " * 10)
        if terminal:
            gate.set()
            await manager.wait()
        await _wait_for(
            pilot,
            lambda: (
                app._turn_status.region.width > 100
                and (
                    app._background_status().failed == 1
                    and manager.snapshot()[0]["delivered"]
                    and app.agent_worker is None
                    and not app._background_status_refresh_pending
                    if terminal
                    else app._background_status().running == 1
                )
            ),
            description="wide static strip",
        )
        assert app._background_status_timer is None
        snapshot = Mock(wraps=manager.status_snapshot)
        monkeypatch.setattr(manager, "status_snapshot", snapshot)
        snapshot.reset_mock()
        await pilot.resize_terminal(26, 32)
        await _wait_for(
            pilot,
            lambda: (
                app._turn_status.content_size.width == 24
                and app._last_turn_status_content == app._turn_status_content(width=24)
            ),
            description="resized cached status",
        )
        assert "Ctrl+G" in Text.from_markup(app._last_turn_status_content or "").plain
        assert snapshot.call_count == 0
        assert app._background_status_timer is None


async def test_covered_inspector_cannot_acknowledge_an_unseen_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app, _, _ = _build_background_app(tmp_path, monkeypatch)
    gate = asyncio.Event()
    async with app.run_test(size=(110, 32)) as pilot:
        manager = app._background_jobs
        assert isinstance(manager, BackgroundAgentManager)
        app._background_paused = True

        async def fail() -> str:
            await gate.wait()
            raise RuntimeError("Covered failure must stay unread")

        receipt = await manager.submit("covered", fail, agent_type="general", task="Covered task")
        app.action_open_sub_agent("covered")
        await _wait_for(pilot, lambda: app._sub_agent_inspector is not None, description="open inspector")
        screen = app._sub_agent_inspector
        assert screen is not None
        cover: ModalScreen[None] = ModalScreen()
        await app.push_screen(cover)
        await cover.mount(Static("Opaque cover"))
        gate.set()
        await manager.wait()
        await _wait_for(
            pilot,
            lambda: (
                f"background-result:{receipt['completion_id']}" in app._sub_agent_activities["covered"].stream_steps
            ),
            description="covered failure projected",
        )
        screen._acknowledge_presented_failure("covered", receipt["completion_id"])
        screen._on_tick()
        await pilot.pause(0.2)
        assert receipt["completion_id"] not in app._background_acknowledged_failures
        assert app._background_status().failed == 1
        app.pop_screen()
        await _wait_for(
            pilot,
            lambda: receipt["completion_id"] in app._background_acknowledged_failures,
            description="uncovered selected result painted",
        )


async def test_failed_retirement_updates_cached_storage_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app, _, _ = _build_background_app(tmp_path, monkeypatch)
    gate = asyncio.Event()
    async with app.run_test(size=(110, 32)) as pilot:
        manager = app._background_jobs
        assert isinstance(manager, BackgroundAgentManager)
        await _start_job(manager, "retire-storage", gate)
        await _wait_for(pilot, lambda: app._background_status().running == 1, description="running retirement job")
        recorder = manager.recorder
        original = recorder.record_background_job_terminal

        def fail(*args: Any, **kwargs: Any) -> None:
            raise SessionJournalError("retirement storage offline")

        monkeypatch.setattr(recorder, "record_background_job_terminal", fail)
        with pytest.raises(SessionJournalError, match="retirement storage offline"):
            await app._retire_background_jobs()
        await _wait_for(
            pilot, lambda: app._background_status().storage_failed == 1, description="retirement warning refreshed"
        )
        assert app._background_status().running == 0
        assert "save failed" in _plain(app)
        assert app._background_jobs is manager
        monkeypatch.setattr(recorder, "record_background_job_terminal", original)
        await app._retire_background_jobs()
        await _wait_for(pilot, lambda: not app._background_status().visible, description="retirement recovered")


async def test_ready_result_stays_visible_until_delivery_can_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app, _, _ = _build_background_app(tmp_path, monkeypatch)
    async with app.run_test(size=(110, 32)) as pilot:
        manager = app._background_jobs
        assert isinstance(manager, BackgroundAgentManager)
        app._background_rebuilding = True

        async def child() -> str:
            return "Ready but not consumed"

        await manager.submit("ready", child, agent_type="general", task="Waiting result")
        await manager.wait()
        await _wait_for(pilot, lambda: app._background_status().ready == 1, description="ready result")
        assert "1 result ready" in _plain(app)
        assert "Esc" not in _plain(app)
        assert app._background_status_timer is None
        assert manager.snapshot()[0]["delivered"] is False
        assert any(key.key == "ctrl+g" for key in app.query_one(ContextFooter).query(FooterKey))
        app._background_rebuilding = False
        app._schedule_pending_inputs()
        await _wait_for(pilot, lambda: not app._background_status().visible, description="ready result consumed")


async def test_delivery_error_is_visible_until_explicit_successful_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app, _, _ = _build_background_app(tmp_path, monkeypatch)
    async with app.run_test(size=(110, 32)) as pilot:
        manager = app._background_jobs
        assert isinstance(manager, BackgroundAgentManager)
        recorder = manager.recorder
        original = recorder.record_background_completion

        def fail(*args: Any, **kwargs: Any) -> None:
            raise SessionJournalError("completion delivery offline")

        monkeypatch.setattr(recorder, "record_background_completion", fail)

        async def child() -> str:
            return "Retained delivery report"

        await manager.submit("delivery", child, agent_type="general", task="Delivery work")
        await _wait_for(pilot, lambda: app._background_status().delivery_failed, description="delivery failure warning")
        assert "delivery failed" in _plain(app)
        assert "Esc" not in _plain(app)
        assert app._background_status_timer is None
        assert manager.snapshot()[0]["delivered"] is False
        monkeypatch.setattr(recorder, "record_background_completion", original)
        app.agent_worker = app.run_worker(app._process_message("Retry delivery"), exit_on_error=False)
        await wait_for_turn_idle(app, pilot)
        await _wait_for(pilot, lambda: not app._background_status().visible, description="delivery error recovered")
        assert manager.snapshot()[0]["delivered"] is True
