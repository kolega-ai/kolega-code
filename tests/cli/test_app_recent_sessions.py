"""Recent-session choices exercise production startup widgets and resume ownership."""

from __future__ import annotations

import asyncio
import json
import threading
import time
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from kolega_code.cli import main as main_module
from kolega_code.cli.app import KolegaCodeApp
from kolega_code.cli.session_store import SessionRecord, SessionStore, SessionStoreError
from kolega_code.cli.tui.settings_screen import ConfirmSettingsActionScreen
from kolega_code.cli.tui.startup import RecentSessionRow, RecentSessionsWidget, StartupEntryWidget
from kolega_code.cli.tui.state import ConversationEntry
from kolega_code.cli.tui.widgets import ChatComposer
from kolega_code.extensions import KolegaExtensionBundle
from kolega_code.llm.models import Message, TextBlock
from textual.pilot import Pilot
from textual.widgets import Collapsible

from ._app_test_utils import _build_mention_test_app, build_test_config, install_fake_agents


async def _wait(pilot: Pilot, predicate: Callable[[], bool]) -> None:
    deadline = time.monotonic() + 8
    while time.monotonic() < deadline:
        await pilot.pause(0.02)
        if predicate():
            return
    raise AssertionError("recent-session state did not settle")


def _saved(store: SessionStore, project: Path, title: str = "Previous work") -> SessionRecord:
    record = store.create(project, "cli", {}, title=title)
    recorder = store.recorder(record.session_id)
    recorder.record_context_message(Message(role="user", content=[TextBlock("saved request")]), actor="user")
    recorder.record_context_message(Message(role="assistant", content=[TextBlock("saved answer")]), actor="agent")
    store.release_session_lock(record.session_id)
    return record


def _row(app: KolegaCodeApp, session_id: str) -> RecentSessionRow:
    return next(
        row for row in app.query(RecentSessionRow) if row._item is not None and row._item.session_id == session_id
    )


@pytest.mark.asyncio
async def test_recent_sessions_are_project_scoped_bounded_and_do_not_steal_focus(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = _build_mention_test_app(tmp_path, monkeypatch)
    records = []
    for index in range(5):
        # Independent, explicit timestamps rather than relying on filesystem or
        # sub-millisecond creation order.
        with monkeypatch.context() as clock:
            clock.setattr("kolega_code.cli.session_store._now", lambda i=index: f"2026-09-{i + 1:02d}T12:00:00+00:00")
            records.append(_saved(app.store, app.project_path, f"Earlier session {index}"))
    _saved(app.store, tmp_path / "other-project", "Wrong project")
    async with app.run_test(size=(120, 45)) as pilot:
        await _wait(pilot, lambda: len(app._startup_recent_sessions()) == 3)
        assert [item.session_id for item in app._startup_recent_sessions()] == [
            records[4].session_id,
            records[3].session_id,
            records[2].session_id,
        ]
        assert app.screen.focused is app.query_one(ChatComposer)
        section = app.query_one(RecentSessionsWidget)
        await _wait(pilot, lambda: section.region.height == 4)
        card = app.query_one(StartupEntryWidget)
        details = card.query_one(".startup-configuration", Collapsible)
        await _wait(pilot, lambda: details.region.y == section.region.bottom)
        assert "Wrong project" not in "".join(str(row.render()) for row in app.query(RecentSessionRow))


@pytest.mark.asyncio
async def test_unchanged_listing_does_not_refresh_the_startup_card(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = _build_mention_test_app(tmp_path, monkeypatch)
    refreshed: list[bool] = []
    monkeypatch.setattr(app, "_ensure_startup_entry", lambda: refreshed.append(True))
    try:
        await app._load_recent_sessions()
        assert not refreshed
        _saved(app.store, app.project_path)
        await app._load_recent_sessions()
        assert len(refreshed) == 1
        await app._load_recent_sessions()
        assert len(refreshed) == 1
    finally:
        app.store.release_session_locks()


@pytest.mark.asyncio
async def test_no_recents_on_empty_or_resumed_startup_and_not_after_first_message_or_reset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = _build_mention_test_app(tmp_path, monkeypatch)
    async with app.run_test(size=(110, 45)) as pilot:
        assert not app.query_one(RecentSessionsWidget).display
        previous = _saved(app.store, app.project_path)
        await app._load_recent_sessions()
        await _wait(pilot, lambda: app.query_one(RecentSessionsWidget).region.height == 2)
        app._add_conversation_entry(ConversationEntry(kind="user", content="start new work"))
        await _wait(pilot, lambda: not app.query_one(RecentSessionsWidget).display)
        app.query_one(StartupEntryWidget).query_one(Collapsible).collapsed = False
        await app._reset_current_thread()
        assert not app._startup_recent_sessions()
        assert not app.query_one(RecentSessionsWidget).display
    resumed = KolegaCodeApp(
        project_path=app.project_path,
        config=app.config,
        mode="cli",
        store=app.store,
        session=previous,
        resuming_session=True,
    )
    async with resumed.run_test() as pilot:
        await pilot.pause()
        assert not resumed._recent_sessions_available
        assert not resumed.query_one(RecentSessionsWidget).display


@pytest.mark.asyncio
async def test_locked_click_preserves_current_ui_and_retry_resumes_after_unlock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = _build_mention_test_app(tmp_path, monkeypatch)
    previous = _saved(app.store, app.project_path)
    owner = SessionStore(app.store.root)
    owner.recorder(previous.session_id)
    notices: list[str] = []
    monkeypatch.setattr(app, "_notify_user", lambda text, **kwargs: notices.append(text))
    before = app.store.events_path_for(previous.session_id).read_bytes()
    try:
        async with app.run_test(size=(120, 45)) as pilot:
            app._set_sidebar_visible(False)
            await _wait(pilot, lambda: bool(app._startup_recent_sessions()))
            row = _row(app, previous.session_id)
            await _wait(pilot, lambda: row.region.height == 1)
            original_region = row.region
            assert await pilot.click(row)
            await _wait(pilot, lambda: bool(notices) and not app._resume_in_progress)
            await _wait(
                pilot,
                lambda: (
                    app._rendered_entry_count == len(app.conversation_entries)
                    and not app._conversation_anchor_pending
                    and row.region == original_region
                ),
            )
            assert "already open in another" in notices[-1]
            assert any(e.kind == "system" and "already open in another" in e.content for e in app.conversation_entries)
            assert previous.session_id not in notices[-1]
            assert app.resume_session is None and not app._quit_cleanly
            assert app.store.events_path_for(previous.session_id).read_bytes() == before
            assert app.screen.focused is app.query_one(ChatComposer)
            assert not app.query_one(ChatComposer).disabled
            owner.release_session_locks()
            assert await pilot.click(row)
            await _wait(pilot, lambda: app._quit_cleanly and not app._resume_in_progress)
            assert app.resume_session is not None
            assert app.resume_session.session_id == previous.session_id
            with pytest.raises(SessionStoreError, match="already open"):
                owner.claim_for_resume(previous.session_id, app.project_path)
    finally:
        owner.release_session_locks()
        app.store.release_session_locks()


@pytest.mark.asyncio
@pytest.mark.parametrize("width", [40, 80, 120])
async def test_draft_confirmation_cancel_preserves_draft_and_releases_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, width: int
) -> None:
    app = _build_mention_test_app(tmp_path, monkeypatch)
    previous = _saved(app.store, app.project_path)
    async with app.run_test(size=(width, 45)) as pilot:
        app._set_sidebar_visible(False)
        await _wait(pilot, lambda: bool(app._startup_recent_sessions()))
        composer = app.query_one(ChatComposer)
        composer.load_text("do not discard this")
        row = _row(app, previous.session_id)
        await _wait(pilot, lambda: row.region.height == 1)
        assert await pilot.click(row)
        await _wait(pilot, lambda: isinstance(app.screen, ConfirmSettingsActionScreen))
        dialog = app.screen.query_one("#settings_action_dialog")
        confirm = app.screen.query_one("#settings_action_confirm")
        cancel = app.screen.query_one("#settings_action_cancel")
        await _wait(
            pilot,
            lambda: (
                confirm.region.width >= 8
                and cancel.region.width >= 8
                and dialog.content_region.contains_region(confirm.region)
                and dialog.content_region.contains_region(cancel.region)
            ),
        )
        await pilot.press("escape")
        await _wait(pilot, lambda: not app._resume_in_progress)
        assert composer.text == "do not discard this"
        assert not composer.disabled
        assert app.resume_session is None
        contender = SessionStore(app.store.root)
        try:
            contender.claim_for_resume(previous.session_id, app.project_path)
        finally:
            contender.release_session_locks()


@pytest.mark.asyncio
async def test_attachment_only_confirmation_cancel_preserves_attachment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = _build_mention_test_app(tmp_path, monkeypatch)
    previous = _saved(app.store, app.project_path)
    attachment = {"type": "image", "media_type": "image/png", "path": "clipboard", "data": "aW1hZ2U="}
    async with app.run_test(size=(120, 45)) as pilot:
        await _wait(pilot, lambda: bool(app._startup_recent_sessions()))
        app._pending_image_attachments.append(attachment)
        assert not app.query_one(ChatComposer).text
        row = _row(app, previous.session_id)
        await _wait(pilot, lambda: row.region.height == 1)
        assert await pilot.click(row)
        await _wait(pilot, lambda: isinstance(app.screen, ConfirmSettingsActionScreen))
        await pilot.press("escape")
        await _wait(pilot, lambda: not app._resume_in_progress)
        assert app._pending_image_attachments == [attachment]
        assert app.resume_session is None
        contender = SessionStore(app.store.root)
        try:
            contender.claim_for_resume(previous.session_id, app.project_path)
        finally:
            contender.release_session_locks()


@pytest.mark.asyncio
async def test_repeated_quit_and_cancellation_wait_for_one_complete_shutdown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = _build_mention_test_app(tmp_path, monkeypatch)
    started, finish = asyncio.Event(), asyncio.Event()
    exited = MagicMock()
    monkeypatch.setattr(app, "exit", exited)
    async with app.run_test() as pilot:
        assert app.agent is not None
        original_cleanup = app.agent.cleanup
        cleanup_calls: list[bool] = []

        async def blocked_cleanup() -> None:
            cleanup_calls.append(True)
            started.set()
            await finish.wait()
            await original_cleanup()

        monkeypatch.setattr(app.agent, "cleanup", blocked_cleanup)
        first = asyncio.create_task(app.action_quit())
        second = None
        try:
            await asyncio.wait_for(started.wait(), 8)
            second = asyncio.create_task(app.action_quit())
            first.cancel()
            await asyncio.sleep(0)
            first.cancel()
            with pytest.raises(asyncio.CancelledError):
                await first
            await pilot.pause()
            assert not second.done()
            assert not app._quit_cleanly
            assert not exited.called
            assert app.agent is None
            assert app._session_shutdown_task is not None and not app._session_shutdown_task.done()
            finish.set()
            await asyncio.wait_for(second, 8)
            assert app._quit_cleanly
            assert cleanup_calls == [True]
            exited.assert_called_once()
        finally:
            finish.set()
            await asyncio.gather(first, *([second] if second is not None else []), return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["agent", "extension", "sink", "save"])
async def test_shutdown_failure_never_authorizes_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    app = _build_mention_test_app(tmp_path, monkeypatch)
    previous = _saved(app.store, app.project_path)
    monkeypatch.setattr(app, "exit", MagicMock())
    async with app.run_test() as pilot:
        await _wait(pilot, lambda: bool(app._startup_recent_sessions()))
        assert app.agent is not None
        failing_cleanup = AsyncMock(side_effect=OSError("shutdown failed"))
        if failure == "agent":
            monkeypatch.setattr(app.agent, "cleanup", failing_cleanup)
        elif failure == "extension":
            app._extension_bundle = KolegaExtensionBundle(cleanup=failing_cleanup)
        elif failure == "sink":
            assert app._usage_sink is not None
            await app._usage_sink.aclose()
            monkeypatch.setattr(app._usage_sink, "aclose", failing_cleanup)
        else:
            monkeypatch.setattr(app, "_save_session_async", failing_cleanup)
        await app._resume_recent_session(previous.session_id)
        assert not app._quit_cleanly
        assert app.resume_session is None
        assert not app._resume_in_progress
        failing_cleanup.assert_awaited_once()
        contender = SessionStore(app.store.root)
        try:
            contender.claim_for_resume(previous.session_id, app.project_path)
        finally:
            contender.release_session_locks()


@pytest.mark.asyncio
async def test_late_listing_cannot_reappear_after_conversation_starts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = _build_mention_test_app(tmp_path, monkeypatch)
    previous = _saved(app.store, app.project_path)
    started, finish = threading.Event(), threading.Event()

    def delayed_list(*, project_path: Path, recover: bool) -> list[SessionRecord]:
        assert not recover
        started.set()
        assert finish.wait(8)
        return [previous]

    monkeypatch.setattr(app.store, "list", delayed_list)
    try:
        async with app.run_test(size=(120, 45)) as pilot:
            await _wait(pilot, started.is_set)
            app._add_conversation_entry(ConversationEntry(kind="user", content="new work"))
            finish.set()
            await _wait(pilot, lambda: not any(w.group == "recent-sessions" for w in app.workers))
            assert not app._startup_recent_sessions()
            assert not app.query_one(RecentSessionsWidget).display
    finally:
        finish.set()


@pytest.mark.asyncio
async def test_cancelled_claim_waits_for_disk_thread_and_releases_reservation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = _build_mention_test_app(tmp_path, monkeypatch)
    previous = _saved(app.store, app.project_path)
    started, finish = threading.Event(), threading.Event()
    original_claim = app.store.claim_for_resume

    def delayed_claim(session_id: str, project_path: Path) -> SessionRecord:
        started.set()
        assert finish.wait(8)
        return original_claim(session_id, project_path)

    monkeypatch.setattr(app.store, "claim_for_resume", delayed_claim)
    try:
        async with app.run_test(size=(120, 45)) as pilot:
            await _wait(pilot, lambda: bool(app._startup_recent_sessions()))
            row = _row(app, previous.session_id)
            await _wait(pilot, lambda: row.region.height == 1)
            assert await pilot.click(row)
            await _wait(pilot, started.is_set)
            worker = next(w for w in app.workers if w.group == "resume-session")
            worker.cancel()
            await asyncio.sleep(0)
            worker.cancel()
            await asyncio.sleep(0)
            finish.set()
            await _wait(pilot, lambda: not app._resume_in_progress)
            assert app.resume_session is None
            contender = SessionStore(app.store.root)
            try:
                contender.claim_for_resume(previous.session_id, app.project_path)
            finally:
                contender.release_session_locks()
    finally:
        finish.set()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["deleted", "invalid_metadata", "invalid_journal"])
async def test_unavailable_target_does_not_close_the_current_ui(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    app = _build_mention_test_app(tmp_path, monkeypatch)
    previous = _saved(app.store, app.project_path)
    notices: list[str] = []
    monkeypatch.setattr(app, "_notify_user", lambda text, **kwargs: notices.append(text))
    async with app.run_test(size=(120, 45)) as pilot:
        await _wait(pilot, lambda: bool(app._startup_recent_sessions()))
        if failure == "deleted":
            app.store.delete(previous.session_id)
        elif failure == "invalid_metadata":
            metadata = previous.to_metadata_dict()
            del metadata["workspace_id"]
            app.store.path_for(previous.session_id).write_text(json.dumps(metadata))
        else:
            app.store.events_path_for(previous.session_id).write_text("not valid json\n")
        row = _row(app, previous.session_id)
        await _wait(pilot, lambda: row.region.height == 1)
        assert await pilot.click(row)
        await _wait(pilot, lambda: bool(notices) and not app._resume_in_progress)
        assert notices[-1].startswith("Could not resume session:")
        assert app.resume_session is None
        assert not app._quit_cleanly
        assert not app.query_one(ChatComposer).disabled


@pytest.mark.asyncio
async def test_saved_workspace_confirmation_cancel_leaves_session_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from kolega_code.worktrees import WorktreeError, WorktreeErrorCode

    app = _build_mention_test_app(tmp_path, monkeypatch)
    previous = _saved(app.store, app.project_path)
    previous.active_project_path = str(tmp_path / "missing-worktree")
    app.store.save(previous)
    before = app.store.events_path_for(previous.session_id).read_bytes()

    def resolve(root: Path, target: str | Path) -> SimpleNamespace:
        if Path(target) != root:
            raise WorktreeError(WorktreeErrorCode.UNKNOWN_WORKTREE, "The saved checkout was removed.")
        return SimpleNamespace(path=root)

    monkeypatch.setattr("kolega_code.worktrees.resolve_worktree", resolve)
    async with app.run_test(size=(120, 45)) as pilot:
        await _wait(pilot, lambda: bool(app._startup_recent_sessions()))
        row = _row(app, previous.session_id)
        await _wait(pilot, lambda: row.region.height == 1)
        assert await pilot.click(row)
        await _wait(pilot, lambda: isinstance(app.screen, ConfirmSettingsActionScreen))
        assert isinstance(app.screen, ConfirmSettingsActionScreen)
        assert "Saved active worktree is unavailable" in app.screen.action_copy
        await pilot.press("escape")
        await _wait(pilot, lambda: not app._resume_in_progress)
        assert app.resume_session is None
        assert app.store.events_path_for(previous.session_id).read_bytes() == before
        assert app.store.load(previous.session_id).active_project_path == previous.active_project_path


@pytest.mark.asyncio
async def test_keyboard_resume_and_confirmed_draft_discard(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    app = _build_mention_test_app(tmp_path, monkeypatch)
    previous = _saved(app.store, app.project_path)
    async with app.run_test(size=(120, 45)) as pilot:
        app._set_sidebar_visible(False)
        await _wait(pilot, lambda: bool(app._startup_recent_sessions()))
        row = _row(app, previous.session_id)
        composer = app.query_one(ChatComposer)
        composer.load_text("unsent draft")
        composer.focus()
        for _ in range(20):
            if app.screen.focused is row:
                break
            await pilot.press("tab")
        assert app.screen.focused is row
        await pilot.press("enter")
        await _wait(pilot, lambda: isinstance(app.screen, ConfirmSettingsActionScreen))
        assert await pilot.click("#settings_action_confirm")
        await _wait(pilot, lambda: app._quit_cleanly and not app._resume_in_progress)
        assert app.resume_session is not None
        assert app.resume_session.session_id == previous.session_id
    app.store.release_session_locks()


def test_real_cli_click_relaunches_and_restores_conversation_and_controls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    install_fake_agents(monkeypatch)
    project = tmp_path / "project"
    project.mkdir()
    config = build_test_config(project)
    store = SessionStore(tmp_path / "state")
    previous = _saved(store, project)
    previous.interaction_mode = "plan"
    previous.permission_mode = "auto"
    previous.task_list_markdown = "- [ ] Saved task"
    previous.latest_plan_markdown = "# Saved plan\n\nKeep this plan."
    store.save(previous)
    monkeypatch.setattr(main_module, "_store_from_args", lambda args: store)
    monkeypatch.setattr(main_module, "build_agent_config", lambda *args, **kwargs: config)
    instances: list[KolegaCodeApp] = []
    hints: list[str] = []
    monkeypatch.setattr(main_module, "_print_quit_resume_hint", hints.append)

    async def no_update(self: KolegaCodeApp) -> None:
        pass

    monkeypatch.setattr(KolegaCodeApp, "_check_for_update_on_startup", no_update)

    async def drive_real_app(self: KolegaCodeApp) -> None:
        instances.append(self)
        async with self.run_test(size=(120, 45)) as pilot:
            self._set_sidebar_visible(False)
            if len(instances) == 1:
                await _wait(pilot, lambda: bool(self._startup_recent_sessions()))
                row = _row(self, previous.session_id)
                await _wait(pilot, lambda: row.region.height == 1)
                assert await pilot.click(row)
                await _wait(pilot, lambda: self._quit_cleanly and not self._resume_in_progress)
            else:
                assert len(instances) == 2
                assert self.session.session_id == previous.session_id
                assert self._resuming_session
                assert self.interaction_mode == "plan"
                assert self.permission_mode.value == "auto"
                assert self.session.task_list_markdown == "- [ ] Saved task"
                assert self._latest_plan == "# Saved plan\n\nKeep this plan."
                assert any(e.content == "saved request" for e in self.conversation_entries)
                assert any(e.content == "saved answer" for e in self.conversation_entries)
                assert self._session_recorder.journal.session_id == previous.session_id
                assert not self.query_one(RecentSessionsWidget).display
                contender = SessionStore(store.root)
                with pytest.raises(SessionStoreError, match="already open"):
                    contender.claim_for_resume(previous.session_id, project)
                old = contender.claim_for_resume(instances[0].session.session_id, project)
                assert old.session_id != previous.session_id
                contender.release_session_locks()
                await self.action_quit()

    monkeypatch.setattr(KolegaCodeApp, "run_async", drive_real_app)
    args = main_module.parse_args(["tui", str(project), "--state-dir", str(store.root)])
    assert main_module._run_tui(args) == 0
    assert len(instances) == 2
    assert hints == [previous.session_id]
    assert not any(event["content"] for event in store.load(instances[0].session.session_id).history)
    contender = SessionStore(store.root)
    try:
        assert contender.claim_for_resume(previous.session_id, project).session_id == previous.session_id
    finally:
        contender.release_session_locks()
