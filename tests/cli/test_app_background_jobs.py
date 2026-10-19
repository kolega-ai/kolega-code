from __future__ import annotations

import asyncio
from contextlib import suppress
import threading
import time
from pathlib import Path
from typing import Any, ClassVar, cast
from unittest.mock import MagicMock

import pytest
from rich.text import Text
from textual.widgets import Collapsible
from textual.widgets._collapsible import CollapsibleTitle
from textual.worker import WorkerCancelled

from kolega_code.cli.config import config_summary
from kolega_code.cli import messages
from kolega_code.cli.session_store import SessionStore
from kolega_code.cli.tui import agent_runtime as agent_runtime_module
from kolega_code.cli.tui.state import ConversationEntry, TurnState
from kolega_code.cli.tui.widgets import ChatComposer, ConversationEntryWidget, ToolEntryWidget
from kolega_code.events import AgentEvent
from kolega_code.llm.models import Message, TextBlock
from kolega_code.session.background import BackgroundAgentManager

from ._app_test_utils import (
    FakeCoderAgent,
    build_test_config,
    install_fake_agents,
    wait_for_session_diff_baseline,
    wait_for_turn_idle,
)
from .test_app_changes_inspector import _init_git_project
from .test_app_sub_agents_workflows import _sub_agent_event
from .test_app_worktree_switch import _approve_switch, _git, _workspace_events


async def _wait_for(
    pilot: Any,
    predicate,
    *,
    timeout: float = 6.0,
    description: str,
) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        await pilot.pause(0.01)
    raise AssertionError(f"Timed out waiting for {description}")


class BackgroundHarnessAgent(FakeCoderAgent):
    instances: ClassVar[list["BackgroundHarnessAgent"]] = []

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.background_jobs: BackgroundAgentManager | None = None
        self.background_needs_turn = False
        self.turn_gate: asyncio.Event | None = None
        self.continue_gate: asyncio.Event | None = None
        self.turn_started = asyncio.Event()
        self.continuation_started = asyncio.Event()
        self.continued = asyncio.Event()
        self.lifecycle_events: list[str] = []
        self.active_turns = 0
        self.max_active_turns = 0
        self.continuations = 0
        self.appended_user_texts: list[str] = []
        type(self).instances.append(self)

    def get_background_jobs(self) -> BackgroundAgentManager:
        if self.background_jobs is None:
            self.background_jobs = BackgroundAgentManager(self.session_recorder)
        return self.background_jobs

    def append_user_message(self, content: list[TextBlock]) -> None:
        self.appended_user_texts.extend(block.text for block in content if isinstance(block, TextBlock))
        blocks: list[Any] = list(content)
        self.history.append(Message(role="user", content=blocks))

    async def _single_chunk_stream(
        self,
        text: str,
        *,
        label: str,
        gate: asyncio.Event | None,
        started: asyncio.Event,
    ):
        self.active_turns += 1
        self.max_active_turns = max(self.max_active_turns, self.active_turns)
        assert self.active_turns == 1, "overlapping parent/background turns"
        self.lifecycle_events.append(f"{label}:start")
        started.set()
        try:
            if gate is not None:
                await gate.wait()
            yield {"type": "response", "content": text, "complete": True, "uuid": f"uuid-{id(self)}"}
        finally:
            self.lifecycle_events.append(f"{label}:end")
            self.active_turns -= 1

    async def process_message_stream(self, message: str, attachments: list[dict] | None = None):
        self.messages.append(message)
        self.attachments.append(attachments)
        self.history.append(Message(role="user", content=[TextBlock(text=message)]))
        async for chunk in self._single_chunk_stream(
            "parent response",
            label="parent",
            gate=self.turn_gate,
            started=self.turn_started,
        ):
            if chunk.get("complete"):
                self.history.append(
                    Message(role="assistant", content=[TextBlock(text=str(chunk["content"]))], stop_reason="end_turn")
                )
            yield chunk

    async def continue_from_history_stream(self):
        self.background_needs_turn = False
        self.continuations += 1
        async for chunk in self._single_chunk_stream(
            "continued with background result",
            label="continuation",
            gate=self.continue_gate,
            started=self.continuation_started,
        ):
            if chunk.get("complete"):
                self.history.append(
                    Message(role="assistant", content=[TextBlock(text=str(chunk["content"]))], stop_reason="end_turn")
                )
            yield chunk
        self.continued.set()

    async def cleanup(self) -> None:
        if self.background_jobs is not None:
            await self.background_jobs.close()
        await super().cleanup()


def _build_background_app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, initialize_git: bool = False):
    pytest.importorskip("textual")

    from kolega_code.cli.app import KolegaCodeApp

    BackgroundHarnessAgent.instances.clear()
    install_fake_agents(monkeypatch, coder_cls=BackgroundHarnessAgent)

    project = tmp_path / "project"
    project.mkdir()
    if initialize_git:
        _init_git_project(project)
    config = build_test_config(project)
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", config_summary(config))
    app = KolegaCodeApp(project_path=project, config=config, mode="code", store=store, session=session)
    return app, store, session


def _background_activity_event(**content: Any) -> AgentEvent:
    event = _sub_agent_event(**content)
    assert event.sub_agent_info is not None
    event.sub_agent_info["background"] = True
    return event


def _history_texts(history: list[Any]) -> list[str]:
    texts: list[str] = []
    for item in history:
        message = item if isinstance(item, Message) else Message.from_dict(item)
        content = getattr(message, "content", None)
        if isinstance(content, str):
            texts.append(content)
            continue
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, TextBlock):
                texts.append(block.text)
    return texts


def _history_contains(history: list[Any], needle: str) -> bool:
    return any(needle in text for text in _history_texts(history))


def _background_row(store: SessionStore, session_id: str, epoch_id: str, job_id: str) -> dict[str, Any]:
    recorder = store.recorder(session_id)
    return next(row for row in recorder.read_background_jobs(epoch_id=epoch_id) if row["job_id"] == job_id)


def _record_turn(recorder: Any, user_text: str) -> None:
    recorder.start_turn(Message(role="user", content=[TextBlock(user_text)]))
    recorder.record_assistant(Message(role="assistant", content=[TextBlock("done")], stop_reason="end_turn"))
    recorder.finish_turn("completed")


async def _completion_widget(app: Any, pilot: Any, entry: ConversationEntry) -> ToolEntryWidget:
    await _wait_for(
        pilot,
        lambda: entry.entry_id in app._entry_widgets and bool(app._entry_widgets[entry.entry_id].query(".tool-body")),
        description="collapsed background report to mount",
    )
    widget = app._entry_widgets[entry.entry_id]
    assert isinstance(widget, ToolEntryWidget)
    return widget


@pytest.mark.parametrize("source", ["live", "restored"])
@pytest.mark.parametrize("status", ["completed", "failed", "cancelled", "interrupted"])
@pytest.mark.asyncio
async def test_background_reports_start_collapsed_and_expand_on_demand(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source: str,
    status: str,
) -> None:
    app, _, _ = _build_background_app(tmp_path, monkeypatch)
    # Include markup-like text and enough output to expose an accidental dump.
    report = "[bold]literal report[/bold]\n" + "\n".join(f"Report detail {i}" for i in range(100))
    completion = {
        "job_id": "job-report",
        "completion_id": "completion-report",
        "agent_type": "investigation",
        "status": status,
        "result": report,
    }
    async with app.run_test(size=(120, 40)) as pilot:
        app._set_sidebar_visible(False)
        if source == "live":
            app._background_completion_delivered(completion)
        else:
            message = Message(
                role="user", content=[TextBlock(BackgroundAgentManager._completion_text(completion))]
            ).to_dict()
            app._restore_conversation_history([message, message])

        entries = [
            entry
            for entry in app.conversation_entries
            if entry.origin == {"kind": "subagent_completion", "job_id": "job-report"}
        ]
        assert len(entries) == 1
        entry = entries[0]
        # Presentation must not truncate or rewrite the retained report.
        assert entry.content.endswith("\n\n" + report)
        await _wait_for(pilot, lambda: entry.entry_id in app._entry_widgets, description="background completion row")
        widget = app._entry_widgets.get(entry.entry_id)
        assert isinstance(widget, ToolEntryWidget)
        widget = await _completion_widget(app, pilot, entry)
        disclosure = widget.query_one(Collapsible)
        title = widget.query_one(CollapsibleTitle)
        assert disclosure.collapsed is True
        assert f"Background investigation {status}" in Text.from_markup(disclosure.title).plain
        assert "job-report" not in disclosure.title
        assert "Report detail" not in disclosure.title
        # Tool rows have one blank separator beneath their single-line title.
        await _wait_for(pilot, lambda: widget.region.height == 2, description="single-line background result")

        title.focus()
        await pilot.press("enter")
        await _wait_for(pilot, lambda: not disclosure.collapsed, description="explicit report expansion")
        assert widget._body is not None
        assert str(widget._body.render()) == entry.content
        await _wait_for(pilot, lambda: widget.region.height > 10, description="full report layout")

        # Resize/refresh must not undo the explicit disclosure choice.
        await pilot.resize_terminal(45, 40)
        widget.refresh_content()
        assert disclosure.collapsed is False
        title.focus()
        await pilot.press("enter")
        await _wait_for(
            pilot, lambda: disclosure.collapsed and widget.region.height <= 3, description="report folded again"
        )
        assert entry.content.endswith("\n\n" + report)


@pytest.mark.asyncio
async def test_ordinary_system_notes_are_not_background_disclosures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app, _, _ = _build_background_app(tmp_path, monkeypatch)
    async with app.run_test():
        entry = ConversationEntry(
            kind="system", content="Background investigation completed (not-a-job):\n\nAn ordinary note."
        )
        assert type(app._make_entry_widget(entry)) is ConversationEntryWidget


@pytest.mark.asyncio
async def test_idle_background_completion_is_inserted_and_continues_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app, _, _ = _build_background_app(tmp_path, monkeypatch)

    async with app.run_test() as pilot:
        await pilot.pause()
        agent = app.agent
        assert isinstance(agent, BackgroundHarnessAgent)
        manager = app._background_jobs
        assert isinstance(manager, BackgroundAgentManager)

        release = asyncio.Event()

        async def child() -> str:
            await release.wait()
            return "idle child report"

        await manager.submit("job-idle", child, agent_type="investigation", task="inspect")
        release.set()

        await asyncio.wait_for(agent.continued.wait(), 2)
        await wait_for_turn_idle(app, pilot)
        await pilot.pause(0.05)

        assert agent.continuations == 1
        assert agent.messages == []
        assert agent.max_active_turns == 1
        assert any("idle child report" in text for text in agent.appended_user_texts)
        assert manager.snapshot("job-idle")[0]["delivered"] is True
        completion_entries = [
            entry
            for entry in app.conversation_entries
            if entry.origin == {"kind": "subagent_completion", "job_id": "job-idle"}
        ]
        assert len(completion_entries) == 1
        assert "Background investigation completed (job-idle):" in completion_entries[0].content
        assert "idle child report" in completion_entries[0].content
        widget = await _completion_widget(app, pilot, completion_entries[0])
        assert widget.query_one(Collapsible).collapsed is True


@pytest.mark.asyncio
async def test_blocked_background_pump_rearms_and_duplicate_starts_do_not_overlap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app, _, _ = _build_background_app(tmp_path, monkeypatch)

    async with app.run_test() as pilot:
        await pilot.pause()
        agent = app.agent
        assert isinstance(agent, BackgroundHarnessAgent)
        manager = app._background_jobs
        assert isinstance(manager, BackgroundAgentManager)

        app._background_rebuilding = True
        release = asyncio.Event()
        agent.continue_gate = asyncio.Event()

        async def child() -> str:
            await release.wait()
            return "held until rebuild clears"

        await manager.submit("job-rearm", child, agent_type="investigation", task="inspect")
        release.set()
        await _wait_for(
            pilot,
            lambda: manager.pending_count == 1,
            description="the blocked completion to accumulate in the mailbox",
        )
        assert agent.continuations == 0
        assert app.agent_worker is None

        app._background_rebuilding = False
        assert app._maybe_start_queued_message() is True
        assert app._maybe_start_queued_message() is False

        await asyncio.wait_for(agent.continuation_started.wait(), 2)
        assert agent.continuations == 1
        assert agent.max_active_turns == 1

        agent.continue_gate.set()
        await asyncio.wait_for(agent.continued.wait(), 2)
        await wait_for_turn_idle(app, pilot)
        await pilot.pause(0.05)

        assert manager.pending_count == 0
        assert agent.continuations == 1
        assert agent.max_active_turns == 1


@pytest.mark.asyncio
async def test_explicit_stop_delivers_background_completion_without_restarting_parent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app, _, _ = _build_background_app(tmp_path, monkeypatch)

    async with app.run_test() as pilot:
        await pilot.pause()
        agent = app.agent
        assert isinstance(agent, BackgroundHarnessAgent)
        manager = app._background_jobs
        assert isinstance(manager, BackgroundAgentManager)

        agent.turn_gate = asyncio.Event()
        app._queue_user_message("work")
        assert app._maybe_start_queued_message() is True
        await asyncio.wait_for(agent.turn_started.wait(), 2)

        async def child() -> str:
            return "stop-path child report"

        await manager.submit("job-stop", child, agent_type="investigation", task="inspect")
        await _wait_for(
            pilot,
            lambda: manager.pending_count == 1,
            description="the completion to remain queued while the parent turn is busy",
        )
        assert agent.continuations == 0

        app.action_cancel_generation()
        await wait_for_turn_idle(app, pilot)
        await pilot.pause(0.05)

        assert agent.messages == ["work"]
        assert agent.continuations == 0
        assert manager.pending_count == 0
        assert any("stop-path child report" in text for text in agent.appended_user_texts)
        completion_entries = [entry for entry in app.conversation_entries if entry.origin is not None]
        assert any(
            entry.origin == {"kind": "subagent_completion", "job_id": "job-stop"} for entry in completion_entries
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("stage", "outcome"),
    [
        ("message", "finish"),
        ("message-finally", "finish"),
        ("completion", "finish"),
        ("completion", "cancel"),
        ("completion", "error"),
        ("stream-drain", "finish"),
        ("stream-drain", "cancel"),
        ("stream-save", "finish"),
        ("stream-save", "cancel"),
    ],
)
async def test_stale_finalizer_preserves_replacement_owner_and_turn_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
    outcome: str,
) -> None:
    """Construct a replacement during teardown; ordinary UI reachability is not asserted."""
    app, _, _ = _build_background_app(tmp_path, monkeypatch)
    async with app.run_test() as pilot:
        await pilot.pause()
        agent = app.agent
        assert isinstance(agent, BackgroundHarnessAgent)
        jobs = app._background_jobs
        assert isinstance(jobs, BackgroundAgentManager)
        entered, release = asyncio.Event(), asyncio.Event()
        old_started = agent.turn_started
        agent.turn_gate = asyncio.Event()
        method = {
            "message": "_apply_pending_workspace_switch",
            "message-finally": "_apply_pending_workspace_switch",
            "completion": "_cleanup_retired_agent_generations",
            "stream-drain": "_drain_pending_events",
            "stream-save": "_save_session_history_async",
        }[stage]
        original = getattr(app, method)
        calls = 0

        async def held_teardown() -> Any:
            nonlocal calls
            calls += 1
            if stage == "message-finally" and calls == 1:
                # Force the awaited fallback in _process_message's finally,
                # without creating a real workspace transition.
                app._pending_workspace_switch = cast(Any, object())
                return None
            result = None if stage == "message-finally" else await original()
            if not entered.is_set():
                entered.set()
                operation = asyncio.create_task(release.wait())
                while not operation.done():
                    try:
                        await asyncio.shield(operation)
                    except asyncio.CancelledError:
                        pass
                if outcome == "cancel":
                    raise asyncio.CancelledError
                if outcome == "error":
                    raise RuntimeError("late retired-generation cleanup failure")
            return result

        if not stage.startswith("stream-"):
            monkeypatch.setattr(app, method, held_teardown)
        if stage == "completion":

            async def child() -> str:
                return "old owner report"

            await jobs.submit("old-owner", child, agent_type="investigation", task="inspect")
        else:
            app._queue_user_message("old owner")
            assert app._maybe_start_queued_message()
        if stage.startswith("stream-"):
            await asyncio.wait_for(old_started.wait(), 4)
            monkeypatch.setattr(app, method, held_teardown)
        agent.turn_gate.set()
        old = replacement = None
        try:
            await asyncio.wait_for(entered.wait(), 4)
            old = app.agent_worker
            assert old is not None
            if stage == "message-finally":
                app._pending_workspace_switch = None
            agent.turn_started = asyncio.Event()
            agent.turn_gate = asyncio.Event()
            replacement = app.run_worker(
                app._process_message("replacement owner"), name="replacement-owner", group="turns", exclusive=True
            )
            app.agent_worker = replacement
            await asyncio.wait_for(agent.turn_started.wait(), 4)
            composer = app.query_one("#composer", ChatComposer)
            timer, started_at = app._turn_timer, app._turn_started_at
            usage = app._usage_ledger.snapshot()
            assert app._turn_active and timer is not None and started_at is not None
            assert composer.placeholder == messages.QUEUE_PLACEHOLDER

            release.set()
            if stage == "completion" and outcome == "cancel":
                with pytest.raises(WorkerCancelled):
                    await asyncio.wait_for(old.wait(), 4)
            else:
                await asyncio.wait_for(old.wait(), 4)

            assert app.agent_worker is replacement
            assert replacement.is_running
            assert app._turn_active
            assert app._turn_timer is timer
            assert app._turn_started_at == started_at
            assert app._status_state.turn_state is TurnState.GENERATING
            assert composer.placeholder == messages.QUEUE_PLACEHOLDER
            assert not composer.disabled
            assert not app._background_paused
            assert not app._background_delivery_failed
            assert app._usage_ledger.snapshot() == usage
            assert not app._maybe_start_queued_message()
            assert agent.active_turns == agent.max_active_turns == 1

            # Exercise the real owner-dependent stop path, not worker.cancel().
            app.action_cancel_generation()
            await asyncio.wait_for(replacement.wait(), 4)
            await wait_for_turn_idle(app, pilot)
            assert app.agent_worker is None
            assert not app._turn_active
            assert app._turn_timer is None and app._turn_started_at is None
            assert app._status_state.turn_state is TurnState.STOPPED
            assert app._background_paused and not app._background_delivery_failed
            assert composer.placeholder == messages.COMPOSER_PLACEHOLDER
            assert agent.active_turns == 0
        finally:
            release.set()
            agent.turn_gate.set()
            if old is not None:
                with suppress(WorkerCancelled):
                    await asyncio.wait_for(old.wait(), 4)
            if replacement is not None:
                await asyncio.wait_for(replacement.wait(), 4)
        await pilot.pause(0.05)


@pytest.mark.asyncio
async def test_agent_rebuild_keeps_background_manager_and_retires_old_generation_until_completion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app, _, _ = _build_background_app(tmp_path, monkeypatch)
    cleaned_bundles: list[object] = []

    async def fake_cleanup_extension_bundle(bundle: object) -> None:
        cleaned_bundles.append(bundle)

    monkeypatch.setattr(agent_runtime_module, "cleanup_extension_bundle", fake_cleanup_extension_bundle)

    async with app.run_test() as pilot:
        await pilot.pause()
        old_agent = app.agent
        assert isinstance(old_agent, BackgroundHarnessAgent)
        manager = app._background_jobs
        assert isinstance(manager, BackgroundAgentManager)
        sentinel_bundle = cast(Any, object())
        app._extension_bundle = sentinel_bundle

        release = asyncio.Event()

        async def child() -> str:
            await release.wait()
            return "rebuilt child report"

        await manager.submit("job-rebuild", child, agent_type="investigation", task="inspect")
        await _wait_for(pilot, lambda: manager.active_count == 1, description="the rebuild-era child to start")
        app._render_event(
            _background_activity_event(agent_id="job-rebuild", status="GENERATING", message="Starting background")
        )
        activity = app._sub_agent_activities["job-rebuild"]

        assert app.config is not None
        await app._build_agent(app.config, rebuild=True)

        new_agent = app.agent
        assert isinstance(new_agent, BackgroundHarnessAgent)
        assert new_agent is not old_agent
        assert app._background_jobs is manager
        assert new_agent.background_jobs is manager
        assert old_agent.background_jobs is None
        assert app._sub_agent_activities["job-rebuild"] is activity
        assert activity.entry in app.conversation_entries
        assert not activity.entry.complete
        assert old_agent.cleanup_calls == 0
        assert len(app._retired_agent_generations) == 1
        retired_agent, retired_bundle = app._retired_agent_generations[0]
        assert retired_agent is old_agent
        assert retired_bundle is sentinel_bundle
        assert cleaned_bundles == []

        release.set()
        await asyncio.wait_for(new_agent.continued.wait(), 2)
        await wait_for_turn_idle(app, pilot)
        await pilot.pause(0.05)

        assert new_agent.continuations == 1
        assert old_agent.continuations == 0
        assert old_agent.cleanup_calls == 1
        assert cleaned_bundles == [sentinel_bundle]
        assert app._retired_agent_generations == []


@pytest.mark.asyncio
async def test_thread_reset_settles_old_epoch_background_jobs_without_restoring_them(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app, _, _ = _build_background_app(tmp_path, monkeypatch)

    async with app.run_test() as pilot:
        await pilot.pause()
        agent = app.agent
        assert isinstance(agent, BackgroundHarnessAgent)
        old_manager = app._background_jobs
        assert isinstance(old_manager, BackgroundAgentManager)
        old_epoch = app._session_recorder.journal.epoch_id

        never = asyncio.Event()

        async def child() -> str:
            await never.wait()
            return "unexpected"

        await old_manager.submit("job-reset", child, agent_type="investigation", task="inspect")
        await _wait_for(pilot, lambda: old_manager.active_count == 1, description="the reset-era child to start")

        await app._reset_current_thread()
        await pilot.pause(0.05)

        assert app._session_recorder.journal.epoch_id != old_epoch
        assert app._background_jobs is not old_manager
        assert app._background_jobs is agent.background_jobs
        assert old_manager.snapshot("job-reset")[0]["status"] == "cancelled"
        assert not old_manager.has_work
        assert not any(
            entry.origin == {"kind": "subagent_completion", "job_id": "job-reset"} for entry in app.conversation_entries
        )
        assert not any("Background investigation" in entry.content for entry in app.conversation_entries)
        assert all(message.role != "user" for message in agent.history)


@pytest.mark.asyncio
async def test_shutdown_joins_parent_turn_and_persists_cancelled_child_before_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app, store, session = _build_background_app(tmp_path, monkeypatch)
    exit_mock = MagicMock()
    monkeypatch.setattr(app, "exit", exit_mock)

    async with app.run_test() as pilot:
        await pilot.pause()
        agent = app.agent
        assert isinstance(agent, BackgroundHarnessAgent)
        manager = app._background_jobs
        assert isinstance(manager, BackgroundAgentManager)
        epoch_id = app._session_recorder.journal.epoch_id

        agent.turn_gate = asyncio.Event()
        app._queue_user_message("work")
        assert app._maybe_start_queued_message() is True
        await asyncio.wait_for(agent.turn_started.wait(), 2)

        child_cancelled = asyncio.Event()
        never = asyncio.Event()

        async def child() -> str:
            agent.lifecycle_events.append("child:start")
            try:
                await never.wait()
            finally:
                agent.lifecycle_events.append("child:cancel")
                child_cancelled.set()
            return "unexpected"

        await manager.submit("job-shutdown", child, agent_type="investigation", task="inspect")
        await _wait_for(pilot, lambda: manager.active_count == 1, description="the shutdown-era child to start")

        cleanup_started = asyncio.Event()
        release_cleanup = asyncio.Event()
        original_cleanup = agent.cleanup

        async def blocked_cleanup() -> None:
            agent.lifecycle_events.append("cleanup:start")
            assert agent.active_turns == 0
            row = _background_row(store, session.session_id, epoch_id, "job-shutdown")
            assert row["status"] == "cancelled"
            assert row["delivered"] is True
            loaded = store.load(session.session_id)
            assert _history_contains(loaded.history, "job-shutdown")
            assert _history_contains(loaded.history, "Background job was cancelled.")
            cleanup_started.set()
            await release_cleanup.wait()
            await original_cleanup()

        monkeypatch.setattr(agent, "cleanup", blocked_cleanup)

        quit_task = asyncio.create_task(app.action_quit())
        await asyncio.wait_for(cleanup_started.wait(), 8)
        await asyncio.wait_for(child_cancelled.wait(), 2)

        assert agent.lifecycle_events.index("parent:end") < agent.lifecycle_events.index("child:cancel")
        assert agent.lifecycle_events.index("child:cancel") < agent.lifecycle_events.index("cleanup:start")
        assert not exit_mock.called

        release_cleanup.set()
        await asyncio.wait_for(quit_task, 8)

        row = _background_row(store, session.session_id, epoch_id, "job-shutdown")
        assert row["status"] == "cancelled"
        assert row["delivered"] is True
        assert app._quit_cleanly
        exit_mock.assert_called_once()


@pytest.mark.asyncio
async def test_workspace_activation_retires_old_jobs_without_late_delivery_into_new_agent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app, store, session = _build_background_app(tmp_path, monkeypatch)
    other = tmp_path / "other-workspace"
    other.mkdir()

    async with app.run_test() as pilot:
        await pilot.pause()
        old_agent = app.agent
        assert isinstance(old_agent, BackgroundHarnessAgent)
        old_manager = app._background_jobs
        assert isinstance(old_manager, BackgroundAgentManager)
        old_epoch = app._session_recorder.journal.epoch_id
        _record_turn(app._session_recorder, "ordinary pre-switch history")
        old_agent.restore_message_history(store.load(session.session_id).history)

        child_cancelled = asyncio.Event()
        never = asyncio.Event()

        async def child() -> str:
            old_agent.lifecycle_events.append("child:start")
            try:
                await never.wait()
            finally:
                old_agent.lifecycle_events.append("child:cancel")
                child_cancelled.set()
            return "unexpected"

        await old_manager.submit("job-workspace", child, agent_type="investigation", task="inspect")
        await _wait_for(pilot, lambda: old_manager.active_count == 1, description="the workspace child to start")

        await app._activate_workspace(other, baseline_label="switched")
        await pilot.pause(0.05)
        await asyncio.wait_for(child_cancelled.wait(), 2)

        new_agent = app.agent
        assert isinstance(new_agent, BackgroundHarnessAgent)
        assert new_agent is not old_agent
        assert app.active_project_path == other
        assert app._background_jobs is not old_manager
        assert new_agent.background_jobs is app._background_jobs

        row = _background_row(store, session.session_id, old_epoch, "job-workspace")
        assert row["status"] == "cancelled"
        assert row["delivered"] is True
        assert old_manager.snapshot("job-workspace")[0]["status"] == "cancelled"
        assert not old_manager.has_work

        assert new_agent.continuations == 0
        assert new_agent.appended_user_texts == []
        assert not new_agent.continuation_started.is_set()
        assert not _history_contains(new_agent.history, "job-workspace")
        assert not _history_contains(new_agent.history, "Background job was cancelled.")
        assert _history_contains(new_agent.history, "ordinary pre-switch history")
        replay = store.load(session.session_id).history
        assert _history_contains(replay, "ordinary pre-switch history")
        assert not _history_contains(replay, "job-workspace")
        assert app._background_jobs is not None
        await app._background_jobs.recover()
        assert not app._background_jobs.has_work
        assert any(
            entry.origin == {"kind": "subagent_completion", "job_id": "job-workspace"}
            for entry in app.conversation_entries
        )

        original_report = next(
            entry.content
            for entry in app.conversation_entries
            if entry.origin == {"kind": "subagent_completion", "job_id": "job-workspace"}
        )
        assert app.config is not None
        await app._build_agent(app.config, rebuild=True)
        rebuilt = app.agent
        assert isinstance(rebuilt, BackgroundHarnessAgent)
        assert not _history_contains(rebuilt.history, "job-workspace")
        assert [
            entry.content
            for entry in app.conversation_entries
            if entry.origin == {"kind": "subagent_completion", "job_id": "job-workspace"}
        ] == [original_report]
        archived_entry = next(
            entry for entry in app.conversation_entries if (entry.origin or {}).get("job_id") == "job-workspace"
        )
        widget = await _completion_widget(app, pilot, archived_entry)
        assert widget.query_one(Collapsible).collapsed is True

    from kolega_code.cli.app import KolegaCodeApp

    resumed = KolegaCodeApp(
        project_path=app.project_path,
        config=app.config,
        mode="code",
        store=store,
        session=store.load(session.session_id),
        resuming_session=True,
    )
    async with resumed.run_test() as pilot:
        await pilot.pause()
        resumed_agent = resumed.agent
        assert isinstance(resumed_agent, BackgroundHarnessAgent)
        assert not _history_contains(resumed_agent.history, "job-workspace")
        assert [
            entry.content
            for entry in resumed.conversation_entries
            if entry.origin == {"kind": "subagent_completion", "job_id": "job-workspace"}
        ] == [original_report]
        assert resumed_agent.continuations == 0
        archived_entry = next(
            entry for entry in resumed.conversation_entries if (entry.origin or {}).get("job_id") == "job-workspace"
        )
        widget = await _completion_widget(resumed, pilot, archived_entry)
        disclosure = widget.query_one(Collapsible)
        assert disclosure.collapsed is True
        disclosure.collapsed = False
        await _wait_for(pilot, lambda: widget.region.height > 2, description="archived report expansion")
        assert widget._body is not None
        assert str(widget._body.render()) == original_report


@pytest.mark.parametrize("boundary", ["close", "archive", "cleanup"])
@pytest.mark.asyncio
async def test_cancelled_workspace_retirement_settles_presentation_and_keeps_current_root_usable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    boundary: str,
) -> None:
    app, store, session = _build_background_app(tmp_path, monkeypatch)
    other = tmp_path / "cancelled-workspace"
    other.mkdir()

    async with app.run_test() as pilot:
        await pilot.pause()
        agent = app.agent
        assert isinstance(agent, BackgroundHarnessAgent)
        manager = app._background_jobs
        assert isinstance(manager, BackgroundAgentManager)
        epoch = app._session_recorder.journal.epoch_id
        started = asyncio.Event()
        entered = asyncio.Event()
        release = threading.Event()
        cleanup_finished = asyncio.Event()
        loop = asyncio.get_running_loop()
        original_cleanup = app._cleanup_retired_agent_generations

        async def checked_cleanup() -> None:
            if boundary == "cleanup":
                entered.set()
                assert await asyncio.to_thread(release.wait, 10)
            await original_cleanup()
            cleanup_finished.set()

        monkeypatch.setattr(app, "_cleanup_retired_agent_generations", checked_cleanup)

        async def child() -> str:
            started.set()
            await asyncio.Event().wait()
            return "unexpected"

        await manager.submit("job-cancel-retirement", child, agent_type="investigation", task="inspect")
        await asyncio.wait_for(started.wait(), 2)
        app._render_event(
            _background_activity_event(
                agent_id="job-cancel-retirement", status="GENERATING", message="Starting background"
            )
        )
        if boundary == "close":
            original_close = manager.close

            async def blocked_close() -> None:
                entered.set()
                assert await asyncio.to_thread(release.wait, 10)
                await original_close()

            monkeypatch.setattr(manager, "close", blocked_close)
        elif boundary == "archive":
            original_commit = app._session_recorder.record_background_completion

            def blocked_commit(*args: Any, **kwargs: Any) -> bool:
                loop.call_soon_threadsafe(entered.set)
                assert release.wait(10)
                return original_commit(*args, **kwargs)

            monkeypatch.setattr(app._session_recorder, "record_background_completion", blocked_commit)

        transition = asyncio.create_task(app._activate_workspace(other, baseline_label="must not switch"))
        try:
            await asyncio.wait_for(entered.wait(), 2)
            app._queue_user_message("queued for the requested workspace")
            transition.cancel()
            await pilot.pause(0.02)
            transition.cancel()
            await pilot.pause(0.02)
            assert not transition.done()
            assert app.active_project_path == app.project_path
            assert app._background_rebuilding
        finally:
            release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(transition, 4)

        assert app.agent is agent
        assert app.active_project_path == app.project_path
        assert not app._background_rebuilding
        assert app._background_jobs is not None
        assert app._background_jobs is not manager
        assert agent.background_jobs is app._background_jobs
        assert cleanup_finished.is_set()
        assert not agent.background_needs_turn
        assert not app._background_resume_needed
        assert agent.continuations == 0
        await pilot.pause(0.05)
        assert agent.messages == []
        assert len(app._queued_messages) == 1
        app._clear_queued_messages()
        assert not _history_contains(agent.history, "job-cancel-retirement")
        assert not _history_contains(store.load(session.session_id).history, "job-cancel-retirement")
        assert _background_row(store, session.session_id, epoch, "job-cancel-retirement")["delivered"]
        reports = [
            entry
            for entry in app.conversation_entries
            if entry.origin == {"kind": "subagent_completion", "job_id": "job-cancel-retirement"}
        ]
        assert len(reports) == 1
        assert "Background job was cancelled." in reports[0].content
        activity = app._sub_agent_activities["job-cancel-retirement"]
        assert activity.status == "cancelled"
        assert activity.entry.complete
        assert not manager.has_work
        await app._background_jobs.recover()
        assert not app._background_jobs.has_work
        assert not app._maybe_start_background_completion()

        async def next_child() -> str:
            return "fresh manager still admits jobs"

        await app._background_jobs.submit("job-after-cancel", next_child, agent_type="investigation", task="inspect")
        await asyncio.wait_for(agent.continued.wait(), 2)
        await wait_for_turn_idle(app, pilot)
        assert agent.continuations == 1
        assert _history_contains(agent.history, "fresh manager still admits jobs")


@pytest.mark.asyncio
@pytest.mark.parametrize("fail_activation", [False, True])
async def test_cancelled_committed_workspace_switch_settles_activation_or_durable_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fail_activation: bool,
) -> None:
    app, store, session = _build_background_app(tmp_path, monkeypatch, initialize_git=True)
    other = tmp_path / "committed-workspace"
    _git(app.project_path, "worktree", "add", "-b", "committed-switch", str(other))

    async with app.run_test() as pilot:
        await pilot.pause()
        agent = app.agent
        assert isinstance(agent, BackgroundHarnessAgent)
        manager = app._background_jobs
        assert manager is not None
        started = asyncio.Event()
        entered = asyncio.Event()
        release = asyncio.Event()

        async def child() -> str:
            started.set()
            await asyncio.Event().wait()
            return "unexpected"

        await manager.submit("job-committed-switch", child, agent_type="investigation", task="inspect")
        await asyncio.wait_for(started.wait(), 2)
        original_close = manager.close

        async def blocked_close() -> None:
            entered.set()
            await release.wait()
            await original_close()

        monkeypatch.setattr(manager, "close", blocked_close)
        original_build = app._build_agent

        async def build(config, rebuild=False, *, restore_transcript=True, preserve_queued=False):
            if fail_activation and app.active_project_path == other:
                raise RuntimeError("new workspace build failed")
            await original_build(
                config, rebuild=rebuild, restore_transcript=restore_transcript, preserve_queued=preserve_queued
            )

        monkeypatch.setattr(app, "_build_agent", build)
        await _approve_switch(app, str(other))
        assert store.load(session.session_id).active_project_path == str(other)
        transition = asyncio.create_task(app._apply_pending_workspace_switch())
        try:
            await asyncio.wait_for(entered.wait(), 2)
            app._queue_user_message("do not automatically continue after cancellation")
            transition.cancel()
            await pilot.pause(0.01)
            transition.cancel()
            await pilot.pause(0.01)
            assert not transition.done()
            assert app._pending_workspace_switch is not None
            assert app._background_rebuilding
            assert not app._maybe_start_queued_message()
        finally:
            release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(transition, 4)
        active = app.project_path if fail_activation else other
        metadata = None if fail_activation else str(other)
        assert app.active_project_path == active
        assert app.session.active_project_path == metadata
        assert store.load(session.session_id).active_project_path == metadata
        assert app._pending_workspace_switch is None
        assert not app._background_rebuilding
        rebuilt = app.agent
        assert isinstance(rebuilt, BackgroundHarnessAgent)
        assert rebuilt is not agent
        assert rebuilt.kwargs["project_path"] == active
        await pilot.pause(0.05)
        assert rebuilt.messages == []
        assert rebuilt.continuations == 0
        assert len(app._queued_messages) == 1
        app._clear_queued_messages()
        assert app._background_jobs is not manager
        assert not rebuilt.background_needs_turn
        assert manager.snapshot("job-committed-switch")[0]["delivered"]
        boundaries = _workspace_events(store, session.session_id)
        assert len(boundaries) == (2 if fail_activation else 1)
        assert boundaries[-1].payload["new_root"] == str(active)


@pytest.mark.parametrize("archive", [False, True])
@pytest.mark.parametrize("failure", ["terminal", "completion"])
@pytest.mark.asyncio
async def test_failed_retirement_retains_report_policy_and_resource_generations_until_reattachment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    archive: bool,
    failure: str,
) -> None:
    app, store, session = _build_background_app(tmp_path, monkeypatch)
    cleaned_bundles: list[object] = []

    async def cleanup_bundle(bundle: object) -> None:
        cleaned_bundles.append(bundle)

    monkeypatch.setattr(agent_runtime_module, "cleanup_extension_bundle", cleanup_bundle)
    async with app.run_test() as pilot:
        await pilot.pause()
        old_agent = app.agent
        assert isinstance(old_agent, BackgroundHarnessAgent)
        manager = app._background_jobs
        assert manager is not None
        epoch = app._session_recorder.journal.epoch_id
        sentinel_bundle = cast(Any, object())
        app._extension_bundle = sentinel_bundle
        child_started = asyncio.Event()
        child_release = asyncio.Event()
        report = "complete original report, never replace this with interrupted"

        async def child() -> str:
            child_started.set()
            await child_release.wait()
            return report

        await manager.submit("job-retained-report", child, agent_type="investigation", task="inspect")
        await asyncio.wait_for(child_started.wait(), 2)
        assert app.config is not None
        await app._build_agent(app.config, rebuild=True)
        agent = app.agent
        assert isinstance(agent, BackgroundHarnessAgent)
        assert agent is not old_agent
        assert app._retired_agent_generations == [(old_agent, sentinel_bundle)]
        app._background_rebuilding = True
        entered = asyncio.Event()
        release = threading.Event()
        failing = True
        loop = asyncio.get_running_loop()
        method = f"record_background_job_{failure}" if failure == "terminal" else "record_background_completion"
        original_write = getattr(app._session_recorder, method)

        def write_outcome(*args: Any, **kwargs: Any) -> Any:
            if failing:
                loop.call_soon_threadsafe(entered.set)
                assert release.wait(10)
                raise OSError("terminal storage unavailable")
            return original_write(*args, **kwargs)

        monkeypatch.setattr(app._session_recorder, method, write_outcome)
        child_release.set()
        if failure == "terminal":
            try:
                await asyncio.wait_for(entered.wait(), 2)
            finally:
                release.set()
            with pytest.raises(OSError, match="storage unavailable"):
                await manager.wait()
        else:
            await manager.wait()
        retirement = asyncio.create_task(app._retire_background_jobs(archive=archive))
        try:
            await asyncio.wait_for(entered.wait(), 2)
        finally:
            release.set()
        with pytest.raises(OSError, match="storage unavailable"):
            await asyncio.wait_for(retirement, 4)
        assert manager.snapshot("job-retained-report")[0]["result"] == report
        assert app._background_jobs is manager
        assert agent.background_jobs is manager
        assert app._background_retirement == (manager, agent, archive)
        assert old_agent.cleanup_calls == 0
        assert cleaned_bundles == []
        app._background_rebuilding = False
        await app._cleanup_retired_agent_generations()
        assert old_agent.cleanup_calls == 0
        app._queue_user_message("must not run before retained retirement settles")
        assert not app._maybe_start_queued_message()
        # A later cleanup request must not detach the current generation or
        # replace the failed retirement's archive choice with its default.
        current_bundle = cast(Any, object())
        app._extension_bundle = current_bundle
        with pytest.raises(OSError, match="storage unavailable"):
            await app._cleanup_agent_generation()
        assert app.agent is agent
        assert app._extension_bundle is current_bundle
        assert agent.cleanup_calls == 0
        assert old_agent.cleanup_calls == 0
        assert app._background_retirement == (manager, agent, archive)
        await pilot.pause(0.05)
        assert agent.messages == []
        app._clear_queued_messages()
        failing = False
        await app._attach_background_jobs()
        assert app._background_retirement is None
        assert app._background_jobs is not manager
        assert agent.background_jobs is app._background_jobs
        assert old_agent.cleanup_calls == 1
        assert cleaned_bundles == [sentinel_bundle]
        assert app._retired_agent_generations == []
        row = _background_row(store, session.session_id, epoch, "job-retained-report")
        assert row["status"] == "completed"
        assert row["result"] == report
        assert row["delivered"]
        assert bool(row.get("archived")) is archive
        assert _history_contains(agent.history, report) is not archive
        assert _history_contains(store.load(session.session_id).history, report) is not archive
        assert not agent.background_needs_turn
        assert not app._background_resume_needed
        await pilot.pause(0.05)
        assert agent.messages == []
        assert agent.continuations == 0


@pytest.mark.asyncio
async def test_archived_background_projection_obeys_rewind_and_epoch_reset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app, store, session = _build_background_app(tmp_path, monkeypatch)

    async with app.run_test() as pilot:
        await pilot.pause()
        agent = app.agent
        assert isinstance(agent, BackgroundHarnessAgent)
        recorder = app._session_recorder
        _record_turn(recorder, "keep this turn")

        async def archive_report(job_id: str) -> None:
            app._background_rebuilding = True
            try:
                manager = app._background_jobs
                assert manager is not None
                child_started = asyncio.Event()
                child_release = asyncio.Event()

                async def child() -> str:
                    child_started.set()
                    await child_release.wait()
                    return f"archived report {job_id}"

                await manager.submit(job_id, child, agent_type="investigation", task="inspect")
                await asyncio.wait_for(child_started.wait(), 2)
                app._render_event(
                    _background_activity_event(agent_id=job_id, status="GENERATING", message="Starting background")
                )
                assert not app._sub_agent_activities[job_id].entry.complete
                child_release.set()
                await manager.wait()
                await app._retire_background_jobs(archive=True)
                await app._attach_background_jobs()
            finally:
                app._background_rebuilding = False

        await archive_report("job-keep-archive")
        _record_turn(recorder, "rewind this turn")
        target = recorder.list_rewindable_turns()[-1]
        await archive_report("job-remove-archive")
        assert set(app._sub_agent_activities) == {"job-keep-archive", "job-remove-archive"}
        assert all(activity.entry.complete for activity in app._sub_agent_activities.values())
        agent.restore_message_history(store.load(session.session_id).history)
        outcome = await asyncio.to_thread(recorder.record_rewind, target.turn_id)
        await app._apply_conversation_rewind(outcome)
        assert [
            entry.origin["job_id"]
            for entry in app.conversation_entries
            if entry.origin and entry.origin.get("kind") == "subagent_completion"
        ] == ["job-keep-archive"]
        assert set(app._sub_agent_activities) == {"job-keep-archive"}
        assert not any("job-remove-archive" in entry.content for entry in app.conversation_entries)
        assert not _history_contains(agent.history, "job-keep-archive")
        assert not _history_contains(agent.history, "job-remove-archive")
        assert app.config is not None
        await app._build_agent(app.config, rebuild=True)
        assert [
            entry.origin["job_id"]
            for entry in app.conversation_entries
            if entry.origin and entry.origin.get("kind") == "subagent_completion"
        ] == ["job-keep-archive"]

        await app._reset_current_thread()
        await app._build_agent(app.config, rebuild=True)
        assert not any(
            entry.origin and entry.origin.get("kind") == "subagent_completion" for entry in app.conversation_entries
        )
        assert app._sub_agent_activities == {}


@pytest.mark.asyncio
async def test_handoff_retires_old_jobs_into_old_session_without_late_results_in_new_history(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app, store, _ = _build_background_app(tmp_path, monkeypatch)
    handoff_doc = "## Goal\nFocus on the next workspace"

    async with app.run_test() as pilot:
        await pilot.pause()
        old_id = app.session.session_id
        old_agent = app.agent
        assert isinstance(old_agent, BackgroundHarnessAgent)
        old_manager = app._background_jobs
        assert isinstance(old_manager, BackgroundAgentManager)
        old_epoch = app._session_recorder.journal.epoch_id

        old_agent.history = [
            Message(role="user", content=[TextBlock("before handoff")]),
            Message(role="assistant", content=[TextBlock("ack")], stop_reason="end_turn"),
        ]

        child_cancelled = asyncio.Event()
        never = asyncio.Event()

        async def child() -> str:
            old_agent.lifecycle_events.append("child:start")
            try:
                await never.wait()
            finally:
                old_agent.lifecycle_events.append("child:cancel")
                child_cancelled.set()
            return "unexpected"

        await old_manager.submit("job-handoff", child, agent_type="investigation", task="inspect")
        await _wait_for(pilot, lambda: old_manager.active_count == 1, description="the handoff child to start")

        await app._handoff_session(handoff_doc)
        await pilot.pause(0.05)
        await asyncio.wait_for(child_cancelled.wait(), 2)

        new_id = app.session.session_id
        new_agent = app.agent
        assert isinstance(new_agent, BackgroundHarnessAgent)
        assert new_id != old_id
        assert app.session.parent_session_id == old_id

        row = _background_row(store, old_id, old_epoch, "job-handoff")
        assert row["status"] == "cancelled"
        assert row["delivered"] is True
        assert old_manager.snapshot("job-handoff")[0]["status"] == "cancelled"
        assert not old_manager.has_work

        new_record = store.load(new_id)
        old_events = store.journal(old_id).read_events()
        assert any(
            event.event_type == "context.message"
            and (event.payload or {}).get("job_id") == "job-handoff"
            and (event.payload or {}).get("status") == "cancelled"
            for event in old_events
        )
        assert _history_contains(new_record.history, handoff_doc)
        assert not _history_contains(new_record.history, "job-handoff")
        assert not _history_contains(new_record.history, "Background job was cancelled.")
        assert not _history_contains(new_agent.history, "job-handoff")
        assert not _history_contains(new_agent.history, "Background job was cancelled.")
        assert new_agent.appended_user_texts == []
        assert new_agent.continuations == 0
        assert not any(
            entry.origin == {"kind": "subagent_completion", "job_id": "job-handoff"}
            for entry in app.conversation_entries
        )


@pytest.mark.usefixtures("hermetic_git_config")
@pytest.mark.asyncio
async def test_rewind_retires_background_jobs_before_file_and_history_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app, store, session = _build_background_app(tmp_path, monkeypatch, initialize_git=True)

    async with app.run_test() as pilot:
        await pilot.pause()
        await wait_for_session_diff_baseline(app)

        agent = app.agent
        assert isinstance(agent, BackgroundHarnessAgent)
        old_manager = app._background_jobs
        assert isinstance(old_manager, BackgroundAgentManager)
        old_epoch = app._session_recorder.journal.epoch_id
        tracker = app._session_diff_tracker
        recorder = app._session_recorder
        assert tracker is not None

        tracker.capture_checkpoint("turn one")
        _record_turn(recorder, "turn one")
        (app.project_path / "src" / "a.py").write_text("turn1 edit\n", encoding="utf-8")

        second = tracker.capture_checkpoint("turn two")
        _record_turn(recorder, "turn two")
        (app.project_path / "src" / "b.py").write_text("turn2 edit\n", encoding="utf-8")

        child_cancelled = asyncio.Event()
        never = asyncio.Event()

        async def child() -> str:
            agent.lifecycle_events.append("child:start")
            try:
                await never.wait()
            finally:
                agent.lifecycle_events.append("child:cancel")
                child_cancelled.set()
            return "unexpected"

        await old_manager.submit("job-rewind", child, agent_type="investigation", task="inspect")
        await _wait_for(pilot, lambda: old_manager.active_count == 1, description="the rewind child to start")

        apply_checks: list[tuple[str, bool]] = []
        original_apply_restore_plan = tracker.apply_restore_plan

        def wrapped_apply_restore_plan(plan):
            row = _background_row(store, session.session_id, old_epoch, "job-rewind")
            apply_checks.append((str(row["status"]), bool(row["delivered"])))
            return original_apply_restore_plan(plan)

        monkeypatch.setattr(tracker, "apply_restore_plan", wrapped_apply_restore_plan)

        rewind_checks: list[tuple[str, bool]] = []
        original_record_rewind = recorder.record_rewind

        def wrapped_record_rewind(target_turn_id: str):
            row = _background_row(store, session.session_id, old_epoch, "job-rewind")
            rewind_checks.append((str(row["status"]), bool(row["delivered"])))
            return original_record_rewind(target_turn_id)

        monkeypatch.setattr(recorder, "record_rewind", wrapped_record_rewind)

        await app._rewind_worker(second.checkpoint_id, "Turn 2")
        await pilot.pause(0.05)
        await asyncio.wait_for(child_cancelled.wait(), 2)

        assert apply_checks == [("cancelled", True)]
        assert rewind_checks == [("cancelled", True)]
        assert old_manager.snapshot("job-rewind")[0]["status"] == "cancelled"
        assert not old_manager.has_work
        assert app._background_jobs is not old_manager

        assert (app.project_path / "src" / "a.py").read_text(encoding="utf-8") == "turn1 edit\n"
        assert (app.project_path / "src" / "b.py").read_text(encoding="utf-8") == "old b\n"
        assert _history_texts(store.load(session.session_id).history) == ["turn one", "done"]
        assert not _history_contains(store.load(session.session_id).history, "job-rewind")
        assert not _history_contains(store.load(session.session_id).history, "Background job was cancelled.")
        # Rewind deliberately removes the insertion from active history, not its
        # acknowledgment: recovery must not resurrect a discarded completion.
        assert app._background_jobs is not None
        await app._background_jobs.recover()
        assert not app._background_jobs.has_work
        assert _background_row(store, session.session_id, old_epoch, "job-rewind")["delivered"] is True


@pytest.mark.asyncio
async def test_parent_turn_finalization_does_not_stop_background_cards(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app, _, _ = _build_background_app(tmp_path, monkeypatch)

    async with app.run_test():
        foreground = _sub_agent_event(agent_id="agent-1", status="GENERATING", message="Starting foreground")
        background = _background_activity_event(
            agent_id="agent-2",
            parent_tool_call_id="tc-2",
            status="GENERATING",
            message="Starting background",
        )
        app._render_event(foreground)
        app._render_event(background)

        activities = list(app._sub_agent_activities.values())
        assert len(activities) == 2
        foreground_activity = next(activity for activity in activities if not activity.background)
        background_activity = next(activity for activity in activities if activity.background)

        app._finalize_sub_agent_activities()

        assert foreground_activity.status == "stopped"
        assert foreground_activity.entry.complete is True
        assert background_activity.status == "running"
        assert background_activity.entry.complete is False


@pytest.mark.asyncio
async def test_restored_background_completion_stays_visible_and_not_as_user_reminder(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app, _, _ = _build_background_app(tmp_path, monkeypatch)

    completion_text = BackgroundAgentManager._completion_text(
        {
            "job_id": "job-restore",
            "completion_id": "completion-restore",
            "agent_type": "investigation",
            "status": "completed",
            "result": "restored child report",
        }
    )
    history = [
        Message(
            role="user",
            content=[TextBlock('<system-reminder source="memory">\nMemory context\n</system-reminder>')],
        ).to_dict(),
        Message(role="user", content=[TextBlock(completion_text)]).to_dict(),
    ]

    async with app.run_test():
        # The same report can appear in both the current root's history and an
        # archived UI-only projection; restoration must not duplicate it.
        app._restore_conversation_history([*history, history[-1]])

        non_startup = [entry for entry in app.conversation_entries if entry.kind != "startup"]
        assert len(non_startup) == 1
        entry = non_startup[0]
        assert entry.kind == "system"
        assert entry.origin == {"kind": "subagent_completion", "job_id": "job-restore"}
        assert "Background investigation completed (job-restore):" in entry.content
        assert "restored child report" in entry.content
        assert not any(item.kind == "user" for item in non_startup)
