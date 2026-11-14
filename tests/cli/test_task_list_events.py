from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest

from kolega_code.cli.app import KolegaCodeApp
from kolega_code.cli.config import config_summary
from kolega_code.cli.session_event_store import FileSessionEventStore
from kolega_code.cli.session_store import SessionStore
from kolega_code.cli.tui.widgets import PlanningMarkdown
from kolega_code.events import AgentEvent, KnownEventType
from kolega_code.session.recording import RecordingConnectionManager
from kolega_code.tools import ToolError

from ._app_test_utils import FakeCoderAgent, build_test_config, extension_by_name, install_fake_agents

pytestmark = pytest.mark.usefixtures("isolated_cli_env", "hermetic_git_config")


@pytest.fixture
def app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> KolegaCodeApp:
    install_fake_agents(monkeypatch, planning_cls=FakeCoderAgent)
    project = tmp_path / "project"
    project.mkdir()
    config = build_test_config(project)
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", config_summary(config))
    return KolegaCodeApp(project_path=project, config=config, mode="code", store=store, session=session)


async def task_events(app: KolegaCodeApp) -> list[AgentEvent]:
    return [
        event
        for event in await FileSessionEventStore(app.store.journal(app.session.session_id)).read(app.session.session_id)
        if event.event_type == KnownEventType.TASK_LIST_UPDATED
    ]


@pytest.mark.asyncio
async def test_live_task_snapshots_are_saved_and_sidebar_refreshed_before_recording(
    app: KolegaCodeApp, monkeypatch: pytest.MonkeyPatch
) -> None:
    async with app.run_test():
        assert isinstance(app.agent, FakeCoderAgent)
        extension = extension_by_name(app.agent.kwargs["tool_extensions"], "cli-shared-task-list")
        initial = await task_events(app)
        assert initial[-1].content == {"markdown": ""}
        observed: list[str] = []
        broadcast = app.recording_connection_manager.broadcast_event

        async def observe(event: AgentEvent, workspace_id: str, thread_id: str) -> None:
            if event.event_type == KnownEventType.TASK_LIST_UPDATED:
                markdown = event.content["markdown"]
                assert app.store.load(app.session.session_id).task_list_markdown == markdown
                sidebar = app.query_one("#status_task_list_markdown", PlanningMarkdown)
                assert sidebar.source == (markdown or "Task List · Not set")
                observed.append(markdown)
            await broadcast(event, workspace_id, thread_id)

        monkeypatch.setattr(app.recording_connection_manager, "broadcast_event", observe)
        markdown = "- [x] inspect\n- [ ] implement"
        assert await extension.tools["update_task_list"](f" \n{markdown}\n ") == "Task list updated."
        assert await extension.tools["update_task_list"](" \n ") == "Task list updated."
        assert observed == [markdown, ""]
        events = await task_events(app)
        assert [event.content["markdown"] for event in events[len(initial) :]] == [markdown, ""]
        for event in events:
            assert event.sender == "cli"
            assert event.session_id == app.session.session_id
            assert event.workspace_id == app.session.workspace_id
            assert event.thread_id == app.session.thread_id
            assert event.sub_agent_info is None
            assert not event.is_streaming
            assert event.seq is not None
        sequences = [event.seq for event in events if event.seq is not None]
        assert sequences == sorted(set(sequences))


@pytest.mark.asyncio
async def test_restored_legacy_metadata_gets_an_initial_recorded_snapshot(app: KolegaCodeApp) -> None:
    markdown = "- [x] written before task events existed\n- [ ] resume"
    app.session.task_list_markdown = markdown
    app.store.save(app.session)
    app.session = app.store.load(app.session.session_id)
    assert await task_events(app) == []

    async with app.run_test():
        events = await task_events(app)
        assert [event.content for event in events] == [{"markdown": markdown}]
        assert app.query_one("#status_task_list_markdown", PlanningMarkdown).source == markdown
        assert app.store.load(app.session.session_id).task_list_markdown == markdown


@pytest.mark.asyncio
async def test_reset_records_a_clear_snapshot_in_the_new_epoch(
    app: KolegaCodeApp, monkeypatch: pytest.MonkeyPatch
) -> None:
    async with app.run_test():
        assert isinstance(app.agent, FakeCoderAgent)
        tools = extension_by_name(app.agent.kwargs["tool_extensions"], "cli-shared-task-list").tools
        await tools["update_task_list"]("- [ ] old work")
        broadcast = app.recording_connection_manager.broadcast_event

        async def observe(event: AgentEvent, workspace_id: str, thread_id: str) -> None:
            if event.event_type == KnownEventType.TASK_LIST_UPDATED:
                assert event.content == {"markdown": ""}
                assert app.store.load(app.session.session_id).task_list_markdown == ""
                assert app.query_one("#status_task_list_markdown", PlanningMarkdown).source == "Task List · Not set"
            await broadcast(event, workspace_id, thread_id)

        monkeypatch.setattr(app.recording_connection_manager, "broadcast_event", observe)
        await app._reset_current_thread()

        events = await task_events(app)
        assert events[-1].content == {"markdown": ""}
        assert app.session.task_list_markdown == ""
        assert app.store.load(app.session.session_id).task_list_markdown == ""
        rows = app.store.journal(app.session.session_id).read_events()
        clear = next(row for row in reversed(rows) if row.event_type == "ui.task_list_updated")
        epoch = next(row for row in reversed(rows) if row.event_type == "context.epoch_started")
        assert clear.seq > epoch.seq
        assert clear.epoch_id == epoch.epoch_id


@pytest.mark.asyncio
async def test_failed_reset_save_does_not_publish_a_non_durable_clear(
    app: KolegaCodeApp, monkeypatch: pytest.MonkeyPatch
) -> None:
    async with app.run_test():
        assert isinstance(app.agent, FakeCoderAgent)
        tools = extension_by_name(app.agent.kwargs["tool_extensions"], "cli-shared-task-list").tools
        await tools["update_task_list"]("- [ ] saved work")
        before = await task_events(app)
        with monkeypatch.context() as scoped:
            scoped.setattr(app, "_save_session_async", AsyncMock(side_effect=OSError("reset save failed")))
            with pytest.raises(OSError, match="reset save failed"):
                await app._reset_current_thread()
            assert await task_events(app) == before
            assert app.store.load(app.session.session_id).task_list_markdown == "- [ ] saved work"


@pytest.mark.asyncio
@pytest.mark.parametrize("blocked_stage", ["save", "emit"])
@pytest.mark.parametrize("preserve_loop", [False, True])
async def test_cancelled_reset_drains_save_and_clear_snapshot_before_releasing_ownership(
    app: KolegaCodeApp,
    monkeypatch: pytest.MonkeyPatch,
    blocked_stage: str,
    preserve_loop: bool,
) -> None:
    async with app.run_test():
        assert isinstance(app.agent, FakeCoderAgent)
        tools = extension_by_name(app.agent.kwargs["tool_extensions"], "cli-shared-task-list").tools
        await tools["update_task_list"]("- [ ] old work")
        before = await task_events(app)
        entered = asyncio.Event()
        release = asyncio.Event()
        save = app._save_session_async
        broadcast = app.recording_connection_manager.broadcast_event

        async def gated_save() -> None:
            if blocked_stage == "save":
                entered.set()
                await release.wait()
            await save()

        async def gated_emit(event: AgentEvent, workspace_id: str, thread_id: str) -> None:
            if event.event_type == KnownEventType.TASK_LIST_UPDATED and blocked_stage == "emit":
                assert app.store.load(app.session.session_id).task_list_markdown == ""
                entered.set()
                await release.wait()
            await broadcast(event, workspace_id, thread_id)

        with monkeypatch.context() as scoped:
            scoped.setattr(app, "_save_session_async", gated_save)
            scoped.setattr(app.recording_connection_manager, "broadcast_event", gated_emit)
            caller = asyncio.create_task(app._reset_current_thread(preserve_loop=preserve_loop))
            try:
                await asyncio.wait_for(entered.wait(), timeout=10)
                for _ in range(3):
                    caller.cancel()
                    await asyncio.sleep(0)
                    assert not caller.done(), "reset must retain the save and clear snapshot owner"
                    assert app._background_rebuilding
                release.set()
                with pytest.raises(asyncio.CancelledError):
                    await asyncio.wait_for(caller, timeout=10)
                events = await task_events(app)
                assert [event.content for event in events[len(before) :]] == [{"markdown": ""}]
                assert app.store.load(app.session.session_id).task_list_markdown == ""
                assert app.query_one("#status_task_list_markdown", PlanningMarkdown).source == "Task List · Not set"
                assert not app._background_rebuilding
                assert not app._pending_session_saves
                rows = app.store.journal(app.session.session_id).read_events()
                clear = next(row for row in reversed(rows) if row.event_type == "ui.task_list_updated")
                epoch = next(row for row in reversed(rows) if row.event_type == "context.epoch_started")
                assert clear.seq > epoch.seq
                assert clear.epoch_id == epoch.epoch_id
            finally:
                release.set()
                await asyncio.gather(caller, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("invalidated_by", ["rebuild", "readonly", "missing_agent", "exit", "session_switch"])
async def test_inactive_task_list_writers_cannot_mutate_save_or_publish(
    app: KolegaCodeApp, monkeypatch: pytest.MonkeyPatch, invalidated_by: str
) -> None:
    async with app.run_test():
        assert app.config is not None
        assert isinstance(app.agent, FakeCoderAgent)
        writer = extension_by_name(app.agent.kwargs["tool_extensions"], "cli-shared-task-list").tools[
            "update_task_list"
        ]
        original_session = app.session
        original_agent = app.agent
        if invalidated_by == "rebuild":
            await app._build_agent(app.config, rebuild=True)
        elif invalidated_by == "readonly":
            await app._set_interaction_mode("plan")
            assert isinstance(app.agent, FakeCoderAgent)
            readonly = extension_by_name(app.agent.kwargs["tool_extensions"], "cli-shared-task-list-readonly")
            assert set(readonly.tools) == {"get_task_list"}
            assert await readonly.tools["get_task_list"]() == "No task list has been set."
            assert readonly.propagate_to_sub_agents is False
        elif invalidated_by == "missing_agent":
            app.agent = None
        elif invalidated_by == "exit":
            app._exit = True
        else:
            app.session = app.store.create(app.project_path, "code", config_summary(app.config))
        before = await task_events(app)
        save = AsyncMock(wraps=app._save_session_async)
        refresh = Mock(wraps=app._refresh_planning_sidebar)
        with monkeypatch.context() as scoped:
            scoped.setattr(app, "_save_session_async", save)
            scoped.setattr(app, "_refresh_planning_sidebar", refresh)
            try:
                with pytest.raises(ToolError, match="inactive agent"):
                    await writer("- [ ] stale write")
                assert original_session.task_list_markdown == ""
                assert app.session.task_list_markdown == ""
                assert await task_events(app) == before
                save.assert_not_awaited()
                refresh.assert_not_called()
            finally:
                app.session = original_session
                app._exit = False
                if invalidated_by == "missing_agent":
                    app.agent = original_agent


@pytest.mark.asyncio
async def test_task_list_save_failure_does_not_refresh_or_publish(
    app: KolegaCodeApp, monkeypatch: pytest.MonkeyPatch
) -> None:
    async with app.run_test():
        assert isinstance(app.agent, FakeCoderAgent)
        tools = extension_by_name(app.agent.kwargs["tool_extensions"], "cli-shared-task-list").tools
        before = await task_events(app)
        refresh = Mock(wraps=app._refresh_planning_sidebar)
        with monkeypatch.context() as scoped:
            scoped.setattr(app, "_save_session_async", AsyncMock(side_effect=OSError("save failed")))
            scoped.setattr(app, "_refresh_planning_sidebar", refresh)
            with pytest.raises(OSError, match="save failed"):
                await tools["update_task_list"]("- [ ] not durable")
            refresh.assert_not_called()
            assert await task_events(app) == before
            assert app.store.load(app.session.session_id).task_list_markdown == ""


@pytest.mark.asyncio
@pytest.mark.parametrize("blocked_stage", ["save", "emit"])
async def test_cancelled_task_list_update_finishes_recording_once_before_caller_exits(
    app: KolegaCodeApp, monkeypatch: pytest.MonkeyPatch, blocked_stage: str
) -> None:
    async with app.run_test():
        assert isinstance(app.agent, FakeCoderAgent)
        tools = extension_by_name(app.agent.kwargs["tool_extensions"], "cli-shared-task-list").tools
        before = await task_events(app)
        entered = asyncio.Event()
        release = asyncio.Event()
        owners: list[asyncio.Task | None] = []
        markdown = "- [x] durable despite repeated cancellation"
        save = app._save_session_async
        broadcast = app.recording_connection_manager.broadcast_event

        async def pause() -> None:
            owners.append(asyncio.current_task())
            entered.set()
            await release.wait()

        async def gated_save() -> None:
            if blocked_stage == "save":
                await pause()
            await save()

        async def gated_emit(event: AgentEvent, workspace_id: str, thread_id: str) -> None:
            if event.event_type == KnownEventType.TASK_LIST_UPDATED and blocked_stage == "emit":
                assert app.store.load(app.session.session_id).task_list_markdown == markdown
                await pause()
            await broadcast(event, workspace_id, thread_id)

        monkeypatch.setattr(app, "_save_session_async", gated_save)
        monkeypatch.setattr(app.recording_connection_manager, "broadcast_event", gated_emit)
        caller = asyncio.create_task(tools["update_task_list"](markdown))
        try:
            await asyncio.wait_for(entered.wait(), timeout=10)
            for _ in range(3):
                caller.cancel()
                await asyncio.sleep(0)
                assert not caller.done()
            assert await task_events(app) == before
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(caller, timeout=10)
            events = await task_events(app)
            assert [event.content for event in events[len(before) :]] == [{"markdown": markdown}]
            assert app.store.load(app.session.session_id).task_list_markdown == markdown
            assert app.query_one("#status_task_list_markdown", PlanningMarkdown).source == markdown
            assert owners[0] is not None and owners[0].done()
            assert not app._pending_session_saves
            await asyncio.sleep(0)
            assert await task_events(app) == events
        finally:
            release.set()
            await asyncio.gather(caller, return_exceptions=True)


@pytest.mark.asyncio
async def test_inflight_update_keeps_its_session_and_recorder_after_surface_switch(
    app: KolegaCodeApp, monkeypatch: pytest.MonkeyPatch
) -> None:
    async with app.run_test():
        assert app.config is not None
        assert isinstance(app.agent, FakeCoderAgent)
        writer = extension_by_name(app.agent.kwargs["tool_extensions"], "cli-shared-task-list").tools[
            "update_task_list"
        ]
        original_session = app.session
        original_recording = app.recording_connection_manager
        other = app.store.create(app.project_path, "code", config_summary(app.config))
        other_events = FileSessionEventStore(app.store.journal(other.session_id))
        replacement = RecordingConnectionManager(app.connection_manager, other_events, session_id=other.session_id)
        saved = asyncio.Event()
        release = asyncio.Event()
        save = app._save_session_async
        markdown = "- [x] belongs to the original session"

        async def gated_save() -> None:
            await save()
            saved.set()
            await release.wait()

        refresh = Mock(wraps=app._refresh_planning_sidebar)
        with monkeypatch.context() as scoped:
            scoped.setattr(app, "_save_session_async", gated_save)
            scoped.setattr(app, "_refresh_planning_sidebar", refresh)
            caller = asyncio.create_task(writer(markdown))
            try:
                await asyncio.wait_for(saved.wait(), timeout=10)
                app.session = other
                app.recording_connection_manager = replacement
                release.set()
                assert await asyncio.wait_for(caller, timeout=10) == "Task list updated."
                refresh.assert_not_called()
                original_events = await FileSessionEventStore(app.store.journal(original_session.session_id)).read(
                    original_session.session_id
                )
                snapshot = next(
                    event for event in reversed(original_events) if event.event_type == KnownEventType.TASK_LIST_UPDATED
                )
                assert snapshot.content == {"markdown": markdown}
                assert snapshot.session_id == original_session.session_id
                assert snapshot.workspace_id == original_session.workspace_id
                assert snapshot.thread_id == original_session.thread_id
                assert await other_events.read(other.session_id) == []
                assert app.store.load(original_session.session_id).task_list_markdown == markdown
                assert app.store.load(other.session_id).task_list_markdown == ""
            finally:
                release.set()
                await asyncio.gather(caller, return_exceptions=True)
                app.session = original_session
                app.recording_connection_manager = original_recording
