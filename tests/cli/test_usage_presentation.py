from __future__ import annotations

import asyncio
from pathlib import Path
import threading
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest

from kolega_code.cli.session_event_store import FileSessionEventStore
from kolega_code.cli.session_store import SessionStore
from kolega_code.cli.session_usage import LLM_RUN_FINISHED_EVENT, SessionUsageSink, derive_session_usage
from kolega_code.events import AgentEvent
from kolega_code.llm.accounting import UsageObservation
from kolega_code.llm.ledger import UsageLedger
from kolega_code.llm.models import Message
from kolega_code.llm.usage import normalize_usage
from kolega_code.session.projection import replay
from kolega_code.session.usage import UsagePresentation

from ._app_test_utils import FakeCoderAgent, build_test_config, install_fake_agents
from .test_app_handoff import _build_handoff_test_app

pytestmark = [pytest.mark.asyncio, pytest.mark.usefixtures("isolated_cli_env")]


def _settle(ledger: UsageLedger, inp: int, out: int) -> tuple[str, Message]:
    usage = normalize_usage({"input_tokens": inp, "output_tokens": out}, "anthropic", "usage-test")
    request_id = ledger.begin("anthropic", "usage-test")
    message = Message(role="assistant", content="done", usage=usage)
    ledger.record_response(request_id, usage, message=message)
    return request_id, message


async def _events(store: SessionStore, session_id: str) -> list[AgentEvent]:
    return await FileSessionEventStore(store.journal(session_id)).read(session_id)


def _root_usage(events: list[AgentEvent]) -> list[AgentEvent]:
    return [event for event in events if event.event_type == "usage_updated" and not event.sub_agent_info]


async def test_resumed_epochs_and_revisions_publish_literal_session_totals_across_rebuilds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from kolega_code.cli.app import KolegaCodeApp

    install_fake_agents(monkeypatch, planning_cls=FakeCoderAgent)
    project = tmp_path / "project"
    project.mkdir()
    config = build_test_config(project)
    store = SessionStore(tmp_path / "state")
    record = store.create(project, "code", {})
    journal = store.journal(record.session_id)
    recorder = store.recorder(record.session_id)
    prior = UsageLedger()
    sink = SessionUsageSink(journal, recorder, prior, mode="tui")
    prior.observer = sink
    await sink.start()
    try:
        recorder.start_turn(Message(role="user", content="prior turn"))
        _, message = _settle(prior, 10, 5)
        recorder.record_assistant(message)
        recorder.finish_turn("completed")
    finally:
        await sink.wait_closed()
    prior_epoch = journal.epoch_id
    recorder.start_epoch("clear")
    assert journal.epoch_id != prior_epoch
    app = KolegaCodeApp(
        project_path=project, config=config, mode="code", store=store, session=store.load(record.session_id)
    )
    constructions: list[tuple[Any, Any]] = []
    initialize = UsagePresentation.__init__

    def observed_init(owner: UsagePresentation, *args: Any, **kwargs: Any) -> None:
        assert app._usage_sink is not None
        assert app._usage_sink._start_task is not None and app._usage_sink._start_task.done()
        constructions.append((app.agent, app._usage_sink))
        initialize(owner, *args, **kwargs)

    monkeypatch.setattr(UsagePresentation, "__init__", observed_init)
    async with app.run_test():
        await asyncio.wait_for(app._startup_complete.wait(), 10)
        owner = app._usage_presentation
        assert owner is not None
        assert constructions == [(app.agent, app._usage_sink)]
        assert app.agent is not None and app._usage_sink is not None
        await owner.flush()
        initial = _root_usage(await _events(store, record.session_id))[-1]
        assert initial.content["usage"]["requests"] == 1
        assert initial.content["usage"]["total_tokens"] == 15

        app._session_recorder.start_turn(Message(role="user", content="new epoch"))
        request_id, message = _settle(app._usage_ledger, 20, 7)
        app._session_recorder.record_assistant(message)
        app._session_recorder.finish_turn("completed")
        assert await app._usage_sink.flush()
        await owner.flush()
        snapshot = _root_usage(await _events(store, record.session_id))[-1]
        assert snapshot.content["usage"]["requests"] == 2
        assert snapshot.content["usage"]["input_tokens"] == 30
        assert snapshot.content["usage"]["output_tokens"] == 12
        assert snapshot.content["usage"]["total_tokens"] == 42
        assert snapshot.content["usage"]["coverage"]["full"] is True

        first_agent = app.agent
        await app._build_agent(config, rebuild=True)
        assert app.agent is not first_agent
        assert app._usage_presentation is owner
        await app._set_interaction_mode("plan")
        assert app._usage_presentation is owner
        assert len(constructions) == 1

        revised = normalize_usage({"input_tokens": 25, "output_tokens": 9}, "anthropic", "usage-test")
        app._usage_ledger.record_observation(request_id, UsageObservation.from_usage(revised))
        assert await app._usage_sink.flush()
        await owner.flush()
        events = await _events(store, record.session_id)
        snapshot = _root_usage(events)[-1]
        assert snapshot.content["scope"] == "session"
        assert snapshot.content["semantics"] == "snapshot"
        assert type(snapshot.content["source_seq"]) is int
        assert snapshot.session_id == record.session_id
        assert snapshot.workspace_id == record.workspace_id
        assert snapshot.thread_id == record.thread_id
        assert snapshot.sub_agent_info is None
        assert snapshot.content["usage"]["requests"] == 2
        assert snapshot.content["usage"]["input_tokens"] == 35
        assert snapshot.content["usage"]["output_tokens"] == 14
        assert snapshot.content["usage"]["total_tokens"] == 49
        facts = [event for event in journal.read_events() if event.event_type == "llm.usage_observed"]
        assert snapshot.content["source_seq"] >= facts[-1].seq
        assert snapshot.seq is not None and snapshot.content["source_seq"] < snapshot.seq

        await app.recording_connection_manager.broadcast_event(
            AgentEvent(
                event_type="usage_updated",
                sender="agent",
                sub_agent_info={"agent_name": "worker", "dispatch_id": "worker-1"},
                content={
                    "scope": "session",
                    "semantics": "snapshot",
                    "source_seq": snapshot.content["source_seq"] + 100,
                    "usage": {"requests": 999, "total_tokens": 999_999},
                },
            ),
            record.workspace_id,
            record.thread_id,
        )
        await owner.flush()
        events = await _events(store, record.session_id)
        assert _root_usage(events)[-1].content == snapshot.content
        assert replay(events).usage == {"source_seq": snapshot.content["source_seq"], **snapshot.content["usage"]}
        await app.action_quit()
    assert owner._closed and owner._unsubscribe is None
    assert owner._worker is not None and owner._worker.done()


async def test_no_config_still_gets_a_session_owned_publisher(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from kolega_code.cli.app import KolegaCodeApp

    project = tmp_path / "project"
    project.mkdir()
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", {})
    app = KolegaCodeApp(project_path=project, config=None, mode="code", store=store, session=session)
    app._onboarding_skipped = True
    ensure_agent = AsyncMock()
    monkeypatch.setattr(app, "_ensure_agent_from_settings", ensure_agent)
    async with app.run_test():
        await asyncio.wait_for(app._startup_complete.wait(), 10)
        ensure_agent.assert_awaited_once()
        assert app.agent is None
        owner = app._usage_presentation
        assert owner is not None
        assert app._usage_sink is not None
        await owner.flush()
        snapshot = _root_usage(await _events(store, session.session_id))[-1]
        assert snapshot.content["usage"]["requests"] == 0
        assert snapshot.content["usage"]["total_tokens"] == 0
        await app.action_quit()
    assert owner._closed and owner._unsubscribe is None


async def test_usage_snapshot_is_recording_only_not_tui_transcript_or_logs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = _build_handoff_test_app(tmp_path, monkeypatch)
    app.show_logs = True
    async with app.run_test():
        try:
            owner = app._usage_presentation
            assert owner is not None
            await owner.flush()
            snapshot = _root_usage(await _events(app.store, app.session.session_id))[-1]
            entries = list(app.conversation_entries)
            write_log = Mock()
            monkeypatch.setattr(app, "_write_log", write_log)
            app._render_event(snapshot)
            write_log.assert_not_called()
            assert app.conversation_entries == entries
        finally:
            await app.action_quit()


async def test_direct_exit_keeps_publisher_until_launcher_finishes_accounting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from kolega_code.cli.main import _close_tui_app

    app = _build_handoff_test_app(tmp_path, monkeypatch)
    try:
        async with app.run_test():
            await asyncio.wait_for(app._startup_complete.wait(), 15)
            owner, sink = app._usage_presentation, app._usage_sink
            assert owner is not None and sink is not None
            _settle(app._usage_ledger, 13, 4)
            assert await sink.flush()
            app.exit()
        assert not owner._closed
        await _close_tui_app(app)
        assert sink.close_result is not None and sink.close_result.committed
        journal = app.store.journal(app.session.session_id)
        final = _root_usage(await _events(app.store, app.session.session_id))[-1]
        assert final.content["usage"] == derive_session_usage(journal.read_events(repair_tail=False))
        assert final.content["usage"]["total_tokens"] == 17
        assert final.content["usage"]["persistence"]["healthy"] is True
        assert final.content["usage"]["coverage"]["unclosed_runs"] == 0
        assert owner._closed and owner._unsubscribe is None
    finally:
        await _close_tui_app(app)
        app.store.release_session_locks()


async def test_quit_during_initial_snapshot_retains_publisher_for_accounting_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from textual.app import App

    from kolega_code.cli.main import _close_tui_app

    app = _build_handoff_test_app(tmp_path, monkeypatch)
    entered, release = threading.Event(), threading.Event()
    read = UsagePresentation._read_snapshot

    def blocked(owner: UsagePresentation, live_run_id: str | None) -> tuple[int, dict[str, Any]]:
        entered.set()
        assert release.wait(30)
        return read(owner, live_run_id)

    monkeypatch.setattr(UsagePresentation, "_read_snapshot", blocked)
    quitting: asyncio.Task[None] | None = None
    try:
        async with App.run_test(app):
            try:
                assert await asyncio.to_thread(entered.wait, 15)
                owner, sink = app._usage_presentation, app._usage_sink
                startup = app._startup_task
                assert owner is not None and sink is not None and startup is not None
                quitting = asyncio.create_task(app.action_quit())
                async with asyncio.timeout(15):
                    while not startup.cancelling():
                        await asyncio.sleep(0)
                await asyncio.sleep(0)
                assert not owner._closing
                assert sink.close_result is None
                assert not quitting.done()
            finally:
                release.set()
                if quitting is not None:
                    await asyncio.wait_for(quitting, 15)
        await _close_tui_app(app)
        assert sink.close_result is not None and sink.close_result.committed
        journal = app.store.journal(app.session.session_id)
        final = _root_usage(await _events(app.store, app.session.session_id))[-1]
        assert final.content["usage"] == derive_session_usage(journal.read_events(repair_tail=False))
        assert final.content["usage"]["persistence"]["healthy"] is True
        assert final.content["usage"]["coverage"]["unclosed_runs"] == 0
        assert owner._closed and owner._unsubscribe is None
    finally:
        release.set()
        if quitting is not None:
            await asyncio.gather(quitting, return_exceptions=True)
        await _close_tui_app(app)
        app.store.release_session_locks()


@pytest.mark.parametrize("action", ["handoff", "quit"])
async def test_cancelled_finish_writer_keeps_publisher_until_sink_drains_and_handoff_is_fresh(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, action: str
) -> None:
    app = _build_handoff_test_app(tmp_path, monkeypatch)
    entered, release = threading.Event(), threading.Event()
    async with app.run_test():
        await asyncio.wait_for(app._startup_complete.wait(), 10)
        old_id = app.session.session_id
        journal = app.store.journal(old_id)
        owner, sink = app._usage_presentation, app._usage_sink
        assert owner is not None and sink is not None
        _settle(app._usage_ledger, 100, 20)
        assert await sink.flush()
        await owner.flush()
        close = owner.aclose
        closes: list[str] = []

        async def observed_close() -> None:
            assert sink.close_result is not None and sink.close_result.committed
            assert app.session.session_id == old_id
            await close()
            closes.append(old_id)

        monkeypatch.setattr(owner, "aclose", observed_close)
        initialize = UsagePresentation.__init__
        handoff_constructions: list[UsagePresentation] = []

        def observed_init(fresh: UsagePresentation, *args: Any, **kwargs: Any) -> None:
            assert app.session.session_id != old_id
            assert app._usage_sink is not None and app._usage_sink is not sink
            assert app._usage_sink._start_task is not None and app._usage_sink._start_task.done()
            initialize(fresh, *args, **kwargs)
            handoff_constructions.append(fresh)

        monkeypatch.setattr(UsagePresentation, "__init__", observed_init)
        append = journal.append

        def blocked_append(event_type: str, **kwargs: Any) -> Any:
            if event_type == LLM_RUN_FINISHED_EVENT:
                entered.set()
                assert release.wait(30), "finish writer was not released"
            return append(event_type, **kwargs)

        monkeypatch.setattr(journal, "append", blocked_append)
        task = asyncio.create_task(
            app._handoff_session("continue with a fresh session") if action == "handoff" else app._shutdown_session()
        )
        try:
            assert await asyncio.to_thread(entered.wait, 10)
            task.cancel()
            cancellation_turn = asyncio.Event()
            asyncio.get_running_loop().call_soon(cancellation_turn.set)
            await cancellation_turn.wait()
            task.cancel()
            assert not task.done()
            assert app._usage_presentation is owner
            assert app.session.session_id == old_id
            assert closes == [] and not owner._closed
            release.set()
            await asyncio.wait_for(task, 10)
            assert closes == [old_id]
            assert owner._closed and owner._unsubscribe is None
            old_events = await _events(app.store, old_id)
            assert _root_usage(old_events)[-1].content["usage"]["requests"] == 1
            assert _root_usage(old_events)[-1].content["usage"]["total_tokens"] == 120
            if action == "handoff":
                fresh = app._usage_presentation
                assert fresh is not None and fresh is not owner
                assert handoff_constructions == [fresh]
                assert app._usage_sink is not None and app._usage_sink is not sink
                assert app.session.session_id != old_id
                await fresh.flush()
                events = await _events(app.store, app.session.session_id)
                assert _root_usage(events)[-1].content["usage"]["requests"] == 0
                assert _root_usage(events)[-1].content["usage"]["total_tokens"] == 0
                assert app._usage_ledger.snapshot().total_tokens == 120
                _settle(app._usage_ledger, 7, 3)
                assert await app._usage_sink.flush()
                await fresh.flush()
                snapshot = _root_usage(await _events(app.store, app.session.session_id))[-1]
                assert snapshot.content["usage"]["requests"] == 1
                assert snapshot.content["usage"]["total_tokens"] == 10
                assert all(
                    event.session_id == app.session.session_id
                    for event in await _events(app.store, app.session.session_id)
                )
                assert await _events(app.store, old_id) == old_events
                await app.action_quit()
                assert fresh._closed
        finally:
            release.set()
            await asyncio.gather(task, return_exceptions=True)
