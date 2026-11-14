from __future__ import annotations

import asyncio
from pathlib import Path
import threading
from typing import Any
from unittest.mock import AsyncMock

import pytest

from kolega_code.acp.agent_factory import AgentFactory
from kolega_code.cli.session_event_store import FileSessionEventStore
from kolega_code.cli.session_usage import derive_session_usage
from kolega_code.events import AgentEvent
from kolega_code.llm.models import Message
from kolega_code.llm.usage import normalize_usage
from kolega_code.session.projection import replay
from kolega_code.session.recording import RecordingConnectionManager
from kolega_code.session.usage import UsagePresentation
from tests.acp.test_cleanup_lifecycle import configured_factory as configured_factory

pytestmark = [pytest.mark.asyncio, pytest.mark.usefixtures("isolated_cli_env")]


async def _events(factory: AgentFactory, session_id: str) -> list[AgentEvent]:
    return await FileSessionEventStore(factory._store.journal(session_id)).read(session_id)


def _snapshots(events: list[AgentEvent]) -> list[AgentEvent]:
    return [event for event in events if event.event_type == "usage_updated"]


async def test_factory_owns_one_neutral_publisher_through_model_and_mode_rebuilds(
    configured_factory: AgentFactory, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    factory = configured_factory
    built: list[Any] = []
    started: list[UsagePresentation] = []
    construct, start = factory._construct_agent, UsagePresentation.start

    async def observed_construct(*args: Any, **kwargs: Any) -> Any:
        result = await construct(*args, **kwargs)
        built.append(result[0])
        return result

    async def observed_start(owner: UsagePresentation) -> None:
        assert built, "publication must start after the initial agent is constructed"
        started.append(owner)
        await start(owner)

    monkeypatch.setattr(factory, "_construct_agent", observed_construct)
    monkeypatch.setattr(UsagePresentation, "start", observed_start)
    session = await factory.open_session(str(tmp_path))
    owner = session.usage_presentation
    assert owner is not None and session.accounting is not None
    accounting = session.accounting
    assert accounting.sink is not None
    sink, ledger = accounting.sink, accounting.ledger
    neutral = owner._broadcast.__self__
    assert type(neutral) is RecordingConnectionManager
    assert neutral.inner is session.manager
    assert started == [owner]
    try:
        recorder = factory._store.recorder(session.session_id)
        recorder.start_turn(Message(role="user", content="accounted response"))
        usage = normalize_usage({"input_tokens": 13, "output_tokens": 4}, "anthropic", "usage-test")
        request_id = ledger.begin("anthropic", "usage-test")
        message = Message(role="assistant", content="done", usage=usage)
        ledger.record_response(request_id, usage, message=message)
        recorder.record_assistant(message)
        recorder.finish_turn("completed")
        assert await sink.flush()
        await owner.flush()
        before = _snapshots(await _events(factory, session.session_id))[-1]
        assert before.content["usage"]["requests"] == 1
        assert before.content["usage"]["total_tokens"] == 17
        assert before.content["usage"]["coverage"]["full"] is True
        old_generation = session.agent._acp_generation
        await factory.apply_model(session, "anthropic", session.config.long_context_config.model)
        assert session.agent is not built[0]
        assert session.agent._acp_generation != old_generation
        assert session.usage_presentation is owner
        await factory.set_interaction_mode(session, "plan")
        assert session.usage_presentation is owner
        assert session.accounting is accounting
        assert started == [owner]
        assert owner._broadcast.__self__ is neutral

        request_id = ledger.begin("anthropic", "usage-test")
        second = normalize_usage({"input_tokens": 7, "output_tokens": 2}, "anthropic", "usage-test")
        ledger.record_response(request_id, second, message=Message(role="assistant", content="helper", usage=second))
        assert await sink.flush()
        await owner.flush()
        events = await _events(factory, session.session_id)
        snapshots = _snapshots(events)
        last = snapshots[-1]
        assert last.content["scope"] == "session"
        assert last.content["semantics"] == "snapshot"
        assert type(last.content["source_seq"]) is int
        assert last.content["usage"]["requests"] == 2
        assert last.content["usage"]["input_tokens"] == 20
        assert last.content["usage"]["output_tokens"] == 6
        assert last.content["usage"]["total_tokens"] == 26
        assert all("acp_generation" not in event.content for event in snapshots)
        assert all(event.sub_agent_info is None for event in snapshots)
        assert all(event.session_id == session.session_id for event in snapshots)
        assert all(event.workspace_id == session.record.workspace_id for event in snapshots)
        assert all(event.thread_id == session.record.thread_id for event in snapshots)
        assert replay(events).usage == {"source_seq": last.content["source_seq"], **last.content["usage"]}
        close = owner.aclose
        closed: list[bool] = []

        async def observed_close() -> None:
            assert accounting._closed
            assert sink.close_result is not None and sink.close_result.committed
            await close()
            closed.append(True)

        monkeypatch.setattr(owner, "aclose", observed_close)
        await factory.close_session(session.session_id)
        assert closed == [True]
        assert owner._closed and owner._unsubscribe is None
        assert owner._worker is not None and owner._worker.done()
        count = len(_snapshots(await _events(factory, session.session_id)))
        ledger.record_failure(ledger.begin("anthropic", "usage-test"), "after close")
        await owner.flush()
        assert len(_snapshots(await _events(factory, session.session_id))) == count
        final = _snapshots(await _events(factory, session.session_id))[-1]
        assert final.content["usage"]["requests"] == 2
        assert final.content["usage"]["total_tokens"] == 26
        assert final.content["usage"]["coverage"]["unclosed_runs"] == 0
    finally:
        await factory.close_session(session.session_id)
        factory._store.release_session_locks()


async def test_failed_publication_start_closes_partial_agent_accounting_and_publisher(
    configured_factory: AgentFactory, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    factory = configured_factory
    captured: dict[str, Any] = {}
    construct, start = factory._construct_agent, UsagePresentation.start
    failure = RuntimeError("publication startup failed")

    async def observed_construct(*args: Any, **kwargs: Any) -> Any:
        captured["accounting"] = kwargs["accounting"]
        agent, manager = await construct(*args, **kwargs)
        cleanup = AsyncMock(wraps=agent.cleanup)
        monkeypatch.setattr(agent, "cleanup", cleanup)
        captured["cleanup"] = cleanup
        captured["session_id"] = args[0].session_id
        return agent, manager

    async def failed_start(owner: UsagePresentation) -> None:
        captured["owner"] = owner
        await start(owner)
        raise failure

    monkeypatch.setattr(factory, "_construct_agent", observed_construct)
    monkeypatch.setattr(UsagePresentation, "start", failed_start)
    try:
        with pytest.raises(RuntimeError) as raised:
            await factory.open_session(str(tmp_path))
        assert raised.value is failure
        owner, accounting = captured["owner"], captured["accounting"]
        captured["cleanup"].assert_awaited_once()
        assert accounting._closed
        assert accounting.sink.close_result.committed
        assert owner._closed and owner._unsubscribe is None
        assert owner._worker.done()
        assert captured["session_id"] not in factory._acp_sessions
        snapshots = _snapshots(await _events(factory, captured["session_id"]))
        assert snapshots[-1].content["usage"]["requests"] == 0
        assert snapshots[-1].content["usage"]["total_tokens"] == 0
    finally:
        factory._store.release_session_locks()


async def test_open_cancel_during_initial_snapshot_drains_accounting_before_final_publication(
    configured_factory: AgentFactory, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    factory = configured_factory
    entered, release = threading.Event(), threading.Event()
    owners: list[UsagePresentation] = []
    read = UsagePresentation._read_snapshot

    def blocked(owner: UsagePresentation, live_run_id: str | None) -> tuple[int, dict[str, Any]]:
        if not owners:
            owners.append(owner)
        entered.set()
        assert release.wait(30)
        return read(owner, live_run_id)

    monkeypatch.setattr(UsagePresentation, "_read_snapshot", blocked)
    opening = asyncio.create_task(factory.open_session(str(tmp_path)))
    try:
        assert await asyncio.to_thread(entered.wait, 15)
        owner = owners[0]
        opening.cancel()
        await asyncio.sleep(0)
        opening.cancel()
        await asyncio.sleep(0)
        assert not opening.done()
        assert not owner._closing
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(opening, 15)
        journal = owner._journal
        final = _snapshots(await _events(factory, journal.session_id))[-1]
        assert final.content["usage"] == derive_session_usage(journal.read_events(repair_tail=False))
        assert final.content["usage"]["persistence"]["healthy"] is True
        assert final.content["usage"]["coverage"]["unclosed_runs"] == 0
        assert owner._closed and owner._unsubscribe is None
        assert journal.session_id not in factory._acp_sessions
    finally:
        release.set()
        await asyncio.gather(opening, return_exceptions=True)
        for session_id in list(factory._acp_sessions):
            await factory.close_session(session_id)
        factory._store.release_session_locks()
