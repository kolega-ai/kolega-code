"""ACP idle updates, serialized automatic turns, and cancellation ownership."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest

from kolega_code.acp.server import AcpAgent
from kolega_code.acp.agent_factory import AgentFactory
from kolega_code.acp.session import AcpSession
from kolega_code.cli.connection import CliConnectionManager
from kolega_code.cli.session_journal import SessionJournal, SessionRecorder
from kolega_code.cli.session_store import SessionRecord
from kolega_code.events import AgentEvent
from kolega_code.llm.models import Message, TextBlock, ToolCall, ToolResult
from tests.acp.test_server import EventStream, StreamingLLM, _FakeConn, _text_event
from tests.agent.compaction_helpers import build_agent
from tests.background_job_helpers import HostAgent


class _GatedNoticeConn(_FakeConn):
    """Fail one report send, with an explicit boundary for inspecting the worker."""

    def __init__(self, fail_at: int = 1) -> None:
        super().__init__()
        self.fail_at: int = fail_at
        self.attempts: list[Any] = []
        self.entered: asyncio.Event = asyncio.Event()
        self.release: asyncio.Event = asyncio.Event()
        self.presented: asyncio.Queue[Any] = asyncio.Queue()

    async def session_update(self, session_id: str, update: Any, source: str = "") -> None:
        notice = getattr(update, "session_update", "") == "agent_thought_chunk" and "child report" in str(update)
        if notice:
            self.attempts.append(update)
            if len(self.attempts) == self.fail_at:
                self.entered.set()
                await self.release.wait()
                raise ConnectionError("temporary transport failure")
        await super().session_update(session_id, update, source)
        if notice:
            self.presented.put_nowait(update)


def host(tmp_path: Path, conn: _FakeConn | None = None) -> tuple[AcpAgent, AcpSession, HostAgent, _FakeConn]:
    agent = HostAgent()
    record = SessionRecord.create(tmp_path, "cli", {})
    session = AcpSession(session_id=record.session_id, record=record, agent=agent, manager=CliConnectionManager())  # type: ignore[arg-type]

    class Factory:
        def config_options_for(self, session: Any) -> list[Any]:
            return []

        def persist(self, session: Any) -> None:
            pass

        async def close_session(self, session_id: str) -> None:
            await agent.background_jobs.close()
            await agent.background_jobs.deliver(agent)
            await agent.cleanup()

    server = AcpAgent(factory=Factory())  # type: ignore[arg-type]
    conn = conn or _FakeConn()
    server.on_connect(conn)  # type: ignore[arg-type]
    server._sessions[session.session_id] = session
    server._start_session_tasks(session)
    return server, session, agent, conn


@pytest.mark.asyncio
async def test_idle_completion_continues_without_prompt_request(tmp_path: Path) -> None:
    server, session, agent, conn = host(tmp_path)
    gate = asyncio.Event()
    agent.background_jobs.launch(gate)
    gate.set()
    try:
        await asyncio.wait_for(agent.continued.wait(), 1)
        assert agent.continuations == 1
        assert agent.messages == []
        assert any("child report" in str(update) for update in conn.updates)
    finally:
        await server.shutdown()
    assert session.event_task is not None and session.event_task.done()
    assert session.background_task is not None and session.background_task.done()


@pytest.mark.asyncio
@pytest.mark.parametrize("paused", [False, True])
@pytest.mark.parametrize("cancel_send", [False, True])
async def test_unsent_completion_retries_only_on_wakeup(tmp_path: Path, paused: bool, cancel_send: bool) -> None:
    conn = _GatedNoticeConn()
    server, session, agent, _ = host(tmp_path, conn)
    gate = asyncio.Event()
    agent.background_jobs.launch(gate)
    gate.set()
    try:
        await asyncio.wait_for(conn.entered.wait(), 1)
        if paused:
            await server.cancel(session.session_id)
        if cancel_send:
            assert session.background_task is not None
            session.background_task.cancel()
            await asyncio.wait_for(asyncio.gather(session.background_task, return_exceptions=True), 1)
        else:
            conn.release.set()
        # Queued behind the gated send; the failure handler has run when this
        # lock is granted. There must be no delivery-generated retry wakeup.
        async with session.execution_lock:
            assert len(conn.attempts) == 1
            assert len(session.completions) == 1
            assert not session.background_wake.is_set()
            assert len(agent.history) == 1
            assert agent.continuations == 0

        # A later session wakeup retries presentation, even when explicit stop
        # forbids a model continuation or the old worker was cancelled.
        await server.load_session(cwd=str(tmp_path), session_id=session.session_id)
        notice = await asyncio.wait_for(conn.presented.get(), 1)
        async with session.execution_lock:
            assert session.completions == []
            assert len(agent.history) == 1
            assert len(conn.attempts) == 2
            assert conn.updates.count(notice) == 1
            assert conn.presented.empty()
            assert agent.continuations == (0 if paused else 1)
            assert agent.max_active_turns <= 1
    finally:
        await server.shutdown()


@pytest.mark.asyncio
async def test_partial_batch_retries_only_unacknowledged_notice(tmp_path: Path) -> None:
    conn = _GatedNoticeConn(fail_at=2)
    server, session, agent, _ = host(tmp_path, conn)
    gate = asyncio.Event()
    try:
        # Ensure both reports enter a single batch before the worker owns it.
        async with session.execution_lock:
            agent.background_jobs.launch(gate)
            agent.background_jobs.launch(gate)
            gate.set()
            await asyncio.wait_for(asyncio.gather(*agent.background_jobs.tasks), 1)
            for completion in agent.background_jobs.pending:
                completion["text"] += f" ({completion['job_id']})"
        await asyncio.wait_for(conn.entered.wait(), 1)
        first = await asyncio.wait_for(conn.presented.get(), 1)
        conn.release.set()
        async with session.execution_lock:
            assert len(agent.history) == 2
            assert len(session.completions) == 1
            assert session.completions[0]["job_id"] == "job-1"
            assert not session.background_wake.is_set()
            assert agent.continuations == 0

        session.background_wake.set()
        second = await asyncio.wait_for(conn.presented.get(), 1)
        async with session.execution_lock:
            assert first != second
            assert conn.attempts == [first, second, second]
            assert len(agent.history) == 2
            assert session.completions == []
            assert conn.updates.count(first) == conn.updates.count(second) == 1
            assert conn.presented.empty()
            assert agent.continuations == 1
            assert agent.max_active_turns == 1
    finally:
        await server.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("fail_notice", [False, True])
async def test_running_base_agent_consumes_report_without_extra_continuation(tmp_path: Path, fail_notice: bool) -> None:
    tool_call = ToolCall(id="tool-1", name="read_file", input={})
    parent_text = "Parent synthesis, distinct from the delegated child report."
    llm = StreamingLLM()
    llm.stream.side_effect = [
        EventStream(
            [],
            Message(role="assistant", content=[tool_call], tool_calls=[tool_call], stop_reason="tool_use"),
        ),
        EventStream(
            [_text_event(parent_text)],
            Message(role="assistant", content=[TextBlock(text=parent_text)], stop_reason="end_turn"),
        ),
    ]
    manager = CliConnectionManager()
    agent, _ = build_agent(tmp_path, connection_manager=manager, llm=llm, model_context_length=20_000)
    assert agent.tool_collection is not None
    agent.tool_collection.cleanup = AsyncMock()
    journal = SessionJournal(agent.thread_id, tmp_path / "journal")
    journal.start_epoch("test")
    agent.session_recorder = SessionRecorder(journal, recover=False)
    tool_started = asyncio.Event()
    release_tool = asyncio.Event()

    async def process_tool_calls(calls: list[ToolCall]) -> list[ToolResult]:
        assert calls == [tool_call]
        tool_started.set()
        await release_tool.wait()
        return [ToolResult(tool_use_id=tool_call.id, name=tool_call.name, content="tool receipt", is_error=False)]

    agent.process_tool_calls = AsyncMock(side_effect=process_tool_calls)
    record = SessionRecord.create(tmp_path, "cli", {})
    session = AcpSession(session_id=record.session_id, record=record, agent=agent, manager=manager)

    class Factory(AgentFactory):
        def persist(self, session: AcpSession) -> None:
            session.record.history = session.agent.dump_message_history()

        async def close_session(self, session_id: str) -> None:
            await agent.get_background_jobs().close()
            await agent.get_background_jobs().deliver(agent)
            await agent.cleanup()

    server = AcpAgent(factory=Factory())
    conn = _GatedNoticeConn(fail_at=1 if fail_notice else 0)
    server.on_connect(conn)  # type: ignore[arg-type]
    server._sessions[session.session_id] = session
    prompt = asyncio.create_task(
        server.prompt(session.session_id, [{"type": "text", "text": "work"}])  # type: ignore[arg-type]
    )
    try:
        await asyncio.wait_for(tool_started.wait(), 3)
        jobs = agent.get_background_jobs()

        async def child() -> str:
            return "child report"

        await jobs.submit("job-1", child, agent_type="investigation", task="research")
        await asyncio.wait_for(jobs.wait_for_completion(), 3)
        assert jobs.pending_count == 1
        release_tool.set()
        assert (await asyncio.wait_for(prompt, 3)).stop_reason == "end_turn"
        if fail_notice:
            await asyncio.wait_for(conn.entered.wait(), 3)
            conn.release.set()
            async with session.execution_lock:
                assert len(session.completions) == 1
                assert not session.background_wake.is_set()
            server._start_session_tasks(session)
        notice = await asyncio.wait_for(conn.presented.get(), 3)
        # The worker sees the in-turn receipt after the root releases ownership.
        # Its presentation must not create a third model request/root turn.
        async with session.execution_lock:
            assert llm.stream.await_count == 2
            assert not agent.background_needs_turn
            assert not session.background_wake.is_set()
            assert session.completions == []
            assert jobs.pending_count == 0
            second_input = llm.stream.call_args_list[1].kwargs["messages"]
            assert sum("Background subagent completion." in msg.get_text_content() for msg in second_input) == 1
            assert sum("Background subagent completion." in msg.get_text_content() for msg in agent.history) == 1
            events = journal.read_events()
            assert sum(event.event_type == "turn.started" for event in events) == 1
            assert (
                sum(
                    event.event_type == "context.message" and bool(event.payload.get("completion_id"))
                    for event in events
                )
                == 1
            )
            assert conn.updates.count(notice) == 1
            assert conn.presented.empty()
            parent_chunks = [
                update.content.text
                for update in conn.updates
                if getattr(update, "session_update", "") == "agent_message_chunk"
            ]
            assert "".join(parent_chunks) == parent_text
    finally:
        release_tool.set()
        await server.shutdown()
        await asyncio.gather(prompt, return_exceptions=True)


@pytest.mark.asyncio
async def test_idle_events_are_consumed_not_discarded_at_next_prompt(tmp_path: Path) -> None:
    server, session, agent, conn = host(tmp_path)
    observed = asyncio.Event()
    original = session.bridge.handle_event

    async def handle_event(session_id: str, event: AgentEvent) -> None:
        observed.set()
        await original(session_id, event)

    session.bridge.handle_event = handle_event
    try:
        session.manager.events.put_nowait(
            AgentEvent(event_type="chat_message", sender="agent", content={"message_type": "status", "content": "idle"})
        )
        await asyncio.wait_for(observed.wait(), 1)
        assert session.event_task is not None and not session.event_task.done()
    finally:
        await server.shutdown()


@pytest.mark.asyncio
async def test_cancelled_prompt_returns_cancelled_then_delivers_without_restart(tmp_path: Path) -> None:
    server, session, agent, _ = host(tmp_path)
    agent.turn_gate = asyncio.Event()
    child_gate = asyncio.Event()
    agent.background_jobs.launch(child_gate)
    prompt = asyncio.create_task(
        server.prompt(session.session_id, [{"type": "text", "text": "work"}])  # type: ignore[arg-type]
    )
    try:
        await asyncio.wait_for(agent.turn_started.wait(), 1)
        child_gate.set()
        await asyncio.wait_for(agent.background_jobs.changed.wait(), 1)
        assert agent.background_jobs.pending_count == 1
        await server.cancel(session.session_id)
        response = await asyncio.wait_for(prompt, 1)
        assert response.stop_reason == "cancelled"
        await asyncio.wait_for(agent.background_jobs.delivered.wait(), 1)
        assert agent.continuations == 0
        assert agent.history[0]["result"] == "child report"
    finally:
        await server.shutdown()


@pytest.mark.asyncio
async def test_close_cancels_and_joins_jobs_before_services(tmp_path: Path) -> None:
    server, session, agent, _ = host(tmp_path)
    agent.background_jobs.launch(asyncio.Event())
    await server.close_session(session.session_id)
    assert agent.cleaned
    assert agent.background_jobs.active_count == 0
    assert agent.background_jobs.outcomes[0]["status"] == "cancelled"
    assert session.event_task is not None and session.event_task.done()
    assert session.background_task is not None and session.background_task.done()


@pytest.mark.asyncio
async def test_completion_during_prompt_waits_for_serial_owner_release(tmp_path: Path) -> None:
    server, session, agent, _ = host(tmp_path)
    agent.turn_gate = asyncio.Event()
    child_gate = asyncio.Event()
    agent.background_jobs.launch(child_gate)
    prompt = asyncio.create_task(
        server.prompt(session.session_id, [{"type": "text", "text": "work"}])  # type: ignore[arg-type]
    )
    try:
        await asyncio.wait_for(agent.turn_started.wait(), 1)
        child_gate.set()
        await asyncio.wait_for(agent.background_jobs.changed.wait(), 1)
        assert agent.background_jobs.pending_count == 1
        assert agent.continuations == 0
        agent.turn_gate.set()
        assert (await asyncio.wait_for(prompt, 1)).stop_reason == "end_turn"
        await asyncio.wait_for(agent.continued.wait(), 1)
        assert agent.max_active_turns == 1
        assert agent.continuations == 1
    finally:
        await server.shutdown()


@pytest.mark.asyncio
async def test_rebuild_keeps_supervisor_and_event_consumer_while_old_jobs_finish(tmp_path: Path) -> None:
    server, session, old, _ = host(tmp_path)
    rebuilt = HostAgent()

    class Factory(AgentFactory):
        def persist(self, session: Any) -> None:
            pass

        async def _construct_agent(self, record: Any, config: Any, **kwargs: Any) -> Any:
            return rebuilt, session.manager

    factory = Factory()
    factory._acp_sessions[session.session_id] = session
    server._factory = factory
    jobs = old.background_jobs
    gate = asyncio.Event()
    jobs.launch(gate)
    event_task = session.event_task
    try:
        await factory._rebuild_agent(session, None)
        assert rebuilt.background_jobs is jobs
        assert not old.cleaned
        assert not jobs.closed
        assert session.event_task is event_task
        gate.set()
        await asyncio.wait_for(rebuilt.continued.wait(), 1)
        # Synchronize with the worker's post-turn retired-generation cleanup.
        async with session.execution_lock:
            assert old.cleaned
        assert old.continuations == 0
        assert rebuilt.continuations == 1
    finally:
        await server.shutdown()


@pytest.mark.asyncio
async def test_close_joins_active_prompt_before_cleaning_session(tmp_path: Path) -> None:
    server, session, agent, _ = host(tmp_path)
    agent.turn_gate = asyncio.Event()
    agent.background_jobs.launch(asyncio.Event())
    prompt = asyncio.create_task(
        server.prompt(session.session_id, [{"type": "text", "text": "work"}])  # type: ignore[arg-type]
    )
    await asyncio.wait_for(agent.turn_started.wait(), 1)
    await asyncio.wait_for(server.close_session(session.session_id), 1)
    assert (await asyncio.wait_for(prompt, 1)).stop_reason == "cancelled"
    assert agent.active_turns == 0
    assert agent.cleaned
    assert not agent.background_jobs.has_work
    assert agent.continuations == 0
