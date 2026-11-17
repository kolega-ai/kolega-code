import asyncio
import threading
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest

from kolega_code.agent.baseagent import BaseAgent
from kolega_code.cli.session_event_store import FileSessionEventStore
from kolega_code.cli.session_journal import SessionEvent, SessionJournal, SessionRecorder
from kolega_code.events import AgentConnectionManager, AgentEvent, KnownEventType
from kolega_code.llm.models import Message, TextBlock, ToolCall, ToolResult
from kolega_code.session.recording import RecordingConnectionManager

pytestmark = pytest.mark.integration

_DEADLINE = 20.0


async def _concurrent_recovery_and_delivery(
    agent: BaseAgent,
    recorder: SessionRecorder,
    monkeypatch: pytest.MonkeyPatch,
    round_index: int,
) -> SessionEvent:
    journal = recorder.journal
    repair_entered = threading.Event()
    append_entered = threading.Event()
    start = threading.Event()
    committed = threading.Event()
    release_append = threading.Event()
    records: list[SessionEvent] = []
    original_repair = recorder.repair_interrupted_tool_results
    original_append = journal.append
    inner = AsyncMock(spec=AgentConnectionManager)
    store = FileSessionEventStore(journal, append_timeout=None)
    manager = RecordingConnectionManager(inner, store, session_id=journal.session_id)
    event = AgentEvent(
        sender="agent",
        event_type=KnownEventType.LOG_MESSAGE,
        content={"text": f"tiny live event {round_index}"},
    )

    def gated_repair(*, defer_projection: bool = False) -> list[ToolResult] | None:
        repair_entered.set()
        assert start.wait(_DEADLINE), "recovery worker was not released"
        return original_repair(defer_projection=defer_projection)

    def gated_append(event_type: str, **kwargs: Any) -> SessionEvent:
        append_entered.set()
        assert start.wait(_DEADLINE), "recording worker was not released"
        record = original_append(event_type, **kwargs)
        records.append(record)
        committed.set()
        assert release_append.wait(_DEADLINE), "committed append was not released"
        return record

    async def observe_live(event: AgentEvent, workspace_id: str, thread_id: str) -> None:
        assert committed.is_set()
        assert len(records) == 1
        assert event.seq == records[0].seq
        assert records[0].payload["event"]["uuid"] == event.uuid
        assert (workspace_id, thread_id) == (agent.workspace_id, agent.thread_id)

    inner.broadcast_event.side_effect = observe_live
    with monkeypatch.context() as gates:
        gates.setattr(recorder, "repair_interrupted_tool_results", gated_repair)
        gates.setattr(journal, "append", gated_append)
        recovery = asyncio.create_task(agent.prepare_background_completions())
        delivery = asyncio.create_task(manager.broadcast_event(event, agent.workspace_id, agent.thread_id))
        try:
            entered = await asyncio.gather(
                asyncio.to_thread(repair_entered.wait, _DEADLINE),
                asyncio.to_thread(append_entered.wait, _DEADLINE),
            )
            assert all(entered), "both workers must enter before either operation proceeds"
            start.set()
            assert await asyncio.to_thread(committed.wait, _DEADLINE), "recording append did not commit"
            await asyncio.wait_for(asyncio.shield(recovery), timeout=_DEADLINE)
            inner.broadcast_event.assert_not_awaited()
            release_append.set()
            await asyncio.wait_for(asyncio.shield(delivery), timeout=_DEADLINE)
            inner.broadcast_event.assert_awaited_once_with(event, agent.workspace_id, agent.thread_id)
        finally:
            start.set()
            release_append.set()
            await asyncio.wait_for(asyncio.gather(recovery, delivery, return_exceptions=True), timeout=_DEADLINE)
    assert event.seq == records[0].seq
    return records[0]


@pytest.mark.asyncio
async def test_warm_recovery_keeps_recording_scan_free_after_successful_tool_cycles(
    isolated_cli_env: None,
    base_agent: BaseAgent,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    journal = SessionJournal(base_agent.thread_id, tmp_path / "session")
    journal.start_epoch("test")
    recorder = SessionRecorder(journal, recover=False)
    base_agent.session_recorder = recorder
    user = Message(role="user", content=[TextBlock(text="work")])
    recorder.start_turn(user)
    base_agent.history = [user]
    for index in range(512):
        journal.append("ui.log_message", actor="agent", payload={"index": index, "text": "x" * 2048})
    journal.read_events()
    assert recorder.repair_interrupted_tool_results() is None

    full_scans: list[str] = []
    delivered: list[SessionEvent] = []

    def reject_read(*, repair_tail: bool) -> bytes:
        full_scans.append("read")
        raise AssertionError("warm recovery re-read the full journal")

    def reject_parse(raw: bytes, *, strict_seq: bool = True) -> list[SessionEvent]:
        full_scans.append("parse")
        raise AssertionError("warm recovery parsed the full journal")

    with monkeypatch.context() as scans:
        scans.setattr(journal, "_read_event_bytes_locked", reject_read)
        scans.setattr(journal, "_parse_event_bytes_locked", reject_parse)
        for round_index in range(3):
            if round_index:
                call = ToolCall(id=f"tool-{round_index}", name="read_file", input={"path": "example.txt"})
                await base_agent._commit_assistant_message(
                    Message(role="assistant", content=[call], tool_calls=[call], stop_reason="tool_use")
                )
                await base_agent._commit_tool_results(
                    [
                        ToolResult(
                            tool_use_id=call.id,
                            name=call.name,
                            content=f"successful output {round_index}",
                            is_error=False,
                        )
                    ]
                )
                recorder.record_context_message(
                    Message(role="user", content=[TextBlock(text=f"context update {round_index}")])
                )
            delivered.append(await _concurrent_recovery_and_delivery(base_agent, recorder, monkeypatch, round_index))
            assert recorder.repair_interrupted_tool_results() is None
            assert full_scans == []

    events = journal.read_events()
    assert [event.seq for event in events] == list(range(1, len(events) + 1))
    for record in delivered:
        persisted = events[record.seq - 1]
        assert persisted == record
        assert persisted.event_type == "ui.log_message"
    assert sum(event.event_type == "assistant.message" for event in events) == 2
    assert sum(event.event_type == "tool.results" for event in events) == 2
    results = [
        block
        for message in base_agent.history
        if isinstance(message.content, list)
        for block in message.content
        if isinstance(block, ToolResult)
    ]
    assert [result.content for result in results] == ["successful output 1", "successful output 2"]
    assert all(not result.is_error for result in results)
