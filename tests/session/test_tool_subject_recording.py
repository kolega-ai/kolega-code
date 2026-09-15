"""Optional display metadata survives recording without adding raw arguments."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import create_autospec

import pytest

from kolega_code.cli.session_event_store import FileSessionEventStore
from kolega_code.cli.session_journal import SessionJournal
from kolega_code.events import AgentConnectionManager, AgentEvent, AgentEventEmitter, KnownEventType
from kolega_code.session.projection import replay
from kolega_code.session.recording import RecordingConnectionManager

pytestmark = pytest.mark.usefixtures("isolated_cli_env")


@pytest.mark.asyncio
@pytest.mark.parametrize("delegated", [False, True])
@pytest.mark.parametrize("message_type,status", [("tool_result", "done"), ("tool_error", "failed")])
@pytest.mark.parametrize("result_subject", [None, "src/updated.py"])
async def test_tool_subject_recording_roundtrip(
    tmp_path: Path, delegated: bool, message_type: str, status: str, result_subject: str | None
) -> None:
    session_id = "tool-subject-recording"
    journal = SessionJournal(session_id, tmp_path)
    journal.start_epoch("test")
    inner = create_autospec(AgentConnectionManager, instance=True)
    recorder = RecordingConnectionManager(inner, FileSessionEventStore(journal), session_id=session_id)
    info = {"dispatch_id": "d1", "agent_name": "reader"} if delegated else None
    emitter = AgentEventEmitter(recorder, "workspace", "thread", "agent", sub_agent_info_provider=lambda: info)

    await emitter.chat("tool_call", "", tool_description="read", tool_call_id="c", tool_subject="src/original.py")
    await emitter.emit(
        AgentEvent(
            sender="agent",
            event_type=KnownEventType.TOOL_STREAMING_UPDATE,
            sub_agent_info=info,
            content={"tool_call_id": "c", "text": "partial", "stream_mode": "append"},
        )
    )
    await emitter.chat(message_type, "finished", tool_call_id="c", tool_subject=result_subject)
    await emitter.chat("tool_call", "", tool_description="read", tool_call_id="legacy")
    await emitter.chat("tool_result", "legacy output", tool_call_id="legacy")
    await recorder.flush()

    # Reopen the journal through a fresh store: not a live in-memory echo.
    stored = await FileSessionEventStore(SessionJournal(session_id, tmp_path)).read(session_id)
    assert len(stored) == 5
    assert stored[0].content["tool_subject"] == "src/original.py"
    assert "tool_subject" not in stored[1].content
    if result_subject is None:
        assert "tool_subject" not in stored[2].content
    else:
        assert stored[2].content["tool_subject"] == result_subject
    assert all("tool_subject" not in event.content for event in stored[3:])
    assert inner.broadcast_event.await_count == 5

    allowed_chat_keys = {"message_type", "text", "tool_description", "tool_call_id", "tool_subject"}
    envelopes = [json.loads(line) for line in journal.events_path.read_text(encoding="utf-8").splitlines()]
    chat_events = [record["payload"]["event"] for record in envelopes if record["type"] == "ui.chat_message"]
    assert len(chat_events) == 4
    # In particular, no input/arguments/kwargs dictionary is added for display.
    assert all(set(event["content"]) <= allowed_chat_keys for event in chat_events)
    assert chat_events[0]["content"]["tool_subject"] == "src/original.py"

    state = replay(stored)
    items = state.sub_agents["d1"].steps if delegated else state.conversation
    assert len(items) == 2
    assert items[0].tool_subject == (result_subject or "src/original.py")
    assert items[0].status == status
    assert items[0].text == "finished"
    assert items[1].tool_subject is None
    encoded = state.to_dict()
    serialized = encoded["sub_agents"]["d1"]["steps"] if delegated else encoded["conversation"]
    assert "tool_subject" not in serialized[1]
