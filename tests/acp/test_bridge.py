"""Unit tests for the ACP event bridge (Phase 1)."""

from __future__ import annotations

from typing import Any, cast

import pytest
from acp.interfaces import Client
from acp.schema import AgentMessageChunk, AgentThoughtChunk, ToolCallProgress, ToolCallStart

from kolega_code.acp.bridge import AcpBridge
from kolega_code.events import AgentEvent


class _FakeConn:
    def __init__(self) -> None:
        self.updates: list[Any] = []

    async def session_update(self, session_id: str, update: Any, source: str = "") -> None:
        self.updates.append(update)


def _chat_event(message_type: str, text: str = "", tool_call_id: str = "", tool_description: str = "") -> AgentEvent:
    content: dict[str, Any] = {"message_type": message_type, "text": text, "tool_call_id": tool_call_id}
    if tool_description:
        content["tool_description"] = tool_description
    return AgentEvent(sender="agent", event_type="chat_message", content=content, is_streaming=False)


@pytest.mark.asyncio
async def test_tool_call_event_starts_tool_call_with_mapped_kind() -> None:
    conn = _FakeConn()
    bridge = AcpBridge(cast(Client, conn))

    await bridge.handle_event("s1", _chat_event("tool_call", "Calling read", "c1", "read"))
    await bridge.handle_event("s1", _chat_event("tool_call", "Calling edit", "c2", "multi_edit"))
    await bridge.handle_event("s1", _chat_event("tool_call", "Calling frobnicate", "c3", "frobnicate"))

    assert len(conn.updates) == 3
    first, second, third = conn.updates
    assert isinstance(first, ToolCallStart)
    assert first.tool_call_id == "c1"
    assert first.kind == "read"
    assert first.status == "pending"
    assert isinstance(second, ToolCallStart)
    assert second.kind == "edit"
    assert isinstance(third, ToolCallStart)
    assert third.kind == "other"


@pytest.mark.asyncio
async def test_tool_result_completes_with_text_content() -> None:
    conn = _FakeConn()
    bridge = AcpBridge(cast(Client, conn))
    await bridge.handle_event("s1", _chat_event("tool_result", "Wrote file", "c1"))
    assert len(conn.updates) == 1
    update = conn.updates[0]
    assert isinstance(update, ToolCallProgress)
    assert update.tool_call_id == "c1"
    assert update.status == "completed"
    assert update.content is not None
    assert update.content[0].content.text == "Wrote file"


@pytest.mark.asyncio
async def test_tool_error_marks_failed() -> None:
    conn = _FakeConn()
    bridge = AcpBridge(cast(Client, conn))
    await bridge.handle_event("s1", _chat_event("tool_error", "boom", "c1"))
    update = conn.updates[0]
    assert update.status == "failed"


@pytest.mark.asyncio
async def test_non_chat_events_are_ignored() -> None:
    conn = _FakeConn()
    bridge = AcpBridge(cast(Client, conn))
    event = AgentEvent(sender="agent", event_type="assistant_delta", content={"text": "hi"}, is_streaming=True)
    await bridge.handle_event("s1", event)
    assert conn.updates == []


@pytest.mark.asyncio
async def test_response_chunk_emits_agent_message() -> None:
    conn = _FakeConn()
    bridge = AcpBridge(cast(Client, conn))
    await bridge.emit_chunk("s1", {"type": "response", "content": "hello", "complete": False, "uuid": "u1"})
    assert len(conn.updates) == 1
    update = conn.updates[0]
    assert isinstance(update, AgentMessageChunk)
    assert update.content.text == "hello"


@pytest.mark.asyncio
async def test_thinking_chunk_emits_thought_update() -> None:
    conn = _FakeConn()
    bridge = AcpBridge(cast(Client, conn))
    await bridge.emit_chunk("s1", {"type": "thinking", "content": "hmm", "complete": False, "uuid": "u1"})
    assert isinstance(conn.updates[0], AgentThoughtChunk)
    assert conn.updates[0].content.text == "hmm"


@pytest.mark.asyncio
async def test_empty_chunks_are_dropped() -> None:
    conn = _FakeConn()
    bridge = AcpBridge(cast(Client, conn))
    await bridge.emit_chunk("s1", {"type": "response", "content": "", "complete": True, "uuid": "u1"})
    assert conn.updates == []


def _terminal_event(event_type: str, content: dict[str, Any]) -> AgentEvent:
    return AgentEvent(sender="agent", event_type=event_type, content=content, is_streaming=False)


@pytest.mark.asyncio
async def test_terminal_command_marks_tool_in_progress_with_command() -> None:
    conn = _FakeConn()
    bridge = AcpBridge(cast(Client, conn))

    await bridge.handle_event("s1", _chat_event("tool_call", "Calling exec_command", "c1", "exec_command"))
    await bridge.handle_event("s1", _terminal_event("terminal_command", {"command": "pytest -q", "terminal_id": "s_1"}))

    command_update = conn.updates[1]
    assert isinstance(command_update, ToolCallProgress)
    assert command_update.status == "in_progress"
    assert command_update.content is not None
    assert command_update.content[0].content.text == "$ pytest -q\n"


@pytest.mark.asyncio
async def test_terminal_output_appends_to_tool_content() -> None:
    conn = _FakeConn()
    bridge = AcpBridge(cast(Client, conn))

    await bridge.handle_event("s1", _chat_event("tool_call", "Calling exec_command", "c1", "exec_command"))
    await bridge.handle_event("s1", _terminal_event("terminal_command", {"command": "pytest", "terminal_id": "s_1"}))
    await bridge.handle_event(
        "s1", _terminal_event("terminal_output", {"display_output": "collecting...\n", "terminal_id": "s_1"})
    )
    await bridge.handle_event("s1", _terminal_event("terminal_output", {"output": "passed\n", "terminal_id": "s_1"}))

    assert conn.updates[2].content[0].content.text == "$ pytest\ncollecting...\n"
    assert conn.updates[3].content[0].content.text == "$ pytest\ncollecting...\npassed\n"


@pytest.mark.asyncio
async def test_terminal_result_uses_terminal_buffer_over_markdown_text() -> None:
    conn = _FakeConn()
    bridge = AcpBridge(cast(Client, conn))

    await bridge.handle_event("s1", _chat_event("tool_call", "Calling exec_command", "c1", "exec_command"))
    await bridge.handle_event("s1", _terminal_event("terminal_command", {"command": "pytest", "terminal_id": "s_1"}))
    await bridge.handle_event("s1", _terminal_event("terminal_output", {"output": "passed\n", "terminal_id": "s_1"}))
    await bridge.handle_event("s1", _chat_event("tool_result", "Status: exited\n\nOutput:\npassed", "c1"))

    final = conn.updates[3]
    assert final.status == "completed"
    assert final.content[0].content.text == "$ pytest\npassed\n"
    assert bridge._terminal_text == {}  # noqa: SLF001


@pytest.mark.asyncio
async def test_terminal_events_without_active_execute_tool_are_ignored() -> None:
    conn = _FakeConn()
    bridge = AcpBridge(cast(Client, conn))

    await bridge.handle_event("s1", _terminal_event("terminal_command", {"command": "pytest", "terminal_id": "s_1"}))
    await bridge.handle_event("s1", _terminal_event("terminal_output", {"output": "passed\n", "terminal_id": "s_1"}))

    assert conn.updates == []


@pytest.mark.asyncio
async def test_background_terminal_noise_does_not_leak_into_other_tools() -> None:
    conn = _FakeConn()
    bridge = AcpBridge(cast(Client, conn))

    await bridge.handle_event("s1", _chat_event("tool_call", "Calling read", "c1", "read"))
    await bridge.handle_event("s1", _terminal_event("terminal_output", {"output": "noise\n", "terminal_id": "s_old"}))

    assert conn.updates[0].status == "pending"
    assert len(conn.updates) == 1
