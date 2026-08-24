"""Diff provider tests: snapshot-sourced old/new content for ACP tool results."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest
from acp.interfaces import Client
from acp.schema import FileEditToolCallContent, ToolCallProgress, ToolCallStart

from kolega_code.acp.bridge import AcpBridge
from kolega_code.acp.diffs import AcpDiffProvider
from kolega_code.events import AgentEvent
from kolega_code.services.file_system import LocalFileSystem
from kolega_code.services.snapshots import SnapshotService


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


def _service(tmp_path: Path) -> SnapshotService:
    return SnapshotService(
        Path(tmp_path),
        "ws",
        "thread",
        "sess",
        LocalFileSystem(root_path=tmp_path),
        root=tmp_path / "state",
    )


def _mutate(service: SnapshotService, tmp_path: Path, mutate: Any) -> None:
    target = tmp_path / "f.txt"
    service.record_mutation(
        tool_name="edit",
        tool_call_id="call-1",
        reason="test",
        paths=[str(target)],
        mutate=mutate,
    )


@pytest.mark.asyncio
async def test_existing_file_edit_produces_old_new_text(tmp_path: Path) -> None:
    service = _service(tmp_path)
    target = tmp_path / "f.txt"
    target.write_text("hello\n")

    def mutate() -> None:
        target.write_text("hello\nworld\n")

    _mutate(service, tmp_path, mutate)

    contents = AcpDiffProvider(service).build_for_tool_result("call-1")
    assert len(contents) == 1
    assert contents[0].path == str(target)
    assert contents[0].old_text == "hello\n"
    assert contents[0].new_text == "hello\nworld\n"


@pytest.mark.asyncio
async def test_new_file_has_no_old_text(tmp_path: Path) -> None:
    service = _service(tmp_path)
    target = tmp_path / "f.txt"

    def mutate() -> None:
        target.write_text("brand new\n")

    _mutate(service, tmp_path, mutate)

    contents = AcpDiffProvider(service).build_for_tool_result("call-1")
    assert len(contents) == 1
    assert contents[0].old_text is None
    assert contents[0].new_text == "brand new\n"


@pytest.mark.asyncio
async def test_unchanged_mutation_produces_no_diff(tmp_path: Path) -> None:
    service = _service(tmp_path)
    target = tmp_path / "f.txt"
    target.write_text("same\n")

    def mutate() -> None:
        target.write_text("same\n")

    _mutate(service, tmp_path, mutate)

    assert AcpDiffProvider(service).build_for_tool_result("call-1") == []


@pytest.mark.asyncio
async def test_unknown_tool_call_produces_no_diff(tmp_path: Path) -> None:
    service = _service(tmp_path)
    target = tmp_path / "f.txt"
    target.write_text("hello\n")

    def mutate() -> None:
        target.write_text("hello\nworld\n")

    _mutate(service, tmp_path, mutate)

    assert AcpDiffProvider(service).build_for_tool_result("call-other") == []


@pytest.mark.asyncio
async def test_no_snapshot_service_produces_no_diff(tmp_path: Path) -> None:
    assert AcpDiffProvider(None).build_for_tool_result("call-1") == []


@pytest.mark.asyncio
async def test_bridge_attaches_diff_content_to_tool_result(tmp_path: Path) -> None:
    service = _service(tmp_path)
    target = tmp_path / "f.txt"
    target.write_text("hello\n")

    def mutate() -> None:
        target.write_text("hello\nworld\n")

    _mutate(service, tmp_path, mutate)

    conn = _FakeConn()
    bridge = AcpBridge(cast(Client, conn), diff_provider=AcpDiffProvider(service))
    await bridge.handle_event("s1", _chat_event("tool_call", "Calling edit", "call-1", "edit"))
    await bridge.handle_event("s1", _chat_event("tool_result", "Edited f.txt", "call-1"))

    assert isinstance(conn.updates[0], ToolCallStart)
    progress = conn.updates[1]
    assert isinstance(progress, ToolCallProgress)
    assert progress.status == "completed"
    assert progress.content is not None
    assert len(progress.content) == 2
    assert isinstance(progress.content[0], FileEditToolCallContent)
    assert progress.content[0].old_text == "hello\n"
    assert progress.content[0].new_text == "hello\nworld\n"
    assert progress.content[1].content.text == "Edited f.txt"


@pytest.mark.asyncio
async def test_bridge_skips_diff_content_on_tool_error(tmp_path: Path) -> None:
    service = _service(tmp_path)
    target = tmp_path / "f.txt"
    target.write_text("hello\n")

    def mutate() -> None:
        target.write_text("hello\nworld\n")

    _mutate(service, tmp_path, mutate)

    conn = _FakeConn()
    bridge = AcpBridge(cast(Client, conn), diff_provider=AcpDiffProvider(service))
    await bridge.handle_event("s1", _chat_event("tool_error", "boom", "call-1"))

    progress = conn.updates[0]
    assert progress.status == "failed"
    assert all(not isinstance(block, FileEditToolCallContent) for block in (progress.content or []))
