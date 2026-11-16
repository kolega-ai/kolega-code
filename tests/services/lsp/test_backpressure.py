from __future__ import annotations

import asyncio
from hashlib import sha256
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest

from kolega_code.agent.prompt_provider import AgentMode
from kolega_code.agent.tool_backend.edit_tool import EditTool
from kolega_code.agent.tool_backend.hashline_v2 import format_line_tag
from kolega_code.agent.tool_backend.lsp_tool import LspEditTool
from kolega_code.services.lsp import LspManager
from kolega_code.services.lsp.client import LspClient
from kolega_code.services.snapshots import SnapshotService


def _make_tool(manager: LspManager, *, lsp_edit: bool = False) -> EditTool | LspEditTool:
    caller = Mock(
        agent_name="test-agent",
        agent_mode=AgentMode.CODE.value,
        sub_agent=False,
        current_tool_execution_id="mutation-call",
    )
    tool_class = LspEditTool if lsp_edit else EditTool
    return tool_class(
        manager._project_path,
        "workspace",
        "thread",
        AsyncMock(),
        Mock(),
        caller,
        lsp_manager=manager,
    )


def _assert_reaped(client: LspClient, proc: asyncio.subprocess.Process) -> None:
    assert proc.returncode is not None
    assert client._proc is None
    assert client._reader_task is None
    assert client._stop_task is not None and client._stop_task.done()
    assert client._pending == {}


def _preview_count(tool: EditTool | LspEditTool) -> int:
    connection = tool.connection_manager
    assert isinstance(connection, AsyncMock)
    return sum(call.args[0].event_type == "file_edit_preview" for call in connection.broadcast_event.await_args_list)


@pytest.mark.asyncio
async def test_noisy_stderr_preserves_requests_large_writes_and_diagnostics(
    fake_lsp_manager: LspManager,
) -> None:
    manager = fake_lsp_manager
    manager._resolved["python"].env["FAKE_LSP_STDERR_BYTES"] = str(2 * 1024 * 1024)
    async with asyncio.timeout(10):
        _, client = await manager._get_or_start_session("src.py")
        assert client is not None
        assert await client.request("workspace/symbol", {"query": "noise"})
        tool = _make_tool(manager)
        assert isinstance(tool, EditTool)
        content = "# " + "x" * (2 * 1024 * 1024) + "\nvalue = undefined_var\n"

        result = await tool.claude_write("large.py", content)

    assert result.startswith("Wrote large.py")
    assert "undefined_var" in result
    assert (manager._project_path / "large.py").read_bytes() == content.encode()
    assert _preview_count(tool) == 1
    assert client.running


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [False, True], ids=["deadline", "cancel"])
async def test_nonreading_server_cannot_strand_applied_write_and_restarts_with_did_open(
    fake_lsp_manager: LspManager,
    monkeypatch: pytest.MonkeyPatch,
    cancel: bool,
) -> None:
    from kolega_code.services.lsp import manager as manager_module

    manager = fake_lsp_manager
    manager._resolved["python"].env["FAKE_LSP_STRICT_OPEN"] = "1"
    await manager.get_diagnostics("src.py")
    client = manager._sessions["python"]
    proc = client._proc
    assert proc is not None and proc.stdin is not None
    await client.request("test/stopReading")
    drain = proc.stdin.drain
    buffered_sizes: list[int] = []
    blocked = asyncio.Event()

    async def record_drain() -> None:
        assert proc.stdin is not None
        buffered = proc.stdin.transport.get_write_buffer_size()
        buffered_sizes.append(buffered)
        if buffered > 65536:
            blocked.set()
        await drain()

    monkeypatch.setattr(proc.stdin, "drain", record_drain)
    tool = _make_tool(manager)
    assert isinstance(tool, EditTool)
    tool.observe_read("src.py")
    content = "# " + "x" * (2 * 1024 * 1024) + "\nvalue = undefined_var\n"
    monkeypatch.setattr(manager_module, "_AUTO_DIAGNOSTICS_TIMEOUT_SECONDS", 30 if cancel else 0.1)
    mutation = asyncio.create_task(tool.claude_write("src.py", content))
    try:
        async with asyncio.timeout(10):
            await blocked.wait()
            assert (manager._project_path / "src.py").read_bytes() == content.encode()
            assert _preview_count(tool) == 1
            if cancel:
                mutation.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await mutation
            else:
                assert await mutation == "Wrote src.py"
    finally:
        mutation.cancel()
        await asyncio.gather(mutation, return_exceptions=True)

    assert max(buffered_sizes) > 65536
    assert (manager._project_path / "src.py").read_bytes() == content.encode()
    _assert_reaped(client, proc)
    assert manager._diag_waiters == {}
    async with asyncio.timeout(10):
        diagnostics = await manager.get_fresh_diagnostics("src.py")
    assert any("undefined_var" in diagnostic.message for diagnostic in diagnostics)
    assert manager._sessions["python"] is not client


@pytest.mark.asyncio
async def test_extra_nonreading_server_is_reaped_without_failing_applied_edit(
    fake_lsp_manager_with_extra_strict: LspManager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from kolega_code.services.lsp import manager as manager_module

    manager = fake_lsp_manager_with_extra_strict
    await manager.get_diagnostics("src.py")
    extra = manager._extra_sessions[("python", "fake-extra")]
    primary = manager._sessions["python"]
    proc = extra._proc
    assert proc is not None
    await extra.request("test/stopReading")
    tool = _make_tool(manager)
    assert isinstance(tool, EditTool)
    content = "# " + "x" * (2 * 1024 * 1024) + "\nvalue = undefined_var\n"
    monkeypatch.setattr(manager_module, "_AUTO_DIAGNOSTICS_TIMEOUT_SECONDS", 0.1)

    async with asyncio.timeout(10):
        result = await tool.claude_write("large.py", content)

    assert result == "Wrote large.py"
    assert (manager._project_path / "large.py").read_bytes() == content.encode()
    _assert_reaped(extra, proc)
    assert primary.running
    assert primary._pending == {}


async def _mutate(tool: EditTool | LspEditTool, operation: str) -> str:
    if isinstance(tool, LspEditTool):
        return await tool.lsp_edit(operation="rename", path="mutation.py", line=1, symbol="before", new_name="after")
    if operation in {"claude_write", "write", "hashline_write"}:
        return await getattr(tool, operation)("mutation.py", "after = 1\n")
    if operation == "claude_edit":
        return await tool.claude_edit("mutation.py", "before", "after")
    if operation in {"edit", "multi_edit"}:
        return await getattr(tool, operation)(
            "mutation.py", "<<<<<<< SEARCH\nbefore = 1\n=======\nafter = 1\n>>>>>>> REPLACE"
        )
    if operation == "hashline_edit":
        return await tool.hashline_edit(
            "mutation.py", [{"op": "set", "tag": format_line_tag(1, "before = 1"), "content": ["after = 1"]}]
        )
    assert operation == "apply_patch"
    return await tool.apply_patch(
        "*** Begin Patch\n*** Update File: mutation.py\n@@\n-before = 1\n+after = 1\n*** End Patch\n"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("operation", "success"),
    [
        ("claude_write", "Wrote mutation.py"),
        ("claude_edit", "Edited mutation.py"),
        ("write", "Wrote mutation.py"),
        ("edit", "Edited mutation.py"),
        ("multi_edit", "Edited mutation.py with 1 replacements"),
        ("hashline_write", "Wrote mutation.py"),
        ("hashline_edit", "Updated mutation.py"),
        ("apply_patch", "Success. Updated the following files:\nM mutation.py"),
        ("lsp_edit", "Applied LSP edit `rename`."),
    ],
)
@pytest.mark.parametrize("cancel", [False, True], ids=["deadline", "cancel"])
async def test_diagnostics_cannot_replay_or_lose_applied_mutation(
    fake_lsp_manager: LspManager,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
    success: str,
    cancel: bool,
) -> None:
    from kolega_code.services.lsp import manager as manager_module

    manager = fake_lsp_manager
    path = manager._project_path / "mutation.py"
    path.write_text("before = 1\n")
    tool = _make_tool(manager, lsp_edit=operation == "lsp_edit")
    snapshots = SnapshotService(
        manager._project_path,
        "workspace",
        "thread",
        "session",
        tool.filesystem,
        root=manager._project_path.parent / "state",
    )
    tool._snapshot_service = snapshots
    if isinstance(tool, EditTool):
        tool.observe_read("mutation.py")
    write = Mock(wraps=tool.filesystem.write_text)
    monkeypatch.setattr(tool.filesystem, "write_text", write)
    entered = asyncio.Event()
    settled = asyncio.Event()

    async def blocked_diagnostics(path: str) -> list[Any]:
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            settled.set()
        return []

    monkeypatch.setattr(manager, "get_fresh_diagnostics", blocked_diagnostics)
    monkeypatch.setattr(manager_module, "_AUTO_DIAGNOSTICS_TIMEOUT_SECONDS", 0.1 if not cancel else 30)
    mutation = asyncio.create_task(_mutate(tool, operation))
    try:
        async with asyncio.timeout(10):
            await entered.wait()
            assert path.read_bytes() == b"after = 1\n"
            assert len(snapshots.list_snapshots()) == 1
            assert _preview_count(tool) == 1
            if cancel:
                mutation.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await mutation
            else:
                result = await mutation
                if operation == "lsp_edit":
                    assert result.startswith(success)
                else:
                    assert result == success
        assert settled.is_set()
        assert path.read_bytes() == b"after = 1\n"
        assert write.call_count == 1
        assert len(snapshots.list_snapshots()) == 1
        if isinstance(tool, EditTool):
            assert tool._read_versions["mutation.py"] == sha256(b"after = 1\n").hexdigest()
        assert manager._diag_waiters == {}
        record = snapshots.latest_snapshot()
        assert record is not None
        snapshots.restore_snapshot(record.snapshot_id)
        assert path.read_bytes() == b"before = 1\n"
    finally:
        mutation.cancel()
        await asyncio.gather(mutation, return_exceptions=True)
