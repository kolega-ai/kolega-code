from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, Mock

import anyio
import pytest
from acp.exceptions import RequestError
from acp.interfaces import Client
from acp.schema import RequestPermissionResponse, TextContentBlock

from kolega_code.acp import agent_factory
from kolega_code.acp.accounting import AcpUsageAccounting
from kolega_code.acp.agent_factory import AgentFactory
from kolega_code.acp.server import AcpAgent
from kolega_code.acp.session import AcpSession
from kolega_code.llm.models import Message, TextBlock
from tests.acp.test_cleanup_lifecycle import configured_factory as configured_factory
from tests.acp.test_server import _FakeConn

pytestmark = [pytest.mark.asyncio, pytest.mark.usefixtures("isolated_cli_env")]


def _server(factory: AgentFactory, monkeypatch: pytest.MonkeyPatch) -> tuple[AcpAgent, _FakeConn]:
    server = AcpAgent(factory)
    conn = _FakeConn()
    server.on_connect(cast(Client, conn))
    monkeypatch.setattr(server, "_schedule_open_updates", lambda _sid: None)
    return server, conn


async def _lock_available(factory: AgentFactory, session_id: str) -> bool:
    code = (
        "from filelock import FileLock, Timeout\n"
        "import sys\n"
        "try:\n"
        "    with FileLock(sys.argv[1], timeout=0):\n"
        "        print('available')\n"
        "except Timeout:\n"
        "    print('owned')\n"
    )
    result = await asyncio.to_thread(
        subprocess.run,
        [sys.executable, "-c", code, str(factory._store.sessions_dir / f".{session_id}.lock")],
        capture_output=True,
        text=True,
        timeout=20,
        check=True,
    )
    return result.stdout.strip() == "available"


@pytest.mark.parametrize("failure", ["construct", "initialize", "restore_history"])
async def test_clear_approval_preserves_nonempty_history_and_epoch_on_rebuild_failure(
    configured_factory: AgentFactory,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    factory = configured_factory
    server, conn = _server(factory, monkeypatch)
    opened = await server.new_session(str(tmp_path))
    session = server._sessions[opened.session_id]
    await server.set_session_mode(session.session_id, "plan")
    old = session.agent
    message = Message(role="user", content=[TextBlock(text="Irreplaceable original context")])
    assert old.session_recorder is not None
    old.session_recorder.record_context_message(message)
    old.append_user_message(message.content)
    plan = "## Implement without losing the pending conversation on failure"
    session.record.latest_plan_markdown = plan
    session.record.plan_pending = True
    factory.persist(session)
    history = old.dump_message_history()
    assert factory._store.load(session.session_id).history == history
    journal = factory._store.journal(session.session_id)
    epoch = journal.epoch_id
    epochs = [event.to_dict() for event in journal.read_events() if event.event_type == "context.epoch_started"]
    conn.request_permission = AsyncMock(
        return_value=RequestPermissionResponse.model_validate(
            {"outcome": {"outcome": "selected", "optionId": "implement_plan_clear"}}
        )
    )
    try:
        assert history
        assert not await _lock_available(factory, session.session_id)
        with monkeypatch.context() as fault:
            error = RuntimeError(f"replacement {failure} failed")
            if failure == "construct":
                fault.setattr(agent_factory, "construct_agent", Mock(side_effect=error))
            elif failure == "initialize":
                from kolega_code.agent.tools import ToolCollection

                fault.setattr(ToolCollection, "initialize", AsyncMock(side_effect=error))
            else:
                from kolega_code.agent import CoderAgent

                fault.setattr(CoderAgent, "restore_message_history", Mock(side_effect=error))
            with pytest.raises(RuntimeError, match=f"replacement {failure} failed"):
                async with session.execution_lock:
                    await server._request_plan_approval(session, plan)
        persisted = factory._store.load(session.session_id)
        assert session.agent is old
        assert old.dump_message_history() == history, (
            f"history lengths: live={len(old.dump_message_history())}, record={len(session.record.history)}, "
            f"persisted={len(persisted.history)}; epoch changed={journal.epoch_id != epoch}"
        )
        assert session.record.history == history
        assert persisted.history == history
        assert journal.epoch_id == epoch
        assert [
            event.to_dict() for event in journal.read_events() if event.event_type == "context.epoch_started"
        ] == epochs
        assert persisted.interaction_mode == "plan" and persisted.plan_pending
        assert session.approved_plan is None and session.idle()
        assert not await _lock_available(factory, session.session_id)
        implementation = AsyncMock(return_value="end_turn")
        monkeypatch.setattr(server, "_run_turn", implementation)
        async with session.execution_lock:
            await server._request_plan_approval(session, plan)
        factory.persist(session)
        implementation.assert_awaited_once()
        assert session.agent is not old
        assert not session.agent.dump_message_history()
        assert journal.epoch_id != epoch
        assert sum(event.event_type == "context.epoch_started" for event in journal.read_events()) == len(epochs) + 1
        persisted = factory._store.load(session.session_id)
        assert not persisted.history and not persisted.plan_pending
        assert persisted.interaction_mode == "build"
    finally:
        await server.shutdown()
        factory._store.release_session_locks()


@pytest.mark.parametrize("operation", ["new", "load"])
@pytest.mark.parametrize("cancel_shutdown", [False, True])
async def test_shutdown_owns_inflight_open_until_construction_and_cleanup_finish(
    configured_factory: AgentFactory,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
    cancel_shutdown: bool,
) -> None:
    factory = configured_factory
    server, _ = _server(factory, monkeypatch)
    if operation == "load":
        seed = await server.new_session(str(tmp_path))
        session_id = seed.session_id
        await server.close_session(session_id)
    else:
        session_id = "opening-during-shutdown"
        monkeypatch.setattr("kolega_code.acp.server.uuid4", lambda: SimpleNamespace(hex=session_id))
    entered, release, shutdown_entered = (asyncio.Event() for _ in range(3))
    construct = factory._construct_agent
    shutdown = server._shutdown
    built: list[AcpSession] = []
    accounting: list[AcpUsageAccounting] = []

    async def blocked_construct(*args: Any, **kwargs: Any) -> Any:
        accounting.append(kwargs["accounting"])
        entered.set()
        await release.wait()
        return await construct(*args, **kwargs)

    async def observed_shutdown() -> None:
        shutdown_entered.set()
        await shutdown()

    monkeypatch.setattr(factory, "_construct_agent", blocked_construct)
    monkeypatch.setattr(server, "_shutdown", observed_shutdown)
    opening = asyncio.create_task(
        server.new_session(str(tmp_path)) if operation == "new" else server.load_session(str(tmp_path), session_id)
    )
    shutting: asyncio.Task[None] | None = None
    try:
        await asyncio.wait_for(entered.wait(), 10)
        assert not await _lock_available(factory, session_id)
        shutting = asyncio.create_task(server.shutdown())
        await asyncio.wait_for(shutdown_entered.wait(), 10)
        if cancel_shutdown:
            shutting.cancel()
            await asyncio.sleep(0)
            shutting.cancel()
        assert server._shutdown_task is not None and not server._shutdown_task.done()
        assert opening in server._session_openings.values()
        for attempt in (
            server.new_session(str(tmp_path)),
            server.load_session(str(tmp_path), session_id),
        ):
            with pytest.raises(RequestError):
                await attempt
        release.set()
        result = await asyncio.wait_for(asyncio.gather(opening, return_exceptions=True), 10)
        assert isinstance(result[0], RequestError), "an opening request succeeded after shutdown began"
        if cancel_shutdown:
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(shutting, 10)
        else:
            await asyncio.wait_for(shutting, 10)
        assert not server._sessions and not factory._acp_sessions
        assert len(accounting) == 1 and accounting[0]._closed
        sink = accounting[0].sink
        assert sink is not None and sink._drainer is not None and sink._drainer.done()
        assert sink.close_result is not None and sink.close_result.status != "pending"
        assert factory._store.load(session_id).session_id == session_id
        assert not await _lock_available(factory, session_id)
        factory._store.release_session_locks()
        assert await _lock_available(factory, session_id)
    finally:
        release.set()
        await asyncio.gather(opening, *([shutting] if shutting is not None else []), return_exceptions=True)
        built.extend(factory._acp_sessions.values())
        for session in built:
            await server.close_session(session.session_id)
        factory._store.release_session_locks()


@pytest.mark.skipif(sys.platform == "win32", reason="Requires POSIX PTYs")
async def test_stalled_task_affine_mcp_cleanup_retains_lock_and_shutdown_owner(
    configured_factory: AgentFactory, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    factory = configured_factory
    server, _ = _server(factory, monkeypatch)
    running, entered, release, exited = (asyncio.Event() for _ in range(4))
    owners: list[asyncio.Task[Any] | None] = []
    discoveries = 0

    async def call_tool(*args: Any, **kwargs: Any) -> Any:
        running.set()
        await asyncio.Event().wait()

    @asynccontextmanager
    async def transport(*args: Any, **kwargs: Any) -> AsyncGenerator[Any, None]:
        nonlocal discoveries
        discoveries += 1
        active = discoveries > 1
        owner = asyncio.current_task()
        if active:
            owners.append(owner)
        try:
            async with anyio.create_task_group():
                try:
                    yield SimpleNamespace(
                        list_tools=AsyncMock(
                            return_value=SimpleNamespace(
                                tools=[SimpleNamespace(name="echo", description="Echo", inputSchema={})]
                            )
                        ),
                        call_tool=call_tool,
                    )
                finally:
                    if active:
                        entered.set()
                        await release.wait()
                    assert asyncio.current_task() is owner
        finally:
            if active:
                exited.set()

    monkeypatch.setattr("kolega_code.mcp.service.open_mcp_session", transport)
    monkeypatch.setattr(agent_factory, "CLEANUP_TIMEOUT", 0.01)
    opened = await server.new_session(
        str(tmp_path), mcp_servers=cast(Any, [{"name": "owned", "command": "unused", "args": [], "env": []}])
    )
    session = server._sessions[opened.session_id]
    healthy_id = (await server.new_session(str(tmp_path))).session_id
    healthy = server._sessions[healthy_id]
    manager = session.agent.tool_collection.terminal_tool.terminal_manager
    process_result = await manager.exec_command("exec sleep 600", workdir=str(tmp_path), yield_time_ms=250)
    assert process_result.session_id is not None
    process = manager.sessions[process_result.session_id]
    extension = next(extension for extension in session.agent.tool_extensions if extension.name == "mcp")

    async def drive(*args: Any, **kwargs: Any) -> str:
        await next(iter(extension.tools.values()))()
        return "end_turn"

    monkeypatch.setattr(server, "_drive_turn", drive)
    assert session.accounting is not None
    close_accounting = AsyncMock(wraps=session.accounting.aclose)
    monkeypatch.setattr(session.accounting, "aclose", close_accounting)
    prompt = asyncio.create_task(server.prompt(session.session_id, [TextContentBlock(type="text", text="call echo")]))
    closing: asyncio.Task[None] | None = None
    shutting: asyncio.Task[None] | None = None
    try:
        await asyncio.wait_for(running.wait(), 10)
        closing = asyncio.create_task(server.close_session(session.session_id))
        await asyncio.wait_for(entered.wait(), 10)
        with pytest.raises(RequestError):
            await asyncio.wait_for(closing, 10)
        owner = session.close_task
        assert owner is not None and not owner.done()
        assert len(owners) == 1 and owners[0] is not owner
        shutting = asyncio.create_task(server.shutdown())
        async with asyncio.timeout(10):
            while healthy.close_task is None:
                await asyncio.sleep(0)
            await asyncio.shield(healthy.close_task)
        assert healthy_id not in factory._acp_sessions
        shutting.cancel()
        await asyncio.sleep(0)
        shutting.cancel()
        assert not shutting.done() and not owner.done()
        close_accounting.assert_not_awaited()
        assert not exited.is_set()
        os.kill(process.pid, 0)
        assert not process.exited.is_set()
        assert not await _lock_available(factory, session.session_id)
        assert server._sessions[session.session_id] is factory._acp_sessions[session.session_id] is session
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(shutting, 10)
        assert exited.is_set() and owner.done() and not owner.cancelled()
        assert process.master_fd is None
        if not process.exited.is_set():
            await asyncio.to_thread(os.waitpid, process.pid, 0)
        with pytest.raises(ProcessLookupError):
            os.kill(process.pid, 0)
        close_accounting.assert_awaited_once()
        assert not server._sessions and not factory._acp_sessions
        factory._store.release_session_locks()
        assert await _lock_available(factory, session.session_id)
    finally:
        release.set()
        await asyncio.gather(
            prompt,
            *([closing] if closing is not None else []),
            *([shutting] if shutting is not None else []),
            return_exceptions=True,
        )
        await server.shutdown()
        await manager.cleanup_all()
        factory._store.release_session_locks()
