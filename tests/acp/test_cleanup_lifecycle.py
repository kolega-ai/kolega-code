from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock

import pytest
from acp.exceptions import RequestError
from acp.interfaces import Client

from kolega_code.acp import agent_factory
from kolega_code.acp.agent_factory import AgentFactory
from kolega_code.acp.server import AcpAgent
from kolega_code.agent import PlanningAgent
from kolega_code.agent.tools import ToolCollection
from tests.acp.test_server import _FakeConn

pytestmark = pytest.mark.usefixtures("isolated_cli_env")


@pytest.fixture
def configured_factory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AgentFactory:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-composition-provider-key")
    monkeypatch.setenv("KOLEGA_CODE_PROVIDER", "anthropic")
    monkeypatch.setenv("KOLEGA_CODE_MODEL", "claude-haiku-4-5-20251001")
    monkeypatch.setattr(ToolCollection, "initialize", AsyncMock(return_value=[]))
    return AgentFactory()


@pytest.mark.parametrize("retired", [False, True])
def test_run_acp_eof_joins_finite_cleanup_before_closing_accounting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, retired: bool
) -> None:
    import acp
    import acp.stdio

    from kolega_code.cli.acp import run_acp

    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-composition-provider-key")
    monkeypatch.setenv("KOLEGA_CODE_PROVIDER", "anthropic")
    monkeypatch.setenv("KOLEGA_CODE_MODEL", "claude-haiku-4-5-20251001")
    monkeypatch.setattr(ToolCollection, "initialize", AsyncMock(return_value=[]))
    monkeypatch.setattr(agent_factory, "CLEANUP_TIMEOUT", 0.02)
    monkeypatch.setattr(acp.stdio, "stdio_streams", AsyncMock(return_value=(object(), object())))
    records: dict[str, Any] = {}
    steps: list[str] = []

    async def transport_eof(server: AcpAgent, **kwargs: Any) -> None:
        server.on_connect(cast(Client, _FakeConn()))
        monkeypatch.setattr(server, "_schedule_open_updates", lambda _sid: None)
        opened = await server.new_session(str(tmp_path))
        session = server._sessions[opened.session_id]
        records.update(server=server, session=session)
        original_cleanup = session.agent.cleanup
        assert session.accounting is not None
        close_accounting = AsyncMock(wraps=session.accounting.aclose)
        monkeypatch.setattr(session.accounting, "aclose", close_accounting)
        records["accounting_close"] = close_accounting

        async def finite_cleanup() -> None:
            steps.append("cleanup-started")
            try:
                await asyncio.sleep(0.15)
                close_accounting.assert_not_awaited()
                assert server._factory._acp_sessions[session.session_id] is session
                await original_cleanup()
                steps.append("cleanup-completed")
            except asyncio.CancelledError:
                steps.append("cleanup-cancelled")
                raise

        monkeypatch.setattr(session.agent, "cleanup", finite_cleanup)
        if retired:
            await server.set_session_mode(session.session_id, "plan")
            assert session.cleanup_tasks

    monkeypatch.setattr(acp, "run_agent", transport_eof)
    assert run_acp(SimpleNamespace(debug=False, acp_log=None, permission_mode="ask")) == 0
    session = records["session"]
    server = records["server"]
    assert steps == ["cleanup-started", "cleanup-completed"]
    assert session.close_task.done() and not session.close_task.cancelled()
    assert not session.cleanup_tasks and not session.retired_agents
    assert session.session_id not in server._sessions
    assert session.session_id not in server._factory._acp_sessions
    records["accounting_close"].assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("normal_cleanup_delay", [0, 0.05], ids=["normal-close", "delayed-close"])
@pytest.mark.parametrize("cancelled", [False, True])
async def test_stalled_mode_cleanup_releases_request_lock_without_releasing_ownership(
    tmp_path: Path,
    configured_factory: AgentFactory,
    monkeypatch: pytest.MonkeyPatch,
    cancelled: bool,
    normal_cleanup_delay: float,
) -> None:
    original_cleanup_timeout = agent_factory.CLEANUP_TIMEOUT
    monkeypatch.setattr(agent_factory, "CLEANUP_TIMEOUT", 0.02, raising=False)
    factory = configured_factory
    server = AcpAgent(factory)
    server.on_connect(cast(Client, _FakeConn()))
    monkeypatch.setattr(server, "_schedule_open_updates", lambda _session_id: None)
    opened = await server.new_session(str(tmp_path))
    session = server._sessions[opened.session_id]
    old = session.agent
    original_cleanup = old.cleanup
    started, release = asyncio.Event(), asyncio.Event()
    calls = 0
    cleanup_tasks: list[asyncio.Task[Any]] = []

    async def cleanup() -> None:
        nonlocal calls
        calls += 1
        cleanup_task = asyncio.current_task()
        assert cleanup_task is not None
        cleanup_tasks.append(cleanup_task)
        started.set()
        await release.wait()
        await original_cleanup()

    old.cleanup = cleanup
    switch = asyncio.create_task(server.set_session_mode(session.session_id, "plan"))
    try:
        await asyncio.wait_for(started.wait(), 5)
        adopted = session.agent
        assert adopted is not old
        normal_cleanup = adopted.cleanup

        async def cleanup_adopted() -> None:
            await asyncio.sleep(normal_cleanup_delay)
            await normal_cleanup()

        adopted_cleanup = AsyncMock(wraps=cleanup_adopted)
        monkeypatch.setattr(adopted, "cleanup", adopted_cleanup)
        assert session.accounting is not None
        close_accounting = AsyncMock(wraps=session.accounting.aclose)
        monkeypatch.setattr(session.accounting, "aclose", close_accounting)
        if cancelled:
            switch.cancel()
            await asyncio.sleep(0)
            switch.cancel()
        done, _ = await asyncio.wait({switch}, timeout=5)
        assert switch in done, "stalled cleanup still holds the request lock"
        if cancelled:
            with pytest.raises(asyncio.CancelledError):
                await switch
        else:
            await switch
        assert not session.execution_lock.locked()
        assert len(cleanup_tasks) == 1
        cleanup_task = cleanup_tasks[0]
        assert not cleanup_task.done()
        assert session.cleanup_tasks[id(old)] is cleanup_task
        assert any(agent is old for agent in session.retired_agents)
        assert not session.idle()
        assert isinstance(adopted, PlanningAgent)
        assert session.record.interaction_mode == "plan"
        assert factory._store.load(session.session_id).interaction_mode == "plan"
        for operation in (
            lambda: factory.close_session(session.session_id),
            lambda: server.load_session(str(tmp_path), session.session_id),
            lambda: factory.load_session(str(tmp_path), session.session_id),
            lambda: factory.open_session(str(tmp_path), session_id=session.session_id),
            lambda: server.set_session_mode(session.session_id, "build"),
            lambda: factory.set_interaction_mode(session, "build"),
            lambda: factory.apply_model(session, "anthropic", session.config.long_context_config.model),
            lambda: server.prompt(session.session_id, []),
        ):
            with pytest.raises(RequestError):
                await operation()
        recovered = asyncio.Event()

        async def recover() -> None:
            recovered.set()

        jobs = SimpleNamespace(recover=recover, pending_count=0, deliver=AsyncMock())
        session.background_wake.set()
        worker = asyncio.create_task(server._background_worker(session, jobs))
        try:
            await asyncio.wait_for(recovered.wait(), 5)
            assert not session.background_wake.is_set()
            jobs.deliver.assert_not_awaited()
        finally:
            worker.cancel()
            await asyncio.gather(worker, return_exceptions=True)
        assert server._sessions[session.session_id] is session
        assert factory._acp_sessions[session.session_id] is session
        assert not session.closed
        assert session.config is not None
        adopted_cleanup.assert_not_awaited()
        close_accounting.assert_not_awaited()
        assert calls == 1
        assert session.cleanup_tasks[id(old)] is cleanup_task
        release.set()
        await asyncio.wait_for(asyncio.shield(cleanup_task), 5)
        assert not session.cleanup_tasks
        assert old not in session.retired_agents
        assert session.idle()
        assert await factory.load_session(str(tmp_path), session.session_id) is session
        monkeypatch.setattr(agent_factory, "CLEANUP_TIMEOUT", original_cleanup_timeout)
        await asyncio.wait_for(server.close_session(session.session_id), 10)
        adopted_cleanup.assert_awaited_once()
        close_accounting.assert_awaited_once()
        assert calls == 1
    finally:
        monkeypatch.setattr(agent_factory, "CLEANUP_TIMEOUT", original_cleanup_timeout)
        release.set()
        await asyncio.gather(switch, return_exceptions=True)
        await asyncio.wait_for(server.close_session(session.session_id), 10)


@pytest.mark.asyncio
async def test_queued_close_rechecks_cleanup_before_removing_session(
    tmp_path: Path, configured_factory: AgentFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_cleanup_timeout = agent_factory.CLEANUP_TIMEOUT
    monkeypatch.setattr(agent_factory, "CLEANUP_TIMEOUT", 0.02)
    factory = configured_factory
    session = await factory.open_session(str(tmp_path))
    server = AcpAgent(factory)
    server._sessions[session.session_id] = session
    old = session.agent
    cleanup = old.cleanup
    release, constructing, finish_construct = asyncio.Event(), asyncio.Event(), asyncio.Event()
    construct = factory._construct_agent

    async def blocked_construct(*args: Any, **kwargs: Any) -> Any:
        constructing.set()
        await finish_construct.wait()
        return await construct(*args, **kwargs)

    async def blocked_cleanup() -> None:
        await release.wait()
        await cleanup()

    monkeypatch.setattr(old, "cleanup", blocked_cleanup)
    monkeypatch.setattr(factory, "_construct_agent", blocked_construct)
    rebuild = asyncio.create_task(server.set_session_mode(session.session_id, "plan"))
    await asyncio.wait_for(constructing.wait(), 5)
    closing = asyncio.create_task(server.close_session(session.session_id))
    try:
        await asyncio.sleep(0)
        assert not closing.done()
        assert session.closed
        finish_construct.set()
        await asyncio.wait_for(rebuild, 5)
        assert session.cleanup_tasks
        with pytest.raises(RequestError):
            await asyncio.wait_for(closing, 5)
        assert server._sessions[session.session_id] is session
        assert factory._acp_sessions[session.session_id] is session
        assert session.closed
    finally:
        monkeypatch.setattr(agent_factory, "CLEANUP_TIMEOUT", original_cleanup_timeout)
        release.set()
        finish_construct.set()
        await asyncio.gather(rebuild, closing, *session.cleanup_tasks.values(), return_exceptions=True)
        await server.close_session(session.session_id)
