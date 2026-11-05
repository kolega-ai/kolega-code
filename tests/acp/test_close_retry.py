from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
from acp.exceptions import RequestError
from acp.interfaces import Client

from kolega_code.acp import agent_factory
from kolega_code.acp.agent_factory import AgentFactory
from kolega_code.acp.server import AcpAgent
from kolega_code.acp.session import AcpSession
from tests.acp.test_cleanup_lifecycle import configured_factory as configured_factory
from tests.acp.test_server import _FakeConn

pytestmark = [pytest.mark.asyncio, pytest.mark.usefixtures("isolated_cli_env")]


class AdapterFailure(RuntimeError):
    pass


class AccountingFailure(RuntimeError):
    pass


@dataclass
class Opened:
    server: AcpAgent
    factory: AgentFactory
    session: AcpSession
    adapter: AsyncMock
    factory_close: AsyncMock
    cleanup: AsyncMock
    accounting: AsyncMock
    hook: AsyncMock
    shutdown_adapter: Callable[[], Awaitable[None]]
    close_factory: Callable[[str], Awaitable[None]]


@pytest_asyncio.fixture
async def opened(
    configured_factory: AgentFactory, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> AsyncIterator[Opened]:
    factory = configured_factory
    server = AcpAgent(factory)
    server.on_connect(cast(Client, _FakeConn()))
    monkeypatch.setattr(server, "_schedule_open_updates", lambda _sid: None)
    response = await server.new_session(str(tmp_path))
    session = server._sessions[response.session_id]
    server._start_session_tasks(session)
    assert session.control_adapter is not None and session.accounting is not None
    shutdown_adapter = session.control_adapter.shutdown
    close_factory = factory.close_session
    close_accounting = session.accounting.aclose
    adapter = AsyncMock(wraps=shutdown_adapter)
    factory_close = AsyncMock(wraps=close_factory)
    cleanup = AsyncMock(wraps=session.agent.cleanup)
    accounting = AsyncMock(wraps=close_accounting)
    hook = AsyncMock(wraps=session.agent.fire_hook)
    monkeypatch.setattr(session.control_adapter, "shutdown", adapter)
    monkeypatch.setattr(factory, "close_session", factory_close)
    monkeypatch.setattr(session.agent, "cleanup", cleanup)
    monkeypatch.setattr(session.accounting, "aclose", accounting)
    monkeypatch.setattr(session.agent, "fire_hook", hook)
    try:
        yield Opened(
            server,
            factory,
            session,
            adapter,
            factory_close,
            cleanup,
            accounting,
            hook,
            shutdown_adapter,
            close_factory,
        )
    finally:
        monkeypatch.setattr(session.control_adapter, "shutdown", shutdown_adapter)
        monkeypatch.setattr(session.accounting, "aclose", close_accounting)
        if session.session_id in factory._acp_sessions:
            await shutdown_adapter()
            await close_factory(session.session_id)
        tasks = [
            task
            for task in (
                session.turn_task,
                session.background_task,
                session.event_task,
                session.plan_approval_task,
                session.close_task,
            )
            if task is not None
        ]
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        server._sessions.pop(session.session_id, None)


async def _fail_adapter(opened: Opened) -> asyncio.Task[None]:
    failure = AdapterFailure("adapter shutdown failed before factory cleanup")
    opened.adapter.side_effect = failure
    with pytest.raises(AdapterFailure) as raised:
        await opened.server.close_session(opened.session.session_id)
    assert raised.value is failure
    task = opened.session.close_task
    assert task is not None and task.done() and task.exception() is failure
    opened.factory_close.assert_not_awaited()
    opened.cleanup.assert_not_awaited()
    opened.accounting.assert_not_awaited()
    opened.hook.assert_not_awaited()
    return task


def _assert_released(opened: Opened) -> None:
    session = opened.session
    opened.factory_close.assert_awaited_once_with(session.session_id)
    opened.cleanup.assert_awaited_once()
    opened.accounting.assert_awaited_once()
    opened.hook.assert_not_awaited()
    assert session.session_id not in opened.server._sessions
    assert session.session_id not in opened.factory._acp_sessions
    assert session.event_task is not None and session.event_task.done()
    assert session.background_task is not None and session.background_task.done()
    assert session.runtime is not None and session.runtime.control._lease is None
    assert session.closed and not session.idle()


@pytest.mark.parametrize("failures", [1, 2])
async def test_completed_adapter_failures_can_retry_without_factory_replay(opened: Opened, failures: int) -> None:
    previous = None
    for _ in range(failures):
        task = await _fail_adapter(opened)
        assert task is not previous
        previous = task
        assert opened.session.session_id in opened.server._sessions
        assert opened.session.session_id in opened.factory._acp_sessions
        assert opened.session.closed and not opened.session.idle()
        assert opened.session.event_task is not None and not opened.session.event_task.done()
    opened.adapter.side_effect = opened.shutdown_adapter
    await opened.server.close_session(opened.session.session_id)
    assert opened.session.close_task is not previous
    assert opened.adapter.await_count == failures + 1
    _assert_released(opened)


@pytest.mark.parametrize("cancel_waiter", [False, True])
async def test_concurrent_retries_share_owner_despite_waiter_cancellation(
    opened: Opened, monkeypatch: pytest.MonkeyPatch, cancel_waiter: bool
) -> None:
    failed = await _fail_adapter(opened)
    entered, release, joined = (asyncio.Event() for _ in range(3))
    owners: list[asyncio.Task[None]] = []
    start_close = opened.server._start_close

    def track_start(session: AcpSession) -> asyncio.Task[None]:
        task = start_close(session)
        owners.append(task)
        if len(owners) == 2:
            joined.set()
        return task

    async def shutdown() -> None:
        entered.set()
        await release.wait()
        await opened.shutdown_adapter()

    monkeypatch.setattr(opened.server, "_start_close", track_start)
    opened.adapter.side_effect = shutdown
    first = asyncio.create_task(opened.server.close_session(opened.session.session_id))
    second: asyncio.Task[None] | None = None
    try:
        await asyncio.wait_for(entered.wait(), 10)
        second = asyncio.create_task(opened.server.close_session(opened.session.session_id))
        await asyncio.wait_for(joined.wait(), 10)
        assert owners[0] is owners[1] is opened.session.close_task
        assert owners[0] is not failed and not owners[0].done()
        opened.factory_close.assert_not_awaited()
        if cancel_waiter:
            first.cancel()
            await asyncio.sleep(0)
            first.cancel()
            await asyncio.sleep(0)
            assert not owners[0].done()
        release.set()
        results = await asyncio.wait_for(asyncio.gather(first, second, return_exceptions=True), 10)
        if cancel_waiter:
            assert isinstance(results[0], asyncio.CancelledError)
        else:
            assert results[0] is None
        assert results[1] is None
        assert opened.session.close_task is owners[0]
        assert opened.adapter.await_count == 2
        _assert_released(opened)
    finally:
        release.set()
        await asyncio.gather(first, *([second] if second is not None else []), return_exceptions=True)


async def test_retry_timeout_keeps_live_owner(opened: Opened, monkeypatch: pytest.MonkeyPatch) -> None:
    await _fail_adapter(opened)
    entered, release = asyncio.Event(), asyncio.Event()

    async def shutdown() -> None:
        entered.set()
        await release.wait()
        await opened.shutdown_adapter()

    opened.adapter.side_effect = shutdown
    monkeypatch.setattr(agent_factory, "CLEANUP_TIMEOUT", 0.01)
    request = asyncio.create_task(opened.server.close_session(opened.session.session_id))
    try:
        await asyncio.wait_for(entered.wait(), 10)
        owner = opened.session.close_task
        assert owner is not None
        with pytest.raises(RequestError) as timed_out:
            await asyncio.wait_for(request, 10)
        assert timed_out.value.data == {"message": "Session close is still pending."}
        with pytest.raises(RequestError) as retried:
            await opened.server.close_session(opened.session.session_id)
        assert retried.value.data == {"message": "Session close is still pending."}
        assert opened.session.close_task is owner and not owner.done()
        assert opened.adapter.await_count == 2
        opened.factory_close.assert_not_awaited()
        release.set()
        await asyncio.wait_for(asyncio.shield(owner), 10)
        _assert_released(opened)
    finally:
        release.set()
        await asyncio.gather(request, return_exceptions=True)


@pytest.mark.parametrize("adapter_failed_first", [False, True])
async def test_later_factory_failure_never_replays_cleanup_or_hooks(opened: Opened, adapter_failed_first: bool) -> None:
    if adapter_failed_first:
        await _fail_adapter(opened)
        opened.adapter.side_effect = opened.shutdown_adapter
    factory_hook = AsyncMock()

    async def close_factory(session_id: str) -> None:
        await factory_hook()
        await opened.close_factory(session_id)

    opened.factory_close.side_effect = close_factory
    failure = AccountingFailure("accounting failed after non-idempotent factory work")
    opened.accounting.side_effect = failure
    with pytest.raises(AccountingFailure) as raised:
        await opened.server.close_session(opened.session.session_id)
    assert raised.value is failure
    failed = opened.session.close_task
    for _ in range(2):
        with pytest.raises(AccountingFailure) as retried:
            await opened.server.close_session(opened.session.session_id)
        assert retried.value is failure
        assert opened.session.close_task is failed
    assert opened.adapter.await_count == 1 + int(adapter_failed_first)
    opened.factory_close.assert_awaited_once()
    factory_hook.assert_awaited_once()
    opened.cleanup.assert_awaited_once()
    opened.accounting.assert_awaited_once()
    opened.hook.assert_not_awaited()
    assert opened.session.session_id in opened.server._sessions
    assert opened.session.session_id in opened.factory._acp_sessions


@pytest.mark.parametrize("adapter_failed_first", [False, True])
async def test_failed_retired_cleanup_remains_fenced(opened: Opened, adapter_failed_first: bool) -> None:
    if adapter_failed_first:
        await _fail_adapter(opened)
        opened.adapter.side_effect = opened.shutdown_adapter
    retired = SimpleNamespace(cleanup=AsyncMock(side_effect=RuntimeError("uncertain retired cleanup")))
    task = AgentFactory.start_retired_cleanup(opened.session, retired)
    completed = asyncio.Event()
    task.add_done_callback(lambda _done: completed.set())
    try:
        await asyncio.wait_for(completed.wait(), 10)
        assert isinstance(task.exception(), RuntimeError)
        failed = None
        for _ in range(2):
            with pytest.raises(RequestError) as blocked:
                await opened.server.close_session(opened.session.session_id)
            assert blocked.value.data == {"message": "Agent cleanup is incomplete; retry after it finishes."}
            if failed is None:
                failed = opened.session.close_task
            assert opened.session.close_task is failed
        retired.cleanup.assert_awaited_once()
        assert opened.session.cleanup_tasks[id(retired)] is task
        assert any(old is retired for old in opened.session.retired_agents)
        assert opened.adapter.await_count == int(adapter_failed_first)
        opened.factory_close.assert_not_awaited()
        opened.cleanup.assert_not_awaited()
        opened.accounting.assert_not_awaited()
        assert opened.session.session_id in opened.server._sessions
        assert opened.session.session_id in opened.factory._acp_sessions
    finally:
        opened.session.cleanup_tasks.clear()
        opened.session.retired_agents.clear()


async def test_cancelled_adapter_owner_is_not_generically_restarted(opened: Opened) -> None:
    opened.adapter.side_effect = asyncio.CancelledError()
    failed = None
    for _ in range(2):
        with pytest.raises(asyncio.CancelledError):
            await opened.server.close_session(opened.session.session_id)
        if failed is None:
            failed = opened.session.close_task
        assert opened.session.close_task is failed
    assert failed is not None and failed.cancelled()
    opened.adapter.assert_awaited_once()
    opened.factory_close.assert_not_awaited()
    opened.cleanup.assert_not_awaited()


async def test_shutdown_recovers_completed_adapter_failure_once(opened: Opened) -> None:
    failed = await _fail_adapter(opened)
    opened.adapter.side_effect = opened.shutdown_adapter
    await opened.server.shutdown()
    await opened.server.shutdown()
    assert opened.session.close_task is not failed
    assert opened.adapter.await_count == 2
    _assert_released(opened)
