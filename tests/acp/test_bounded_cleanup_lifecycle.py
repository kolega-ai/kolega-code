from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, Mock

import pytest
from acp.exceptions import RequestError
from acp.interfaces import Client
from acp.schema import RequestPermissionResponse, SetSessionModeResponse, TextContentBlock

from kolega_code.acp import agent_factory
from kolega_code.acp.agent_factory import AgentFactory
from kolega_code.acp.server import AcpAgent
from kolega_code.acp.session import AcpSession
from kolega_code.agent import CoderAgent
from kolega_code.agent.tools import ToolCollection
from kolega_code.llm.models import Message, TextBlock
from tests.acp.test_composition import configured_factory as configured_factory
from tests.acp.test_server import _FakeConn

pytestmark = [pytest.mark.asyncio, pytest.mark.usefixtures("isolated_cli_env")]


async def _open(
    factory: AgentFactory, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[AcpAgent, AcpSession, _FakeConn]:
    server = AcpAgent(factory)
    conn = _FakeConn()
    server.on_connect(cast(Client, conn))
    monkeypatch.setattr(server, "_schedule_open_updates", lambda _session_id: None)
    opened = await server.new_session(str(tmp_path))
    return server, server._sessions[opened.session_id], conn


@pytest.fixture(params=[False, True], ids=["normal-setup", "slow-setup"])
def slow_setup_cleanup(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    if not request.param:
        return
    # This case holds the previous agent's cleanup open until the mode switch
    # returns, so the join always runs to its bound: shrink it below the 5 s
    # production timeout, as the sibling ACP cleanup suites already do.
    monkeypatch.setattr(agent_factory, "CLEANUP_TIMEOUT", 0.05)
    set_mode = AcpAgent.set_session_mode
    setup_timeout = agent_factory.CLEANUP_TIMEOUT

    async def slow_set_mode(
        server: AcpAgent, session_id: str, mode_id: str, **kwargs: Any
    ) -> SetSessionModeResponse | None:
        assert mode_id == "plan"
        assert agent_factory.CLEANUP_TIMEOUT == setup_timeout
        session = server._sessions[session_id]
        old = session.agent
        cleanup = old.cleanup
        release = asyncio.Event()

        async def delayed_cleanup() -> None:
            await release.wait()
            assert agent_factory.CLEANUP_TIMEOUT == setup_timeout
            await cleanup()

        monkeypatch.setattr(old, "cleanup", delayed_cleanup)
        try:
            result = await set_mode(server, session_id, mode_id, **kwargs)
            assert id(old) in session.cleanup_tasks
            assert not session.idle()
            return result
        finally:
            release.set()

    monkeypatch.setattr(AcpAgent, "set_session_mode", slow_set_mode)


async def test_close_fences_prompt_before_cancellation_finishes(
    configured_factory: AgentFactory, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    server, session, _ = await _open(configured_factory, tmp_path, monkeypatch)
    running, cancelling, release, leaked = (asyncio.Event() for _ in range(4))
    clean = AsyncMock(wraps=session.agent.cleanup)
    monkeypatch.setattr(session.agent, "cleanup", clean)

    async def draining() -> None:
        running.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            session.turn_task = None
            cancelling.set()
            await release.wait()

    async def new_turn(*args: Any, **kwargs: Any) -> str:
        leaked.set()
        await release.wait()
        return "end_turn"

    monkeypatch.setattr(server, "_run_turn", new_turn)
    turn = session.turn_task = asyncio.create_task(draining())
    await asyncio.wait_for(running.wait(), 5)
    closing = asyncio.create_task(server.close_session(session.session_id))
    request: asyncio.Task[Any] | None = None
    observed: asyncio.Task[bool] | None = None
    try:
        await asyncio.wait_for(cancelling.wait(), 5)
        request = asyncio.create_task(
            server.prompt(session.session_id, [TextContentBlock(type="text", text="new prompt")])
        )
        observed = asyncio.create_task(leaked.wait())
        done, _ = await asyncio.wait({request, observed}, timeout=5, return_when=asyncio.FIRST_COMPLETED)
        assert request in done, "close admitted a new prompt while joining cancellation"
        with pytest.raises(RequestError):
            await request
        assert session.closed
        clean.assert_not_awaited()
        release.set()
        await asyncio.wait_for(closing, 5)
        clean.assert_awaited_once()
    finally:
        release.set()
        if observed is not None:
            observed.cancel()
        await asyncio.gather(
            turn,
            closing,
            *([request] if request is not None else []),
            *([observed] if observed is not None else []),
            return_exceptions=True,
        )
        await server.shutdown()


@pytest.mark.parametrize("cancel_close", [False, True])
@pytest.mark.parametrize("cancel_shutdown", [False, True])
async def test_shutdown_drains_healthy_sessions_and_retains_pending_close(
    configured_factory: AgentFactory,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cancel_close: bool,
    cancel_shutdown: bool,
) -> None:
    monkeypatch.setattr(agent_factory, "CLEANUP_TIMEOUT", 0.02, raising=False)
    factory = configured_factory
    server, session, _ = await _open(factory, tmp_path, monkeypatch)
    healthy_id = (await server.new_session(str(tmp_path))).session_id
    healthy = server._sessions[healthy_id]
    healthy_cleanup = AsyncMock(wraps=healthy.agent.cleanup)
    monkeypatch.setattr(healthy.agent, "cleanup", healthy_cleanup)
    old = session.agent
    original_cleanup = old.cleanup
    started, release = asyncio.Event(), asyncio.Event()
    actual: list[asyncio.Task[Any]] = []
    calls = 0

    async def cleanup() -> None:
        nonlocal calls
        calls += 1
        current = asyncio.current_task()
        assert current is not None
        actual.append(current)
        started.set()
        await release.wait()
        await original_cleanup()

    monkeypatch.setattr(old, "cleanup", cleanup)
    switching = asyncio.create_task(server.set_session_mode(session.session_id, "plan"))
    retry: asyncio.Task[None] | None = None
    shutting: asyncio.Task[None] | None = None
    try:
        await asyncio.wait_for(started.wait(), 5)
        done, _ = await asyncio.wait({switching}, timeout=5)
        assert switching in done, "cleanup did not release the mode request"
        await switching
        adopted_cleanup = AsyncMock(wraps=session.agent.cleanup)
        monkeypatch.setattr(session.agent, "cleanup", adopted_cleanup)
        assert session.accounting is not None
        close_accounting = AsyncMock(wraps=session.accounting.aclose)
        monkeypatch.setattr(session.accounting, "aclose", close_accounting)
        assert session.control_adapter is not None
        close_control = AsyncMock(wraps=session.control_adapter.shutdown)
        monkeypatch.setattr(session.control_adapter, "shutdown", close_control)
        shutting = asyncio.create_task(server.shutdown())
        async with asyncio.timeout(5):
            while healthy.close_task is None:
                await asyncio.sleep(0)
            await asyncio.shield(healthy.close_task)
        healthy_cleanup.assert_awaited_once()
        assert healthy_id not in server._sessions
        assert healthy_id not in factory._acp_sessions
        assert session.closed and not session.idle()
        assert server._sessions[session.session_id] is session
        assert factory._acp_sessions[session.session_id] is session
        assert session.cleanup_tasks[id(old)] is actual[0]
        close_task = session.close_task
        assert close_task is not None and not close_task.done()
        assert not shutting.done()
        if cancel_shutdown:
            shutting.cancel()
            await asyncio.sleep(0)
            shutting.cancel()
            await asyncio.sleep(0)
            assert not shutting.done()
            assert server._shutdown_task is not None and not server._shutdown_task.done()
        retry = asyncio.create_task(server.close_session(session.session_id))
        if cancel_close:
            await asyncio.sleep(0)
            retry.cancel()
            await asyncio.sleep(0)
            retry.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(retry, 5)
        else:
            with pytest.raises(RequestError):
                await asyncio.wait_for(retry, 5)
        assert session.close_task is close_task
        adopted_cleanup.assert_not_awaited()
        close_accounting.assert_not_awaited()
        close_control.assert_not_awaited()
        for operation in (
            lambda: server.new_session(str(tmp_path)),
            lambda: server.load_session(str(tmp_path), session.session_id),
            lambda: factory.load_session(str(tmp_path), session.session_id),
            lambda: factory.open_session(str(tmp_path), session_id=session.session_id),
            lambda: server.set_config_option("permission-auto", session.session_id, True),
        ):
            with pytest.raises(RequestError):
                await operation()
        release.set()
        await asyncio.wait_for(asyncio.shield(close_task), 5)
        if cancel_shutdown:
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(shutting, 5)
        else:
            await asyncio.wait_for(shutting, 5)
        assert calls == 1
        adopted_cleanup.assert_awaited_once()
        close_accounting.assert_awaited_once()
        close_control.assert_awaited_once()
        assert session.session_id not in server._sessions
        assert session.session_id not in factory._acp_sessions
        assert not session.cleanup_tasks and not session.retired_agents
    finally:
        release.set()
        await asyncio.gather(
            switching,
            *([retry] if retry is not None else []),
            *([shutting] if shutting is not None else []),
            return_exceptions=True,
        )
        await asyncio.gather(*getattr(session, "cleanup_tasks", {}).values(), return_exceptions=True)
        await server.shutdown()


@pytest.mark.parametrize("option", ["implement_plan", "implement_plan_clear"])
@pytest.mark.parametrize("failure", ["construct", "initialize", "restore_history", "persist", "invalidate_context"])
async def test_failed_approval_preparation_preserves_plan_and_allows_public_retries(
    configured_factory: AgentFactory,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    option: str,
    failure: str,
) -> None:
    server, session, conn = await _open(configured_factory, tmp_path, monkeypatch)
    await server.set_session_mode(session.session_id, "plan")
    old = session.agent
    plan = "## Retry this exact pending plan after preparation fails"
    modes: list[str] = []

    async def drive(current: AcpSession, bridge: Any, text: str | None, attachments: Any) -> str:
        mode = current.record.interaction_mode
        modes.append(mode)
        if mode == "plan":
            setattr(current.agent, "_completed_plan", plan)
        return "end_turn"

    monkeypatch.setattr(server, "_drive_turn", drive)
    conn.request_permission = AsyncMock(
        return_value=RequestPermissionResponse.model_validate({"outcome": {"outcome": "selected", "optionId": option}})
    )
    try:
        with monkeypatch.context() as fault:
            error = RuntimeError(f"failed {failure}")
            if failure == "construct":
                fault.setattr(agent_factory, "construct_agent", Mock(side_effect=error))
            elif failure == "initialize":
                fault.setattr(ToolCollection, "initialize", AsyncMock(side_effect=error))
            elif failure == "restore_history":
                fault.setattr(CoderAgent, "restore_message_history", Mock(side_effect=error))
            elif failure == "persist":
                fault.setattr(configured_factory, "persist", Mock(side_effect=error))
            else:
                assert session.accounting is not None
                fault.setattr(session.accounting, "invalidate_context", Mock(side_effect=error))
            with pytest.raises(RuntimeError, match=f"failed {failure}"):
                await server.prompt(session.session_id, [TextContentBlock(type="text", text="Make a plan")])
        assert session.agent is old
        assert session.record.interaction_mode == "plan"
        assert session.record.plan_pending
        assert session.record.latest_plan_markdown == plan
        assert session.approved_plan is None
        assert session.plan_approval_task is None
        assert not session.cleanup_tasks
        assert session.idle()
        assert modes == ["plan"]
        await server.set_session_mode(session.session_id, "plan")
        await server.set_config_option("mode", session.session_id, "plan")
        result = await server.prompt(session.session_id, [TextContentBlock(type="text", text="Retry the plan")])
        assert result.stop_reason == "end_turn"
        async with asyncio.timeout(5):
            async with session.execution_lock:
                pass
        assert modes == ["plan", "plan", "build"]
        assert conn.request_permission.await_count == 2
        assert not session.record.plan_pending
        assert session.record.latest_plan_markdown == plan
        assert session.approved_plan is None
        assert session.idle()
    finally:
        await server.shutdown()


@pytest.mark.parametrize("mode", ["plan", "build"])
@pytest.mark.parametrize("failure", ["close_jobs", "deliver_jobs", "clear_context"])
async def test_failed_clear_context_preparation_releases_approval(
    configured_factory: AgentFactory,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    failure: str,
) -> None:
    server, session, conn = await _open(configured_factory, tmp_path, monkeypatch)
    if mode == "plan":
        await server.set_session_mode(session.session_id, mode)
    old = session.agent
    plan = "## Preserve this plan when clearing context fails"
    session.record.latest_plan_markdown = plan
    session.record.plan_pending = True
    conn.request_permission = AsyncMock(
        return_value=RequestPermissionResponse.model_validate(
            {"outcome": {"outcome": "selected", "optionId": "implement_plan_clear"}}
        )
    )
    run = AsyncMock(return_value="end_turn")
    monkeypatch.setattr(server, "_run_turn", run)
    try:
        with monkeypatch.context() as fault:
            error = RuntimeError(f"failed {failure}")
            if failure == "clear_context":
                fault.setattr(old, "reset_volatile_context", Mock(side_effect=error))
            else:
                jobs = old.get_background_jobs()
                fault.setattr(jobs, "close" if failure == "close_jobs" else "deliver", AsyncMock(side_effect=error))
            with pytest.raises(RuntimeError, match=f"failed {failure}"):
                async with session.execution_lock:
                    await server._request_plan_approval(session, plan)
        assert session.agent is old
        assert session.record.interaction_mode == mode
        assert session.record.plan_pending
        assert session.record.latest_plan_markdown == plan
        assert session.approved_plan is None
        assert session.idle()
        run.assert_not_awaited()
        async with session.execution_lock:
            await server._request_plan_approval(session, plan)
        run.assert_awaited_once()
        assert conn.request_permission.await_count == 2
        assert session.record.interaction_mode == "build"
        assert not session.record.plan_pending
        assert session.approved_plan is None
        assert session.idle()
    finally:
        await server.shutdown()


@pytest.mark.parametrize("option", ["implement_plan", "implement_plan_clear"])
@pytest.mark.usefixtures("slow_setup_cleanup")
async def test_post_adoption_error_retains_approved_plan_until_cleanup_finishes(
    configured_factory: AgentFactory, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, option: str
) -> None:
    server, session, conn = await _open(configured_factory, tmp_path, monkeypatch)
    await server.set_session_mode(session.session_id, "plan")
    await asyncio.wait_for(asyncio.gather(*session.cleanup_tasks.values()), 30)
    assert session.idle()
    monkeypatch.setattr(agent_factory, "CLEANUP_TIMEOUT", 0.02)
    old = session.agent
    cleanup = old.cleanup
    plan = "## Resume the adopted plan despite a mode-update error"
    release, implemented, finish = (asyncio.Event() for _ in range(3))
    seen: list[str | None] = []

    async def delayed_cleanup() -> None:
        await release.wait()
        await cleanup()

    async def drive(current: AcpSession, bridge: Any, text: str | None, attachments: Any) -> str:
        seen.append(text)
        if current.agent is old:
            setattr(old, "_completed_plan", plan)
        else:
            implemented.set()
            await finish.wait()
        return "end_turn"

    monkeypatch.setattr(old, "cleanup", delayed_cleanup)
    monkeypatch.setattr(server, "_drive_turn", drive)
    monkeypatch.setattr(server, "_emit_current_mode", AsyncMock(side_effect=RuntimeError("mode update failed")))
    conn.request_permission = AsyncMock(
        return_value=RequestPermissionResponse.model_validate({"outcome": {"outcome": "selected", "optionId": option}})
    )
    try:
        with pytest.raises(RuntimeError, match="mode update failed"):
            await server.prompt(session.session_id, [TextContentBlock(type="text", text="Make a plan")])
        assert session.agent is not old
        assert session.record.interaction_mode == "build"
        assert session.cleanup_tasks
        assert session.approved_plan == plan
        assert session.record.plan_pending
        assert session.record.latest_plan_markdown == plan
        assert not session.idle()
        assert len(seen) == 1
        release.set()
        await asyncio.wait_for(asyncio.gather(*session.cleanup_tasks.values()), 5)
        await asyncio.wait_for(implemented.wait(), 5)
        assert len(seen) == 2
        assert seen[1] is not None and plan in seen[1]
        assert session.approved_plan is None
        assert not session.record.plan_pending
        finish.set()
        async with asyncio.timeout(5):
            async with session.execution_lock:
                pass
        assert session.idle()
        conn.request_permission.assert_awaited_once()
    finally:
        release.set()
        finish.set()
        await asyncio.gather(*session.cleanup_tasks.values(), return_exceptions=True)
        await server.shutdown()


@pytest.mark.parametrize("option", ["implement_plan", "implement_plan_clear"])
async def test_cancelled_approval_request_does_not_resume_after_cleanup(
    configured_factory: AgentFactory, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, option: str
) -> None:
    monkeypatch.setattr(agent_factory, "CLEANUP_TIMEOUT", 0.02)
    server, session, conn = await _open(configured_factory, tmp_path, monkeypatch)
    session.record.interaction_mode = "plan"
    session.record.plan_pending = True
    started, release = asyncio.Event(), asyncio.Event()
    cleanup = session.agent.cleanup
    run = AsyncMock(return_value="end_turn")
    monkeypatch.setattr(server, "_run_turn", run)
    conn.request_permission = AsyncMock(
        return_value=RequestPermissionResponse.model_validate({"outcome": {"outcome": "selected", "optionId": option}})
    )

    async def delayed_cleanup() -> None:
        started.set()
        await release.wait()
        await cleanup()

    monkeypatch.setattr(session.agent, "cleanup", delayed_cleanup)

    async def approve() -> None:
        async with session.execution_lock:
            await server._request_plan_approval(session, "## Cancel this accepted request")

    approving = asyncio.create_task(approve())
    try:
        await asyncio.wait_for(started.wait(), 5)
        approving.cancel()
        await asyncio.sleep(0)
        approving.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(approving, 5)
        assert session.cleanup_tasks
        assert session.approved_plan is None and session.record.plan_pending
        release.set()
        await asyncio.wait_for(asyncio.gather(*session.cleanup_tasks.values()), 5)
        async with session.execution_lock:
            pass
        run.assert_not_awaited()
    finally:
        release.set()
        await asyncio.gather(approving, *session.cleanup_tasks.values(), return_exceptions=True)
        await server.shutdown()


async def test_worker_reuses_cleanup_task_after_automatic_plan_rebuild(
    configured_factory: AgentFactory, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(agent_factory, "CLEANUP_TIMEOUT", 0.02, raising=False)
    factory = configured_factory
    server, session, _ = await _open(factory, tmp_path, monkeypatch)
    session.record.interaction_mode = "plan"
    session.record.plan_pending = True
    old = session.agent
    original_cleanup = old.cleanup
    release, started, persisted = asyncio.Event(), asyncio.Event(), asyncio.Event()
    actual: list[asyncio.Task[Any]] = []
    calls = 0
    errors: list[dict[str, Any]] = []
    loop = asyncio.get_running_loop()
    handler = loop.get_exception_handler()
    loop.set_exception_handler(lambda _loop, context: errors.append(context))

    async def cleanup() -> None:
        nonlocal calls
        calls += 1
        current = asyncio.current_task()
        assert current is not None
        actual.append(current)
        started.set()
        if calls == 1:
            await release.wait()
        await original_cleanup()

    async def automatic_turn(*args: Any, **kwargs: Any) -> str:
        if session.agent is old:
            await server._request_plan_approval(session, "## Implement this plan")
        return "end_turn"

    persist = factory.persist

    def record_persist(current: AcpSession) -> None:
        persist(current)
        if current.agent is not old and started.is_set():
            persisted.set()

    jobs = SimpleNamespace(
        recover=AsyncMock(),
        deliver=AsyncMock(),
        close=AsyncMock(),
        set_callbacks=lambda **kwargs: None,
        active_count=0,
        pending_count=0,
    )
    monkeypatch.setattr(old, "cleanup", cleanup)
    monkeypatch.setattr(old, "background_jobs", jobs)
    monkeypatch.setattr(old, "background_needs_turn", True)
    monkeypatch.setattr(server, "_run_turn", automatic_turn)
    monkeypatch.setattr(factory, "persist", record_persist)
    session.bridge = SimpleNamespace(send=AsyncMock())
    session.completions.append({"text": "child report"})
    session.background_wake.set()
    worker = session.background_task = asyncio.create_task(server._background_worker(session, jobs))
    try:
        await asyncio.wait_for(started.wait(), 5)
        await asyncio.wait_for(persisted.wait(), 5)
        async with asyncio.timeout(5):
            async with session.execution_lock:
                retained = session.cleanup_tasks.get(id(old))
                calls_after_sweep = calls
        release.set()
        await asyncio.wait_for(asyncio.shield(actual[0]), 5)
        await asyncio.sleep(0)
        assert calls_after_sweep == 1, f"worker invoked cleanup twice; callback errors: {errors!r}"
        assert retained is actual[0]
        assert not errors
        assert not session.cleanup_tasks and not session.retired_agents
        assert not worker.done()
    finally:
        release.set()
        worker.cancel()
        await asyncio.gather(worker, *actual, return_exceptions=True)
        loop.set_exception_handler(handler)
        session.bridge = None
        await server.shutdown()


@pytest.mark.parametrize("option", ["implement_plan", "implement_plan_clear"])
async def test_plan_approval_does_not_start_implementation_through_cleanup_fence(
    configured_factory: AgentFactory, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, option: str
) -> None:
    monkeypatch.setattr(agent_factory, "CLEANUP_TIMEOUT", 0.02, raising=False)
    server, session, conn = await _open(configured_factory, tmp_path, monkeypatch)
    plan = "## Safe implementation plan"
    session.record.interaction_mode = "plan"
    session.record.latest_plan_markdown = plan
    session.record.plan_pending = True
    conn.request_permission = AsyncMock(
        return_value=RequestPermissionResponse.model_validate({"outcome": {"outcome": "selected", "optionId": option}})
    )
    old = session.agent
    original_cleanup = old.cleanup
    started, release = asyncio.Event(), asyncio.Event()
    calls = 0

    async def cleanup() -> None:
        nonlocal calls
        calls += 1
        started.set()
        await release.wait()
        await original_cleanup()

    monkeypatch.setattr(old, "cleanup", cleanup)
    implemented = asyncio.Event()

    async def implement(*args: Any, **kwargs: Any) -> str:
        implemented.set()
        return "end_turn"

    run = AsyncMock(side_effect=implement)
    monkeypatch.setattr(server, "_run_turn", run)
    approving = asyncio.create_task(server._request_plan_approval(session, plan))
    try:
        await asyncio.wait_for(started.wait(), 5)
        done, _ = await asyncio.wait({approving}, timeout=5)
        assert approving in done, "approval still holds the request during cleanup"
        await approving
        run.assert_not_awaited()
        assert session.cleanup_tasks and session.record.plan_pending
        assert session.record.latest_plan_markdown == plan
        assert session.record.interaction_mode == "build"
        await server._request_plan_approval(session, plan)
        assert conn.request_permission.await_count == 1
        run.assert_not_awaited()
        tasks = list(session.cleanup_tasks.values())
        release.set()
        await asyncio.wait_for(asyncio.gather(*tasks), 5)
        await asyncio.wait_for(implemented.wait(), 5)
        await server._request_plan_approval(session, plan)
        run.assert_awaited_once()
        assert conn.request_permission.await_count == 1
        assert session.approved_plan is None
        assert not session.record.plan_pending
        assert calls == 1
    finally:
        release.set()
        await asyncio.gather(approving, return_exceptions=True)
        await asyncio.gather(*getattr(session, "cleanup_tasks", {}).values(), return_exceptions=True)
        await server.shutdown()


@pytest.mark.parametrize("failure", ["error", "cancelled"])
async def test_terminal_cleanup_failure_keeps_exact_task_and_resources_fenced(
    configured_factory: AgentFactory, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    cleanup_timeout = agent_factory.CLEANUP_TIMEOUT
    monkeypatch.setattr(agent_factory, "CLEANUP_TIMEOUT", 0.02)
    server, session, _ = await _open(configured_factory, tmp_path, monkeypatch)
    old = session.agent
    original_cleanup = old.cleanup
    started, release = asyncio.Event(), asyncio.Event()
    calls = 0

    async def cleanup() -> None:
        nonlocal calls
        calls += 1
        started.set()
        await release.wait()
        if failure == "error":
            raise RuntimeError("cleanup failed")
        raise asyncio.CancelledError

    monkeypatch.setattr(old, "cleanup", cleanup)
    switching = asyncio.create_task(server.set_session_mode(session.session_id, "plan"))
    try:
        await asyncio.wait_for(started.wait(), 5)
        done, _ = await asyncio.wait({switching}, timeout=5)
        assert switching in done
        await switching
        retained = session.cleanup_tasks[id(old)]
        adopted_cleanup = AsyncMock(wraps=session.agent.cleanup)
        monkeypatch.setattr(session.agent, "cleanup", adopted_cleanup)
        assert session.accounting is not None and session.control_adapter is not None
        close_accounting = AsyncMock(wraps=session.accounting.aclose)
        close_control = AsyncMock(wraps=session.control_adapter.shutdown)
        monkeypatch.setattr(session.accounting, "aclose", close_accounting)
        monkeypatch.setattr(session.control_adapter, "shutdown", close_control)
        release.set()
        await asyncio.gather(retained, return_exceptions=True)
        with pytest.raises(RuntimeError, match="ACP shutdown incomplete"):
            await server.shutdown()
        with pytest.raises(RequestError):
            await server.close_session(session.session_id)
        assert AgentFactory.start_retired_cleanup(session, old) is retained
        assert session.cleanup_tasks[id(old)] is retained
        assert any(agent is old for agent in session.retired_agents)
        assert session.closed and not session.idle()
        assert configured_factory._acp_sessions[session.session_id] is session
        adopted_cleanup.assert_not_awaited()
        close_accounting.assert_not_awaited()
        close_control.assert_not_awaited()
        assert calls == 1
    finally:
        monkeypatch.setattr(agent_factory, "CLEANUP_TIMEOUT", cleanup_timeout)
        release.set()
        await asyncio.gather(switching, *session.cleanup_tasks.values(), return_exceptions=True)
        await original_cleanup()
        session.cleanup_tasks.clear()
        session.retired_agents.clear()
        session.close_task = None
        await server.close_session(session.session_id)


@pytest.mark.parametrize("option", ["implement_plan", "implement_plan_clear"])
@pytest.mark.parametrize("interrupt", [None, "cancel", "close"])
@pytest.mark.usefixtures("slow_setup_cleanup")
async def test_public_approved_plan_resumes_once_unless_cancelled_or_closed(
    configured_factory: AgentFactory,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    option: str,
    interrupt: str | None,
) -> None:
    server, session, conn = await _open(configured_factory, tmp_path, monkeypatch)
    await server.set_session_mode(session.session_id, "plan")
    await asyncio.wait_for(asyncio.gather(*session.cleanup_tasks.values()), 30)
    assert session.idle()
    monkeypatch.setattr(agent_factory, "CLEANUP_TIMEOUT", 0.02)
    old = session.agent
    original_cleanup = old.cleanup
    plan = "## UNIQUE APPROVED PLAN: edit important.txt to add validation"
    release, implemented, finish_turn = (asyncio.Event() for _ in range(3))
    seen: list[tuple[str | None, str]] = []
    calls = 0

    async def cleanup() -> None:
        nonlocal calls
        calls += 1
        await release.wait()
        await original_cleanup()

    async def drive(current: AcpSession, bridge: Any, text: str | None, attachments: Any) -> str:
        seen.append((text, type(current.agent).__name__))
        current.agent.append_user_message([TextBlock(text=text or "")])
        if current.agent is old:
            current.agent.history.append(Message(role="assistant", content=[TextBlock(text=plan)]))
            setattr(old, "_completed_plan", plan)
        else:
            implemented.set()
            await finish_turn.wait()
        return "end_turn"

    construct = AsyncMock(wraps=configured_factory._construct_agent)
    monkeypatch.setattr(configured_factory, "_construct_agent", construct)
    monkeypatch.setattr(old, "cleanup", cleanup)
    monkeypatch.setattr(server, "_drive_turn", drive)
    conn.request_permission = AsyncMock(
        return_value=RequestPermissionResponse.model_validate({"outcome": {"outcome": "selected", "optionId": option}})
    )
    try:
        initial = await server.prompt(session.session_id, [TextContentBlock(type="text", text="Make a plan")])
        assert initial.stop_reason == "end_turn"
        assert session.approved_plan == plan and session.record.plan_pending
        assert session.record.latest_plan_markdown == plan
        assert session.record.interaction_mode == "build"
        if option.endswith("_clear"):
            assert "UNIQUE APPROVED PLAN" not in str(session.agent.dump_message_history())
        with pytest.raises(RequestError):
            await server.prompt(session.session_id, [TextContentBlock(type="text", text="retry")])
        await server._request_plan_approval(session, plan)
        conn.request_permission.assert_awaited_once()
        assert len(seen) == 1
        if interrupt == "cancel":
            await server.cancel(session.session_id)
        elif interrupt == "close":
            with pytest.raises(RequestError):
                await server.close_session(session.session_id)
        tasks = tuple(session.cleanup_tasks.values())
        assert tasks
        release.set()
        await asyncio.wait_for(asyncio.gather(*tasks), 5)
        if interrupt is None:
            await asyncio.wait_for(implemented.wait(), 5)
            assert len(seen) == 2
            assert seen[1][1] == "CoderAgent"
            assert seen[1][0] is not None and plan in seen[1][0]
            assert not session.record.plan_pending and session.approved_plan is None
            await server._request_plan_approval(session, plan)
            await server.load_session(str(tmp_path), session.session_id)
            with pytest.raises(RequestError):
                await server.prompt(session.session_id, [TextContentBlock(type="text", text="duplicate")])
        finish_turn.set()
        async with asyncio.timeout(5):
            async with session.execution_lock:
                pass
        assert len(seen) == (2 if interrupt is None else 1)
        assert calls == 1
        construct.assert_awaited_once()
        conn.request_permission.assert_awaited_once()
        if interrupt is not None:
            assert session.approved_plan is None
            assert session.record.plan_pending
        if interrupt == "close":
            assert session.close_task is not None
            await asyncio.wait_for(asyncio.shield(session.close_task), 5)
            assert session.session_id not in server._sessions
    finally:
        release.set()
        finish_turn.set()
        await asyncio.gather(*session.cleanup_tasks.values(), return_exceptions=True)
        await server.shutdown()


@pytest.mark.parametrize("interrupt", ["cancel", "close"])
async def test_cancel_and_close_fence_late_plan_approval(
    configured_factory: AgentFactory, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, interrupt: str
) -> None:
    server, session, conn = await _open(configured_factory, tmp_path, monkeypatch)
    session.record.interaction_mode = "plan"
    session.record.plan_pending = True
    entered, cancelled, release = (asyncio.Event() for _ in range(3))
    run = AsyncMock(return_value="end_turn")
    switch = AsyncMock(wraps=configured_factory.set_interaction_mode)

    async def late_permission(*args: Any, **kwargs: Any) -> RequestPermissionResponse:
        entered.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            cancelled.set()
            await release.wait()
        return RequestPermissionResponse.model_validate(
            {"outcome": {"outcome": "selected", "optionId": "implement_plan_clear"}}
        )

    conn.request_permission = late_permission
    monkeypatch.setattr(server, "_run_turn", run)
    monkeypatch.setattr(configured_factory, "set_interaction_mode", switch)

    async def approve() -> None:
        async with session.execution_lock:
            await server._request_plan_approval(session, "## Do not implement after cancellation")

    approving = asyncio.create_task(approve())
    closing: asyncio.Task[None] | None = None
    try:
        await asyncio.wait_for(entered.wait(), 5)
        if interrupt == "cancel":
            await server.cancel(session.session_id)
        else:
            closing = asyncio.create_task(server.close_session(session.session_id))
        await asyncio.wait_for(cancelled.wait(), 5)
        release.set()
        await asyncio.wait_for(approving, 5)
        if closing is not None:
            await asyncio.wait_for(closing, 5)
        switch.assert_not_awaited()
        run.assert_not_awaited()
        assert session.approved_plan is None
        assert session.record.plan_pending
        assert session.record.interaction_mode == "plan"
    finally:
        release.set()
        await asyncio.gather(approving, *([closing] if closing is not None else []), return_exceptions=True)
        await server.shutdown()


async def test_cancel_during_plan_finalization_does_not_offer_approval(
    configured_factory: AgentFactory, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    server, session, conn = await _open(configured_factory, tmp_path, monkeypatch)
    await server.set_session_mode(session.session_id, "plan")
    server._start_session_tasks(session)
    finalizing, release = asyncio.Event(), asyncio.Event()
    plan = "## Retain this plan without implementing after cancellation"
    conn.request_permission = AsyncMock()
    switch = AsyncMock(wraps=configured_factory.set_interaction_mode)

    async def drive(*args: Any, **kwargs: Any) -> str:
        setattr(session.agent, "_completed_plan", plan)
        return "end_turn"

    async def emit_usage(*args: Any, **kwargs: Any) -> None:
        finalizing.set()
        await release.wait()

    monkeypatch.setattr(server, "_drive_turn", drive)
    monkeypatch.setattr(session.bridge, "emit_usage", emit_usage)
    monkeypatch.setattr(configured_factory, "set_interaction_mode", switch)
    prompting = asyncio.create_task(
        server.prompt(session.session_id, [TextContentBlock(type="text", text="Make a plan")])
    )
    try:
        await asyncio.wait_for(finalizing.wait(), 5)
        await server.cancel(session.session_id)
        release.set()
        await asyncio.wait_for(prompting, 5)
        assert session.record.latest_plan_markdown == plan and session.record.plan_pending
        assert session.approved_plan is None
        conn.request_permission.assert_not_awaited()
        switch.assert_not_awaited()
    finally:
        release.set()
        await asyncio.gather(prompting, return_exceptions=True)
        await server.shutdown()
