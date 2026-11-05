from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import cast
from unittest.mock import AsyncMock

import pytest
from acp.interfaces import Client

from kolega_code.acp.agent_factory import AgentFactory
from kolega_code.acp.server import AcpAgent
from kolega_code.acp.session import AcpSession
from tests.acp.test_cleanup_lifecycle import configured_factory as configured_factory
from tests.acp.test_server import _FakeConn

pytestmark = pytest.mark.usefixtures("isolated_cli_env")


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_waiter", [False, True])
@pytest.mark.parametrize("cancelled_close", [False, True])
async def test_shutdown_drains_all_healthy_sessions_before_reporting_all_failures(
    configured_factory: AgentFactory,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cancel_waiter: bool,
    cancelled_close: bool,
) -> None:
    factory = configured_factory
    server = AcpAgent(factory)
    server.on_connect(cast(Client, _FakeConn()))
    monkeypatch.setattr(server, "_schedule_open_updates", lambda _sid: None)
    sessions: list[AcpSession] = []
    for _ in range(4):
        opened = await server.new_session(str(tmp_path))
        sessions.append(server._sessions[opened.session_id])
    healthy = sessions[::2]
    failed = sessions[1::2]
    failures = {
        failed[0].session_id: RuntimeError("first close failed"),
        failed[1].session_id: asyncio.CancelledError("second close cancelled")
        if cancelled_close
        else ValueError("second close failed"),
    }
    started = {session.session_id: asyncio.Event() for session in sessions}
    release = asyncio.Event()
    close = factory.close_session
    accounting: dict[str, AsyncMock] = {}
    for session in sessions:
        assert session.accounting is not None
        accounting[session.session_id] = AsyncMock(wraps=session.accounting.aclose)
        monkeypatch.setattr(session.accounting, "aclose", accounting[session.session_id])

    async def controlled_close(session_id: str) -> None:
        started[session_id].set()
        if session_id in failures:
            raise failures[session_id]
        await release.wait()
        await close(session_id)

    closing = AsyncMock(side_effect=controlled_close)
    monkeypatch.setattr(factory, "close_session", closing)
    shutting = asyncio.create_task(server.shutdown())
    try:
        async with asyncio.timeout(30):
            await asyncio.gather(*(event.wait() for event in started.values()))
            failed_tasks: list[asyncio.Task[None]] = []
            for session in failed:
                assert session.close_task is not None
                failed_tasks.append(session.close_task)
            await asyncio.gather(*failed_tasks, return_exceptions=True)
            assert not shutting.done()
            assert server._shutdown_task is not None
            shutdown_task = server._shutdown_task
            if cancel_waiter:
                shutting.cancel()
            assert all(session.closed for session in sessions)
            for session in healthy:
                assert session.close_task is not None and not session.close_task.cancelled()
            release.set()
            if cancel_waiter:
                with pytest.raises(asyncio.CancelledError):
                    await shutting
            else:
                with pytest.raises(RuntimeError, match="ACP shutdown incomplete"):
                    await shutting
            for session in healthy:
                assert session.session_id not in server._sessions
                assert session.session_id not in factory._acp_sessions
                assert session.close_task is not None
                assert session.close_task.done() and not session.close_task.cancelled()
                accounting[session.session_id].assert_awaited_once()
            for session in failed:
                assert server._sessions[session.session_id] is session
                assert factory._acp_sessions[session.session_id] is session
                accounting[session.session_id].assert_not_awaited()
            with pytest.raises(RuntimeError, match="ACP shutdown incomplete") as caught:
                await server.shutdown()
            cause = caught.value.__cause__
            assert isinstance(cause, BaseExceptionGroup)
            assert cause.exceptions[0] is failures[failed[0].session_id]
            if cancelled_close:
                assert isinstance(cause.exceptions[1], asyncio.CancelledError)
            else:
                assert cause.exceptions[1] is failures[failed[1].session_id]
            assert server._shutdown_task is shutdown_task
            assert closing.await_count == len(sessions)
    finally:
        release.set()
        await asyncio.gather(shutting, return_exceptions=True)
        for session in failed:
            await close(session.session_id)


@pytest.mark.skipif(sys.platform == "win32", reason="Requires POSIX PTYs")
def test_real_stdio_eof_fails_with_retained_agent_pty_and_accounting(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    bootstrap = tmp_path / "bootstrap"
    bootstrap.mkdir()
    shutil.copyfile(Path(__file__).with_name("_accounting_subprocess_bootstrap.py"), bootstrap / "sitecustomize.py")
    environment = {key: value for key, value in os.environ.items() if key in ("PATH", "SYSTEMROOT", "LANG")}
    environment.update(
        HOME=str(tmp_path),
        KOLEGA_CODE_STATE_DIR=str(tmp_path / "state"),
        KOLEGA_CODE_LSP="off",
        KOLEGA_CODE_PROVIDER="anthropic",
        KOLEGA_CODE_MODEL="claude-haiku-4-5-20251001",
        ANTHROPIC_API_KEY="fake-shutdown-provider-key",
        PYTHONPATH=os.pathsep.join((str(bootstrap), str(root))),
        CI="1",
        PYTHON_DOTENV_DISABLED="1",
    )
    result = subprocess.run(
        [sys.executable, "-m", "tests.acp._shutdown_failure_process", str(tmp_path)],
        input="",
        capture_output=True,
        text=True,
        cwd=root,
        env=environment,
        timeout=90,
        check=False,
    )
    assert result.returncode != 0, result.stderr
    assert "ACP shutdown incomplete" in result.stderr
    assert json.loads(result.stdout) == {
        "transport_eof": True,
        "pty_alive": True,
        "adopted_cleanup_calls": 0,
        "accounting_close_calls": 0,
        "server_retained": True,
        "factory_retained": True,
        "healthy_drained": True,
    }
