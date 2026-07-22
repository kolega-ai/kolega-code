import asyncio
import os
import sys
from unittest.mock import patch

import pytest

from kolega_code.events import AgentConnectionManager
from kolega_code.services.terminal import LocalTerminalManager, PtySession, _strip_runtime_venv


class _RecordingConnectionManager(AgentConnectionManager):
    def __init__(self):
        self.events = []

    async def broadcast_event(self, event, workspace_id, thread_id):
        self.events.append(event)

    async def connect(self, websocket, workspace_id, thread_id, connection_type, user_info=None) -> None:
        return None

    def disconnect(self, websocket, workspace_id, thread_id, connection_type) -> None:
        return None

    def get_connection_count(self, workspace_id, thread_id) -> dict:
        return {}


@pytest.fixture
def manager():
    # A no-op connection manager: output broadcasting is recorded but unused in tests.
    return LocalTerminalManager("workspace", "thread", _RecordingConnectionManager())


@pytest.mark.asyncio
async def test_exec_command_success(manager):
    result = await manager.exec_command("echo hello world", yield_time_ms=5000)
    assert result.status == "exited"
    assert result.exit_code == 0
    assert "hello world" in result.output
    assert result.session_id is None


@pytest.mark.asyncio
async def test_exec_command_nonzero_exit_code(manager):
    result = await manager.exec_command("exit 7", yield_time_ms=5000)
    assert result.status == "exited"
    assert result.exit_code == 7


@pytest.mark.asyncio
async def test_exec_command_failing_command_is_nonzero(manager):
    result = await manager.exec_command("ls /this_path_does_not_exist_xyz", yield_time_ms=5000)
    assert result.status == "exited"
    assert result.exit_code != 0


@pytest.mark.asyncio
async def test_long_running_returns_session_then_completes(manager):
    result = await manager.exec_command("echo start; sleep 1; echo done", yield_time_ms=250)
    assert result.status == "running"
    assert result.session_id is not None
    session_id = result.session_id

    for _ in range(40):
        result = await manager.write_stdin(session_id, "", yield_time_ms=2000)
        if result.status == "exited":
            break
    assert result.status == "exited"
    assert result.exit_code == 0
    assert "done" in result.output


@pytest.mark.asyncio
async def test_interactive_stdin(manager):
    result = await manager.exec_command('printf "P> "; read x; echo got=$x', yield_time_ms=400)
    assert result.status == "running"
    assert "P>" in result.output

    result = await manager.write_stdin(result.session_id, "ada\n", yield_time_ms=3000)
    assert result.status == "exited"
    assert "got=ada" in result.output


@pytest.mark.asyncio
async def test_kill_session_interrupt_reports_130(manager):
    result = await manager.exec_command("sleep 30", yield_time_ms=300)
    assert result.status == "running"
    killed = await manager.kill_session(result.session_id, "INT")
    assert killed.status == "exited"
    assert killed.exit_code == 130


@pytest.mark.asyncio
async def test_kill_session_term(manager):
    result = await manager.exec_command("sleep 30", yield_time_ms=300)
    killed = await manager.kill_session(result.session_id, "TERM")
    assert killed.status == "exited"
    # SIGTERM -> 143, or SIGKILL fallback -> 137
    assert killed.exit_code in (143, 137)


@pytest.mark.asyncio
async def test_list_sessions_tracks_running_and_clears(manager):
    result = await manager.exec_command("sleep 5", yield_time_ms=200)
    sessions = await manager.list_sessions()
    assert result.session_id in sessions
    assert sessions[result.session_id]["running"] is True

    await manager.kill_session(result.session_id, "TERM")
    assert result.session_id not in await manager.list_sessions()


@pytest.mark.asyncio
async def test_write_stdin_unknown_session_raises(manager):
    with pytest.raises(KeyError):
        await manager.write_stdin("does_not_exist")


@pytest.mark.asyncio
async def test_kill_unknown_session_raises(manager):
    with pytest.raises(KeyError):
        await manager.kill_session("does_not_exist")


@pytest.mark.asyncio
async def test_run_command_convenience_accumulates_output(manager):
    output = await manager.run_command("echo a; echo b; echo c")
    assert "a" in output and "b" in output and "c" in output


@pytest.mark.asyncio
async def test_workdir_is_respected(manager, tmp_path):
    result = await manager.exec_command("pwd", workdir=str(tmp_path), yield_time_ms=3000)
    assert result.status == "exited"
    # macOS resolves symlinks (/var -> /private/var); the leaf dir is enough.
    assert result.output.strip().endswith(tmp_path.name)


@pytest.mark.asyncio
async def test_constructed_default_workdir_is_used_when_workdir_omitted(tmp_path):
    manager = LocalTerminalManager("workspace", "thread", _RecordingConnectionManager(), default_workdir=tmp_path)
    result = await manager.exec_command("pwd", yield_time_ms=3000)
    assert result.status == "exited"
    # Commands must run in the configured project root, not the process cwd.
    assert result.output.strip().endswith(tmp_path.name)


@pytest.mark.asyncio
async def test_no_cwd_persistence_between_calls(manager, tmp_path):
    sub = tmp_path / "sub"
    sub.mkdir()
    # cd in one call must NOT affect the next (fresh process per exec).
    await manager.exec_command(f"cd {sub}", workdir=str(tmp_path), yield_time_ms=3000)
    result = await manager.exec_command("pwd", workdir=str(tmp_path), yield_time_ms=3000)
    assert result.output.strip().endswith(tmp_path.name)


def test_strip_runtime_venv_removes_own_venv():
    env = {
        "VIRTUAL_ENV": "/opt/app/.venv",
        "VIRTUAL_ENV_PROMPT": "app",
        "UV": "/usr/local/bin/uv",
        "UV_RUN_RECURSION_DEPTH": "1",
        "PATH": "/opt/app/.venv/bin:/usr/local/bin:/usr/bin",
        "HOME": "/Users/u",
    }
    out = _strip_runtime_venv(env, runtime_prefix="/opt/app/.venv")
    for var in ("VIRTUAL_ENV", "VIRTUAL_ENV_PROMPT", "UV", "UV_RUN_RECURSION_DEPTH"):
        assert var not in out
    assert out["PATH"] == "/usr/local/bin:/usr/bin"
    assert out["HOME"] == "/Users/u"


def test_strip_runtime_venv_keeps_user_activated_venv():
    env = {
        "VIRTUAL_ENV": "/Users/u/venvs/proj",
        "PATH": "/Users/u/venvs/proj/bin:/usr/bin",
        "UV": "/usr/local/bin/uv",
    }
    out = _strip_runtime_venv(dict(env), runtime_prefix="/opt/app/.venv")
    assert out == env


def test_strip_runtime_venv_without_active_venv_is_untouched():
    env = {"PATH": "/usr/local/bin:/usr/bin", "HOME": "/Users/u"}
    assert _strip_runtime_venv(dict(env), runtime_prefix="/opt/app/.venv") == env


@pytest.mark.asyncio
async def test_child_shell_does_not_inherit_runtime_venv(manager, tmp_path):
    # Simulate launching from the checkout via `uv run`: the app's own venv is
    # exported and prepended to PATH. The child shell must see neither.
    polluted = {"VIRTUAL_ENV": sys.prefix, "PATH": f"{sys.prefix}/bin{os.pathsep}" + os.environ.get("PATH", "")}
    with patch.dict(os.environ, polluted):
        # workdir without a .venv so auto-activation doesn't re-set VIRTUAL_ENV
        result = await manager.exec_command(
            'echo "V=${VIRTUAL_ENV:-unset}"; echo "P=$PATH"', workdir=str(tmp_path), yield_time_ms=5000
        )
    assert result.status == "exited"
    assert "V=unset" in result.output
    path_line = next(line for line in result.output.splitlines() if line.startswith("P="))
    assert f"{sys.prefix}/bin" not in path_line


@pytest.mark.asyncio
async def test_clean_env_overlay(manager):
    result = await manager.exec_command("echo $NO_COLOR-$TERM-$PAGER", yield_time_ms=3000)
    assert "1-dumb-cat" in result.output


@pytest.mark.asyncio
async def test_close_all_terminates_sessions(manager):
    await manager.exec_command("sleep 30", yield_time_ms=200)
    await manager.exec_command("sleep 30", yield_time_ms=200)
    assert len(manager.sessions) == 2
    await manager.close_all()
    assert len(manager.sessions) == 0


@pytest.mark.asyncio
async def test_pty_display_broadcast_decodes_split_utf8_without_replacement(tmp_path):
    connection = _RecordingConnectionManager()
    session = PtySession("s_test", "echo", str(tmp_path), connection, "workspace", "thread")
    session._broadcast_task = asyncio.create_task(session._broadcast_worker())

    session._broadcast("€".encode("utf-8")[:1])
    session._broadcast("€".encode("utf-8")[1:])
    await session._broadcast_queue.join()
    session._broadcast_queue.put_nowait(None)
    await session._broadcast_task

    display = "".join(event.content.get("display_output", "") for event in connection.events)
    assert display == "€"
    assert "�" not in display


@pytest.mark.asyncio
async def test_pty_display_broadcast_preserves_chunk_order(tmp_path):
    connection = _RecordingConnectionManager()
    session = PtySession("s_test", "echo", str(tmp_path), connection, "workspace", "thread")
    session._broadcast_task = asyncio.create_task(session._broadcast_worker())

    for chunk in (b"a", b"b", b"c"):
        session._broadcast(chunk)
    await session._broadcast_queue.join()
    session._broadcast_queue.put_nowait(None)
    await session._broadcast_task

    assert "".join(event.content.get("display_output", "") for event in connection.events) == "abc"
