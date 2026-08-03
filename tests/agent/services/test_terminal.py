import asyncio
import contextlib
import os
import signal
import sys
from unittest.mock import patch

import pytest

from kolega_code.events import AgentConnectionManager
from kolega_code.services.terminal import (
    DetachedSession,
    LocalTerminalManager,
    PtySession,
    _signal_group,
    _strip_runtime_venv,
)


def _process_alive(pid: int) -> bool:
    """True if ``pid`` is a live, non-zombie process.

    ``os.kill(pid, 0)`` reports a zombie as alive, so reap any of our own
    finished children first (in the real deployment init reaps survivors after
    kolega-code exits; in-process we must do it to tell dead from zombie).
    """
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    try:
        reaped, _ = os.waitpid(pid, os.WNOHANG)
        if reaped == pid:
            return False
    except ChildProcessError:
        pass
    return True


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
    explicit_workdir = tmp_path / "subdir"
    explicit_workdir.mkdir()
    result = await manager.exec_command("pwd", workdir=str(explicit_workdir / ".." / "subdir"), yield_time_ms=3000)
    assert result.status == "exited"
    command_event = next(event for event in manager.connection_manager.events if event.event_type == "terminal_command")
    assert command_event.content == {
        "command": "pwd",
        "terminal_id": "s_1",
        "session_id": "s_1",
        "workdir": str(explicit_workdir.resolve()),
    }
    assert result.output.strip() == str(explicit_workdir.resolve())


@pytest.mark.asyncio
async def test_constructed_default_workdir_is_used_when_workdir_omitted(tmp_path):
    default_workdir = tmp_path / "default"
    default_workdir.mkdir()
    connection_manager = _RecordingConnectionManager()
    manager = LocalTerminalManager(
        "workspace",
        "thread",
        connection_manager,
        default_workdir=default_workdir / ".." / "default",
    )
    result = await manager.exec_command("pwd", yield_time_ms=3000)
    assert result.status == "exited"
    command_event = next(event for event in connection_manager.events if event.event_type == "terminal_command")
    assert command_event.content == {
        "command": "pwd",
        "terminal_id": "s_1",
        "session_id": "s_1",
        "workdir": str(default_workdir.resolve()),
    }
    # Commands must run in the configured project root, not the process cwd.
    assert result.output.strip() == str(default_workdir.resolve())


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
async def test_warns_when_command_leaves_background_jobs(manager):
    # Backgrounded (`&`) processes die with the shell that started them; the
    # result must say so loudly instead of letting the agent believe a server
    # it launched is still listening, and must point at the supported recipe
    # (exec_command background=true, which is detached and survives). The
    # warning must NOT teach raw detach recipes (nohup/disown) — durability is
    # the terminal manager's job, not the model's.
    result = await manager.exec_command("sleep 30 & echo main-done", yield_time_ms=8000)
    assert result.status == "exited"
    assert "main-done" in result.output
    assert "[kolega-code] WARNING: background process(es)" in result.output
    assert "exec_command background=true" in result.output
    assert "nohup" not in result.output
    assert "disown" not in result.output


# --------------------------------------------------------------------------- #
# background=true durability: detached sessions outlive the kolega-code process
# (matching claude-code's run_in_background), while foreground commands do not.
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_background_launch_is_detached_and_running(manager):
    result = await manager.exec_command("sleep 30", yield_time_ms=500, background=True)
    assert result.status == "running"
    assert result.session_id is not None
    session = manager.sessions[result.session_id]
    assert isinstance(session, DetachedSession)
    assert session.detached is True
    await manager.kill_session(result.session_id, "TERM")


@pytest.mark.asyncio
async def test_background_survives_cleanup_all(manager, tmp_path):
    # A heartbeat "server" that appends a line every 100ms; cleanup_all() stands
    # in for kolega-code tearing down at the end of a run.
    marker = tmp_path / "beats.log"
    cmd = f"while true; do echo beat >> {marker}; sleep 0.1; done"
    result = await manager.exec_command(cmd, yield_time_ms=800, background=True)
    pid = manager.sessions[result.session_id].pid
    assert result.status == "running" and pid is not None

    await asyncio.sleep(0.2)
    before = marker.read_text().count("beat")

    await manager.cleanup_all()  # == kolega-code exit
    assert len(manager.sessions) == 0

    await asyncio.sleep(0.4)
    after = marker.read_text().count("beat")
    try:
        assert _process_alive(pid), "detached background process was reaped on cleanup_all"
        assert after > before, "detached background process stopped writing after cleanup_all"
    finally:
        # It really is a survivor, so we must clean it up ourselves.
        with contextlib.suppress(OSError):
            os.killpg(os.getpgid(pid), signal.SIGKILL)


@pytest.mark.asyncio
async def test_background_stopped_by_kill_session(manager):
    result = await manager.exec_command("sleep 30", yield_time_ms=500, background=True)
    pid = manager.sessions[result.session_id].pid
    killed = await manager.kill_session(result.session_id, "TERM")
    assert killed.status == "exited"
    await asyncio.sleep(0.1)
    assert not _process_alive(pid)
    assert result.session_id not in manager.sessions


@pytest.mark.asyncio
async def test_background_poll_captures_incremental_output(manager):
    # Initial call captures early output; a later poll captures new output while
    # the process keeps running (write_stdin clamps its poll window to >=5s).
    result = await manager.exec_command(
        "echo first; sleep 1; echo second; sleep 30", yield_time_ms=600, background=True
    )
    assert result.status == "running"
    assert "first" in result.output
    poll = await manager.write_stdin(result.session_id, "")
    assert poll.status == "running"
    assert "second" in poll.output
    await manager.kill_session(result.session_id, "TERM")


@pytest.mark.asyncio
async def test_foreground_long_runner_is_killed_by_cleanup_all(manager):
    # Contrast with the background case: a normal command must NOT survive.
    result = await manager.exec_command("sleep 300", yield_time_ms=500)
    pid = manager.sessions[result.session_id].pid
    assert result.status == "running"
    await manager.cleanup_all()
    await asyncio.sleep(0.2)
    assert not _process_alive(pid)


def test_signal_group_none_pid_is_noop():
    # Defensive: signalling before a process exists must not raise.
    _signal_group(None, signal.SIGTERM)


@pytest.mark.asyncio
async def test_no_background_warning_for_quiet_commands(manager):
    result = await manager.exec_command("echo plain-done", yield_time_ms=5000)
    assert result.status == "exited"
    assert "WARNING" not in result.output


@pytest.mark.asyncio
async def test_no_background_warning_when_jobs_finished(manager):
    result = await manager.exec_command("sleep 0.2 & wait; echo waited-done", yield_time_ms=5000)
    assert result.status == "exited"
    assert "waited-done" in result.output
    assert "WARNING" not in result.output


@pytest.mark.asyncio
async def test_no_background_warning_when_foreground_session_killed(manager):
    # kill_command on a foreground process leaves a stale job-table entry when
    # the EXIT trap fires; the liveness filter must keep that from warning.
    result = await manager.exec_command("sleep 30", yield_time_ms=300)
    assert result.status == "running"
    killed = await manager.kill_session(result.session_id, "TERM")
    assert killed.status == "exited"
    assert "WARNING" not in killed.output


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
