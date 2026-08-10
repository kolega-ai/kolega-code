import asyncio
import contextlib
import os
import shlex
import signal
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from kolega_code.events import AgentConnectionManager
from kolega_code.services.base import TerminalCommandCancelled
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


def _marker_command(marker: Path, final_output: str) -> str:
    script = "\n".join(
        [
            "import pathlib, time",
            f"marker = pathlib.Path({str(marker)!r})",
            "print('started', flush=True)",
            "while not marker.exists():",
            "    time.sleep(0.01)",
            f"print({final_output!r}, flush=True)",
        ]
    )
    return f"{shlex.quote(sys.executable)} -c {shlex.quote(script)}"


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
async def test_list_sessions_hides_naturally_exited_pty_until_final_poll(manager, tmp_path):
    marker = tmp_path / "finish-pty"
    result = await manager.exec_command(
        _marker_command(marker, "pty-final"),
        workdir=str(tmp_path),
        yield_time_ms=250,
    )
    assert result.status == "running"
    session_id = result.session_id
    assert session_id is not None
    session = manager.sessions[session_id]
    assert isinstance(session, PtySession)

    marker.touch()
    await asyncio.wait_for(session.exited.wait(), timeout=3)

    assert session_id not in await manager.list_sessions()
    assert manager.sessions[session_id] is session
    assert session.master_fd is None

    final = await manager.write_stdin(session_id, "", yield_time_ms=250)
    assert final.status == "exited"
    assert final.exit_code == 0
    assert "pty-final" in final.output
    assert session_id not in manager.sessions


@pytest.mark.asyncio
async def test_list_sessions_hides_naturally_exited_detached_session_and_closes_raw_resources(manager, tmp_path):
    marker = tmp_path / "finish-detached"
    result = await manager.exec_command(
        _marker_command(marker, "detached-final"),
        workdir=str(tmp_path),
        yield_time_ms=250,
        background=True,
    )
    assert result.status == "running"
    session_id = result.session_id
    assert session_id is not None
    session = manager.sessions[session_id]
    assert isinstance(session, DetachedSession)
    assert session.log_path is not None
    assert session.stdin_path is not None
    log_path = Path(session.log_path)
    stdin_path = Path(session.stdin_path)

    marker.touch()
    await asyncio.wait_for(session.exited.wait(), timeout=3)
    assert log_path.exists()
    assert stdin_path.exists()

    assert session_id not in await manager.list_sessions()
    assert manager.sessions[session_id] is session
    assert session.log_path is None
    assert session.stdin_path is None
    assert not log_path.exists()
    assert not stdin_path.exists()

    final = await manager.kill_session(session_id, "TERM")
    assert final.status == "exited"
    assert final.exit_code == 0
    assert "detached-final" in final.output
    assert session_id not in manager.sessions


@pytest.mark.asyncio
async def test_list_sessions_reports_only_live_sessions_when_completed_results_are_retained(manager, tmp_path):
    live = await manager.exec_command("sleep 30", workdir=str(tmp_path), yield_time_ms=250)
    marker = tmp_path / "finish-mixed"
    completed = await manager.exec_command(
        _marker_command(marker, "mixed-final"),
        workdir=str(tmp_path),
        yield_time_ms=250,
    )
    assert live.session_id is not None
    assert completed.session_id is not None

    try:
        completed_session = manager.sessions[completed.session_id]
        marker.touch()
        await asyncio.wait_for(completed_session.exited.wait(), timeout=3)

        sessions = await manager.list_sessions()
        assert set(sessions) == {live.session_id}
        assert sessions[live.session_id]["running"] is True
        assert completed.session_id in manager.sessions

        final = await manager.write_stdin(completed.session_id, "", yield_time_ms=250)
        assert final.status == "exited"
        assert "mixed-final" in final.output
    finally:
        if live.session_id in manager.sessions:
            await manager.kill_session(live.session_id, "TERM")


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
async def test_run_command_timeout_keeps_uncapped_final_delta(manager):
    script = "import sys,time;sys.stdout.write('BEGIN-' + 'x' * 60000 + '-END');sys.stdout.flush();time.sleep(30)"
    command = f"{shlex.quote(sys.executable)} -c {shlex.quote(script)}"

    output = await manager.run_command(command, timeout=1)

    assert output.startswith("BEGIN-")
    assert "-END" in output
    assert len(output) > 60_000


@pytest.mark.asyncio
async def test_exec_command_spills_complete_output_to_session_path(manager, tmp_path):
    spill_root = tmp_path / "terminal-output"
    manager.configure_output_spill(spill_root)
    script = (
        "import sys;"
        "sys.stdout.write('BEGIN\\n' + ''.join(f'line-{i:04d}-' + 'x' * 90 + '\\n' "
        "for i in range(900)) + 'END\\n')"
    )
    command = f"{shlex.quote(sys.executable)} -c {shlex.quote(script)}"

    result = await manager.exec_command(
        command,
        yield_time_ms=10_000,
        max_output_tokens=1_000_000,
    )

    assert result.status == "exited"
    assert result.spill_path is not None
    spill_path = os.path.realpath(result.spill_path)
    assert os.path.commonpath([spill_path, str(spill_root.resolve())]) == str(spill_root.resolve())
    complete_bytes = Path(spill_path).read_bytes()
    complete = complete_bytes.decode("utf-8")
    assert complete.startswith("BEGIN")
    assert "line-0450-" in complete
    assert complete.rstrip().endswith("END")
    assert "line-0450-" not in result.output
    assert len(result.output) <= 40_000
    assert result.spill_bytes == len(complete_bytes)


@pytest.mark.asyncio
async def test_cancelled_exec_returns_recoverable_spill_result(manager, tmp_path):
    spill_root = tmp_path / "terminal-output"
    manager.configure_output_spill(spill_root)
    script = (
        "import sys,time;sys.stdout.write('BEGIN\\n' + 'x' * 70000 + '\\nEND\\n');sys.stdout.flush();time.sleep(30)"
    )
    command = f"{shlex.quote(sys.executable)} -c {shlex.quote(script)}"

    task = asyncio.create_task(
        manager.exec_command(command, yield_time_ms=30_000),
    )
    for _ in range(200):
        if list(spill_root.glob("*.log")):
            break
        await asyncio.sleep(0.01)
    else:
        pytest.fail("terminal output did not spill before cancellation")

    task.cancel()
    with pytest.raises(TerminalCommandCancelled) as cancelled:
        await task

    result = cancelled.value.result
    assert result.status == "cancelled"
    assert result.exit_code in (143, 137)
    assert result.spill_path is not None
    complete_bytes = Path(result.spill_path).read_bytes()
    complete = complete_bytes.decode("utf-8")
    lines = complete.splitlines()
    assert lines[0] == "BEGIN"
    assert "END" in lines
    assert result.spill_bytes == len(complete_bytes)
    assert await manager.list_sessions() == {}


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
async def test_exec_command_session_env_override(manager):
    # Values are shell-quoted, so spaces survive.
    result = await manager.exec_command(
        'echo "SP=$KOLEGA_SCRATCHPAD"', env={"KOLEGA_SCRATCHPAD": "/tmp/sp dir"}, yield_time_ms=5000
    )
    assert result.status == "exited"
    assert result.exit_code == 0
    assert "SP=/tmp/sp dir" in result.output


@pytest.mark.asyncio
@pytest.mark.parametrize("background", [False, True])
async def test_session_env_override_survives_login_profile(manager, tmp_path, background):
    # A login shell sources profile files before the command string runs, and a
    # profile may reset or unset session overrides. The in-command re-export
    # runs after any profile, so the override must win. Covers the PTY and
    # detached session paths.
    home = tmp_path / "home"
    home.mkdir()
    hostile = "export PROFILE_RAN=1\nexport KOLEGA_SCRATCHPAD=/clobbered\n"
    for name in (".bash_profile", ".profile"):
        (home / name).write_text(hostile)
    result = await manager.exec_command(
        'echo "RAN=${PROFILE_RAN:-0} SP=$KOLEGA_SCRATCHPAD"',
        env={"KOLEGA_SCRATCHPAD": "/real/scratchpad", "HOME": str(home)},
        login=True,
        background=background,
        yield_time_ms=10000,
    )
    assert result.status == "exited"
    assert result.exit_code == 0
    assert "RAN=1" in result.output  # the profile really ran...
    assert "SP=/real/scratchpad" in result.output  # ...and still lost


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
async def test_background_spill_is_normalized_and_raw_spool_is_internal(manager, tmp_path):
    spill_root = tmp_path / "terminal-output"
    manager.configure_output_spill(spill_root)
    script = (
        "import sys,time;sys.stdout.write('BEGIN\\n' + 'x' * 60000 + '\\nEND\\n');sys.stdout.flush();time.sleep(30)"
    )
    command = f"{shlex.quote(sys.executable)} -c {shlex.quote(script)}"

    result = await manager.exec_command(command, yield_time_ms=1000, background=True)

    assert result.status == "running"
    assert result.session_id is not None
    session = manager.sessions[result.session_id]
    raw_spool = session.log_path
    assert raw_spool is not None
    assert result.spill_path is not None
    assert os.path.realpath(raw_spool) != os.path.realpath(result.spill_path)
    spill = Path(result.spill_path).read_bytes()
    assert spill.startswith(b"BEGIN\n")
    assert spill.endswith(b"\nEND\n")
    assert result.spill_bytes == len(spill)

    await manager.kill_session(result.session_id, "TERM")

    assert not os.path.exists(raw_spool)
    assert os.path.exists(result.spill_path)


@pytest.mark.asyncio
async def test_foreground_long_runner_is_killed_by_cleanup_all(manager):
    # Contrast with the background case: a normal command must NOT survive.
    result = await manager.exec_command("sleep 300", yield_time_ms=500)
    pid = manager.sessions[result.session_id].pid
    assert result.status == "running"
    await manager.cleanup_all()
    await asyncio.sleep(0.2)
    assert not _process_alive(pid)


@pytest.mark.asyncio
async def test_background_write_stdin_delivers_input(manager):
    result = await manager.exec_command(
        'while read line; do echo "got:$line"; done', yield_time_ms=500, background=True
    )
    assert result.status == "running"
    sid = result.session_id
    out = (await manager.write_stdin(sid, "hello\n", yield_time_ms=500)).output
    for _ in range(20):  # bounded: this suite has no pytest-timeout
        if "got:hello" in out:
            break
        out += (await manager.write_stdin(sid, "hello\n", yield_time_ms=500)).output
    assert "got:hello" in out
    await manager.kill_session(sid, "TERM")


@pytest.mark.asyncio
async def test_background_multiple_writes_no_eof_between(manager):
    # `while read` exits the moment stdin hits EOF, so three separate writes
    # landing while the loop is still running proves the FIFO never delivers
    # EOF between write_stdin calls.
    result = await manager.exec_command(
        'while read line; do echo "got:$line"; done', yield_time_ms=500, background=True
    )
    sid = result.session_id
    out = ""
    for word in ("one", "two", "three"):
        out += (await manager.write_stdin(sid, f"{word}\n", yield_time_ms=500)).output
    for _ in range(20):
        if all(f"got:{w}" in out for w in ("one", "two", "three")):
            break
        out += (await manager.write_stdin(sid, "ping\n", yield_time_ms=500)).output
    for word in ("one", "two", "three"):
        assert f"got:{word}" in out
    assert manager.sessions[sid].running is True, "reader loop exited — stdin saw EOF"
    await manager.kill_session(sid, "TERM")


@pytest.mark.asyncio
async def test_background_stdin_reader_survives_cleanup_all(manager, tmp_path):
    # A stdin-draining child would die the instant its stdin hit EOF; surviving
    # cleanup_all proves the FIFO never loses its last writer when we exit.
    marker = tmp_path / "lines.log"
    cmd = f'while read line; do echo "$line" >> {marker}; done'
    result = await manager.exec_command(cmd, yield_time_ms=500, background=True)
    assert result.status == "running"
    session = manager.sessions[result.session_id]
    pid, fifo = session.pid, session.stdin_path
    assert pid is not None and fifo is not None

    await manager.write_stdin(result.session_id, "before\n", yield_time_ms=500)
    await manager.cleanup_all()  # == kolega-code exit
    assert len(manager.sessions) == 0

    await asyncio.sleep(0.3)
    try:
        assert _process_alive(pid), "stdin-reading background process died on cleanup_all"
        # Delivery is still possible after our exit: any writer can open the FIFO.
        fd = os.open(fifo, os.O_WRONLY | os.O_NONBLOCK)
        try:
            os.write(fd, b"after\n")
        finally:
            os.close(fd)
        for _ in range(40):
            if marker.exists() and "after" in marker.read_text():
                break
            await asyncio.sleep(0.05)
        text = marker.read_text()
        assert "before" in text and "after" in text
    finally:
        # It really is a survivor, so we must clean it up ourselves.
        with contextlib.suppress(OSError):
            os.killpg(os.getpgid(pid), signal.SIGKILL)


@pytest.mark.asyncio
async def test_detached_write_true_then_false_after_exit(manager):
    result = await manager.exec_command("read x; echo done-$x", yield_time_ms=500, background=True)
    session = manager.sessions[result.session_id]
    fifo = session.stdin_path
    assert fifo is not None
    assert await session.write("go\n") is True
    for _ in range(100):  # bounded wait; the pump polls every 0.05s
        if session.exited.is_set():
            break
        await asyncio.sleep(0.05)
    assert session.exited.is_set()
    assert await session.write("late\n") is False  # ENXIO: no reader left
    final = await manager.write_stdin(result.session_id, "")  # exited -> drain returns instantly
    assert final.status == "exited" and final.exit_code == 0
    assert "done-go" in (result.output + final.output)
    assert not os.path.exists(fifo)  # finished session unlinked its FIFO


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
