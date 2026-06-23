"""Local terminal backend: codex-style unified exec over real PTYs.

Each command runs as its own process under a pseudo-terminal (``PtySession``).
We stream output into a bounded head-tail buffer, report the real exit code via
``waitpid``, write stdin to running sessions, and signal/kill the process group
for interrupts and cleanup. There is no persistent shell: ``cd``/``export`` do
not carry across separate ``exec_command`` calls (use ``cd x && ...`` or pass a
``workdir``).
"""

import asyncio
import fcntl
import os
import pty
import shlex
import signal
import struct
import termios
import time
from typing import Any, Dict, Optional

from ..events import AgentConnectionManager
from ..events import AgentEvent
from .base import ExecResult, TerminalManager
from .terminal_buffer import (
    DEFAULT_YIELD_MS,
    MAX_POLL_MS,
    MAX_YIELD_MS,
    HeadTailBuffer,
    cap_tokens,
    clamp_yield,
)

# Default PTY window size so size-aware programs (git, less) format predictably.
DEFAULT_ROWS = 40
DEFAULT_COLS = 120

# Environment overlay that keeps program output clean for the model: no colors,
# no pager (which would otherwise block waiting for input), unbuffered Python.
CLEAN_ENV = {
    "TERM": "dumb",
    "NO_COLOR": "1",
    "CLICOLOR": "0",
    "PAGER": "cat",
    "GIT_PAGER": "cat",
    "PYTHONUNBUFFERED": "1",
}

_VENV_ACTIVATE_REL = os.path.join(".venv", "bin", "activate")


def _pick_shell() -> str:
    for shell in ("/bin/bash", "/bin/sh"):
        if os.path.exists(shell):
            return shell
    return "/bin/sh"


def _normalize_exit_code(status: int) -> int:
    """Translate a waitpid status into a shell-style exit code.

    Processes killed by a signal report ``128 + signum`` (e.g. 130 for SIGINT),
    matching what a shell would put in ``$?``.
    """
    code = os.waitstatus_to_exitcode(status)
    if code < 0:
        return 128 + (-code)
    return code


class PtySession:
    """A single command running under its own PTY."""

    def __init__(
        self,
        session_id: str,
        command: str,
        workdir: str,
        connection_manager: AgentConnectionManager,
        workspace_id: str,
        thread_id: str,
        *,
        login: bool = False,
        env: Optional[Dict[str, str]] = None,
        auto_activate_venv: bool = True,
    ):
        self.session_id = session_id
        self.command = command
        self.workdir = workdir
        self.connection_manager = connection_manager
        self.workspace_id = workspace_id
        self.thread_id = thread_id
        self.login = login
        self.env = env or {}
        self.auto_activate_venv = auto_activate_venv

        self.pid: Optional[int] = None
        self.master_fd: Optional[int] = None
        self.exit_code: Optional[int] = None
        self.start_time = time.monotonic()

        self.exited = asyncio.Event()
        self._new_output = asyncio.Event()
        self._buffer = HeadTailBuffer()  # output since the last read (delta)
        self._reader_added = False
        self._closed = False

    # -- lifecycle ---------------------------------------------------------

    async def start(self) -> None:
        env = os.environ.copy()
        if self.env:
            env.update(self.env)
        env.update(CLEAN_ENV)

        command = self.command
        if self.auto_activate_venv:
            activate = os.path.join(str(self.workdir), _VENV_ACTIVATE_REL)
            if os.path.isfile(activate):
                command = f"source {shlex.quote(activate)} 2>/dev/null; {command}"

        shell = _pick_shell()
        shell_args = ["-lc", command] if self.login else ["-c", command]

        pid, master_fd = pty.fork()
        if pid == 0:
            # Child: pty.fork() already called setsid(), so we are our own
            # session/process-group leader. cd into the workdir and exec.
            try:
                os.chdir(str(self.workdir))
            except Exception:
                pass
            try:
                os.execvpe(shell, [shell, *shell_args], env)
            except Exception:
                os._exit(127)
        else:
            self.pid = pid
            self.master_fd = master_fd
            try:
                fcntl.ioctl(master_fd, termios.TIOCSWINSZ, struct.pack("HHHH", DEFAULT_ROWS, DEFAULT_COLS, 0, 0))
            except OSError:
                pass
            flags = fcntl.fcntl(master_fd, fcntl.F_GETFL)
            fcntl.fcntl(master_fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)
            asyncio.get_event_loop().add_reader(master_fd, self._on_readable)
            self._reader_added = True

    def _remove_reader(self) -> None:
        if self._reader_added and self.master_fd is not None:
            try:
                asyncio.get_event_loop().remove_reader(self.master_fd)
            except Exception:
                pass
            self._reader_added = False

    # -- output reading ----------------------------------------------------

    def _on_readable(self) -> None:
        try:
            data = os.read(self.master_fd, 65536)
        except (BlockingIOError, InterruptedError):
            return
        except OSError:
            # EIO on macOS when the child has exited; treat as EOF.
            data = b""
        if not data:
            self._handle_eof()
            return
        self._buffer.append(data)
        self._new_output.set()
        self._broadcast(data)

    def _handle_eof(self) -> None:
        self._remove_reader()
        self._reap()
        self._new_output.set()
        if not self.exited.is_set():
            asyncio.ensure_future(self._await_exit())

    async def _await_exit(self) -> None:
        # The slave fd closed but the child may not be reaped yet; poll briefly.
        while not self.exited.is_set():
            await asyncio.sleep(0.02)
            self._reap()
        self._new_output.set()

    def _reap(self) -> None:
        if self.exited.is_set() or self.pid is None:
            return
        try:
            pid, status = os.waitpid(self.pid, os.WNOHANG)
        except ChildProcessError:
            if self.exit_code is None:
                self.exit_code = -1
            self.exited.set()
            return
        if pid == self.pid:
            self.exit_code = _normalize_exit_code(status)
            self.exited.set()

    def _broadcast(self, data: bytes) -> None:
        if not self.connection_manager:
            return
        text = data.decode("utf-8", errors="replace")
        event = AgentEvent(
            event_type="terminal_output",
            sender="agent",
            content={
                "output": text,
                "terminal_id": self.session_id,
                "session_id": self.session_id,
                "thread_id": self.thread_id,
            },
        )
        asyncio.ensure_future(self._safe_broadcast(event))

    async def _safe_broadcast(self, event: AgentEvent) -> None:
        try:
            await self.connection_manager.broadcast_event(event, self.workspace_id, self.thread_id)
        except Exception:
            pass

    # -- interaction -------------------------------------------------------

    async def drain(self, yield_ms: int) -> None:
        """Wait until the process exits or ``yield_ms`` elapses."""
        try:
            await asyncio.wait_for(self.exited.wait(), timeout=yield_ms / 1000)
        except asyncio.TimeoutError:
            pass

    def read_delta(self, max_output_tokens: int):
        text = self._buffer.text()
        self._buffer.reset()
        return cap_tokens(text, max_output_tokens)

    async def write(self, chars: str) -> bool:
        if self.master_fd is None:
            return False
        try:
            os.write(self.master_fd, chars.encode())
            return True
        except OSError:
            return False

    def _signal_group(self, sig: int) -> None:
        if self.pid is None:
            return
        try:
            os.killpg(os.getpgid(self.pid), sig)
        except (ProcessLookupError, OSError):
            try:
                os.kill(self.pid, sig)
            except (ProcessLookupError, OSError):
                pass

    async def kill(self, signame: str = "TERM") -> None:
        if signame == "INT":
            self._signal_group(signal.SIGINT)
        else:
            self._signal_group(signal.SIGTERM)
        try:
            await asyncio.wait_for(self.exited.wait(), timeout=2.0)
        except asyncio.TimeoutError:
            self._signal_group(signal.SIGKILL)
            try:
                await asyncio.wait_for(self.exited.wait(), timeout=2.0)
            except asyncio.TimeoutError:
                pass

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._remove_reader()
        if not self.exited.is_set():
            self._signal_group(signal.SIGKILL)
            self._reap()
        if self.master_fd is not None:
            try:
                os.close(self.master_fd)
            except OSError:
                pass
            self.master_fd = None

    @property
    def running(self) -> bool:
        return not self.exited.is_set()


class LocalTerminalManager(TerminalManager):
    """Registry of local PTY sessions implementing the unified-exec interface."""

    def __init__(self, workspace_id: str, thread_id: str, connection_manager: AgentConnectionManager):
        self.workspace_id = workspace_id
        self.thread_id = thread_id
        self.connection_manager = connection_manager
        self.sessions: Dict[str, PtySession] = {}
        self._counter = 0
        self.auto_activate_venv = True
        # Default working directory for commands that don't pass one.
        self.default_workdir = os.getcwd()

    def _next_session_id(self) -> str:
        self._counter += 1
        return f"s_{self._counter}"

    async def _emit_command(self, session_id: str, command: str) -> None:
        if not self.connection_manager:
            return
        try:
            await self.connection_manager.broadcast_event(
                AgentEvent(
                    event_type="terminal_command",
                    sender="agent",
                    content={"command": command, "terminal_id": session_id, "session_id": session_id},
                ),
                self.workspace_id,
                self.thread_id,
            )
        except Exception:
            pass

    async def _emit_output(self, session_id: str, text: str) -> None:
        if not self.connection_manager:
            return
        try:
            await self.connection_manager.broadcast_event(
                AgentEvent(
                    event_type="terminal_output",
                    sender="agent",
                    content={
                        "output": text,
                        "terminal_id": session_id,
                        "session_id": session_id,
                        "thread_id": self.thread_id,
                    },
                ),
                self.workspace_id,
                self.thread_id,
            )
        except Exception:
            pass

    def _result_from(self, session: PtySession, max_output_tokens: int, duration_ms: int) -> ExecResult:
        capped = session.read_delta(max_output_tokens)
        if session.exited.is_set():
            return ExecResult(
                status="exited",
                session_id=None,
                exit_code=session.exit_code,
                output=capped.text,
                truncated=capped.truncated,
                original_token_count=capped.original_token_count,
                duration_ms=duration_ms,
            )
        return ExecResult(
            status="running",
            session_id=session.session_id,
            exit_code=None,
            output=capped.text,
            truncated=capped.truncated,
            original_token_count=capped.original_token_count,
            duration_ms=duration_ms,
        )

    async def _finish_if_exited(self, session: PtySession) -> None:
        await self._emit_output(session.session_id, f"[exited {session.exit_code}]\n")
        await session.close()
        self.sessions.pop(session.session_id, None)

    async def exec_command(
        self,
        command: str,
        *,
        workdir: Optional[str] = None,
        yield_time_ms: int = DEFAULT_YIELD_MS,
        max_output_tokens: int = 10000,
        login: bool = False,
        env: Optional[Dict[str, str]] = None,
    ) -> ExecResult:
        yield_ms = clamp_yield(yield_time_ms, poll=False)
        wd = workdir or self.default_workdir or os.getcwd()
        session_id = self._next_session_id()

        await self._emit_command(session_id, command)
        session = PtySession(
            session_id,
            command,
            wd,
            self.connection_manager,
            self.workspace_id,
            self.thread_id,
            login=login,
            env=env,
            auto_activate_venv=self.auto_activate_venv,
        )
        start = time.monotonic()
        await session.start()
        self.sessions[session_id] = session

        await session.drain(yield_ms)
        duration_ms = int((time.monotonic() - start) * 1000)
        result = self._result_from(session, max_output_tokens, duration_ms)
        if result.status == "exited":
            await self._finish_if_exited(session)
        return result

    async def write_stdin(
        self,
        session_id: str,
        chars: str = "",
        *,
        yield_time_ms: int = DEFAULT_YIELD_MS,
        max_output_tokens: int = 10000,
    ) -> ExecResult:
        session = self.sessions.get(session_id)
        if session is None:
            raise KeyError(f"No such session: {session_id}")

        yield_ms = clamp_yield(yield_time_ms, poll=(chars == ""))
        start = time.monotonic()
        if chars:
            await session.write(chars)
        await session.drain(yield_ms)
        duration_ms = int((time.monotonic() - start) * 1000)
        result = self._result_from(session, max_output_tokens, duration_ms)
        if result.status == "exited":
            await self._finish_if_exited(session)
        return result

    async def kill_session(self, session_id: str, signal: str = "TERM") -> ExecResult:
        session = self.sessions.get(session_id)
        if session is None:
            raise KeyError(f"No such session: {session_id}")
        start = time.monotonic()
        await session.kill(signal)
        duration_ms = int((time.monotonic() - start) * 1000)
        capped = session.read_delta(10000)
        exit_code = session.exit_code
        await self._emit_output(session_id, f"[exited {exit_code}]\n")
        await session.close()
        self.sessions.pop(session_id, None)
        return ExecResult(
            status="exited",
            session_id=None,
            exit_code=exit_code,
            output=capped.text,
            truncated=capped.truncated,
            original_token_count=capped.original_token_count,
            duration_ms=duration_ms,
        )

    async def list_sessions(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for session_id, session in self.sessions.items():
            result[session_id] = {
                "command": session.command,
                "workdir": str(session.workdir),
                "runtime_s": round(time.monotonic() - session.start_time, 1),
                "running": session.running,
            }
        return result

    async def close_all(self) -> None:
        for session in list(self.sessions.values()):
            try:
                await session.close()
            except Exception:
                pass
        self.sessions.clear()

    async def cleanup_all(self) -> None:
        await self.close_all()

    async def run_command(self, command: str, cwd: Optional[str] = None, timeout: Optional[int] = None) -> str:
        """Run a command to completion and return its combined output.

        Convenience method for internal callers (builds, sandbox setup). Streams
        through the session model, accumulating output across poll windows.
        """
        deadline = time.monotonic() + (timeout if timeout and timeout > 0 else 600)
        result = await self.exec_command(command, workdir=cwd, yield_time_ms=MAX_YIELD_MS, max_output_tokens=200000)
        parts = [result.output]
        session_id = result.session_id
        while result.status == "running" and time.monotonic() < deadline:
            result = await self.write_stdin(session_id, "", yield_time_ms=MAX_POLL_MS, max_output_tokens=200000)
            parts.append(result.output)
        if result.status == "running" and session_id is not None:
            killed = await self.kill_session(session_id, "TERM")
            parts.append(killed.output)
        return "".join(parts)
