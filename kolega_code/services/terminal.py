"""Local terminal backend: codex-style unified exec over real PTYs.

Each command runs as its own process under a pseudo-terminal (``PtySession``).
We stream output into a bounded head-tail buffer, report the real exit code via
``waitpid``, write stdin to running sessions, and signal/kill the process group
for interrupts and cleanup. There is no persistent shell: ``cd``/``export`` do
not carry across separate ``exec_command`` calls (use ``cd x && ...`` or pass a
``workdir``).
"""

import asyncio
import codecs
import contextlib
import fcntl
import os
import pty
import shlex
import signal
import struct
import subprocess
import sys
import tempfile
import termios
import time
from pathlib import Path
from typing import Any, Dict, Optional, Union

from ..events import AgentConnectionManager
from ..events import AgentEvent
from .base import ExecResult, TerminalCommandCancelled, TerminalManager
from .terminal_buffer import (
    DEFAULT_YIELD_MS,
    GLOBAL_MAX_TOOL_OUTPUT_TOKENS,
    MAX_YIELD_MS,
    MIN_YIELD_MS,
    TerminalOutputAccumulator,
    TerminalSpillStore,
    clamp_output_tokens,
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

# Env vars `uv run` injects when launching this app from a source checkout.
_RUNTIME_VENV_VARS = ("VIRTUAL_ENV", "VIRTUAL_ENV_PROMPT", "UV", "UV_RUN_RECURSION_DEPTH")

# Shell preamble installed ahead of every command. Each exec runs as a pty
# session leader, so when its shell exits the kernel SIGHUPs the whole
# foreground process group — silently killing anything the command left
# backgrounded with `&`. The EXIT trap turns that silent kill into an explicit
# warning that names the surviving-till-exit pids and the supported patterns.
# (`jobs -p` is empty when no background jobs linger, so quiet commands stay
# quiet; bash and sh both evaluate the trap the same way. The message avoids
# backticks and extra `%` specifiers: the trap body is re-evaluated when it
# fires, and printf reuses its format for surplus arguments.)
#
# Two guards keep the warning accurate:
# - The TERM/INT/HUP traps record signal-driven exits (kill_command) in
#   `_kolega_signaled` and re-exit with the equivalent 128+signal status. A
#   killed foreground command lingers in the job table when the EXIT trap
#   fires, so without this flag kill_command would warn as if `&` had been
#   misused. (Checking job state or pid liveness alone races under load —
#   the doomed child may not have been reaped yet.)
# - `kill -0` liveness filters jobs that already died on their own before a
#   normal exit, so the warning only names processes the shell exit actually
#   kills.
_BACKGROUND_JOB_TRAP = (
    "trap '_kolega_signaled=1; exit 143' TERM; "
    "trap '_kolega_signaled=1; exit 130' INT; "
    "trap '_kolega_signaled=1; exit 129' HUP; "
    'trap \'if [ -z "${_kolega_signaled:-}" ]; then pids=""; for p in $(jobs -p); do '
    'kill -0 $p 2>/dev/null && pids="$pids $p"; done; '
    'pids="${pids# }"; if [ -n "$pids" ]; then '
    'printf "\\n[kolega-code] WARNING: background process(es) (%s) backgrounded with & were terminated when this '
    "command ended. To keep a process running, re-run it with exec_command background=true; it is detached and "
    'keeps running after this command and after the kolega-code session ends (stop it with kill_command).\\n" "$pids" >&2; fi; fi\' EXIT; '
)


def _strip_runtime_venv(env: Dict[str, str], runtime_prefix: Optional[str] = None) -> Dict[str, str]:
    """Remove this process's own venv from ``env`` (mutating and returning it).

    Launching from a source checkout via ``uv run`` exports the app's dev venv
    (VIRTUAL_ENV + a PATH prepend), which child shells would otherwise inherit —
    making ``python``/``pip`` resolve into the app's venv instead of behaving
    like a fresh user terminal. Only the app's own venv is stripped: a venv the
    user activated deliberately is a different prefix and passes through.
    """
    venv = env.get("VIRTUAL_ENV")
    prefix = runtime_prefix or sys.prefix
    if not venv or os.path.realpath(venv) != os.path.realpath(prefix):
        return env
    for var in _RUNTIME_VENV_VARS:
        env.pop(var, None)
    venv_bin = os.path.realpath(os.path.join(venv, "bin"))
    path = env.get("PATH")
    if path:
        env["PATH"] = os.pathsep.join(entry for entry in path.split(os.pathsep) if os.path.realpath(entry) != venv_bin)
    return env


def build_child_env(extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """Assemble the environment for a model-facing child shell.

    The user's own environment minus the app's runtime venv, plus per-session
    overrides, plus the CLEAN_ENV overlay. Overrides are applied after the
    scrub so a caller can still set VIRTUAL_ENV deliberately.
    """
    env = _strip_runtime_venv(os.environ.copy())
    if extra:
        env.update(extra)
    env.update(CLEAN_ENV)
    return env


def _export_prefix(extra: Optional[Dict[str, str]]) -> str:
    """Shell prefix re-exporting per-session env overrides inside the command.

    Overrides are already in the spawn environment (build_child_env), but a
    login shell (``bash -lc``) sources profile files before the command string
    runs, and a profile may reset or unset them. Re-exporting inside the command
    string runs after any profile, so per-session overrides (e.g. the session
    scratchpad path) hold in every shell. Values are quoted; names come from
    internal callers and are shell-safe identifiers.
    """
    if not extra:
        return ""
    return "".join(f"export {name}={shlex.quote(value)}; " for name, value in extra.items())


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


def _signal_group(pid: Optional[int], sig: int) -> None:
    """Signal ``pid``'s whole process group, falling back to the bare pid.

    Every session leader (a ``setsid`` child) owns its process group, so
    ``killpg`` reaches the command and everything it spawned. Shared by the PTY
    and detached-background sessions.
    """
    if pid is None:
        return
    try:
        os.killpg(os.getpgid(pid), sig)
    except (ProcessLookupError, OSError):
        try:
            os.kill(pid, sig)
        except (ProcessLookupError, OSError):
            pass


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
        spill_store: Optional[TerminalSpillStore] = None,
        retain_full_delta: bool = False,
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
        self._output = TerminalOutputAccumulator(spill_store, retain_full_delta=retain_full_delta)
        self._display_decoder = codecs.getincrementaldecoder("utf-8")("replace")
        self._broadcast_queue: asyncio.Queue[tuple[str, str] | None] = asyncio.Queue()
        self._broadcast_task: Optional[asyncio.Task] = None
        self._reader_added = False
        self._reap_task: Optional[asyncio.Task] = None
        self._closed = False

    # -- lifecycle ---------------------------------------------------------

    async def start(self) -> None:
        env = build_child_env(self.env)

        command = self.command
        if self.auto_activate_venv:
            activate = os.path.join(str(self.workdir), _VENV_ACTIVATE_REL)
            if os.path.isfile(activate):
                command = f"source {shlex.quote(activate)} 2>/dev/null; {command}"
        # Install the background-job warning ahead of everything else so it
        # survives venv activation and still fires if the command backgrounds
        # processes and exits (the trap is overwritten only if the command
        # sets its own EXIT trap). Session env overrides are re-exported before
        # venv activation so an activate script still wins on conflicts.
        command = _BACKGROUND_JOB_TRAP + _export_prefix(self.env) + command

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
            if self.connection_manager:
                self._broadcast_task = asyncio.create_task(self._broadcast_worker())
            asyncio.get_event_loop().add_reader(master_fd, self._on_readable)
            self._reader_added = True
            self._reap_task = asyncio.create_task(self._poll_for_exit())

    def _remove_reader(self) -> None:
        if self._reader_added and self.master_fd is not None:
            try:
                asyncio.get_event_loop().remove_reader(self.master_fd)
            except Exception:
                pass
            self._reader_added = False

    # -- output reading ----------------------------------------------------

    def _on_readable(self) -> None:
        if self.master_fd is None:
            return
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
        self._output.append_bytes(data)
        self._new_output.set()
        self._broadcast(data)

    def _handle_eof(self) -> None:
        self._remove_reader()
        self._output.finalize()
        self._flush_display_decoder()
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

    async def _poll_for_exit(self) -> None:
        """Reap the shell when it exits without PTY EOF.

        PTY EOF is the normal exit signal, but it is withheld while any
        process still holds the slave open — and a backgrounded (``&``)
        process does exactly that on Linux, where the kernel does not
        reliably SIGHUP it when the session leader exits. Reaping the shell
        itself marks the command finished on every platform; ``_detach_pty``
        then releases the PTY, and the hangup that release causes kills any
        surviving background process — which keeps the EXIT trap's warning
        accurate.
        """
        while not self.exited.is_set():
            await asyncio.sleep(_REAP_POLL_INTERVAL)
            if not self._poll_reap():
                continue
            # The shell's last bytes (e.g. the background-job warning) may
            # still sit in the PTY buffer; yield once so the reader drains
            # them before the output is finalized and `exited` wakes waiters.
            await asyncio.sleep(_REAP_SETTLE_S)
            self._detach_pty()
            self.exited.set()
            return

    def _poll_reap(self) -> bool:
        """Reap the shell if it exited; True when this call did the reaping.

        Unlike :meth:`_reap` this does not set ``exited``: the poller must
        finalize the output first so a waiter (``drain``) only wakes once
        every byte is in the accumulator.
        """
        if self.exited.is_set() or self.pid is None:
            return False
        try:
            pid, status = os.waitpid(self.pid, os.WNOHANG)
        except ChildProcessError:
            return False  # the EOF path already reaped it
        if pid != self.pid:
            return False
        self.exit_code = _normalize_exit_code(status)
        return True

    def _detach_pty(self) -> None:
        """Release the PTY after the shell exited without EOF.

        The non-EOF analogue of :meth:`_handle_eof`: drop the reader, close
        the master (hanging up the slave, which SIGHUPs any surviving
        background process), finalize the output, and flush the display
        decoder.
        """
        self._remove_reader()
        if self.master_fd is not None:
            try:
                os.close(self.master_fd)
            except OSError:
                pass
            self.master_fd = None
        self._output.finalize()
        self._flush_display_decoder()
        self._new_output.set()

    def _broadcast(self, data: bytes) -> None:
        if not self.connection_manager:
            return
        raw_text = data.decode("utf-8", errors="replace")
        display_text = self._display_decoder.decode(data, final=False)
        self._enqueue_broadcast(raw_text, display_text)

    def _flush_display_decoder(self) -> None:
        if not self.connection_manager:
            return
        display_text = self._display_decoder.decode(b"", final=True)
        self._enqueue_broadcast(display_text, display_text)

    def _enqueue_broadcast(self, raw_text: str, display_text: str) -> None:
        if not raw_text and not display_text:
            return
        self._broadcast_queue.put_nowait((raw_text, display_text))

    async def _broadcast_worker(self) -> None:
        while True:
            item = await self._broadcast_queue.get()
            try:
                if item is None:
                    return
                raw_text, display_text = item
                event = AgentEvent(
                    event_type="terminal_output",
                    sender="agent",
                    content={
                        "output": raw_text,
                        "display_output": display_text,
                        "terminal_id": self.session_id,
                        "session_id": self.session_id,
                        "thread_id": self.thread_id,
                    },
                )
                await self._safe_broadcast(event)
            finally:
                self._broadcast_queue.task_done()

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

    def read_delta(self, max_output_tokens: int, *, hard_limit: bool = True):
        return self._output.read_delta(max_output_tokens, hard_limit=hard_limit)

    async def write(self, chars: str) -> bool:
        if self.master_fd is None:
            return False
        try:
            os.write(self.master_fd, chars.encode())
            return True
        except OSError:
            return False

    def _signal_group(self, sig: int) -> None:
        _signal_group(self.pid, sig)

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
        if self._reap_task is not None:
            self._reap_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._reap_task
            self._reap_task = None
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
        self._output.finalize()
        if self._broadcast_task is not None:
            with contextlib.suppress(Exception):
                await asyncio.wait_for(self._broadcast_queue.join(), timeout=0.5)
            self._broadcast_queue.put_nowait(None)
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await asyncio.wait_for(self._broadcast_task, timeout=0.5)
            self._broadcast_task = None

    @property
    def running(self) -> bool:
        return not self.exited.is_set()


# How often the detached-session pump tails its log file and checks liveness.
_DETACHED_POLL_INTERVAL = 0.05

# How often a PTY session reaps its shell. PTY EOF is the normal exit signal,
# but it is withheld while any process still holds the slave open — and a
# backgrounded (`&`) process does exactly that on Linux, where the kernel does
# not reliably SIGHUP it when the session leader exits. Without this poll,
# `sleep 30 & echo main-done` would report "running" for the whole 30s even
# though the shell finished. `waitpid(WNOHANG)` is cheap; 0.05s keeps exit
# detection invisible to callers.
_REAP_POLL_INTERVAL = 0.05
# Grace period after reaping the shell before the output is finalized: the
# shell's last bytes (e.g. the background-job warning) may still sit in the
# PTY buffer, and the reader callback needs a loop tick to drain them so the
# first read after `exited` sees the complete stream.
_REAP_SETTLE_S = 0.02


class DetachedSession:
    """A ``background=true`` command that outlives the kolega-code process.

    Unlike :class:`PtySession`, this is spawned **detached** — ``setsid`` (its own
    session, no controlling terminal), stdin from a private FIFO the child holds
    open O_RDWR as fd 0, and stdout+stderr to a temp log file. That combination
    is what makes it durable:

    - No controlling TTY, so it never gets SIGHUP when our terminal/PTY goes away.
    - Output goes to a real file (never a pipe/PTY we hold), so a write can't fail
      with EIO once we exit.
    - It is not tracked by an asyncio child-transport, so nothing SIGKILLs it when
      our event loop tears down.
    - Its stdin FIFO never loses its last writer (the child's own fd 0 is one), so
      the child never reads EOF — blocking reads just wait, like an idle terminal,
      including after we exit.

    The process therefore keeps running after the agent session ends (reparented
    to init) — matching claude-code's ``run_in_background``. It is stopped only by
    ``kill_command`` (SIGTERM/-KILL to its group) or by the container going away.
    ``write`` delivers real input through the FIFO; there is no echo (the FIFO is
    not a TTY), and stdin-draining commands (``while read``, REPLs, bare ``cat``)
    never see EOF, so they run until killed. Output may be block-buffered by the
    child (its stdout is a file, not a TTY), so verify a server as a client (curl)
    rather than waiting on its log.
    """

    detached = True

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
        spill_store: Optional[TerminalSpillStore] = None,
        retain_full_delta: bool = False,
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
        self.spill_store = spill_store
        self.retain_full_delta = retain_full_delta

        self.pid: Optional[int] = None
        self.exit_code: Optional[int] = None
        self.start_time = time.monotonic()
        self.log_path: Optional[str] = None
        self.stdin_path: Optional[str] = None

        self.exited = asyncio.Event()
        self._new_output = asyncio.Event()
        self._output: Optional[TerminalOutputAccumulator] = None
        self._proc: Optional[subprocess.Popen] = None
        self._reader = None
        self._read_offset = 0
        self._pump_task: Optional[asyncio.Task] = None
        self._closed = False

    async def start(self) -> None:
        env = build_child_env(self.env)

        command = self.command
        if self.auto_activate_venv:
            activate = os.path.join(str(self.workdir), _VENV_ACTIVATE_REL)
            if os.path.isfile(activate):
                command = f"source {shlex.quote(activate)} 2>/dev/null; {command}"
        # Same re-export as PtySession: session env overrides must survive
        # login-shell profiles, and run before venv activation.
        command = _export_prefix(self.env) + command

        shell = _pick_shell()
        shell_args = ["-lc", command] if self.login else ["-c", command]

        # The child writes raw bytes to an internal OS-temp spool. The pump
        # incrementally normalizes those bytes before the accumulator decides
        # whether to create a session-owned recoverable spill.
        fd, self.log_path = tempfile.mkstemp(prefix="kolega-bg-", suffix=".log")
        os.close(fd)
        self._output = TerminalOutputAccumulator(
            self.spill_store,
            retain_full_delta=self.retain_full_delta,
        )

        # Keep the detached stdin FIFO in OS temp storage. The session-owned log
        # may be deleted with its owning session; unlinking this pathname does
        # not disrupt the child's already-open fd 0, but it would prevent a
        # later write_stdin from reconnecting to a still-running process.
        fifo_fd, fifo_name = tempfile.mkstemp(prefix="kolega-bg-", suffix=".stdin")
        os.close(fifo_fd)
        os.unlink(fifo_name)
        self.stdin_path = fifo_name
        os.mkfifo(self.stdin_path, 0o600)
        self._reader = open(self.log_path, "rb")
        writer = open(self.log_path, "wb")
        # The FIFO is opened O_RDWR and handed to the child as fd 0. O_RDWR never
        # blocks waiting for a peer, and because the child's own fd 0 is then both
        # a reader AND a writer, the FIFO never loses its last writer while the
        # child (or any descendant that inherited fd 0) lives: the child never
        # reads EOF on stdin — not at startup, not between write() calls, not
        # after we exit. POSIX leaves O_RDWR-on-FIFO undefined; Linux documents it
        # as non-blocking (fifo(7)) and macOS/BSD behaves the same — the only
        # supported runtimes.
        stdin_fd = os.open(self.stdin_path, os.O_RDWR)
        try:
            # start_new_session=True -> setsid in the child: its own session and
            # process group, no controlling terminal. plain Popen (not asyncio)
            # so no child-transport kills it when our loop closes.
            self._proc = subprocess.Popen(  # noqa: S603 - shell chosen internally, args are controlled
                [shell, *shell_args],
                cwd=str(self.workdir),
                env=env,
                stdin=stdin_fd,
                stdout=writer,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                close_fds=True,  # std handles passed above are exempt
            )
        except Exception:
            # Launch failed: don't leak the reader fd or the temp files.
            with contextlib.suppress(OSError):
                self._reader.close()
            self._reader = None
            with contextlib.suppress(OSError):
                os.unlink(self.log_path)
            self.log_path = None
            with contextlib.suppress(OSError):
                os.unlink(self.stdin_path)
            self.stdin_path = None
            if self._output is not None:
                self._output.finalize()
                self._output = None
            raise
        finally:
            writer.close()  # the child holds its own dup of the fd
            os.close(stdin_fd)  # ditto: the child's fd 0 keeps the FIFO alive
        self.pid = self._proc.pid
        self._pump_task = asyncio.create_task(self._pump())

    def _read_new_bytes(self) -> bytes:
        if self._reader is None:
            return b""
        try:
            self._reader.seek(self._read_offset)
            data = self._reader.read()
            self._read_offset = self._reader.tell()
            return data
        except (OSError, ValueError):
            return b""

    async def _broadcast(self, data: bytes) -> None:
        if not self.connection_manager or not data:
            return
        text = data.decode("utf-8", errors="replace")
        try:
            await self.connection_manager.broadcast_event(
                AgentEvent(
                    event_type="terminal_output",
                    sender="agent",
                    content={
                        "output": text,
                        "display_output": text,
                        "terminal_id": self.session_id,
                        "session_id": self.session_id,
                        "thread_id": self.thread_id,
                    },
                ),
                self.workspace_id,
                self.thread_id,
            )
        except Exception:
            pass

    async def _pump(self) -> None:
        """Tail the log file and watch for exit until the process finishes."""
        while True:
            chunk = self._read_new_bytes()
            if chunk:
                assert self._output is not None
                self._output.append_bytes(chunk)
                self._new_output.set()
                await self._broadcast(chunk)
            rc = self._proc.poll() if self._proc else 0
            if rc is not None:
                tail = self._read_new_bytes()
                if tail:
                    assert self._output is not None
                    self._output.append_bytes(tail)
                    self._new_output.set()
                    await self._broadcast(tail)
                assert self._output is not None
                self._output.finalize()
                self.exit_code = rc if rc >= 0 else 128 + (-rc)
                self.exited.set()
                if self._closed:
                    self._cleanup_finished_files()
                self._pump_task = None
                return
            await asyncio.sleep(_DETACHED_POLL_INTERVAL)

    async def drain(self, yield_ms: int) -> None:
        try:
            await asyncio.wait_for(self.exited.wait(), timeout=yield_ms / 1000)
        except asyncio.TimeoutError:
            pass

    def read_delta(self, max_output_tokens: int, *, hard_limit: bool = True):
        assert self._output is not None
        return self._output.read_delta(max_output_tokens, hard_limit=hard_limit)

    async def write(self, chars: str) -> bool:
        """Deliver ``chars`` to the child's stdin FIFO.

        Opens the FIFO O_WRONLY|O_NONBLOCK per call: POSIX guarantees this
        succeeds while any process holds the read side (the child's own O_RDWR
        fd 0, for its whole life) and fails with ENXIO once the child and its
        fd-0 heirs are gone. There is no echo — the FIFO is not a TTY, so these
        bytes appear in the log only if the child prints them. ``stdin_path`` is
        always set between start() and close(), the only window in which the
        manager calls write().
        """
        assert self.stdin_path is not None
        try:
            fd = os.open(self.stdin_path, os.O_WRONLY | os.O_NONBLOCK)
        except OSError:
            # ENXIO: no reader left — the child (and anything that inherited
            # its fd 0) exited or closed stdin. Mirrors PtySession's False.
            return False
        try:
            data = chars.encode()
            while data:
                # Loop for partial writes (payloads > PIPE_BUF: 512 on macOS,
                # 4096 on Linux). CPython ignores SIGPIPE, so a lost-reader race
                # raises BrokenPipeError here, not a signal; a full pipe (child
                # not draining its backlog) raises BlockingIOError. Both are
                # OSError -> False.
                data = data[os.write(fd, data) :]
            return True
        except OSError:
            return False
        finally:
            os.close(fd)

    async def kill(self, signame: str = "TERM") -> None:
        first = signal.SIGINT if signame == "INT" else signal.SIGTERM
        _signal_group(self.pid, first)
        try:
            await asyncio.wait_for(self.exited.wait(), timeout=2.0)
        except asyncio.TimeoutError:
            _signal_group(self.pid, signal.SIGKILL)
            try:
                await asyncio.wait_for(self.exited.wait(), timeout=2.0)
            except asyncio.TimeoutError:
                pass

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        # Detached sessions are not signalled on manager close. Keep the pump
        # alive as a reaper/normalizer until the child exits, then remove the
        # internal raw spool and FIFO. This avoids zombies and preserves every
        # byte emitted after the session registry lets go of the command.
        if not self.exited.is_set():
            return
        if self._pump_task is not None:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._pump_task
            self._pump_task = None
        self._cleanup_finished_files()

    def _cleanup_finished_files(self) -> None:
        if self._reader is not None:
            with contextlib.suppress(OSError):
                self._reader.close()
            self._reader = None
        if self.log_path:
            with contextlib.suppress(OSError):
                os.unlink(self.log_path)
            self.log_path = None
        if self.stdin_path:
            with contextlib.suppress(OSError):
                os.unlink(self.stdin_path)
            self.stdin_path = None

    @property
    def running(self) -> bool:
        return not self.exited.is_set()


# A managed session is either a foreground PTY command or a detached background
# one; both expose the same start/drain/read_delta/write/kill/close interface.
Session = Union[PtySession, DetachedSession]


class LocalTerminalManager(TerminalManager):
    """Registry of local PTY sessions implementing the unified-exec interface."""

    def __init__(
        self,
        workspace_id: str,
        thread_id: str,
        connection_manager: AgentConnectionManager,
        default_workdir: Optional[Union[str, os.PathLike]] = None,
        spill_root: Optional[Union[str, os.PathLike]] = None,
    ):
        self.workspace_id = workspace_id
        self.thread_id = thread_id
        self.connection_manager = connection_manager
        # Naturally exited sessions remain here until write_stdin/kill_session
        # returns their final unread delta. list_sessions hides them and closes
        # their process resources without discarding that recoverable result.
        self.sessions: Dict[str, Session] = {}
        self._counter = 0
        self.auto_activate_venv = True
        # Default working directory for commands that don't pass one. Callers
        # that serve a project should pass its root; the process cwd is only a
        # fallback for standalone use.
        self.default_workdir = str(default_workdir) if default_workdir else os.getcwd()
        self._spill_store = TerminalSpillStore(Path(spill_root)) if spill_root is not None else None

    def configure_output_spill(self, root: Optional[Path]) -> None:
        """Set the session-owned root used for subsequent terminal sessions."""
        if root is None:
            return
        resolved = Path(root)
        if self._spill_store is not None and self._spill_store.root == resolved:
            return
        if self.sessions:
            raise RuntimeError("Cannot change terminal spill storage while sessions are running")
        self._spill_store = TerminalSpillStore(resolved)

    def _next_session_id(self) -> str:
        self._counter += 1
        return f"s_{self._counter}"

    @property
    def has_running_sessions(self) -> bool:
        """Whether a retained PTY could still mutate its original workdir."""
        return any(session.running for session in self.sessions.values())

    async def _emit_command(self, session_id: str, command: str, workdir: str) -> None:
        if not self.connection_manager:
            return
        try:
            await self.connection_manager.broadcast_event(
                AgentEvent(
                    event_type="terminal_command",
                    sender="agent",
                    content={
                        "command": command,
                        "terminal_id": session_id,
                        "session_id": session_id,
                        "workdir": workdir,
                    },
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

    def _result_from(
        self,
        session: Session,
        max_output_tokens: int,
        duration_ms: int,
        *,
        hard_limit: bool = True,
    ) -> ExecResult:
        capped = session.read_delta(max_output_tokens, hard_limit=hard_limit)
        if session.exited.is_set():
            return ExecResult(
                status="exited",
                session_id=None,
                exit_code=session.exit_code,
                output=capped.text,
                truncated=capped.truncated,
                original_token_count=capped.original_token_count,
                duration_ms=duration_ms,
                spill_path=capped.spill_path,
                spill_bytes=capped.spill_bytes,
                line_truncated_count=capped.line_truncated_count,
                line_truncated_bytes=capped.line_truncated_bytes,
                preview_omitted_bytes=capped.preview_omitted_bytes,
                preview_omitted_lines=capped.preview_omitted_lines,
            )
        return ExecResult(
            status="running",
            session_id=session.session_id,
            exit_code=None,
            output=capped.text,
            truncated=capped.truncated,
            original_token_count=capped.original_token_count,
            duration_ms=duration_ms,
            spill_path=capped.spill_path,
            spill_bytes=capped.spill_bytes,
            line_truncated_count=capped.line_truncated_count,
            line_truncated_bytes=capped.line_truncated_bytes,
            preview_omitted_bytes=capped.preview_omitted_bytes,
            preview_omitted_lines=capped.preview_omitted_lines,
        )

    async def _finish_if_exited(self, session: Session) -> None:
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
        background: bool = False,
    ) -> ExecResult:
        return await self._exec_command(
            command,
            workdir=workdir,
            yield_time_ms=yield_time_ms,
            max_output_tokens=max_output_tokens,
            login=login,
            env=env,
            background=background,
            hard_limit=True,
        )

    async def _exec_command(
        self,
        command: str,
        *,
        workdir: Optional[str],
        yield_time_ms: int,
        max_output_tokens: int,
        login: bool,
        env: Optional[Dict[str, str]],
        background: bool,
        hard_limit: bool,
    ) -> ExecResult:
        max_output_tokens = clamp_output_tokens(max_output_tokens) if hard_limit else max(1, int(max_output_tokens))
        yield_ms = clamp_yield(yield_time_ms, poll=False)
        # workdir should be absolute; a relative path would chdir relative to
        # this process's cwd, not default_workdir.
        wd = os.path.realpath(os.path.abspath(workdir or self.default_workdir))
        session_id = self._next_session_id()

        await self._emit_command(session_id, command, wd)
        # background=true launches a detached session that survives our exit; a
        # normal command runs under a PTY tied to this process's lifetime.
        session_cls = DetachedSession if background else PtySession
        session = session_cls(
            session_id,
            command,
            wd,
            self.connection_manager,
            self.workspace_id,
            self.thread_id,
            login=login,
            env=env,
            auto_activate_venv=self.auto_activate_venv,
            spill_store=self._spill_store,
            retain_full_delta=not hard_limit,
        )
        start = time.monotonic()
        await session.start()
        self.sessions[session_id] = session

        try:
            await session.drain(yield_ms)
        except asyncio.CancelledError:
            killed = await asyncio.shield(self._kill_session(session_id, "TERM", hard_limit=hard_limit))
            killed.status = "cancelled"
            raise TerminalCommandCancelled(killed) from None
        duration_ms = int((time.monotonic() - start) * 1000)
        result = self._result_from(session, max_output_tokens, duration_ms, hard_limit=hard_limit)
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
        return await self._write_stdin(
            session_id,
            chars,
            yield_time_ms=yield_time_ms,
            max_output_tokens=max_output_tokens,
            hard_limit=True,
        )

    async def _write_stdin(
        self,
        session_id: str,
        chars: str,
        *,
        yield_time_ms: int,
        max_output_tokens: int,
        hard_limit: bool,
        enforce_poll_minimum: bool = True,
    ) -> ExecResult:
        max_output_tokens = clamp_output_tokens(max_output_tokens) if hard_limit else max(1, int(max_output_tokens))
        session = self.sessions.get(session_id)
        if session is None:
            raise KeyError(f"No such session: {session_id}")

        yield_ms = clamp_yield(
            yield_time_ms,
            poll=(chars == "" and enforce_poll_minimum),
        )
        start = time.monotonic()
        if chars:
            # Both session types return a success bool; it is intentionally
            # discarded (parity between the PTY and detached paths). A write
            # that failed because the process died surfaces as status="exited"
            # in this same result: the drain below observes the exit well
            # within the >=250ms write window.
            await session.write(chars)
        try:
            await session.drain(yield_ms)
        except asyncio.CancelledError:
            killed = await asyncio.shield(self._kill_session(session_id, "TERM", hard_limit=hard_limit))
            killed.status = "cancelled"
            raise TerminalCommandCancelled(killed) from None
        duration_ms = int((time.monotonic() - start) * 1000)
        result = self._result_from(session, max_output_tokens, duration_ms, hard_limit=hard_limit)
        if result.status == "exited":
            await self._finish_if_exited(session)
        return result

    async def kill_session(self, session_id: str, signal: str = "TERM") -> ExecResult:
        return await self._kill_session(session_id, signal, hard_limit=True)

    async def _kill_session(
        self,
        session_id: str,
        signal: str,
        *,
        hard_limit: bool,
    ) -> ExecResult:
        session = self.sessions.get(session_id)
        if session is None:
            raise KeyError(f"No such session: {session_id}")
        start = time.monotonic()
        await session.kill(signal)
        duration_ms = int((time.monotonic() - start) * 1000)
        capped = session.read_delta(
            GLOBAL_MAX_TOOL_OUTPUT_TOKENS if hard_limit else 200_000,
            hard_limit=hard_limit,
        )
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
            spill_path=capped.spill_path,
            spill_bytes=capped.spill_bytes,
            line_truncated_count=capped.line_truncated_count,
            line_truncated_bytes=capped.line_truncated_bytes,
            preview_omitted_bytes=capped.preview_omitted_bytes,
            preview_omitted_lines=capped.preview_omitted_lines,
        )

    async def list_sessions(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        completed: list[Session] = []
        for session_id, session in list(self.sessions.items()):
            running = session.running
            if not running:
                completed.append(session)
                continue
            result[session_id] = {
                "command": session.command,
                "workdir": str(session.workdir),
                "runtime_s": round(time.monotonic() - session.start_time, 1),
                "running": True,
            }
        if completed:
            # Release PTY descriptors, detached pumps, raw spools, and FIFOs,
            # but retain each finalized output accumulator for one final poll.
            await asyncio.gather(*(session.close() for session in completed), return_exceptions=True)
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
        initial_yield_ms = min(
            MAX_YIELD_MS,
            max(MIN_YIELD_MS, int((deadline - time.monotonic()) * 1000)),
        )
        result = await self._exec_command(
            command,
            workdir=cwd,
            yield_time_ms=initial_yield_ms,
            max_output_tokens=200000,
            login=False,
            env=None,
            background=False,
            hard_limit=False,
        )
        parts = [result.output]
        session_id = result.session_id
        while result.status == "running" and session_id is not None and time.monotonic() < deadline:
            remaining_ms = max(1, int((deadline - time.monotonic()) * 1000))
            result = await self._write_stdin(
                session_id,
                "",
                yield_time_ms=min(MAX_YIELD_MS, remaining_ms),
                max_output_tokens=200000,
                hard_limit=False,
                enforce_poll_minimum=False,
            )
            parts.append(result.output)
        if result.status == "running" and session_id is not None:
            killed = await self._kill_session(session_id, "TERM", hard_limit=False)
            parts.append(killed.output)
        return "".join(parts)
