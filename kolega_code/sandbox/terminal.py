"""Terminal manager implementation for sandbox environments."""

import asyncio
import base64
import os
import re
import shlex
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional, Callable, Awaitable
from datetime import datetime, timezone

from ..services.base import ExecResult, TerminalCommandCancelled, TerminalManager
from ..services.terminal_buffer import (
    GLOBAL_MAX_TOOL_OUTPUT_TOKENS,
    TerminalOutputAccumulator,
    TerminalSpillStore,
    clamp_output_tokens,
    clamp_yield,
)
from kolega_code.events import AgentEvent


class BoundedTerminalOutputs(list):
    """List-compatible terminal history with a hard in-memory byte ceiling."""

    MAX_BYTES = 1_000_000

    def __init__(self, values=()):
        super().__init__()
        self._bytes = 0
        self.extend(values)

    @staticmethod
    def _item_bytes(item: Any) -> int:
        if not isinstance(item, dict):
            return 0
        return len(str(item.get("data", "")).encode("utf-8"))

    def append(self, item: Any) -> None:
        super().append(item)
        self._bytes += self._item_bytes(item)
        while self._bytes > self.MAX_BYTES and len(self) > 1:
            removed = super().pop(0)
            self._bytes -= self._item_bytes(removed)

    def extend(self, values) -> None:
        for value in values:
            self.append(value)


class SandboxTerminalManager(TerminalManager):
    """Terminal manager that operates within a sandbox."""

    output_buffer_type = BoundedTerminalOutputs

    def __init__(self, sandbox: Any, workspace_id: str, thread_id: str, connection_manager: Any = None):
        """
        Initialize sandbox terminal manager.

        Args:
            sandbox: The sandbox instance (e.g., E2B Sandbox)
            workspace_id: ID of the workspace
            thread_id: ID of the thread
            connection_manager: Connection manager for broadcasting events (optional)
        """
        self.sandbox = sandbox
        self.workspace_id = workspace_id
        self.thread_id = thread_id
        self.connection_manager = connection_manager
        self.terminals: Dict[str, Dict[str, Any]] = {}
        self.outputs: Dict[str, list] = {}
        self._default_terminal_id: Optional[str] = None
        self._spill_store: Optional[TerminalSpillStore] = None
        self._output_accumulators: Dict[str, TerminalOutputAccumulator] = {}
        self._command_tasks: Dict[str, asyncio.Task] = {}

        # Track commands and their status (for interface parity)
        self.command_history: Dict[str, Dict[str, Any]] = {}
        self.command_counter = 0

    def configure_output_spill(self, root: Optional[Path]) -> None:
        if root is None:
            return
        resolved = Path(root)
        if self._spill_store is not None and self._spill_store.root == resolved:
            return
        if any(info.get("status") == "running" for info in self.command_history.values()):
            raise RuntimeError("Cannot change terminal spill storage while sessions are running")
        self._spill_store = TerminalSpillStore(resolved)

    def set_connection_manager(self, connection_manager: Any) -> None:
        """
        Set the connection manager for streaming terminal output.
        This allows setting it after creation when it becomes available.

        Args:
            connection_manager: Connection manager for broadcasting events
        """
        self.connection_manager = connection_manager

    async def _ensure_default_terminal(self) -> str:
        """Ensure a default terminal exists and return its ID."""
        if self._default_terminal_id is None or self._default_terminal_id not in self.terminals:
            self._default_terminal_id = await self.launch_terminal()
        return self._default_terminal_id

    # -- unified-exec interface (codex-style) ------------------------------
    #
    # These adapt the new session model onto the existing e2b machinery: a
    # "session" is a tracked background command (send_command_tracked), stdin is
    # written via send_input/send_stdin, and the exit code comes from the real
    # e2b handle.wait(). The legacy terminal/command bookkeeping is preserved so
    # state serialization keeps working.

    async def _wait_for_command(self, command_id: str, yield_ms: int) -> None:
        """Wait until a tracked command finishes or the yield window elapses."""
        deadline = time.monotonic() + yield_ms / 1000
        while time.monotonic() < deadline:
            info = self.command_history.get(command_id)
            if not info or info.get("status") in ("completed", "failed", "terminated"):
                return
            await asyncio.sleep(0.05)

    def _render_delta(self, command_id: str, max_output_tokens: int):
        accumulator = self._output_accumulators.get(command_id)
        if accumulator is None:
            return TerminalOutputAccumulator(None).read_delta(max_output_tokens)
        return accumulator.read_delta(max_output_tokens)

    async def _sync_sandbox_spill(self, command_id: str, capped) -> None:
        """Mirror newly persisted normalized bytes to a sandbox-readable file."""
        if not capped.spill_path:
            return
        info = self.command_history.get(command_id)
        if info is None:
            return

        remote_path = info.get("sandbox_spill_path")
        if remote_path is None:
            remote_path = f"/home/user/.kolega/terminal-output/{uuid.uuid4().hex}.exec_command.log"
            parent = os.path.dirname(remote_path)
            result = await self.sandbox.commands.run(
                f"mkdir -p {shlex.quote(parent)} && : > {shlex.quote(remote_path)}",
                timeout=60,
            )
            if getattr(result, "exit_code", 1) != 0:
                return
            info["sandbox_spill_path"] = remote_path
            info["sandbox_spill_bytes"] = 0

        synced_bytes = int(info.get("sandbox_spill_bytes", 0))
        target_bytes = capped.spill_bytes
        if target_bytes > synced_bytes:
            try:
                with Path(capped.spill_path).open("rb") as source:
                    source.seek(synced_bytes)
                    while synced_bytes < target_bytes:
                        chunk = source.read(min(48 * 1024, target_bytes - synced_bytes))
                        if not chunk:
                            break
                        encoded = base64.b64encode(chunk).decode("ascii")
                        result = await self.sandbox.commands.run(
                            (f"printf %s {shlex.quote(encoded)} | base64 -d >> {shlex.quote(remote_path)}"),
                            timeout=60,
                        )
                        if getattr(result, "exit_code", 1) != 0:
                            return
                        synced_bytes += len(chunk)
                        info["sandbox_spill_bytes"] = synced_bytes
            except OSError:
                return

        if synced_bytes < target_bytes:
            return
        capped.spill_path = remote_path
        if info.get("status") in ("completed", "failed", "terminated"):
            result = await self.sandbox.commands.run(
                f"chmod 0400 {shlex.quote(remote_path)}",
                timeout=60,
            )
            if getattr(result, "exit_code", 1) != 0:
                return

    async def _exec_result(self, command_id: str, max_output_tokens: int, duration_ms: int) -> ExecResult:
        capped = self._render_delta(command_id, max_output_tokens)
        await self._sync_sandbox_spill(command_id, capped)
        info = self.command_history.get(command_id) or {}
        status = info.get("status")
        if status in ("completed", "failed", "terminated") or status is None:
            self._output_accumulators.pop(command_id, None)
            return ExecResult(
                status="exited",
                session_id=None,
                exit_code=info.get("return_code"),
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
            session_id=command_id,
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

    async def exec_command(
        self,
        command: str,
        *,
        workdir: Optional[str] = None,
        yield_time_ms: int = 10000,
        max_output_tokens: int = 10000,
        login: bool = False,
        env: Optional[Dict[str, str]] = None,
        background: bool = False,
    ) -> ExecResult:
        # background is a lifetime hint for the local backend; here the command
        # runs in the sandbox's persistent terminal, which already outlives any
        # single exec and lives until the sandbox is destroyed, so no special
        # handling.
        _ = background
        max_output_tokens = clamp_output_tokens(max_output_tokens)
        yield_ms = clamp_yield(yield_time_ms, poll=False)
        terminal_id = await self._ensure_default_terminal()
        info = self.terminals[terminal_id]
        if workdir:
            if hasattr(workdir, "__fspath__"):
                workdir = str(workdir)
            info["cwd"] = workdir
        if env:
            merged = dict(info.get("env") or {})
            merged.update(env)
            info["env"] = merged

        start = time.monotonic()
        command_id = await self.send_command_tracked(terminal_id, command, timeout=0)
        if command_id is None:
            return ExecResult(status="exited", session_id=None, exit_code=1, output="Failed to start command")
        try:
            await self._wait_for_command(command_id, yield_ms)
        except asyncio.CancelledError:
            killed = await asyncio.shield(self.kill_session(command_id, "TERM"))
            killed.status = "cancelled"
            raise TerminalCommandCancelled(killed) from None
        duration_ms = int((time.monotonic() - start) * 1000)
        return await self._exec_result(command_id, max_output_tokens, duration_ms)

    async def write_stdin(
        self,
        session_id: str,
        chars: str = "",
        *,
        yield_time_ms: int = 10000,
        max_output_tokens: int = 10000,
    ) -> ExecResult:
        info = self.command_history.get(session_id)
        if not info:
            raise KeyError(f"No such session: {session_id}")
        max_output_tokens = clamp_output_tokens(max_output_tokens)
        terminal_id = info["terminal_id"]
        yield_ms = clamp_yield(yield_time_ms, poll=(chars == ""))
        start = time.monotonic()
        if chars:
            try:
                await self.send_input(terminal_id, chars, submit=False, command_id=session_id)
            except ValueError:
                # The command may have just finished; report its final status.
                pass
        try:
            await self._wait_for_command(session_id, yield_ms)
        except asyncio.CancelledError:
            killed = await asyncio.shield(self.kill_session(session_id, "TERM"))
            killed.status = "cancelled"
            raise TerminalCommandCancelled(killed) from None
        duration_ms = int((time.monotonic() - start) * 1000)
        return await self._exec_result(session_id, max_output_tokens, duration_ms)

    async def kill_session(self, session_id: str, signal: str = "TERM") -> ExecResult:
        info = self.command_history.get(session_id)
        if not info:
            raise KeyError(f"No such session: {session_id}")
        terminal_id = info["terminal_id"]
        pid = info.get("pid")
        handle = info.get("handle")
        info["kill_requested"] = signal
        # e2b exposes a single kill; for INT, best-effort send Ctrl-C via stdin.
        if signal == "INT" and pid is not None:
            try:
                await self.sandbox.commands.send_stdin(pid, "\x03")
            except Exception:
                pass
        if handle is not None:
            try:
                await handle.kill()
            except Exception:
                pass
        task = self._command_tasks.get(session_id)
        if task is not None:
            try:
                await asyncio.wait_for(asyncio.shield(task), timeout=5)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                pass
        if info.get("status") == "running":
            info["status"] = "terminated"
            if info.get("return_code") is None:
                info["return_code"] = 130 if signal == "INT" else 143
        active = self.terminals.get(terminal_id, {}).get("active_commands")
        if active is not None:
            active.pop(session_id, None)
        accumulator = self._output_accumulators.get(session_id)
        if accumulator is not None:
            accumulator.finalize()
        capped = self._render_delta(session_id, GLOBAL_MAX_TOOL_OUTPUT_TOKENS)
        await self._sync_sandbox_spill(session_id, capped)
        self._output_accumulators.pop(session_id, None)
        return ExecResult(
            status="exited",
            session_id=None,
            exit_code=info.get("return_code"),
            output=capped.text,
            truncated=capped.truncated,
            original_token_count=capped.original_token_count,
            spill_path=capped.spill_path,
            spill_bytes=capped.spill_bytes,
            line_truncated_count=capped.line_truncated_count,
            line_truncated_bytes=capped.line_truncated_bytes,
            preview_omitted_bytes=capped.preview_omitted_bytes,
            preview_omitted_lines=capped.preview_omitted_lines,
        )

    async def list_sessions(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for command_id, info in self.command_history.items():
            if info.get("status") == "running":
                terminal_id = info.get("terminal_id")
                if terminal_id is not None:
                    result[command_id] = {
                        "command": info.get("command", ""),
                        "workdir": self.terminals.get(terminal_id, {}).get("cwd"),
                        "running": True,
                    }
        return result

    async def get_last_command(self, terminal_id: str) -> str:
        """Get the last command sent to a terminal."""
        if terminal_id not in self.terminals:
            raise KeyError(f"Terminal {terminal_id} not found")

        terminal_info = self.terminals[terminal_id]
        return terminal_info.get("last_command", "")

    async def get_last_command_purpose(self, terminal_id: str) -> str:
        """Get the purpose of the last command sent to a terminal."""
        if terminal_id not in self.terminals:
            raise KeyError(f"Terminal {terminal_id} not found")

        terminal_info = self.terminals[terminal_id]
        return terminal_info.get("last_command_purpose", "")

    def _handle_cd_command(self, command: str, current_dir: str, terminal_info: Dict[str, Any]) -> None:
        """
        Check if command is a cd command and update terminal's working directory if so.

        Args:
            command: The command that was executed
            current_dir: The current working directory
            terminal_info: Terminal info dict to update
        """
        # Match cd followed by path, stopping at ; or && or ||
        cd_match = re.match(r"^\s*cd\s+([^;&|]+)", command.strip())
        if not cd_match:
            return

        new_dir = cd_match.group(1).strip()

        if (new_dir.startswith('"') and new_dir.endswith('"')) or (new_dir.startswith("'") and new_dir.endswith("'")):
            new_dir = new_dir[1:-1]

        if new_dir.startswith("/"):
            new_working_dir = new_dir
        elif new_dir == "..":
            new_working_dir = os.path.dirname(current_dir.rstrip("/"))
            if not new_working_dir:
                new_working_dir = "/"
        elif new_dir == ".":
            new_working_dir = current_dir
        elif new_dir == "~":
            new_working_dir = "/home/user"
        else:
            new_working_dir = os.path.join(current_dir, new_dir)

        # Normalize the path (handle double slashes)
        new_working_dir = os.path.normpath(new_working_dir)
        # Ensure single leading slash for absolute paths
        if new_working_dir.startswith("//"):
            new_working_dir = new_working_dir[1:]

        terminal_info["cwd"] = new_working_dir

    async def _create_output_handler(
        self,
        terminal_id: str,
        output_type: str,
        command_id: Optional[str] = None,
    ) -> Callable[[str], Awaitable[None]]:
        """
        Create an async output handler for streaming.

        Args:
            terminal_id: ID of the terminal
            output_type: Type of output ('stdout' or 'stderr')

        Returns:
            Async callback function for handling output
        """

        async def handler(data: str) -> None:
            self.outputs[terminal_id].append(
                {"type": output_type, "data": data, "timestamp": datetime.now(timezone.utc)}
            )
            if command_id is not None:
                accumulator = self._output_accumulators.get(command_id)
                if accumulator is not None:
                    accumulator.append_text(data)

            # Broadcast output immediately for streaming
            if self.connection_manager:
                try:
                    terminal_output_event = AgentEvent(
                        event_type="terminal_output",
                        sender="agent",
                        content={
                            "output": data,
                            "terminal_id": terminal_id,
                            "thread_id": self.thread_id,
                        },
                    )
                    await self.connection_manager.broadcast_event(
                        terminal_output_event, self.workspace_id, self.thread_id
                    )
                except Exception:
                    # Don't let broadcast errors affect command execution
                    pass

        return handler

    async def run_command(self, command: str, cwd: Optional[str] = None, timeout: Optional[int] = None) -> str:
        """
        Run a command directly (convenience method for utilities).

        Args:
            command: Command to execute
            cwd: Optional working directory (defaults to /home/user/workspace)
            timeout: Optional timeout in seconds (0 for no timeout, None for default 60s)

        Returns:
            Command output as string
        """
        working_dir = cwd if cwd is not None else "/home/user/workspace"

        # Convert Path objects to strings for E2B compatibility
        if hasattr(working_dir, "__fspath__"):
            working_dir = str(working_dir)

        # Ensure the working directory exists (E2B specific fix)
        if working_dir != "/home/user":
            try:
                await self.sandbox.commands.run(f"test -d {working_dir} || mkdir -p {working_dir}")
            except Exception:
                # If we can't create it, fall back to /home/user
                working_dir = "/home/user"

        try:
            sandbox_timeout = timeout if timeout is not None else 60  # Default to 60s for backward compatibility

            # For utility commands, we don't need streaming
            if sandbox_timeout == 0:
                result = await self.sandbox.commands.run(command, cwd=working_dir, timeout=0)
            else:
                # Use timeout with buffer for asyncio
                result = await asyncio.wait_for(
                    self.sandbox.commands.run(command, cwd=working_dir, timeout=sandbox_timeout),
                    timeout=sandbox_timeout + 5,  # Give 5 seconds more than the sandbox timeout
                )

            output = ""
            if result.stdout:
                output += result.stdout
            if result.stderr:
                if output:
                    output += "\n"
                output += result.stderr

            return output

        except asyncio.TimeoutError:
            return f"Command execution timed out after {sandbox_timeout + 5} seconds"
        except Exception as e:
            return f"Command failed: {str(e)}"

    async def launch_terminal(self, terminal_id: Optional[str] = None, **terminal_kwargs) -> str:
        """
        Launch a new terminal session.

        Args:
            terminal_id: Optional ID for the terminal. If not provided, generates UUID.
            **terminal_kwargs: Additional terminal options:
                - cwd: Working directory (default: /home/user/workspace)
                - env: Environment variables (default: {})

        Returns:
            Terminal ID
        """
        if terminal_id is None:
            terminal_id = str(uuid.uuid4())

        cwd = terminal_kwargs.get("cwd", "/home/user/workspace")
        env = terminal_kwargs.get("env", {})

        # Convert Path objects to strings for E2B compatibility
        if hasattr(cwd, "__fspath__"):
            cwd = str(cwd)

        try:
            await self.sandbox.commands.run(f"test -d {cwd}")
        except Exception:
            try:
                await self.sandbox.commands.run(f"mkdir -p {cwd}")
            except Exception as e:
                # If we can't create the directory, fall back to /home/user
                print(f"Warning: Could not ensure directory {cwd} exists: {e}")
                cwd = "/home/user"

        self.terminals[terminal_id] = {
            "created_at": datetime.now(timezone.utc),
            "cwd": cwd,
            "env": env,
            "process": None,
            "last_command": "",
            "last_command_purpose": "",
            "active_commands": {},
        }
        self.outputs[terminal_id] = BoundedTerminalOutputs()

        return terminal_id

    async def send_command(
        self, terminal_id: str, command: str, purpose: Optional[str] = None, timeout: Optional[int] = None
    ) -> bool:
        """
        Send a command to a terminal.

        Args:
            terminal_id: ID of the terminal
            command: Command to execute
            purpose: Optional description of command purpose
            timeout: Optional timeout in seconds (0 or None for no timeout)

        Returns:
            True if command was sent successfully

        Raises:
            KeyError: If terminal doesn't exist
        """
        if terminal_id not in self.terminals:
            raise KeyError(f"Terminal {terminal_id} not found")

        terminal_info = self.terminals[terminal_id]

        terminal_info["last_command"] = command.rstrip("\n")
        terminal_info["last_command_purpose"] = purpose or ""

        working_dir = terminal_info["cwd"]

        # Convert Path objects to strings for E2B compatibility
        if hasattr(working_dir, "__fspath__"):
            working_dir = str(working_dir)

        # Forward the terminal's environment to the sandbox command.
        env = terminal_info.get("env") or {}

        self.outputs[terminal_id].append(
            {"type": "command", "data": command, "timestamp": datetime.now(timezone.utc), "purpose": purpose}
        )

        if self.connection_manager:
            try:
                await self._broadcast_output(terminal_id, f"$ {command}\n")
            except Exception:
                pass  # Don't fail if broadcast fails

        stdout_handler = await self._create_output_handler(terminal_id, "stdout")
        stderr_handler = await self._create_output_handler(terminal_id, "stderr")

        try:
            sandbox_timeout = timeout if timeout is not None else 0

            if sandbox_timeout == 0:
                result = await self.sandbox.commands.run(
                    command,
                    cwd=working_dir,
                    envs=env,
                    on_stdout=stdout_handler,
                    on_stderr=stderr_handler,
                    timeout=0,
                )
            else:
                # Use timeout with buffer for asyncio
                result = await asyncio.wait_for(
                    self.sandbox.commands.run(
                        command,
                        cwd=working_dir,
                        envs=env,
                        on_stdout=stdout_handler,
                        on_stderr=stderr_handler,
                        timeout=sandbox_timeout,
                    ),
                    timeout=sandbox_timeout + 5,  # Give 5 seconds more than the sandbox timeout
                )

            self.outputs[terminal_id].append(
                {
                    "type": "exit",
                    "data": f"Process exited with code {result.exit_code}",
                    "exit_code": result.exit_code,
                    "timestamp": datetime.now(timezone.utc),
                }
            )

            if self.connection_manager:
                await self._broadcast_output(terminal_id, f"Process exited with code {result.exit_code}\n")

            if result.exit_code == 0:
                self._handle_cd_command(command, working_dir, terminal_info)

            return result.exit_code == 0

        except asyncio.TimeoutError:
            error_msg = f"Command execution timed out after {sandbox_timeout + 5} seconds"
            self.outputs[terminal_id].append(
                {"type": "stderr", "data": error_msg, "timestamp": datetime.now(timezone.utc)}
            )

            if self.connection_manager:
                await self._broadcast_output(terminal_id, error_msg)

            self.outputs[terminal_id].append(
                {
                    "type": "exit",
                    "data": "Process exited with code 1",
                    "exit_code": 1,
                    "timestamp": datetime.now(timezone.utc),
                }
            )

            if self.connection_manager:
                await self._broadcast_output(terminal_id, "Process exited with code 1\n")

            return False  # Command failed due to timeout

        except Exception as e:
            error_msg = f"Command failed: {str(e)}"
            self.outputs[terminal_id].append(
                {"type": "stderr", "data": error_msg, "timestamp": datetime.now(timezone.utc)}
            )

            if self.connection_manager:
                await self._broadcast_output(terminal_id, error_msg)

            self.outputs[terminal_id].append(
                {
                    "type": "exit",
                    "data": "Process exited with code 1",
                    "exit_code": 1,
                    "timestamp": datetime.now(timezone.utc),
                }
            )

            if self.connection_manager:
                await self._broadcast_output(terminal_id, "Process exited with code 1\n")

            return False  # Command failed

    async def send_input(
        self, terminal_id: str, text: str, submit: bool = True, command_id: Optional[str] = None
    ) -> bool:
        """
        Send input to an active tracked command in the sandbox.
        """
        if terminal_id not in self.terminals:
            raise KeyError(f"Terminal {terminal_id} not found")

        terminal_info = self.terminals[terminal_id]
        active_commands = terminal_info["active_commands"]

        if command_id is None:
            if not active_commands:
                raise ValueError(f"No active command is running in terminal {terminal_id}")
            if len(active_commands) > 1:
                raise ValueError(f"Multiple active commands are running in terminal {terminal_id}; provide command_id")
            command_id = next(iter(active_commands))

        command_info = self.command_history.get(command_id)
        if not command_info or command_info.get("terminal_id") != terminal_id:
            raise ValueError(f"Command ID {command_id} not found in terminal {terminal_id}")
        if command_info.get("status") != "running":
            raise ValueError(f"Command {command_id} is not running in terminal {terminal_id}")

        pid = command_info.get("pid")
        if pid is None:
            raise ValueError(f"Command {command_id} is not ready for input yet")

        payload = text
        if submit and not payload.endswith("\n"):
            payload += "\n"

        try:
            await self.sandbox.commands.send_stdin(pid, payload)
            return True
        except AttributeError as exc:
            raise ValueError("Sandbox command stdin is not supported by this E2B SDK version") from exc

    async def _broadcast_output(self, terminal_id: str, output: str):
        """Broadcast terminal output to connected clients."""
        if not self.connection_manager:
            return

        try:
            terminal_output_event = AgentEvent(
                event_type="terminal_output",
                sender="agent",
                content={
                    "output": output,
                    "terminal_id": terminal_id,
                    "thread_id": self.thread_id,
                },
            )
            await self.connection_manager.broadcast_event(terminal_output_event, self.workspace_id, self.thread_id)
        except Exception:
            # Don't let broadcast errors affect command execution
            pass

    async def send_command_tracked(
        self, terminal_id: str, command: str, purpose: Optional[str] = None, timeout: Optional[int] = None
    ) -> Optional[str]:
        """
        Send a command and return a command ID for tracking.

        Args:
            terminal_id: ID of the terminal to send command to
            command: The command to execute
            purpose: Optional description of the command's purpose
            timeout: Optional timeout in seconds (0 or None for no timeout)

        Returns:
            Command ID for tracking, or None if command couldn't be sent

        Raises:
            KeyError: If terminal doesn't exist
        """
        if terminal_id not in self.terminals:
            raise KeyError(f"Terminal {terminal_id} not found")

        self.command_counter += 1
        command_id = f"{terminal_id}_{self.command_counter}"

        start_time = datetime.now(timezone.utc)
        self.command_history[command_id] = {
            "command": command.strip(),
            "purpose": purpose,
            "terminal_id": terminal_id,
            "start_time": start_time,
            "status": "running",
            "return_code": None,
            "pid": None,
            "handle": None,
            "kill_requested": None,
        }
        self._output_accumulators[command_id] = TerminalOutputAccumulator(self._spill_store)

        self.terminals[terminal_id]["active_commands"][command_id] = self.command_history[command_id]

        terminal_info = self.terminals[terminal_id]
        working_dir = terminal_info["cwd"]

        self.outputs[terminal_id].append(
            {"type": "command", "data": command, "timestamp": datetime.now(timezone.utc), "purpose": purpose}
        )

        if self.connection_manager:
            await self._broadcast_output(terminal_id, f"$ {command}\n")

        task = asyncio.create_task(self._execute_command_async(command_id, terminal_id, command, working_dir, timeout))
        self._command_tasks[command_id] = task

        return command_id

    async def _execute_command_async(
        self, command_id: str, terminal_id: str, command: str, working_dir: str, timeout: Optional[int] = None
    ):
        """Execute a command asynchronously and track its status."""
        try:
            # Convert Path objects to strings for E2B compatibility
            if hasattr(working_dir, "__fspath__"):
                working_dir = str(working_dir)

            # Forward the terminal's environment to the sandbox command.
            env = self.terminals.get(terminal_id, {}).get("env") or {}

            # Create streaming output handlers
            stdout_handler = await self._create_output_handler(terminal_id, "stdout", command_id)
            stderr_handler = await self._create_output_handler(terminal_id, "stderr", command_id)

            sandbox_timeout = timeout if timeout is not None else 0

            # Execute with streaming and keep stdin open for interactive prompts.
            try:
                if sandbox_timeout == 0:
                    handle = await self.sandbox.commands.run(
                        command,
                        background=True,
                        cwd=working_dir,
                        envs=env,
                        on_stdout=stdout_handler,
                        on_stderr=stderr_handler,
                        stdin=True,
                        timeout=0,
                    )
                    self.command_history[command_id]["pid"] = handle.pid
                    self.command_history[command_id]["handle"] = handle
                    if self.command_history[command_id].get("kill_requested"):
                        await handle.kill()
                    result = await handle.wait()
                else:
                    handle = await self.sandbox.commands.run(
                        command,
                        background=True,
                        cwd=working_dir,
                        envs=env,
                        on_stdout=stdout_handler,
                        on_stderr=stderr_handler,
                        stdin=True,
                        timeout=sandbox_timeout,
                    )
                    self.command_history[command_id]["pid"] = handle.pid
                    self.command_history[command_id]["handle"] = handle
                    if self.command_history[command_id].get("kill_requested"):
                        await handle.kill()
                    result = await asyncio.wait_for(
                        handle.wait(),
                        timeout=sandbox_timeout + 5,  # Give 5 seconds more than the sandbox timeout
                    )
            except asyncio.TimeoutError:
                # If the sandbox itself times out or hangs
                raise Exception(f"Command execution timed out after {sandbox_timeout + 5} seconds")

            accumulator = self._output_accumulators.get(command_id)
            if accumulator is not None:
                accumulator.finalize()

            if self.command_history[command_id].get("kill_requested"):
                self.command_history[command_id]["status"] = "terminated"
            else:
                self.command_history[command_id]["status"] = "completed"
            self.command_history[command_id]["return_code"] = result.exit_code
            self.command_history[command_id]["end_time"] = datetime.now(timezone.utc)

            self.outputs[terminal_id].append(
                {
                    "type": "exit",
                    "data": f"Process exited with code {result.exit_code}",
                    "exit_code": result.exit_code,
                    "timestamp": datetime.now(timezone.utc),
                }
            )

            if self.connection_manager:
                await self._broadcast_output(terminal_id, f"Process exited with code {result.exit_code}\n")

            if result.exit_code == 0 and terminal_id in self.terminals:
                terminal_info = self.terminals[terminal_id]
                self._handle_cd_command(command, working_dir, terminal_info)

            if terminal_id in self.terminals:
                self.terminals[terminal_id]["active_commands"].pop(command_id, None)
        except Exception as e:
            self.command_history[command_id]["status"] = "failed"
            self.command_history[command_id]["return_code"] = 1
            self.command_history[command_id]["end_time"] = datetime.now(timezone.utc)

            error_msg = f"Command failed: {str(e)}"
            self.outputs[terminal_id].append(
                {"type": "stderr", "data": error_msg, "timestamp": datetime.now(timezone.utc)}
            )
            accumulator = self._output_accumulators.get(command_id)
            if accumulator is not None:
                accumulator.append_text(error_msg)
                accumulator.finalize()

            if self.connection_manager:
                await self._broadcast_output(terminal_id, error_msg)

            self.outputs[terminal_id].append(
                {
                    "type": "exit",
                    "data": "Process exited with code 1",
                    "exit_code": 1,
                    "timestamp": datetime.now(timezone.utc),
                }
            )

            if self.connection_manager:
                await self._broadcast_output(terminal_id, "Process exited with code 1\n")

            if terminal_id in self.terminals:
                self.terminals[terminal_id]["active_commands"].pop(command_id, None)
        finally:
            self._command_tasks.pop(command_id, None)

    def read_output(self, terminal_id: str, num_chars: int = 1024, offset: int = 0) -> str:
        """
        Read characters from a terminal's output buffer.

        Args:
            terminal_id: ID of the terminal to read output from
            num_chars: Number of characters to read (default: 1024).
            offset: Number of characters from the end to start reading from (default: 0).

        Returns:
            The requested characters from the terminal's output buffer.

        Raises:
            KeyError: If terminal doesn't exist
        """
        if terminal_id not in self.terminals:
            raise KeyError(f"Terminal {terminal_id} not found")

        full_output = ""
        for output in self.outputs[terminal_id]:
            if output["type"] == "command":
                full_output += f"$ {output['data']}\n"
            elif output["type"] in ["stdout", "stderr"]:
                full_output += output["data"]
                if not output["data"].endswith("\n"):
                    full_output += "\n"
            elif output["type"] == "exit":
                full_output += f"{output['data']}\n"

        # Apply offset and num_chars logic similar to LocalTerminalManager
        total_chars = len(full_output)

        if total_chars == 0:
            return ""

        if offset == 0:
            if total_chars <= num_chars:
                return full_output
            else:
                return full_output[-num_chars:]
        else:
            # With offset: read num_chars characters starting from (end - offset - num_chars)
            start_pos = max(0, total_chars - offset - num_chars)
            end_pos = max(0, total_chars - offset)

            if start_pos >= end_pos:
                return ""

            return full_output[start_pos:end_pos]

    def get_command_status(self, terminal_id: str, command_id: str) -> dict:
        """
        Get the status of a specific command.

        Args:
            terminal_id: ID of the terminal
            command_id: ID of the command to check

        Returns:
            Dictionary containing command status information

        Raises:
            KeyError: If terminal doesn't exist
        """
        if terminal_id not in self.terminals:
            raise KeyError(f"Terminal {terminal_id} not found")

        if command_id not in self.command_history:
            return {"status": "not_found"}

        command_info = self.command_history[command_id]

        if "end_time" in command_info:
            duration = (command_info["end_time"] - command_info["start_time"]).total_seconds()
        else:
            duration = (datetime.now(timezone.utc) - command_info["start_time"]).total_seconds()

        return {
            "status": command_info["status"],
            "command": command_info["command"],
            "purpose": command_info.get("purpose"),
            "duration": duration,
            "return_code": command_info.get("return_code"),
            "child_pids": [],
        }

    async def get_terminal_status(self, terminal_id: str) -> dict:
        """
        Get comprehensive status of a terminal including active commands.

        Args:
            terminal_id: ID of the terminal to check

        Returns:
            Dictionary containing terminal status and active commands

        Raises:
            KeyError: If terminal doesn't exist
        """
        if terminal_id not in self.terminals:
            raise KeyError(f"Terminal {terminal_id} not found")

        terminal_info = self.terminals[terminal_id]

        active_commands = {}
        for cmd_id, cmd_info in terminal_info["active_commands"].items():
            active_commands[cmd_id] = self.get_command_status(terminal_id, cmd_id)

        return {
            "running": True,  # Sandbox terminals are always "running"
            "ready_for_commands": len(active_commands) == 0,
            "active_commands": active_commands,
            "last_command": terminal_info.get("last_command", ""),
        }

    async def cleanup_all(self):
        """Clean up all terminals - useful for interrupt handling"""
        print(f"Cleaning up {len(self.terminals)} terminal(s)")

        terminal_ids = list(self.terminals.keys())
        for terminal_id in terminal_ids:
            try:
                await self.close_terminal(terminal_id)
                print(f"Closed terminal {terminal_id}")
            except Exception as e:
                print(f"Error closing terminal {terminal_id}: {e}")
        for accumulator in self._output_accumulators.values():
            accumulator.finalize()
        self.terminals.clear()
        self.outputs.clear()
        self.command_history.clear()
        self._output_accumulators.clear()
        self._command_tasks.clear()

    def _handle_output(self, terminal_id: str, data: str, stream: str):
        """Handle output from process."""
        self.outputs[terminal_id].append({"type": stream, "data": data, "timestamp": datetime.now(timezone.utc)})

    async def get_output(self, terminal_id: str, **kwargs) -> str:
        """
        Get output from a terminal.

        Args:
            terminal_id: ID of the terminal
            **kwargs: Optional filters (last_n_lines, since_timestamp, etc.)

        Returns:
            Terminal output as string

        Raises:
            KeyError: If terminal doesn't exist
        """
        if terminal_id not in self.terminals:
            raise KeyError(f"Terminal {terminal_id} not found")

        outputs = self.outputs[terminal_id]

        last_n_lines = kwargs.get("last_n_lines")
        since_timestamp = kwargs.get("since_timestamp")

        filtered_outputs = outputs

        if since_timestamp:
            filtered_outputs = [o for o in filtered_outputs if o["timestamp"] > since_timestamp]

        lines = []
        for output in filtered_outputs:
            if output["type"] in ["stdout", "stderr"]:
                lines.append(output["data"])
            elif output["type"] == "command":
                lines.append(f"$ {output['data']}")
            elif output["type"] == "exit":
                lines.append(output["data"])

        if last_n_lines and len(lines) > last_n_lines:
            lines = lines[-last_n_lines:]

        return "\n".join(lines)

    async def close_terminal(self, terminal_id: str) -> None:
        """
        Close a terminal.

        Args:
            terminal_id: ID of the terminal to close

        Raises:
            KeyError: If terminal doesn't exist
        """
        if terminal_id not in self.terminals:
            raise KeyError(f"Terminal {terminal_id} not found")

        for command_id in list(self.terminals[terminal_id]["active_commands"]):
            try:
                await self.kill_session(command_id, "TERM")
            except Exception:
                accumulator = self._output_accumulators.pop(command_id, None)
                if accumulator is not None:
                    accumulator.finalize()
        del self.terminals[terminal_id]
        del self.outputs[terminal_id]

    async def close_all(self) -> None:
        """Close all terminals."""
        terminal_ids = list(self.terminals.keys())
        for terminal_id in terminal_ids:
            await self.close_terminal(terminal_id)
        for accumulator in self._output_accumulators.values():
            accumulator.finalize()
        self._output_accumulators.clear()
        self._command_tasks.clear()

    async def list_terminals(self) -> Dict[str, Any]:
        """
        Get information about all terminals.

        Returns:
            Dictionary mapping terminal IDs to terminal info
        """
        result = {}
        for terminal_id, info in self.terminals.items():
            result[terminal_id] = {
                "created_at": info["created_at"].isoformat(),
                "cwd": info["cwd"],
                "has_running_process": info.get("process") is not None,
                "running": True,  # Match LocalTerminalManager format
                "last_command": info.get("last_command", ""),
            }
        return result
