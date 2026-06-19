"""Terminal manager implementation for sandbox environments."""

import uuid
import asyncio
import re
import os
import time
from typing import Any, Dict, Optional, Callable, Awaitable
from datetime import datetime, timezone

from ..services.base import ExecResult, TerminalManager
from ..services.terminal_buffer import cap_tokens, clamp_yield
from kolega_code.events import AgentEvent


class SandboxTerminalManager(TerminalManager):
    """Terminal manager that operates within a sandbox."""

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

        # Track commands and their status (for interface parity)
        self.command_history: Dict[str, Dict[str, Any]] = {}
        self.command_counter = 0

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

    def _render_delta(self, terminal_id: str, start_index: int, max_output_tokens: int):
        parts = []
        for entry in self.outputs.get(terminal_id, [])[start_index:]:
            if entry.get("type") in ("stdout", "stderr"):
                parts.append(entry.get("data", ""))
        return cap_tokens("".join(parts), max_output_tokens)

    def _exec_result(
        self, command_id: str, terminal_id: str, start_index: int, max_output_tokens: int, duration_ms: int
    ) -> ExecResult:
        capped = self._render_delta(terminal_id, start_index, max_output_tokens)
        info = self.command_history.get(command_id) or {}
        status = info.get("status")
        if status in ("completed", "failed", "terminated") or status is None:
            return ExecResult(
                status="exited",
                session_id=None,
                exit_code=info.get("return_code"),
                output=capped.text,
                truncated=capped.truncated,
                original_token_count=capped.original_token_count,
                duration_ms=duration_ms,
            )
        return ExecResult(
            status="running",
            session_id=command_id,
            exit_code=None,
            output=capped.text,
            truncated=capped.truncated,
            original_token_count=capped.original_token_count,
            duration_ms=duration_ms,
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
    ) -> ExecResult:
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

        start_index = len(self.outputs.get(terminal_id, []))
        start = time.monotonic()
        command_id = await self.send_command_tracked(terminal_id, command, timeout=0)
        if command_id is None:
            return ExecResult(status="exited", session_id=None, exit_code=1, output="Failed to start command")
        await self._wait_for_command(command_id, yield_ms)
        duration_ms = int((time.monotonic() - start) * 1000)
        return self._exec_result(command_id, terminal_id, start_index, max_output_tokens, duration_ms)

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
        terminal_id = info["terminal_id"]
        yield_ms = clamp_yield(yield_time_ms, poll=(chars == ""))
        start_index = len(self.outputs.get(terminal_id, []))
        start = time.monotonic()
        if chars:
            try:
                await self.send_input(terminal_id, chars, submit=False, command_id=session_id)
            except ValueError:
                # The command may have just finished; report its final status.
                pass
        await self._wait_for_command(session_id, yield_ms)
        duration_ms = int((time.monotonic() - start) * 1000)
        return self._exec_result(session_id, terminal_id, start_index, max_output_tokens, duration_ms)

    async def kill_session(self, session_id: str, signal: str = "TERM") -> ExecResult:
        info = self.command_history.get(session_id)
        if not info:
            raise KeyError(f"No such session: {session_id}")
        terminal_id = info["terminal_id"]
        start_index = len(self.outputs.get(terminal_id, []))
        pid = info.get("pid")
        handle = info.get("handle")
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
        if info.get("status") == "running":
            info["status"] = "terminated"
            if info.get("return_code") is None:
                info["return_code"] = 130 if signal == "INT" else 143
        active = self.terminals.get(terminal_id, {}).get("active_commands")
        if active is not None:
            active.pop(session_id, None)
        capped = self._render_delta(terminal_id, start_index, 10000)
        return ExecResult(
            status="exited",
            session_id=None,
            exit_code=info.get("return_code"),
            output=capped.text,
            truncated=capped.truncated,
            original_token_count=capped.original_token_count,
        )

    async def list_sessions(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for command_id, info in self.command_history.items():
            if info.get("status") == "running":
                terminal_id = info.get("terminal_id")
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
        # Check if this is a cd command
        # Match cd followed by path, stopping at ; or && or ||
        cd_match = re.match(r"^\s*cd\s+([^;&|]+)", command.strip())
        if not cd_match:
            return

        new_dir = cd_match.group(1).strip()

        # Remove quotes if present
        if (new_dir.startswith('"') and new_dir.endswith('"')) or (new_dir.startswith("'") and new_dir.endswith("'")):
            new_dir = new_dir[1:-1]

        # Handle relative and absolute paths
        if new_dir.startswith("/"):
            # Absolute path
            new_working_dir = new_dir
        elif new_dir == "..":
            # Parent directory
            new_working_dir = os.path.dirname(current_dir.rstrip("/"))
            if not new_working_dir:
                new_working_dir = "/"
        elif new_dir == ".":
            # Current directory (no change)
            new_working_dir = current_dir
        elif new_dir == "~":
            # Home directory
            new_working_dir = "/home/user"
        else:
            # Relative path
            new_working_dir = os.path.join(current_dir, new_dir)

        # Normalize the path (handle double slashes)
        new_working_dir = os.path.normpath(new_working_dir)
        # Ensure single leading slash for absolute paths
        if new_working_dir.startswith("//"):
            new_working_dir = new_working_dir[1:]

        # Update the terminal's stored working directory
        terminal_info["cwd"] = new_working_dir

    async def _create_output_handler(self, terminal_id: str, output_type: str) -> Callable[[str], Awaitable[None]]:
        """
        Create an async output handler for streaming.

        Args:
            terminal_id: ID of the terminal
            output_type: Type of output ('stdout' or 'stderr')

        Returns:
            Async callback function for handling output
        """

        async def handler(data: str) -> None:
            # Store output
            self.outputs[terminal_id].append(
                {"type": output_type, "data": data, "timestamp": datetime.now(timezone.utc)}
            )

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
                # Try to create the directory if it doesn't exist
                await self.sandbox.commands.run(f"test -d {working_dir} || mkdir -p {working_dir}")
            except Exception:
                # If we can't create it, fall back to /home/user
                working_dir = "/home/user"

        try:
            # Determine timeout settings
            sandbox_timeout = timeout if timeout is not None else 60  # Default to 60s for backward compatibility

            # For utility commands, we don't need streaming
            if sandbox_timeout == 0:
                # No timeout - let it run indefinitely
                result = await self.sandbox.commands.run(command, cwd=working_dir, timeout=0)
            else:
                # Use timeout with buffer for asyncio
                result = await asyncio.wait_for(
                    self.sandbox.commands.run(command, cwd=working_dir, timeout=sandbox_timeout),
                    timeout=sandbox_timeout + 5,  # Give 5 seconds more than the sandbox timeout
                )

            # Return the combined output (stdout + stderr)
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

        # Extract terminal options
        cwd = terminal_kwargs.get("cwd", "/home/user/workspace")
        env = terminal_kwargs.get("env", {})

        # Convert Path objects to strings for E2B compatibility
        if hasattr(cwd, "__fspath__"):  # Check if it's a Path-like object
            cwd = str(cwd)

        # Ensure the directory exists (try to create if it doesn't)
        try:
            # Check if directory exists
            await self.sandbox.commands.run(f"test -d {cwd}")
        except Exception:
            # Directory doesn't exist, try to create it
            try:
                await self.sandbox.commands.run(f"mkdir -p {cwd}")
                # Directory now exists (either already existed or was created)
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
            "active_commands": {},  # Track commands for this terminal
        }
        self.outputs[terminal_id] = []

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

        # Update last command info
        terminal_info["last_command"] = command.rstrip("\n")  # Strip trailing newline for consistency
        terminal_info["last_command_purpose"] = purpose or ""

        # Use terminal's working directory
        working_dir = terminal_info["cwd"]

        # Convert Path objects to strings for E2B compatibility
        if hasattr(working_dir, "__fspath__"):  # Check if it's a Path-like object
            working_dir = str(working_dir)

        # Store command in output
        self.outputs[terminal_id].append(
            {"type": "command", "data": command, "timestamp": datetime.now(timezone.utc), "purpose": purpose}
        )

        # Broadcast command if connection manager available
        if self.connection_manager:
            try:
                await self._broadcast_output(terminal_id, f"$ {command}\n")
            except Exception:
                pass  # Don't fail if broadcast fails

        # Create streaming output handlers
        stdout_handler = await self._create_output_handler(terminal_id, "stdout")
        stderr_handler = await self._create_output_handler(terminal_id, "stderr")

        # Execute command in sandbox with streaming
        try:
            # Determine timeout settings
            sandbox_timeout = timeout if timeout is not None else 0  # Default to no timeout

            # If no timeout requested (0), don't use asyncio.wait_for
            if sandbox_timeout == 0:
                result = await self.sandbox.commands.run(
                    command,
                    cwd=working_dir,
                    on_stdout=stdout_handler,
                    on_stderr=stderr_handler,
                    timeout=0,  # No timeout for sandbox
                )
            else:
                # Use timeout with buffer for asyncio
                result = await asyncio.wait_for(
                    self.sandbox.commands.run(
                        command,
                        cwd=working_dir,
                        on_stdout=stdout_handler,
                        on_stderr=stderr_handler,
                        timeout=sandbox_timeout,
                    ),
                    timeout=sandbox_timeout + 5,  # Give 5 seconds more than the sandbox timeout
                )

            # Store exit code
            self.outputs[terminal_id].append(
                {
                    "type": "exit",
                    "data": f"Process exited with code {result.exit_code}",
                    "exit_code": result.exit_code,
                    "timestamp": datetime.now(timezone.utc),
                }
            )

            # Broadcast exit status
            if self.connection_manager:
                await self._broadcast_output(terminal_id, f"Process exited with code {result.exit_code}\n")

            # If it was a successful cd command, update the terminal's working directory
            if result.exit_code == 0:
                self._handle_cd_command(command, working_dir, terminal_info)

            return result.exit_code == 0  # Return True only if command succeeded

        except asyncio.TimeoutError:
            # Handle timeout specifically
            error_msg = f"Command execution timed out after {sandbox_timeout + 5} seconds"
            self.outputs[terminal_id].append(
                {"type": "stderr", "data": error_msg, "timestamp": datetime.now(timezone.utc)}
            )

            # Broadcast error
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

            # Broadcast exit status
            if self.connection_manager:
                await self._broadcast_output(terminal_id, "Process exited with code 1\n")

            return False  # Command failed due to timeout

        except Exception as e:
            # Store error
            error_msg = f"Command failed: {str(e)}"
            self.outputs[terminal_id].append(
                {"type": "stderr", "data": error_msg, "timestamp": datetime.now(timezone.utc)}
            )

            # Broadcast error
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

            # Broadcast exit status
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

        # Generate command ID
        self.command_counter += 1
        command_id = f"{terminal_id}_{self.command_counter}"

        # Record command in history
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
        }

        # Also track in terminal's active commands
        self.terminals[terminal_id]["active_commands"][command_id] = self.command_history[command_id]

        # Get terminal info
        terminal_info = self.terminals[terminal_id]
        working_dir = terminal_info["cwd"]

        # Store command in output
        self.outputs[terminal_id].append(
            {"type": "command", "data": command, "timestamp": datetime.now(timezone.utc), "purpose": purpose}
        )

        # Broadcast command
        if self.connection_manager:
            await self._broadcast_output(terminal_id, f"$ {command}\n")

        # Start command execution asynchronously without waiting
        asyncio.create_task(self._execute_command_async(command_id, terminal_id, command, working_dir, timeout))

        return command_id

    async def _execute_command_async(
        self, command_id: str, terminal_id: str, command: str, working_dir: str, timeout: Optional[int] = None
    ):
        """Execute a command asynchronously and track its status."""
        try:
            # Convert Path objects to strings for E2B compatibility
            if hasattr(working_dir, "__fspath__"):  # Check if it's a Path-like object
                working_dir = str(working_dir)

            # Create streaming output handlers
            stdout_handler = await self._create_output_handler(terminal_id, "stdout")
            stderr_handler = await self._create_output_handler(terminal_id, "stderr")

            # Determine timeout settings
            sandbox_timeout = timeout if timeout is not None else 0  # Default to no timeout

            # Execute with streaming and keep stdin open for interactive prompts.
            try:
                if sandbox_timeout == 0:
                    handle = await self.sandbox.commands.run(
                        command,
                        background=True,
                        cwd=working_dir,
                        on_stdout=stdout_handler,
                        on_stderr=stderr_handler,
                        stdin=True,
                        timeout=0,
                    )
                    self.command_history[command_id]["pid"] = handle.pid
                    self.command_history[command_id]["handle"] = handle
                    result = await handle.wait()
                else:
                    handle = await self.sandbox.commands.run(
                        command,
                        background=True,
                        cwd=working_dir,
                        on_stdout=stdout_handler,
                        on_stderr=stderr_handler,
                        stdin=True,
                        timeout=sandbox_timeout,
                    )
                    self.command_history[command_id]["pid"] = handle.pid
                    self.command_history[command_id]["handle"] = handle
                    result = await asyncio.wait_for(
                        handle.wait(),
                        timeout=sandbox_timeout + 5,  # Give 5 seconds more than the sandbox timeout
                    )
            except asyncio.TimeoutError:
                # If the sandbox itself times out or hangs
                raise Exception(f"Command execution timed out after {sandbox_timeout + 5} seconds")

            # Command completed
            self.command_history[command_id]["status"] = "completed"
            self.command_history[command_id]["return_code"] = result.exit_code
            self.command_history[command_id]["end_time"] = datetime.now(timezone.utc)

            # Store exit code
            self.outputs[terminal_id].append(
                {
                    "type": "exit",
                    "data": f"Process exited with code {result.exit_code}",
                    "exit_code": result.exit_code,
                    "timestamp": datetime.now(timezone.utc),
                }
            )

            # Broadcast exit status
            if self.connection_manager:
                await self._broadcast_output(terminal_id, f"Process exited with code {result.exit_code}\n")

            # If it was a successful cd command, update the terminal's working directory
            if result.exit_code == 0 and terminal_id in self.terminals:
                terminal_info = self.terminals[terminal_id]
                self._handle_cd_command(command, working_dir, terminal_info)

            # Remove from active commands
            if terminal_id in self.terminals:
                self.terminals[terminal_id]["active_commands"].pop(command_id, None)

        except Exception as e:
            # Command failed
            self.command_history[command_id]["status"] = "failed"
            self.command_history[command_id]["return_code"] = 1
            self.command_history[command_id]["end_time"] = datetime.now(timezone.utc)

            # Store error
            error_msg = f"Command failed: {str(e)}"
            self.outputs[terminal_id].append(
                {"type": "stderr", "data": error_msg, "timestamp": datetime.now(timezone.utc)}
            )

            # Broadcast error
            if self.connection_manager:
                await self._broadcast_output(terminal_id, error_msg)

            # Store exit info
            self.outputs[terminal_id].append(
                {
                    "type": "exit",
                    "data": "Process exited with code 1",
                    "exit_code": 1,
                    "timestamp": datetime.now(timezone.utc),
                }
            )

            # Broadcast exit status
            if self.connection_manager:
                await self._broadcast_output(terminal_id, "Process exited with code 1\n")

            # Remove from active commands
            if terminal_id in self.terminals:
                self.terminals[terminal_id]["active_commands"].pop(command_id, None)

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

        # Reconstruct full output from stored outputs
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
            # Default behavior: read last num_chars characters
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

        # Calculate duration
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
            "child_pids": [],  # No child process tracking in sandbox
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

        # Get active commands for this terminal
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
        self.terminals.clear()
        self.outputs.clear()
        self.command_history.clear()

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

        # Apply filters if provided
        last_n_lines = kwargs.get("last_n_lines")
        since_timestamp = kwargs.get("since_timestamp")

        filtered_outputs = outputs

        if since_timestamp:
            filtered_outputs = [o for o in filtered_outputs if o["timestamp"] > since_timestamp]

        # Convert to string
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

        # Clean up
        del self.terminals[terminal_id]
        del self.outputs[terminal_id]

    async def close_all(self) -> None:
        """Close all terminals."""
        terminal_ids = list(self.terminals.keys())
        for terminal_id in terminal_ids:
            await self.close_terminal(terminal_id)

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
