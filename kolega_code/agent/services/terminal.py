import asyncio
import fcntl
import os
import platform
import psutil
import pty
import re
import select
import signal
import time
import uuid
from typing import Dict, List, Optional

from ..connection_manager import AgentConnectionManager
from ..models.public import AgentEvent
from .base import TerminalManager


class AsyncPersistentTerminal:
    """
    An asynchronous class for maintaining a persistent terminal process that can receive commands
    and return output over time.
    """

    COMMAND_MONITOR_TIMEOUT_SECONDS = 300

    def __init__(
        self,
        workspace_id: str,
        thread_id: str,
        terminal_id: str,
        connection_manager: AgentConnectionManager,
        terminal_cmd: Optional[List[str]] = None,
        cwd: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
        auto_activate_venv: bool = True,
    ):
        """
        Initialize a persistent terminal process.

        Args:
            terminal: Command to start the terminal. Defaults to bash/cmd.exe based on platform.
            cwd: Working directory for the terminal. Defaults to current directory.
            env: Environment variables for the terminal. Defaults to current environment.
        """
        self.terminal_id = terminal_id
        self.workspace_id = workspace_id
        self.thread_id = thread_id
        self.connection_manager = connection_manager
        self.auto_activate_venv = auto_activate_venv

        # Check platform - this implementation only works on Unix-like systems
        if platform.system() == "Windows":
            raise RuntimeError("PTY-based shell is not supported on Windows")

        # Attributes to be set during startup
        self.process = None
        self.master_fd = None
        self.slave_fd = None
        self.is_running = False
        self.shell_cleaned = False
        self.pid = None

        # Store initialization parameters
        if terminal_cmd is None:
            # Try to find the best shell available
            for shell in ["/bin/zsh", "/bin/bash", "/bin/sh"]:
                if os.path.exists(shell):
                    self.terminal_cmd = [shell]
                    break
            else:
                self.terminal_cmd = ["/bin/sh", "-f"]  # Fallback to /bin/sh
        else:
            self.terminal_cmd = terminal_cmd

        self.cwd = cwd

        if env is None:
            self.env = os.environ.copy()
        else:
            self.env = env

        # Force environment to use unbuffered Python output
        self.env["PYTHONUNBUFFERED"] = "1"
        # Set TERM to a simple terminal type
        self.env["TERM"] = "xterm"
        self.env["PROMPT"] = "$ "

        # Buffer for output that hasn't been read yet
        self.output_buffer = bytearray()

        # Buffer that keeps all output (doesn't get cleared)
        self.persistent_output_buffer = bytearray()

        # Tracking the last time output was received
        self.last_output_time = 0

        # Track the last command sent to the terminal
        self.last_command = ""
        self.last_command_purpose = ""

        # Command tracking for process-based completion detection
        self.active_commands = {}  # command_id -> command_info
        self.command_counter = 0
        self.shell_prompt_detected = True  # Track if shell is ready for new commands

    def _strip_ansi_codes(self, text: str) -> str:
        """
        Remove ANSI escape sequences (color/formatting codes) from text.

        Args:
            text: The text containing ANSI codes

        Returns:
            Text with ANSI codes removed
        """
        ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
        return ansi_escape.sub("", text)

    async def start(self) -> None:
        """Start the terminal process."""
        self.pid, self.master_fd = pty.fork()

        if self.pid == 0:
            # Child process - execute the shell
            if self.cwd:
                os.chdir(self.cwd)

            # Replace the child process with the shell
            os.execvpe(self.terminal_cmd[0], self.terminal_cmd, self.env)
        else:
            # Parent process
            # Set master to non-blocking mode
            flags = fcntl.fcntl(self.master_fd, fcntl.F_GETFL)
            fcntl.fcntl(self.master_fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)

            self.is_running = True
            self.last_output_time = time.time()

            # Start the background task to read output
            self._read_task = asyncio.create_task(self._read_output())

            # Wait a bit for the shell to initialize
            await asyncio.sleep(0.2)

        if self.auto_activate_venv:
            activation_script = await self.detect_venv()
            if activation_script:
                await self.send_command(activation_script)

        # Clean up the shell
        await self.send_command("unsetopt prompt_cr prompt_sp")  # Disable prompt % marker
        await self.send_command("unsetopt zle")  # Turn off zsh line editing
        await self.send_command('PS1=""')  # Set empty prompt
        await self.send_command('PROMPT=""')  # Alternative way to set prompt
        await self.send_command('RPROMPT=""')
        await self.get_new_output(inactivity_timeout=0.5)
        self.shell_cleaned = True

        # Clear last_command after shell initialization
        self.last_command = ""
        self.last_command_purpose = ""

        self.is_running = True
        self.last_output_time = time.time()

        # Start the background task to read output
        self._read_task = asyncio.create_task(self._read_output())

    async def detect_venv(self) -> str:
        """
        Detect if a Python virtual environment exists in the project directory.

        Returns:
            Path to the activation script if a virtual environment was found, empty string otherwise
        """
        # Common virtual environment directory names
        venv_dirs = [".venv", "venv", "env", ".env"]

        for venv_dir in venv_dirs:
            # Check if the virtual environment directory exists
            venv_path = os.path.join(str(self.cwd), venv_dir)

            # Check if directory exists
            if not os.path.isdir(venv_path):
                continue

            # Check for the activation script based on OS
            activate_script = os.path.join(venv_path, "bin", "activate")
            windows_script = os.path.join(venv_path, "Scripts", "activate")

            if os.path.isfile(activate_script):
                return f"source {activate_script}"

            if os.path.isfile(windows_script):
                return f"source {windows_script}"

        return ""

    async def _read_output(self) -> None:
        """Background task to continuously read from process output."""
        READ_SIZE = 1024

        while self.is_running:
            # Use asyncio-friendly way to check for data
            await asyncio.sleep(0.01)  # Small sleep to prevent CPU thrashing

            try:
                # Try a non-blocking read
                r, _, _ = select.select([self.master_fd], [], [], 0)
                if self.master_fd in r:
                    try:
                        data = os.read(self.master_fd, READ_SIZE)
                        if data:
                            self.last_output_time = time.time()
                            self.output_buffer.extend(data)
                            self.persistent_output_buffer.extend(data)

                            data_str = bytes(data).decode("utf-8", errors="replace")

                            if self.shell_cleaned:
                                output_ansi_stripped = self._strip_ansi_codes(data_str)
                                terminal_output_event = AgentEvent(
                                    event_type="terminal_output",
                                    sender="agent",
                                    content={
                                        "output": output_ansi_stripped,
                                        "terminal_id": self.terminal_id,
                                        "thread_id": self.thread_id,
                                    },
                                )
                                await self.connection_manager.broadcast_event(
                                    terminal_output_event, self.workspace_id, self.thread_id
                                )
                        else:
                            # EOF - process has exited
                            self.is_running = False
                            break
                    except (OSError, IOError):
                        # Check if process has exited
                        try:
                            pid, status = os.waitpid(self.pid, os.WNOHANG)
                            if pid == self.pid:
                                self.is_running = False
                                break
                        except ChildProcessError:
                            self.is_running = False
                            break

                        # If read error but process still running, just continue
                        continue
            except Exception:
                self.is_running = False
                break

        # Process has terminated
        self.is_running = False

    async def send_command(self, command: str, purpose: Optional[str] = None) -> bool:
        """
        Send a command to the terminal.

        Args:
            command: The command to execute

        Returns:
            True if command was sent successfully, False if process is not running
        """
        if not self.is_running or self.master_fd is None:
            return False

        if not command.endswith("\n"):
            command += "\n"

        try:
            os.write(self.master_fd, command.encode())
            self.last_command = command
            self.last_command_purpose = purpose
            return True
        except (BrokenPipeError, ConnectionResetError, OSError):
            self.is_running = False
            return False

    async def send_input(self, text: str, submit: bool = True) -> bool:
        """
        Send raw input to the foreground process without changing command tracking.
        """
        if not self.is_running or self.master_fd is None:
            return False

        payload = text
        if submit and not payload.endswith("\n"):
            payload += "\n"

        try:
            os.write(self.master_fd, payload.encode())
            return True
        except (BrokenPipeError, ConnectionResetError, OSError):
            self.is_running = False
            return False

    async def get_new_output(self, inactivity_timeout: float = 2.0, max_wait: Optional[float] = 30) -> str:
        """
        Get output that hasn't been read yet, waiting until the terminal stops producing
        output for inactivity_timeout seconds or until max_wait seconds have passed.

        Args:
            inactivity_timeout: Seconds of inactivity that signals completion
            max_wait: Maximum seconds to wait overall

        Returns:
            Any new output from the terminal
        """
        start_time = time.time()

        while True:
            current_time = time.time()

            # Check for process termination
            if not self.is_running and len(self.output_buffer) == 0:
                break

            # Check if we've been inactive long enough to stop
            inactive_time = current_time - self.last_output_time
            if len(self.output_buffer) > 0 and inactive_time >= inactivity_timeout:
                break

            # Check if we've exceeded the maximum wait time
            if max_wait is not None and current_time - start_time >= max_wait:
                break

            # Wait a bit before checking again
            await asyncio.sleep(0.1)

        # Get and clear the current buffer
        output = bytes(self.output_buffer).decode("utf-8", errors="replace")
        self.output_buffer.clear()

        return output

    def read_output(self, num_chars: int = 1024, offset: int = 0) -> str:
        """
        Read characters from the persistent output buffer.

        Args:
            num_chars: Number of characters to read (default: 1024).
                      If buffer is smaller than num_chars, returns entire buffer.
            offset: Number of characters from the end to start reading from (default: 0).
                   If offset is 0, reads the last num_chars characters.
                   If offset is > 0, reads num_chars characters starting from that offset from the end.

        Returns:
            The requested characters from the output buffer as a UTF-8 decoded string.
        """
        # Get the length of the buffer
        buffer_length = len(self.persistent_output_buffer)

        if buffer_length == 0:
            return ""

        # First decode the entire buffer to get proper character boundaries
        full_output = bytes(self.persistent_output_buffer).decode("utf-8", errors="replace")
        total_chars = len(full_output)

        if total_chars == 0:
            return ""

        # Calculate start and end positions
        if offset == 0:
            # Default behavior: read last num_chars characters
            if total_chars <= num_chars:
                return full_output
            else:
                return full_output[-num_chars:]
        else:
            # With offset: read num_chars characters starting from (end - offset - num_chars)
            # This means we skip the last 'offset' characters and read 'num_chars' before that
            start_pos = max(0, total_chars - offset - num_chars)
            end_pos = max(0, total_chars - offset)

            # Ensure we don't have negative ranges
            if start_pos >= end_pos:
                return ""

            return full_output[start_pos:end_pos]

    async def is_alive(self) -> bool:
        """Check if the terminal process is still running."""
        if self.pid is None:
            return False

        try:
            # Check if process is still running
            pid, status = os.waitpid(self.pid, os.WNOHANG)
            if pid == self.pid:
                self.is_running = False
        except ChildProcessError:
            self.is_running = False

        return self.is_running

    async def close(self) -> None:
        """Close the terminal process and clean up resources."""
        if not self.is_running or self.pid is None:
            return

        self.is_running = False

        # Try to terminate the process gracefully
        try:
            os.kill(self.pid, signal.SIGTERM)
            # Give it a moment to terminate
            await asyncio.sleep(0.5)

            # Force kill if still running
            if await self.is_alive():
                os.kill(self.pid, signal.SIGKILL)
        except ProcessLookupError:
            # Process is already gone
            pass

        # Close the master FD
        if self.master_fd is not None:
            os.close(self.master_fd)
            self.master_fd = None

        # Cancel the read task
        if hasattr(self, "_read_task") and not self._read_task.done():
            self._read_task.cancel()
            try:
                await self._read_task
            except asyncio.CancelledError:
                pass

    async def send_command_tracked(self, command: str, purpose: Optional[str] = None) -> Optional[str]:
        """
        Send a command and return a unique command ID for tracking.

        Args:
            command: The command to execute
            purpose: Optional description of the command's purpose

        Returns:
            Command ID for tracking, or None if command couldn't be sent
        """
        if not self.is_running or self.master_fd is None:
            return None

        self.command_counter += 1
        command_id = f"{self.terminal_id}_{self.command_counter}"

        # Record the command
        self.active_commands[command_id] = {
            "command": command.strip(),
            "purpose": purpose,
            "start_time": time.time(),
            "status": "running",
            "child_pids": set(),
            "return_code": None,
        }

        # Send the command
        success = await self.send_command(command, purpose)
        if success:
            self.shell_prompt_detected = False
            # Start monitoring for completion
            asyncio.create_task(self._monitor_command_completion(command_id))
            return command_id
        else:
            del self.active_commands[command_id]
            return None

    async def _monitor_command_completion(self, command_id: str):
        """Monitor a command for completion by checking child processes."""
        command_info = self.active_commands.get(command_id)
        if not command_info:
            return

        # Initial delay to let command start
        await asyncio.sleep(0.1)

        max_monitor_time = self.COMMAND_MONITOR_TIMEOUT_SECONDS
        check_interval = 0.2  # Check every 200ms for faster response
        consecutive_no_children = 0

        start_time = time.time()

        while command_info["status"] == "running":
            current_time = time.time()

            # Timeout protection - don't monitor forever (>= so a zero timeout
            # deterministically times out on the first check)
            if current_time - start_time >= max_monitor_time:
                command_info["status"] = "monitor_timeout"
                command_info["return_code"] = None
                command_info["monitor_timeout_seconds"] = max_monitor_time
                break

            try:
                # Get current child processes of the shell
                try:
                    shell_process = psutil.Process(self.pid)
                    current_children = {child.pid for child in shell_process.children(recursive=True)}
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    # Shell process might have died
                    command_info["status"] = "terminated"
                    command_info["return_code"] = -1
                    break

                # Update tracked child PIDs
                command_info["child_pids"].update(current_children)

                # If no children, start counting consecutive checks
                if not current_children:
                    consecutive_no_children += 1

                    # If we've had no children for several consecutive checks, likely done
                    # Use shorter requirement for quick commands
                    required_consecutive = 3 if current_time - start_time < 5 else 2

                    if consecutive_no_children >= required_consecutive:
                        # Double-check with prompt detection if we can
                        prompt_detected = self._check_for_shell_prompt()

                        # Mark as completed if:
                        # 1. We have no children for several checks AND
                        # 2. Either we detect a prompt OR enough time has passed
                        if prompt_detected or consecutive_no_children >= 5:
                            command_info["status"] = "completed"
                            command_info["return_code"] = 0  # Assume success
                            self.shell_prompt_detected = True
                            break
                else:
                    # Reset counter if we see children again
                    consecutive_no_children = 0

            except Exception:
                # If we can't monitor, assume command is still running
                # But don't let it run forever
                if current_time - start_time > 30:  # 30 second fallback
                    command_info["status"] = "completed"
                    command_info["return_code"] = 0
                    break

            await asyncio.sleep(check_interval)

    def _check_for_shell_prompt(self) -> bool:
        """Check if the recent output contains a shell prompt pattern."""
        recent_output = self.read_output(200)  # Last 200 chars

        # Look for common prompt patterns - be more inclusive to avoid hanging
        prompt_patterns = [
            r"\$\s*$",  # $ at end (bash)
            r">\s*$",  # > at end
            r"#\s*$",  # # at end (root)
            r"%\s*$",  # % at end (zsh)
            r"❯\s*$",  # Fish shell prompt
            r"➜.*$",  # Oh My Zsh arrow prompts
            r"»\s*$",  # Custom prompt with »
            r"λ\s*$",  # Lambda prompt
            r"⚡\s*$",  # Lightning prompt
            r"\]\s*\$\s*$",  # [user@host dir]$ pattern
            r"\)\s*\$\s*$",  # (venv) $ pattern
            r":\s*\$\s*$",  # dir:$ pattern
            r"~\s*\$\s*$",  # ~$ pattern
            r"\w+\s*\$\s*$",  # word$ pattern (simplified)
        ]

        for pattern in prompt_patterns:
            if re.search(pattern, recent_output.strip(), re.MULTILINE):
                return True
        return False

    def get_command_status(self, command_id: str) -> dict:
        """Get the status of a specific command."""
        if command_id not in self.active_commands:
            return {"status": "not_found"}

        command_info = self.active_commands[command_id]
        return {
            "status": command_info["status"],
            "command": command_info["command"],
            "purpose": command_info.get("purpose"),
            "duration": time.time() - command_info["start_time"],
            "return_code": command_info.get("return_code"),
            "monitor_timeout_seconds": command_info.get("monitor_timeout_seconds"),
            "child_pids": list(command_info["child_pids"]),
        }

    def get_active_commands(self) -> dict:
        """Get all currently active commands."""
        return {
            cmd_id: self.get_command_status(cmd_id)
            for cmd_id in self.active_commands
            if self.active_commands[cmd_id]["status"] in {"running", "monitor_timeout"}
        }

    def is_ready_for_commands(self) -> bool:
        """Check if the terminal is ready to accept new commands."""
        return self.shell_prompt_detected and len(self.get_active_commands()) == 0


class LocalTerminalManager(TerminalManager):
    """
    A manager class that helps handle multiple terminal instances.
    """

    def __init__(self, workspace_id: str, thread_id: str, connection_manager: AgentConnectionManager):
        """Initialize an empty terminal manager."""
        self.terminals: Dict[str, AsyncPersistentTerminal] = {}
        self.workspace_id = workspace_id
        self.thread_id = thread_id
        self.connection_manager = connection_manager

    async def get_last_command(self, terminal_id: str) -> str:
        if terminal_id not in self.terminals:
            raise KeyError(f"Terminal with ID {terminal_id} not found")

        # Strip trailing newline for consistency
        return self.terminals[terminal_id].last_command.rstrip("\n")

    async def get_last_command_purpose(self, terminal_id: str) -> str:
        if terminal_id not in self.terminals:
            raise KeyError(f"Terminal with ID {terminal_id} not found")

        return self.terminals[terminal_id].last_command_purpose

    async def launch_terminal(self, terminal_id: Optional[str] = None, **terminal_kwargs) -> str:
        """
        Create a new terminal instance with the given ID or generate a random ID.

        Args:
            terminal_id: Optional identifier for the terminal (random UUID if not provided)
            **terminal_kwargs: Arguments to pass to AsyncPersistentTerminal constructor

        Returns:
            The ID of the created terminal
        """
        # Generate a random ID if none provided
        if terminal_id is None:
            terminal_id = str(uuid.uuid4())

        # Create new terminal instance
        term = AsyncPersistentTerminal(
            workspace_id=self.workspace_id,
            thread_id=self.thread_id,
            terminal_id=terminal_id,
            connection_manager=self.connection_manager,
            **terminal_kwargs,
        )
        await term.start()

        # Store in our dictionary
        self.terminals[terminal_id] = term
        return terminal_id

    async def send_command(
        self, term_id: str, command: str, purpose: Optional[str] = None, timeout: Optional[int] = None
    ) -> bool:
        """
        Send a command to a specific terminal.

        Args:
            term_id: ID of the terminal to send command to
            command: The command to execute
            purpose: Optional description of command purpose
            timeout: Optional timeout in seconds (not implemented for local terminals)

        Returns:
            True if command was sent successfully

        Raises:
            KeyError: If terminal_id doesn't exist
        """
        if term_id not in self.terminals:
            raise KeyError(f"Terminal with ID {term_id} not found")

        # Note: timeout is not implemented for local terminals as they use persistent shell sessions
        return await self.terminals[term_id].send_command(command, purpose=purpose)

    async def send_input(
        self, term_id: str, text: str, submit: bool = True, command_id: Optional[str] = None
    ) -> bool:
        """
        Send input to a running command in a local terminal.
        """
        if term_id not in self.terminals:
            raise KeyError(f"Terminal with ID {term_id} not found")

        terminal = self.terminals[term_id]
        active_commands = terminal.get_active_commands()

        if command_id is not None:
            status = terminal.get_command_status(command_id)
            if status["status"] == "not_found":
                raise ValueError(f"Command ID {command_id} not found in terminal {term_id}")
            if status["status"] not in {"running", "monitor_timeout"}:
                raise ValueError(f"Command {command_id} is not running in terminal {term_id}")
        elif not active_commands:
            raise ValueError(f"No active command is running in terminal {term_id}")
        elif len(active_commands) > 1:
            raise ValueError(f"Multiple active commands are running in terminal {term_id}; provide command_id")

        return await terminal.send_input(text, submit=submit)

    async def get_output(self, terminal_id: str, **kwargs) -> str:
        """
        Get output from a specific terminal.

        Args:
            terminal_id: ID of the terminal to get output from
            **kwargs: Arguments to pass to get_new_output

        Returns:
            Output from the specified terminal

        Raises:
            KeyError: If terminal_id doesn't exist
        """
        if terminal_id not in self.terminals:
            raise KeyError(f"Terminal with ID {terminal_id} not found")

        return await self.terminals[terminal_id].get_new_output(**kwargs)

    def read_output(self, terminal_id: str, num_chars: int = 1024, offset: int = 0) -> str:
        """
        Read characters from a terminal's persistent output buffer.

        Args:
            terminal_id: ID of the terminal to read output from
            num_chars: Number of characters to read (default: 1024).
                      If buffer is smaller than num_chars, returns entire buffer.
            offset: Number of characters from the end to start reading from (default: 0).
                   If offset is 0, reads the last num_chars characters.
                   If offset is > 0, reads num_chars characters starting from that offset from the end.

        Returns:
            The requested characters from the terminal's output buffer as a UTF-8 decoded string.

        Raises:
            KeyError: If terminal_id doesn't exist
        """
        if terminal_id not in self.terminals:
            raise KeyError(f"Terminal with ID {terminal_id} not found")

        return self.terminals[terminal_id].read_output(num_chars=num_chars, offset=offset)

    async def close_terminal(self, terminal_id: str) -> None:
        """
        Close a specific terminal.

        Args:
            terminal_id: ID of the terminal to close

        Raises:
            KeyError: If terminal_id doesn't exist
        """
        if terminal_id not in self.terminals:
            raise KeyError(f"Terminal with ID {terminal_id} not found.")

        await self.terminals[terminal_id].close()
        del self.terminals[terminal_id]

    async def close_all(self) -> None:
        """Close all terminal instances."""
        # Get a copy of keys to avoid modification during iteration
        term_ids = list(self.terminals.keys())
        for term_id in term_ids:
            await self.close_terminal(term_id)

    async def list_terminals(self) -> Dict[str, bool]:
        """
        Get a dictionary of all terminals IDs and their running status.

        Returns:
            Dictionary mapping terminal IDs to boolean running status
        """
        result = {}
        for term_id, term in self.terminals.items():
            result[term_id] = {"running": await term.is_alive(), "last_command": term.last_command}
        return result

    async def send_command_tracked(
        self, term_id: str, command: str, purpose: Optional[str] = None, timeout: Optional[int] = None
    ) -> Optional[str]:
        """
        Send a command and return a command ID for tracking.

        Args:
            term_id: ID of the terminal to send command to
            command: The command to execute
            purpose: Optional description of the command's purpose
            timeout: Optional timeout in seconds (not implemented for local terminals)

        Returns:
            Command ID for tracking, or None if command couldn't be sent

        Raises:
            KeyError: If terminal_id doesn't exist
        """
        if term_id not in self.terminals:
            raise KeyError(f"Terminal with ID {term_id} not found")

        # Note: timeout is not implemented for local terminals as they use persistent shell sessions
        return await self.terminals[term_id].send_command_tracked(command, purpose)

    def get_command_status(self, term_id: str, command_id: str) -> dict:
        """
        Get the status of a specific command.

        Args:
            term_id: ID of the terminal
            command_id: ID of the command to check

        Returns:
            Dictionary containing command status information

        Raises:
            KeyError: If terminal_id doesn't exist
        """
        if term_id not in self.terminals:
            raise KeyError(f"Terminal with ID {term_id} not found")

        return self.terminals[term_id].get_command_status(command_id)

    async def get_terminal_status(self, term_id: str) -> dict:
        """
        Get comprehensive status of a terminal including active commands.

        Args:
            term_id: ID of the terminal to check

        Returns:
            Dictionary containing terminal status and active commands

        Raises:
            KeyError: If terminal_id doesn't exist
        """
        if term_id not in self.terminals:
            raise KeyError(f"Terminal with ID {term_id} not found")

        terminal = self.terminals[term_id]
        return {
            "running": await terminal.is_alive(),
            "ready_for_commands": terminal.is_ready_for_commands(),
            "active_commands": terminal.get_active_commands(),
            "last_command": terminal.last_command,
        }

    async def cleanup_all(self):
        """Clean up all terminals - useful for interrupt handling"""
        print(f"Cleaning up {len(self.terminals)} terminal(s)")
        for terminal_id, terminal in list(self.terminals.items()):
            try:
                await terminal.close()
                print(f"Closed terminal {terminal_id}")
            except Exception as e:
                print(f"Error closing terminal {terminal_id}: {e}")
        self.terminals.clear()

    async def run_command(self, command: str, cwd: Optional[str] = None, timeout: Optional[int] = None) -> str:
        """
        Run a command directly (convenience method for utilities).

        Args:
            command: Command to execute
            cwd: Optional working directory
            timeout: Optional timeout in seconds

        Returns:
            Command output as string
        """
        # Create a temporary terminal for this command
        terminal_kwargs = {}
        if cwd:
            terminal_kwargs["cwd"] = cwd

        terminal_id = await self.launch_terminal(**terminal_kwargs)
        try:
            # Send command and wait for completion
            await self.send_command(terminal_id, command)

            # Get output (with a reasonable timeout)
            import asyncio

            await asyncio.sleep(0.1)  # Brief wait for command to start

            # Wait for command completion by checking if it's ready for new commands
            max_wait = timeout or 30  # Default 30 second timeout
            waited = 0
            while waited < max_wait:
                if self.terminals[terminal_id].is_ready_for_commands():
                    break
                await asyncio.sleep(0.5)
                waited += 0.5

            # Get the output
            output = await self.get_output(terminal_id)
            return output

        finally:
            # Clean up the temporary terminal
            await self.close_terminal(terminal_id)
