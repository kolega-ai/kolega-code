import asyncio
import json
import os.path
import re
from pathlib import Path
from typing import Optional, Tuple, Union

from .. import prompts
from kolega_code.config import AgentConfig
from kolega_code.llm.client import LLMClient
from kolega_code.llm.models import Message, MessageHistory, TextBlock
from kolega_code.llm.specs import get_model_specs
from kolega_code.events import AgentEvent
from kolega_code.services.base import ExecResult
from kolega_code.services.terminal import LocalTerminalManager
from .base_tool import BaseTool


class TerminalTool(BaseTool):
    def __init__(
        self,
        project_path: Union[str, Path],
        workspace_id: str,
        thread_id: str,
        connection_manager,
        config: AgentConfig,
        caller,
        filesystem=None,
        terminal_manager=None,
    ):
        super().__init__(
            project_path,
            workspace_id,
            thread_id,
            connection_manager,
            config,
            caller,
            filesystem,
            terminal_manager=terminal_manager,
        )

        self.auto_activate_venv = True
        self.venv_activation_command = None
        self.initialized = False
        self.security_check_enabled = False

        # Use injected terminal_manager if provided, otherwise create local one
        if self.terminal_manager is None:
            self.terminal_manager = LocalTerminalManager(
                workspace_id=workspace_id,
                thread_id=thread_id,
                connection_manager=connection_manager,
                default_workdir=self.project_path,
            )

    async def _run_command_security_check(self, command: str) -> Tuple[bool, str]:
        provider = self.config.fast_config.provider
        api_key = self.config.get_api_key(provider)
        rate_limits = self.config.fast_config.rate_limits

        client = LLMClient(
            provider=provider.value,
            api_key=api_key or "",
            max_retries=rate_limits.max_retries,
            requests_per_minute=rate_limits.requests_per_minute,
            tokens_per_minute=rate_limits.tokens_per_minute,
            token_manager=self.config.get_chatgpt_token_manager(),
        )

        try:
            model_specs = get_model_specs(self.config.fast_config.provider, self.config.fast_config.model)

            system_message = Message(role="system", content=[TextBlock(text=prompts.SHELL_SAFETY_SYSTEM_PROMPT)])

            messages = MessageHistory(
                [
                    Message(
                        role="user",
                        content=[
                            TextBlock(text=f"Project directory:\n{str(self.caller.project_path)}\nCommand:\n{command}")
                        ],
                    )
                ]
            )

            response = await client.generate(
                model=self.config.fast_config.model,
                max_completion_tokens=model_specs["max_completion_tokens"],
                system=system_message,
                messages=messages,
            )

            response_text = response.get_text_content()

            if response_text == "safe":
                return True, ""
            else:
                return False, response_text
        except Exception as ex:
            error_msg = f"Command not executed. Could not verify safety: {str(ex)}"
            return False, error_msg

    # -- model-facing unified-exec tools -----------------------------------

    def _format_result(self, result: ExecResult) -> str:
        return json.dumps(
            {
                "status": result.status,
                "exit_code": result.exit_code,
                "session_id": result.session_id,
                "output": result.output,
                "truncated": result.truncated,
                "original_token_count": result.original_token_count,
                "duration_ms": result.duration_ms,
            }
        )

    async def exec_command(
        self,
        command: str,
        workdir: Optional[str] = None,
        yield_time_ms: int = 10000,
        max_output_tokens: int = 10000,
        login: bool = False,
    ) -> str:
        """Run a shell command as a fresh process and return its output.

        The command runs under a pseudo-terminal so interactive programs behave
        normally. Output is collected for up to yield_time_ms milliseconds. If
        the process exits within that window, the full result with its real exit
        code is returned. If it is still running, a session_id is returned that
        you can drive with write_stdin (to send input or poll for more output)
        and stop with kill_command.

        The working directory does NOT persist between calls. Pass `workdir`, or
        chain commands in one call with `cd path && ...`. Defaults to the
        project root.

        Args:
            command: Shell command line, executed via `bash -c`.
            workdir: Working directory for the command. Relative paths resolve
                     against the project root. Defaults to project root.
            yield_time_ms: How long to wait for output/exit before returning,
                           in milliseconds (clamped to 250–30000).
            max_output_tokens: Maximum tokens of output to return in this call.
            login: Run the shell as a login shell (sources profile). Default false.

        Returns:
            A JSON object: {"status": "exited"|"running", "exit_code",
            "session_id", "output", "truncated", "original_token_count",
            "duration_ms"}.
        """
        if self.security_check_enabled:
            allowed, denied_reason = await self._run_command_security_check(command)
            if not allowed:
                return denied_reason

        # Resolve relative workdirs against the project root (same contract as
        # the file tools), never against this process's cwd.
        wd = os.path.normpath(str(self.project_path / workdir)) if workdir else str(self.project_path)
        try:
            result = await self.terminal_manager.exec_command(
                command,
                workdir=wd,
                yield_time_ms=yield_time_ms,
                max_output_tokens=max_output_tokens,
                login=login,
            )
        except Exception as exc:
            return json.dumps({"status": "error", "error": str(exc)})
        return self._format_result(result)

    async def write_stdin(
        self,
        session_id: str,
        chars: str = "",
        yield_time_ms: int = 10000,
        max_output_tokens: int = 10000,
    ) -> str:
        """Write input to a running session's stdin and read recent output.

        Pass chars="" to poll (read new output without writing). Use this to
        answer prompts (e.g. send "y\\n"), drive a REPL, or send control
        characters (e.g. "\\x03" for Ctrl-C). The text is sent raw — include a
        trailing "\\n" to submit a line. Waits up to yield_time_ms (clamped to
        250–30000 when writing, 5000–300000 when polling) for more output or for
        the process to exit.

        Args:
            session_id: The id returned by exec_command when status == "running".
            chars: Bytes to write to stdin. An empty string polls only.
            yield_time_ms: Wait window in milliseconds.
            max_output_tokens: Maximum tokens of output to return in this call.

        Returns:
            A JSON object with the same shape as exec_command.
        """
        try:
            result = await self.terminal_manager.write_stdin(
                session_id, chars, yield_time_ms=yield_time_ms, max_output_tokens=max_output_tokens
            )
        except KeyError as exc:
            return json.dumps({"status": "error", "error": str(exc)})
        return self._format_result(result)

    async def kill_command(self, session_id: str, signal: str = "TERM") -> str:
        """Terminate a running session and its process group.

        Sends SIGTERM (then SIGKILL after a short grace period). Use
        signal="INT" to send Ctrl-C (SIGINT) instead.

        Args:
            session_id: The id of the session to stop.
            signal: "TERM" (default, graceful) or "INT" (Ctrl-C).

        Returns:
            A JSON object describing the final state of the session.
        """
        try:
            result = await self.terminal_manager.kill_session(session_id, signal)
        except KeyError as exc:
            return json.dumps({"status": "error", "error": str(exc)})
        return self._format_result(result)

    async def list_sessions(self) -> str:
        """List currently running exec sessions.

        Returns:
            A JSON object mapping each running session id to its command,
            working directory, and runtime in seconds.
        """
        sessions = await self.terminal_manager.list_sessions()
        return json.dumps({"sessions": sessions})

    # -- internal one-shot helper (not exposed to the model) ---------------

    async def execute_terminal_command(self, command: str, strip_colors: bool = True) -> str:
        """
        Execute a command and display output in terminal.

        Args:
            command: The command to execute
            strip_colors: Whether to strip ANSI color codes from output (default: True)
        """
        # Log the command
        await self.log_info(f"Executing command: {command}", sender=self.caller.agent_name)

        # Check security if enabled
        if self.security_check_enabled:
            allowed, denied_reason = await self._run_command_security_check(command)
            if not allowed:
                await self.log_error(
                    f"Command blocked by security check: {denied_reason}", sender=self.caller.agent_name
                )
                return f"Command execution blocked: {denied_reason}"

        # Initialize terminal environment if not already done
        if not self.initialized and self.auto_activate_venv:
            await self.initialize_terminal()

        # Prepend virtual environment activation command if available and requested
        full_command = command
        if self.venv_activation_command:
            # Use a subshell to maintain environment for this command
            full_command = f"(source {self.venv_activation_command} && {command})"

        try:
            # Send command to terminal (show the original command to the user)
            terminal_command_event = AgentEvent(
                event_type="terminal_command", sender="agent", content={"command": command}
            )
            await self.connection_manager.broadcast_event(terminal_command_event, self.workspace_id, self.thread_id)

            # Execute the command (with potential venv activation)
            process = await asyncio.create_subprocess_shell(
                full_command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self.project_path),
                shell=True,
            )

            try:
                # Set a timeout of 15 seconds
                stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=15)
                output = stdout.decode() + stderr.decode()

                # Strip ANSI color codes if requested
                if strip_colors:
                    output = self._strip_ansi_codes(output)

            except asyncio.TimeoutError:
                # If the command doesn't return after 15 seconds
                output = "Command timed out after 15 seconds"
                await self.log_info(f"Command timed out: {command}", sender=self.caller.agent_name)

                # Try to terminate the process
                try:
                    process.terminate()
                except Exception:
                    pass

            terminal_output_event = AgentEvent(event_type="terminal_output", sender="agent", content={"output": output})
            await self.connection_manager.broadcast_event(terminal_output_event, self.workspace_id, self.thread_id)

            return output
        except Exception as e:
            error_msg = f"Command execution failed: {str(e)}"
            await self.log_error(error_msg, sender=self.caller.agent_name)
            return error_msg

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

    def configure(
        self, auto_activate_venv: Optional[bool] = None, security_check_enabled: Optional[bool] = None
    ) -> None:
        """
        Configure the terminal tool settings.

        Args:
            auto_activate_venv: Whether to automatically detect and activate virtual environments
            security_check_enabled: Whether to perform security checks on commands before execution
        """
        if auto_activate_venv is not None:
            self.auto_activate_venv = auto_activate_venv
            # Reset initialization state if configuration changes
            if not auto_activate_venv:
                self.venv_activation_command = None
            self.initialized = False

        if security_check_enabled is not None:
            self.security_check_enabled = security_check_enabled

    async def detect_venv(self) -> str:
        """
        Detect if a Python virtual environment exists in the project directory.

        Returns:
            Path to the activation script if a virtual environment was found, empty string otherwise
        """
        if not self.auto_activate_venv:
            await self.log_info("Virtual environment auto-activation is disabled", sender=self.caller.agent_name)
            return ""

        # Common virtual environment directory names
        venv_dirs = [".venv", "venv", "env", ".env"]

        for venv_dir in venv_dirs:
            # Check if the virtual environment directory exists
            venv_path = os.path.join(str(self.project_path), venv_dir)

            # Check if directory exists
            if not os.path.isdir(venv_path):
                continue

            # Check for the activation script based on OS
            activate_script = os.path.join(venv_path, "bin", "activate")
            windows_script = os.path.join(venv_path, "Scripts", "activate")

            if os.path.isfile(activate_script):
                await self.log_info(f"Found virtual environment at {venv_dir}", sender=self.caller.agent_name)
                return activate_script

            if os.path.isfile(windows_script):
                await self.log_info(
                    f"Found virtual environment at {venv_dir} (Windows style)", sender=self.caller.agent_name
                )
                return windows_script

        await self.log_info("No virtual environment found in the project directory", sender=self.caller.agent_name)
        return ""

    async def initialize_terminal(self) -> None:
        """
        Initialize the terminal by detecting virtual environments.
        Sets the venv_activation_command for use in subsequent commands.
        """
        await self.log_info("Initializing terminal environment...", sender=self.caller.agent_name)

        # Detect virtual environment
        activation_script = await self.detect_venv()

        if activation_script:
            self.venv_activation_command = activation_script
            await self.log_info(
                f"Virtual environment activation script found at: {activation_script}", sender=self.caller.agent_name
            )
        else:
            self.venv_activation_command = None

        self.initialized = True
        await self.log_info("Terminal initialization complete", sender=self.caller.agent_name)
