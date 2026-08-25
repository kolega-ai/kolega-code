import asyncio
import os
import re
from pathlib import Path
from typing import Optional, Tuple, Union

from .. import prompts
from kolega_code.config import AgentConfig
from kolega_code.llm.client import LLMClient
from kolega_code.llm.ledger import helper_origin, llm_call_origin
from kolega_code.llm.models import Message, MessageHistory, TextBlock
from kolega_code.llm.specs import get_model_specs
from kolega_code.events import AgentEvent
from kolega_code.services.base import ExecResult, TerminalCommandCancelled
from kolega_code.services.terminal import LocalTerminalManager, build_child_env
from kolega_code.services.terminal_buffer import (
    GLOBAL_MAX_TOOL_OUTPUT_TOKENS,
    cap_chars,
    clamp_output_tokens,
)
from .base_tool import BaseTool

# Startup window for background=true launches: long enough to capture a dev
# server's "ready" line or an early crash (port in use, missing env), short
# enough that the agent is not blocked on a process meant to outlive the call.
BACKGROUND_SETTLE_MS = 2000

_BACKGROUND_NOTE = (
    "Running in background, detached: it keeps running after this command and even "
    "after the agent session ends, until you stop it with kill_command(session_id) "
    "(or the environment goes away). write_stdin(session_id, chars) sends real input "
    "to its stdin; write_stdin(session_id) polls new output. Input is not echoed and "
    "its stdin never reaches EOF, so verify effects from the output (or as a client) "
    "and stop stdin-reading commands with kill_command. See all running shells with "
    "list_sessions. Output may be buffered, so verify a server by reaching it as a "
    "client (e.g. curl), not by waiting on this log."
)


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
        # Ids of sessions started with background=true that are still running.
        # Best-effort annotation metadata for list_sessions; the sessions
        # themselves are always tracked manager-wide.
        self._background_sessions: set[str] = set()

        # Use injected terminal_manager if provided, otherwise create local one
        if self.terminal_manager is None:
            self.terminal_manager = LocalTerminalManager(
                workspace_id=workspace_id,
                thread_id=thread_id,
                connection_manager=connection_manager,
                default_workdir=self.project_path,
            )
        self._configure_terminal_spill()

    def _configure_terminal_spill(self) -> None:
        """Bind terminal output to durable session storage when available."""
        root: Optional[Path] = None
        recorder = getattr(self.caller, "session_recorder", None)
        journal = getattr(recorder, "journal", None)
        session_dir = getattr(journal, "session_dir", None)
        if isinstance(session_dir, (str, os.PathLike)) and str(session_dir):
            root = Path(session_dir).expanduser().resolve() / "terminal-output"
        else:
            scratchpad_dir = getattr(self.caller, "scratchpad_dir", None)
            if isinstance(scratchpad_dir, (str, os.PathLike)) and str(scratchpad_dir):
                root = Path(scratchpad_dir).expanduser().resolve() / "terminal-output"
        self.terminal_manager.configure_output_spill(root)

    async def _run_command_security_check(self, command: str) -> Tuple[bool, str]:
        provider = self.config.fast_config.provider
        api_key = self.config.get_api_key(provider)
        rate_limits = self.config.fast_config.rate_limits
        endpoint = self.config.custom_endpoint_for(self.config.fast_config)

        client = LLMClient(
            provider=provider.value,
            api_key=api_key or "",
            model=self.config.fast_config.model,
            max_retries=rate_limits.max_retries,
            requests_per_minute=rate_limits.requests_per_minute,
            tokens_per_minute=rate_limits.tokens_per_minute,
            token_manager=self.config.get_chatgpt_token_manager(),
            usage_ledger=getattr(self.caller, "usage_ledger", None),
            trace_sink=getattr(self.caller, "llm_trace_sink", None),
            base_url=endpoint.base_url if endpoint else None,
            api_style=endpoint.api_style if endpoint else None,
        )

        try:
            model_specs = get_model_specs(self.config.fast_config.provider, self.config.fast_config.model)

            system_message = Message(role="system", content=[TextBlock(text=prompts.SHELL_SAFETY_SYSTEM_PROMPT)])

            scratchpad_dir = getattr(self.caller, "scratchpad_dir", None)
            directory_context = f"Project directory:\n{str(self.caller.project_path)}"
            if isinstance(scratchpad_dir, (str, Path)) and scratchpad_dir:
                directory_context += f"\nScratchpad directory (session-writable):\n{scratchpad_dir}"
            messages = MessageHistory(
                [
                    Message(
                        role="user",
                        content=[TextBlock(text=f"{directory_context}\nCommand:\n{command}")],
                    )
                ]
            )

            with llm_call_origin(helper_origin("terminal_security")):
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

    def _session_env(self) -> Optional[dict]:
        """Per-session env overrides injected into every exec session.

        The scratchpad path is documented to the model through the scratchpad
        prompt extension, not the tool description, so the tool schema stays
        byte-identical. The messaging socket address lets a command post a
        message back into the session that spawned it (own-child delivery).
        """
        env: dict[str, str] = {}
        scratchpad_dir = getattr(self.caller, "scratchpad_dir", None)
        if scratchpad_dir:
            env["KOLEGA_SCRATCHPAD"] = str(scratchpad_dir)
        socket_path = getattr(self.caller, "messaging_socket_path", None)
        if socket_path:
            env["KOLEGA_MESSAGING_SOCKET"] = str(socket_path)
        return env or None

    def _format_result(
        self,
        result: ExecResult,
        *,
        background: bool = False,
        max_output_tokens: int = GLOBAL_MAX_TOOL_OUTPUT_TOKENS,
    ) -> str:
        effective_tokens = clamp_output_tokens(max_output_tokens)
        original_tokens = result.original_token_count or ((len(result.output) + 3) // 4 if result.output else 0)
        truncated = result.truncated
        capped = None
        output_prefix = ""

        for _ in range(2):
            header = self._result_header(
                result,
                truncated=truncated,
                original_tokens=original_tokens,
                background=background,
            )
            output_prefix = f"{header}\n\nOutput:\n"
            # Recovery metadata is more important than honoring an unusually
            # tiny requested budget. It may exceed that tiny request, but the
            # complete tool result always remains below the 10,000-token global
            # ceiling and the ordinary spill path is never shortened.
            global_chars = GLOBAL_MAX_TOOL_OUTPUT_TOKENS * 4
            requested_chars = effective_tokens * 4
            total_budget = min(global_chars, max(requested_chars, len(output_prefix)))
            capped = cap_chars(
                result.output,
                max(0, total_budget - len(output_prefix)),
                marker=f"\n[... output truncated to fit {effective_tokens} tokens ...]\n",
                original_token_count=original_tokens,
            )
            final_truncated = result.truncated or capped.truncated
            if final_truncated == truncated:
                return output_prefix + capped.text
            truncated = final_truncated

        assert capped is not None
        return output_prefix + capped.text

    @staticmethod
    def _result_header(
        result: ExecResult,
        *,
        truncated: bool,
        original_tokens: int,
        background: bool,
    ) -> str:
        lines = [
            f"Status: {result.status}",
            f"Exit code: {result.exit_code if result.exit_code is not None else 'none'}",
            f"Session: {result.session_id or 'none'}",
            f"Duration: {result.duration_ms} ms",
            f"Output truncated: {'yes' if truncated else 'no'}",
            f"Original output: ~{original_tokens:,} tokens",
        ]
        if result.spill_path:
            lines.extend(
                [
                    f"Full output: {result.spill_path}",
                    f"Spill size: {result.spill_bytes:,} bytes",
                ]
            )
        if result.preview_omitted_bytes:
            lines.append(
                "Preview middle omitted: "
                f"{result.preview_omitted_bytes:,} bytes across "
                f"{result.preview_omitted_lines:,} "
                f"{'line' if result.preview_omitted_lines == 1 else 'lines'}"
            )
        if result.line_truncated_count or result.line_truncated_bytes:
            lines.append(
                "Long lines shortened: "
                f"{result.line_truncated_count:,} "
                f"{'line' if result.line_truncated_count == 1 else 'lines'}, "
                f"{result.line_truncated_bytes:,} bytes omitted"
            )
        if background and result.status == "running" and result.session_id is not None:
            lines.extend(["Background: true", f"Note: {_BACKGROUND_NOTE}"])
        return "\n".join(lines)

    @staticmethod
    def _format_error(error: object) -> str:
        return f"Status: error\nError: {error}"

    async def exec_command(
        self,
        command: str,
        workdir: Optional[str] = None,
        yield_time_ms: int = 10000,
        max_output_tokens: int = 10000,
        login: bool = False,
        background: bool = False,
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

        Running long-lived processes: pass background=true for dev servers,
        watchers, and long builds you want to keep running while you do other
        work. It returns after a short startup window with a session_id. The
        process is launched detached, so it keeps running until you stop it with
        kill_command — including after this agent session ends (it does NOT die
        when you finish). Do NOT use shell `&` for this — processes backgrounded
        that way are killed when the command that started them ends. Drive
        background sessions with write_stdin: chars sends real input, chars=""
        polls new output. Input is not echoed (their stdin is not a TTY) and
        never reaches EOF, so commands that read stdin run until kill_command
        stops them. Use list_sessions to see all running shells. Always verify
        a server answers (e.g. curl) before handing its URL to the browser
        agent — do not rely on its log, which may be buffered.

        Args:
            command: Shell command line, executed via `bash -c`.
            workdir: Working directory for the command. Relative paths resolve
                     against the project root. Defaults to project root.
            yield_time_ms: How long to wait for output/exit before returning,
                           in milliseconds (clamped to 250–30000).
            max_output_tokens: Maximum tokens of output to return in this call.
            login: Run the shell as a login shell (sources profile). Default false.
            background: Launch detached and return after a short startup window
                        (~2s) with a session_id. The process outlives this call
                        and the agent session until kill_command stops it; it
                        accepts write_stdin input (no echo; stdin never reaches
                        EOF). Commands that exit within the startup window
                        report their real exit code.

        Returns:
            Structured text with status, session id, exit code, duration,
            truncation metadata, and model-visible output. Oversized streams
            include an ordinary ``Full output:`` filesystem path. Background
            launches that are still running also include ``Background: true``
            and management guidance.
        """
        if self.security_check_enabled:
            allowed, denied_reason = await self._run_command_security_check(command)
            if not allowed:
                return denied_reason

        # Resolve relative workdirs against the project root (same contract as
        # the file tools), never against this process's cwd.
        wd = os.path.normpath(str(self.project_path / workdir)) if workdir else str(self.project_path)
        effective_max_tokens = clamp_output_tokens(max_output_tokens)
        try:
            result = await self.terminal_manager.exec_command(
                command,
                workdir=wd,
                yield_time_ms=min(yield_time_ms, BACKGROUND_SETTLE_MS) if background else yield_time_ms,
                max_output_tokens=effective_max_tokens,
                login=login,
                env=self._session_env(),
                background=background,
            )
        except TerminalCommandCancelled as exc:
            return self._format_result(
                exc.result,
                background=background,
                max_output_tokens=effective_max_tokens,
            )
        except Exception as exc:
            return self._format_error(exc)
        if background and result.status == "running" and result.session_id is not None:
            self._background_sessions.add(result.session_id)
        return self._format_result(result, background=background, max_output_tokens=effective_max_tokens)

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
        trailing "\\n" to submit a line. Waits up to yield_time_ms for more
        output or for the process to exit.

        Works for background sessions too: input is delivered to their stdin
        but not echoed, so verify the effect from the command's output; their
        stdin never reaches EOF, so stop stdin-reading commands with
        kill_command.

        Args:
            session_id: The id returned by exec_command when status == "running".
            chars: Bytes to write to stdin. An empty string polls only.
            yield_time_ms: How long to wait for more output or the process to
                           exit, in milliseconds (clamped to 250–30000 when
                           writing, 5000–300000 when polling).
            max_output_tokens: Maximum tokens of output to return in this call.

        Returns:
            Structured text with the same fields and spill behavior as
            exec_command.
        """
        effective_max_tokens = clamp_output_tokens(max_output_tokens)
        try:
            result = await self.terminal_manager.write_stdin(
                session_id,
                chars,
                yield_time_ms=yield_time_ms,
                max_output_tokens=effective_max_tokens,
            )
        except TerminalCommandCancelled as exc:
            return self._format_result(
                exc.result,
                max_output_tokens=effective_max_tokens,
            )
        except KeyError as exc:
            return self._format_error(exc)
        return self._format_result(result, max_output_tokens=effective_max_tokens)

    async def kill_command(self, session_id: str, signal: str = "TERM") -> str:
        """Terminate a running session and its process group.

        Sends SIGTERM (then SIGKILL after a short grace period). Use
        signal="INT" to send Ctrl-C (SIGINT) instead.

        Args:
            session_id: The id of the session to stop.
            signal: "TERM" (default, graceful) or "INT" (Ctrl-C).

        Returns:
            Structured text describing the final state of the session.
        """
        try:
            result = await self.terminal_manager.kill_session(session_id, signal)
        except KeyError as exc:
            return self._format_error(exc)
        self._background_sessions.discard(session_id)
        return self._format_result(result)

    async def list_sessions(self) -> str:
        """List currently running exec sessions.

        Includes sessions started with exec_command background=true, annotated
        with "background": true.

        Returns:
            Structured text listing each running session id, command, working
            directory, runtime in seconds, and background flag. Returns
            ``No running sessions.`` when the registry is empty.
        """
        sessions = await self.terminal_manager.list_sessions()
        self._background_sessions.intersection_update(sessions)
        if not sessions:
            return "No running sessions."

        lines = [f"Running terminal sessions: {len(sessions)}"]
        for session_id, info in sessions.items():
            command = str(info.get("command", ""))
            command_lines = command.splitlines() or [""]
            lines.extend(
                [
                    "",
                    f"Session: {session_id}",
                    f"  Command: {command_lines[0]}",
                ]
            )
            lines.extend(f"           {line}" for line in command_lines[1:])
            lines.extend(
                [
                    f"  Workdir: {info.get('workdir') or 'unknown'}",
                    f"  Runtime: {info.get('runtime_s', 0)} s",
                    f"  Running: {'true' if info.get('running', True) else 'false'}",
                    f"  Background: {'true' if session_id in self._background_sessions else 'false'}",
                ]
            )
        text = "\n".join(lines)
        return cap_chars(
            text,
            GLOBAL_MAX_TOOL_OUTPUT_TOKENS * 4,
            marker="\n[... additional sessions omitted ...]\n",
        ).text

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
            workdir = str(self.project_path.resolve())
            terminal_command_event = AgentEvent(
                event_type="terminal_command",
                sender="agent",
                content={"command": command, "workdir": workdir},
            )
            await self.connection_manager.broadcast_event(terminal_command_event, self.workspace_id, self.thread_id)

            # Execute the command (with potential venv activation)
            process = await asyncio.create_subprocess_shell(
                full_command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=workdir,
                env=build_child_env(),
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
