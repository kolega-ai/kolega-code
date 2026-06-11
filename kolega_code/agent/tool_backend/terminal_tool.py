import asyncio
import os.path
import re
import time
from pathlib import Path
from typing import Optional, Tuple, Union

from .. import prompts
from kolega_code.config import AgentConfig
from kolega_code.llm.client import LLMClient
from kolega_code.llm.models import Message, MessageHistory, TextBlock
from kolega_code.llm.specs import get_model_specs
from kolega_code.events import AgentEvent
from kolega_code.services.terminal import LocalTerminalManager
from .base_tool import BaseTool


def _generate_compression_notice(terminal_id: str, full_char_count: int, threshold: int, compressed_output: str) -> str:
    """
    Generate a properly formatted compression notice.

    Args:
        terminal_id: ID of the terminal
        full_char_count: Actual character count of the full output
        threshold: The compression threshold that was exceeded
        compressed_output: The compressed/summarized output

    Returns:
        Formatted compression notice string
    """
    # Recommend reading at the threshold limit to avoid re-compression
    recommended_char_count = threshold

    return f"""⚠️  **OUTPUT COMPRESSED** ⚠️

The terminal output ({full_char_count:,} characters) exceeded the compression threshold ({threshold:,} characters) and has been summarized below.

**To read uncompressed output (up to {threshold:,} characters), use:**
`read_terminal({terminal_id}, num_chars={recommended_char_count})`

**To read specific portions without compression, use the offset parameter:**
`read_terminal({terminal_id}, num_chars=<chars>, offset=<chars_from_end>)`

**Note:** Reading more than {threshold:,} characters without an offset will result in compression again.

---

**Compressed Summary:**
{compressed_output}

---

💡 **Tip:** Use the read_terminal tool with {recommended_char_count:,} characters to get the most recent uncompressed output, or use the offset parameter to read specific portions.
"""


class TerminalTool(BaseTool):

    output_compression_threshold = 4000
    DEFAULT_COMMAND_WAIT_TIMEOUT_SECONDS = 120
    MAX_COMMAND_WAIT_TIMEOUT_SECONDS = 300

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
                workspace_id=workspace_id, thread_id=thread_id, connection_manager=connection_manager
            )

    async def _run_command_security_check(self, command: str) -> Tuple[bool, str]:
        provider = self.config.fast_config.provider
        api_key = self.config.get_api_key(provider)
        rate_limits = self.config.fast_config.rate_limits

        client = LLMClient(
            provider=provider.value,
            api_key=api_key,
            max_retries=rate_limits.max_retries,
            requests_per_minute=rate_limits.requests_per_minute,
            tokens_per_minute=rate_limits.tokens_per_minute,
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
                return True, None
            else:
                return False, response_text
        except Exception as ex:
            error_msg = f"Command not executed. Could not verify safety: {str(ex)}"
            return False, error_msg

    async def _compress_terminal_output(
        self, output: str, last_command: str, command_purpose: Optional[str] = None
    ) -> str:
        provider = self.config.fast_config.provider
        api_key = self.config.get_api_key(provider)
        rate_limits = self.config.fast_config.rate_limits

        client = LLMClient(
            provider=provider.value,
            api_key=api_key,
            max_retries=rate_limits.max_retries,
            requests_per_minute=rate_limits.requests_per_minute,
            tokens_per_minute=rate_limits.tokens_per_minute,
        )

        try:
            model_specs = get_model_specs(self.config.fast_config.provider, self.config.fast_config.model)

            system_message = Message(role="system", content=[TextBlock(text=prompts.SHELL_COMPRESSION_SYSTEM_PROMPT)])

            messages = MessageHistory(
                [
                    Message(
                        role="user",
                        content=[
                            TextBlock(
                                text=f"Last command:\n{str(last_command)}\nCommand purpose:\n{command_purpose}\nOutput:\n{output}"
                            )
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

            compressed = response.get_text_content()

            return compressed
        except Exception:
            # Return full output as fallback if compression fails.
            return output

    async def launch_terminal(self, terminal_id: Optional[str] = None) -> str:
        terminal_id = await self.terminal_manager.launch_terminal(cwd=self.project_path)

        terminal_launched_event = AgentEvent(
            event_type="terminal_launched", sender="agent", content={"terminal_id": terminal_id}
        )
        await self.connection_manager.broadcast_event(terminal_launched_event, self.workspace_id, self.thread_id)

        return f"Launched new terminal with terminal_id {terminal_id}"

    async def run_command(self, terminal_id: str, command: str, purpose: str) -> str:
        if self.security_check_enabled:
            allowed, denied_reason = await self._run_command_security_check(command)

            if not allowed:
                return denied_reason

        # For sandbox environments, use timeout=0 (no timeout) to allow long-running processes
        # LocalTerminalManager will ignore this parameter as it uses persistent shell sessions
        success = await self.terminal_manager.send_command(terminal_id, command, purpose=purpose, timeout=0)

        if success:
            return f"Ran command `{command}` in terminal {terminal_id}. Use read_terminal to read the output."
        else:
            return f"Failed to run command `{command}` in terminal {terminal_id}. Terminal may not be running."

    async def read_terminal(self, terminal_id: str, num_chars: int = 1024, offset: int = 0) -> str:
        """
        Read characters from a terminal's persistent output buffer.

        Args:
            terminal_id: ID of the terminal to read output from
            num_chars: Number of characters to read (default: 1024).
                      If buffer is smaller than num_chars, returns entire buffer.
            offset: Number of characters from the end to start reading from (default: 0).
                   If offset is 0, reads the last num_chars characters.
                   If offset is > 0, reads num_chars characters starting from that offset from the end.
                   Note: When offset > 0, compression is bypassed to allow reading specific portions.

        Returns:
            The requested characters from the terminal's output buffer, formatted in markdown code blocks.
            When offset is used, compression is skipped to preserve the exact requested content.
        """
        output = self.terminal_manager.read_output(terminal_id, num_chars=num_chars, offset=offset)
        output_ansi_stripped = self._strip_ansi_codes(output)

        # Skip compression when using offset to allow reading specific portions
        if offset > 0:
            return f"```\n{output_ansi_stripped}```\n"

        # Apply compression logic only when reading from the end (offset = 0)
        if len(output_ansi_stripped) > self.output_compression_threshold:
            last_command = await self.terminal_manager.get_last_command(terminal_id)
            command_purpose = await self.terminal_manager.get_last_command_purpose(terminal_id)
            compressed_output = await self._compress_terminal_output(
                output_ansi_stripped, last_command, command_purpose
            )

            # Calculate the full character count
            full_char_count = len(output_ansi_stripped)

            compression_notice = _generate_compression_notice(
                terminal_id, full_char_count, self.output_compression_threshold, compressed_output
            )
            return compression_notice

        return f"```\n{output_ansi_stripped}```\n"

    async def close_terminal(self, terminal_id: str) -> str:
        await self.terminal_manager.close_terminal(terminal_id)

        terminal_closed_event = AgentEvent(
            event_type="terminal_closed", sender="agent", content={"terminal_id": terminal_id}
        )
        await self.connection_manager.broadcast_event(terminal_closed_event, self.workspace_id, self.thread_id)

        return f"Terminal with ID {terminal_id} closed."

    async def list_terminals(self):
        results = await self.terminal_manager.list_terminals()

        formatted_results = "# Terminal Sessions\n\n"

        if not results:
            formatted_results += "No active terminals found.\n"
        else:
            formatted_results += "| Terminal ID | Status | Last Command |\n"
            formatted_results += "|-------------|--------|-------------|\n"

            for terminal_id, terminal_info in results.items():
                status = "Running" if terminal_info["running"] else "Stopped"
                last_command = terminal_info["last_command"] or "None"
                # Truncate long commands for better display
                if len(last_command) > 50:
                    last_command = last_command[:47] + "..."
                formatted_results += f"| {terminal_id} | {status} | {last_command} |\n"

        return formatted_results

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

    def configure(self, auto_activate_venv: bool = None, security_check_enabled: bool = None) -> None:
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

    async def run_command_tracked(self, terminal_id: str, command: str, purpose: str) -> str:
        """
        Run a command with tracking and return a command ID.

        This version provides reliable command completion detection by monitoring
        the actual process status rather than interpreting output.

        Args:
            terminal_id: The terminal to run the command in
            command: The command to execute
            purpose: Description of what the command is meant to do

        Returns:
            Command ID that can be used to check status, or error message
        """
        if self.security_check_enabled:
            allowed, denied_reason = await self._run_command_security_check(command)
            if not allowed:
                return denied_reason

        try:
            # For sandbox environments, use timeout=0 (no timeout) to allow long-running processes
            command_id = await self.terminal_manager.send_command_tracked(terminal_id, command, purpose, timeout=0)
            if command_id:
                return command_id  # Return just the command ID, not a message
            else:
                return f"Failed to start command `{command}` in terminal {terminal_id}. Terminal may not be running."
        except KeyError as e:
            return str(e)

    async def send_terminal_input(
        self, terminal_id: str, text: str, submit: bool = True, command_id: Optional[str] = None
    ) -> str:
        """
        Send input to an already-running command in a terminal.

        Args:
            terminal_id: The terminal where the command is running
            text: Text to send to the process
            submit: Whether to append a newline before sending
            command_id: Optional command ID when more than one command is active

        Returns:
            Confirmation that input was sent, or a readable error
        """
        try:
            success = await self.terminal_manager.send_input(
                terminal_id, text, submit=submit, command_id=command_id
            )
        except (KeyError, ValueError) as e:
            return str(e)

        if not success:
            return f"Failed to send input to terminal {terminal_id}. Terminal may not be running."

        command_suffix = f" for command {command_id}" if command_id else ""
        submit_suffix = " and submitted it" if submit else ""
        return f"Sent input to terminal {terminal_id}{command_suffix}{submit_suffix}."

    async def check_command_status(self, terminal_id: str, command_id: str) -> str:
        """
        Check the status of a specific command using process monitoring.

        This provides reliable completion detection without LLM interpretation.

        Args:
            terminal_id: The terminal the command is running in
            command_id: The command ID returned from run_command_tracked

        Returns:
            Formatted status information including completion state
        """
        try:
            status = self.terminal_manager.get_command_status(terminal_id, command_id)

            if status["status"] == "not_found":
                return f"❌ Command ID {command_id} not found"

            duration_str = f"{status['duration']:.1f}s"

            if status["status"] == "running":
                child_pids = status.get("child_pids") or []
                child_info = f" ({len(child_pids)} child processes)" if child_pids else ""
                return (
                    f"🔄 Command still running in terminal {terminal_id} after {duration_str}{child_info}\n"
                    f"Command: {status['command']}"
                )
            elif status["status"] == "completed":
                return_code = status.get("return_code", "unknown")
                return (
                    f"✅ Command completed in {duration_str} with exit code {return_code}\nCommand: {status['command']}"
                )
            elif status["status"] == "terminated":
                return f"❌ Command terminated after {duration_str}\nCommand: {status['command']}"
            elif status["status"] == "failed":
                return_code = status.get("return_code", 1)
                return f"❌ Command failed in {duration_str} with exit code {return_code}\nCommand: {status['command']}"
            elif status["status"] == "monitor_timeout":
                return (
                    f"⚠️ Command monitoring stopped after {duration_str}; command may still be running in "
                    f"terminal {terminal_id}.\nCommand: {status['command']}\n"
                    f'Use `check_command_status("{terminal_id}", "{command_id}")` to check it again.'
                )
            else:
                return f"❓ Command status: {status['status']} after {duration_str}\nCommand: {status['command']}"

        except KeyError as e:
            return str(e)

    async def check_terminal_status(self, terminal_id: str) -> str:
        """
        Get comprehensive status of a terminal including all active commands.

        Args:
            terminal_id: The terminal to check

        Returns:
            Formatted terminal status including readiness and active commands
        """
        try:
            status = await self.terminal_manager.get_terminal_status(terminal_id)

            result = f"# Terminal {terminal_id} Status\n\n"
            result += f"**Running:** {'Yes' if status['running'] else 'No'}\n"
            result += f"**Ready for new commands:** {'Yes' if status['ready_for_commands'] else 'No'}\n\n"

            active_commands = status["active_commands"]
            if active_commands:
                result += "**Active Commands:**\n"
                for cmd_id, cmd_status in active_commands.items():
                    duration = f"{cmd_status['duration']:.1f}s"
                    result += f"- `{cmd_id}`: {cmd_status['command']} (running {duration})\n"
            else:
                result += "**Active Commands:** None\n"

            return result

        except KeyError as e:
            return str(e)

    def _normalize_command_wait_timeout(self, timeout: Optional[int]) -> int:
        try:
            wait_timeout = int(timeout)
        except (TypeError, ValueError):
            return self.DEFAULT_COMMAND_WAIT_TIMEOUT_SECONDS

        if wait_timeout <= 0:
            return self.DEFAULT_COMMAND_WAIT_TIMEOUT_SECONDS
        return min(wait_timeout, self.MAX_COMMAND_WAIT_TIMEOUT_SECONDS)

    def _format_command_wait_timeout(self, terminal_id: str, command_id: str, timeout: int) -> str:
        return (
            f"⏰ Timeout: Command {command_id} is still running in terminal {terminal_id} after {timeout} seconds.\n"
            f'Use `check_command_status("{terminal_id}", "{command_id}")` to check it again, '
            f'or `close_terminal("{terminal_id}")` if you want to stop that terminal.'
        )

    async def wait_for_command_completion(
        self, terminal_id: str, command_id: str, timeout: Optional[int] = DEFAULT_COMMAND_WAIT_TIMEOUT_SECONDS
    ) -> str:
        """
        Wait for a specific command to complete with reliable process monitoring.

        Args:
            terminal_id: The terminal the command is running in
            command_id: The command ID to wait for
            timeout: Maximum time to wait in seconds

        Returns:
            Completion status or timeout message
        """
        wait_timeout = self._normalize_command_wait_timeout(timeout)
        start_time = time.time()

        while time.time() - start_time < wait_timeout:
            try:
                status = self.terminal_manager.get_command_status(terminal_id, command_id)

                if status["status"] in ["completed", "terminated", "not_found", "failed", "monitor_timeout"]:
                    return await self.check_command_status(terminal_id, command_id)

            except KeyError as e:
                return str(e)

            remaining = wait_timeout - (time.time() - start_time)
            if remaining <= 0:
                break
            await asyncio.sleep(min(1, remaining))  # Check every second

        return self._format_command_wait_timeout(terminal_id, command_id, wait_timeout)
