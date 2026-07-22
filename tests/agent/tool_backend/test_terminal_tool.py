import os
import asyncio
import json
from unittest.mock import AsyncMock, Mock, patch

import pytest
import uuid

from kolega_code.config import AgentConfig, ModelConfig, ModelProvider, RateLimitConfig
from kolega_code.events import AgentEvent
from kolega_code.services.base import ExecResult
from kolega_code.agent.tool_backend.terminal_tool import TerminalTool

# Check if running in CI environment
SKIP_IN_CI = bool(os.getenv("CI")) or bool(os.getenv("GITLAB_CI"))


@pytest.fixture
def mock_connection_manager():
    return AsyncMock()


@pytest.fixture
def project_path(tmp_path):
    return tmp_path


@pytest.fixture
def agent_config():
    return AgentConfig(
        anthropic_api_key="test_key",
        openai_api_key="test_key",
        long_context_config=ModelConfig(
            provider=ModelProvider.ANTHROPIC, model="test-model", rate_limits=RateLimitConfig()
        ),
        fast_config=ModelConfig(provider=ModelProvider.ANTHROPIC, model="test-model", rate_limits=RateLimitConfig()),
        thinking_config=ModelConfig(
            provider=ModelProvider.ANTHROPIC,
            model="test-model",
            rate_limits=RateLimitConfig(),
            thinking_effort="medium",
        ),
    )


@pytest.fixture
def mock_base_agent():
    mock = Mock()
    mock.agent_name = "test_agent"
    return mock


@pytest.fixture
def terminal_tool(project_path, mock_connection_manager, agent_config, mock_base_agent):
    tool = TerminalTool(
        project_path, "test_workspace", str(uuid.uuid4()), mock_connection_manager, agent_config, mock_base_agent
    )
    # Set initialized=True to prevent auto-initialization during tests
    tool.initialized = True
    return tool


class TestTerminalTool:
    @pytest.mark.asyncio
    async def test_execute_terminal_command_success(self, terminal_tool, mock_connection_manager):
        # Mock the subprocess
        mock_process = AsyncMock()
        mock_process.communicate = AsyncMock(return_value=(b"stdout", b""))

        with patch("asyncio.create_subprocess_shell", return_value=mock_process):
            result = await terminal_tool.execute_terminal_command('echo "Hello World"')

            assert result == "stdout"

            # Verify connection manager was called correctly
            mock_connection_manager.broadcast_event.assert_called()
            calls = mock_connection_manager.broadcast_event.call_args_list

            # Check log message broadcast (first call)
            assert isinstance(calls[0][0][0], AgentEvent)
            assert calls[0][0][0].event_type == "log_message"
            assert calls[0][0][0].content["text"] == 'Executing command: echo "Hello World"'
            assert calls[0][0][0].content["level"] == "info"
            assert calls[0][0][1] == "test_workspace"

            # Check command broadcast (second call)
            assert isinstance(calls[1][0][0], AgentEvent)
            assert calls[1][0][0].event_type == "terminal_command"
            assert calls[1][0][0].content["command"] == 'echo "Hello World"'
            assert calls[1][0][1] == "test_workspace"

            # Check output broadcast (third call)
            assert isinstance(calls[2][0][0], AgentEvent)
            assert calls[2][0][0].event_type == "terminal_output"
            assert calls[2][0][0].content["output"] == "stdout"
            assert calls[2][0][1] == "test_workspace"

    @pytest.mark.asyncio
    async def test_execute_terminal_command_with_stderr(self, terminal_tool, mock_connection_manager):
        # Mock the subprocess
        mock_process = AsyncMock()
        mock_process.communicate = AsyncMock(return_value=(b"stdout", b"stderr"))

        with patch("asyncio.create_subprocess_shell", return_value=mock_process):
            result = await terminal_tool.execute_terminal_command("test command")

            assert result == "stdoutstderr"

    @pytest.mark.asyncio
    async def test_execute_terminal_command_timeout(self, terminal_tool, mock_connection_manager):
        # Mock the subprocess to simulate a timeout
        mock_process = AsyncMock()
        mock_process.communicate = AsyncMock(side_effect=asyncio.TimeoutError())
        mock_process.terminate = Mock()

        with patch("asyncio.create_subprocess_shell", return_value=mock_process):
            result = await terminal_tool.execute_terminal_command("slow command")

            assert result == "Command timed out after 15 seconds"
            mock_process.terminate.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_terminal_command_process_error(self, terminal_tool, mock_connection_manager):
        # Mock the subprocess to raise an exception
        mock_process = AsyncMock()
        mock_process.communicate = AsyncMock(side_effect=Exception("Process error"))

        with patch("asyncio.create_subprocess_shell", return_value=mock_process):
            result = await terminal_tool.execute_terminal_command("error command")

            assert "Command execution failed: Process error" in result

    @pytest.mark.asyncio
    async def test_execute_terminal_command_empty_output(self, terminal_tool, mock_connection_manager):
        # Mock the subprocess with empty output
        mock_process = AsyncMock()
        mock_process.communicate = AsyncMock(return_value=(b"", b""))

        with patch("asyncio.create_subprocess_shell", return_value=mock_process):
            result = await terminal_tool.execute_terminal_command("empty command")

            assert result == ""

    @pytest.mark.asyncio
    async def test_execute_terminal_command_working_directory(self, terminal_tool, mock_connection_manager):
        # Mock the subprocess
        mock_process = AsyncMock()
        mock_process.communicate = AsyncMock(return_value=(b"stdout", b""))

        with patch("asyncio.create_subprocess_shell", return_value=mock_process) as mock_create:
            await terminal_tool.execute_terminal_command("pwd")

            # Verify the command was executed in the correct directory
            mock_create.assert_called_once()
            assert mock_create.call_args[1]["cwd"] == str(terminal_tool.project_path)


class TestUnifiedExecTools:
    """Tests for the codex-style exec_command / write_stdin / kill_command tools."""

    @pytest.mark.asyncio
    async def test_exec_command_returns_json_and_defaults_workdir(self, terminal_tool):
        terminal_tool.terminal_manager = Mock()
        terminal_tool.terminal_manager.exec_command = AsyncMock(
            return_value=ExecResult(status="exited", exit_code=0, output="hi", duration_ms=12)
        )

        out = await terminal_tool.exec_command("echo hi")
        data = json.loads(out)
        assert data["status"] == "exited"
        assert data["exit_code"] == 0
        assert data["output"] == "hi"

        kwargs = terminal_tool.terminal_manager.exec_command.call_args.kwargs
        assert kwargs["workdir"] == str(terminal_tool.project_path)

    @pytest.mark.asyncio
    async def test_exec_command_resolves_relative_workdir_against_project_root(self, terminal_tool):
        terminal_tool.terminal_manager = Mock()
        terminal_tool.terminal_manager.exec_command = AsyncMock(
            return_value=ExecResult(status="exited", exit_code=0, output="", duration_ms=1)
        )

        # "." must mean the project root, not the process cwd the CLI was
        # launched from (regression: commands ran in the launch directory).
        await terminal_tool.exec_command("pwd", workdir=".")
        kwargs = terminal_tool.terminal_manager.exec_command.call_args.kwargs
        assert kwargs["workdir"] == str(terminal_tool.project_path)

        await terminal_tool.exec_command("pwd", workdir="sub/dir")
        kwargs = terminal_tool.terminal_manager.exec_command.call_args.kwargs
        assert kwargs["workdir"] == str(terminal_tool.project_path / "sub" / "dir")

    @pytest.mark.asyncio
    async def test_exec_command_end_to_end_runs_in_project_root(self, terminal_tool):
        # Real PTY through the default manager: even when the CLI process runs
        # elsewhere, workdir="." must land in the project root.
        data = json.loads(await terminal_tool.exec_command("pwd", workdir=".", yield_time_ms=5000))
        assert data["status"] == "exited"
        assert data["output"].strip().endswith(terminal_tool.project_path.name)

    @pytest.mark.asyncio
    async def test_exec_command_passes_absolute_workdir_through(self, terminal_tool):
        terminal_tool.terminal_manager = Mock()
        terminal_tool.terminal_manager.exec_command = AsyncMock(
            return_value=ExecResult(status="exited", exit_code=0, output="", duration_ms=1)
        )

        await terminal_tool.exec_command("pwd", workdir="/somewhere/else")
        kwargs = terminal_tool.terminal_manager.exec_command.call_args.kwargs
        assert kwargs["workdir"] == "/somewhere/else"

    @pytest.mark.asyncio
    async def test_exec_command_running_returns_session_id(self, terminal_tool):
        terminal_tool.terminal_manager = Mock()
        terminal_tool.terminal_manager.exec_command = AsyncMock(
            return_value=ExecResult(status="running", session_id="s_1", output="partial")
        )

        data = json.loads(await terminal_tool.exec_command("sleep 5"))
        assert data["status"] == "running"
        assert data["session_id"] == "s_1"

    @pytest.mark.asyncio
    async def test_write_stdin_returns_json(self, terminal_tool):
        terminal_tool.terminal_manager = Mock()
        terminal_tool.terminal_manager.write_stdin = AsyncMock(
            return_value=ExecResult(status="exited", exit_code=0, output="done")
        )

        data = json.loads(await terminal_tool.write_stdin("s_1", "y\n"))
        assert data["status"] == "exited"
        assert data["output"] == "done"
        terminal_tool.terminal_manager.write_stdin.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_write_stdin_unknown_session_returns_error(self, terminal_tool):
        terminal_tool.terminal_manager = Mock()
        terminal_tool.terminal_manager.write_stdin = AsyncMock(side_effect=KeyError("No such session: s_9"))

        data = json.loads(await terminal_tool.write_stdin("s_9"))
        assert data["status"] == "error"

    @pytest.mark.asyncio
    async def test_kill_command_returns_json(self, terminal_tool):
        terminal_tool.terminal_manager = Mock()
        terminal_tool.terminal_manager.kill_session = AsyncMock(return_value=ExecResult(status="exited", exit_code=143))

        data = json.loads(await terminal_tool.kill_command("s_1", "TERM"))
        assert data["status"] == "exited"
        assert data["exit_code"] == 143
        terminal_tool.terminal_manager.kill_session.assert_awaited_once_with("s_1", "TERM")

    @pytest.mark.asyncio
    async def test_kill_command_unknown_session_returns_error(self, terminal_tool):
        terminal_tool.terminal_manager = Mock()
        terminal_tool.terminal_manager.kill_session = AsyncMock(side_effect=KeyError("No such session"))

        data = json.loads(await terminal_tool.kill_command("s_9"))
        assert data["status"] == "error"

    @pytest.mark.asyncio
    async def test_list_sessions_returns_json(self, terminal_tool):
        terminal_tool.terminal_manager = Mock()
        terminal_tool.terminal_manager.list_sessions = AsyncMock(
            return_value={"s_1": {"command": "sleep 5", "running": True}}
        )

        data = json.loads(await terminal_tool.list_sessions())
        assert "s_1" in data["sessions"]

    @pytest.mark.asyncio
    async def test_exec_command_blocked_by_security_check(self, terminal_tool):
        terminal_tool.security_check_enabled = True
        terminal_tool._run_command_security_check = AsyncMock(return_value=(False, "blocked: dangerous"))
        terminal_tool.terminal_manager = Mock()
        terminal_tool.terminal_manager.exec_command = AsyncMock()

        result = await terminal_tool.exec_command("rm -rf /")
        assert "blocked" in result
        terminal_tool.terminal_manager.exec_command.assert_not_awaited()
