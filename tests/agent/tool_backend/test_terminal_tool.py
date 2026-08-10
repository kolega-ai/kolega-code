import os
import asyncio
import shlex
import sys
from unittest.mock import AsyncMock, Mock, patch

import pytest
import uuid

from kolega_code.config import AgentConfig, ModelConfig, ModelProvider, RateLimitConfig
from kolega_code.events import AgentEvent
from kolega_code.services.base import ExecResult, TerminalCommandCancelled
from kolega_code.agent.tool_backend.terminal_tool import BACKGROUND_SETTLE_MS, TerminalTool

# Check if running in CI environment
SKIP_IN_CI = bool(os.getenv("CI")) or bool(os.getenv("GITLAB_CI"))


def _result_field(result: str, name: str) -> str:
    prefix = f"{name}: "
    return next(line.removeprefix(prefix) for line in result.splitlines() if line.startswith(prefix))


def _marker_command(marker, final_output: str) -> str:
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
    )


@pytest.fixture
def mock_base_agent():
    mock = Mock()
    mock.agent_name = "test_agent"
    # A bare Mock attribute would read as a truthy scratchpad dir; sessions
    # without a scratchpad are the default in these tests.
    mock.scratchpad_dir = None
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
            assert calls[1][0][0].content == {
                "command": 'echo "Hello World"',
                "workdir": str(terminal_tool.project_path.resolve()),
            }
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
    async def test_execute_terminal_command_env_scrubbed_and_clean(self, terminal_tool, mock_connection_manager):
        mock_process = AsyncMock()
        mock_process.communicate = AsyncMock(return_value=(b"", b""))

        # Simulate `uv run` from the checkout: the app's own venv leaks into the env.
        polluted = {"VIRTUAL_ENV": sys.prefix, "PATH": f"{sys.prefix}/bin{os.pathsep}" + os.environ.get("PATH", "")}
        with patch.dict(os.environ, polluted):
            with patch("asyncio.create_subprocess_shell", return_value=mock_process) as mock_create:
                await terminal_tool.execute_terminal_command("pwd")

        env = mock_create.call_args.kwargs["env"]
        assert "VIRTUAL_ENV" not in env
        assert f"{sys.prefix}/bin" not in env["PATH"].split(os.pathsep)
        # CLEAN_ENV overlay now applies to this path too
        assert env["TERM"] == "dumb"
        assert env["NO_COLOR"] == "1"

    @pytest.mark.asyncio
    async def test_execute_terminal_command_working_directory(self, terminal_tool, mock_connection_manager):
        # Mock the subprocess
        mock_process = AsyncMock()
        mock_process.communicate = AsyncMock(return_value=(b"stdout", b""))

        with patch("asyncio.create_subprocess_shell", return_value=mock_process) as mock_create:
            await terminal_tool.execute_terminal_command("pwd")

            # Verify the command was executed in the correct directory
            mock_create.assert_called_once()
            assert mock_create.call_args[1]["cwd"] == str(terminal_tool.project_path.resolve())


class TestUnifiedExecTools:
    """Tests for the codex-style exec_command / write_stdin / kill_command tools."""

    @pytest.mark.asyncio
    async def test_exec_command_returns_structured_text_and_defaults_workdir(self, terminal_tool):
        terminal_tool.terminal_manager = Mock()
        terminal_tool.terminal_manager.exec_command = AsyncMock(
            return_value=ExecResult(status="exited", exit_code=0, output="hi", duration_ms=12)
        )

        out = await terminal_tool.exec_command("echo hi")
        assert _result_field(out, "Status") == "exited"
        assert _result_field(out, "Exit code") == "0"
        assert "Output:\nhi" in out

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
        result = await terminal_tool.exec_command("pwd", workdir=".", yield_time_ms=5000)
        assert _result_field(result, "Status") == "exited"
        assert f"Output:\n{terminal_tool.project_path}" in result

    @pytest.mark.asyncio
    async def test_exec_command_injects_scratchpad_env(self, terminal_tool, mock_base_agent, tmp_path):
        mock_base_agent.scratchpad_dir = tmp_path / "scratchpad"
        terminal_tool.terminal_manager = Mock()
        terminal_tool.terminal_manager.exec_command = AsyncMock(
            return_value=ExecResult(status="exited", exit_code=0, output="", duration_ms=1)
        )

        await terminal_tool.exec_command("env")
        kwargs = terminal_tool.terminal_manager.exec_command.call_args.kwargs
        assert kwargs["env"] == {"KOLEGA_SCRATCHPAD": str(tmp_path / "scratchpad")}

    @pytest.mark.asyncio
    async def test_exec_command_without_scratchpad_injects_nothing(self, terminal_tool):
        terminal_tool.terminal_manager = Mock()
        terminal_tool.terminal_manager.exec_command = AsyncMock(
            return_value=ExecResult(status="exited", exit_code=0, output="", duration_ms=1)
        )

        await terminal_tool.exec_command("env")
        kwargs = terminal_tool.terminal_manager.exec_command.call_args.kwargs
        assert kwargs["env"] is None

    @pytest.mark.asyncio
    async def test_exec_command_end_to_end_scratchpad_env(self, terminal_tool, mock_base_agent, tmp_path):
        # Real PTY: the model-facing session sees $KOLEGA_SCRATCHPAD.
        mock_base_agent.scratchpad_dir = tmp_path / "scratchpad"
        result = await terminal_tool.exec_command('echo "SP=$KOLEGA_SCRATCHPAD"', yield_time_ms=5000)
        assert _result_field(result, "Status") == "exited"
        assert f"SP={tmp_path / 'scratchpad'}" in result

    @pytest.mark.asyncio
    async def test_write_stdin_session_keeps_scratchpad_env(self, terminal_tool, mock_base_agent, tmp_path):
        # A session created by exec_command and driven with write_stdin keeps
        # the env it was spawned with.
        mock_base_agent.scratchpad_dir = tmp_path / "scratchpad"
        result = await terminal_tool.exec_command(
            'read line; echo "GOT=$KOLEGA_SCRATCHPAD"',
            yield_time_ms=300,
        )
        assert _result_field(result, "Status") == "running"
        session_id = _result_field(result, "Session")
        result = await terminal_tool.write_stdin(session_id, "go\n", yield_time_ms=5000)
        assert _result_field(result, "Status") == "exited"
        assert f"GOT={tmp_path / 'scratchpad'}" in result

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

        result = await terminal_tool.exec_command("sleep 5")
        assert _result_field(result, "Status") == "running"
        assert _result_field(result, "Session") == "s_1"

    @pytest.mark.asyncio
    async def test_cancelled_exec_command_returns_structured_recovery_result(self, terminal_tool, tmp_path):
        spill_path = tmp_path / "terminal-output" / "000001.exec_command.log"
        terminal_tool.terminal_manager = Mock()
        terminal_tool.terminal_manager.exec_command = AsyncMock(
            side_effect=TerminalCommandCancelled(
                ExecResult(
                    status="cancelled",
                    exit_code=143,
                    output="partial",
                    truncated=True,
                    original_token_count=20_000,
                    spill_path=str(spill_path),
                    spill_bytes=80_000,
                )
            )
        )

        result = await terminal_tool.exec_command("produce forever")

        assert _result_field(result, "Status") == "cancelled"
        assert _result_field(result, "Exit code") == "143"
        assert f"Full output: {spill_path}" in result
        assert "Output:\npartial" in result

    @pytest.mark.asyncio
    async def test_structured_result_hard_caps_complete_text_and_preserves_spill_path(
        self,
        terminal_tool,
        tmp_path,
    ):
        spill_path = tmp_path / "terminal-output" / "000001.exec_command.log"
        terminal_tool.terminal_manager = Mock()
        terminal_tool.terminal_manager.exec_command = AsyncMock(
            return_value=ExecResult(
                status="exited",
                exit_code=0,
                output="x" * 100_000,
                truncated=True,
                original_token_count=25_000,
                spill_path=str(spill_path),
                spill_bytes=100_000,
                line_truncated_count=2,
                line_truncated_bytes=500,
                preview_omitted_bytes=60_000,
                preview_omitted_lines=600,
            )
        )

        result = await terminal_tool.exec_command(
            "produce output",
            max_output_tokens=1_000_000,
        )

        assert len(result) <= 40_000
        assert f"Full output: {spill_path}" in result
        assert "Output truncated: yes" in result
        assert "Original output: ~25,000 tokens" in result
        assert "Long lines shortened: 2 lines, 500 bytes omitted" in result
        assert "Preview middle omitted: 60,000 bytes across 600 lines" in result

    @pytest.mark.asyncio
    async def test_write_stdin_returns_structured_text(self, terminal_tool):
        terminal_tool.terminal_manager = Mock()
        terminal_tool.terminal_manager.write_stdin = AsyncMock(
            return_value=ExecResult(status="exited", exit_code=0, output="done")
        )

        result = await terminal_tool.write_stdin("s_1", "y\n")
        assert _result_field(result, "Status") == "exited"
        assert "Output:\ndone" in result
        terminal_tool.terminal_manager.write_stdin.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_cancelled_write_stdin_returns_structured_recovery_result(self, terminal_tool, tmp_path):
        spill_path = tmp_path / "terminal-output" / "000001.exec_command.log"
        terminal_tool.terminal_manager = Mock()
        terminal_tool.terminal_manager.write_stdin = AsyncMock(
            side_effect=TerminalCommandCancelled(
                ExecResult(
                    status="cancelled",
                    exit_code=137,
                    output="latest",
                    spill_path=str(spill_path),
                    spill_bytes=70_000,
                )
            )
        )

        result = await terminal_tool.write_stdin("s_1")

        assert _result_field(result, "Status") == "cancelled"
        assert _result_field(result, "Exit code") == "137"
        assert f"Full output: {spill_path}" in result
        assert "Output:\nlatest" in result

    @pytest.mark.asyncio
    async def test_write_stdin_unknown_session_returns_error(self, terminal_tool):
        terminal_tool.terminal_manager = Mock()
        terminal_tool.terminal_manager.write_stdin = AsyncMock(side_effect=KeyError("No such session: s_9"))

        result = await terminal_tool.write_stdin("s_9")
        assert _result_field(result, "Status") == "error"

    @pytest.mark.asyncio
    async def test_kill_command_returns_structured_text(self, terminal_tool):
        terminal_tool.terminal_manager = Mock()
        terminal_tool.terminal_manager.kill_session = AsyncMock(return_value=ExecResult(status="exited", exit_code=143))

        result = await terminal_tool.kill_command("s_1", "TERM")
        assert _result_field(result, "Status") == "exited"
        assert _result_field(result, "Exit code") == "143"
        terminal_tool.terminal_manager.kill_session.assert_awaited_once_with("s_1", "TERM")

    @pytest.mark.asyncio
    async def test_kill_command_unknown_session_returns_error(self, terminal_tool):
        terminal_tool.terminal_manager = Mock()
        terminal_tool.terminal_manager.kill_session = AsyncMock(side_effect=KeyError("No such session"))

        result = await terminal_tool.kill_command("s_9")
        assert _result_field(result, "Status") == "error"

    @pytest.mark.asyncio
    async def test_list_sessions_returns_structured_text(self, terminal_tool):
        terminal_tool.terminal_manager = Mock()
        terminal_tool.terminal_manager.list_sessions = AsyncMock(
            return_value={"s_1": {"command": "sleep 5", "running": True}}
        )

        result = await terminal_tool.list_sessions()
        assert "Session: s_1" in result
        assert "Command: sleep 5" in result

    @pytest.mark.asyncio
    async def test_list_sessions_empty_is_plain_text(self, terminal_tool):
        terminal_tool.terminal_manager = Mock()
        terminal_tool.terminal_manager.list_sessions = AsyncMock(return_value={})

        assert await terminal_tool.list_sessions() == "No running sessions."

    @pytest.mark.asyncio
    async def test_exec_command_blocked_by_security_check(self, terminal_tool):
        terminal_tool.security_check_enabled = True
        terminal_tool._run_command_security_check = AsyncMock(return_value=(False, "blocked: dangerous"))
        terminal_tool.terminal_manager = Mock()
        terminal_tool.terminal_manager.exec_command = AsyncMock()

        result = await terminal_tool.exec_command("rm -rf /")
        assert "blocked" in result
        terminal_tool.terminal_manager.exec_command.assert_not_awaited()


class TestBackgroundExec:
    """Tests for exec_command(background=True) dev-server style launches."""

    @pytest.mark.asyncio
    async def test_background_running_caps_yield_and_annotates_result(self, terminal_tool):
        terminal_tool.terminal_manager = Mock()
        terminal_tool.terminal_manager.exec_command = AsyncMock(
            return_value=ExecResult(status="running", session_id="s_1", output="ready", duration_ms=2000)
        )

        result = await terminal_tool.exec_command("npm run dev", background=True, yield_time_ms=30000)

        assert _result_field(result, "Status") == "running"
        assert _result_field(result, "Session") == "s_1"
        assert _result_field(result, "Background") == "true"
        assert "kill_command" in result
        kwargs = terminal_tool.terminal_manager.exec_command.call_args.kwargs
        assert kwargs["yield_time_ms"] == BACKGROUND_SETTLE_MS
        assert "s_1" in terminal_tool._background_sessions

    @pytest.mark.asyncio
    async def test_background_early_exit_returns_real_exit_code(self, terminal_tool):
        terminal_tool.terminal_manager = Mock()
        terminal_tool.terminal_manager.exec_command = AsyncMock(
            return_value=ExecResult(status="exited", exit_code=1, output="EADDRINUSE", duration_ms=300)
        )

        result = await terminal_tool.exec_command("npm run dev", background=True)

        assert _result_field(result, "Status") == "exited"
        assert _result_field(result, "Exit code") == "1"
        assert "Background:" not in result
        assert terminal_tool._background_sessions == set()

    @pytest.mark.asyncio
    async def test_default_exec_is_not_annotated_as_background(self, terminal_tool):
        terminal_tool.terminal_manager = Mock()
        terminal_tool.terminal_manager.exec_command = AsyncMock(
            return_value=ExecResult(status="running", session_id="s_2", output="", duration_ms=10000)
        )

        result = await terminal_tool.exec_command("sleep 5", yield_time_ms=30000)

        assert _result_field(result, "Status") == "running"
        assert "Background:" not in result
        kwargs = terminal_tool.terminal_manager.exec_command.call_args.kwargs
        assert kwargs["yield_time_ms"] == 30000
        assert terminal_tool._background_sessions == set()

    @pytest.mark.asyncio
    async def test_list_sessions_marks_background_and_prunes_dead(self, terminal_tool):
        terminal_tool.terminal_manager = Mock()
        terminal_tool.terminal_manager.exec_command = AsyncMock(
            return_value=ExecResult(status="running", session_id="s_1", output="", duration_ms=2000)
        )
        await terminal_tool.exec_command("npm run dev", background=True)
        terminal_tool._background_sessions.add("s_stale")

        terminal_tool.terminal_manager.list_sessions = AsyncMock(
            return_value={"s_1": {"command": "npm run dev", "running": True}}
        )
        result = await terminal_tool.list_sessions()

        assert "Session: s_1" in result
        assert "Background: true" in result
        # Ids no longer running are pruned from the tracking set.
        assert terminal_tool._background_sessions == {"s_1"}

    @pytest.mark.asyncio
    async def test_naturally_exited_background_session_is_hidden_but_final_output_remains(
        self,
        terminal_tool,
        tmp_path,
        monkeypatch,
    ):
        monkeypatch.setattr(
            "kolega_code.agent.tool_backend.terminal_tool.BACKGROUND_SETTLE_MS",
            250,
        )
        marker = tmp_path / "finish-background-tool"
        result = await terminal_tool.exec_command(
            _marker_command(marker, "background-final"),
            workdir=str(tmp_path),
            background=True,
        )
        assert _result_field(result, "Status") == "running"
        session_id = _result_field(result, "Session")
        session = terminal_tool.terminal_manager.sessions[session_id]

        marker.touch()
        await asyncio.wait_for(session.exited.wait(), timeout=3)

        assert await terminal_tool.list_sessions() == "No running sessions."
        assert terminal_tool._background_sessions == set()

        final = await terminal_tool.write_stdin(session_id, "", yield_time_ms=250)
        assert _result_field(final, "Status") == "exited"
        assert _result_field(final, "Exit code") == "0"
        assert "background-final" in final

    @pytest.mark.asyncio
    async def test_kill_command_forgets_background_session(self, terminal_tool):
        terminal_tool.terminal_manager = Mock()
        terminal_tool.terminal_manager.kill_session = AsyncMock(return_value=ExecResult(status="exited", exit_code=143))
        terminal_tool._background_sessions.add("s_1")

        await terminal_tool.kill_command("s_1", "TERM")

        assert terminal_tool._background_sessions == set()

    @pytest.mark.asyncio
    async def test_background_end_to_end_dev_server_pattern(self, terminal_tool):
        # Real PTY through the default manager: a backgrounded server process
        # stays alive across later tool calls, appears in list_sessions, and
        # dies via kill_command — the dev-server flow the browser agent needs.
        result = await terminal_tool.exec_command("sleep 30", background=True)
        assert _result_field(result, "Status") == "running"
        assert _result_field(result, "Background") == "true"
        session_id = _result_field(result, "Session")

        sessions = await terminal_tool.list_sessions()
        assert f"Session: {session_id}" in sessions
        assert "Background: true" in sessions

        killed = await terminal_tool.kill_command(session_id)
        assert _result_field(killed, "Status") == "exited"
        assert f"Session: {session_id}" not in await terminal_tool.list_sessions()
