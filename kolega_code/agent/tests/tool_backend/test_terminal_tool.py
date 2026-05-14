import os
import asyncio
from unittest.mock import AsyncMock, Mock, patch

import pytest
import uuid

from kolega_code.agent.config import AgentConfig, ModelConfig, ModelProvider, RateLimitConfig
from kolega_code.agent.models.public import AgentEvent
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
            provider=ModelProvider.ANTHROPIC, model="test-model", rate_limits=RateLimitConfig(), thinking_tokens=1024
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
    async def test_execute_terminal_command_terminate_error(self, terminal_tool, mock_connection_manager):
        # Mock the subprocess to simulate a timeout and raise error on terminate
        mock_process = AsyncMock()
        mock_process.communicate = AsyncMock(side_effect=asyncio.TimeoutError())
        mock_process.terminate = Mock(side_effect=Exception("Terminate error"))

        with patch("asyncio.create_subprocess_shell", return_value=mock_process):
            result = await terminal_tool.execute_terminal_command("slow command")

            assert result == "Command timed out after 15 seconds"
            mock_process.terminate.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_terminal_command_empty_output(self, terminal_tool, mock_connection_manager):
        # Mock the subprocess with empty output
        mock_process = AsyncMock()
        mock_process.communicate = AsyncMock(return_value=(b"", b""))

        with patch("asyncio.create_subprocess_shell", return_value=mock_process):
            result = await terminal_tool.execute_terminal_command("empty command")

            assert result == ""

    @pytest.mark.asyncio
    async def test_execute_terminal_command_unicode_output(self, terminal_tool, mock_connection_manager):
        # Mock the subprocess with unicode output
        mock_process = AsyncMock()
        mock_process.communicate = AsyncMock(return_value=(b"Hello \xe2\x9c\xa8", b""))

        with patch("asyncio.create_subprocess_shell", return_value=mock_process):
            result = await terminal_tool.execute_terminal_command("unicode command")

            assert result == "Hello ✨"

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


class TestTerminalToolCommandTracking:
    """Tests for the command tracking functionality"""

    @pytest.mark.asyncio
    async def test_run_command_tracked_success(self, terminal_tool, mock_connection_manager):
        # Mock the terminal manager
        mock_terminal_manager = AsyncMock()
        mock_terminal_manager.send_command_tracked = AsyncMock(return_value="terminal_1_1")
        terminal_tool.terminal_manager = mock_terminal_manager

        result = await terminal_tool.run_command_tracked("terminal_1", "echo test", "Test command")

        assert result == "terminal_1_1"
        mock_terminal_manager.send_command_tracked.assert_called_once_with(
            "terminal_1", "echo test", "Test command", timeout=0
        )

    @pytest.mark.asyncio
    async def test_run_command_tracked_failure(self, terminal_tool, mock_connection_manager):
        # Mock the terminal manager to return None (failure)
        mock_terminal_manager = AsyncMock()
        mock_terminal_manager.send_command_tracked = AsyncMock(return_value=None)
        terminal_tool.terminal_manager = mock_terminal_manager

        result = await terminal_tool.run_command_tracked("terminal_1", "echo test", "Test command")

        assert "Failed to start command `echo test` in terminal terminal_1" in result

    @pytest.mark.asyncio
    async def test_run_command_tracked_terminal_not_found(self, terminal_tool, mock_connection_manager):
        # Mock the terminal manager to raise KeyError
        mock_terminal_manager = AsyncMock()
        mock_terminal_manager.send_command_tracked = AsyncMock(
            side_effect=KeyError("Terminal with ID invalid not found")
        )
        terminal_tool.terminal_manager = mock_terminal_manager

        result = await terminal_tool.run_command_tracked("invalid", "echo test", "Test command")

        assert "Terminal with ID invalid not found" in result

    @pytest.mark.asyncio
    async def test_check_command_status_running(self, terminal_tool, mock_connection_manager):
        # Mock the terminal manager
        mock_terminal_manager = Mock()
        mock_terminal_manager.get_command_status = Mock(
            return_value={"status": "running", "command": "sleep 10", "duration": 5.2, "child_pids": [1234, 5678]}
        )
        terminal_tool.terminal_manager = mock_terminal_manager

        result = await terminal_tool.check_command_status("terminal_1", "cmd_1")

        assert "🔄 Command still running after 5.2s (2 child processes)" in result
        assert "Command: sleep 10" in result

    @pytest.mark.asyncio
    async def test_check_command_status_completed(self, terminal_tool, mock_connection_manager):
        # Mock the terminal manager
        mock_terminal_manager = Mock()
        mock_terminal_manager.get_command_status = Mock(
            return_value={
                "status": "completed",
                "command": "echo test",
                "duration": 1.5,
                "return_code": 0,
                "child_pids": [],
            }
        )
        terminal_tool.terminal_manager = mock_terminal_manager

        result = await terminal_tool.check_command_status("terminal_1", "cmd_1")

        assert "✅ Command completed in 1.5s with exit code 0" in result
        assert "Command: echo test" in result

    @pytest.mark.asyncio
    async def test_check_command_status_terminated(self, terminal_tool, mock_connection_manager):
        # Mock the terminal manager
        mock_terminal_manager = Mock()
        mock_terminal_manager.get_command_status = Mock(
            return_value={
                "status": "terminated",
                "command": "invalid_command",
                "duration": 0.8,
                "return_code": -1,
                "child_pids": [],
            }
        )
        terminal_tool.terminal_manager = mock_terminal_manager

        result = await terminal_tool.check_command_status("terminal_1", "cmd_1")

        assert "❌ Command terminated after 0.8s" in result
        assert "Command: invalid_command" in result

    @pytest.mark.asyncio
    async def test_check_command_status_not_found(self, terminal_tool, mock_connection_manager):
        # Mock the terminal manager
        mock_terminal_manager = Mock()
        mock_terminal_manager.get_command_status = Mock(return_value={"status": "not_found"})
        terminal_tool.terminal_manager = mock_terminal_manager

        result = await terminal_tool.check_command_status("terminal_1", "invalid_cmd")

        assert "❌ Command ID invalid_cmd not found" in result

    @pytest.mark.asyncio
    async def test_check_terminal_status_with_active_commands(self, terminal_tool, mock_connection_manager):
        # Mock the terminal manager
        mock_terminal_manager = AsyncMock()
        mock_terminal_manager.get_terminal_status = AsyncMock(
            return_value={
                "running": True,
                "ready_for_commands": False,
                "active_commands": {
                    "terminal_1_1": {"command": "sleep 30", "duration": 15.3},
                    "terminal_1_2": {"command": "npm test", "duration": 45.7},
                },
                "last_command": "npm test",
            }
        )
        terminal_tool.terminal_manager = mock_terminal_manager

        result = await terminal_tool.check_terminal_status("terminal_1")

        assert "# Terminal terminal_1 Status" in result
        assert "**Running:** Yes" in result
        assert "**Ready for new commands:** No" in result
        assert "`terminal_1_1`: sleep 30 (running 15.3s)" in result
        assert "`terminal_1_2`: npm test (running 45.7s)" in result

    @pytest.mark.asyncio
    async def test_check_terminal_status_no_active_commands(self, terminal_tool, mock_connection_manager):
        # Mock the terminal manager
        mock_terminal_manager = AsyncMock()
        mock_terminal_manager.get_terminal_status = AsyncMock(
            return_value={"running": True, "ready_for_commands": True, "active_commands": {}, "last_command": None}
        )
        terminal_tool.terminal_manager = mock_terminal_manager

        result = await terminal_tool.check_terminal_status("terminal_1")

        assert "**Ready for new commands:** Yes" in result
        assert "**Active Commands:** None" in result

    @pytest.mark.asyncio
    async def test_wait_for_command_completion_success(self, terminal_tool, mock_connection_manager):
        # Mock the terminal manager to return completed status
        mock_terminal_manager = Mock()
        mock_terminal_manager.get_command_status = Mock(
            return_value={"status": "completed", "command": "echo test", "duration": 2.1, "return_code": 0}
        )
        terminal_tool.terminal_manager = mock_terminal_manager

        result = await terminal_tool.wait_for_command_completion("terminal_1", "cmd_1", timeout=5)

        assert "✅ Command completed in 2.1s with exit code 0" in result

    @pytest.mark.asyncio
    @pytest.mark.skipif(SKIP_IN_CI, reason="Skipping slow test in CI environment")
    async def test_wait_for_command_completion_timeout(self, terminal_tool, mock_connection_manager):
        # Mock the terminal manager to always return running status
        mock_terminal_manager = Mock()
        mock_terminal_manager.get_command_status = Mock(
            return_value={"status": "running", "command": "sleep 100", "duration": 10.0}
        )
        terminal_tool.terminal_manager = mock_terminal_manager

        result = await terminal_tool.wait_for_command_completion("terminal_1", "cmd_1", timeout=1)

        assert "⏰ Timeout: Command cmd_1 still running after 1 seconds" in result

    @pytest.mark.asyncio
    async def test_wait_for_command_completion_terminal_not_found(self, terminal_tool, mock_connection_manager):
        # Mock the terminal manager to raise KeyError
        mock_terminal_manager = Mock()
        mock_terminal_manager.get_command_status = Mock(side_effect=KeyError("Terminal with ID invalid not found"))
        terminal_tool.terminal_manager = mock_terminal_manager

        result = await terminal_tool.wait_for_command_completion("invalid", "cmd_1", timeout=5)

        assert "Terminal with ID invalid not found" in result

    @pytest.mark.asyncio
    async def test_read_terminal_with_offset_skips_compression(self, terminal_tool, mock_connection_manager):
        """Test that read_terminal with offset > 0 skips compression even for large output."""
        # Create a large output that would normally trigger compression
        large_output = "A" * 5000  # Exceeds the 4000 character threshold
        terminal_tool.terminal_manager.read_output = Mock(return_value=large_output)
        terminal_tool.terminal_manager.get_last_command = AsyncMock(return_value="test command")
        terminal_tool.terminal_manager.get_last_command_purpose = AsyncMock(return_value="test purpose")

        # Test with offset = 0 (should compress)
        result_no_offset = await terminal_tool.read_terminal("test_terminal", num_chars=5000, offset=0)
        assert "OUTPUT COMPRESSED" in result_no_offset
        assert "offset parameter" in result_no_offset

        # Test with offset > 0 (should not compress)
        result_with_offset = await terminal_tool.read_terminal("test_terminal", num_chars=1000, offset=100)
        assert "OUTPUT COMPRESSED" not in result_with_offset
        assert result_with_offset.startswith("```\n")
        assert result_with_offset.endswith("```\n")

    @pytest.mark.asyncio
    async def test_read_terminal_offset_parameter_passed_through(self, terminal_tool, mock_connection_manager):
        """Test that the offset parameter is correctly passed through to the terminal manager."""
        terminal_tool.terminal_manager.read_output = Mock(return_value="test output")

        # Test with various offset values
        await terminal_tool.read_terminal("test_terminal", num_chars=500, offset=0)
        terminal_tool.terminal_manager.read_output.assert_called_with("test_terminal", num_chars=500, offset=0)

        await terminal_tool.read_terminal("test_terminal", num_chars=200, offset=50)
        terminal_tool.terminal_manager.read_output.assert_called_with("test_terminal", num_chars=200, offset=50)

        await terminal_tool.read_terminal("test_terminal", num_chars=1000, offset=100)
        terminal_tool.terminal_manager.read_output.assert_called_with("test_terminal", num_chars=1000, offset=100)
