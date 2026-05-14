import os
import asyncio
import time
from unittest.mock import AsyncMock, Mock, patch, call

import pytest
import psutil


# Check if running in CI environment
SKIP_IN_CI = bool(os.getenv("CI")) or bool(os.getenv("GITLAB_CI"))
from kolega_code.agent.services.terminal import AsyncPersistentTerminal, LocalTerminalManager


@pytest.fixture
def mock_connection_manager():
    return AsyncMock()


@pytest.fixture
def workspace_id():
    return "test_workspace"


@pytest.fixture
def terminal_id():
    return "test_terminal"


class TestAsyncPersistentTerminalCommandTracking:
    """Tests for command tracking in AsyncPersistentTerminal"""

    @pytest.fixture
    def mock_terminal(self, workspace_id, terminal_id, mock_connection_manager):
        """Create a mock terminal without actually starting it"""
        with patch("kolega_code.agent.services.terminal.pty.fork"), patch(
            "kolega_code.agent.services.terminal.os.execvpe"
        ), patch("kolega_code.agent.services.terminal.fcntl.fcntl"):
            # Don't patch asyncio.create_task here - let each test handle it

            terminal = AsyncPersistentTerminal(
                workspace_id=workspace_id,
                thread_id="test_thread",
                terminal_id=terminal_id,
                connection_manager=mock_connection_manager,
                cwd="/tmp",
                auto_activate_venv=False,
            )

            # Mock the required attributes for testing
            terminal.is_running = True
            terminal.master_fd = 123
            terminal.pid = 12345
            terminal.shell_cleaned = True

            return terminal

    def test_command_tracking_initialization(self, mock_terminal):
        """Test that command tracking attributes are initialized correctly"""
        assert mock_terminal.active_commands == {}
        assert mock_terminal.command_counter == 0
        assert mock_terminal.shell_prompt_detected == True

    @pytest.mark.asyncio
    async def test_send_command_tracked_success(self, mock_terminal):
        """Test successful command tracking"""
        # Mock the send_command method
        mock_terminal.send_command = AsyncMock(return_value=True)

        # Mock the monitoring task creation with a function that properly handles the coroutine
        def mock_create_task(coro):
            # Close the coroutine to avoid warnings
            if hasattr(coro, "close"):
                coro.close()
            return Mock()

        with patch("asyncio.create_task", side_effect=mock_create_task) as mock_create_task_patch:
            command_id = await mock_terminal.send_command_tracked("echo test", "Test purpose")

            assert command_id == "test_terminal_1"
            assert "test_terminal_1" in mock_terminal.active_commands

            command_info = mock_terminal.active_commands["test_terminal_1"]
            assert command_info["command"] == "echo test"
            assert command_info["purpose"] == "Test purpose"
            assert command_info["status"] == "running"
            assert command_info["child_pids"] == set()
            assert command_info["return_code"] is None
            assert isinstance(command_info["start_time"], float)

            # Verify monitoring task was created
            assert mock_create_task_patch.call_count == 1
            assert mock_terminal.shell_prompt_detected == False

    @pytest.mark.asyncio
    async def test_send_command_tracked_failure(self, mock_terminal):
        """Test command tracking when send fails"""
        # Mock the send_command method to fail
        mock_terminal.send_command = AsyncMock(return_value=False)

        command_id = await mock_terminal.send_command_tracked("echo test", "Test purpose")

        assert command_id is None
        assert len(mock_terminal.active_commands) == 0

    @pytest.mark.asyncio
    async def test_send_command_tracked_terminal_not_running(self, mock_terminal):
        """Test command tracking when terminal is not running"""
        mock_terminal.is_running = False

        command_id = await mock_terminal.send_command_tracked("echo test", "Test purpose")

        assert command_id is None
        assert len(mock_terminal.active_commands) == 0

    def test_get_command_status_running(self, mock_terminal):
        """Test getting status of a running command"""
        # Add a test command
        start_time = time.time()
        mock_terminal.active_commands["test_cmd"] = {
            "command": "sleep 10",
            "purpose": "Test sleep",
            "start_time": start_time,
            "status": "running",
            "child_pids": {1234, 5678},
            "return_code": None,
        }

        status = mock_terminal.get_command_status("test_cmd")

        assert status["status"] == "running"
        assert status["command"] == "sleep 10"
        assert status["purpose"] == "Test sleep"
        assert status["duration"] >= 0
        assert status["return_code"] is None
        assert set(status["child_pids"]) == {1234, 5678}

    def test_get_command_status_not_found(self, mock_terminal):
        """Test getting status of non-existent command"""
        status = mock_terminal.get_command_status("nonexistent")

        assert status["status"] == "not_found"

    def test_get_active_commands(self, mock_terminal):
        """Test getting all active commands"""
        # Add some test commands
        mock_terminal.active_commands = {
            "cmd1": {
                "command": "running_cmd",
                "start_time": time.time(),
                "status": "running",
                "child_pids": set(),
                "return_code": None,
            },
            "cmd2": {
                "command": "completed_cmd",
                "start_time": time.time() - 10,
                "status": "completed",
                "child_pids": set(),
                "return_code": 0,
            },
        }

        active = mock_terminal.get_active_commands()

        assert len(active) == 1
        assert "cmd1" in active
        assert "cmd2" not in active

    def test_is_ready_for_commands_true(self, mock_terminal):
        """Test terminal ready state when no active commands"""
        mock_terminal.shell_prompt_detected = True
        mock_terminal.active_commands = {}

        assert mock_terminal.is_ready_for_commands() == True

    def test_is_ready_for_commands_false_no_prompt(self, mock_terminal):
        """Test terminal not ready when no prompt detected"""
        mock_terminal.shell_prompt_detected = False
        mock_terminal.active_commands = {}

        assert mock_terminal.is_ready_for_commands() == False

    def test_is_ready_for_commands_false_active_commands(self, mock_terminal):
        """Test terminal not ready when commands are active"""
        mock_terminal.shell_prompt_detected = True
        mock_terminal.active_commands = {
            "cmd1": {
                "command": "running_cmd",
                "status": "running",
                "start_time": time.time(),
                "child_pids": set(),
                "return_code": None,
            }
        }

        assert mock_terminal.is_ready_for_commands() == False

    def test_check_for_shell_prompt_detected(self, mock_terminal):
        """Test shell prompt detection"""
        # Mock read_output to return output with prompt
        mock_terminal.read_output = Mock(return_value="some output\n$ ")

        assert mock_terminal._check_for_shell_prompt() == True

    def test_check_for_shell_prompt_not_detected(self, mock_terminal):
        """Test shell prompt not detected"""
        # Mock read_output to return output without prompt
        mock_terminal.read_output = Mock(return_value="some output\nstill processing...")

        assert mock_terminal._check_for_shell_prompt() == False

    @pytest.mark.asyncio
    @pytest.mark.skipif(SKIP_IN_CI, reason="Skipping slow test in CI environment")
    async def test_monitor_command_completion_success(self, mock_terminal):
        """Test command completion monitoring"""
        # Set up a test command
        command_id = "test_cmd"
        mock_terminal.active_commands[command_id] = {
            "command": "echo test",
            "start_time": time.time(),
            "status": "running",
            "child_pids": set(),
            "return_code": None,
        }

        # Mock psutil to simulate no child processes
        with patch("psutil.Process") as mock_process_class:
            mock_process = Mock()
            mock_process.children.return_value = []
            mock_process_class.return_value = mock_process

            # Mock prompt detection to return True
            mock_terminal._check_for_shell_prompt = Mock(return_value=True)

            # Run the monitoring
            await mock_terminal._monitor_command_completion(command_id)

            # Check that command was marked as completed
            assert mock_terminal.active_commands[command_id]["status"] == "completed"
            assert mock_terminal.active_commands[command_id]["return_code"] == 0
            assert mock_terminal.shell_prompt_detected == True

    @pytest.mark.asyncio
    @pytest.mark.skipif(SKIP_IN_CI, reason="Skipping slow test in CI environment")
    async def test_monitor_command_completion_process_died(self, mock_terminal):
        """Test command completion monitoring when shell process dies"""
        command_id = "test_cmd"
        mock_terminal.active_commands[command_id] = {
            "command": "echo test",
            "start_time": time.time(),
            "status": "running",
            "child_pids": set(),
            "return_code": None,
        }

        # Mock psutil to raise NoSuchProcess
        with patch("psutil.Process", side_effect=psutil.NoSuchProcess(123)):
            await mock_terminal._monitor_command_completion(command_id)

            # Check that command was marked as terminated
            assert mock_terminal.active_commands[command_id]["status"] == "terminated"
            assert mock_terminal.active_commands[command_id]["return_code"] == -1


class TestLocalTerminalManagerCommandTracking:
    """Tests for command tracking in LocalTerminalManager"""

    @pytest.fixture
    def terminal_manager(self, workspace_id, mock_connection_manager):
        return LocalTerminalManager(workspace_id, "test-thread-id", mock_connection_manager)

    @pytest.fixture
    def mock_terminal(self):
        terminal = Mock()
        terminal.send_command_tracked = AsyncMock(return_value="terminal_1_1")
        terminal.get_command_status = Mock(return_value={"status": "running"})
        terminal.is_alive = AsyncMock(return_value=True)
        terminal.is_ready_for_commands = Mock(return_value=True)
        terminal.get_active_commands = Mock(return_value={})
        terminal.last_command = "echo test"
        return terminal

    @pytest.mark.asyncio
    async def test_send_command_tracked_success(self, terminal_manager, mock_terminal):
        """Test sending tracked command through manager"""
        terminal_manager.terminals["terminal_1"] = mock_terminal

        command_id = await terminal_manager.send_command_tracked("terminal_1", "echo test", "Test purpose")

        assert command_id == "terminal_1_1"
        mock_terminal.send_command_tracked.assert_called_once_with("echo test", "Test purpose")

    @pytest.mark.asyncio
    async def test_send_command_tracked_terminal_not_found(self, terminal_manager):
        """Test sending tracked command to non-existent terminal"""
        with pytest.raises(KeyError, match="Terminal with ID invalid not found"):
            await terminal_manager.send_command_tracked("invalid", "echo test", "Test purpose")

    def test_get_command_status_success(self, terminal_manager, mock_terminal):
        """Test getting command status through manager"""
        terminal_manager.terminals["terminal_1"] = mock_terminal

        status = terminal_manager.get_command_status("terminal_1", "cmd_1")

        assert status["status"] == "running"
        mock_terminal.get_command_status.assert_called_once_with("cmd_1")

    def test_get_command_status_terminal_not_found(self, terminal_manager):
        """Test getting command status from non-existent terminal"""
        with pytest.raises(KeyError, match="Terminal with ID invalid not found"):
            terminal_manager.get_command_status("invalid", "cmd_1")

    @pytest.mark.asyncio
    async def test_get_terminal_status_success(self, terminal_manager, mock_terminal):
        """Test getting terminal status through manager"""
        terminal_manager.terminals["terminal_1"] = mock_terminal

        status = await terminal_manager.get_terminal_status("terminal_1")

        assert status["running"] == True
        assert status["ready_for_commands"] == True
        assert status["active_commands"] == {}
        assert status["last_command"] == "echo test"

    @pytest.mark.asyncio
    async def test_get_terminal_status_terminal_not_found(self, terminal_manager):
        """Test getting status from non-existent terminal"""
        with pytest.raises(KeyError, match="Terminal with ID invalid not found"):
            await terminal_manager.get_terminal_status("invalid")
