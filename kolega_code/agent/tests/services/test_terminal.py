import asyncio
from unittest.mock import AsyncMock, Mock, patch

import pytest

from kolega_code.agent.services.terminal import AsyncPersistentTerminal, LocalTerminalManager


class TestAsyncPersistentTerminal:
    """Test class for AsyncPersistentTerminal read_output with offset functionality."""

    def test_read_output_with_offset(self):
        """Test read_output method with offset parameter."""
        terminal = AsyncPersistentTerminal(
            workspace_id="test_workspace",
            thread_id="test_thread",
            terminal_id="test_terminal",
            connection_manager=Mock(),
            auto_activate_venv=False,
        )

        # Create test output
        test_output = "0123456789ABCDEFGHIJ"  # 20 characters
        terminal.persistent_output_buffer = bytearray(test_output.encode())

        # Test default behavior (offset = 0)
        result = terminal.read_output(num_chars=5, offset=0)
        assert result == "FGHIJ"  # Last 5 characters

        # Test with offset = 3 (skip last 3 characters, read 5 before that)
        result = terminal.read_output(num_chars=5, offset=3)
        assert result == "CDEFG"  # Characters at indices 12-16 (skipping last 3: HIJ)

        # Test with offset = 10 (skip last 10 characters, read 5 before that)
        result = terminal.read_output(num_chars=5, offset=10)
        assert result == "56789"  # Characters at indices 5-9

        # Test edge case: offset + num_chars > total length
        result = terminal.read_output(num_chars=15, offset=5)
        assert result == "0123456789ABCDE"  # Should read from start to (total - offset)

        # Test edge case: offset >= total length
        result = terminal.read_output(num_chars=5, offset=25)
        assert result == ""  # Should return empty string

        # Test when trying to read more than available
        result = terminal.read_output(num_chars=10, offset=5)
        assert result == "56789ABCDE"  # Characters at indices 5-14 (skipping last 5: FGHIJ)

        # Test with empty buffer
        terminal.persistent_output_buffer = bytearray()
        result = terminal.read_output(num_chars=5, offset=3)
        assert result == ""

    def test_read_output_with_unicode_and_offset(self):
        """Test read_output method with offset parameter and unicode characters."""
        terminal = AsyncPersistentTerminal(
            workspace_id="test_workspace",
            thread_id="test_thread",
            terminal_id="test_terminal",
            connection_manager=Mock(),
            auto_activate_venv=False,
        )

        # Create test output with unicode characters
        test_output = "Hello 🌍 World 🚀 Test"  # Mix of ASCII and unicode
        terminal.persistent_output_buffer = bytearray(test_output.encode())

        # Test reading with offset (should handle unicode properly)
        result = terminal.read_output(num_chars=10, offset=5)
        expected_total_chars = len(test_output)
        expected_start = max(0, expected_total_chars - 5 - 10)
        expected_end = max(0, expected_total_chars - 5)
        expected = test_output[expected_start:expected_end]
        assert result == expected

    @pytest.mark.asyncio
    async def test_send_input_appends_newline_without_mutating_last_command(self):
        terminal = AsyncPersistentTerminal(
            workspace_id="test_workspace",
            thread_id="test_thread",
            terminal_id="test_terminal",
            connection_manager=Mock(),
            auto_activate_venv=False,
        )
        terminal.is_running = True
        terminal.master_fd = 123
        terminal.last_command = "python prompt.py\n"
        terminal.last_command_purpose = "Prompt test"

        with patch("kolega_code.agent.services.terminal.os.write") as mock_write:
            result = await terminal.send_input("Ada", submit=True)

        assert result is True
        mock_write.assert_called_once_with(123, b"Ada\n")
        assert terminal.last_command == "python prompt.py\n"
        assert terminal.last_command_purpose == "Prompt test"

    @pytest.mark.asyncio
    async def test_send_input_can_send_raw_text(self):
        terminal = AsyncPersistentTerminal(
            workspace_id="test_workspace",
            thread_id="test_thread",
            terminal_id="test_terminal",
            connection_manager=Mock(),
            auto_activate_venv=False,
        )
        terminal.is_running = True
        terminal.master_fd = 123

        with patch("kolega_code.agent.services.terminal.os.write") as mock_write:
            result = await terminal.send_input("A", submit=False)

        assert result is True
        mock_write.assert_called_once_with(123, b"A")


@pytest.mark.asyncio
async def test_local_terminal_manager_can_answer_python_prompt(tmp_path):
    manager = LocalTerminalManager("test_workspace", "test_thread", AsyncMock())
    terminal_id = await manager.launch_terminal(cwd=tmp_path, auto_activate_venv=False)
    command_id = None

    try:
        command_id = await manager.send_command_tracked(
            terminal_id,
            'python -c "prompt = \'READY_\' + \'FOR_\' + \'INPUT>\'; name = input(prompt); print(\'hello \' + name)"',
            "Prompt for a name and echo it",
        )
        assert command_id

        prompt_seen = False
        for _ in range(50):
            if "READY_FOR_INPUT>" in manager.read_output(terminal_id, num_chars=500):
                prompt_seen = True
                break
            await asyncio.sleep(0.1)

        assert prompt_seen
        assert await manager.send_input(terminal_id, "Ada", command_id=command_id)

        completed = False
        for _ in range(50):
            status = manager.get_command_status(terminal_id, command_id)
            if status["status"] == "completed":
                completed = True
                break
            await asyncio.sleep(0.1)

        assert completed
        assert "hello Ada" in manager.read_output(terminal_id, num_chars=1000)
    finally:
        if terminal_id in manager.terminals:
            await manager.close_terminal(terminal_id)
