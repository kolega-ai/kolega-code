"""Unit tests for terminal state serialization."""

from datetime import datetime
from unittest.mock import Mock
from kolega_code.sandbox.serializer import TerminalStateSerializer
from kolega_code.models.sandbox_terminal_state import SandboxTerminalState, TerminalInfo, TerminalOutput


class TestTerminalStateSerializer:
    """Test terminal state serialization and deserialization."""

    def test_serialize_empty_terminal_manager(self):
        """Test serializing an empty terminal manager."""
        terminal_manager = Mock()
        terminal_manager.terminals = {}
        terminal_manager.outputs = {}

        state = TerminalStateSerializer.serialize_to_model(terminal_manager, "workspace-123", "sandbox-456")

        assert state.workspace_id == "workspace-123"
        assert state.sandbox_id == "sandbox-456"
        assert len(state.terminals) == 0
        assert len(state.outputs) == 0
        assert state.total_output_size == 0

    def test_serialize_with_terminals(self):
        """Test serializing terminal manager with multiple terminals."""
        terminal_manager = Mock()
        terminal_manager.terminals = {
            "term1": {
                "created_at": datetime.now(),
                "cwd": "/home/user/workspace",
                "env": {"PATH": "/usr/bin"},
                "last_command": "ls -la",
                "last_command_purpose": "List files",
            },
            "term2": {
                "created_at": datetime.now(),
                "cwd": "/tmp",
                "env": {},
                "last_command": "pwd",
                "last_command_purpose": "Check directory",
            },
        }
        terminal_manager.outputs = {
            "term1": [
                {"type": "command", "data": "ls -la", "timestamp": datetime.now(), "purpose": "List files"},
                {
                    "type": "stdout",
                    "data": "total 24\ndrwxr-xr-x 2 user user 4096 Jan 1 12:00 .\n",
                    "timestamp": datetime.now(),
                },
            ],
            "term2": [
                {"type": "command", "data": "pwd", "timestamp": datetime.now(), "purpose": "Check directory"},
                {"type": "stdout", "data": "/tmp\n", "timestamp": datetime.now()},
            ],
        }
        terminal_manager._default_terminal_id = "term1"

        state = TerminalStateSerializer.serialize_to_model(terminal_manager, "workspace-123", "sandbox-456")

        assert len(state.terminals) == 2
        assert "term1" in state.terminals
        assert "term2" in state.terminals
        assert state.terminals["term1"].last_command == "ls -la"
        assert state.terminals["term2"].cwd == "/tmp"
        assert state.default_terminal_id == "term1"

        # Check outputs
        assert len(state.outputs["term1"]) == 2
        assert len(state.outputs["term2"]) == 2
        assert state.outputs["term1"][0].type == "command"
        assert state.outputs["term1"][1].type == "stdout"
        assert state.total_output_size > 0

    def test_serialize_with_size_limits(self):
        """Test that serialization respects size limits."""
        terminal_manager = Mock()
        terminal_manager.terminals = {
            "term1": {
                "created_at": datetime.now(),
                "cwd": "/home/user",
                "env": {},
                "last_command": "cat large_file.txt",
                "last_command_purpose": "View file",
            }
        }

        # Create large output that exceeds limit
        large_output = "x" * 300000  # 300KB, exceeds 256KB limit
        terminal_manager.outputs = {
            "term1": [
                {"type": "command", "data": "cat large_file.txt", "timestamp": datetime.now(), "purpose": "View file"},
                {"type": "stdout", "data": large_output, "timestamp": datetime.now()},
            ]
        }

        state = TerminalStateSerializer.serialize_to_model(terminal_manager, "workspace-123", "sandbox-456")

        # Should have truncation notice
        assert any(output.type == "truncation" for output in state.outputs["term1"])
        # Total size should be under limit
        assert state.total_output_size <= state.MAX_OUTPUT_SIZE

    def test_restore_from_model(self):
        """Test restoring terminal manager from model."""
        # Create a state model
        state = SandboxTerminalState(
            workspace_id="workspace-123",
            sandbox_id="sandbox-456",
            terminals={
                "term1": TerminalInfo(
                    terminal_id="term1",
                    created_at=datetime.now(),
                    cwd="/home/user/workspace",
                    env={"FOO": "bar"},
                    last_command="echo hello",
                    last_command_purpose="Test echo",
                )
            },
            outputs={
                "term1": [
                    TerminalOutput(type="command", data="echo hello", timestamp=datetime.now(), purpose="Test echo"),
                    TerminalOutput(type="stdout", data="hello\n", timestamp=datetime.now()),
                ]
            },
            default_terminal_id="term1",
        )

        # Create mock terminal manager
        terminal_manager = Mock()
        terminal_manager.terminals = {}
        terminal_manager.outputs = {}
        terminal_manager._default_terminal_id = None

        # Restore
        TerminalStateSerializer.restore_from_model(terminal_manager, state)

        # Verify restoration
        assert len(terminal_manager.terminals) == 1
        assert "term1" in terminal_manager.terminals
        assert terminal_manager.terminals["term1"]["cwd"] == "/home/user/workspace"
        assert terminal_manager.terminals["term1"]["env"]["FOO"] == "bar"
        assert terminal_manager.terminals["term1"]["last_command"] == "echo hello"
        assert terminal_manager.terminals["term1"]["process"] is None  # Can't restore process

        # Check outputs
        assert len(terminal_manager.outputs["term1"]) == 2
        assert terminal_manager.outputs["term1"][0]["type"] == "command"
        assert terminal_manager.outputs["term1"][1]["data"] == "hello\n"

        # Check default terminal
        assert terminal_manager._default_terminal_id == "term1"

    def test_to_frontend_format(self):
        """Test converting state to frontend format."""
        state = SandboxTerminalState(
            workspace_id="workspace-123",
            sandbox_id="sandbox-456",
            terminals={
                "term1": TerminalInfo(
                    terminal_id="term1",
                    created_at=datetime.now(),
                    cwd="/home/user",
                    env={},
                    last_command="ls",
                    last_command_purpose="",
                ),
                "term2": TerminalInfo(
                    terminal_id="term2",
                    created_at=datetime.now(),
                    cwd="/tmp",
                    env={},
                    last_command="pwd",
                    last_command_purpose="",
                ),
            },
            outputs={
                "term1": [
                    TerminalOutput(type="command", data="ls", timestamp=datetime.now()),
                    TerminalOutput(type="stdout", data="file1.txt\nfile2.txt\n", timestamp=datetime.now()),
                    TerminalOutput(
                        type="exit", data="Process exited with code 0", timestamp=datetime.now(), exit_code=0
                    ),
                ],
                "term2": [
                    TerminalOutput(type="command", data="pwd", timestamp=datetime.now()),
                    TerminalOutput(type="stdout", data="/tmp", timestamp=datetime.now()),
                ],
            },
        )

        frontend_data = TerminalStateSerializer.to_frontend_format(state)

        assert "terminals" in frontend_data
        assert len(frontend_data["terminals"]) == 2

        # Find terminals by ID
        term1_data = next(t for t in frontend_data["terminals"] if t["id"] == "term1")
        term2_data = next(t for t in frontend_data["terminals"] if t["id"] == "term2")

        # Check content formatting
        assert "$ ls" in term1_data["content"]
        assert "file1.txt\nfile2.txt" in term1_data["content"]
        assert "Process exited with code 0" in term1_data["content"]

        assert "$ pwd" in term2_data["content"]
        assert "/tmp" in term2_data["content"]

    def test_get_recent_outputs(self):
        """Test getting recent outputs with line limit."""
        outputs = []

        # Add many outputs
        for i in range(20):
            outputs.append({"type": "command", "data": f"echo line{i}", "timestamp": datetime.now()})
            outputs.append({"type": "stdout", "data": f"line{i}\n" * 10, "timestamp": datetime.now()})  # 10 lines each

        # Get recent outputs with limit
        recent = TerminalStateSerializer.get_recent_outputs(outputs, max_lines=50)

        # Should have truncated older outputs
        assert len(recent) < len(outputs)

        # Count total lines
        total_lines = 0
        for output in recent:
            if output["type"] == "command":
                total_lines += 1
            elif output["type"] in ["stdout", "stderr"]:
                total_lines += output["data"].count("\n") + 1

        # Should be close to limit (may be slightly over due to partial output)
        assert total_lines <= 60  # Some buffer for partial outputs

    def test_serialize_handles_missing_terminal_manager_attrs(self):
        """Test serialization handles terminal managers without expected attributes."""
        # Terminal manager without 'terminals' attribute (e.g., local terminal manager)
        terminal_manager = Mock(spec=[])  # No attributes

        state = TerminalStateSerializer.serialize_to_model(terminal_manager, "workspace-123", "sandbox-456")

        # Should return empty state
        assert state.workspace_id == "workspace-123"
        assert state.sandbox_id == "sandbox-456"
        assert len(state.terminals) == 0
        assert len(state.outputs) == 0

    def test_restore_handles_none_state(self):
        """Test restore handles None state gracefully."""
        terminal_manager = Mock()
        terminal_manager.terminals = {"existing": {}}
        terminal_manager.outputs = {"existing": []}

        # Should not raise exception
        TerminalStateSerializer.restore_from_model(terminal_manager, None)

        # Should not modify terminal manager
        assert "existing" in terminal_manager.terminals
        assert "existing" in terminal_manager.outputs
