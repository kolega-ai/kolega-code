"""Terminal manager state serialization utilities."""

from typing import Dict, Any, List
from datetime import datetime, timezone
from kolega_code.models.sandbox_terminal_state import SandboxTerminalState, TerminalInfo, TerminalOutput


class TerminalStateSerializer:
    """Handles serialization and deserialization of terminal manager state."""

    @staticmethod
    def serialize_to_model(terminal_manager, workspace_id: str, sandbox_id: str) -> SandboxTerminalState:
        """Serialize terminal manager state to a SandboxTerminalState model."""
        state = SandboxTerminalState(workspace_id=workspace_id, sandbox_id=sandbox_id)

        if not hasattr(terminal_manager, "terminals"):
            return state  # Can't serialize local terminal managers

        # Serialize terminals
        for terminal_id, info in terminal_manager.terminals.items():
            state.terminals[terminal_id] = TerminalInfo(
                terminal_id=terminal_id,
                created_at=info["created_at"],
                cwd=info["cwd"],
                env=info["env"],
                last_command=info.get("last_command", ""),
                last_command_purpose=info.get("last_command_purpose", ""),
            )

        # Serialize outputs with size tracking
        total_size = 0
        for terminal_id, outputs in terminal_manager.outputs.items():
            terminal_outputs = []
            terminal_size = 0

            # Process outputs in reverse order (keep most recent)
            for output in reversed(outputs):
                terminal_output = TerminalOutput(
                    type=output["type"],
                    data=output["data"],
                    timestamp=output["timestamp"],
                    purpose=output.get("purpose"),
                    exit_code=output.get("exit_code"),
                )
                output_size = len(output["data"].encode("utf-8"))

                # Check if adding this output would exceed limits
                if (
                    terminal_size + output_size > state.MAX_OUTPUT_PER_TERMINAL
                    or total_size + output_size > state.MAX_OUTPUT_SIZE
                ):
                    # Add truncation notice at the beginning
                    if not any(o.type == "truncation" for o in terminal_outputs):
                        terminal_outputs.insert(
                            0,
                            TerminalOutput(
                                type="truncation",
                                data="[Earlier terminal output truncated due to size limits]",
                                timestamp=datetime.now(timezone.utc),
                            ),
                        )
                    break

                terminal_outputs.insert(0, terminal_output)  # Insert at beginning to maintain order
                terminal_size += output_size
                total_size += output_size

            state.outputs[terminal_id] = terminal_outputs

            # Break if we've hit total size limit
            if total_size >= state.MAX_OUTPUT_SIZE:
                break

        state.total_output_size = total_size
        state.default_terminal_id = getattr(terminal_manager, "_default_terminal_id", None)

        return state

    @staticmethod
    def restore_from_model(terminal_manager, state: SandboxTerminalState) -> None:
        """Restore terminal manager state from a SandboxTerminalState model."""
        if not state or not hasattr(terminal_manager, "terminals"):
            return

        # Clear existing state
        terminal_manager.terminals.clear()
        terminal_manager.outputs.clear()

        # Restore terminals
        for terminal_id, terminal_info in state.terminals.items():
            terminal_manager.terminals[terminal_id] = {
                "created_at": terminal_info.created_at,
                "cwd": terminal_info.cwd,
                "env": terminal_info.env,
                "process": None,  # Process references can't be restored
                "last_command": terminal_info.last_command,
                "last_command_purpose": terminal_info.last_command_purpose,
                "active_commands": {},  # Active commands start fresh
            }

        # Restore outputs
        for terminal_id, outputs in state.outputs.items():
            terminal_manager.outputs[terminal_id] = [
                {
                    "type": output.type,
                    "data": output.data,
                    "timestamp": output.timestamp,
                    "purpose": output.purpose,
                    "exit_code": output.exit_code,
                }
                for output in outputs
                if output.type != "truncation"  # Skip truncation notices
            ]

        # Restore default terminal ID
        if hasattr(terminal_manager, "_default_terminal_id"):
            terminal_manager._default_terminal_id = state.default_terminal_id

    @staticmethod
    def to_frontend_format(state: SandboxTerminalState) -> Dict[str, Any]:
        """Convert terminal state to frontend format."""
        terminal_tabs = []

        for terminal_id, outputs in state.outputs.items():
            content = ""
            for output in outputs:
                if output.type == "command":
                    content += f"$ {output.data}\n"
                elif output.type == "truncation":
                    content += f"{output.data}\n"
                elif output.type in ["stdout", "stderr"]:
                    content += output.data
                    if not output.data.endswith("\n"):
                        content += "\n"
                elif output.type == "exit":
                    content += f"{output.data}\n"

            if content:  # Only add tabs with content
                terminal_tabs.append({"id": terminal_id, "content": content.rstrip()})  # Remove trailing whitespace

        return {"terminals": terminal_tabs}

    @staticmethod
    def get_recent_outputs(outputs: List[Dict[str, Any]], max_lines: int = 100) -> List[Dict[str, Any]]:
        """Get the most recent outputs, limited by line count."""
        if not outputs:
            return []

        recent_outputs = []
        line_count = 0

        # Process outputs in reverse order
        for output in reversed(outputs):
            if output["type"] in ["stdout", "stderr"]:
                lines = output["data"].count("\n") + 1
                if line_count + lines > max_lines:
                    # Partial output to fit within limit
                    remaining_lines = max_lines - line_count
                    if remaining_lines > 0:
                        lines_data = output["data"].split("\n")
                        partial_output = output.copy()
                        partial_output["data"] = "\n".join(lines_data[-remaining_lines:])
                        recent_outputs.insert(0, partial_output)
                    break
                line_count += lines

            recent_outputs.insert(0, output)

            if output["type"] == "command":
                line_count += 1

            if line_count >= max_lines:
                break

        return recent_outputs
