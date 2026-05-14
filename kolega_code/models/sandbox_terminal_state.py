"""Sandbox terminal state model for persisting terminal sessions."""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import uuid

from pydantic import BaseModel, ConfigDict, Field


class TerminalInfo(BaseModel):
    """Information about a single terminal."""

    terminal_id: str = Field(..., description="Terminal identifier")
    created_at: datetime = Field(..., description="When the terminal was created")
    cwd: str = Field(..., description="Working directory of the terminal")
    env: Dict[str, str] = Field(default_factory=dict, description="Environment variables")
    last_command: str = Field(default="", description="Last command executed")
    last_command_purpose: str = Field(default="", description="Purpose of last command")


class TerminalOutput(BaseModel):
    """Single output entry from a terminal."""

    type: str = Field(..., description="Type of output: command, stdout, stderr, exit")
    data: str = Field(..., description="Output data")
    timestamp: datetime = Field(..., description="When the output was generated")
    purpose: Optional[str] = Field(None, description="Purpose for commands")
    exit_code: Optional[int] = Field(None, description="Exit code for exit type")


class SandboxTerminalState(BaseModel):
    """Model for persisting sandbox terminal state."""

    model_config = ConfigDict(populate_by_name=True, arbitrary_types_allowed=True)

    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique identifier")
    workspace_id: str = Field(..., description="Associated workspace ID")
    sandbox_id: str = Field(..., description="Associated sandbox ID")
    terminals: Dict[str, TerminalInfo] = Field(default_factory=dict)
    outputs: Dict[str, List[TerminalOutput]] = Field(default_factory=dict)
    default_terminal_id: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    total_output_size: int = 0
    MAX_OUTPUT_SIZE: int = 1048576
    MAX_OUTPUT_PER_TERMINAL: int = 262144

