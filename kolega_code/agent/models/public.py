"""Models for WebSocket messages."""

import uuid
from datetime import datetime
from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field


class AgentEvent(BaseModel):
    uuid: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    event_type: Literal[
        "system_message",
        "chat_message",
        "log_message",
        "terminal_command",
        "terminal_output",
        "terminal_launched",
        "terminal_closed",
        "browser_launched",
        "browser_closed",
        "status_update",
        "llm_status_update",
        "credit_alert",
        "llm_context_update",
        "tool_streaming_update",
        "memory_suggestions",
    ]
    sender: str
    recipient: Optional[str] = None
    content: dict = Field(default_factory=dict)
    is_streaming: bool = False
    sub_agent_info: Optional[dict] = None


class AgentStatus(Enum):
    """
    Enum representing the current status of an agent.

    Values:
        STOPPED: The agent is not currently generating content.
        GENERATING: The agent is actively generating content.
        INTERRUPT_REQUESTED: The agent has received a request to stop generation but hasn't fully stopped yet.
    """

    STOPPED = "stopped"
    GENERATING = "generating"
    INTERRUPT_REQUESTED = "interrupt_requested"
