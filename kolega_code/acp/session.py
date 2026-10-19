"""One ACP session: the kolega-code session record plus its live agent.

The ACP server drives one ``CoderAgent`` per session; the session is the
durable ``SessionRecord`` so ACP-driven threads replay in the TUI, export,
and thread import like any other session.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from kolega_code.agent import CoderAgent
from kolega_code.cli.connection import CliConnectionManager
from kolega_code.cli.session_store import SessionRecord


@dataclass
class AcpSession:
    """Live state for one ACP session."""

    session_id: str
    record: SessionRecord
    agent: CoderAgent
    manager: CliConnectionManager
    #: Bound at build time so a model/mode-switch rebuild reattaches the same callback.
    permission_callback: Any = None
    #: The AgentConfig the agent was built with, reused by mode-switch rebuilds.
    config: Any = None
    #: Shared with the ask_user_choice tool extension; the server binds the
    #: ``elicit`` callable here so questions survive agent rebuilds.
    question_state: dict[str, Any] = field(default_factory=dict)
    #: In-flight turn task; None when the session is idle. Set and cleared by the server.
    turn_task: asyncio.Task | None = field(default=None, repr=False)
    execution_lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)
    event_task: asyncio.Task[None] | None = field(default=None, repr=False)
    background_task: asyncio.Task[None] | None = field(default=None, repr=False)
    bridge: Any = field(default=None, repr=False)
    background_wake: asyncio.Event = field(default_factory=asyncio.Event, repr=False)
    completions: list[dict[str, Any]] = field(default_factory=list)
    auto_continue_paused: bool = False
    closed: bool = False
    retired_agents: list[Any] = field(default_factory=list, repr=False)

    def idle(self) -> bool:
        return not self.execution_lock.locked() and (self.turn_task is None or self.turn_task.done())
