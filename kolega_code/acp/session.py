"""One ACP session: the kolega-code session record plus its live agent.

The ACP server drives one ``CoderAgent`` per session; the session is the
durable ``SessionRecord`` so ACP-driven threads replay in the TUI, export,
and thread import like any other session.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from kolega_code.agent import CoderAgent
from kolega_code.cli.connection import CliConnectionManager
from kolega_code.cli.session_store import SessionRecord

if TYPE_CHECKING:
    from kolega_code.acp.accounting import AcpUsageAccounting
    from kolega_code.acp.control import AcpControlAdapter
    from kolega_code.mcp.config import MCPServerConfig
    from kolega_code.session.runtime import SessionRuntime
    from kolega_code.session.usage import UsagePresentation


@dataclass
class AcpSession:
    """Live state for one ACP session."""

    session_id: str
    record: SessionRecord
    agent: CoderAgent
    manager: CliConnectionManager
    #: Bound at build time so a model/mode-switch rebuild reattaches the same callback.
    permission_callback: Any = None
    #: Shared permission-policy owner and control channel for this session.
    runtime: SessionRuntime | None = None
    control_adapter: AcpControlAdapter | None = field(default=None, repr=False)
    #: The AgentConfig the agent was built with, reused by mode-switch rebuilds.
    config: Any = field(default=None, repr=False)
    client_mcp_servers: dict[str, MCPServerConfig] = field(default_factory=dict, repr=False)
    #: Shared with the ask_user_choice tool extension; the adapter binds the
    #: ``elicit`` callable here so questions survive agent rebuilds.
    question_state: dict[str, Any] = field(default_factory=dict)
    #: In-flight turn task; None when the session is idle. Set and cleared by the server.
    turn_task: asyncio.Task | None = field(default=None, repr=False)
    execution_lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)
    event_task: asyncio.Task[None] | None = field(default=None, repr=False)
    background_task: asyncio.Task[None] | None = field(default=None, repr=False)
    close_task: asyncio.Task[None] | None = field(default=None, repr=False)
    close_retryable: bool = field(default=False, repr=False)
    bridge: Any = field(default=None, repr=False)
    background_wake: asyncio.Event = field(default_factory=asyncio.Event, repr=False)
    completions: list[dict[str, Any]] = field(default_factory=list)
    auto_continue_paused: bool = False
    approved_plan: str | None = field(default=None, repr=False)
    plan_approval_generation: int = 0
    plan_approval_task: asyncio.Task[Any] | None = field(default=None, repr=False)
    closed: bool = False
    retired_agents: list[Any] = field(default_factory=list, repr=False)
    cleanup_tasks: dict[int, asyncio.Task[None]] = field(default_factory=dict, repr=False)
    accounting: AcpUsageAccounting | None = field(default=None, repr=False)
    usage_presentation: UsagePresentation | None = field(default=None, repr=False)

    def idle(self) -> bool:
        return (
            not self.closed
            and not self.cleanup_tasks
            and self.approved_plan is None
            and not self.execution_lock.locked()
            and (self.turn_task is None or self.turn_task.done())
        )
