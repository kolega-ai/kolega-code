"""One ACP session: the kolega-code session record plus its live agent.

The ACP server drives one ``CoderAgent`` per session; the session is the
durable ``SessionRecord`` so ACP-driven threads replay in the TUI, export,
and thread import like any other session.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

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
    #: In-flight turn task; None when the session is idle. Set and cleared by the server.
    turn_task: asyncio.Task | None = field(default=None, repr=False)

    def idle(self) -> bool:
        return self.turn_task is None or self.turn_task.done()

    def drain_events(self) -> None:
        """Drop any events left over from a previous turn (stale mapping inputs)."""
        while not self.manager.events.empty():
            try:
                self.manager.events.get_nowait()
            except asyncio.QueueEmpty:
                break
