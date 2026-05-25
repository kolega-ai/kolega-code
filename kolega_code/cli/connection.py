"""CLI event bridge for agent broadcasts."""

from __future__ import annotations

import asyncio
from collections import Counter
from typing import Any

from kolega_code.agent.connection_manager import AgentConnectionManager
from kolega_code.agent.models.public import AgentEvent


class CliConnectionManager(AgentConnectionManager):
    """Connection manager that exposes agent broadcasts through an asyncio queue."""

    def __init__(self) -> None:
        self.events: asyncio.Queue[AgentEvent] = asyncio.Queue()
        self._connections: Counter[tuple[str, str, str]] = Counter()

    async def connect(
        self,
        websocket: Any,
        workspace_id: str,
        thread_id: str,
        connection_type: str,
        user_info=None,
    ) -> None:
        self._connections[(workspace_id, thread_id, connection_type)] += 1

    def disconnect(self, websocket: Any, workspace_id: str, thread_id: str, connection_type: str) -> None:
        key = (workspace_id, thread_id, connection_type)
        if self._connections[key] > 1:
            self._connections[key] -= 1
        else:
            self._connections.pop(key, None)

    async def broadcast_event(self, event: AgentEvent, workspace_id: str, thread_id: str) -> None:
        await self.events.put(event)

    def get_connection_count(self, workspace_id: str, thread_id: str) -> dict:
        counts: dict[str, int] = {}
        for (ws_id, th_id, connection_type), count in self._connections.items():
            if ws_id == workspace_id and th_id == thread_id:
                counts[connection_type] = count
        return counts

    async def next_event(self) -> AgentEvent:
        """Return the next broadcast event."""
        return await self.events.get()
