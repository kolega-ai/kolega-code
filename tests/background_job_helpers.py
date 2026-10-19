"""Event-gated doubles for testing host ownership, not supervisor internals."""

from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator, Callable

from kolega_code.permissions import PermissionMode


class Jobs:
    def __init__(self) -> None:
        self.on_ready: Callable[[], None] | None = None
        self.on_delivered: Callable[[dict[str, Any]], None] | None = None
        self.tasks: set[asyncio.Task[None]] = set()
        self.pending: list[dict[str, Any]] = []
        self.outcomes: list[dict[str, Any]] = []
        self.changed = asyncio.Event()
        self.waiting = asyncio.Event()
        self.closed = False
        self.delivered = asyncio.Event()

    def set_callbacks(self, *, on_ready: Any, on_delivered: Any) -> None:
        self.on_ready = on_ready
        self.on_delivered = on_delivered

    async def recover(self) -> None:
        if self.pending and self.on_ready is not None:
            self.on_ready()

    @property
    def active_count(self) -> int:
        return sum(not task.done() for task in self.tasks)

    @property
    def pending_count(self) -> int:
        return len(self.pending)

    @property
    def has_work(self) -> bool:
        return bool(self.active_count or self.pending_count)

    def launch(self, gate: asyncio.Event) -> None:
        assert not self.closed

        async def child() -> None:
            status = "completed"
            try:
                await gate.wait()
            except asyncio.CancelledError:
                status = "cancelled"
            completion = {
                "job_id": f"job-{len(self.outcomes)}",
                "agent_type": "investigation",
                "status": status,
                "result": "child report",
                "text": f"Background investigation {status}: child report",
            }
            self.outcomes.append(completion)
            self.pending.append(completion)
            self.changed.set()
            if self.on_ready is not None:
                self.on_ready()

        self.tasks.add(asyncio.create_task(child()))

    async def deliver(self, agent: Any) -> int:
        assert not agent.active_turns, "host edited history during an active parent turn"
        count = len(self.pending)
        while self.pending:
            completion = self.pending.pop(0)
            agent.history.append(completion)
            agent.background_needs_turn = True
            if self.on_delivered is not None:
                self.on_delivered(completion)
        if count:
            self.delivered.set()
        return count

    async def wait_for_completion(self) -> None:
        self.waiting.set()
        while not self.pending and self.active_count:
            self.changed.clear()
            await self.changed.wait()

    async def wait(self) -> None:
        await asyncio.shield(asyncio.gather(*self.tasks))

    async def close(self) -> None:
        self.closed = True
        # Let newly admitted children enter their cancellation-safe body.
        await asyncio.sleep(0)
        for task in self.tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*self.tasks)


class HostAgent:
    """Only the agent surface used by the headless/protocol execution owners."""

    def __init__(self, jobs: Jobs | None = None) -> None:
        self.background_jobs = jobs or Jobs()
        self.background_needs_turn = False
        self.history: list[dict[str, Any]] = []
        self.permission_mode = PermissionMode.AUTO
        self.active_turns = 0
        self.max_active_turns = 0
        self.continuations = 0
        self.messages: list[str] = []
        self.continued = asyncio.Event()
        self.turn_started = asyncio.Event()
        self.turn_gate: asyncio.Event | None = None
        self.cleaned = False
        self.session_recorder: Any = None

    def get_background_jobs(self) -> Jobs:
        if self.background_jobs is None:
            self.background_jobs = Jobs()
        return self.background_jobs

    async def _stream(self, text: str) -> AsyncIterator[dict[str, Any]]:
        self.active_turns += 1
        self.max_active_turns = max(self.max_active_turns, self.active_turns)
        assert self.active_turns == 1, "overlapping parent turns"
        self.turn_started.set()
        try:
            if self.turn_gate is not None:
                await self.turn_gate.wait()
            yield {"type": "response", "content": text, "complete": True}
        finally:
            self.active_turns -= 1

    async def process_message_stream(self, text: str, attachments: Any = None) -> AsyncIterator[dict[str, Any]]:
        self.messages.append(text)
        async for chunk in self._stream("parent response"):
            yield chunk

    async def continue_from_history_stream(self) -> AsyncIterator[dict[str, Any]]:
        self.background_needs_turn = False
        self.continuations += 1
        async for chunk in self._stream("continued with child report"):
            yield chunk
        self.continued.set()

    def dump_message_history(self) -> list[dict[str, Any]]:
        return list(self.history)

    def dump_compaction_state(self) -> dict[str, Any]:
        return {}

    def set_permission_mode(self, mode: PermissionMode) -> None:
        self.permission_mode = mode

    async def cleanup(self) -> None:
        if self.background_jobs is not None:
            await self.background_jobs.close()
        self.cleaned = True
