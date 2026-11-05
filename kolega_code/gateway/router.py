"""Per-chat session registry: LRU ordering with an idle-TTL reaper.

The gateway keeps one agent session per chat. Sessions are expensive (an LLM
client, tool collection, journal handles), so a long-lived daemon must not
hold every chat it has ever seen: the registry caps the live set and evicts
idle entries through an ``on_evict`` callback the session host supplies
(persist + cleanup).
"""

from __future__ import annotations

import asyncio
import inspect
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional

from kolega_code.gateway.adapters.base import ChatRef

#: Called once per evicted entry so the host can persist and clean up.
#: May be sync or async.
EvictCallback = Callable[[Any], Optional[Awaitable[None]]]


@dataclass
class SessionEntry:
    """One live chat session held by the registry."""

    chat_ref: ChatRef
    #: Host-owned state (the agent runtime and its bookkeeping).
    payload: Any
    last_active: float


class SessionRegistry:
    """Bounded, LRU-ordered registry of live per-chat sessions."""

    def __init__(
        self,
        *,
        max_sessions: int = 50,
        idle_ttl_seconds: Optional[float] = 3600.0,
        on_evict: Optional[EvictCallback] = None,
        is_pinned: Optional[Callable[[Any], bool]] = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._max_sessions = max_sessions
        self._idle_ttl_seconds = idle_ttl_seconds
        self._on_evict = on_evict
        self._is_pinned = is_pinned or (lambda payload: False)
        self._clock = clock
        self._entries: OrderedDict[str, SessionEntry] = OrderedDict()

    def get(self, chat_ref: ChatRef) -> Optional[SessionEntry]:
        """Return the live entry for a chat, marking it recently active."""
        entry = self._entries.get(chat_ref.key)
        if entry is not None:
            entry.last_active = self._clock()
            self._entries.move_to_end(chat_ref.key)
        return entry

    def put(self, chat_ref: ChatRef, payload: Any) -> SessionEntry:
        """Register (or replace) the live session for a chat."""
        entry = SessionEntry(chat_ref=chat_ref, payload=payload, last_active=self._clock())
        self._entries[chat_ref.key] = entry
        self._entries.move_to_end(chat_ref.key)
        return entry

    def remove(self, chat_ref: ChatRef) -> Optional[SessionEntry]:
        """Drop a chat without invoking the evict callback (e.g. ``/new``)."""
        return self._entries.pop(chat_ref.key, None)

    def active_count(self) -> int:
        return len(self._entries)

    async def prune(self) -> list[SessionEntry]:
        """Evict expired and over-capacity entries, least-recently-active first."""
        evicted: list[SessionEntry] = []
        now = self._clock()
        if self._idle_ttl_seconds and self._idle_ttl_seconds > 0:
            expired_keys = [
                key
                for key, entry in self._entries.items()
                if now - entry.last_active > self._idle_ttl_seconds and not self._is_pinned(entry.payload)
            ]
            for key in expired_keys:
                evicted.append(self._entries.pop(key))
        while len(self._entries) > self._max_sessions:
            candidate = next((key for key, entry in self._entries.items() if not self._is_pinned(entry.payload)), None)
            if candidate is None:
                break  # temporary over-capacity is safer than cancelling owned work
            evicted.append(self._entries.pop(candidate))
        await self._notify_evicted(evicted)
        return evicted

    async def clear(self) -> list[SessionEntry]:
        """Evict everything (shutdown path), calling the evict callback."""
        evicted = self.take_all()
        await self._notify_evicted(evicted)
        return evicted

    def take_all(self) -> list[SessionEntry]:
        evicted = list(self._entries.values())
        self._entries.clear()
        return evicted

    async def _notify_evicted(self, entries: list[SessionEntry]) -> None:
        if self._on_evict is None:
            return
        # Synchronous callbacks retain every removed entry before any waiter can cancel.
        pending = []
        failures: list[BaseException] = []
        for entry in entries:
            try:
                result = self._on_evict(entry.payload)
            except BaseException as exc:
                failures.append(exc)
            else:
                if inspect.isawaitable(result):
                    pending.append(result)
        outcomes = await asyncio.gather(*pending, return_exceptions=True)
        failures.extend(outcome for outcome in outcomes if isinstance(outcome, BaseException))
        if failures:
            raise failures[0]
