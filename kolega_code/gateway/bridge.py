"""Stream bridge: mirror an agent turn into chat messages.

Consumes the two parallel output surfaces of a turn —

- ``process_message_stream`` chunks (``{"type": "response"|"thinking",
  "content", "complete"}``, incremental segments), and
- ``chat_message`` tool events from the session's connection-manager queue —

and renders them through the adapter the way Hermes/OpenClaw render Telegram
turns:

- a placeholder reply is created on first content and edited in place on a
  throttle until the turn ends (final-only transports get one message at the
  end);
- each round of tool activity lands in its own "working…" message below the
  newest reply bubble: a round that starts after text was streamed opens a
  fresh message (Telegram cannot move messages, so appending to an older one
  would surface above the latest text), and the messages persist as a trail
  of what the agent did during the turn;
- text is chunked at the adapter's limit; the first chunk is the edited
  message and overflow chunks are sent as continuations on finalize.

Chunk content is *incremental*, so the renderer appends and re-renders the
accumulated text — it never replaces with the last segment.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, AsyncIterator, Optional

from kolega_code.events import AgentEvent
from kolega_code.gateway.adapters.base import GatewayAdapter
from kolega_code.gateway.adapters.telegram.formatting import chunk_text

logger = logging.getLogger(__name__)

#: Cap on a round message's lines so a chatty turn cannot spam an unbounded edit.
MAX_STATUS_LINES = 8
#: Rendered status line: (icon, tool description, tool call id).
StatusLine = tuple[str, str, str]


class TurnRenderer:
    """Renders one turn of stream chunks and tool events into a chat."""

    def __init__(
        self,
        adapter: GatewayAdapter,
        chat_id: str,
        *,
        event_queue: asyncio.Queue[AgentEvent],
        chunk_limit: Optional[int] = None,
        edit_throttle_seconds: float = 1.0,
        monotonic: Any = time.monotonic,
    ) -> None:
        self._adapter = adapter
        self._chat_id = chat_id
        self._events = event_queue
        self._chunk_limit = chunk_limit or adapter.capabilities.text_chunk_limit
        self._edit_throttle = edit_throttle_seconds
        self._monotonic = monotonic
        # Per-segment state, mirroring the TUI's stream fold: a segment is one
        # contiguous assistant response identified by its chunk uuid (the
        # agent rotates it at tool rounds and thinking transitions), and each
        # segment becomes one chat bubble.
        self._segment_uuid: Optional[str] = None
        self._segment_text = ""
        self._total_text = ""
        self._reply_id: Optional[str] = None
        self._edited_text = ""
        self._last_flush = 0.0
        #: The current tool round's message, one per burst of tool activity.
        #: Older rounds' messages stay in the chat as a trail of the turn.
        self._status_id: Optional[str] = None
        self._status_lines: list[StatusLine] = []
        self._status_sent_text = ""
        self._last_status_flush = 0.0
        #: Set when a reply message was sent after the current round message:
        #: the next tool event then opens a fresh round message below the
        #: newest bubble (Telegram cannot move messages) instead of appending
        #: to the stale one higher up the chat.
        self._status_stale = False

    async def run(self, chunks: AsyncIterator[dict[str, Any]]) -> str:
        """Consume the turn generator and mirror it into the chat.

        Returns the complete accumulated response text. Cancellation (e.g.
        ``/stop``) propagates after the cleanup below has run, so the partial
        reply stays visible.
        """
        pump: Optional[asyncio.Task[None]] = None
        if self._events is not None:
            pump = asyncio.create_task(self._event_pump(), name="gateway-turn-event-pump")
        try:
            await self._adapter.set_typing(self._chat_id, True)
            async for chunk in chunks:
                if chunk.get("type") != "response":
                    # Thinking chunks stay off the wire; they are reasoning, not output.
                    continue
                content = chunk.get("content") or ""
                chunk_uuid = str(chunk.get("uuid") or "")
                complete = bool(chunk.get("complete"))
                self._total_text += content
                if self._reply_id is not None and chunk_uuid and chunk_uuid != self._segment_uuid:
                    # The agent rotated the segment uuid: settle this bubble,
                    # the next content opens a fresh one (the TUI's rule).
                    await self._finalize_segment()
                if content:
                    if self._reply_id is None:
                        # An empty first chunk never opens a bubble (the agent
                        # flushes one after every tool round) — same drop rule
                        # as the TUI.
                        self._segment_uuid = chunk_uuid
                    self._segment_text += content
                    await self._flush_segment(final=complete)
                if complete and (self._reply_id is not None or self._segment_text):
                    await self._finalize_segment()
            await self._finalize_segment()
            return self._total_text
        finally:
            if pump is not None:
                pump.cancel()
                try:
                    await pump
                except asyncio.CancelledError:
                    pass
            await self._adapter.set_typing(self._chat_id, False)

    # -- Reply messages (one per response segment) -------------------------

    async def _finalize_segment(self) -> None:
        await self._flush_segment(final=True)
        # A new bubble for the next segment.
        self._reply_id = None
        self._edited_text = ""
        self._segment_text = ""
        self._segment_uuid = None

    async def _flush_segment(self, final: bool) -> None:
        text = self._segment_text
        if not text:
            return
        now = self._monotonic()
        if not final and now - self._last_flush < self._edit_throttle:
            return
        self._last_flush = now
        chunks = chunk_text(text, self._chunk_limit)
        if self._reply_id is None:
            if not final and not self._adapter.capabilities.supports_edits:
                return  # final-only transports send once, at the end
            self._reply_id = await self._adapter.send_text(self._chat_id, chunks[0])
            self._edited_text = chunks[0]
            self._status_stale = True
            return
        if not final:
            if len(chunks) > 1:
                # The first message is frozen at the chunk limit; the rest
                # land as continuations when the segment ends.
                return
            await self._edit_reply(chunks[0])
            return
        await self._edit_reply(chunks[0])
        for extra in chunks[1:]:
            await self._adapter.send_text(self._chat_id, extra)
            self._status_stale = True

    async def _edit_reply(self, text: str) -> None:
        if text == self._edited_text:
            return
        if self._adapter.capabilities.supports_edits:
            await self._adapter.edit_text(self._chat_id, self._reply_id or "", text)
        self._edited_text = text

    # -- Tool round messages ----------------------------------------------

    async def _event_pump(self) -> None:
        try:
            while True:
                event = await self._events.get()
                record = self._line_record_for(event)
                if record is None:
                    continue
                if self._status_stale:
                    # The newest reply bubble was sent after the current round
                    # message: open a fresh round message below it instead of
                    # appending to the stale one higher up the chat.
                    self._status_stale = False
                    self._status_id = None
                    self._status_lines = []
                self._record_line(record)
                await self._flush_status()
        except asyncio.CancelledError:
            raise

    @staticmethod
    def _line_record_for(event: AgentEvent) -> Optional[StatusLine]:
        if event.event_type != "chat_message":
            return None
        message_type = event.content.get("message_type")
        description = str(event.content.get("tool_description") or "tool")
        call_id = str(event.content.get("tool_call_id") or "")
        if message_type == "tool_call":
            return ("⏳", description, call_id)
        if message_type == "tool_result":
            return ("✅", description, call_id)
        if message_type == "tool_error":
            return ("❌", description, call_id)
        return None

    def _record_line(self, record: StatusLine) -> None:
        icon, _, call_id = record
        if icon != "⏳" and call_id:
            # A finished call replaces its pending line so the round message
            # reads as a settled trail (⏳ bash -> ✅ bash), not an append log.
            for index, (pending_icon, _, pending_id) in enumerate(self._status_lines):
                if pending_icon == "⏳" and pending_id == call_id:
                    self._status_lines[index] = record
                    return
        self._status_lines.append(record)
        self._status_lines = self._status_lines[-MAX_STATUS_LINES:]

    async def _flush_status(self) -> None:
        if not self._status_lines:
            return
        text = "\n".join(f"{icon} {description}" for icon, description, _ in self._status_lines)
        if text == self._status_sent_text:
            # Line-cap truncation can make a new event's rendered text
            # identical to the previous one; re-sending the same content is a
            # Telegram "message is not modified" error, not progress.
            return
        now = self._monotonic()
        if self._status_id is not None and now - self._last_status_flush < self._edit_throttle:
            return
        self._last_status_flush = now
        if self._status_id is None:
            # No round message on transports that cannot edit it away.
            if not self._adapter.capabilities.supports_edits:
                return
            self._status_id = await self._adapter.send_text(self._chat_id, text)
        else:
            await self._adapter.edit_text(self._chat_id, self._status_id, text)
        self._status_sent_text = text
