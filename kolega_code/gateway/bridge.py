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
- tool activity lands in a separate "working…" status message, edited as
  events arrive and deleted when the turn finishes;
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

#: Cap on status-message lines so a chatty turn cannot spam an unbounded edit.
MAX_STATUS_LINES = 8
#: Events the status pump renders; everything else on the stream is ignored.
STATUS_MESSAGE_TYPES = {"tool_call", "tool_result", "tool_error"}


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
        self._full_text = ""
        self._reply_id: Optional[str] = None
        self._edited_text = ""
        self._last_flush = 0.0
        self._status_id: Optional[str] = None
        self._status_lines: list[str] = []
        self._last_status_flush = 0.0

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
                if chunk.get("type") == "response":
                    self._full_text += chunk.get("content") or ""
                    await self._flush(final=bool(chunk.get("complete")))
                # Thinking chunks stay off the wire; they are reasoning, not output.
            await self._flush(final=True)
            return self._full_text
        finally:
            if pump is not None:
                pump.cancel()
                try:
                    await pump
                except asyncio.CancelledError:
                    pass
            await self._adapter.set_typing(self._chat_id, False)
            await self._finalize_status()

    # -- Reply message -----------------------------------------------------

    async def _flush(self, final: bool) -> None:
        text = self._full_text
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
            return
        if not final:
            if len(chunks) > 1:
                # The first message is frozen at the chunk limit; the rest
                # land as continuations when the turn ends.
                return
            await self._edit_reply(chunks[0])
            return
        await self._edit_reply(chunks[0])
        for extra in chunks[1:]:
            await self._adapter.send_text(self._chat_id, extra)

    async def _edit_reply(self, text: str) -> None:
        if text == self._edited_text:
            return
        if self._adapter.capabilities.supports_edits:
            await self._adapter.edit_text(self._chat_id, self._reply_id or "", text)
        self._edited_text = text

    # -- Tool status message ----------------------------------------------

    async def _event_pump(self) -> None:
        try:
            while True:
                event = await self._events.get()
                line = self._status_line_for(event)
                if line is None:
                    continue
                self._status_lines.append(line)
                self._status_lines = self._status_lines[-MAX_STATUS_LINES:]
                await self._flush_status()
        except asyncio.CancelledError:
            raise

    @staticmethod
    def _status_line_for(event: AgentEvent) -> Optional[str]:
        if event.event_type != "chat_message":
            return None
        message_type = event.content.get("message_type")
        if message_type == "tool_call":
            description = event.content.get("tool_description") or "tool"
            return f"⏳ {description}"
        if message_type == "tool_result":
            return "✅ tool finished"
        if message_type == "tool_error":
            return "❌ tool failed"
        return None

    async def _flush_status(self) -> None:
        if not self._status_lines:
            return
        now = self._monotonic()
        if self._status_id is not None and now - self._last_status_flush < self._edit_throttle:
            return
        self._last_status_flush = now
        text = "\n".join(self._status_lines)
        if self._status_id is None:
            # No status message on transports that cannot edit it away.
            if not self._adapter.capabilities.supports_edits:
                return
            self._status_id = await self._adapter.send_text(self._chat_id, text)
            return
        await self._adapter.edit_text(self._chat_id, self._status_id, text)

    async def _finalize_status(self) -> None:
        if self._status_id is None:
            return
        message_id, self._status_id = self._status_id, None
        if self._adapter.capabilities.supports_delete:
            try:
                await self._adapter.delete_message(self._chat_id, message_id)
            except Exception:  # noqa: BLE001 — a stuck status message is cosmetic
                logger.debug("gateway: status delete failed for chat %s", self._chat_id)
