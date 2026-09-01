"""Turn handlers: the gateway's seam between a chat message and a reply.

A turn handler receives a routed message and produces the reply (or its
streaming updates) through the adapter. The daemon never knows *how* a reply
is made — an echo handler proves the pipeline in Phase 0, and the real agent
session host (``kolega_code.gateway.sessions``) plugs into the same seam.
"""

from __future__ import annotations

from typing import Any

from kolega_code.gateway.adapters.base import ChatRef, GatewayAdapter, InboundMessage


class EchoTurnHandler:
    """Phase-0 handler: reply with the text that arrived.

    Exercises the full daemon → handler → adapter outbound path without an
    LLM, so the transport plumbing is verifiable in isolation.
    """

    def __init__(self, adapter: GatewayAdapter) -> None:
        self._adapter = adapter

    def status(self) -> dict[str, Any]:
        return {"active_sessions": 0}

    async def handle(self, chat_ref: ChatRef, message: InboundMessage) -> None:
        if message.text.strip():
            await self._adapter.send_text(chat_ref.chat_id, f"echo: {message.text}")

    async def shutdown(self) -> None:
        pass
