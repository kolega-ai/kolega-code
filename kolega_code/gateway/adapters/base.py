"""Gateway adapter contract: the platform-agnostic messaging boundary.

A messaging gateway moves conversations between chat platforms (Telegram today;
WhatsApp, Discord, … later) and agent sessions. Every platform speaks a
different protocol, so adapters exist to normalize them onto one small envelope
in each direction:

- **Inbound**: adapters publish :class:`InboundMessage` records on their
  ``inbound`` queue, which the gateway drains. Adapters own the platform
  connection, its auth (bot token, QR pairing, …), reconnect behaviour, and the
  work of translating platform payloads (reply quotes, attachments, button taps)
  into envelope fields.

- **Outbound**: the gateway calls ``send_text``/``edit_text``/… with the
  envelope's ``chat_id``. Capability flags tell the gateway what the platform
  can do (edit in place, inline buttons, typing indicators, chunk limits) so the
  router and stream bridge never need to know which platform they are talking
  to.

The contract deliberately does not assume in-process async libraries: a future
WhatsApp adapter will be a Node (Baileys) sidecar subprocess speaking local
HTTP, so adapters only need to be plain async objects with a queue and
send/edit primitives.

Button taps round-trip as ordinary inbound messages: ``send_buttons`` returns a
callback token, and a tap arrives as an :class:`InboundMessage` whose
``callback_token``/``callback_option`` fields name the tapped button. This keeps
the control relay transport-agnostic.
"""

from __future__ import annotations

import abc
import asyncio
from dataclasses import dataclass
from typing import Any, Optional, Sequence

STREAMING_FINAL_ONLY = "final_only"
STREAMING_EDIT_IN_PLACE = "edit_in_place"
STREAMING_DRAFT = "draft"


@dataclass(frozen=True)
class Attachment:
    """One media item attached to a message, already downloaded or fetchable."""

    kind: str
    source: str
    file_name: Optional[str] = None


@dataclass(frozen=True)
class ReplyContext:
    """What the sender quoted, so the model sees the surrounding conversation."""

    text: str = ""
    sender_id: str = ""


@dataclass(frozen=True)
class InboundMessage:
    """A normalized message arriving from any platform."""

    channel: str
    chat_id: str
    sender_id: str
    message_id: str
    text: str = ""
    topic_id: Optional[str] = None
    sender_name: str = ""
    attachments: tuple[Attachment, ...] = ()
    reply_to: Optional[ReplyContext] = None
    callback_token: Optional[str] = None
    callback_option: Optional[str] = None
    is_group: bool = False
    bot_mentioned: bool = False


@dataclass(frozen=True)
class ChatRef:
    """Address of one conversation, the gateway's routing key.

    ``key`` is stable across adapters and includes the channel so a Telegram
    chat and a Discord channel with the same numeric id never collide.
    """

    channel: str
    chat_id: str
    topic_id: Optional[str] = None

    @property
    def key(self) -> str:
        base = f"{self.channel}:{self.chat_id}"
        if self.topic_id:
            return f"{base}:{self.topic_id}"
        return base

    @classmethod
    def from_message(cls, message: InboundMessage) -> "ChatRef":
        return cls(
            channel=message.channel,
            chat_id=message.chat_id,
            topic_id=message.topic_id,
        )


@dataclass(frozen=True)
class AdapterCapabilities:
    """What one platform can do; the gateway reads these, never platform SDKs."""

    supports_edits: bool = False
    supports_delete: bool = False
    supports_inline_buttons: bool = False
    supports_typing: bool = False
    supports_groups: bool = False
    supports_voice_notes: bool = False
    text_chunk_limit: int = 4096
    max_media_mb: float = 5.0
    streaming_mode: str = STREAMING_FINAL_ONLY


@dataclass(frozen=True)
class ButtonOption:
    """One tappable choice in an inline button prompt."""

    option_id: str
    label: str


class UnsupportedCapability(RuntimeError):
    """The adapter's platform cannot perform the requested operation."""


class GatewayAdapter(abc.ABC):
    """Contract every chat-platform adapter implements.

    Subclasses set ``name`` and ``capabilities``, push inbound messages onto
    ``self.inbound``, and implement the outbound operations their platform
    supports (the rest inherit ``UnsupportedCapability`` defaults).
    """

    name: str = "base"
    capabilities: AdapterCapabilities = AdapterCapabilities()

    def __init__(self) -> None:
        self.inbound: asyncio.Queue[InboundMessage] = asyncio.Queue()

    async def start(self) -> None:
        """Connect to the platform and begin publishing inbound messages."""

    async def stop(self) -> None:
        """Disconnect and stop background tasks. Must be idempotent."""

    def health(self) -> dict[str, Any]:
        """Current connection state plus adapter-specific detail."""
        return {"state": "stopped"}

    async def send_text(
        self,
        chat_id: str,
        text: str,
        *,
        reply_to_message_id: Optional[str] = None,
    ) -> str:
        """Send a message; returns an adapter-assigned message id."""
        raise NotImplementedError

    async def edit_text(self, chat_id: str, message_id: str, text: str) -> None:
        raise UnsupportedCapability(f"{self.name} does not support editing messages")

    async def delete_message(self, chat_id: str, message_id: str) -> None:
        raise UnsupportedCapability(f"{self.name} does not support deleting messages")

    async def send_media(self, chat_id: str, items: Sequence[Attachment], *, caption: str = "") -> str:
        raise UnsupportedCapability(f"{self.name} does not support sending media")

    async def send_buttons(self, chat_id: str, prompt: str, options: Sequence[ButtonOption]) -> str:
        """Send a prompt with tappable options; returns a callback token.

        A tap later arrives as an inbound message carrying the token and the
        tapped ``option_id``. The returned token is the adapter's own opaque id.
        """
        raise UnsupportedCapability(f"{self.name} does not support inline buttons")

    async def set_typing(self, chat_id: str, active: bool) -> None:
        """Show/hide a typing indicator for the chat."""
