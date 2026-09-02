"""Messaging gateway: drive Kolega Code agents from chat platforms.

The gateway is a long-running daemon that owns connections to messaging
platforms (Telegram first; more later), routes each chat to a dedicated agent
session, streams turns back into the chat, and relays mid-turn prompts
(permission approvals, ``ask_user_choice`` questions) to the user's phone.

Layout:

- :mod:`kolega_code.gateway.adapters` — the platform-agnostic adapter contract
  (``GatewayAdapter``) and per-platform implementations.
- :mod:`kolega_code.gateway.config` — settings.json-driven configuration
  (the ``gateway`` section plus the saved Telegram token).
- :mod:`kolega_code.gateway.daemon` — process lifecycle, lock file, inbound
  dispatch, access control.
- :mod:`kolega_code.gateway.router` — per-chat session registry (LRU + idle TTL).
- :mod:`kolega_code.gateway.bridge` — stream/tool-event rendering into chat.
- :mod:`kolega_code.gateway.sessions` — the real agent-session host
  (``SessionRuntime`` + the ACP-style agent-construction recipe).
- :mod:`kolega_code.gateway.commands` — slash commands (``/new``, ``/status``, …).
- :mod:`kolega_code.gateway.control_relay` — permission/question prompts as
  inline buttons (later phase).
- :mod:`kolega_code.gateway.redaction` — keep credentials out of chat text.
"""

from kolega_code.gateway.adapters.base import (
    AdapterCapabilities,
    Attachment,
    ButtonOption,
    ChatRef,
    GatewayAdapter,
    InboundMessage,
    ReplyContext,
    UnsupportedCapability,
)
from kolega_code.gateway.config import GatewayConfig, GatewayConfigError, load_gateway_config

__all__ = [
    "AdapterCapabilities",
    "Attachment",
    "ButtonOption",
    "ChatRef",
    "GatewayAdapter",
    "GatewayConfig",
    "GatewayConfigError",
    "InboundMessage",
    "ReplyContext",
    "UnsupportedCapability",
    "load_gateway_config",
]
