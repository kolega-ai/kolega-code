"""Sender access control: allowlist plus pairing-code onboarding.

The daemon's allowlist comes from the stored gateway settings (``settings.json`` → ``gateway.allowed_users``),
but a running daemon must also learn newly approved senders without a
restart, and approvals happen from a separate process (``kolega-code gateway
pairing approve <code>``). Both sides therefore persist to small files under
the state dir, re-read on every check:

- ``gateway_allowlist.json`` — sender id -> approval record.
- ``gateway_pairing.json`` — pairing code -> pending request.

When pairing is enabled, an unknown sender gets a short one-hour code to hand
to the operator; approving the code moves the sender into the persisted
allowlist, and their next message goes through.
"""

from __future__ import annotations

import json
import logging
import os
import secrets
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from kolega_code.gateway.adapters.base import InboundMessage

logger = logging.getLogger(__name__)

ALLOWLIST_FILE_NAME = "gateway_allowlist.json"
PAIRING_FILE_NAME = "gateway_pairing.json"
#: Unambiguous alphabet for codes the operator types by hand.
_CODE_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
CODE_LENGTH = 6
MAX_PENDING_CODES = 50


class AccessControlError(RuntimeError):
    """Raised when a pairing approval cannot be completed."""


@dataclass(frozen=True)
class PairingRequest:
    """One pending onboarding request."""

    code: str
    sender_id: str
    sender_name: str
    channel: str
    chat_id: str
    created_at: float
    expires_at: float


def _read_json(path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("gateway access: unreadable %s (%s); treating as empty", path, exc)
        return {}
    return raw if isinstance(raw, dict) else {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


class GatewayAccessControl:
    """Allowlist + pairing state shared between the daemon and the CLI."""

    def __init__(
        self,
        *,
        state_dir: Path,
        allowed_users: tuple[str, ...] = (),
        pairing_enabled: bool = False,
        code_ttl_seconds: float = 3600.0,
        now: Any = time.time,
    ) -> None:
        self._state_dir = state_dir
        self._configured_users = set(allowed_users)
        self._pairing_enabled = pairing_enabled
        self._code_ttl_seconds = code_ttl_seconds
        self._now = now
        self._allowlist_path = state_dir / ALLOWLIST_FILE_NAME
        self._pairing_path = state_dir / PAIRING_FILE_NAME

    # -- Daemon side -------------------------------------------------------

    def is_allowed(self, sender_id: str) -> bool:
        if not self._configured_users:
            # No allowlist configured: anyone may talk to the gateway.
            return True
        if sender_id in self._configured_users:
            return True
        return sender_id in _read_json(self._allowlist_path)

    def on_unknown_sender(self, message: InboundMessage) -> Optional[str]:
        """Return the pairing reply to send, or None to drop silently.

        Pairing only applies while an allowlist is configured; without one the
        gateway is open and no sender is ever "unknown".
        """
        if not self._configured_users or not self._pairing_enabled:
            return None
        pending = self._pending()
        existing = next((entry for entry in pending.values() if entry["sender_id"] == message.sender_id), None)
        if existing is not None and self._now() < float(existing["expires_at"]):
            code = str(existing["code"])
        else:
            code = self._issue_code(pending)
            pending[code] = {
                "code": code,
                "sender_id": message.sender_id,
                "sender_name": message.sender_name,
                "channel": message.channel,
                "chat_id": message.chat_id,
                "created_at": self._now(),
                "expires_at": self._now() + self._code_ttl_seconds,
            }
            self._write_pairings(pending)
        return f"🔑 I don't know you yet. Ask the gateway's owner to run:\n\nkolega-code gateway pairing approve {code}"

    def pending(self) -> list[PairingRequest]:
        """Pending requests, newest first (for ``gateway pairing list``)."""
        now = self._now()
        requests = [
            PairingRequest(
                code=str(entry["code"]),
                sender_id=str(entry["sender_id"]),
                sender_name=str(entry.get("sender_name") or ""),
                channel=str(entry.get("channel") or ""),
                chat_id=str(entry.get("chat_id") or ""),
                created_at=float(entry["created_at"]),
                expires_at=float(entry["expires_at"]),
            )
            for entry in self._pending().values()
            if now < float(entry["expires_at"])
        ]
        return sorted(requests, key=lambda request: request.created_at, reverse=True)

    # -- Operator side -----------------------------------------------------

    def approve(self, code: str) -> str:
        """Approve a pending code, returning the sender id that was admitted.

        Raises ``AccessControlError`` for unknown or expired codes.
        """
        normalized = code.strip().upper()
        pending = self._pending()
        entry = pending.get(normalized)
        if entry is None or self._now() >= float(entry["expires_at"]):
            pending.pop(normalized, None)
            self._write_pairings(pending)
            raise AccessControlError(f"Unknown or expired pairing code: {code}")
        sender_id = str(entry["sender_id"])
        allowlist = _read_json(self._allowlist_path)
        allowlist[sender_id] = {
            "name": entry.get("sender_name") or "",
            "approved_at": self._now(),
        }
        pending.pop(normalized, None)
        _write_json(self._allowlist_path, allowlist)
        self._write_pairings(pending)
        return sender_id

    # -- Internals ---------------------------------------------------------

    def _pending(self) -> dict[str, dict[str, Any]]:
        return {str(code): dict(entry) for code, entry in _read_json(self._pairing_path).items()}

    def _write_pairings(self, pending: dict[str, dict[str, Any]]) -> None:
        # Prune expired codes and cap the set so an abandoned gateway cannot
        # accumulate state forever.
        now = self._now()
        live = {code: entry for code, entry in pending.items() if now < float(entry["expires_at"])}
        while len(live) > MAX_PENDING_CODES:
            oldest = min(live, key=lambda code: float(live[code]["created_at"]))
            live.pop(oldest)
        _write_json(self._pairing_path, live)

    def _issue_code(self, pending: dict[str, dict[str, Any]]) -> str:
        for _ in range(100):
            code = "".join(secrets.choice(_CODE_ALPHABET) for _ in range(CODE_LENGTH))
            if code not in pending:
                return code
        raise AccessControlError("Could not generate a unique pairing code")
