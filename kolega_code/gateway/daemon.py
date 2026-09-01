"""Gateway process: adapter lifecycle, inbound dispatch, access control.

One daemon owns one adapter (later, one per account). Its job between the
transport and the session host is deliberately small:

- single-instance lock file and pid file,
- inbound dedup (adapter retries must never double-run a turn),
- the sender allowlist and group policy (a chat app is a control plane;
  strangers get nothing),
- dispatch into the turn handler, which owns per-chat session state and
  ordering,
- a heartbeat status file so ``gateway status`` can report on the running
  daemon without an IPC channel.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from collections import deque
from dataclasses import asdict, dataclass
from typing import Any, Optional, Protocol

from filelock import BaseFileLock, FileLock
from filelock import Timeout as FileLockTimeout

from kolega_code.events import utc_now_iso
from kolega_code.gateway.access import GatewayAccessControl
from kolega_code.gateway.adapters.base import ChatRef, GatewayAdapter, InboundMessage
from kolega_code.gateway.config import GatewayConfig

logger = logging.getLogger(__name__)

LOCK_FILE_NAME = "gateway.lock"
PID_FILE_NAME = "gateway.pid"
STATUS_FILE_NAME = "gateway.status.json"
#: Heartbeat cadence for the on-disk status file.
STATUS_HEARTBEAT_SECONDS = 15.0
#: A status file older than this is treated as stale by `gateway status`.
STATUS_STALE_SECONDS = 90.0
#: Short, secret-free failure notice. The real exception goes to the log only.
ERROR_REPLY = "⚠️ The gateway hit an error handling that message (see the gateway log)."
#: Bound on remembered message ids for dedup; adapter retries are near-term.
DEDUP_WINDOW = 1024


class GatewayDaemonError(RuntimeError):
    """Raised when the daemon cannot start (lock conflict, adapter failure)."""


class TurnHandler(Protocol):
    """The seam the daemon dispatches routed messages into.

    Implementations own per-chat session state and ordering, and return
    ``active_sessions`` for ``gateway status``.
    """

    def status(self) -> dict[str, Any]: ...

    async def handle(self, chat_ref: ChatRef, message: InboundMessage) -> None: ...

    async def shutdown(self) -> None: ...


@dataclass(frozen=True)
class DaemonStatus:
    """Snapshot for ``gateway status``."""

    running: bool
    adapter: str
    adapter_state: dict[str, Any]
    active_sessions: int
    pid: Optional[int]
    started_at: Optional[str]
    recent_errors: int


class GatewayDaemon:
    """Runs one adapter and dispatches its inbound messages."""

    def __init__(self, config: GatewayConfig, adapter: GatewayAdapter, turn_handler: TurnHandler) -> None:
        self._config = config
        self._adapter = adapter
        self._turn_handler = turn_handler
        self._lock_path = config.state_dir / LOCK_FILE_NAME
        self._pid_path = config.state_dir / PID_FILE_NAME
        self._lock: Optional[BaseFileLock] = None
        self._dispatch_task: Optional[asyncio.Task[None]] = None
        self._heartbeat_task: Optional[asyncio.Task[None]] = None
        self._started_at: Optional[str] = None
        self._recent_errors = 0
        self._seen_message_ids: deque[str] = deque(maxlen=DEDUP_WINDOW)
        self._status_path = config.state_dir / STATUS_FILE_NAME
        self._access = GatewayAccessControl(
            state_dir=config.state_dir,
            allowed_users=config.allowed_users,
            pairing_enabled=config.pairing_enabled,
            code_ttl_seconds=config.pairing_code_ttl_seconds,
        )

    # -- Lifecycle ---------------------------------------------------------

    async def start(self) -> None:
        self._acquire_lock()
        self._config.state_dir.mkdir(parents=True, exist_ok=True)
        # The default project (~/kolega-code-workspace) may not exist yet.
        self._config.project_path.mkdir(parents=True, exist_ok=True)
        self._write_pid()
        await self._adapter.start()
        self._started_at = utc_now_iso()
        self._dispatch_task = asyncio.create_task(self._dispatch_loop(), name="gateway-dispatch")
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop(), name="gateway-heartbeat")
        self._write_status()

    async def stop(self) -> None:
        if self._dispatch_task is not None:
            self._dispatch_task.cancel()
            try:
                await self._dispatch_task
            except asyncio.CancelledError:
                pass
            self._dispatch_task = None
        if self._heartbeat_task is not None:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
            self._heartbeat_task = None
        await self._turn_handler.shutdown()
        await self._adapter.stop()
        self._remove_status()
        self._release_lock()
        self._remove_pid()

    def status(self) -> DaemonStatus:
        handler_status = self._turn_handler.status()
        return DaemonStatus(
            running=self._dispatch_task is not None,
            adapter=self._adapter.name,
            adapter_state=self._adapter.health(),
            active_sessions=int(handler_status.get("active_sessions", 0)),
            pid=os.getpid() if self._dispatch_task is not None else None,
            started_at=self._started_at,
            recent_errors=self._recent_errors,
        )

    # -- Status file -------------------------------------------------------

    def _status_payload(self) -> dict[str, Any]:
        return {**asdict(self.status()), "heartbeat_at": utc_now_iso()}

    def _write_status(self) -> None:
        try:
            temporary = self._status_path.with_name(f"{self._status_path.name}.tmp")
            temporary.write_text(json.dumps(self._status_payload()), encoding="utf-8")
            os.replace(temporary, self._status_path)
        except OSError as exc:
            logger.warning("gateway: could not write status file (%s)", exc)

    def _remove_status(self) -> None:
        try:
            self._status_path.unlink(missing_ok=True)
        except OSError:
            pass

    async def _heartbeat_loop(self) -> None:
        try:
            while True:
                await asyncio.sleep(STATUS_HEARTBEAT_SECONDS)
                self._write_status()
        except asyncio.CancelledError:
            raise

    # -- Dispatch ----------------------------------------------------------

    async def _dispatch_loop(self) -> None:
        while True:
            message = await self._adapter.inbound.get()
            await self._handle_message(message)

    async def _handle_message(self, message: InboundMessage) -> None:
        if self._is_duplicate(message):
            logger.info("gateway: dropping duplicate message %s", message.message_id)
            return
        chat_ref = ChatRef.from_message(message)
        if message.is_group and not self._group_allowed(message):
            logger.info(
                "gateway: dropping unaddressed group message in chat %s from %s",
                message.chat_id,
                message.sender_id,
            )
            return
        if not self._access.is_allowed(message.sender_id):
            logger.info(
                "gateway: dropping message from unauthorized sender %s on %s",
                message.sender_id,
                message.channel,
            )
            pairing_reply = self._access.on_unknown_sender(message)
            if pairing_reply is not None:
                try:
                    await self._adapter.send_text(message.chat_id, pairing_reply)
                except Exception:  # noqa: BLE001 — the drop is already logged
                    logger.debug("gateway: pairing reply failed", exc_info=True)
            return
        try:
            await self._turn_handler.handle(chat_ref, message)
        except asyncio.CancelledError:
            raise
        except Exception:
            self._recent_errors += 1
            logger.exception("gateway: turn failed for chat %s", chat_ref.key)
            try:
                await self._adapter.send_text(chat_ref.chat_id, ERROR_REPLY)
            except Exception:
                logger.exception("gateway: could not deliver error reply for chat %s", chat_ref.key)

    def _is_duplicate(self, message: InboundMessage) -> bool:
        if not message.message_id:
            return False
        if message.message_id in self._seen_message_ids:
            return True
        self._seen_message_ids.append(message.message_id)
        return False

    def _group_allowed(self, message: InboundMessage) -> bool:
        """Group policy: the bot must be addressed, and only listed groups pass."""
        if self._config.group_ids and message.chat_id not in self._config.group_ids:
            return False
        return message.bot_mentioned

    # -- Lock and pid ------------------------------------------------------

    def _acquire_lock(self) -> None:
        lock = FileLock(str(self._lock_path))
        try:
            lock.acquire(timeout=0)
        except FileLockTimeout as exc:
            raise GatewayDaemonError(
                f"Another gateway instance is running (lock held: {self._lock_path}). "
                "Stop it first, or run `kolega-code gateway status`."
            ) from exc
        self._lock = lock

    def _release_lock(self) -> None:
        if self._lock is not None:
            try:
                self._lock.release()
            except Exception:  # noqa: BLE001 — release is best-effort on shutdown
                logger.exception("gateway: lock release failed")
            self._lock = None

    def _write_pid(self) -> None:
        self._pid_path.write_text(f"{os.getpid()}\n", encoding="utf-8")

    def _remove_pid(self) -> None:
        try:
            self._pid_path.unlink(missing_ok=True)
        except OSError:
            pass
