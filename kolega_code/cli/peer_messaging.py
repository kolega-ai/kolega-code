"""Headless peer inbox for long-lived `ask --goal` / `--loop` workers.

A worker has no interactive surface, so the inbound pipeline is narrower than
the TUI's: messages are either accepted into an in-memory queue or dropped with
a notice — a hold can never be answered here, and parking one would recreate
exactly the silent-black-hole failure mode cross-session messaging must avoid.
Accepted messages reach the agent two ways: mid-turn at tool boundaries via
BaseAgent's queued-input provider, and between turns as their own turn (see
``pending_texts``, drained by the run loop).

The socket lifecycle is deliberately forgiving: any bind failure degrades to
in-process silence with a stderr notice instead of failing the run.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Optional

from kolega_code.events import AgentEvent
from kolega_code.session.inbox import (
    MAX_QUEUED_PEER_MESSAGES,
    InboundNotice,
    InboundNoticeKind,
    PeerMessage,
    PeerRepeatGuard,
    PeerSocketServer,
    bind_session_socket,
    deliver_inbound,
    peer_model_text,
    peer_origin,
)


class HeadlessPeerInbox:
    """Binds one worker session's socket and queues accepted peer messages."""

    def __init__(
        self,
        *,
        store_root: Path,
        session_id: str,
        settings: Any,
        permission_mode_value: str,
        json_mode: bool,
        journal_factory: Any,
    ) -> None:
        self.store_root = store_root
        self.session_id = session_id
        self.settings = settings
        self.permission_mode_value = permission_mode_value
        self.json_mode = json_mode
        #: Zero-arg callable returning the session's FileSessionEventStore.
        self.journal_factory = journal_factory
        self._server: Optional[PeerSocketServer] = None
        self._queue: list[PeerMessage] = []
        self._turn_active = False
        self._repeat_guard = PeerRepeatGuard()

    # -- Lifecycle ---------------------------------------------------------

    async def start(self, agent: Any) -> None:
        """Bind the session's socket; any bind failure degrades to in-process silence."""
        server = await bind_session_socket(
            store_root=self.store_root,
            session_id=self.session_id,
            describe_status=lambda: "busy" if self._queue or self._active else "idle",
            deliver_message=self._deliver,
            on_unavailable=lambda reason: self._notice(f"peers: socket unavailable ({reason})"),
        )
        if server is None:
            return
        self._server = server
        self._notice(f"peers: listening on {server.path}")
        agent.set_queued_input_provider(self._provide_queued_inputs)

    def _notice(self, text: str) -> None:
        if not self.json_mode:
            print(text, file=sys.stderr)

    async def stop(self) -> None:
        server, self._server = self._server, None
        if server is not None:
            try:
                await server.stop()
            except Exception:  # noqa: BLE001 — shutdown must always complete
                pass

    # -- Queue -------------------------------------------------------------

    @property
    def _active(self) -> bool:
        return self._turn_active

    def mark_busy(self) -> None:
        self._turn_active = True

    def mark_idle(self) -> None:
        self._turn_active = False

    def pending_texts(self) -> list[str]:
        """Model-facing texts of queued messages, oldest first (non-destructive)."""
        return [peer_model_text(message) for message in self._queue]

    def pop_all(self) -> list[str]:
        texts = self.pending_texts()
        self._queue.clear()
        return texts

    async def _provide_queued_inputs(self) -> list[Any]:
        from kolega_code.agent.baseagent import QueuedUserInput

        if not self._queue:
            return []
        inputs = [
            QueuedUserInput(text=peer_model_text(message), origin=peer_origin(message)) for message in self._queue
        ]
        self._queue.clear()
        return inputs

    # -- Inbound -----------------------------------------------------------

    async def _deliver(self, message: PeerMessage) -> str:
        """Inbound hook shared with the socket transport. No surface, no holds.

        All sequencing lives in :func:`deliver_inbound`; a hold degrades to an
        explicit drop here because nobody can answer one.
        """

        def notify(notice: InboundNotice) -> None:
            if notice.kind is InboundNoticeKind.REPEAT_DROPPED:
                return  # loop protection is silent toward the sender
            sender = notice.message.sender_title
            if notice.kind is InboundNoticeKind.POLICY_DROPPED:
                self._notice(f"peers: dropped inbound message from {sender} (policy {notice.policy})")
            else:
                self._notice(
                    f"peers: dropped inbound message from {sender} (queue cap {MAX_QUEUED_PEER_MESSAGES} full)"
                )

        return await deliver_inbound(
            message,
            policy=self.settings.get_cross_session_inbound(),
            recipient_mode=self.permission_mode_value,
            repeat_guard=self._repeat_guard,
            queued_count=lambda: len(self._queue),
            record_event=lambda event_type: self._record(event_type, message),
            enqueue=self._queue.append,
            notify=notify,
        )

    async def _record(self, event_type: str, message: PeerMessage) -> None:
        try:
            events = self.journal_factory()
            await events.append(
                AgentEvent(
                    sender="peer-inbox",
                    event_type=event_type,
                    session_id=self.session_id,
                    content=message.event_content(),
                )
            )
        except Exception as exc:  # noqa: BLE001 — observability never blocks delivery
            self._notice(f"peers: could not record {event_type}: {exc}")


async def start_headless_peer_inbox(
    *,
    store_root: Path,
    session_id: str,
    agent: Any,
    settings: Any,
    permission_mode_value: str,
    json_mode: bool,
    journal_factory: Any,
    enabled: bool,
) -> Optional[HeadlessPeerInbox]:
    """Create, bind, and attach a worker's peer inbox. None when not enabled.

    Callers pass ``enabled`` for their own worker-shape rules (only --goal and
    --loop workers on persisted sessions have a lifetime to receive into).
    Never raises for bind failures — the transport degrades to in-process
    silence with a stderr notice so the run itself always proceeds.
    """
    if not enabled:
        return None
    inbox = HeadlessPeerInbox(
        store_root=store_root,
        session_id=session_id,
        settings=settings,
        permission_mode_value=permission_mode_value,
        json_mode=json_mode,
        journal_factory=journal_factory,
    )
    await inbox.start(agent)
    return inbox


def store_title_lookup(store: Any) -> Any:
    """A discovery ``title_lookup`` backed by the shared session store."""
    titles_by_id = {record.session_id: record.title for record in store.list()}
    return titles_by_id.get


def format_peer_table(peers: Any) -> str:
    """The one true rendering of discovered peers, shared by the tool and /peers."""
    if not peers:
        return "No other live sessions to message."
    lines = ["Live peer sessions:"]
    for peer in peers:
        project = Path(peer.project_path).name if peer.project_path else ""
        location = f" — {project}" if project else ""
        remote = "" if peer.source == "process" else " (remote)"
        lines.append(f"- {peer.title} — {peer.status}{remote}{location} — id {peer.session_id[:8]}")
    return "\n".join(lines)


#: Display-name cap shared by /rename, --name, and the send/receive paths.
PEER_NAME_MAX_CHARS = 64
