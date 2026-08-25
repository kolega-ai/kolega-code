"""Control channel: client-to-agent round trips over the event stream.

Everything the agent tells a frontend already travels one way, as events. Some
interactions need an answer back — "may I run this command?", "which option do
you want?" — and those were previously a direct in-process callback the terminal
UI supplied. That made the terminal UI the only frontend that could ever answer,
because a callback cannot cross a process boundary.

Here a request is an event like any other, and a response is an ordinary call:

1. The agent calls :meth:`ControlChannel.request` and awaits.
2. The channel emits ``control_requested`` onto the session's event stream, so it
   reaches every attached client and is recorded in the session's history.
3. Whichever client holds the control lease renders a prompt.
4. That client calls :meth:`respond`, which resolves the agent's wait.
5. The channel emits ``control_resolved`` so other clients stop showing a prompt
   that has already been answered.

There is deliberately no privileged path for a co-located UI. A frontend that
answered prompts through a side channel would leave the real path unexercised,
and the first remote client would be the one to discover it did not work.

**The lease.** Exactly one client may answer at a time. Sharing a session is
read-only, so viewers receive the request events — they can see that a prompt is
pending — but a response from anyone but the lease holder is refused. With no
holder at all, a request resolves immediately to its caller-supplied default
rather than hanging: an agent waiting forever on a prompt nobody can see is worse
than a denial.
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional

from kolega_code.events import AgentEvent, KnownEventType

#: How long to wait for the lease holder before falling back to the default.
#: Generous, because the answer is a human decision, but not unbounded: a client
#: that closed its window without answering must not strand the turn.
DEFAULT_REQUEST_TIMEOUT_SECONDS = 900.0

#: Emitter signature. Supplied by the runtime so requests travel the same
#: transport as every other event and land in the session recording.
EventEmitter = Callable[[AgentEvent], Awaitable[None]]


class ControlLeaseError(RuntimeError):
    """Raised when a client tries to take a lease another client holds."""


@dataclass
class ControlRequest:
    """One outstanding question awaiting an answer."""

    request_id: str
    #: "permission", "question", or anything a future prompt needs.
    kind: str
    payload: dict[str, Any]
    #: Returned if nobody answers, or if no client can.
    default: dict[str, Any]
    future: "asyncio.Future[dict[str, Any]]" = field(repr=False, compare=False, default=None)  # type: ignore[assignment]

    def describe(self) -> dict[str, Any]:
        """Wire form of the request, without the local future."""
        return {"request_id": self.request_id, "kind": self.kind, "payload": self.payload}


class ControlChannel:
    """Request/response transport for interactions that need a human answer."""

    def __init__(
        self,
        *,
        session_id: str,
        emit: EventEmitter,
        sender: str = "system",
        timeout: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
    ) -> None:
        self.session_id = session_id
        self.timeout = timeout
        self._emit = emit
        self._sender = sender
        self._pending: dict[str, ControlRequest] = {}
        self._lease: Optional[str] = None

    # -- Lease -------------------------------------------------------------

    @property
    def controller(self) -> Optional[str]:
        """Id of the client currently allowed to answer, if any."""
        return self._lease

    def acquire(self, client_id: str) -> None:
        """Claim the right to answer control requests."""
        if self._lease is not None and self._lease != client_id:
            raise ControlLeaseError(
                f"Control is held by {self._lease}; {client_id} may observe but not answer",
            )
        self._lease = client_id

    def release(self, client_id: str) -> None:
        """Give up the lease. Outstanding requests fall back to their defaults."""
        if self._lease != client_id:
            return
        self._lease = None
        # Nobody can answer now, so do not leave the agent waiting for a prompt
        # that no longer has anyone in front of it.
        for request in list(self._pending.values()):
            self._settle(request, request.default, reason="controller_left")

    # -- Agent side --------------------------------------------------------

    async def request(
        self,
        kind: str,
        payload: dict[str, Any],
        *,
        default: dict[str, Any],
        timeout: Optional[float] = None,
    ) -> dict[str, Any]:
        """Ask the controlling client a question and wait for its answer.

        Returns ``default`` if there is no controller, if the wait times out, or
        if the lease is released before an answer arrives. Never raises for those
        cases: a prompt that cannot be answered has to resolve one way or another,
        and the caller decides which way by choosing the default.

        ``timeout`` overrides the channel default for this one request — a
        short-lived approval (a held peer message, say) must not inherit the
        generous turn-prompt budget.
        """
        effective_timeout = self.timeout if timeout is None else timeout
        request = ControlRequest(
            request_id=str(uuid.uuid4()),
            kind=kind,
            payload=payload,
            default=default,
            future=asyncio.get_running_loop().create_future(),
        )
        self._pending[request.request_id] = request

        # Emitted even when nobody can answer, so the recording shows that the
        # agent reached a decision point and how it was resolved.
        await self._announce(KnownEventType.CONTROL_REQUESTED, request, extra={"has_controller": self.has_controller})

        if not self.has_controller:
            self._settle(request, default, reason="no_controller")
            return await request.future

        try:
            return await asyncio.wait_for(asyncio.shield(request.future), timeout=effective_timeout)
        except asyncio.TimeoutError:
            self._settle(request, default, reason="timeout")
            return default
        except asyncio.CancelledError:
            # The turn was cancelled; stop tracking so a late answer cannot
            # resolve a request nobody is waiting on.
            self._pending.pop(request.request_id, None)
            raise

    @property
    def has_controller(self) -> bool:
        return self._lease is not None

    # -- Client side -------------------------------------------------------

    def pending(self) -> list[ControlRequest]:
        """Outstanding requests, so a client attaching mid-prompt can catch up."""
        return list(self._pending.values())

    def respond(
        self,
        request_id: str,
        response: dict[str, Any],
        *,
        client_id: Optional[str] = None,
    ) -> bool:
        """Answer a request. Returns False if it is unknown or already settled.

        A response from a client that does not hold the lease is refused, which is
        what keeps a shared session read-only for viewers.
        """
        if client_id is not None and self._lease is not None and client_id != self._lease:
            raise ControlLeaseError(f"{client_id} does not hold control; {self._lease} does")
        request = self._pending.get(request_id)
        if request is None:
            return False
        self._settle(request, response, reason="answered")
        return True

    # -- Internals ---------------------------------------------------------

    def _settle(self, request: ControlRequest, response: dict[str, Any], *, reason: str) -> None:
        self._pending.pop(request.request_id, None)
        if request.future is not None and not request.future.done():
            request.future.set_result(response)
        # Fire-and-forget: settling must work from synchronous UI callbacks, and a
        # failure to announce must not strip an answer the agent is waiting for.
        try:
            asyncio.get_running_loop().create_task(
                self._announce(
                    KnownEventType.CONTROL_RESOLVED,
                    request,
                    extra={"response": response, "reason": reason},
                )
            )
        except RuntimeError:
            pass

    async def _announce(
        self,
        event_type: str,
        request: ControlRequest,
        *,
        extra: Optional[dict[str, Any]] = None,
    ) -> None:
        event = AgentEvent(
            sender=self._sender,
            event_type=event_type,
            session_id=self.session_id,
            content={**request.describe(), **(extra or {})},
        )
        try:
            await self._emit(event)
        except Exception:
            # Announcing is observability. The round trip itself must still work
            # if the transport is unavailable.
            pass
