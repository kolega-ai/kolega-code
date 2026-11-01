"""Control channel: round trips, the lease, and the ways a prompt can go unanswered.

The failure modes matter more than the happy path here. An agent that waits
forever on a prompt nobody can see is a hang, and a viewer of a shared session
that can answer prompts is a security hole, so both are pinned down.
"""

from __future__ import annotations

import asyncio

import pytest

from kolega_code.events import AgentEvent, KnownEventType
from kolega_code.session.control import ControlChannel, ControlLeaseError

DENY = {"allowed": False, "reason": "default"}
ALLOW = {"allowed": True}


def _channel(*, timeout: float = 5.0) -> tuple[ControlChannel, list[AgentEvent]]:
    emitted: list[AgentEvent] = []

    async def emit(event: AgentEvent) -> None:
        emitted.append(event)

    return ControlChannel(session_id="s1", emit=emit, timeout=timeout), emitted


async def _settle() -> None:
    """Let the fire-and-forget resolution announcement run."""
    await asyncio.sleep(0)
    await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_round_trip_delivers_the_controllers_answer() -> None:
    channel, emitted = _channel()
    channel.acquire("tui")

    async def answer() -> None:
        await asyncio.sleep(0.01)
        (request,) = channel.pending()
        assert channel.respond(request.request_id, ALLOW, client_id="tui") is True

    responder = asyncio.create_task(answer())
    result = await channel.request("permission", {"tool": "terminal"}, default=DENY)
    await responder
    await _settle()

    assert result == ALLOW
    kinds = [event.event_type for event in emitted]
    assert kinds == [KnownEventType.CONTROL_REQUESTED, KnownEventType.CONTROL_RESOLVED]
    assert emitted[0].content["payload"] == {"tool": "terminal"}
    assert emitted[1].content["reason"] == "answered"


@pytest.mark.asyncio
async def test_request_is_announced_so_viewers_can_see_it() -> None:
    """A prompt must appear on the stream, not only in the controller's UI."""
    channel, emitted = _channel()
    channel.acquire("tui")

    async def answer() -> None:
        await asyncio.sleep(0.01)
        channel.respond(channel.pending()[0].request_id, ALLOW, client_id="tui")

    task = asyncio.create_task(answer())
    await channel.request("permission", {"command": "rm -rf build"}, default=DENY)
    await task
    await _settle()

    announcement = emitted[0]
    assert announcement.session_id == "s1"
    assert announcement.content["kind"] == "permission"
    assert announcement.content["has_controller"] is True
    assert "request_id" in announcement.content


@pytest.mark.asyncio
async def test_no_controller_resolves_to_the_default_immediately() -> None:
    """An agent must never block on a prompt nobody is able to answer."""
    channel, emitted = _channel(timeout=30.0)

    result = await asyncio.wait_for(channel.request("permission", {}, default=DENY), timeout=1.0)
    await _settle()

    assert result == DENY
    assert emitted[0].content["has_controller"] is False
    resolved = [event for event in emitted if event.event_type == KnownEventType.CONTROL_RESOLVED]
    assert resolved and resolved[0].content["reason"] == "no_controller"


@pytest.mark.asyncio
async def test_silent_controller_times_out_to_the_default() -> None:
    channel, emitted = _channel(timeout=0.05)
    channel.acquire("tui")

    result = await channel.request("permission", {}, default=DENY)
    await _settle()

    assert result == DENY, "a controller that never answers must not strand the turn"
    resolved = [event for event in emitted if event.event_type == KnownEventType.CONTROL_RESOLVED]
    assert resolved and resolved[0].content["reason"] == "timeout"
    assert channel.pending() == []


@pytest.mark.asyncio
async def test_releasing_the_lease_settles_outstanding_requests() -> None:
    """Closing the only client that could answer must not leave the agent waiting."""
    channel, _ = _channel(timeout=30.0)
    channel.acquire("tui")

    async def leave() -> None:
        await asyncio.sleep(0.01)
        channel.release("tui")

    task = asyncio.create_task(leave())
    result = await asyncio.wait_for(channel.request("permission", {}, default=DENY), timeout=2.0)
    await task

    assert result == DENY
    assert channel.pending() == []


@pytest.mark.asyncio
async def test_a_viewer_cannot_answer() -> None:
    """Shared sessions are read-only, so only the lease holder may respond."""
    channel, _ = _channel()
    channel.acquire("tui")

    async def attempt() -> None:
        await asyncio.sleep(0.01)
        request_id = channel.pending()[0].request_id
        with pytest.raises(ControlLeaseError):
            channel.respond(request_id, ALLOW, client_id="viewer")
        channel.respond(request_id, ALLOW, client_id="tui")

    task = asyncio.create_task(attempt())
    result = await channel.request("permission", {}, default=DENY)
    await task

    assert result == ALLOW, "the lease holder's answer must still be accepted"


@pytest.mark.asyncio
async def test_lease_is_exclusive() -> None:
    channel, _ = _channel()
    channel.acquire("tui")
    with pytest.raises(ControlLeaseError):
        channel.acquire("web")
    # Re-acquiring by the same client is idempotent, so a reconnect is not fatal.
    channel.acquire("tui")
    channel.release("tui")
    channel.acquire("web")
    assert channel.controller == "web"


@pytest.mark.asyncio
async def test_responding_twice_is_reported_not_raised() -> None:
    channel, _ = _channel()
    channel.acquire("tui")

    async def answer_twice() -> None:
        await asyncio.sleep(0.01)
        request_id = channel.pending()[0].request_id
        assert channel.respond(request_id, ALLOW, client_id="tui") is True
        assert channel.respond(request_id, DENY, client_id="tui") is False

    task = asyncio.create_task(answer_twice())
    result = await channel.request("permission", {}, default=DENY)
    await task

    assert result == ALLOW, "a duplicate answer must not overwrite the first"


@pytest.mark.asyncio
async def test_unknown_request_id_is_reported() -> None:
    channel, _ = _channel()
    channel.acquire("tui")
    assert channel.respond("not-a-request", ALLOW, client_id="tui") is False


@pytest.mark.asyncio
async def test_pending_lets_a_client_catch_up_mid_prompt() -> None:
    """A client attaching while a prompt is open must be able to find it."""
    channel, _ = _channel(timeout=30.0)
    channel.acquire("tui")

    async def inspect_then_answer() -> None:
        await asyncio.sleep(0.01)
        pending = channel.pending()
        assert len(pending) == 1
        assert pending[0].kind == "question"
        assert pending[0].describe()["payload"] == {"question": "which one?"}
        channel.respond(pending[0].request_id, {"choice": 1}, client_id="tui")

    task = asyncio.create_task(inspect_then_answer())
    result = await channel.request("question", {"question": "which one?"}, default={"choice": None})
    await task

    assert result == {"choice": 1}


@pytest.mark.asyncio
@pytest.mark.parametrize("settlement", ["answer", "cancel", "release", "timeout", "no_controller"])
async def test_is_pending_tracks_the_request_until_settlement(settlement: str) -> None:
    emitted: list[AgentEvent] = []
    announced = asyncio.Event()
    proceed = asyncio.Event()

    async def emit(event: AgentEvent) -> None:
        emitted.append(event)
        if event.event_type == KnownEventType.CONTROL_REQUESTED:
            announced.set()
            await proceed.wait()

    channel = ControlChannel(session_id="s1", emit=emit, timeout=0.01 if settlement == "timeout" else 30.0)
    if settlement != "no_controller":
        channel.acquire("tui")
    assert not channel.is_pending("unknown")
    task = asyncio.create_task(channel.request("permission", {}, default=DENY))
    try:
        await asyncio.wait_for(announced.wait(), 5)
        request_id = emitted[0].content["request_id"]
        assert channel.is_pending(request_id)
        assert not channel.is_pending("unknown")
        if settlement == "answer":
            assert channel.respond(request_id, ALLOW, client_id="tui")
        elif settlement == "release":
            channel.release("tui")
        elif settlement == "cancel":
            task.cancel()
        proceed.set()
        if settlement == "cancel":
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            assert await asyncio.wait_for(task, 5) == (ALLOW if settlement == "answer" else DENY)
        assert not channel.is_pending(request_id)
        assert channel.pending() == []
        await _settle()
    finally:
        proceed.set()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_cancelling_a_turn_drops_the_request() -> None:
    """A late answer must not resolve a request nobody is waiting on."""
    channel, _ = _channel(timeout=30.0)
    channel.acquire("tui")

    task = asyncio.create_task(channel.request("permission", {}, default=DENY))
    await asyncio.sleep(0.01)
    request_id = channel.pending()[0].request_id
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert channel.pending() == []
    assert channel.respond(request_id, ALLOW, client_id="tui") is False


@pytest.mark.asyncio
async def test_emit_failure_does_not_break_the_round_trip() -> None:
    """Announcing is observability; the answer must still get through."""

    async def broken_emit(event: AgentEvent) -> None:
        raise RuntimeError("transport down")

    channel = ControlChannel(session_id="s1", emit=broken_emit, timeout=5.0)
    channel.acquire("tui")

    async def answer() -> None:
        await asyncio.sleep(0.01)
        channel.respond(channel.pending()[0].request_id, ALLOW, client_id="tui")

    task = asyncio.create_task(answer())
    result = await channel.request("permission", {}, default=DENY)
    await task

    assert result == ALLOW


@pytest.mark.asyncio
async def test_delivery_marker_is_idempotent_and_stale_ids_are_inert() -> None:
    channel, _ = _channel()
    channel.acquire("tui")
    task = asyncio.create_task(channel.request("question", {"question": "?"}, default=DENY))
    await asyncio.sleep(0)
    (request,) = channel.pending()
    assert channel.mark_delivered(request.request_id) is request
    assert channel.mark_delivered(request.request_id) is None
    channel.respond(request.request_id, ALLOW, client_id="tui")
    assert await task == ALLOW
    assert channel.mark_delivered(request.request_id) is None
    assert channel.mark_delivered("unknown") is None


@pytest.mark.asyncio
async def test_no_controller_cannot_acknowledge_or_answer_during_announcement() -> None:
    attempted: list[tuple[object, bool]] = []

    async def emit(event: AgentEvent) -> None:
        if event.event_type == KnownEventType.CONTROL_REQUESTED:
            request_id = event.content["request_id"]
            attempted.append((channel.mark_delivered(request_id), channel.respond(request_id, ALLOW)))

    channel = ControlChannel(session_id="s1", emit=emit)
    assert await channel.request("permission", {}, default=DENY) == DENY
    assert attempted == [(None, False)]


@pytest.mark.asyncio
async def test_cancellation_during_announcement_drops_pending_request() -> None:
    announced = asyncio.Event()

    async def emit(event: AgentEvent) -> None:
        if event.event_type == KnownEventType.CONTROL_REQUESTED:
            announced.set()
            await asyncio.Event().wait()

    channel = ControlChannel(session_id="s1", emit=emit)
    channel.acquire("tui")
    task = asyncio.create_task(channel.request("permission", {}, default=DENY))
    await asyncio.wait_for(announced.wait(), 5)
    request_id = channel.pending()[0].request_id
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert channel.pending() == []
    assert channel.mark_delivered(request_id) is None
    assert channel.respond(request_id, ALLOW) is False
