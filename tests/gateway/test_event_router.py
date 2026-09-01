"""EventRouter: type-based fan-out from one agent event source."""

import asyncio

import pytest

from kolega_code.events import AgentEvent
from kolega_code.gateway.event_router import EventRouter


def event(event_type: str) -> AgentEvent:
    return AgentEvent(event_type=event_type, sender="system", content={"k": event_type})


@pytest.mark.asyncio
async def test_routes_events_to_subscribed_kinds_only() -> None:
    source: asyncio.Queue[AgentEvent] = asyncio.Queue()
    router = EventRouter(source)
    chat_queue = router.subscribe("chat_message")
    control_queue = router.subscribe("control_requested", "control_resolved")
    router.start()

    await source.put(event("chat_message"))
    await source.put(event("control_requested"))
    await source.put(event("control_resolved"))
    await source.put(event("some_future_event_type"))

    first = await chat_queue.get()
    second = await control_queue.get()
    third = await control_queue.get()
    assert first.event_type == "chat_message"
    assert second.event_type == "control_requested"
    assert third.event_type == "control_resolved"
    # Unsubscribed kinds are dropped, and no queue receives the wrong kind.
    assert chat_queue.empty()
    assert control_queue.empty()
    await router.stop()


@pytest.mark.asyncio
async def test_multiple_subscribers_share_a_kind_without_stealing() -> None:
    source: asyncio.Queue[AgentEvent] = asyncio.Queue()
    router = EventRouter(source)
    first = router.subscribe("chat_message")
    second = router.subscribe("chat_message")
    router.start()

    await source.put(event("chat_message"))
    # One event fans out to every subscriber of its kind.
    assert (await first.get()).event_type == "chat_message"
    assert (await second.get()).event_type == "chat_message"
    await router.stop()


@pytest.mark.asyncio
async def test_stop_cancels_the_router_task() -> None:
    source: asyncio.Queue[AgentEvent] = asyncio.Queue()
    router = EventRouter(source)
    queue = router.subscribe("chat_message")
    router.start()
    await router.stop()
    # A second stop is a no-op.
    await router.stop()
    await source.put(event("chat_message"))
    assert queue.empty()
