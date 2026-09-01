"""TurnRenderer: edit-in-place streaming, chunking, status messages, cancellation."""

import asyncio
from typing import Any, AsyncIterator

import pytest

from kolega_code.events import AgentEvent
from kolega_code.gateway.adapters.base import AdapterCapabilities, GatewayAdapter
from kolega_code.gateway.bridge import TurnRenderer


class FakeAdapter(GatewayAdapter):
    name = "fake"

    def __init__(self, *, edits: bool = True, chunk_limit: int = 100) -> None:
        super().__init__()
        self.capabilities = AdapterCapabilities(
            supports_edits=edits,
            supports_delete=True,
            streaming_mode="edit_in_place" if edits else "final_only",
            text_chunk_limit=chunk_limit,
        )
        self.calls: list[tuple[Any, ...]] = []
        self._seq = 0

    async def send_text(self, chat_id: str, text: str, *, reply_to_message_id: str | None = None) -> str:
        await asyncio.sleep(0)  # model network I/O: yields so the event pump runs
        self._seq += 1
        message_id = f"m-{self._seq}"
        self.calls.append(("send", chat_id, message_id, text))
        return message_id

    async def edit_text(self, chat_id: str, message_id: str, text: str) -> None:
        await asyncio.sleep(0)
        self.calls.append(("edit", chat_id, message_id, text))

    async def delete_message(self, chat_id: str, message_id: str) -> None:
        await asyncio.sleep(0)
        self.calls.append(("delete", chat_id, message_id))

    async def set_typing(self, chat_id: str, active: bool) -> None:
        await asyncio.sleep(0)
        self.calls.append(("typing", chat_id, active))

    def sent_texts(self) -> list[str]:
        return [str(call[3]) for call in self.calls if call[0] == "send"]


async def chunks(*segments: str) -> AsyncIterator[dict[str, Any]]:
    for index, content in enumerate(segments):
        yield {"type": "response", "content": content, "complete": index == len(segments) - 1, "uuid": "u-1"}


def renderer(adapter: FakeAdapter, **kwargs: Any) -> TurnRenderer:
    return TurnRenderer(adapter, "chat-1", event_queue=asyncio.Queue(), edit_throttle_seconds=0.0, **kwargs)


def tool_event(description: str = "bash") -> AgentEvent:
    return AgentEvent(
        event_type="chat_message",
        sender="system",
        content={"message_type": "tool_call", "text": "", "tool_description": description, "tool_call_id": "t-1"},
    )


@pytest.mark.asyncio
async def test_streams_with_edit_in_place() -> None:
    adapter = FakeAdapter()
    text = await renderer(adapter).run(chunks("hello", " world"))
    assert text == "hello world"
    assert adapter.calls[0][0] == "typing" and adapter.calls[0][2] is True
    assert ("send", "chat-1", "m-1", "hello") in adapter.calls
    assert ("edit", "chat-1", "m-1", "hello world") in adapter.calls
    # Final edit is a no-op (text unchanged), and typing is turned off.
    assert adapter.calls[-1] == ("typing", "chat-1", False)
    assert adapter.calls.count(("edit", "chat-1", "m-1", "hello world")) == 1


@pytest.mark.asyncio
async def test_final_only_transport_sends_once() -> None:
    adapter = FakeAdapter(edits=False)
    text = await renderer(adapter).run(chunks("hello", " world"))
    assert text == "hello world"
    sends = [call for call in adapter.calls if call[0] == "send"]
    assert sends == [("send", "chat-1", "m-1", "hello world")]
    assert not [call for call in adapter.calls if call[0] == "edit"]


@pytest.mark.asyncio
async def test_over_limit_replies_send_continuations_on_finalize() -> None:
    adapter = FakeAdapter(chunk_limit=6)
    text = await renderer(adapter).run(chunks("hello ", "world"))
    assert text == "hello world"
    # The first chunk streams into the edited reply (no-op edit skipped at
    # finalize since the text is unchanged); the overflow lands as a second
    # message once the turn is complete.
    assert ("send", "chat-1", "m-2", "world") in adapter.calls
    assert ("edit", "chat-1", "m-1", "hello ") not in adapter.calls
    assert adapter.calls[-1] == ("typing", "chat-1", False)


@pytest.mark.asyncio
async def test_edit_throttle_skips_intermediate_edits() -> None:
    adapter = FakeAdapter()
    r = TurnRenderer(adapter, "chat-1", event_queue=asyncio.Queue(), edit_throttle_seconds=10.0)
    text = await r.run(chunks("hello", " world"))
    assert text == "hello world"
    # The second chunk arrives within the throttle window: no edit until the
    # final flush, which bypasses the throttle.
    assert adapter.calls == [
        ("typing", "chat-1", True),
        ("send", "chat-1", "m-1", "hello"),
        ("edit", "chat-1", "m-1", "hello world"),
        ("typing", "chat-1", False),
    ]


@pytest.mark.asyncio
async def test_tool_events_render_as_status_message_and_are_deleted() -> None:
    adapter = FakeAdapter()
    queue: asyncio.Queue[AgentEvent] = asyncio.Queue()
    r = TurnRenderer(adapter, "chat-1", event_queue=queue, edit_throttle_seconds=0.0)
    await queue.put(tool_event("bash"))
    await queue.put(tool_event("read"))

    # A real turn runs long enough for the pump to drain the queued events;
    # model that with a gate the pump can pass while the stream is suspended.
    gate = asyncio.Event()

    async def slow_chunks() -> AsyncIterator[dict[str, Any]]:
        yield {"type": "response", "content": "do", "complete": False, "uuid": "u-1"}
        await gate.wait()
        yield {"type": "response", "content": "ne", "complete": True, "uuid": "u-1"}

    run_task = asyncio.create_task(r.run(slow_chunks()))
    await asyncio.sleep(0.05)  # pump drains both tool events
    gate.set()
    text = await run_task
    assert text == "done"
    status_id = [call[2] for call in adapter.calls if call[0] == "send" and call[3] != "done"][0]
    assert ("send", "chat-1", status_id, "⏳ bash") in adapter.calls
    assert ("edit", "chat-1", status_id, "⏳ bash\n⏳ read") in adapter.calls
    assert ("delete", "chat-1", status_id) in adapter.calls


@pytest.mark.asyncio
async def test_thinking_only_turns_send_nothing() -> None:
    adapter = FakeAdapter()

    async def thinking_only() -> AsyncIterator[dict[str, Any]]:
        yield {"type": "thinking", "content": "reasoning", "complete": True, "uuid": "u-1"}

    text = await renderer(adapter).run(thinking_only())
    assert text == ""
    assert not [call for call in adapter.calls if call[0] == "send"]


@pytest.mark.asyncio
async def test_cancellation_keeps_partial_reply_and_cleans_up() -> None:
    adapter = FakeAdapter()

    async def cancelled_stream() -> AsyncIterator[dict[str, Any]]:
        yield {"type": "response", "content": "partial ", "complete": False, "uuid": "u-1"}
        raise asyncio.CancelledError

    r = renderer(adapter)
    with pytest.raises(asyncio.CancelledError):
        await r.run(cancelled_stream())
    assert "partial " in adapter.sent_texts()
    assert adapter.calls[-1] == ("typing", "chat-1", False)
