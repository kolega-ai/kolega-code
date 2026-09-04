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


def tool_event(description: str = "bash", call_id: str = "t-1") -> AgentEvent:
    return AgentEvent(
        event_type="chat_message",
        sender="system",
        content={"message_type": "tool_call", "text": "", "tool_description": description, "tool_call_id": call_id},
    )


def tool_result_event(description: str = "bash", call_id: str = "t-1", *, error: bool = False) -> AgentEvent:
    return AgentEvent(
        event_type="chat_message",
        sender="system",
        content={
            "message_type": "tool_error" if error else "tool_result",
            "text": "",
            "tool_description": description,
            "tool_call_id": call_id,
        },
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
async def test_typing_is_refreshed_while_the_turn_runs() -> None:
    """Chat platforms expire the typing action after ~5s: a slow turn must
    re-send it so the indicator keeps showing through thinking and tool
    runs, and stops when the turn ends."""
    adapter = FakeAdapter()

    async def slow_chunks() -> AsyncIterator[dict[str, Any]]:
        for _ in range(3):
            await asyncio.sleep(0.05)
            yield {"type": "response", "content": "x", "complete": False, "uuid": "u-1"}
        yield {"type": "response", "content": "!", "complete": True, "uuid": "u-1"}

    r = TurnRenderer(
        adapter,
        "chat-1",
        event_queue=asyncio.Queue(),
        edit_throttle_seconds=0.0,
        typing_refresh_seconds=0.01,
    )
    await r.run(slow_chunks())
    typing_on = [call for call in adapter.calls if call == ("typing", "chat-1", True)]
    # The initial action plus refreshes across the turn's quiet stretches.
    assert len(typing_on) >= 3
    assert adapter.calls[-1] == ("typing", "chat-1", False)


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
async def test_tool_round_renders_as_persistent_message() -> None:
    adapter = FakeAdapter()
    queue: asyncio.Queue[AgentEvent] = asyncio.Queue()
    r = TurnRenderer(adapter, "chat-1", event_queue=queue, edit_throttle_seconds=0.0)
    await queue.put(tool_event("bash"))
    await queue.put(tool_event("read"))

    async def slow_chunks() -> AsyncIterator[dict[str, Any]]:
        # The pump drains the pre-queued round before the stream starts.
        await asyncio.sleep(0.1)
        yield {"type": "response", "content": "do", "complete": False, "uuid": "u-1"}
        yield {"type": "response", "content": "ne", "complete": True, "uuid": "u-1"}

    text = await r.run(slow_chunks())
    assert text == "done"
    status_id = [call[2] for call in adapter.calls if call[0] == "send" and call[3] != "done"][0]
    assert ("send", "chat-1", status_id, "⏳ bash") in adapter.calls
    assert ("edit", "chat-1", status_id, "⏳ bash\n⏳ read") in adapter.calls
    # The round message persists as the turn's trail: no cleanup at the end.
    assert not [call for call in adapter.calls if call[0] == "delete"]


@pytest.mark.asyncio
async def test_new_tool_round_opens_message_below_latest_text() -> None:
    """A tool round that starts after a reply bubble was sent opens its own
    message below the latest text — it never appends to the previous round's
    message higher up the chat, and round messages persist as the trail."""
    adapter = FakeAdapter()
    queue: asyncio.Queue[AgentEvent] = asyncio.Queue()
    r = TurnRenderer(adapter, "chat-1", event_queue=queue, edit_throttle_seconds=0.0)
    await queue.put(tool_event("bash"))
    gate = asyncio.Event()

    async def tool_rounds() -> AsyncIterator[dict[str, Any]]:
        yield {"type": "response", "content": "before", "complete": False, "uuid": "u-1"}
        await gate.wait()
        yield {"type": "response", "content": "", "complete": True, "uuid": "u-1"}
        yield {"type": "response", "content": "after", "complete": True, "uuid": "u-2"}
        await queue.put(tool_event("read"))
        await asyncio.sleep(0.1)  # pump opens the second round message
        yield {"type": "response", "content": "", "complete": True, "uuid": "u-2"}

    run_task = asyncio.create_task(r.run(tool_rounds()))
    await asyncio.sleep(0.05)  # pump creates the first round message
    gate.set()
    await run_task

    sends = [call for call in adapter.calls if call[0] == "send"]
    # Round 1 first, then the reply bubbles, then round 2 as a fresh message
    # below "after" carrying only its own tool line.
    assert [call[3] for call in sends] == ["⏳ bash", "before", "after", "⏳ read"]
    # Nothing is ever deleted: round messages are the turn's persistent trail.
    assert not [call for call in adapter.calls if call[0] == "delete"]


@pytest.mark.asyncio
async def test_tool_results_replace_pending_lines_in_place() -> None:
    """A finished call settles its pending line (⏳ bash -> ✅ bash) instead of
    appending a generic 'tool finished' line, so the round message reads as a
    settled trail in call order."""
    adapter = FakeAdapter()
    queue: asyncio.Queue[AgentEvent] = asyncio.Queue()
    r = TurnRenderer(adapter, "chat-1", event_queue=queue, edit_throttle_seconds=0.0)
    await queue.put(tool_event("bash", call_id="t-1"))
    await queue.put(tool_event("read", call_id="t-2"))
    await queue.put(tool_result_event("bash", call_id="t-1"))
    await queue.put(tool_result_event("read", call_id="t-2"))

    async def slow_chunks() -> AsyncIterator[dict[str, Any]]:
        # The pump drains the pre-queued round before the stream starts.
        await asyncio.sleep(0.1)
        yield {"type": "response", "content": "hi!", "complete": True, "uuid": "u-1"}

    text = await r.run(slow_chunks())
    assert text == "hi!"

    status_id = [call[2] for call in adapter.calls if call[0] == "send" and call[3] != "hi!"][0]
    assert ("send", "chat-1", status_id, "⏳ bash") in adapter.calls
    assert ("edit", "chat-1", status_id, "⏳ bash\n⏳ read") in adapter.calls
    assert ("edit", "chat-1", status_id, "✅ bash\n⏳ read") in adapter.calls
    assert ("edit", "chat-1", status_id, "✅ bash\n✅ read") in adapter.calls


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


@pytest.mark.asyncio
async def test_segment_uuid_rotation_opens_new_bubbles() -> None:
    """A tool round rotates the stream uuid: the pre-tool text settles as one
    bubble and the post-tool text arrives as a new one (the TUI's fold rule)."""
    adapter = FakeAdapter()

    async def tool_round() -> AsyncIterator[dict[str, Any]]:
        yield {"type": "response", "content": "before tool", "complete": False, "uuid": "u-1"}
        yield {"type": "response", "content": "", "complete": True, "uuid": "u-1"}
        # The agent rotates the uuid after the tool round.
        yield {"type": "response", "content": "after tool", "complete": True, "uuid": "u-2"}

    text = await renderer(adapter).run(tool_round())
    assert text == "before toolafter tool"
    # Two separate bubbles, not one merged/edit-chained message.
    sends = [call for call in adapter.calls if call[0] == "send"]
    assert [call[3] for call in sends] == ["before tool", "after tool"]
    assert not any(call[0] == "edit" and call[3] == "before toolafter tool" for call in adapter.calls)


@pytest.mark.asyncio
async def test_status_message_skips_unchanged_text() -> None:
    """Repeated identical status lines (line-cap truncation) must not re-edit
    the same content — Telegram rejects that as 'message is not modified'."""
    adapter = FakeAdapter()
    queue: asyncio.Queue[AgentEvent] = asyncio.Queue()
    r = TurnRenderer(adapter, "chat-1", event_queue=queue, edit_throttle_seconds=0.0)
    for _ in range(9):
        await queue.put(tool_event("bash"))

    run_task = asyncio.create_task(r.run(_slow_done_chunks()))
    await asyncio.sleep(0.1)  # pump drains all nine events (capped at 8 lines)
    status_id = next(call[2] for call in adapter.calls if call[0] == "send" and call[3].startswith("⏳"))
    await run_task
    edits = [call for call in adapter.calls if call[0] == "edit" and call[2] == status_id]
    # First event sends, events 2-8 grow the text, and the ninth renders the
    # same capped 8-line text: seven edits, no duplicate-content re-edit.
    assert len(edits) == 7


async def _slow_done_chunks() -> AsyncIterator[dict[str, Any]]:
    await asyncio.sleep(0.15)
    yield {"type": "response", "content": "done", "complete": True, "uuid": "u-1"}
