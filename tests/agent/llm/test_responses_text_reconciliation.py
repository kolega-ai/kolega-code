from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any

import pytest

from kolega_code.llm.exceptions import LLMIncompleteResponseError
from kolega_code.llm.models import Message, MessageChunk, ToolCall
from kolega_code.llm.providers.responses_common import ResponsesStreamWrapper


class _FakeStream:
    def __init__(self, events: list[SimpleNamespace]) -> None:
        self._events = events

    async def __aiter__(self) -> AsyncIterator[SimpleNamespace]:
        for event in self._events:
            yield event


def _delta(text: str, **identity: Any) -> SimpleNamespace:
    return SimpleNamespace(type="response.output_text.delta", delta=text, **identity)


def _message(text: str, item_id: str = "msg_1") -> SimpleNamespace:
    return SimpleNamespace(
        type="message",
        id=item_id,
        content=[SimpleNamespace(type="output_text", text=text)],
    )


def _completed(output: list[SimpleNamespace] | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        type="response.completed",
        response=SimpleNamespace(output=output or [], usage=None, status="completed", incomplete_details=None),
    )


_SNAPSHOT_SOURCES = (
    "response.output_text.done",
    "response.content_part.done",
    "response.output_item.done",
    "response.completed",
)


def _snapshot(source: str, text: str) -> SimpleNamespace:
    if source == "response.completed":
        return _completed([_message(text)])
    if source == "response.output_item.done":
        return SimpleNamespace(type=source, output_index=0, item=_message(text))
    payload = (
        {"text": text}
        if source == "response.output_text.done"
        else {"part": SimpleNamespace(type="output_text", text=text)}
    )
    return SimpleNamespace(type=source, item_id="msg_1", output_index=0, content_index=0, **payload)


async def _drain(events: list[SimpleNamespace]) -> tuple[list[MessageChunk], Message]:
    wrapper = ResponsesStreamWrapper(_FakeStream(events))
    async with wrapper:
        chunks = [chunk async for chunk in wrapper]
    return chunks, await wrapper.get_final_message()


@pytest.mark.asyncio
async def test_terminal_snapshot_replaces_partial_delta_for_same_part() -> None:
    chunks, message = await _drain(
        [
            _delta("vision-", item_id="msg_1", output_index=0, content_index=0),
            _completed([_message("vision-ok")]),
        ]
    )

    assert [chunk.text for chunk in chunks if chunk.type == "text"] == ["vision-"]
    assert message.get_text_content() == "vision-ok"


@pytest.mark.asyncio
@pytest.mark.parametrize("source", _SNAPSHOT_SOURCES)
@pytest.mark.parametrize("deltas", [[], ["vision-"], ["vision-", "ok"]], ids=["snapshot-only", "partial", "full"])
async def test_completed_text_reconciles_each_snapshot_source(source: str, deltas: list[str]) -> None:
    chunks, message = await _drain(
        [_delta(text, item_id="msg_1", output_index=0, content_index=0) for text in deltas]
        + [_snapshot(source, "vision-ok")]
        + ([_completed()] if source != "response.completed" else [])
    )

    assert [chunk.text for chunk in chunks if chunk.type == "text"] == deltas
    assert message.get_text_content() == "vision-ok"
    assert message.stop_reason == "end_turn"


@pytest.mark.asyncio
@pytest.mark.parametrize("text", ["corrected", "vis", ""])
@pytest.mark.parametrize("source", _SNAPSHOT_SOURCES)
async def test_snapshot_replaces_same_part_even_when_not_an_extension(source: str, text: str) -> None:
    _, message = await _drain(
        [
            _delta("vision-", item_id="msg_1", output_index=0, content_index=0),
            _snapshot(source, text),
        ]
        + ([_completed()] if source != "response.completed" else [])
    )

    assert message.get_text_content() == text


@pytest.mark.asyncio
async def test_all_snapshot_layers_reconcile_once_and_final_snapshot_wins() -> None:
    chunks, message = await _drain(
        [
            _delta("vision-", item_id="msg_1", output_index=0, content_index=0),
            _snapshot("response.output_text.done", "vision-o"),
            _snapshot("response.output_text.done", "vision-o"),
            _snapshot("response.content_part.done", "vision-ok"),
            _snapshot("response.output_item.done", "vision-ok"),
            _snapshot("response.completed", "vision-okay"),
        ]
    )

    assert [chunk.text for chunk in chunks if chunk.type == "text"] == ["vision-"]
    assert message.get_text_content() == "vision-okay"


@pytest.mark.asyncio
async def test_empty_repeated_terminal_does_not_erase_captured_snapshot() -> None:
    _, message = await _drain(
        [
            _delta("vision-", item_id="msg_1", output_index=0, content_index=0),
            _completed([_message("vision-ok")]),
            _completed(),
        ]
    )

    assert message.get_text_content() == "vision-ok"


@pytest.mark.asyncio
async def test_reconciliation_preserves_distinct_parts_and_items_with_identical_text() -> None:
    first = _message("same")
    first.content.append(SimpleNamespace(type="output_text", text="same"))
    _, message = await _drain(
        [
            _delta("sa", item_id="msg_1", output_index=0, content_index=0),
            _delta("same", item_id="msg_1", output_index=0, content_index=1),
            _delta("sa", item_id="msg_2", output_index=1, content_index=0),
            _completed([first, _message("same", "msg_2"), _message("same", "msg_3")]),
        ]
    )

    assert message.get_text_content() == "samesamesamesame"


@pytest.mark.asyncio
async def test_interleaved_part_deltas_are_reconciled_independently() -> None:
    _, message = await _drain(
        [
            _delta("first-", item_id="msg_1", output_index=0, content_index=0),
            _delta("second", item_id="msg_1", output_index=0, content_index=1),
            _delta("part", item_id="msg_1", output_index=0, content_index=0),
            SimpleNamespace(
                type="response.content_part.done",
                item_id="msg_1",
                output_index=0,
                content_index=0,
                part=SimpleNamespace(type="output_text", text="first-part!"),
            ),
            _completed(),
        ]
    )

    assert message.get_text_content() == "first-part!second"


@pytest.mark.asyncio
async def test_terminal_recovers_earlier_parts_and_items_in_output_order() -> None:
    first = _message("first-")
    first.content.append(SimpleNamespace(type="output_text", text="part"))
    _, message = await _drain(
        [
            _delta("last", item_id="msg_2", output_index=1, content_index=0),
            _delta("pa", item_id="msg_1", output_index=0, content_index=1),
            _completed([first, _message("last", "msg_2")]),
        ]
    )

    assert message.get_text_content() == "first-partlast"


@pytest.mark.asyncio
async def test_unindexed_items_keep_first_seen_order_with_parts_in_content_order() -> None:
    _, message = await _drain(
        [
            _delta("second", item_id="msg_2", content_index=1),
            _delta("first-", item_id="msg_2", content_index=0),
            _delta("last", item_id="msg_1", content_index=0),
            _completed(),
        ]
    )

    assert message.get_text_content() == "first-secondlast"


@pytest.mark.asyncio
async def test_similar_text_in_a_distinct_item_is_not_used_as_a_replacement() -> None:
    _, message = await _drain(
        [
            _delta("vision-", item_id="msg_1", output_index=0, content_index=0),
            SimpleNamespace(type="response.output_item.done", output_index=1, item=_message("vision-ok", "msg_2")),
            _completed(),
        ]
    )

    assert message.get_text_content() == "vision-vision-ok"


@pytest.mark.asyncio
@pytest.mark.parametrize("with_added_event", [False, True])
async def test_index_only_deltas_reconcile_when_item_id_becomes_available(with_added_event: bool) -> None:
    events = [_delta("vision-", output_index=2, content_index=0)]
    if with_added_event:
        events.append(SimpleNamespace(type="response.output_item.added", output_index=2, item=_message("")))
    events.extend(
        [
            SimpleNamespace(
                type="response.output_text.done", item_id="msg_1", output_index=2, content_index=0, text="vision-ok"
            ),
            SimpleNamespace(
                type="response.content_part.done",
                output_index=2,
                content_index=0,
                part=SimpleNamespace(type="output_text", text="vision-okay"),
            ),
            _completed(),
        ]
    )
    _, message = await _drain(events)

    assert message.get_text_content() == "vision-okay"


@pytest.mark.asyncio
async def test_item_id_reconciles_when_done_event_has_no_output_index() -> None:
    _, message = await _drain(
        [
            _delta("vision-", item_id="msg_1", content_index=0),
            SimpleNamespace(type="response.output_item.done", item=_message("vision-ok")),
            _completed(),
        ]
    )

    assert message.get_text_content() == "vision-ok"


@pytest.mark.asyncio
@pytest.mark.parametrize("snapshot_has_id", [False, True])
async def test_legacy_unidentified_deltas_reconcile_with_terminal_output(snapshot_has_id: bool) -> None:
    item = _message("vision-ok")
    if not snapshot_has_id:
        del item.id
    _, message = await _drain([_delta("vision-"), _completed([item])])

    assert message.get_text_content() == "vision-ok"


@pytest.mark.asyncio
async def test_empty_terminal_output_keeps_all_delta_only_parts() -> None:
    _, message = await _drain(
        [
            _delta("one", item_id="msg_1", output_index=0, content_index=0),
            _delta("two", item_id="msg_1", output_index=0, content_index=1),
            _delta("three", item_id="msg_2", output_index=1, content_index=0),
            _completed(),
        ]
    )

    assert message.get_text_content() == "onetwothree"


@pytest.mark.asyncio
async def test_non_text_part_done_is_not_added_to_assistant_text() -> None:
    _, message = await _drain(
        [
            _delta("answer", item_id="msg_1", output_index=0, content_index=0),
            SimpleNamespace(
                type="response.content_part.done",
                item_id="msg_1",
                output_index=0,
                content_index=1,
                part=SimpleNamespace(type="refusal", text="not output text"),
            ),
            _completed(),
        ]
    )

    assert message.get_text_content() == "answer"


def _tool_item() -> SimpleNamespace:
    return SimpleNamespace(
        type="function_call", id="fc_1", call_id="call_1", name="read_file", arguments='{"path":"a.py"}'
    )


@pytest.mark.asyncio
async def test_terminal_only_text_and_tool_fallback_remains_intact() -> None:
    _, message = await _drain([_completed([_message("vision-ok"), _tool_item()])])

    assert message.get_text_content() == "vision-ok"
    assert len(message.tool_calls) == 1
    assert message.tool_calls[0].id == "call_1"
    assert message.tool_calls[0].input == {"path": "a.py"}
    assert message.stop_reason == "tool_use"


@pytest.mark.asyncio
async def test_snapshot_without_terminal_response_still_rejects_tool_dispatch() -> None:
    wrapper = ResponsesStreamWrapper(
        _FakeStream(
            [
                _delta("vision-", item_id="msg_1", output_index=0, content_index=0),
                _snapshot("response.output_item.done", "vision-ok"),
                SimpleNamespace(type="response.output_item.added", output_index=1, item=_tool_item()),
            ]
        )
    )
    async with wrapper:
        chunks = [chunk async for chunk in wrapper]

    with pytest.raises(LLMIncompleteResponseError) as caught:
        await wrapper.get_final_message()

    assert all(chunk.type != "tool_use_start" for chunk in chunks)
    assert caught.value.pending_tool_calls == 1
    assert caught.value.partial_output is True
    partial = caught.value.partial_message
    assert partial is not None
    assert partial.get_text_content() == "vision-ok"
    assert partial.stop_reason == "interrupted"
    assert all(not isinstance(block, ToolCall) for block in partial.content)


@pytest.mark.asyncio
@pytest.mark.parametrize("source", _SNAPSHOT_SOURCES[:-1])
async def test_empty_snapshot_without_terminal_retains_streamed_output_signal(source: str) -> None:
    wrapper = ResponsesStreamWrapper(
        _FakeStream(
            [
                _delta("already visible", item_id="msg_1", output_index=0, content_index=0),
                _snapshot(source, ""),
            ]
        )
    )
    async with wrapper:
        chunks = [chunk async for chunk in wrapper]

    assert [chunk.text for chunk in chunks if chunk.type == "text"] == ["already visible"]
    with pytest.raises(LLMIncompleteResponseError) as caught:
        await wrapper.get_final_message()

    assert caught.value.partial_output is True
    assert caught.value.partial_message is None
    assert caught.value.pending_tool_calls == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("event_type", ["response.completed", "response.incomplete"])
async def test_incomplete_terminal_reconciles_text_but_still_suppresses_tools(event_type: str) -> None:
    terminal = _completed([_message("vision-ok"), _tool_item()])
    terminal.type = event_type
    terminal.response.status = "incomplete"
    terminal.response.incomplete_details = SimpleNamespace(reason="max_output_tokens")
    chunks, message = await _drain(
        [
            _delta("vision-", item_id="msg_1", output_index=0, content_index=0),
            SimpleNamespace(type="response.output_item.added", output_index=1, item=_tool_item()),
            terminal,
        ]
    )

    assert all(chunk.type != "tool_use_start" for chunk in chunks)
    assert message.get_text_content() == "vision-ok"
    assert message.tool_calls == []
    assert all(not isinstance(block, ToolCall) for block in message.content)
    assert message.stop_reason == "max_tokens"


@pytest.mark.asyncio
async def test_reconciled_final_message_is_idempotent_and_retains_usage_and_tools() -> None:
    terminal = _completed([_message("vision-ok")])
    terminal.response.usage = SimpleNamespace(
        input_tokens=5, output_tokens=3, total_tokens=8, input_tokens_details=None
    )
    wrapper = ResponsesStreamWrapper(
        _FakeStream(
            [
                _delta("vision-", item_id="msg_1", output_index=0, content_index=0),
                SimpleNamespace(type="response.output_item.added", output_index=1, item=_tool_item()),
                SimpleNamespace(type="response.output_item.done", output_index=1, item=_tool_item()),
                terminal,
            ]
        ),
        provider_name="deepseek",
    )
    async with wrapper:
        chunks = [chunk async for chunk in wrapper]
    first = await wrapper.get_final_message()
    second = await wrapper.get_final_message()

    assert [chunk.type for chunk in chunks].count("tool_use_start") == 1
    assert first.get_text_content() == second.get_text_content() == "vision-ok"
    assert len(first.tool_calls) == len(second.tool_calls) == 1
    assert first.tool_calls[0].execution_id == second.tool_calls[0].execution_id
    assert first.usage_metadata["prompt_tokens"] == 5
    assert first.usage_metadata["completion_tokens"] == 3
    assert first.usage_metadata["provider"] == "deepseek"
    assert first.stop_reason == second.stop_reason == "tool_use"
