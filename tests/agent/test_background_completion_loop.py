import asyncio
from contextlib import aclosing
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from kolega_code.agent.baseagent import BaseAgent
from kolega_code.cli.session_journal import TERMINAL_TURN_EVENTS, SessionJournal, SessionJournalError, SessionRecorder
from kolega_code.cli.session_store import SessionStore
from kolega_code.llm.exceptions import LLMError
from kolega_code.llm.models import Message, TextBlock, ToolCall, ToolResult
from kolega_code.llm.providers.models import TokenCount

from .compaction_helpers import FakeLLM


@dataclass
class _StreamEvent:
    type: str
    text: str = ""
    thinking: str = ""
    tool_call_delta: dict[str, Any] | None = None


class _GatedStream:
    def __init__(
        self,
        final_message: Message,
        *,
        events: list[_StreamEvent] | None = None,
        entered: asyncio.Event | None = None,
        release: asyncio.Event | None = None,
    ) -> None:
        self._final_message = final_message
        self._events = list(events or [])
        self._entered = entered
        self._release = release
        self._released = release is None
        self._index = 0

    async def __aenter__(self) -> "_GatedStream":
        if self._entered is not None:
            self._entered.set()
        return self

    async def __aexit__(self, *exc: object) -> bool:
        _ = exc
        return False

    def __aiter__(self) -> "_GatedStream":
        return self

    async def __anext__(self) -> _StreamEvent:
        if not self._released:
            assert self._release is not None
            self._released = True
            await self._release.wait()
        if self._index >= len(self._events):
            raise StopAsyncIteration
        event = self._events[self._index]
        self._index += 1
        return event

    async def get_final_message(self) -> Message:
        return self._final_message


class _RecordingLLM:
    def __init__(self, streams: list[_GatedStream]) -> None:
        self._streams = list(streams)
        self.calls: list[dict[str, Any]] = []
        self.provider = MagicMock(base_url="https://api.test.example/v1")
        self.count_tokens = AsyncMock(return_value=TokenCount(input_tokens=0))
        self.generate = AsyncMock()
        self.stream = AsyncMock(side_effect=self._stream)

    async def _stream(self, *args: Any, **kwargs: Any) -> _GatedStream:
        self.calls.append(kwargs)
        return self._streams.pop(0)


def _tool_call_message(tool_call: ToolCall) -> Message:
    return Message(role="assistant", content=[tool_call], stop_reason="tool_use", tool_calls=[tool_call])


def _tool_call_text_message(tool_call: ToolCall, text: str) -> Message:
    return Message(
        role="assistant",
        content=[TextBlock(text=text), tool_call],
        stop_reason="tool_use",
        tool_calls=[tool_call],
    )


def _text_message(text: str = "done") -> Message:
    return Message(role="assistant", content=[TextBlock(text=text)], stop_reason="end_turn")


def _tool_result(tool_call: ToolCall, content: str = "tool ok") -> ToolResult:
    return ToolResult(tool_use_id=tool_call.id, name=tool_call.name, content=content, is_error=False)


def _configure_agent(base_agent, llm: Any) -> None:
    base_agent.system_prompt = Message(role="system", content=[TextBlock(text="sys")])
    base_agent.tool_collection = MagicMock()
    base_agent.tool_collection.get_tool_list = MagicMock(return_value=[])
    base_agent.llm = llm
    base_agent.log_info = AsyncMock()
    base_agent.log_error = AsyncMock()
    base_agent.log_warning = AsyncMock()


async def _complete_background_job(base_agent, job_id: str = "job-1", result: str = "child report") -> None:
    jobs = base_agent.get_background_jobs()

    async def run() -> str:
        return result

    await jobs.submit(job_id, run, agent_type="investigation", task="research")
    await asyncio.wait_for(jobs.wait_for_completion(), timeout=3)


def _is_completion_message(message: Message) -> bool:
    if not isinstance(message.content, list):
        return False
    return any(
        isinstance(block, TextBlock) and "Background subagent completion." in block.text for block in message.content
    )


def _find_message_index(messages: list[Message], predicate) -> int:
    for index, message in enumerate(messages):
        if predicate(message):
            return index
    raise AssertionError("matching message not found")


def _completion_messages(messages: list[Message]) -> list[Message]:
    return [message for message in messages if _is_completion_message(message)]


def _tool_results(messages: list[Message]) -> list[ToolResult]:
    return [block for message in messages for block in (message.content or []) if isinstance(block, ToolResult)]


def _attach_session_recorder(base_agent, tmp_path: Path) -> SessionRecorder:
    journal = SessionJournal(base_agent.thread_id, tmp_path / "session")
    journal.start_epoch("test")
    recorder = SessionRecorder(journal, recover=False)
    base_agent.session_recorder = recorder
    return recorder


def _turn_events(recorder: SessionRecorder) -> list[Any]:
    events = recorder.journal.read_events()
    turn_ids = [event.turn_id for event in events if event.event_type == "turn.started"]
    assert turn_ids, "turn never started"
    turn_id = turn_ids[-1]
    return [event for event in events if event.turn_id == turn_id]


def _terminal_turn_events(recorder: SessionRecorder) -> list[Any]:
    return [event for event in _turn_events(recorder) if event.event_type in TERMINAL_TURN_EVENTS]


@pytest.mark.asyncio
async def test_completion_during_text_only_model_delivers_after_turn_and_leaves_needs_turn(base_agent) -> None:
    entered = asyncio.Event()
    release = asyncio.Event()
    llm = _RecordingLLM(
        [
            _GatedStream(
                _text_message("done"),
                events=[_StreamEvent(type="text", text="done")],
                entered=entered,
                release=release,
            )
        ]
    )
    _configure_agent(base_agent, llm)

    async def run_turn() -> list[dict[str, Any]]:
        return [chunk async for chunk in base_agent.process_message_stream("work")]

    task = asyncio.create_task(run_turn())
    await asyncio.wait_for(entered.wait(), timeout=3)

    await _complete_background_job(base_agent)

    assert base_agent.get_background_jobs().pending_count == 1
    assert _completion_messages(base_agent.history) == []
    assert base_agent.background_needs_turn is False

    release.set()
    chunks = await asyncio.wait_for(task, timeout=3)

    assert [chunk["content"] for chunk in chunks if chunk["type"] == "response"] == ["done"]
    completions = _completion_messages(base_agent.history)
    assert len(completions) == 1
    assert "child report" in completions[0].get_text_content()
    assert base_agent.background_needs_turn is True
    assert base_agent.get_background_jobs().pending_count == 0
    assert llm.stream.await_count == 1

    assistant_index = _find_message_index(
        base_agent.history,
        lambda message: message.role == "assistant" and message.get_text_content() == "done",
    )
    completion_index = _find_message_index(base_agent.history, _is_completion_message)
    assert assistant_index < completion_index

    await base_agent.get_background_jobs().close()


@pytest.mark.asyncio
async def test_completion_ready_before_tool_receipt_waits_for_tool_batch_then_is_consumed(base_agent) -> None:
    tool_call = ToolCall(id="tool-1", name="read_file", input={})
    model_entered = asyncio.Event()
    release_model = asyncio.Event()
    tool_started = asyncio.Event()
    release_tool = asyncio.Event()
    llm = _RecordingLLM(
        [
            _GatedStream(_tool_call_message(tool_call), entered=model_entered, release=release_model),
            _GatedStream(_text_message("done"), events=[_StreamEvent(type="text", text="done")]),
        ]
    )
    _configure_agent(base_agent, llm)

    async def process_tool_calls(_tool_calls: list[ToolCall]) -> list[ToolResult]:
        tool_started.set()
        await release_tool.wait()
        return [_tool_result(tool_call)]

    base_agent.process_tool_calls = AsyncMock(side_effect=process_tool_calls)

    async def run_turn() -> list[dict[str, Any]]:
        return [chunk async for chunk in base_agent.process_message_stream("work")]

    task = asyncio.create_task(run_turn())
    await asyncio.wait_for(model_entered.wait(), timeout=3)

    await _complete_background_job(base_agent)

    assert base_agent.get_background_jobs().pending_count == 1
    assert _completion_messages(base_agent.history) == []

    release_model.set()
    await asyncio.wait_for(tool_started.wait(), timeout=3)
    assert _completion_messages(base_agent.history) == []

    release_tool.set()
    await asyncio.wait_for(task, timeout=3)

    assert llm.stream.await_count == 2
    assert base_agent.background_needs_turn is False
    assert base_agent.get_background_jobs().pending_count == 0

    second_request_messages = list(llm.calls[1]["messages"])
    tool_result_index = _find_message_index(
        second_request_messages,
        lambda message: (
            isinstance(message.content, list) and any(isinstance(block, ToolResult) for block in message.content)
        ),
    )
    completion_index = _find_message_index(second_request_messages, _is_completion_message)
    assert tool_result_index < completion_index

    history_tool_result_index = _find_message_index(
        base_agent.history,
        lambda message: (
            isinstance(message.content, list) and any(isinstance(block, ToolResult) for block in message.content)
        ),
    )
    history_completion_index = _find_message_index(base_agent.history, _is_completion_message)
    final_assistant_index = _find_message_index(
        base_agent.history,
        lambda message: message.role == "assistant" and message.get_text_content() == "done",
    )
    assert history_tool_result_index < history_completion_index < final_assistant_index

    await base_agent.get_background_jobs().close()


@pytest.mark.asyncio
async def test_cancelled_tool_batch_settles_tool_results_before_background_completion(base_agent) -> None:
    tool_call = ToolCall(id="tool-1", name="read_file", input={})
    started = asyncio.Event()
    llm = _RecordingLLM([_GatedStream(_tool_call_message(tool_call))])
    _configure_agent(base_agent, llm)

    async def swallowing_tool_batch(_tool_calls: list[ToolCall]) -> list[ToolResult]:
        started.set()
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            return [_tool_result(tool_call, content="Status: cancelled")]
        raise AssertionError("tool batch should have been cancelled")

    base_agent.process_tool_calls = AsyncMock(side_effect=swallowing_tool_batch)

    async def run_turn() -> list[dict[str, Any]]:
        return [chunk async for chunk in base_agent.process_message_stream("work")]

    task = asyncio.create_task(run_turn())
    await asyncio.wait_for(started.wait(), timeout=3)

    await _complete_background_job(base_agent)
    assert base_agent.get_background_jobs().pending_count == 1

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=3)

    assert llm.stream.await_count == 1
    assert base_agent.background_needs_turn is True
    assert base_agent.get_background_jobs().pending_count == 0

    tool_result_index = _find_message_index(
        base_agent.history,
        lambda message: (
            isinstance(message.content, list) and any(isinstance(block, ToolResult) for block in message.content)
        ),
    )
    completion_index = _find_message_index(base_agent.history, _is_completion_message)
    assert tool_result_index < completion_index

    await base_agent.get_background_jobs().close()


@pytest.mark.asyncio
async def test_cancel_during_threaded_tool_call_persistence_waits_for_commit_and_keeps_completion_pending(
    base_agent, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tool_call = ToolCall(id="tool-1", name="read_file", input={})
    recorder = _attach_session_recorder(base_agent, tmp_path)
    llm = FakeLLM(token_script=[100], final_message=_tool_call_message(tool_call))
    _configure_agent(base_agent, llm)

    entered = asyncio.Event()
    release = threading.Event()
    loop = asyncio.get_running_loop()
    ordering: list[str] = []

    original_record_assistant = recorder.record_assistant
    original_write = recorder.journal._write_event_locked
    original_record_background_completion = recorder.record_background_completion
    original_append_user_message = base_agent.append_user_message

    def blocked_record_assistant(
        message: Message, *, reasoning_effort: str | None = None, defer_projection: bool = False
    ) -> None:
        ordering.append("assistant_record_started")
        loop.call_soon_threadsafe(entered.set)
        assert release.wait(timeout=10), "test did not release assistant commit"
        original_record_assistant(message, reasoning_effort=reasoning_effort, defer_projection=defer_projection)
        ordering.append("assistant_record_committed")

    def record_event(event) -> None:
        if event.event_type == "tool.results":
            ordering.append("tool_results_journal")
        original_write(event)

    def record_background_completion(*args: Any, **kwargs: Any) -> bool:
        ordering.append("completion_journal")
        return original_record_background_completion(*args, **kwargs)

    def append_user_message(blocks) -> None:
        if any(isinstance(block, ToolResult) for block in blocks):
            ordering.append("tool_results_live")
        if any(isinstance(block, TextBlock) and "Background subagent completion." in block.text for block in blocks):
            ordering.append("completion_live")
        original_append_user_message(blocks)

    monkeypatch.setattr(recorder, "record_assistant", blocked_record_assistant)
    monkeypatch.setattr(recorder.journal, "_write_event_locked", record_event)
    monkeypatch.setattr(recorder, "record_background_completion", record_background_completion)
    monkeypatch.setattr(base_agent, "append_user_message", append_user_message)

    async def run_turn() -> list[dict[str, Any]]:
        return [chunk async for chunk in base_agent.process_message_stream("work")]

    task = asyncio.create_task(run_turn())
    await asyncio.wait_for(entered.wait(), timeout=3)

    await _complete_background_job(base_agent)
    assert base_agent.get_background_jobs().pending_count == 1

    task.cancel()
    await asyncio.sleep(0.05)
    assert not task.done()

    release.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=3)

    assert "assistant_record_committed" in ordering
    assert "tool_results_journal" in ordering
    assert ordering.index("assistant_record_committed") < ordering.index("tool_results_journal")
    assert ordering.index("tool_results_journal") < ordering.index("tool_results_live")
    assert "completion_journal" not in ordering
    assert "completion_live" not in ordering

    results = _tool_results(base_agent.history)
    assert len(results) == 1
    assert results[0].tool_use_id == tool_call.id
    assert results[0].name == tool_call.name
    assert results[0].is_error is True
    assert "Tool execution was interrupted" in str(results[0].content)

    assert _completion_messages(base_agent.history) == []
    assert base_agent.get_background_jobs().pending_count == 1
    assert base_agent.background_needs_turn is False

    turn_events = _turn_events(recorder)
    event_types = [event.event_type for event in turn_events]
    assert "assistant.message" in event_types
    assert "tool.results" in event_types
    assert _terminal_turn_events(recorder)[-1].event_type == "turn.cancelled"
    assert event_types.index("assistant.message") < event_types.index("tool.results")

    await base_agent.get_background_jobs().close()


async def _close_or_cancel_after_first_chunk(base_agent, mode: str) -> dict[str, Any]:
    if mode == "close":
        gen = base_agent.process_message_stream("work")
        first = await gen.__anext__()
        await gen.aclose()
        return first

    if mode == "cancel":
        pause = asyncio.Event()
        observed: list[dict[str, Any]] = []

        async def consume() -> None:
            async with aclosing(base_agent.process_message_stream("work")) as stream:
                async for chunk in stream:
                    observed.append(chunk)
                    await pause.wait()

        task = asyncio.create_task(consume())
        deadline = asyncio.get_running_loop().time() + 3
        while not observed and asyncio.get_running_loop().time() < deadline:
            await asyncio.sleep(0.01)
        assert observed, "first chunk was never yielded"
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=3)
        return observed[0]

    raise AssertionError(f"unexpected mode: {mode}")


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["close", "cancel"])
async def test_early_generator_close_or_cancel_after_tool_call_chunk_records_interrupted_result_and_turn_terminal(
    base_agent, tmp_path: Path, mode: str
) -> None:
    tool_call = ToolCall(id="tool-1", name="read_file", input={})
    recorder = _attach_session_recorder(base_agent, tmp_path)
    text = "ok"
    llm = FakeLLM(token_script=[100])
    llm.stream = AsyncMock(
        return_value=_GatedStream(
            _tool_call_text_message(tool_call, text),
            events=[_StreamEvent(type="text", text=text)],
        )
    )
    _configure_agent(base_agent, llm)
    base_agent.process_tool_calls = AsyncMock(return_value=[_tool_result(tool_call, content="tool unexpectedly ran")])

    first = await _close_or_cancel_after_first_chunk(base_agent, mode)

    assert first["type"] == "response"
    assert first["content"] == text
    assert first["complete"] is True

    results = _tool_results(base_agent.history)
    assert len(results) == 1
    assert results[0].tool_use_id == tool_call.id
    assert results[0].name == tool_call.name
    assert "Tool execution was interrupted" in str(results[0].content)
    assert str(results[0].content) != "tool unexpectedly ran"

    terminal_events = _terminal_turn_events(recorder)
    assert terminal_events, "turn never reached a terminal journal event"
    assert terminal_events[-1].event_type in {"turn.cancelled", "turn.failed"}
    assert recorder.current_turn_id is None

    turn_events = _turn_events(recorder)
    event_types = [event.event_type for event in turn_events]
    assert "assistant.message" in event_types
    assert "tool.results" in event_types
    assert (
        event_types.index("assistant.message")
        < event_types.index("tool.results")
        < max(index for index, name in enumerate(event_types) if name in TERMINAL_TURN_EVENTS)
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", ["idle", "new-prompt", "continuation", "rebuilt-idle", "archive"])
@pytest.mark.parametrize("projection_failure", ["none", "before-append", "after-append"])
async def test_failed_tool_result_write_retries_original_output_before_new_context(
    base_agent, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, boundary: str, projection_failure: str
) -> None:
    call = ToolCall(id="original-tool", name="read", input={}, execution_id="original-execution")
    recorder = _attach_session_recorder(base_agent, tmp_path)
    entered, release = asyncio.Event(), asyncio.Event()
    llm = _RecordingLLM(
        [
            _GatedStream(_tool_call_message(call), entered=entered, release=release),
            _GatedStream(_text_message("continued")),
        ]
    )
    _configure_agent(base_agent, llm)
    original_result = _tool_result(call, "ORIGINAL SUCCESSFUL OUTPUT")
    original_result.execution_id = call.execution_id
    base_agent.process_tool_calls = AsyncMock(return_value=[original_result])
    original_write = recorder.journal._write_event_locked

    def fail_results(event) -> None:
        if event.event_type == "tool.results":
            raise SessionJournalError("tool-result disk unavailable")
        original_write(event)

    monkeypatch.setattr(recorder.journal, "_write_event_locked", fail_results)

    async def run_turn() -> None:
        async for _ in base_agent.process_message_stream("original prompt"):
            pass

    running = asyncio.create_task(run_turn())
    manager = base_agent.get_background_jobs()
    try:
        await asyncio.wait_for(entered.wait(), timeout=3)
        await _complete_background_job(base_agent, result="retained child report")
        release.set()
        with pytest.raises(SessionJournalError, match="tool-result disk unavailable"):
            await asyncio.wait_for(running, timeout=3)
        assert recorder.current_turn_id is None
        assert manager.pending_count == 1
        assert not _completion_messages(base_agent.history)
        # Continued storage failure cannot acknowledge a report or start a request.
        with pytest.raises(SessionJournalError, match="tool-result disk unavailable"):
            await manager.deliver(base_agent)
        assert manager.pending_count == 1
        assert not _completion_messages(base_agent.history)
        assert llm.stream.await_count == 1

        monkeypatch.setattr(recorder.journal, "_write_event_locked", original_write)
        if projection_failure != "none":
            original_append = base_agent.append_user_message

            def fail_projection(blocks) -> None:
                if any(isinstance(block, ToolResult) for block in blocks):
                    if projection_failure == "after-append":
                        original_append(blocks)
                    raise RuntimeError("synthetic live projection failure")
                original_append(blocks)

            monkeypatch.setattr(base_agent, "append_user_message", fail_projection)
            with pytest.raises(RuntimeError, match="synthetic live projection failure"):
                await manager.deliver(base_agent)
            assert manager.pending_count == 1
            assert not _completion_messages(base_agent.history)
            monkeypatch.setattr(base_agent, "append_user_message", original_append)
        target = base_agent
        if boundary == "rebuilt-idle":
            target = BaseAgent(context=base_agent.context)
            target.restore_message_history([message.to_dict() for message in base_agent.history])
            target.session_recorder = recorder
            target.background_jobs = manager
            _configure_agent(target, llm)
        if boundary in {"idle", "rebuilt-idle", "archive"}:
            assert await manager.deliver(target, archive=boundary == "archive") == 1
        if boundary == "new-prompt":
            stream = target.process_message_stream("next prompt")
        else:
            stream = target.continue_from_history_stream()
        async with aclosing(stream):
            async for _ in stream:
                pass

        base_agent.process_tool_calls.assert_awaited_once()
        assert manager.pending_count == 0
        events = recorder.journal.read_events()
        batches = [event for event in events if event.event_type == "tool.results"]
        assert len(batches) == 1
        completion = next(event for event in events if event.payload.get("origin") == {"kind": "subagent_completion"})
        assert batches[0].seq < completion.seq
        assert batches[0].turn_id == next(event.turn_id for event in events if event.event_type == "turn.started")
        replayed, _ = SessionStore(tmp_path / "replay")._replay(events, recorder.journal)
        for history in (
            target.history,
            llm.calls[1]["messages"],
            [Message.from_dict(message) for message in replayed],
        ):
            results = _tool_results(history)
            assert len(results) == 1
            assert results[0].content == "ORIGINAL SUCCESSFUL OUTPUT"
            assert results[0].execution_id == "original-execution"
            assert results[0].is_error is False
            if boundary != "archive":
                assert _find_message_index(
                    history, lambda message: bool(_tool_results([message]))
                ) < _find_message_index(history, _is_completion_message)
            else:
                assert not _completion_messages(history)
    finally:
        release.set()
        await asyncio.gather(running, return_exceptions=True)
        await manager.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", ["idle", "rebuilt-idle", "new-prompt"])
async def test_ambiguous_assistant_commit_is_paired_before_completion_acknowledgment(
    base_agent, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, boundary: str
) -> None:
    call = ToolCall(id="ambiguous-call", name="read", input={}, execution_id="ambiguous-execution")
    recorder = _attach_session_recorder(base_agent, tmp_path)
    entered, release = asyncio.Event(), asyncio.Event()
    llm = _RecordingLLM(
        [
            _GatedStream(_tool_call_message(call), entered=entered, release=release),
            _GatedStream(_text_message("next request")),
        ]
    )
    _configure_agent(base_agent, llm)
    base_agent.process_tool_calls = AsyncMock()
    original_write = recorder.journal._write_event_locked
    failed = False

    def ambiguous_assistant(event) -> None:
        nonlocal failed
        original_write(event)
        if event.event_type == "assistant.message" and not failed:
            failed = True
            raise SessionJournalError("assistant commit acknowledgment lost")

    monkeypatch.setattr(recorder.journal, "_write_event_locked", ambiguous_assistant)

    async def consume() -> None:
        async for _ in base_agent.process_message_stream("original prompt"):
            pass

    running = asyncio.create_task(consume())
    manager = base_agent.get_background_jobs()
    try:
        await asyncio.wait_for(entered.wait(), timeout=3)
        await _complete_background_job(base_agent)
        release.set()
        with pytest.raises(SessionJournalError, match="assistant commit acknowledgment lost"):
            await asyncio.wait_for(running, timeout=3)
        assert recorder.current_turn_id is None
        assert manager.pending_count == 1
        base_agent.process_tool_calls.assert_not_awaited()
        target = base_agent
        if boundary == "rebuilt-idle":
            target = BaseAgent(context=base_agent.context)
            replayed, _ = SessionStore(tmp_path / "restore")._replay(recorder.journal.read_events(), recorder.journal)
            target.restore_message_history(replayed)
            target.session_recorder = recorder
            target.background_jobs = manager
            _configure_agent(target, llm)
        if boundary != "new-prompt":
            assert await manager.deliver(target) == 1
        stream = (
            target.process_message_stream("next prompt")
            if boundary == "new-prompt"
            else target.continue_from_history_stream()
        )
        async with aclosing(stream):
            async for _ in stream:
                pass
        events = recorder.journal.read_events()
        assistants = [event for event in events if event.event_type == "assistant.message"]
        assert len(assistants) == 2
        batches = [event for event in events if event.event_type == "tool.results"]
        assert len(batches) == 1
        completion = next(event for event in events if event.payload.get("origin") == {"kind": "subagent_completion"})
        assert assistants[0].seq < batches[0].seq < completion.seq
        assert batches[0].turn_id == assistants[0].turn_id
        replayed, _ = SessionStore(tmp_path / "final")._replay(events, recorder.journal)
        for history in (target.history, llm.calls[1]["messages"], [Message.from_dict(item) for item in replayed]):
            calls = [item for message in history for item in message.tool_calls or []]
            assert [item.execution_id for item in calls] == ["ambiguous-execution"]
            results = _tool_results(history)
            assert len(results) == 1
            assert results[0].execution_id == "ambiguous-execution"
            assert results[0].is_error is True
            assert "interrupted" in str(results[0].content).lower()
            assert _find_message_index(history, lambda message: bool(_tool_results([message]))) < _find_message_index(
                history, _is_completion_message
            )
    finally:
        release.set()
        await asyncio.gather(running, return_exceptions=True)
        await manager.close()


@pytest.mark.asyncio
async def test_failed_interruption_write_on_generator_close_still_closes_turn_and_retries_before_next_prompt(
    base_agent, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    call = ToolCall(id="closed-call", name="read", input={}, execution_id="closed-execution")
    recorder = _attach_session_recorder(base_agent, tmp_path)
    llm = _RecordingLLM([_GatedStream(_tool_call_message(call)), _GatedStream(_text_message("new turn"))])
    _configure_agent(base_agent, llm)
    base_agent.process_tool_calls = AsyncMock()
    original_write = recorder.journal._write_event_locked

    def fail_results(event) -> None:
        if event.event_type == "tool.results":
            raise SessionJournalError("interruption result disk unavailable")
        original_write(event)

    stream = base_agent.process_message_stream("interrupted prompt")
    try:
        first = await stream.__anext__()
        assert first["type"] == "response"
        monkeypatch.setattr(recorder.journal, "_write_event_locked", fail_results)
        await stream.aclose()
        assert recorder.current_turn_id is None
        assert _terminal_turn_events(recorder)[-1].event_type == "turn.cancelled"
        base_agent.process_tool_calls.assert_not_awaited()
        monkeypatch.setattr(recorder.journal, "_write_event_locked", original_write)
        async for _ in base_agent.process_message_stream("next prompt"):
            pass
        assert llm.stream.await_count == 2
        events = recorder.journal.read_events()
        starts = [event for event in events if event.event_type == "turn.started"]
        assert len(starts) == 2
        assert starts[1].payload["message"]["content"] == [{"type": "text", "text": "next prompt"}]
        batches = [event for event in events if event.event_type == "tool.results"]
        assert len(batches) == 1
        assert batches[0].turn_id == starts[0].turn_id
        assert batches[0].seq < starts[1].seq
        assert len(_tool_results(llm.calls[1]["messages"])) == 1
    finally:
        await stream.aclose()


@pytest.mark.asyncio
async def test_second_cancellation_during_interruption_commit_cannot_skip_turn_terminal(
    base_agent, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    call = ToolCall(id="cancelled-call", name="read", input={}, execution_id="cancelled-execution")
    recorder = _attach_session_recorder(base_agent, tmp_path)
    llm = _RecordingLLM([_GatedStream(_tool_call_message(call)), _GatedStream(_text_message("new turn"))])
    _configure_agent(base_agent, llm)
    base_agent.process_tool_calls = AsyncMock()
    chunk_seen, commit_entered = asyncio.Event(), asyncio.Event()
    release_commit = threading.Event()
    loop = asyncio.get_running_loop()
    original_write = recorder.journal._write_event_locked

    def block_results(event) -> None:
        if event.event_type == "tool.results":
            loop.call_soon_threadsafe(commit_entered.set)
            assert release_commit.wait(timeout=10), "interruption commit was never released"
        original_write(event)

    monkeypatch.setattr(recorder.journal, "_write_event_locked", block_results)

    async def consume() -> None:
        async with aclosing(base_agent.process_message_stream("cancelled prompt")) as stream:
            async for _ in stream:
                chunk_seen.set()
                await asyncio.Event().wait()

    running = asyncio.create_task(consume())
    try:
        await asyncio.wait_for(chunk_seen.wait(), timeout=3)
        running.cancel()
        await asyncio.wait_for(commit_entered.wait(), timeout=3)
        running.cancel()
        release_commit.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(running, timeout=3)
        assert recorder.current_turn_id is None
        terminal = _terminal_turn_events(recorder)
        assert len(terminal) == 1
        assert terminal[0].event_type == "turn.cancelled"
        assert len(_tool_results(base_agent.history)) == 1
        async for _ in base_agent.process_message_stream("next prompt"):
            pass
        assert llm.stream.await_count == 2
    finally:
        release_commit.set()
        if not running.done():
            running.cancel()
        await asyncio.gather(running, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("append_first", [False, True])
async def test_committed_tool_result_projection_failure_retains_success_before_next_model_request(
    base_agent, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, append_first: bool
) -> None:
    call = ToolCall(id="committed-call", name="read", input={}, execution_id="committed-execution")
    recorder = _attach_session_recorder(base_agent, tmp_path)
    entered, release = asyncio.Event(), asyncio.Event()
    llm = _RecordingLLM(
        [
            _GatedStream(_tool_call_message(call), entered=entered, release=release),
            _GatedStream(_text_message("continued")),
        ]
    )
    _configure_agent(base_agent, llm)
    result = _tool_result(call, "COMMITTED SUCCESSFUL OUTPUT")
    result.execution_id = call.execution_id
    base_agent.process_tool_calls = AsyncMock(return_value=[result])
    original_append = base_agent.append_user_message

    def fail_projection(blocks) -> None:
        if any(isinstance(block, ToolResult) for block in blocks):
            if append_first:
                original_append(blocks)
            raise RuntimeError("committed result projection failed")
        original_append(blocks)

    monkeypatch.setattr(base_agent, "append_user_message", fail_projection)

    async def consume() -> None:
        async for _ in base_agent.process_message_stream("original prompt"):
            pass

    running = asyncio.create_task(consume())
    manager = base_agent.get_background_jobs()
    try:
        await asyncio.wait_for(entered.wait(), timeout=3)
        await _complete_background_job(base_agent)
        release.set()
        with pytest.raises(LLMError, match="committed result projection failed"):
            await asyncio.wait_for(running, timeout=3)
        assert len(llm.calls) == 1
        assert manager.pending_count == 1
        assert recorder.current_turn_id is None
        base_agent.process_tool_calls.assert_awaited_once()
        replacement = BaseAgent(context=base_agent.context)
        replacement.restore_message_history([message.to_dict() for message in base_agent.history])
        replacement.session_recorder = recorder
        replacement.background_jobs = manager
        _configure_agent(replacement, llm)
        assert await manager.deliver(replacement) == 1
        async for _ in replacement.continue_from_history_stream():
            pass
        events = recorder.journal.read_events()
        batches = [event for event in events if event.event_type == "tool.results"]
        assert len(batches) == 1
        completion = next(event for event in events if event.payload.get("origin") == {"kind": "subagent_completion"})
        assert batches[0].seq < completion.seq
        replayed, _ = SessionStore(tmp_path / "replay")._replay(events, recorder.journal)
        for history in (replacement.history, llm.calls[1]["messages"], [Message.from_dict(item) for item in replayed]):
            results = _tool_results(history)
            assert len(results) == 1
            assert results[0].content == "COMMITTED SUCCESSFUL OUTPUT"
            assert results[0].execution_id == "committed-execution"
            assert results[0].is_error is False
    finally:
        release.set()
        await asyncio.gather(running, return_exceptions=True)
        await manager.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("append_first", [False, True])
async def test_failed_assistant_projection_survives_agent_rebuild_without_duplicate_tool_calls(
    base_agent, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, append_first: bool
) -> None:
    call = ToolCall(id="projected-call", name="read", input={}, execution_id="projected-execution")
    recorder = _attach_session_recorder(base_agent, tmp_path)
    entered, release = asyncio.Event(), asyncio.Event()
    llm = _RecordingLLM(
        [
            _GatedStream(_tool_call_message(call), entered=entered, release=release),
            _GatedStream(_text_message("continued")),
        ]
    )
    _configure_agent(base_agent, llm)
    base_agent.process_tool_calls = AsyncMock()
    original_append = base_agent.append_assistant_message

    def fail_projection(message: Message) -> None:
        if message.tool_calls:
            if append_first:
                original_append(message)
            raise RuntimeError("assistant live projection failed")
        original_append(message)

    monkeypatch.setattr(base_agent, "append_assistant_message", fail_projection)

    async def consume() -> None:
        async for _ in base_agent.process_message_stream("original prompt"):
            pass

    running = asyncio.create_task(consume())
    manager = base_agent.get_background_jobs()
    try:
        await asyncio.wait_for(entered.wait(), timeout=3)
        await _complete_background_job(base_agent)
        release.set()
        with pytest.raises(LLMError, match="assistant live projection failed"):
            await asyncio.wait_for(running, timeout=3)
        assert len(llm.calls) == 1
        assert recorder.current_turn_id is None
        assert manager.pending_count == 1
        base_agent.process_tool_calls.assert_not_awaited()
        replacement = BaseAgent(context=base_agent.context)
        replacement.restore_message_history([message.to_dict() for message in base_agent.history])
        replacement.session_recorder = recorder
        replacement.background_jobs = manager
        _configure_agent(replacement, llm)
        assert await manager.deliver(replacement) == 1
        async for _ in replacement.continue_from_history_stream():
            pass
        events = recorder.journal.read_events()
        assert len([event for event in events if event.event_type == "tool.results"]) == 1
        replayed, _ = SessionStore(tmp_path / "replay")._replay(events, recorder.journal)
        for history in (
            replacement.history,
            llm.calls[1]["messages"],
            [Message.from_dict(message) for message in replayed],
        ):
            calls = [item for message in history for item in message.tool_calls or []]
            assert [item.execution_id for item in calls] == ["projected-execution"]
            results = _tool_results(history)
            assert len(results) == 1
            assert results[0].execution_id == "projected-execution"
            assert results[0].is_error is True
    finally:
        release.set()
        await asyncio.gather(running, return_exceptions=True)
        await manager.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("commit_first", [False, True])
async def test_failed_terminal_write_is_retried_before_processing_the_next_prompt(
    base_agent, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, commit_first: bool
) -> None:
    recorder = _attach_session_recorder(base_agent, tmp_path)
    llm = _RecordingLLM([_GatedStream(_text_message("first response")), _GatedStream(_text_message("second response"))])
    _configure_agent(base_agent, llm)
    original_write = recorder.journal._write_event_locked
    failed = False

    def fail_terminal(event) -> None:
        nonlocal failed
        if event.event_type in TERMINAL_TURN_EVENTS and not failed:
            failed = True
            if commit_first:
                original_write(event)
            raise SessionJournalError("terminal commit acknowledgment lost")
        original_write(event)

    monkeypatch.setattr(recorder.journal, "_write_event_locked", fail_terminal)
    with pytest.raises(SessionJournalError, match="terminal commit acknowledgment lost"):
        async for _ in base_agent.process_message_stream("first prompt"):
            pass
    async for _ in base_agent.process_message_stream("second prompt"):
        pass
    assert recorder.current_turn_id is None
    assert llm.stream.await_count == 2
    events = recorder.journal.read_events()
    starts = [event for event in events if event.event_type == "turn.started"]
    terminals = [event for event in events if event.event_type in TERMINAL_TURN_EVENTS]
    assert len(starts) == len(terminals) == 2
    assert [event.turn_id for event in terminals] == [event.turn_id for event in starts]
    assert terminals[0].seq < starts[1].seq
    assert starts[1].payload["message"]["content"] == [{"type": "text", "text": "second prompt"}]
