"""Offline stream setup, cleanup, output accounting, and retry deadlines."""

import asyncio
import json
import time
from collections.abc import AsyncIterator
from contextlib import suppress
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
import pytest_asyncio
from openai import AsyncOpenAI

from kolega_code.agent import baseagent as agent_module
from kolega_code.agent.baseagent import BaseAgent
from kolega_code.llm.exceptions import LLMRateLimitError, LLMStreamStalled, LLMTimeout
from kolega_code.llm.ledger import current_llm_call_origin
from kolega_code.llm.models import Message, MessageChunk, TextBlock, ToolDefinition
from kolega_code.llm.providers.responses_common import ResponsesStreamWrapper
from kolega_code.permissions import PermissionMode
from kolega_code.tools import Tool, ToolRegistry

from ..compaction_helpers import FakeLLM


@pytest_asyncio.fixture(autouse=True, params=["lazy", "eager"])
async def stream_task_factory(request: pytest.FixtureRequest) -> AsyncIterator[None]:
    factory = getattr(asyncio, "eager_task_factory", None) if request.param == "eager" else None
    if request.param == "eager" and factory is None:
        pytest.skip("Eager tasks require Python 3.12+")
    loop = asyncio.get_running_loop()
    original_factory = loop.get_task_factory()
    loop.set_task_factory(factory)
    try:
        yield
    finally:
        loop.set_task_factory(original_factory)


class _Body(httpx.AsyncByteStream):
    def __init__(self, events: tuple[dict[str, Any], ...] = (), *, resist_cancel: bool = False) -> None:
        self.events = events
        self.resist_cancel = resist_cancel
        self.reading = asyncio.Event()
        self.release = asyncio.Event()
        self.closed = asyncio.Event()

    async def __aiter__(self):
        for event in self.events:
            yield f"data: {json.dumps(event)}\n\n".encode()
        self.reading.set()
        try:
            await self.release.wait()
        except asyncio.CancelledError:
            if not self.resist_cancel:
                raise
            await self.release.wait()

    async def aclose(self) -> None:
        self.closed.set()


class _FiniteEventStream:
    def __init__(self, events: list[Any]) -> None:
        self._events = list(events)

    def __aiter__(self) -> AsyncIterator[Any]:
        async def generate() -> AsyncIterator[Any]:
            for event in self._events:
                yield event

        return generate()


def _client(handler: Any, *, max_retries: int = 0) -> AsyncOpenAI:
    return AsyncOpenAI(
        api_key="offline-test",
        base_url="https://offline.invalid/v1",
        max_retries=max_retries,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )


def _response(body: _Body) -> httpx.Response:
    return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=body)


async def _wrapper(client: AsyncOpenAI) -> ResponsesStreamWrapper:
    raw = await client.responses.create(model="offline-test", input="hello", stream=True)
    return ResponsesStreamWrapper(raw)


@pytest.fixture
def guarded(base_agent: BaseAgent, monkeypatch: pytest.MonkeyPatch) -> BaseAgent:
    base_agent._stream_first_chunk_timeout = 0.08
    base_agent._stream_idle_timeout = 0.08
    base_agent._stream_heartbeat_seconds = 0.01
    monkeypatch.setattr(agent_module, "_STREAM_CLEANUP_TIMEOUT", 0.01)
    monkeypatch.setattr(base_agent.emitter, "llm_status", AsyncMock())
    return base_agent


async def _drain(agent: BaseAgent, setup: Any) -> None:
    async with agent._open_stream_with_stall_guard(setup) as (stream, state):
        async for _ in agent._iter_stream_with_stall_guard(stream, state=state):
            pass


@pytest.mark.asyncio
@pytest.mark.parametrize("exit_kind", ["break", "exception", "cancel"])
async def test_real_sdk_stream_closes_on_early_exit(exit_kind: str) -> None:
    body = _Body(({"type": "response.output_text.delta", "delta": "hello"},))
    async with _client(lambda _: _response(body)) as client:
        wrapper = await _wrapper(client)

        async def consume() -> None:
            async with wrapper:
                async for chunk in wrapper:
                    assert chunk.text == "hello"
                    if exit_kind == "break":
                        break
                    if exit_kind == "exception":
                        raise ValueError("consumer failed")

        task = asyncio.create_task(consume())
        if exit_kind == "cancel":
            await asyncio.wait_for(body.reading.wait(), 1)
            task.cancel()
        with suppress(ValueError, asyncio.CancelledError):
            await task
        assert body.closed.is_set()
        assert wrapper._closed


@pytest.mark.asyncio
async def test_delayed_sdk_headers_have_first_event_deadline_and_heartbeats(guarded: BaseAgent) -> None:
    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def handler(_: httpx.Request) -> httpx.Response:
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()
        raise AssertionError("unreachable")

    async with _client(handler) as client:
        with pytest.raises(LLMStreamStalled) as raised:
            await _drain(guarded, lambda: _wrapper(client))

    assert started.is_set()
    assert cancelled.is_set()
    assert raised.value.stage == "first_chunk"
    assert raised.value.zero_output
    heartbeats = cast(AsyncMock, guarded.emitter.llm_status).await_args_list
    assert heartbeats
    assert all(call.kwargs["stage"] == "first_chunk" for call in heartbeats)


@pytest.mark.asyncio
async def test_sdk_retry_sleep_is_inside_setup_deadline(guarded: BaseAgent) -> None:
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(429, json={"error": {"message": "slow down"}}, headers={"retry-after": "5"})

    async with _client(handler, max_retries=2) as client:
        with pytest.raises(LLMStreamStalled):
            await _drain(guarded, lambda: _wrapper(client))
    assert len(requests) == 1
    assert cast(AsyncMock, guarded.emitter.llm_status).await_count > 0


@pytest.mark.asyncio
async def test_create_enter_and_first_read_share_one_deadline(guarded: BaseAgent) -> None:
    guarded._stream_first_chunk_timeout = 0.25
    body = _Body()

    async def handler(_: httpx.Request) -> httpx.Response:
        await asyncio.sleep(0.1)
        return _response(body)

    wrapper: ResponsesStreamWrapper | None = None

    class DelayedEnter:
        async def __aenter__(self) -> ResponsesStreamWrapper:
            await asyncio.sleep(0.1)
            assert wrapper is not None
            return await wrapper.__aenter__()

        async def __aexit__(self, *args: Any) -> bool:
            assert wrapper is not None
            return await wrapper.__aexit__(*args)

    async with _client(handler) as client:

        async def setup() -> DelayedEnter:
            nonlocal wrapper
            wrapper = await _wrapper(client)
            return DelayedEnter()

        started = time.monotonic()
        with pytest.raises(LLMStreamStalled) as raised:
            await _drain(guarded, setup)
        elapsed = time.monotonic() - started

    assert raised.value.stage == "first_chunk"
    assert 0.24 <= elapsed < 0.4
    assert body.closed.is_set()


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["create", "enter"])
@pytest.mark.parametrize("cancel_turn", [False, True])
async def test_cancellation_resistant_setup_is_closed_when_it_eventually_returns(
    guarded: BaseAgent, phase: str, cancel_turn: bool
) -> None:
    body = _Body()
    entered = asyncio.Event()
    cancelled = asyncio.Event()
    release = asyncio.Event()

    async def block() -> None:
        entered.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            cancelled.set()
            await release.wait()

    async def handler(_: httpx.Request) -> httpx.Response:
        if phase == "create":
            await block()
        return _response(body)

    wrapper: ResponsesStreamWrapper | None = None

    class DelayedEnter:
        async def __aenter__(self) -> ResponsesStreamWrapper:
            await block()
            assert wrapper is not None
            return await wrapper.__aenter__()

        async def __aexit__(self, *args: Any) -> bool:
            assert wrapper is not None
            return await wrapper.__aexit__(*args)

    async with _client(handler) as client:

        async def setup() -> Any:
            nonlocal wrapper
            wrapper = await _wrapper(client)
            return DelayedEnter() if phase == "enter" else wrapper

        task = asyncio.create_task(_drain(guarded, setup))
        try:
            await asyncio.wait_for(entered.wait(), 1)
            if cancel_turn:
                task.cancel()
            done, _ = await asyncio.wait({task}, timeout=0.3)
            assert task in done, "source cancellation must not hold the turn open"
            with pytest.raises(asyncio.CancelledError if cancel_turn else LLMStreamStalled):
                task.result()
            assert cancelled.is_set()
            assert not body.closed.is_set()
        finally:
            release.set()
            await asyncio.wait_for(body.closed.wait(), 1)
            with suppress(asyncio.CancelledError, LLMStreamStalled):
                await task
        assert wrapper is not None and wrapper._closed


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_turn", [False, True])
async def test_cancellation_resistant_read_does_not_hold_turn_and_is_eventually_closed(
    guarded: BaseAgent, cancel_turn: bool
) -> None:
    body = _Body(resist_cancel=True)
    async with _client(lambda _: _response(body)) as client:
        task = asyncio.create_task(_drain(guarded, lambda: _wrapper(client)))
        try:
            await asyncio.wait_for(body.reading.wait(), 1)
            if cancel_turn:
                task.cancel()
            done, _ = await asyncio.wait({task}, timeout=0.3)
            assert task in done, "a read swallowing cancellation must not defeat the deadline"
            with pytest.raises(asyncio.CancelledError if cancel_turn else LLMStreamStalled):
                task.result()
        finally:
            body.release.set()
            await asyncio.wait_for(body.closed.wait(), 1)
            with suppress(asyncio.CancelledError, LLMStreamStalled):
                await task


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "event_type", ["response.function_call_arguments.delta", "response.custom_tool_call_input.delta"]
)
async def test_tool_only_stream_is_partial_output_not_zero_output(guarded: BaseAgent, event_type: str) -> None:
    payload = "x" * 35_000
    body = _Body(({"type": event_type, "item_id": "tool_1", "delta": payload},))
    async with _client(lambda _: _response(body)) as client:
        with pytest.raises(LLMStreamStalled) as raised:
            await _drain(guarded, lambda: _wrapper(client))
    assert raised.value.stage == "stream"
    assert raised.value.bytes_received == len(payload)
    assert raised.value.partial_output
    assert not raised.value.zero_output
    assert body.closed.is_set()


@pytest.mark.asyncio
async def test_tool_done_does_not_double_count_argument_deltas(guarded: BaseAgent) -> None:
    payload = '{"path":"hello"}'
    wrapper = ResponsesStreamWrapper(None)
    chunks = [
        wrapper._handle_event(
            SimpleNamespace(type="response.function_call_arguments.delta", item_id="f", delta=payload)
        ),
        wrapper._handle_event(
            SimpleNamespace(type="response.function_call_arguments.done", item_id="f", arguments=payload)
        ),
    ]
    assert chunks[0].type == "tool_use_delta"
    assert chunks[0].tool_call_delta == {"input_delta": payload}
    assert sum(guarded._stream_event_chars(chunk) for chunk in chunks) == len(payload)
    assert guarded._stream_event_chars(MessageChunk(type="text", text="a")) == 1
    assert guarded._stream_event_chars(MessageChunk(type="thinking", thinking="bc")) == 2


@pytest.mark.asyncio
async def test_agent_retries_incomplete_responses_stream_without_dispatching_partial_tool(
    guarded: BaseAgent, monkeypatch: pytest.MonkeyPatch
) -> None:
    guarded.system_prompt = Message(role="system", content=[TextBlock(text="offline test")])
    guarded.permission_mode = PermissionMode.AUTO
    handler_calls: list[str] = []

    async def exec_command(command: str) -> str:
        handler_calls.append(command)
        return "unexpected"

    registry = ToolRegistry(
        [
            Tool(
                name="exec_command",
                definition=ToolDefinition(name="exec_command", description="", parameters=[]),
                handler=exec_command,
            )
        ]
    )
    registry_call = AsyncMock(wraps=registry.call)
    monkeypatch.setattr(registry, "call", registry_call)
    guarded.tool_collection = cast(
        Any,
        SimpleNamespace(
            get_tool_list=MagicMock(return_value=registry.definitions()),
            registry=lambda: registry,
        ),
    )

    first = ResponsesStreamWrapper(
        _FiniteEventStream(
            [
                SimpleNamespace(
                    type="response.output_item.added",
                    item=SimpleNamespace(
                        type="function_call",
                        id="fc_1",
                        call_id="call_1",
                        name="exec_command",
                    ),
                ),
                SimpleNamespace(
                    type="response.function_call_arguments.delta",
                    item_id="fc_1",
                    delta='{"command":',
                ),
            ]
        ),
        provider_name="deepseek",
    )
    second = ResponsesStreamWrapper(
        _FiniteEventStream(
            [
                SimpleNamespace(type="response.output_text.delta", delta="recovered"),
                SimpleNamespace(
                    type="response.completed",
                    response=SimpleNamespace(
                        output=[],
                        usage=None,
                        status="completed",
                        incomplete_details=None,
                    ),
                ),
            ]
        ),
        provider_name="deepseek",
    )
    llm = FakeLLM(token_script=[100])
    llm.stream = AsyncMock(side_effect=[first, second])
    guarded.llm = cast(Any, llm)
    monkeypatch.setattr(agent_module.random, "uniform", lambda *_: 0)

    chunks = [chunk async for chunk in guarded._agent_loop_stream()]

    assert any(chunk.get("content") == "recovered" for chunk in chunks)
    assert llm.stream.await_count == 2
    assert guarded._consecutive_llm_retries == 0
    registry_call.assert_not_awaited()
    assert handler_calls == []
    error_events = [
        call.args[0]
        for call in cast(AsyncMock, guarded.emitter.connection_manager.broadcast_event).await_args_list
        if call.args[0].event_type == "llm_error"
    ]
    assert any(event.content["error_type"] == "LLMIncompleteResponseError" for event in error_events)


@pytest.mark.asyncio
async def test_agent_continues_after_visible_mixed_output_without_releasing_partial_tool(
    guarded: BaseAgent, monkeypatch: pytest.MonkeyPatch
) -> None:
    guarded.system_prompt = Message(role="system", content=[TextBlock(text="offline test")])
    guarded.permission_mode = PermissionMode.AUTO
    guarded.tool_collection = cast(
        Any,
        SimpleNamespace(
            get_tool_list=MagicMock(return_value=[]),
            registry=MagicMock(),
        ),
    )
    partial_thinking = "t" * (guarded.STREAM_FLUSH_CHARS + 1)
    partial_response = "x" * (guarded.STREAM_FLUSH_CHARS + 1)
    first = ResponsesStreamWrapper(
        _FiniteEventStream(
            [
                SimpleNamespace(type="response.reasoning_text.delta", delta=partial_thinking),
                SimpleNamespace(type="response.output_text.delta", delta=partial_response),
                SimpleNamespace(
                    type="response.output_item.added",
                    item=SimpleNamespace(
                        type="function_call",
                        id="fc_partial",
                        call_id="call_partial",
                        name="exec_command",
                    ),
                ),
                SimpleNamespace(
                    type="response.function_call_arguments.delta",
                    item_id="fc_partial",
                    delta='{"command":"echo unfinished',
                ),
            ]
        ),
        provider_name="deepseek",
    )
    second = ResponsesStreamWrapper(
        _FiniteEventStream(
            [
                SimpleNamespace(type="response.output_text.delta", delta="recovered"),
                SimpleNamespace(
                    type="response.completed",
                    response=SimpleNamespace(
                        output=[],
                        usage=None,
                        status="completed",
                        incomplete_details=None,
                    ),
                ),
            ]
        ),
        provider_name="deepseek",
    )
    llm = FakeLLM(token_script=[100])
    llm.stream = AsyncMock(side_effect=[first, second])
    guarded.llm = cast(Any, llm)
    monkeypatch.setattr(agent_module.random, "uniform", lambda *_: 0)

    chunks = [chunk async for chunk in guarded._agent_loop_stream()]

    response_chunk = next(chunk for chunk in chunks if chunk.get("content") == partial_response)
    assert response_chunk["complete"] is False
    assert any(
        chunk.get("type") == "response"
        and chunk.get("uuid") == response_chunk["uuid"]
        and chunk.get("complete") is True
        for chunk in chunks
    )
    assert any(chunk.get("content") == "recovered" for chunk in chunks)
    assert llm.stream.await_count == 2
    partial_message, continuation_nudge, recovered_message = guarded.history[-3:]
    assert [partial_message.role, continuation_nudge.role, recovered_message.role] == [
        "assistant",
        "user",
        "assistant",
    ]
    assert partial_message.get_text_content() == partial_response
    assert partial_message.stop_reason == "interrupted"
    assert partial_message.tool_calls == []
    assert partial_message.usage_metadata["stream_incomplete"] is True
    assert "transport failure" in continuation_nudge.get_text_content()
    assert recovered_message.get_text_content() == "recovered"
    chat_events = [
        call.args[0]
        for call in cast(AsyncMock, guarded.emitter.connection_manager.broadcast_event).await_args_list
        if call.args[0].event_type == "chat_message"
    ]
    assert all(event.content.get("message_type") != "tool_call" for event in chat_events)
    status_calls = cast(AsyncMock, guarded.emitter.llm_status).await_args_list
    assert any(call.kwargs.get("continuation") is True for call in status_calls)


@pytest.mark.asyncio
async def test_incomplete_responses_stream_does_not_announce_partial_write(
    guarded: BaseAgent, monkeypatch: pytest.MonkeyPatch
) -> None:
    guarded.system_prompt = Message(role="system", content=[TextBlock(text="offline test")])
    guarded.permission_mode = PermissionMode.AUTO
    guarded.tool_collection = cast(
        Any,
        SimpleNamespace(
            get_tool_list=MagicMock(return_value=[]),
            registry=MagicMock(),
        ),
    )
    first = ResponsesStreamWrapper(
        _FiniteEventStream(
            [
                SimpleNamespace(
                    type="response.output_item.added",
                    item=SimpleNamespace(
                        type="function_call",
                        id="fc_write",
                        call_id="call_write",
                        name="write",
                    ),
                ),
                SimpleNamespace(
                    type="response.function_call_arguments.delta",
                    item_id="fc_write",
                    delta='{"path":"partial.py","content":"unfinished',
                ),
            ]
        ),
        provider_name="deepseek",
    )
    second = ResponsesStreamWrapper(
        _FiniteEventStream(
            [
                SimpleNamespace(type="response.output_text.delta", delta="recovered"),
                SimpleNamespace(
                    type="response.completed",
                    response=SimpleNamespace(
                        output=[],
                        usage=None,
                        status="completed",
                        incomplete_details=None,
                    ),
                ),
            ]
        ),
        provider_name="deepseek",
    )
    llm = FakeLLM(token_script=[100])
    llm.stream = AsyncMock(side_effect=[first, second])
    guarded.llm = cast(Any, llm)
    monkeypatch.setattr(agent_module.random, "uniform", lambda *_: 0)

    chunks = [chunk async for chunk in guarded._agent_loop_stream()]

    assert any(chunk.get("content") == "recovered" for chunk in chunks)
    chat_events = [
        call.args[0]
        for call in cast(AsyncMock, guarded.emitter.connection_manager.broadcast_event).await_args_list
        if call.args[0].event_type == "chat_message"
    ]
    assert all(event.content.get("message_type") != "tool_call" for event in chat_events)


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["create", "enter", "read", "active"])
async def test_remaining_retry_budget_bounds_subsequent_attempt(guarded: BaseAgent, phase: str) -> None:
    guarded._stream_first_chunk_timeout = 5
    guarded._stream_idle_timeout = 5
    guarded._llm_retry_budget_seconds = 0.04
    guarded._llm_retry_budget_started = time.monotonic()
    closed = asyncio.Event()

    class Stream:
        async def __aenter__(self) -> Any:
            if phase == "enter":
                await asyncio.Event().wait()
            return self

        async def __aexit__(self, *args: Any) -> bool:
            closed.set()
            return False

        def __aiter__(self) -> Any:
            return self

        async def __anext__(self) -> MessageChunk:
            if phase == "active":
                await asyncio.sleep(0.005)
                return MessageChunk(type="text", text="x")
            await asyncio.Event().wait()
            raise StopAsyncIteration

    async def setup() -> Stream:
        if phase == "create":
            await asyncio.Event().wait()
        return Stream()

    started = time.monotonic()
    with pytest.raises(LLMTimeout, match="retry wall-clock budget"):
        await _drain(guarded, setup)
    assert time.monotonic() - started < 0.3
    if phase != "create":
        assert closed.is_set()


@pytest.mark.asyncio
async def test_initial_attempt_does_not_start_retry_budget(guarded: BaseAgent) -> None:
    guarded._llm_retry_budget_seconds = 0.001
    body = _Body()
    body.release.set()

    async def handler(_: httpx.Request) -> httpx.Response:
        await asyncio.sleep(0.03)
        return _response(body)

    async with _client(handler) as client:
        await _drain(guarded, lambda: _wrapper(client))
    assert guarded._llm_retry_budget_started is None


@pytest.mark.asyncio
async def test_expired_retry_budget_does_not_start_another_request(guarded: BaseAgent) -> None:
    guarded._llm_retry_budget_started = time.monotonic() - 1
    guarded._llm_retry_budget_seconds = 0.1
    setup = AsyncMock()
    with pytest.raises(LLMTimeout, match="retry wall-clock budget"):
        await _drain(guarded, setup)
    setup.assert_not_called()


@pytest.mark.asyncio
async def test_expired_retry_budget_cannot_send_warmed_sdk_request(guarded: BaseAgent) -> None:
    body = _Body()
    body.release.set()
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return _response(body)

    async with _client(handler) as client:
        async with await _wrapper(client) as stream:
            async for _ in stream:
                pass
        assert len(requests) == 1
        guarded._llm_retry_budget_started = time.monotonic() - 1
        guarded._llm_retry_budget_seconds = 0.1
        with pytest.raises(LLMTimeout, match="retry wall-clock budget"):
            await _drain(guarded, lambda: _wrapper(client))
        assert len(requests) == 1


@pytest.mark.asyncio
async def test_late_setup_result_cannot_escape_retry_deadline(guarded: BaseAgent) -> None:
    guarded._llm_retry_budget_seconds = 0.01
    stream = MagicMock()
    stream.__aenter__ = AsyncMock(return_value=stream)
    stream.__aexit__ = AsyncMock(return_value=False)

    async def setup() -> Any:
        # Armed here, not before _drain: the guard's entry budget check raises the
        # same LLMTimeout, and any stall before setup starts would skip this path.
        guarded._llm_retry_budget_started = time.monotonic()
        time.sleep(0.03)
        return stream

    with pytest.raises(LLMTimeout, match="retry wall-clock budget"):
        await _drain(guarded, setup)
    stream.__aexit__.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("retry_after", [None, 45.0])
async def test_retry_backoff_cannot_exceed_remaining_budget(
    guarded: BaseAgent, monkeypatch: pytest.MonkeyPatch, retry_after: float | None
) -> None:
    guarded._llm_retry_budget_seconds = 0.03
    monkeypatch.setattr(agent_module.random, "uniform", lambda *_: 10.0)
    error = LLMRateLimitError("offline rate limit")
    if retry_after is not None:
        setattr(error, "retry_after", retry_after)
    started = time.monotonic()
    with pytest.raises(LLMRateLimitError):
        await guarded.handle_llm_error(error)
    assert time.monotonic() - started < 0.3
    assert guarded._llm_retry_budget_started is not None
    calls = cast(AsyncMock, guarded.emitter.llm_status).await_args_list
    retry_status = next(call for call in calls if call.kwargs.get("retry"))
    assert retry_status.kwargs["retry_in_s"] <= 0.03
    assert calls[-1].args[0] == "error"
    assert guarded._consecutive_llm_retries == 1


@pytest.mark.asyncio
async def test_entry_error_is_not_masked_by_cleanup_failure(guarded: BaseAgent) -> None:
    error = LLMRateLimitError("retry this entry error")

    class Stream:
        async def __aenter__(self) -> Any:
            raise error

        async def __aexit__(self, *args: Any) -> bool:
            raise RuntimeError("cleanup failed")

    setup = AsyncMock(return_value=Stream())
    with pytest.raises(LLMRateLimitError) as raised:
        await _drain(guarded, setup)
    assert raised.value is error


@pytest.mark.asyncio
@pytest.mark.parametrize("stall", [False, True])
async def test_agent_loop_guards_real_sdk_setup_and_preserves_origin_and_retry_policy(
    guarded: BaseAgent, monkeypatch: pytest.MonkeyPatch, stall: bool
) -> None:
    if not stall:
        # Real SDK event-model initialization can exceed the fixture's tiny
        # stall deadline even though the mocked transport responds immediately.
        guarded._stream_first_chunk_timeout = 1.0
        guarded._stream_idle_timeout = 1.0
    guarded.system_prompt = Message(role="system", content=[TextBlock(text="offline test")])
    guarded.tool_collection = MagicMock()
    guarded.tool_collection.get_tool_list = MagicMock(return_value=[])
    llm = FakeLLM(token_script=[100])
    guarded.llm = cast(Any, llm)
    monkeypatch.setattr(agent_module.random, "uniform", lambda *_: 0)
    requests = []
    body = _Body(
        (
            {"type": "response.output_text.delta", "delta": "done"},
            {
                "type": "response.completed",
                "sequence_number": 2,
                "response": {
                    "id": "resp_mock",
                    "created_at": 0,
                    "model": "offline-test",
                    "object": "response",
                    "output": [],
                    "parallel_tool_calls": False,
                    "tool_choice": "auto",
                    "tools": [],
                    "status": "completed",
                },
            },
        )
    )
    body.release.set()

    async def handler(request: httpx.Request) -> httpx.Response:
        assert current_llm_call_origin() is not None
        requests.append(request)
        if stall:
            await asyncio.Event().wait()
        return _response(body)

    async with _client(handler) as client:

        async def stream(*args: Any, **kwargs: Any) -> ResponsesStreamWrapper:
            return await _wrapper(client)

        llm.stream = AsyncMock(side_effect=stream)
        if stall:
            with pytest.raises(LLMStreamStalled):
                async for _ in guarded._agent_loop_stream():
                    pass
            assert len(requests) == 2
            assert guarded._zero_output_stalls == 2
        else:
            chunks = [chunk async for chunk in guarded._agent_loop_stream()]
            assert any(chunk.get("content") == "done" for chunk in chunks)
            assert body.closed.is_set()
            assert guarded._consecutive_llm_retries == 0
            assert guarded._llm_retry_budget_started is None
    assert current_llm_call_origin() is None
