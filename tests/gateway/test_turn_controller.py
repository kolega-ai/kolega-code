from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextvars import ContextVar
from dataclasses import replace
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest
import pytest_asyncio

from kolega_code.agent.prompts import build_implement_plan_prompt
from kolega_code.gateway.adapters.base import InboundMessage
from kolega_code.gateway.daemon import ERROR_REPLY
from kolega_code.gateway.sessions import ChatLifecycleIncomplete
from kolega_code.session.turns import TurnController, TurnQueue

from .test_background_jobs import host, queue_reports
from .test_sessions import chat_ref, inbound, wait_for

pytestmark = pytest.mark.usefixtures("isolated_cli_env")


@pytest_asyncio.fixture(autouse=True, params=[False, True], ids=["lazy", "eager"])
async def turn_scheduling(request: pytest.FixtureRequest) -> AsyncIterator[None]:
    loop = asyncio.get_running_loop()
    original = loop.get_task_factory()
    if request.param:
        factory = getattr(asyncio, "eager_task_factory", None)
        if factory is None:
            pytest.skip("Eager task factories require Python 3.12")
        loop.set_task_factory(factory)
    try:
        yield
    finally:
        loop.set_task_factory(original)


@pytest.mark.asyncio
@pytest.mark.parametrize("interrupt", ["stop", "transport_failure"])
async def test_renderer_interruption_closes_source_in_owned_task_and_keeps_fifo_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, interrupt: str
) -> None:
    handler, adapter, _ = host(tmp_path)
    entry = await handler._ensure_session(chat_ref())
    turns = entry.runtime.turns
    assert isinstance(turns, TurnController)
    run = AsyncMock(wraps=turns.run)
    monkeypatch.setattr(turns, "run", run)
    sending, cleaning, release = (asyncio.Event() for _ in range(3))
    scope: ContextVar[str] = ContextVar("gateway-turn-scope", default="outside")
    started: list[str] = []
    closed: list[str] = []
    source_tasks: list[asyncio.Task[Any] | None] = []
    send = adapter.send_text

    async def send_text(chat_id: str, text: str, **kwargs: Any) -> str:
        if text == "first response":
            sending.set()
            if interrupt == "transport_failure":
                raise OSError("renderer transport failed")
            await asyncio.Future()
        return await send(chat_id, text, **kwargs)

    async def stream(text: str, attachments: Any = None) -> AsyncIterator[dict[str, Any]]:
        task = asyncio.current_task()
        assert task is entry.turn_task is turns.execution_task is turns.owner
        assert turns._active_run is not None
        assert scope.get() == "outside"
        token = scope.set(text)
        started.append(text)
        source_tasks.append(task)
        try:
            yield {"type": "response", "content": f"{text} response", "complete": True}
        finally:
            if text == "first":
                cleaning.set()
                await release.wait()
            assert asyncio.current_task() is task
            assert scope.get() == text
            scope.reset(token)
            closed.append(text)

    monkeypatch.setattr(adapter, "send_text", send_text)
    monkeypatch.setattr(entry.runtime, "send_message", stream)
    try:
        await handler.handle(chat_ref(), inbound("first"))
        await asyncio.wait_for(sending.wait(), 10)
        worker = handler._workers[chat_ref().key]
        task = entry.turn_task
        assert task is not None
        if interrupt == "stop":
            assert await handler._stop_turn(chat_ref())
        await asyncio.wait_for(cleaning.wait(), 10)
        await handler.handle(chat_ref(), inbound("second"))
        await handler.handle(chat_ref(), inbound("third"))
        queue = handler._queues[chat_ref().key]
        assert queue is turns.inputs
        assert queue.qsize() == 2
        assert started == ["first"] and closed == []
        assert entry.turn_task is task and not task.done()
        assert not await turns.wait_settled(timeout=0)
        assert not worker.done()
        if interrupt == "stop":
            for _ in range(3):
                assert await handler._stop_turn(chat_ref())
            assert task.cancelling() == 1
        release.set()
        assert await wait_for(lambda: closed == ["first", "second", "third"] and entry.turn_task is None)
        assert started == ["first", "second", "third"]
        assert scope.get() == "outside"
        assert all(task is not worker for task in source_tasks)
        assert handler._workers[chat_ref().key] is worker and not worker.done()
        assert turns.execution_task is turns.owner is None
        assert queue.empty()
        assert run.await_count == 3
        assert ((chat_ref().chat_id, ERROR_REPLY) in adapter.sent) == (interrupt == "transport_failure")
    finally:
        release.set()
        await handler.shutdown()


@pytest.mark.asyncio
async def test_background_wake_and_continuation_use_controller_queue_and_completion_hooks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    handler, adapter, agents = host(tmp_path)
    entry = await handler._ensure_session(chat_ref())
    turns = entry.runtime.turns
    run = AsyncMock(wraps=turns.run)
    monkeypatch.setattr(turns, "run", run)
    order: list[str] = []
    persist = handler._persist

    async def continuation() -> AsyncIterator[dict[str, Any]]:
        assert asyncio.current_task() is turns.execution_task is turns.owner
        assert turns._active_run is not None
        assert any("child report" in text for _, text in adapter.sent)
        order.append("stream")
        try:
            yield {"type": "response", "content": "continued test-key", "complete": True}
        finally:
            order.append("closed")

    def save(payload: Any) -> None:
        assert order[:2] == ["stream", "closed"]
        order.append("persist")
        persist(payload)

    monkeypatch.setattr(agents[0], "continue_from_history_stream", continuation)
    monkeypatch.setattr(handler, "_persist", save)
    try:
        queue_reports(agents[0], "child report")
        queue = handler._queues[chat_ref().key]
        assert queue is turns.inputs
        assert isinstance(queue, TurnQueue)
        assert queue.items == [None]
        assert await wait_for(lambda: order[:3] == ["stream", "closed", "persist"] and entry.turn_task is None)
        assert run.await_count == 1
        assert run.await_args is not None
        assert run.await_args.kwargs["hooks"].after
        assert "test-key" not in "\n".join(text for _, text in adapter.sent)
    finally:
        monkeypatch.setattr(handler, "_persist", persist)
        await handler.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("pending", [False, True], ids=["empty", "pending"])
async def test_new_runtime_reuses_existing_controller_fifo(tmp_path: Path, pending: bool) -> None:
    handler, _, agents = host(tmp_path)
    queue: TurnQueue[InboundMessage | None] = TurnQueue()
    handler._queues[chat_ref().key] = queue
    try:
        original = await handler._ensure_session(chat_ref())
        assert original.runtime.turns.inputs is queue
        handler._registry.remove(chat_ref())
        owner = handler._retain_close(original)
        await handler._wait_operation(owner)
        assert original.runtime.turns.closed
        assert handler._queues[chat_ref().key] is queue
        if pending:
            queue.put_nowait(inbound("already queued"))
        rebuilt = await handler._ensure_session(chat_ref())
        assert rebuilt.runtime is not original.runtime
        assert rebuilt.runtime.turns is not original.runtime.turns
        assert rebuilt.runtime.turns.inputs is queue
        await handler.handle(chat_ref(), inbound("new input"))
        expected = ["already queued", "new input"] if pending else ["new input"]
        assert await wait_for(lambda: agents[-1].messages == expected and rebuilt.turn_task is None)
        assert handler._queues[chat_ref().key] is queue and queue.empty()
        await handler._apply_model(chat_ref(), rebuilt, "anthropic/claude-opus-5")
        assert rebuilt.runtime.turns.inputs is queue
        await handler.handle(chat_ref(), inbound("after model switch"))
        assert await wait_for(lambda: agents[-1].messages == ["after model switch"] and rebuilt.turn_task is None)
    finally:
        await handler.shutdown()


@pytest.mark.asyncio
async def test_stop_after_publication_before_body_never_enters_cancelled_turn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    handler, _, agents = host(tmp_path)
    entry = await handler._ensure_session(chat_ref())
    turns = entry.runtime.turns
    track = turns.track
    published: list[asyncio.Task[Any]] = []
    run = AsyncMock(wraps=turns.run)
    monkeypatch.setattr(turns, "run", run)

    def publish(task: asyncio.Task[Any] | None = None) -> None:
        track(task)
        assert task is not None and turns.execution_task is turns.owner is task
        published.append(task)
        if len(published) == 1:
            assert turns.request_stop()

    monkeypatch.setattr(turns, "track", publish)
    try:
        await handler.handle(chat_ref(), inbound("cancel before entry"))
        await handler.handle(chat_ref(), inbound("next message"))
        assert await wait_for(lambda: agents[0].messages == ["next message"] and entry.turn_task is None)
        assert len(published) == 2 and published[0].cancelled()
        assert run.await_count == 1
        assert not turns.stop_pending and turns.owner is None
        assert not handler._workers[chat_ref().key].done()
    finally:
        await handler.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("interrupt", ["stop", "input"])
async def test_implementation_tracks_retained_lifecycle_task_until_source_closes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, interrupt: str
) -> None:
    handler, adapter, _ = host(tmp_path)
    await handler.handle(chat_ref(), inbound("/mode plan"))
    entry = await handler._ensure_session(chat_ref())
    turns = entry.runtime.turns
    run = AsyncMock(wraps=turns.run)
    monkeypatch.setattr(turns, "run", run)
    implementing, cleaning, release = (asyncio.Event() for _ in range(3))
    plan = "# Full plan\n\nImplement and verify."
    implementation_prompt = build_implement_plan_prompt(plan)
    started: list[str] = []
    closed: list[str] = []

    async def stream(text: str, attachments: Any = None) -> AsyncIterator[dict[str, Any]]:
        task = asyncio.current_task()
        assert task is turns.execution_task is turns.owner
        assert turns._active_run is not None
        started.append(text)
        try:
            if text == implementation_prompt:
                owner = handler._operations[chat_ref().key]
                assert owner.task is task
                assert entry.plan_approval_task is not task
                implementing.set()
                yield {"type": "response", "content": "Implementing.", "complete": True}
                await asyncio.Future()
            else:
                yield {"type": "response", "content": plan if text == "plan" else text, "complete": True}
        finally:
            if text == implementation_prompt:
                cleaning.set()
                await release.wait()
            assert task is asyncio.current_task()
            closed.append(text)

    monkeypatch.setattr(entry.runtime, "send_message", stream)
    try:
        await handler.handle(chat_ref(), inbound("plan"))
        assert await wait_for(lambda: bool(adapter.tokens) and entry.turn_task is None)
        token = adapter.tokens[-1]
        await handler.handle(chat_ref(), replace(inbound(""), callback_token=token, callback_option="implement_plan"))
        await asyncio.wait_for(implementing.wait(), 10)
        owner = handler._operations[chat_ref().key]
        implementation = entry.turn_task
        approval = entry.plan_approval_task
        assert implementation is owner.task and implementation is not None
        assert owner.entry is entry and not owner.cleaning and not owner.released.is_set()
        await handler.handle(chat_ref(), inbound("/model anthropic/claude-opus-5"))
        assert "A turn is running" in adapter.sent[-1][1]
        if interrupt == "stop":
            await handler.handle(chat_ref(), inbound("/stop"))
        await handler.handle(chat_ref(), inbound("follow up"))
        await handler.handle(chat_ref(), inbound("last input"))
        await asyncio.wait_for(cleaning.wait(), 10)
        assert handler._operations[chat_ref().key] is owner and not owner.released.is_set()
        assert entry.turn_task is implementation and not implementation.done()
        assert turns.execution_task is implementation
        assert started == ["plan", implementation_prompt]
        assert closed == ["plan"]
        assert not handler._workers[chat_ref().key].done()
        release.set()
        assert await wait_for(
            lambda: closed == ["plan", implementation_prompt, "follow up", "last input"] and entry.turn_task is None
        )
        assert implementation.cancelled()
        assert approval is not None and approval.cancelled()
        assert owner.released.is_set() and not handler._operations
        assert run.await_count == 4
        assert not entry.turn_lock.locked() and turns.owner is None
    finally:
        release.set()
        await handler.shutdown()


@pytest.mark.asyncio
async def test_shutdown_retains_controller_execution_and_worker_until_task_affine_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    handler, _, agents = host(tmp_path)
    entry = await handler._ensure_session(chat_ref())
    turns = entry.runtime.turns
    entered, cleaning, release = (asyncio.Event() for _ in range(3))
    closed: list[asyncio.Task[Any] | None] = []

    async def stream(text: str, attachments: Any = None) -> AsyncIterator[dict[str, Any]]:
        task = asyncio.current_task()
        assert task is turns.execution_task
        entered.set()
        try:
            await asyncio.Future()
            yield {}
        finally:
            cleaning.set()
            await release.wait()
            assert task is asyncio.current_task()
            closed.append(task)

    monkeypatch.setattr(entry.runtime, "send_message", stream)
    try:
        await handler.handle(chat_ref(), inbound("blocked"))
        await asyncio.wait_for(entered.wait(), 10)
        task = entry.turn_task
        worker = handler._workers[chat_ref().key]
        handler.begin_shutdown()
        await asyncio.wait_for(cleaning.wait(), 10)
        assert task is not None and entry.turn_task is task and not task.done()
        assert not worker.done() and not await turns.wait_settled(timeout=0)
        with pytest.raises(ChatLifecycleIncomplete):
            await handler.handle(chat_ref(), inbound("too late"))
        release.set()
        await handler.wait_closed()
        assert task.cancelled() and worker.cancelled()
        assert closed == [task]
        assert turns.closed and turns.execution_task is turns.owner is None
        assert agents[0].cleaned
        assert not handler._workers and not handler._queues
    finally:
        release.set()
        await handler.wait_closed()


@pytest.mark.asyncio
async def test_owner_change_during_renderer_await_fences_completion_hooks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    handler, adapter, _ = host(tmp_path)
    entry = await handler._ensure_session(chat_ref())
    turns = entry.runtime.turns
    sending, release, closed = (asyncio.Event() for _ in range(3))
    save = Mock(wraps=handler._persist)
    monkeypatch.setattr(handler, "_persist", save)
    send = adapter.send_text

    async def send_text(chat_id: str, text: str, **kwargs: Any) -> str:
        sending.set()
        await release.wait()
        return await send(chat_id, text, **kwargs)

    async def stream(text: str, attachments: Any = None) -> AsyncIterator[dict[str, Any]]:
        task = asyncio.current_task()
        try:
            yield {"type": "response", "content": "already rendering", "complete": True}
            yield {"type": "response", "content": "stale output", "complete": True}
        finally:
            assert asyncio.current_task() is task
            closed.set()

    monkeypatch.setattr(adapter, "send_text", send_text)
    monkeypatch.setattr(entry.runtime, "send_message", stream)
    run = AsyncMock(wraps=turns.run)
    monkeypatch.setattr(turns, "run", run)
    try:
        await handler.handle(chat_ref(), inbound("fenced"))
        await asyncio.wait_for(sending.wait(), 10)
        turns.owner = object()
        release.set()
        assert await wait_for(lambda: closed.is_set() and entry.turn_task is None)
        assert run.await_count == 1
        save.assert_not_called()
        assert adapter.sent == [(chat_ref().chat_id, "already rendering")]
        assert not entry.record.plan_pending
    finally:
        turns.owner = None
        release.set()
        await handler.shutdown()


@pytest.mark.asyncio
async def test_background_turn_reclaims_ownership_after_intervening_plan_implementation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    handler, adapter, agents = host(tmp_path)
    implementing, release_implementation = asyncio.Event(), asyncio.Event()
    continuing, release_continuation = asyncio.Event(), asyncio.Event()
    reporting, release_report = asyncio.Event(), asyncio.Event()
    background_tasks: list[asyncio.Task[Any]] = []
    send = adapter.send_text

    async def slow_report(chat_id: str, text: str, **kwargs: Any) -> str:
        if text == "child report":
            reporting.set()
            await release_report.wait()
        return await send(chat_id, text, **kwargs)

    monkeypatch.setattr(adapter, "send_text", slow_report)
    try:
        await handler.handle(chat_ref(), inbound("/mode plan"))
        entry = await handler._ensure_session(chat_ref())
        turns = entry.runtime.turns

        async def stream(text: str, attachments: Any = None) -> AsyncIterator[dict[str, Any]]:
            if text == "plan":
                yield {"type": "response", "content": "# Plan\n\nBuild it.", "complete": True}
            else:
                implementing.set()
                await release_implementation.wait()
                yield {"type": "response", "content": "implemented", "complete": True}

        monkeypatch.setattr(entry.runtime, "send_message", stream)
        await handler.handle(chat_ref(), inbound("plan"))
        assert await wait_for(lambda: adapter.tokens and entry.turn_task is None)
        request = next(request for request in entry.runtime.control.pending() if request.kind == "plan_approval")
        start = turns.start

        def observe_start(execute: Any, **kwargs: Any) -> asyncio.Task[Any]:
            task = start(execute, **kwargs)
            background_tasks.append(task)
            monkeypatch.setattr(turns, "start", start)
            return task

        monkeypatch.setattr(turns, "start", observe_start)
        assert entry.runtime.answer_plan_approval(request.request_id, "implement_plan", client_id=chat_ref().key)
        queue_reports(agents[-1], "child report")
        await asyncio.wait_for(implementing.wait(), 10)
        implementation_task = entry.turn_task
        assert len(background_tasks) == 1
        background_task = background_tasks[0]
        assert not background_task.done()
        assert background_task is not implementation_task

        async def continuation() -> AsyncIterator[dict[str, Any]]:
            continuing.set()
            await release_continuation.wait()
            yield {"type": "response", "content": "continuation done", "complete": True}

        monkeypatch.setattr(entry.runtime.agent, "continue_from_history_stream", continuation)
        release_implementation.set()
        await asyncio.wait_for(reporting.wait(), 10)
        assert await wait_for(lambda: entry.plan_approval_task is None)
        release_report.set()
        await asyncio.wait_for(continuing.wait(), 10)
        assert entry.turn_task is turns.owner is background_task
        assert not background_task.done()
        assert await handler._stop_turn(chat_ref())
        assert background_task.cancelling() == 1
        assert await wait_for(lambda: entry.turn_task is None)
        assert background_task.cancelled()
        assert not handler._workers[chat_ref().key].done()
    finally:
        release_report.set()
        release_implementation.set()
        release_continuation.set()
        await handler.shutdown()
