from __future__ import annotations

import asyncio
from contextvars import ContextVar
from typing import Any, AsyncIterator

import pytest
import pytest_asyncio

from kolega_code.session.turns import TurnBusyError, TurnController, TurnHooks, TurnOutcome


@pytest_asyncio.fixture(params=["default", "eager"])
async def scheduling(request: pytest.FixtureRequest) -> AsyncIterator[None]:
    loop = asyncio.get_running_loop()
    previous = loop.get_task_factory()
    if request.param == "eager":
        factory = getattr(asyncio, "eager_task_factory", None)
        if factory is None:
            pytest.skip("Eager tasks require Python 3.12")
        loop.set_task_factory(factory)
    try:
        yield
    finally:
        loop.set_task_factory(previous)


async def _consume(stream: AsyncIterator[dict[str, Any]]) -> None:
    async for _ in stream:
        pass


async def _one_chunk() -> AsyncIterator[dict[str, Any]]:
    yield {"type": "response", "content": "answer"}


@pytest.mark.asyncio
async def test_immediate_execution_publishes_ownership_before_entry(scheduling: None) -> None:
    controller: TurnController[str] = TurnController()
    seen: list[bool] = []

    async def execute() -> None:
        seen.append(controller.execution_task is asyncio.current_task())
        seen.append(controller.owner is asyncio.current_task())
        await controller.run(_one_chunk, hooks=TurnHooks(consume=_consume))

    task = controller.start(execute)
    assert controller.execution_task is task
    await task
    assert seen == [True, True]
    assert controller.execution_task is None
    assert controller.owner is None


@pytest.mark.asyncio
async def test_preentry_cancel_releases_the_reservation_without_starting_work(scheduling: None) -> None:
    controller: TurnController[str] = TurnController()
    entered = False

    async def execute() -> None:
        nonlocal entered
        entered = True

    task = controller.start(execute)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not entered
    assert controller.execution_task is None
    assert controller.owner is None
    await controller.start(execute)
    assert entered


@pytest.mark.asyncio
async def test_body_settlement_and_teardown_keep_the_original_task_and_context(scheduling: None) -> None:
    controller: TurnController[str] = TurnController()
    binding: ContextVar[object] = ContextVar("turn-binding")
    expected = object()
    trace: list[str] = []

    async def stream() -> AsyncIterator[dict[str, Any]]:
        token = binding.set(expected)
        try:
            trace.append("body")
            yield {"type": "response", "content": "answer"}
            await asyncio.Event().wait()
        finally:
            assert controller.execution_task is asyncio.current_task()
            assert binding.get() is expected
            binding.reset(token)
            trace.append("stream-close")

    async def consume(chunks: AsyncIterator[dict[str, Any]]) -> None:
        async for _ in chunks:
            assert binding.get() is expected
            raise ValueError("presentation failed")

    async def settle(result: Any) -> None:
        assert controller.execution_task is asyncio.current_task()
        assert result.outcome is TurnOutcome.FAILED
        trace.append("settle")

    def teardown(result: Any) -> None:
        assert result.outcome is TurnOutcome.FAILED
        trace.append("teardown")

    async def execute() -> None:
        await controller.run(stream, hooks=TurnHooks(consume=consume, settle=settle, teardown=teardown))

    with pytest.raises(ValueError, match="presentation failed"):
        await controller.start(execute)
    assert trace == ["body", "stream-close", "settle", "teardown"]


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["prepare", "stream", "save"])
async def test_cancellation_settles_once_then_tears_down(stage: str, scheduling: None) -> None:
    controller: TurnController[str] = TurnController()
    entered = asyncio.Event()
    trace: list[str] = []

    async def gate() -> None:
        entered.set()
        await asyncio.Event().wait()

    async def stream() -> AsyncIterator[dict[str, Any]]:
        if stage == "stream":
            await gate()
        yield {"type": "response", "content": "answer"}

    async def settle(result: Any) -> None:
        assert result.outcome is TurnOutcome.CANCELLED
        trace.append("settle")

    def teardown(result: Any) -> None:
        assert result.outcome is TurnOutcome.CANCELLED
        trace.append("teardown")

    async def execute() -> Any:
        return await controller.run(
            stream,
            hooks=TurnHooks(
                consume=_consume,
                before=(gate,) if stage == "prepare" else (),
                after=(gate,) if stage == "save" else (),
                settle=settle,
                teardown=teardown,
            ),
            suppress_cancel=True,
        )

    task = controller.start(execute)
    await asyncio.wait_for(entered.wait(), 10)
    assert controller.request_stop()
    result = await task
    assert result.outcome is TurnOutcome.CANCELLED
    assert trace == ["settle", "teardown"]
    assert not controller.stop_pending


@pytest.mark.asyncio
async def test_detached_stream_keeps_execution_fenced_and_discards_late_output(scheduling: None) -> None:
    controller: TurnController[str] = TurnController()
    entered, cancelled, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
    output: list[str] = []
    trace: list[str] = []

    async def stream() -> AsyncIterator[dict[str, Any]]:
        yield {"content": "before"}
        entered.set()
        while not release.is_set():
            try:
                await release.wait()
            except asyncio.CancelledError:
                cancelled.set()
        yield {"content": "late"}
        raise RuntimeError("late detached failure")

    async def consume(chunks: AsyncIterator[dict[str, Any]]) -> None:
        async for chunk in chunks:
            output.append(chunk["content"])

    async def after() -> None:
        trace.append("after")

    def teardown(result: Any) -> None:
        trace.append("teardown")

    async def execute() -> None:
        await controller.run(stream, hooks=TurnHooks(consume=consume, after=(after,), teardown=teardown))

    task = controller.start(execute)
    await asyncio.wait_for(entered.wait(), 10)
    controller.request_stop()
    await asyncio.wait_for(cancelled.wait(), 10)
    controller.detach()
    controller.inputs.put_nowait("follow-up")
    try:
        assert controller.owner is None
        assert task in controller.draining_tasks
        assert not await controller.wait_settled(timeout=0)
        with pytest.raises(TurnBusyError):
            controller.start(execute)
        assert controller.take_inputs() == []
        assert controller.inputs.qsize() == 1
    finally:
        release.set()
        with pytest.raises(RuntimeError, match="late detached failure"):
            await task
    assert output == ["before"]
    assert trace == []
    assert not controller.draining_tasks
    assert controller.take_inputs() == ["follow-up"]


@pytest.mark.asyncio
async def test_boundary_drain_checks_ownership_after_the_frontend_await(scheduling: None) -> None:
    controller: TurnController[dict[str, Any]] = TurnController()
    entered, release = asyncio.Event(), asyncio.Event()
    payload = {"text": "wrapped", "display_text": "raw", "origin": {"kind": "peer"}, "attachments": [{"id": "image"}]}
    controller.inputs.put_nowait(payload)
    taken: list[dict[str, Any]] = []

    async def execute() -> None:
        entered.set()
        await release.wait()
        taken.extend(controller.take_inputs(mid_turn=True))

    task = controller.start(execute)
    await asyncio.wait_for(entered.wait(), 10)
    controller.detach(cancel=lambda: None)
    release.set()
    await task
    assert taken == []
    assert controller.take_inputs() == [payload]


@pytest.mark.asyncio
async def test_cancelled_drain_waiter_does_not_cancel_owned_execution(scheduling: None) -> None:
    controller: TurnController[str] = TurnController()
    entered, release = asyncio.Event(), asyncio.Event()

    async def execute() -> None:
        entered.set()
        await release.wait()

    task = controller.start(execute)
    await asyncio.wait_for(entered.wait(), 10)
    controller.close_admission()
    waiter = asyncio.create_task(controller.wait_settled())
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    assert not task.cancelling()
    with pytest.raises(TurnBusyError):
        controller.start(execute)
    release.set()
    await task
    assert await controller.wait_settled()


@pytest.mark.asyncio
async def test_republishing_retained_execution_registers_completion_only_once(scheduling: None) -> None:
    controller: TurnController[str] = TurnController()
    entered, release = asyncio.Event(), asyncio.Event()
    finished: list[asyncio.Task[Any]] = []
    controller.on_finished = lambda task, draining: finished.append(task)

    async def execute() -> None:
        entered.set()
        await release.wait()
        assert controller.execution_task is controller.owner is asyncio.current_task()

    retained = controller.start(execute)
    try:
        await asyncio.wait_for(entered.wait(), 10)
        intervening = asyncio.create_task(asyncio.sleep(0))
        controller.owner = intervening
        controller.track(intervening)
        await intervening
        assert controller.execution_task is controller.owner is None
        assert controller.busy
        controller.owner = retained
        controller.track(retained)
    finally:
        release.set()
        await retained
    await asyncio.sleep(0)
    assert finished == [intervening, retained]
    assert controller.execution_task is controller.owner is None
    assert not controller.busy
