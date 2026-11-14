from __future__ import annotations

import asyncio
import subprocess
import sys
import textwrap
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import anyio
import pytest
import pytest_asyncio

from kolega_code.session.turns import TurnBusyError, TurnController, TurnHooks, TurnOutcome, TurnQueue, TurnResult


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


async def _wait(event: asyncio.Event) -> None:
    await asyncio.wait_for(event.wait(), 10)


async def _consume(stream: AsyncIterator[dict[str, Any]]) -> None:
    async for _ in stream:
        pass


async def _one_chunk() -> AsyncIterator[dict[str, Any]]:
    yield {"content": "answer"}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("failure_at", "recoverable"),
    [
        (None, False),
        ("before", False),
        ("stream", False),
        ("after", False),
        ("before", True),
        ("stream", True),
        ("after", True),
    ],
)
async def test_run_orders_all_phases_and_preserves_the_original_exception(
    failure_at: str | None, recoverable: bool, scheduling: None
) -> None:
    controller: TurnController[str] = TurnController()
    entered, release = asyncio.Event(), asyncio.Event()
    trace: list[str] = []
    results: list[TurnResult] = []
    failure = ValueError("presentation rejected")

    async def before_one() -> None:
        trace.append("before-one")
        if failure_at == "before":
            raise failure

    async def before_two() -> None:
        trace.append("before-two")

    async def stream() -> AsyncIterator[dict[str, Any]]:
        try:
            trace.append("stream")
            entered.set()
            await release.wait()
            if failure_at == "stream":
                raise failure
            yield {"content": "answer"}
        finally:
            trace.append("stream-close")

    async def consume(chunks: AsyncIterator[dict[str, Any]]) -> None:
        trace.append("consume")
        async for _ in chunks:
            trace.append("chunk")

    async def after_one() -> None:
        trace.append("after-one")
        if failure_at == "after":
            raise failure

    async def after_two() -> None:
        trace.append("after-two")

    async def settle(result: TurnResult) -> None:
        assert asyncio.current_task() is controller.execution_task
        trace.append("settle")
        results.append(result)

    def teardown(result: TurnResult) -> None:
        assert asyncio.current_task() is controller.execution_task
        trace.append("teardown")
        results.append(result)

    async def execute() -> TurnResult:
        return await controller.run(
            stream,
            hooks=TurnHooks(
                consume=consume,
                before=(before_one, before_two),
                after=(after_one, after_two),
                settle=settle,
                teardown=teardown,
            ),
            recoverable_errors=(ValueError,) if recoverable else (),
            format_error=lambda exc: f"formatted: {exc}",
        )

    task = controller.start(execute)
    try:
        if failure_at != "before":
            await _wait(entered)
            assert trace == ["before-one", "before-two", "consume", "stream"]
            assert results == []
        release.set()
        if failure_at is not None and not recoverable:
            with pytest.raises(ValueError) as raised:
                await task
            assert raised.value is failure
        else:
            result = await task
            assert result is results[-1]
        expected = ["before-one"]
        if failure_at != "before":
            expected += ["before-two", "consume", "stream"]
            if failure_at != "stream":
                expected.append("chunk")
            expected.append("stream-close")
            if failure_at != "stream":
                expected.append("after-one")
                if failure_at != "after":
                    expected.append("after-two")
        if failure_at is not None:
            expected.append("settle")
            assert len(results) == 2
            assert results[0] is results[1]
            assert results[0].outcome is TurnOutcome.FAILED
            assert results[0].exception is failure
            assert results[0].error == "formatted: presentation rejected"
        else:
            assert len(results) == 1
            assert results[0] == TurnResult(TurnOutcome.COMPLETED)
        assert trace == expected + ["teardown"]
        assert not controller.busy
    finally:
        release.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("resist_settlement_cancel", [False, True])
@pytest.mark.parametrize("suppress_cancel", [False, True])
async def test_repeated_cancellation_during_settlement_always_tears_down_once(
    resist_settlement_cancel: bool, suppress_cancel: bool, scheduling: None
) -> None:
    controller: TurnController[str] = TurnController()
    streaming, settling, release, finished = asyncio.Event(), asyncio.Event(), asyncio.Event(), asyncio.Event()
    controller.on_finished = lambda task, draining: finished.set()
    observed: asyncio.Queue[int] = asyncio.Queue()
    trace: list[str] = []
    results: list[TurnResult] = []

    async def stream() -> AsyncIterator[dict[str, Any]]:
        try:
            streaming.set()
            await asyncio.Event().wait()
            yield {}
        finally:
            trace.append("stream-close")

    async def settle(result: TurnResult) -> None:
        results.append(result)
        trace.append("settle")
        settling.set()
        try:
            while not release.is_set():
                try:
                    await release.wait()
                except asyncio.CancelledError:
                    task = asyncio.current_task()
                    assert task is controller.execution_task
                    assert task is not None
                    observed.put_nowait(task.cancelling())
                    if not resist_settlement_cancel:
                        raise
        finally:
            trace.append("settle-exit")

    def teardown(result: TurnResult) -> None:
        assert result is results[0]
        trace.append("teardown")

    async def execute() -> TurnResult:
        return await controller.run(
            stream,
            hooks=TurnHooks(consume=_consume, settle=settle, teardown=teardown),
            suppress_cancel=suppress_cancel,
        )

    task = controller.start(execute)
    try:
        await _wait(streaming)
        assert controller.request_stop()
        await _wait(settling)
        task.cancel()
        assert await asyncio.wait_for(observed.get(), 10) == 2
        if resist_settlement_cancel:
            task.cancel()
            assert await asyncio.wait_for(observed.get(), 10) == 3
            assert trace == ["stream-close", "settle"]
            assert controller.busy
        release.set()
        if suppress_cancel and resist_settlement_cancel:
            assert await task is results[0]
        else:
            with pytest.raises(asyncio.CancelledError):
                await task
        await _wait(finished)
        assert trace == ["stream-close", "settle", "settle-exit", "teardown"]
        assert len(results) == 1
        assert results[0].outcome is TurnOutcome.CANCELLED
        assert isinstance(results[0].exception, asyncio.CancelledError)
        assert not controller.stop_pending
        assert not controller.busy
        await controller.start(lambda: controller.run(_one_chunk, hooks=TurnHooks(consume=_consume)))
    finally:
        release.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("exit_kind", ["break", "failure", "cancel"])
async def test_source_cancel_scope_is_closed_in_the_consumers_original_task(exit_kind: str, scheduling: None) -> None:
    controller: TurnController[str] = TurnController()
    consuming, finished = asyncio.Event(), asyncio.Event()
    controller.on_finished = lambda task, draining: finished.set()
    trace: list[str] = []
    failure = RuntimeError("consumer failed")
    original: asyncio.Task[Any] | None = None

    async def stream() -> AsyncIterator[dict[str, Any]]:
        assert asyncio.current_task() is original
        with anyio.CancelScope():
            try:
                trace.append("stream")
                yield {"content": "answer"}
                await asyncio.Event().wait()
            finally:
                assert asyncio.current_task() is original
                trace.append("close")
        trace.append("scope-exit")

    async def consume(chunks: AsyncIterator[dict[str, Any]]) -> None:
        assert asyncio.current_task() is original
        async for _ in chunks:
            consuming.set()
            if exit_kind == "failure":
                raise failure
            if exit_kind == "cancel":
                await asyncio.Event().wait()
            break

    def teardown(result: TurnResult) -> None:
        assert asyncio.current_task() is original
        trace.append("teardown")

    async def execute() -> TurnResult:
        nonlocal original
        original = asyncio.current_task()
        return await controller.run(stream, hooks=TurnHooks(consume=consume, teardown=teardown))

    task = controller.start(execute)
    try:
        await _wait(consuming)
        if exit_kind == "cancel":
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        elif exit_kind == "failure":
            with pytest.raises(RuntimeError) as raised:
                await task
            assert raised.value is failure
        else:
            assert (await task).outcome is TurnOutcome.COMPLETED
        await _wait(finished)
        assert trace == ["stream", "close", "teardown"]
        assert not controller.busy
    finally:
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("save_fails", [False, True])
async def test_detach_retains_a_cancellation_resistant_late_save_until_actual_completion(
    save_fails: bool, scheduling: None
) -> None:
    controller: TurnController[str] = TurnController()
    saving, cancelled, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
    trace: list[str] = []
    finished: list[tuple[asyncio.Task[Any], bool]] = []
    failure = OSError("late save failed")
    controller.on_finished = lambda task, draining: finished.append((task, draining))

    async def save() -> None:
        saving.set()
        while not release.is_set():
            try:
                await release.wait()
            except asyncio.CancelledError:
                cancelled.set()
        trace.append("save-finished")
        if save_fails:
            raise failure

    async def after_save() -> None:
        trace.append("after-save")

    async def settle(result: TurnResult) -> None:
        trace.append("settle")

    def teardown(result: TurnResult) -> None:
        trace.append("teardown")

    async def execute() -> TurnResult:
        return await controller.run(
            _one_chunk,
            hooks=TurnHooks(consume=_consume, after=(save, after_save), settle=settle, teardown=teardown),
        )

    task = controller.start(execute)
    try:
        await _wait(saving)
        controller.detach()
        await _wait(cancelled)
        controller.inputs.put_nowait("queued")
        assert controller.owner is None
        assert controller.execution_task is task
        assert task in controller.draining_tasks
        assert controller.busy
        assert finished == []
        assert not await controller.wait_settled(timeout=0)
        assert controller.take_inputs(mid_turn=True) == []
        with pytest.raises(asyncio.QueueEmpty):
            controller.take_next()
        with pytest.raises(TurnBusyError):
            controller.start(execute)
        release.set()
        if save_fails:
            with pytest.raises(OSError) as raised:
                await task
            assert raised.value is failure
        else:
            assert (await task).outcome is TurnOutcome.COMPLETED
        assert trace == ["save-finished"]
        assert finished == [(task, True)]
        assert not controller.draining_tasks
        assert not controller.busy
        assert controller.take_next() == "queued"
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_queue_fifo_take_all_clears_readiness_without_losing_the_next_wakeup(scheduling: None) -> None:
    queue: TurnQueue[object] = TurnQueue()
    first, second, third = object(), object(), object()
    queue.put_nowait(first)
    queue.put_nowait(second)
    queue.put_nowait(third)
    assert await queue.get() is first
    assert queue.take_all() == [second, third]
    assert queue.empty()
    assert queue.qsize() == 0
    assert queue.take_all() == []
    with pytest.raises(asyncio.QueueEmpty):
        queue.get_nowait()
    entered = asyncio.Event()

    async def read() -> object:
        entered.set()
        return await queue.get()

    reader = asyncio.create_task(read())
    try:
        await _wait(entered)
        assert not reader.done()
        queue.put_nowait(third)
        assert await asyncio.wait_for(reader, 10) is third
        assert queue.empty()
    finally:
        if not reader.done():
            reader.cancel()
        await asyncio.gather(reader, return_exceptions=True)


@pytest.mark.asyncio
async def test_only_the_active_execution_can_take_a_mid_turn_fifo_boundary(scheduling: None) -> None:
    queue: TurnQueue[dict[str, Any]] = TurnQueue()
    controller = TurnController(inputs=queue)
    first = {"text": "one", "attachments": [{"id": "image"}]}
    second = {"text": "two", "origin": {"kind": "peer"}}
    queue.put_nowait(first)
    queue.put_nowait(second)
    entered, release = asyncio.Event(), asyncio.Event()
    taken: list[dict[str, Any]] = []

    async def execute() -> None:
        entered.set()
        await release.wait()
        assert controller.take_inputs() == []
        taken.extend(controller.take_inputs(mid_turn=True))

    task = controller.start(execute)
    try:
        await _wait(entered)
        assert controller.inputs is queue
        assert controller.take_inputs(mid_turn=True) == []
        with pytest.raises(asyncio.QueueEmpty):
            controller.take_next()
        with pytest.raises(TurnBusyError):
            controller.start(execute)
        release.set()
        await task
        assert taken == [first, second]
        assert taken[0] is first
        assert taken[1] is second
        queue.put_nowait(first)
        assert controller.take_next() is first
        controller.close_admission()
        queue.put_nowait(second)
        assert controller.take_inputs(mid_turn=True) == []
        with pytest.raises(asyncio.QueueEmpty):
            controller.take_next()
        with pytest.raises(TurnBusyError):
            controller.start(execute)
        assert queue.take_all() == [second]
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("replace_owner", [False, True])
async def test_track_replacement_requires_a_changed_opaque_owner_and_retains_old_work(
    replace_owner: bool, scheduling: None
) -> None:
    controller: TurnController[str] = TurnController()
    old_owner, new_owner = object(), object()
    old_entered, new_entered = asyncio.Event(), asyncio.Event()
    old_release, new_release = asyncio.Event(), asyncio.Event()

    async def old_work() -> None:
        old_entered.set()
        await old_release.wait()

    async def new_work() -> None:
        new_entered.set()
        await new_release.wait()

    old_task = asyncio.create_task(old_work())
    new_task = asyncio.create_task(new_work())
    try:
        controller.owner = old_owner
        controller.track(old_task)
        await _wait(old_entered)
        await _wait(new_entered)
        if not replace_owner:
            with pytest.raises(TurnBusyError):
                controller.track(new_task)
            assert controller.execution_task is old_task
            assert controller.owner is old_owner
        else:
            controller.owner = new_owner
            controller.track(new_task)
            assert controller.execution_task is new_task
            old_release.set()
            await old_task
            assert controller.owner is new_owner
            assert controller.execution_task is new_task
            assert not await controller.wait_settled(timeout=0)
            new_release.set()
            await new_task
            assert controller.owner is new_owner
        assert controller.busy
        with pytest.raises(TurnBusyError):
            controller.start(new_work)
    finally:
        old_release.set()
        new_release.set()
        await asyncio.gather(old_task, new_task, return_exceptions=True)
        controller.owner = None
        controller.disarm_watchdog()


@pytest.mark.asyncio
@pytest.mark.parametrize("owner_state", ["retained", "released", "replacement-owner", "replacement-execution"])
async def test_execution_completion_clears_only_its_matching_stop_state(owner_state: str, scheduling: None) -> None:
    controller: TurnController[str] = TurnController()
    entered, release, successor_release = asyncio.Event(), asyncio.Event(), asyncio.Event()
    original_owner = controller.owner = object()
    escalated: list[str] = []

    async def execute() -> None:
        entered.set()
        await release.wait()

    task = asyncio.create_task(execute())
    successor: asyncio.Task[bool] | None = None
    controller.track(task)
    try:
        await _wait(entered)
        controller.request_stop(cancel=lambda: None, force_after=3600, on_escalate=lambda: escalated.append("old"))
        old_handle = controller.watchdog
        assert old_handle is not None
        old_callback = old_handle._callback
        assert old_callback is not None
        replaced = owner_state.startswith("replacement")
        if replaced:
            controller.owner = object()
            if owner_state == "replacement-execution":
                successor = asyncio.create_task(successor_release.wait())
                controller.track(successor)
            controller.arm_watchdog(3600, lambda: escalated.append("new"))
        elif owner_state == "released":
            controller.owner = None
        expected_owner, expected_watchdog = controller.owner, controller.watchdog
        release.set()
        await task
        assert controller.owner is expected_owner
        assert controller.execution_task is successor
        assert controller.stop_pending is replaced
        if replaced:
            assert controller.owner is not original_owner
            assert controller.watchdog is expected_watchdog
            assert expected_watchdog is not None and not expected_watchdog.cancelled()
        else:
            assert controller.watchdog is None
            assert old_handle.cancelled()
        old_callback()
        assert escalated == []
        assert controller.stop_pending is replaced
        assert controller.watchdog is (expected_watchdog if replaced else None)
    finally:
        release.set()
        successor_release.set()
        await asyncio.gather(task, *([successor] if successor is not None else []), return_exceptions=True)
        controller.owner = None
        controller.disarm_watchdog()


@pytest.mark.asyncio
async def test_replaced_execution_remains_in_shutdown_waits_after_successor_finishes(scheduling: None) -> None:
    controller: TurnController[str] = TurnController()
    old_release, new_release = asyncio.Event(), asyncio.Event()

    async def old_work() -> None:
        await old_release.wait()

    async def new_work() -> None:
        await new_release.wait()

    old_task = asyncio.create_task(old_work())
    new_task = asyncio.create_task(new_work())
    try:
        controller.owner = object()
        controller.track(old_task)
        controller.owner = object()
        controller.track(new_task)
        new_release.set()
        await new_task
        assert not old_task.done()
        assert not await controller.wait_settled(timeout=0)
        old_release.set()
        await old_task
        assert await controller.wait_settled(timeout=0)
    finally:
        old_release.set()
        new_release.set()
        await asyncio.gather(old_task, new_task, return_exceptions=True)
        controller.owner = None


@pytest.mark.asyncio
async def test_watchdog_manually_fires_for_the_current_owner(scheduling: None) -> None:
    controller: TurnController[str] = TurnController()
    owner = controller.owner = object()
    escalated: list[object] = []
    assert controller.request_stop(cancel=lambda: None, force_after=3600, on_escalate=lambda: escalated.append(owner))
    handle = controller.watchdog
    assert handle is not None
    callback = handle._callback
    assert callback is not None
    handle.cancel()
    try:
        callback()
        assert escalated == [owner]
        assert controller.watchdog is None
        assert controller.owner is owner
        assert controller.stop_pending
    finally:
        controller.disarm_watchdog()


@pytest.mark.asyncio
async def test_obsolete_watchdog_callback_cannot_escalate_or_erase_successors_timer(scheduling: None) -> None:
    controller: TurnController[str] = TurnController()
    controller.owner = object()
    escalated: list[str] = []
    controller.request_stop(cancel=lambda: None, force_after=3600, on_escalate=lambda: escalated.append("old"))
    old_handle = controller.watchdog
    assert old_handle is not None
    old_callback = old_handle._callback
    assert old_callback is not None
    successor = controller.owner = object()
    controller.arm_watchdog(3600, lambda: escalated.append("new"))
    new_handle = controller.watchdog
    assert new_handle is not None
    try:
        old_callback()
        assert escalated == []
        assert controller.owner is successor
        assert controller.stop_pending
        assert controller.watchdog is new_handle
        assert not new_handle.cancelled()
    finally:
        old_handle.cancel()
        new_handle.cancel()
        controller.disarm_watchdog()


@pytest.mark.asyncio
@pytest.mark.parametrize("detached", [False, True])
async def test_cancelling_an_entered_shutdown_waiter_never_cancels_execution_or_extra_save(
    detached: bool, scheduling: None
) -> None:
    controller: TurnController[str] = TurnController()
    running, saving, waiting = asyncio.Event(), asyncio.Event(), asyncio.Event()
    turn_release, save_release = asyncio.Event(), asyncio.Event()

    async def execute() -> None:
        running.set()
        await turn_release.wait()

    async def save() -> None:
        saving.set()
        await save_release.wait()

    task = controller.start(execute)
    save_task = asyncio.create_task(save())

    async def shutdown_wait() -> bool:
        waiting.set()
        return await controller.wait_settled(extra=(save_task,))

    waiter: asyncio.Task[bool] | None = None
    try:
        await _wait(running)
        await _wait(saving)
        if detached:
            controller.detach(cancel=lambda: None)
        controller.close_admission()
        waiter = asyncio.create_task(shutdown_wait())
        await _wait(waiting)
        assert not waiter.done()
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        assert task.cancelling() == 0
        assert save_task.cancelling() == 0
        assert not task.done()
        assert not save_task.done()
        with pytest.raises(TurnBusyError):
            controller.start(execute)
        turn_release.set()
        await task
        assert not await controller.wait_settled(timeout=0, extra=(save_task,))
        save_release.set()
        await save_task
        assert await controller.wait_settled(extra=(save_task,))
        assert controller.closed
    finally:
        turn_release.set()
        save_release.set()
        if waiter is not None:
            await asyncio.gather(waiter, return_exceptions=True)
        await asyncio.gather(task, save_task, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("take_boundary", [False, True])
async def test_owner_replaced_during_post_stream_await_cannot_mutate_successor(
    take_boundary: bool, scheduling: None
) -> None:
    controller: TurnController[str] = TurnController()
    entered, release = asyncio.Event(), asyncio.Event()
    old_owner, successor = object(), object()
    trace: list[str] = []
    taken: list[str] = []

    async def save() -> None:
        entered.set()
        await release.wait()
        if take_boundary:
            taken.extend(controller.take_inputs(mid_turn=True))

    async def next_after() -> None:
        trace.append("next-after")

    def teardown(result: TurnResult) -> None:
        trace.append("teardown")

    async def execute() -> TurnResult:
        return await controller.run(
            _one_chunk, hooks=TurnHooks(consume=_consume, after=(save, next_after), teardown=teardown)
        )

    controller.owner = old_owner
    task = asyncio.create_task(execute())
    controller.track(task)
    successor_watchdog: asyncio.TimerHandle | None = None
    try:
        await _wait(entered)
        controller.inputs.put_nowait("successor-input")
        controller.owner = successor
        controller.stop_pending = True
        controller.arm_watchdog(3600, lambda: trace.append("successor-escalation"))
        successor_watchdog = controller.watchdog
        release.set()
        await task
        assert taken == []
        assert controller.inputs.items == ["successor-input"]
        assert trace == []
        assert controller.owner is successor
        assert controller.stop_pending
        assert controller.watchdog is successor_watchdog
        assert successor_watchdog is not None
        assert not successor_watchdog.cancelled()
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        if successor_watchdog is not None:
            successor_watchdog.cancel()
        controller.disarm_watchdog()
        controller.owner = None


@pytest.mark.asyncio
async def test_overlapping_runs_with_replaced_opaque_owner_keep_successors_run_guard_and_teardown(
    scheduling: None,
) -> None:
    controller: TurnController[str] = TurnController()
    old_entered, new_entered = asyncio.Event(), asyncio.Event()
    old_release, new_release, new_begin = asyncio.Event(), asyncio.Event(), asyncio.Event()
    old_owner, successor = object(), object()
    trace: list[str] = []
    output: list[str] = []

    async def old_save() -> None:
        old_entered.set()
        await old_release.wait()

    async def old_after() -> None:
        trace.append("old-after")

    def old_teardown(result: TurnResult) -> None:
        trace.append("old-teardown")

    async def old_execute() -> TurnResult:
        return await controller.run(
            _one_chunk,
            hooks=TurnHooks(consume=_consume, after=(old_save, old_after), teardown=old_teardown),
        )

    async def new_stream() -> AsyncIterator[dict[str, Any]]:
        yield {"content": "successor"}
        new_entered.set()
        await new_release.wait()

    async def new_consume(chunks: AsyncIterator[dict[str, Any]]) -> None:
        async for chunk in chunks:
            output.append(chunk["content"])

    def new_teardown(result: TurnResult) -> None:
        assert result.outcome is TurnOutcome.COMPLETED
        trace.append("new-teardown")

    async def new_execute() -> TurnResult:
        await new_begin.wait()
        return await controller.run(new_stream, hooks=TurnHooks(consume=new_consume, teardown=new_teardown))

    controller.owner = old_owner
    old_task = asyncio.create_task(old_execute())
    controller.track(old_task)
    new_task: asyncio.Task[TurnResult] | None = None
    successor_watchdog: asyncio.TimerHandle | None = None
    try:
        await _wait(old_entered)
        controller.owner = successor
        new_task = asyncio.create_task(new_execute())
        controller.track(new_task)
        new_begin.set()
        await _wait(new_entered)
        controller.stop_pending = True
        controller.arm_watchdog(3600, lambda: trace.append("new-escalation"))
        successor_watchdog = controller.watchdog
        old_release.set()
        assert (await old_task).outcome is TurnOutcome.COMPLETED
        assert controller.owner is successor
        assert controller.execution_task is new_task
        assert controller.stop_pending
        assert controller.watchdog is successor_watchdog
        assert trace == []
        assert output == ["successor"]
        with pytest.raises(TurnBusyError):
            await controller.run(_one_chunk, hooks=TurnHooks(consume=_consume))
        with pytest.raises(TurnBusyError):
            controller.start(new_execute)
        new_release.set()
        assert (await new_task).outcome is TurnOutcome.COMPLETED
        assert trace == ["new-teardown"]
        assert not controller.stop_pending
        assert controller.watchdog is None
        assert controller.owner is successor
    finally:
        old_release.set()
        new_begin.set()
        new_release.set()
        await asyncio.gather(old_task, return_exceptions=True)
        if new_task is not None:
            await asyncio.gather(new_task, return_exceptions=True)
        if successor_watchdog is not None:
            successor_watchdog.cancel()
        controller.disarm_watchdog()
        controller.owner = None


def test_headless_import_and_controller_runtime_with_textual_and_rich_blocked() -> None:
    program = textwrap.dedent(
        """
        import asyncio
        import importlib.abc
        import sys

        class NoUI(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split(".", 1)[0] in {"textual", "rich"}:
                    raise AssertionError("UI dependency imported: " + fullname)
                return None

        sys.meta_path.insert(0, NoUI())
        from kolega_code.session.turns import TurnController, TurnHooks, TurnOutcome

        async def main():
            controller = TurnController()
            output = []
            async def stream():
                yield {"content": "headless"}
            async def consume(chunks):
                async for chunk in chunks:
                    output.append(chunk["content"])
            result = await controller.start(
                lambda: controller.run(stream, hooks=TurnHooks(consume=consume))
            )
            assert result.outcome is TurnOutcome.COMPLETED
            assert output == ["headless"]
            assert not controller.busy
            controller.close_admission()
            assert await controller.wait_settled()

        asyncio.run(main())
        assert not any(name.split(".", 1)[0] in {"textual", "rich"} for name in sys.modules)
        print("headless-controller-ok")
        """
    )
    result = subprocess.run(
        [sys.executable, "-B", "-c", program],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == "headless-controller-ok"
