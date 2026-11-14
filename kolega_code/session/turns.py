"""UI-free execution, cancellation, and input ownership for a session."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable, Iterable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Generic, TypeVar

InputT = TypeVar("InputT")
ResultT = TypeVar("ResultT")
Chunk = dict[str, Any]
StreamFactory = Callable[[], AsyncIterator[Chunk]]
AsyncStep = Callable[[], Awaitable[None]]


class TurnBusyError(RuntimeError):
    """Execution is still owned, draining, or closed to new work."""


class TurnOutcome(StrEnum):
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


@dataclass(frozen=True)
class TurnResult:
    outcome: TurnOutcome
    error: str = ""
    exception: BaseException | None = None


@dataclass(frozen=True)
class TurnHooks:
    consume: Callable[[AsyncIterator[Chunk]], Awaitable[None]]
    before: tuple[AsyncStep, ...] = ()
    after: tuple[AsyncStep, ...] = ()
    settle: Callable[[TurnResult], Awaitable[None]] | None = None
    teardown: Callable[[TurnResult], None] | None = None


class TurnQueue(Generic[InputT]):
    def __init__(self) -> None:
        self.items: list[InputT] = []
        self._ready = asyncio.Event()

    def put_nowait(self, item: InputT) -> None:
        self.items.append(item)
        self._ready.set()

    def get_nowait(self) -> InputT:
        if not self.items:
            raise asyncio.QueueEmpty
        item = self.items.pop(0)
        if not self.items:
            self._ready.clear()
        return item

    async def get(self) -> InputT:
        while not self.items:
            self._ready.clear()
            await self._ready.wait()
        return self.get_nowait()

    def take_all(self) -> list[InputT]:
        taken = list(self.items)
        self.items.clear()
        self._ready.clear()
        return taken

    def empty(self) -> bool:
        return not self.items

    def qsize(self) -> int:
        return len(self.items)


class TurnController(Generic[InputT]):
    def __init__(self, *, inputs: TurnQueue[InputT] | None = None) -> None:
        self.inputs: TurnQueue[InputT] = inputs if inputs is not None else TurnQueue()
        self.owner: object | None = None
        self.execution_task: asyncio.Task[Any] | None = None
        self.draining_tasks: set[asyncio.Task[Any]] = set()
        self.stop_pending = False
        self.closed = False
        self.watchdog: asyncio.TimerHandle | None = None
        self.on_finished: Callable[[asyncio.Task[Any], bool], None] | None = None
        self._tasks: set[asyncio.Task[Any]] = set()
        self._execution_owner: object | None = None
        self._active_run: object | None = None
        self._run_owner: object | None = None

    @property
    def busy(self) -> bool:
        return (
            self.owner is not None
            or bool(self.draining_tasks)
            or self._active_run is not None
            or any(not task.done() for task in self._tasks)
            or (self.execution_task is not None and not self.execution_task.done())
        )

    def track(self, task: asyncio.Task[Any] | None = None) -> None:
        task = task or asyncio.current_task()
        if task is None or self.execution_task is task:
            return
        previous = self.execution_task
        if previous is not None and not previous.done() and self.owner is self._execution_owner:
            raise TurnBusyError("The previous execution is still owned.")
        self.execution_task = task
        self._execution_owner = self.owner
        if task not in self._tasks:
            self._tasks.add(task)
            task.add_done_callback(self._execution_finished)

    def release_task(self, task: asyncio.Task[Any]) -> bool:
        if not task.cancelled():
            task.exception()
        owns_stop = self.owner is task or (
            self.execution_task is task and (self.owner is None or self.owner is self._execution_owner)
        )
        if self.execution_task is task:
            self.execution_task = None
            self._execution_owner = None
        self._tasks.discard(task)
        if self.owner is task:
            self.owner = None
        if owns_stop:
            # Frontends may release their opaque owner before the task finishes.
            # Clear that execution's Stop state, never a successor's request.
            self.stop_pending = False
            self.disarm_watchdog()
        draining = task in self.draining_tasks
        self.draining_tasks.discard(task)
        return draining

    def _execution_finished(self, task: asyncio.Task[Any]) -> None:
        draining = self.release_task(task)
        if self.on_finished is not None:
            self.on_finished(task, draining)

    def start(self, execute: Callable[[], Awaitable[ResultT]], *, name: str | None = None) -> asyncio.Task[ResultT]:
        if self.closed or self.busy:
            raise TurnBusyError("The previous execution is still owned or admission is closed.")

        async def published_execution() -> ResultT:
            # An eager factory must not enter the body before the task is published.
            await asyncio.sleep(0)
            return await execute()

        task = asyncio.create_task(published_execution(), name=name)
        self.owner = task
        self.track(task)
        return task

    def request_stop(
        self,
        *,
        cancel: Callable[[], None] | None = None,
        force_after: float | None = None,
        on_escalate: Callable[[], None] | None = None,
    ) -> bool:
        if self.stop_pending:
            return False
        self.stop_pending = True
        if cancel is not None:
            cancel()
        elif self.execution_task is not None and not self.execution_task.done():
            self.execution_task.cancel()
        if force_after is not None and on_escalate is not None:
            self.arm_watchdog(force_after, on_escalate)
        return True

    def arm_watchdog(self, delay: float, escalate: Callable[[], None]) -> None:
        self.disarm_watchdog()
        owner = self.owner

        def fire() -> None:
            if self.watchdog is not handle:
                return
            self.watchdog = None
            if self.stop_pending and self.owner is owner:
                escalate()

        handle = asyncio.get_running_loop().call_later(max(0.0, delay), fire)
        self.watchdog = handle

    def disarm_watchdog(self) -> None:
        watchdog, self.watchdog = self.watchdog, None
        if watchdog is not None:
            watchdog.cancel()

    def detach(self, *, cancel: Callable[[], None] | None = None, cancel_execution: bool = True) -> None:
        self.owner = None
        self.stop_pending = False
        self.disarm_watchdog()
        task = self.execution_task
        if task is not None and not task.done():
            self.draining_tasks.add(task)
        if not cancel_execution:
            return
        if cancel is not None:
            cancel()
        elif task is not None and not task.done():
            task.cancel()

    def take_next(self) -> InputT:
        if self.closed or self.busy:
            raise asyncio.QueueEmpty
        return self.inputs.get_nowait()

    def take_inputs(self, *, mid_turn: bool = False) -> list[InputT]:
        if self.closed or self.draining_tasks:
            return []
        if self.busy and (
            not mid_turn or asyncio.current_task() is not self.execution_task or self.owner is not self._execution_owner
        ):
            return []
        return self.inputs.take_all()

    def close_admission(self) -> None:
        self.closed = True
        self.disarm_watchdog()

    async def wait_settled(self, *, timeout: float | None = None, extra: Iterable[asyncio.Future[Any]] = ()) -> bool:
        waiters = set(extra) | self.draining_tasks | self._tasks
        if self.execution_task is not None:
            waiters.add(self.execution_task)
        if not waiters:
            return True
        _, pending = await asyncio.wait(waiters, timeout=timeout)
        return not pending

    async def run(
        self,
        stream_factory: StreamFactory,
        *,
        hooks: TurnHooks,
        recoverable_errors: tuple[type[Exception], ...] = (),
        format_error: Callable[[Exception], str] = str,
        suppress_cancel: bool = False,
    ) -> TurnResult:
        if self.closed or self.draining_tasks:
            return TurnResult(TurnOutcome.CANCELLED)
        if self._active_run is not None and self._run_owner is self.owner:
            raise TurnBusyError("A stream is already running.")
        run = self._active_run = object()
        owner = self._run_owner = self.owner
        result = TurnResult(TurnOutcome.COMPLETED)

        def current() -> bool:
            return self._active_run is run and self.owner is owner

        async def live_stream() -> AsyncGenerator[Chunk, None]:
            source = stream_factory()
            try:
                async for chunk in source:
                    if current():
                        yield chunk
            finally:
                close = getattr(source, "aclose", None)
                if close is not None:
                    await close()

        try:
            for step in hooks.before:
                await step()
                if not current():
                    return result
            stream = live_stream()
            try:
                await hooks.consume(stream)
            finally:
                # Closing in this task preserves ContextVar and AnyIO scope affinity.
                await stream.aclose()
            if current():
                for step in hooks.after:
                    await step()
                    if not current():
                        break
        except asyncio.CancelledError as exc:
            result = TurnResult(TurnOutcome.CANCELLED, exception=exc)
            if current() and hooks.settle is not None:
                await hooks.settle(result)
            if not suppress_cancel:
                raise
        except Exception as exc:
            result = TurnResult(TurnOutcome.FAILED, format_error(exc), exc)
            if current() and hooks.settle is not None:
                await hooks.settle(result)
            if not isinstance(exc, recoverable_errors):
                raise
        finally:
            try:
                if current():
                    self.stop_pending = False
                    self.disarm_watchdog()
                    if hooks.teardown is not None:
                        hooks.teardown(result)
            finally:
                if self._active_run is run:
                    self._active_run = None
                    self._run_owner = None
        return result
