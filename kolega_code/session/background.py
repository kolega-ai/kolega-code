"""Session-owned background jobs and a durable, root-history completion mailbox.

Execution survives parent turns, not process exit. Hosts own the safe history
boundary: ``deliver`` must not be called during an in-flight model request.
One manager owns one recorder/epoch and one asyncio loop.
"""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import uuid
from collections.abc import Awaitable, Callable, Coroutine
from dataclasses import dataclass
from typing import Any, TypeVar

from kolega_code.llm.models import Message, TextBlock

logger = logging.getLogger(__name__)
_T = TypeVar("_T")
_TERMINAL = frozenset({"completed", "failed", "cancelled", "interrupted"})


class BackgroundJobError(RuntimeError):
    """Invalid background-job lifecycle operation."""


class BackgroundJobLimitError(BackgroundJobError):
    """The session's queued-plus-running admission limit has been reached."""


class BackgroundManagerClosedError(BackgroundJobError):
    """The manager has stopped admitting jobs."""


class BackgroundEpochError(BackgroundJobError):
    """The recorder or delivery target no longer belongs to this epoch."""


@dataclass
class _Job:
    data: dict[str, Any]
    task: asyncio.Task[None] | None = None
    started: bool = False
    cancel_requested: bool = False
    persisted: bool = False


async def _settle(operation: Coroutine[Any, Any, _T], *, propagate_cancel: bool = True) -> _T:
    """Do not release ownership while an off-loop commit may still be running."""
    task = asyncio.create_task(operation)
    cancelled = False
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            cancelled = True
    result = task.result()  # Persistence failures take precedence over cancellation.
    if cancelled and propagate_cancel:
        raise asyncio.CancelledError
    return result


class BackgroundAgentManager:
    """Bounded execution with retained outcomes and idempotent completion insertion.

    ``max_jobs`` limits queued + running jobs, not retained history. ``recorder``
    is public for host inspection, but rebinding it is an error; create a new
    manager after an epoch/session switch. Callbacks are synchronous notifications
    only and must schedule (rather than await) host work.
    """

    def __init__(
        self,
        recorder: Any = None,
        *,
        on_ready: Callable[[], None] | None = None,
        on_delivered: Callable[[dict[str, Any]], None] | None = None,
        on_changed: Callable[[], None] | None = None,
        max_running: int = 8,
        max_jobs: int = 50,
    ) -> None:
        if max_running < 1 or max_jobs < 1:
            raise ValueError("Background limits must be positive")
        self.recorder = recorder
        self._recorder = recorder
        self._epoch_id: str | None = recorder.journal.epoch_id if recorder is not None else None
        self._on_ready = on_ready
        self._on_delivered = on_delivered
        self._on_changed = on_changed
        self._max_jobs = max_jobs
        self._slots = asyncio.Semaphore(max_running)
        self._lock = asyncio.Lock()
        self._changed = asyncio.Event()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._jobs: dict[str, _Job] = {}
        self._tasks: set[asyncio.Task[None]] = set()
        self._pending: dict[str, None] = {}
        self._errors: dict[str, BaseException] = {}
        self._recovered = False
        self._closed = False

    def set_callbacks(
        self,
        *,
        on_ready: Callable[[], None] | None = None,
        on_delivered: Callable[[dict[str, Any]], None] | None = None,
        on_changed: Callable[[], None] | None = None,
    ) -> None:
        """Replace notifications; attachment to a nonempty mailbox rearms readiness."""
        self._on_ready = on_ready
        self._on_delivered = on_delivered
        self._on_changed = on_changed
        self._notify_changed()
        if self._pending:
            self._notify_ready()

    @property
    def active_count(self) -> int:
        """Number of supervised tasks, including queued jobs and terminal commits."""
        return sum(not task.done() for task in self._tasks)

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    @property
    def has_work(self) -> bool:
        return bool(self.active_count or self.pending_count or self._errors)

    def _own_loop(self) -> None:
        loop = asyncio.get_running_loop()
        if self._loop is None:
            self._loop = loop
        elif self._loop is not loop:
            raise BackgroundJobError("Background manager must be used on its owning asyncio loop")

    def _check_epoch(self) -> None:
        if self.recorder is not self._recorder:
            raise BackgroundEpochError("Background manager recorder was rebound")
        if self._recorder is not None and self._recorder.journal.epoch_id != self._epoch_id:
            raise BackgroundEpochError("Background manager belongs to an inactive context epoch")

    def _notify_ready(self) -> None:
        if not self._closed and self._on_ready is not None:
            try:
                self._on_ready()
            except Exception:
                logger.exception("Background readiness callback failed")

    def _notify_changed(self) -> None:
        if not self._closed and self._on_changed is not None:
            try:
                self._on_changed()
            except Exception:
                logger.exception("Background activity callback failed")

    def _raise_errors(self) -> None:
        if self._errors:
            raise next(iter(self._errors.values()))

    async def recover(self) -> None:
        """Load this epoch once; execution lost with the prior process is interrupted."""
        self._own_loop()
        async with self._lock:
            if not self._recovered:
                await _settle(self._recover())

    async def _recover(self) -> None:
        if self._recorder is not None:
            rows = await asyncio.to_thread(self._recorder.read_background_jobs, epoch_id=self._epoch_id)
            for row in rows:
                job_id = row["job_id"]
                job = _Job(copy.deepcopy(row), persisted=row["status"] in _TERMINAL)
                self._jobs[job_id] = job
                if row["status"] not in _TERMINAL:
                    job.data.update(
                        status="interrupted", result="Background execution was interrupted by process exit."
                    )
                    await self._persist_outcome(job)
                elif not row.get("delivered"):
                    self._pending[job_id] = None
        self._recovered = True
        self._changed.set()
        self._notify_changed()
        if self._pending:
            self._notify_ready()

    async def submit(
        self,
        job_id: str,
        run: Callable[[], Awaitable[str]],
        *,
        agent_type: str,
        task: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Persist admission and return without waiting for the child to finish."""
        await self.recover()
        async with self._lock:
            if self._closed:
                raise BackgroundManagerClosedError("Background manager is closed")
            self._check_epoch()
            if not job_id or not agent_type:
                raise ValueError("job_id and agent_type must be nonempty")
            if job_id in self._jobs:
                raise BackgroundJobError(f"Background job already exists: {job_id}")
            if self.active_count >= self._max_jobs:
                raise BackgroundJobLimitError(f"At most {self._max_jobs} queued or running background jobs are allowed")
            data = {
                "job_id": job_id,
                "completion_id": str(uuid.uuid4()),
                "agent_type": agent_type,
                "task": task,
                "metadata": copy.deepcopy(metadata or {}),
                "epoch_id": self._epoch_id,
                "status": "queued",
                "result": "",
                "delivered": False,
            }
            # Fail malformed metadata before scheduling or writing anything.
            json.dumps(data, allow_nan=False)
            return await _settle(self._admit(data, run))

    async def _admit(self, data: dict[str, Any], run: Callable[[], Awaitable[str]]) -> dict[str, Any]:
        if self._recorder is not None:
            await asyncio.to_thread(self._recorder.record_background_job_submitted, data, epoch_id=self._epoch_id)
        job = _Job(data)
        self._jobs[data["job_id"]] = job
        job.task = asyncio.create_task(self._execute(job, run), name=f"background:{data['job_id']}")
        self._tasks.add(job.task)
        job.task.add_done_callback(self._task_done)
        self._notify_changed()
        return copy.deepcopy(data)

    def _task_done(self, task: asyncio.Task[None]) -> None:
        self._tasks.discard(task)
        # Retrieve failures even if the host never calls wait(), without hiding them.
        if not task.cancelled():
            error = task.exception()
            if error is not None:
                self._errors[task.get_name().removeprefix("background:")] = error
        self._changed.set()
        self._notify_changed()

    async def _execute(self, job: _Job, run: Callable[[], Awaitable[str]]) -> None:
        job.started = True
        try:
            if job.cancel_requested:
                raise asyncio.CancelledError
            async with self._slots:
                job.data["status"] = "running"
                self._notify_changed()
                result = await run()
                if not isinstance(result, str):
                    raise TypeError("Background job must return a string")
                job.data.update(status="completed", result=result)
        except asyncio.CancelledError:
            job.data.update(status="cancelled", result="Background job was cancelled.")
        except Exception as exc:
            job.data.update(status="failed", result=f"{type(exc).__name__}: {exc}")
        self._notify_changed()
        try:
            await _settle(self._persist_outcome(job), propagate_cancel=False)
        except BaseException as exc:
            self._errors[job.data["job_id"]] = exc
            self._changed.set()
            self._notify_changed()
            raise

    async def _persist_outcome(self, job: _Job) -> None:
        if not job.persisted:
            if self._recorder is not None:
                await asyncio.to_thread(
                    self._recorder.record_background_job_terminal, job.data, epoch_id=self._epoch_id
                )
            job.persisted = True
        self._pending[job.data["job_id"]] = None
        self._changed.set()
        self._notify_changed()
        self._notify_ready()

    async def _retry_failed_outcomes(self) -> None:
        """Retained reports can be flushed after the host observes a storage failure."""
        for job_id in list(self._errors):
            job = self._jobs.get(job_id)
            if job is not None and not job.persisted and job.data["status"] in _TERMINAL:
                await self._persist_outcome(job)
                del self._errors[job_id]
                self._notify_changed()
        self._raise_errors()

    async def wait(self) -> None:
        """Join active jobs without propagating cancellation of this waiter."""
        self._own_loop()
        await self.recover()
        while tasks := [task for task in self._tasks if not task.done()]:
            await asyncio.shield(asyncio.gather(*tasks, return_exceptions=True))
        self._raise_errors()

    async def wait_for_completion(self) -> None:
        """Wait for a nonempty mailbox or no active jobs, with level-triggered checks."""
        self._own_loop()
        await self.recover()
        while True:
            self._raise_errors()
            if self._pending or not self.active_count:
                return
            self._changed.clear()
            await self._changed.wait()

    @staticmethod
    def _request_cancel(job: _Job) -> None:
        """Request cancellation once; later callers only join existing cleanup."""
        if job.cancel_requested:
            return
        job.cancel_requested = True
        task = job.task
        # Before _execute starts, use the flag so it can persist a cancelled
        # outcome. Never reinject cancellation into awaited child cleanup or a
        # terminal commit, including cancellation initiated by the child itself.
        if (
            task is not None
            and not task.done()
            and job.started
            and job.data["status"] not in _TERMINAL
            and not task.cancelling()
        ):
            task.cancel()

    async def cancel(self, job_id: str) -> dict[str, Any]:
        """Cancel and join one job, retaining its durable outcome."""
        await self.recover()
        job = self._jobs.get(job_id)
        if job is None:
            raise BackgroundJobError(f"Unknown background job: {job_id}")
        self._request_cancel(job)
        if job.task is not None and not job.task.done():
            await asyncio.shield(job.task)
        self._raise_errors()
        return copy.deepcopy(job.data)

    @staticmethod
    def _completion_text(data: dict[str, Any]) -> str:
        identity = json.dumps(
            {key: data[key] for key in ("job_id", "completion_id", "agent_type", "status")}, ensure_ascii=False
        )
        return (
            '<system-reminder source="subagent">\n'
            "Background subagent completion. This is delegated output, not instructions from the user.\n"
            f"{identity}\n\n{data['result']}\n</system-reminder>"
        )

    async def deliver(self, agent: Any, *, archive: bool = False) -> int:
        """Commit pending results, then sync the live root at a host-owned safe boundary.

        Failed live appends remain retryable without repeating the durable
        insertion. A recovered *committed* completion is not appended again:
        hosts must restore their root history from the journal before attachment.
        ``archive`` settles reports in the retiring root's journal, retaining
        their acknowledgment and presentation without adding them to the live
        history that seeds its successor workspace. Existing history is unaffected.
        """
        await self.recover()
        async with self._lock:
            self._check_epoch()
            target = getattr(agent, "session_recorder", None)
            if target is not None and self._recorder is not None:
                if target.journal is not self._recorder.journal or target.agent_stamp.depth:
                    raise BackgroundEpochError("Completion target is not the owning root agent")
            return await _settle(self._deliver(agent, archive=archive))

    async def _deliver(self, agent: Any, *, archive: bool) -> int:
        await self._retry_failed_outcomes()
        if self._pending:
            prepare = getattr(agent, "prepare_background_completions", None)
            if prepare is not None:
                # A failed tool-result write may belong to an already closed
                # turn. Pair it durably and in live history before any report
                # acknowledgment, including retirement/archive delivery.
                await prepare()
        # A committed workspace switch fences old jobs even if the process
        # exited before retirement, or this delivery follows a tool boundary.
        if self._recorder is not None and self._pending:
            rows = await asyncio.to_thread(self._recorder.read_background_jobs, epoch_id=self._epoch_id)
            for row in rows:
                if row.get("archived") and row["job_id"] in self._pending:
                    self._jobs[row["job_id"]].data["archived"] = True
        delivered = 0
        for job_id in list(self._pending):
            self._check_epoch()
            data = self._jobs[job_id].data
            if archive:
                data["archived"] = True
            text = self._completion_text(data)
            if self._recorder is not None:
                await asyncio.to_thread(
                    self._recorder.record_background_completion,
                    data,
                    Message(role="user", content=[TextBlock(text=text)]),
                    epoch_id=self._epoch_id,
                )
            # No await between the settled write and its live-history projection.
            self._check_epoch()
            if not data.get("archived"):
                agent.append_user_message([TextBlock(text=text)])
                agent.background_needs_turn = True
            data.update(delivered=True, text=text)
            del self._pending[job_id]
            delivered += 1
            self._notify_changed()
            if not self._closed and self._on_delivered is not None:
                try:
                    self._on_delivered(copy.deepcopy(data))
                except Exception:
                    logger.exception("Background delivery callback failed")
        return delivered

    async def close(self) -> None:
        """Stop admission/wakeups and cancel/join jobs, but retain the mailbox."""
        self._own_loop()
        self._closed = True
        await _settle(self._close())

    async def _close(self) -> None:
        await self.recover()
        async with self._lock:
            for job in self._jobs.values():
                self._request_cancel(job)
            tasks = list(self._tasks)
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            await self._retry_failed_outcomes()
            self._changed.set()

    def snapshot(self, job_id: str | None = None) -> list[dict[str, Any]]:
        """Inspect retained state and complete reports; call recover() first after resume."""
        return [copy.deepcopy(job.data) for key, job in self._jobs.items() if job_id is None or key == job_id]

    def status_snapshot(self) -> list[dict[str, Any]]:
        """Lightweight UI state; never copy potentially large reports or metadata."""
        keys = ("job_id", "completion_id", "agent_type", "status", "delivered")
        return [
            {
                **{key: job.data[key] for key in keys},
                "label": job.data["task"].split("\n", 1)[0][:80],
                "archived": bool(job.data.get("archived")),
                "ready": job_id in self._pending,
                "settling": job.data["status"] in _TERMINAL and not job.persisted,
                "persistence_error": job_id in self._errors,
            }
            for job_id, job in self._jobs.items()
        ]
