"""The gigacode workflow runtime: the primitives a script orchestrates with.

A :class:`WorkflowRuntime` binds together a ``dispatch`` callable (runs one
sub-agent), an ``emit`` callable (publishes progress), a :class:`RunJournal`
(artifacts + resume), and a :class:`Budget`. It exposes the five primitives
plus ``workflow()`` that the executor injects as globals.

Concurrency, the lifetime agent cap, and the per-call fan-out cap are enforced
here so a runaway script cannot exhaust the machine.
"""

from __future__ import annotations

import asyncio
import inspect
import os
import random
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Optional

from .accounting import (
    WorkflowRunAccounting,
    reset_current_agent_reservation,
    set_current_agent_reservation,
)
from .budget import Budget
from .errors import WorkflowScriptError
from .executor import DEFAULT_MAX_AGENT_DEPTH, run_script, safe_builtins
from .journal import RunJournal
from .types import AgentRunSpec, DispatchFn, EmitFn, WorkflowResolver

# Largest single parallel()/pipeline() fan-out, and the lifetime agent backstop.
MAX_FANOUT = 4096
DEFAULT_AGENT_CAP = 1000

# A batch admitted through the semaphore together would otherwise fire LLM requests at the
# same instant; a small random pre-dispatch delay de-synchronizes them so concurrent
# sub-agents don't collectively spike the account rate limit.
START_STAGGER_SECONDS = 0.75


def default_concurrency() -> int:
    """Concurrent-agent cap: a few below the core count, clamped to [1, 8].

    Kept modest so a fan-out doesn't burst enough simultaneous LLM requests to trip
    account-level rate limits; the jittered start-stagger further de-correlates them.
    """
    cpus = os.cpu_count() or 4
    return max(1, min(8, cpus - 2))


def _call_with_arity(fn: Callable[..., Any], *args: Any) -> Any:
    """Call ``fn`` passing only as many positional args as it accepts.

    Python (unlike JS) errors on surplus positional args, so a single-arg stage
    like ``lambda r: agent(...)`` would break if always handed
    ``(prevResult, originalItem, index)``. This trims the tuple to the callable's
    arity (passing all of it when the callable takes ``*args``).
    """
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return fn(*args)
    params = list(sig.parameters.values())
    if any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in params):
        return fn(*args)
    positional = [p for p in params if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
    return fn(*args[: len(positional)])


class WorkflowRuntime:
    def __init__(
        self,
        *,
        dispatch: DispatchFn,
        emit: EmitFn,
        journal: RunJournal,
        budget: Budget,
        concurrency: Optional[int] = None,
        agent_cap: int = DEFAULT_AGENT_CAP,
        accounting: Optional[WorkflowRunAccounting] = None,
        max_agent_depth: int = DEFAULT_MAX_AGENT_DEPTH,
        resume_cache: Optional[Dict[int, Any]] = None,
        workflow_resolver: Optional[WorkflowResolver] = None,
    ) -> None:
        self._dispatch = dispatch
        self._emit = emit
        self._journal = journal
        self._accounting = accounting or WorkflowRunAccounting(budget, agent_cap)
        if self._accounting.budget is not budget:
            raise ValueError("workflow accounting must own the runtime budget")
        self.budget = self._accounting.budget
        self._sem = asyncio.Semaphore(concurrency or default_concurrency())
        self._max_agent_depth = max_agent_depth
        self._resolver = workflow_resolver

        self._call_index = 0
        self._current_phase: Optional[str] = None
        self._nested_depth = 0
        self._resume_cache: Dict[int, Any] = resume_cache or {}
        self._pending_emits: "set[asyncio.Task]" = set()

    # ------------------------------------------------------------------ run
    async def execute(self, source: str, args: Any) -> Any:
        """Run the top-level script and drain any fire-and-forget progress emits."""
        namespace = self._build_namespace(args=args, nested=False)
        try:
            return await run_script(source, namespace)
        finally:
            if self._pending_emits:
                await asyncio.gather(*self._pending_emits, return_exceptions=True)

    def _build_namespace(self, *, args: Any, nested: bool) -> Dict[str, Any]:
        return {
            "__builtins__": safe_builtins(),
            "__name__": "__workflow__",
            "agent": self.agent,
            "parallel": self.parallel,
            "pipeline": self.pipeline,
            "phase": self.phase,
            "log": self.log,
            "workflow": self._nested_blocked if nested else self.workflow,
            "args": args,
            "budget": self.budget,
        }

    # ----------------------------------------------------------- primitives
    async def agent(
        self,
        prompt: str,
        *,
        label: Optional[str] = None,
        phase: Optional[str] = None,
        schema: Optional[dict] = None,
        model: Optional[str] = None,
        effort: Optional[str] = None,
        agent_type: Optional[str] = None,
    ) -> Any:
        """Dispatch one sub-agent. Returns its recap text, or the validated dict
        when ``schema`` is given, or ``None`` if the agent failed/was skipped.
        """
        if not isinstance(prompt, str) or not prompt.strip():
            raise WorkflowScriptError("agent() requires a non-empty prompt string")

        # Index is assigned synchronously (no await before this point), so it is
        # deterministic across runs for a given script — the basis for resume.
        index = self._call_index
        self._call_index += 1
        spec = AgentRunSpec(
            prompt=prompt,
            label=label,
            phase=phase or self._current_phase,
            schema=schema,
            model=model,
            effort=effort,
            agent_type=agent_type,
            max_agent_depth=self._max_agent_depth,
            call_index=index,
        )
        key = spec.cache_key()

        cached = self._resume_cache.get(index)
        if cached is not None and cached[0] == key:
            cached_value = cached[1]
            label = spec.label or prompt[:60]
            self._journal.append_transcript_event(
                {
                    "type": "agent_cached",
                    "index": index,
                    "label": label,
                    "phase": spec.phase,
                    "value": cached_value,
                }
            )
            self._emit_soon("workflow_agent_cached", {"label": label, "phase": spec.phase})
            return cached_value

        async with self._sem:
            # Stagger starts within the admitted batch so they don't hit the API in lockstep.
            if START_STAGGER_SECONDS:
                await asyncio.sleep(random.uniform(0, START_STAGGER_SECONDS))
            reservation = self._accounting.reserve_agent()
            result = None
            reservation_token = set_current_agent_reservation(reservation)
            try:
                result = await self._dispatch(spec)
            finally:
                reset_current_agent_reservation(reservation_token)
                reservation.report_total(result.tokens if result is not None else None)

        assert result is not None
        value = result.value
        self._journal.record(
            index,
            key,
            spec.label,
            value,
            status=result.status,
            phase=spec.phase,
            agent_type=spec.agent_type,
            max_agent_depth=spec.max_agent_depth,
            tokens=result.tokens,
            error=result.error,
            transcript_path=result.transcript_path,
            transcript_markdown_path=result.transcript_markdown_path,
        )
        self._journal.append_transcript_event(
            {
                "type": "agent_call",
                "index": index,
                "label": spec.label,
                "phase": spec.phase,
                "agent_type": spec.agent_type,
                "max_agent_depth": spec.max_agent_depth,
                "status": result.status,
                "tokens": result.tokens,
                "error": result.error,
                "value": value,
                "transcript_path": result.transcript_path,
                "transcript_markdown_path": result.transcript_markdown_path,
            }
        )
        return value

    async def parallel(self, thunks: Iterable[Callable[[], Awaitable[Any]]]) -> List[Any]:
        """Run thunks concurrently and wait for all (a barrier). A thunk that
        raises resolves to ``None`` — the call itself never rejects.
        """
        items = list(thunks)
        if len(items) > MAX_FANOUT:
            raise WorkflowScriptError(f"parallel() accepts at most {MAX_FANOUT} items, got {len(items)}")
        return await asyncio.gather(*[self._invoke(t) for t in items])

    async def pipeline(self, items: Iterable[Any], *stages: Callable[..., Any]) -> List[Any]:
        """Run each item through all stages independently — no barrier between
        stages. A stage that throws drops that item to ``None`` and skips its
        remaining stages. Each stage receives ``(prevResult, originalItem, index)``.
        """
        materialized = list(items)
        if len(materialized) > MAX_FANOUT:
            raise WorkflowScriptError(f"pipeline() accepts at most {MAX_FANOUT} items, got {len(materialized)}")

        async def chain(original: Any, index: int) -> Any:
            value: Any = original
            for stage in stages:
                try:
                    out = _call_with_arity(stage, value, original, index)
                    if inspect.isawaitable(out):
                        out = await out
                    value = out
                except Exception:
                    return None
            return value

        return await asyncio.gather(*[chain(item, i) for i, item in enumerate(materialized)])

    def phase(self, title: str) -> None:
        """Start a new phase; subsequent ``agent()`` calls group under it."""
        self._current_phase = str(title)
        self._emit_soon("workflow_phase", {"title": str(title)})

    def log(self, message: str) -> None:
        """Emit a narrator line above the progress tree."""
        self._emit_soon("workflow_log", {"message": str(message)})

    async def workflow(self, name_or_ref: Any, args: Any = None) -> Any:
        """Run another workflow inline as a sub-step (one level deep only).

        Shares this run's counters, budget, journal, and concurrency cap.
        """
        if self._resolver is None:
            raise WorkflowScriptError("nested workflows are unavailable in this run")
        source = self._resolver(name_or_ref)
        namespace = self._build_namespace(args=args, nested=True)
        self._nested_depth += 1
        try:
            return await run_script(source, namespace)
        finally:
            self._nested_depth -= 1

    async def _nested_blocked(self, *_args: Any, **_kwargs: Any) -> Any:
        raise WorkflowScriptError("workflow() cannot be nested more than one level deep")

    # ------------------------------------------------------------- helpers
    async def _invoke(self, thunk: Any) -> Any:
        """Await a thunk (zero-arg callable) or a bare awaitable; failures -> None."""
        try:
            result = thunk() if callable(thunk) else thunk
            if inspect.isawaitable(result):
                result = await result
            return result
        except Exception:
            return None

    def _emit_soon(self, event_type: str, content: dict) -> None:
        """Fire-and-forget a progress emit from a synchronous primitive.

        The task is tracked so it isn't garbage-collected mid-flight and is
        drained at the end of :meth:`execute`.
        """
        try:
            task = asyncio.ensure_future(self._emit(event_type, content))
        except RuntimeError:
            # No running loop (e.g. a unit test calling phase() directly) — skip.
            return
        self._pending_emits.add(task)
        task.add_done_callback(self._pending_emits.discard)
