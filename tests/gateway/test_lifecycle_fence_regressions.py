from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio

from kolega_code.cli.session_event_store import FileSessionEventStore
from kolega_code.gateway import sessions
from kolega_code.gateway.adapters.base import ChatRef, InboundMessage
from kolega_code.gateway.daemon import GatewayShutdownIncomplete
from kolega_code.gateway.sessions import AgentTurnHandler, ChatLifecycleIncomplete, ChatSession
from kolega_code.llm.models import Message
from kolega_code.llm.usage import normalize_usage

from .test_sessions import RecordingAdapter, inbound, wait_for, wait_for_turn_done
from .test_shared_composition import CHAT, _RecorderAwareAgent
from .test_shared_composition import gateway as gateway

pytestmark = pytest.mark.usefixtures("isolated_cli_env")


@pytest_asyncio.fixture(autouse=True, params=[False, True], ids=["lazy", "eager"])
async def lifecycle_scheduling(request: pytest.FixtureRequest) -> AsyncIterator[None]:
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


def background_jobs(deliver: Any) -> SimpleNamespace:
    return SimpleNamespace(
        deliver=AsyncMock(side_effect=deliver),
        active_count=0,
        pending_count=0,
        has_work=False,
        close=AsyncMock(),
        set_callbacks=lambda **_kwargs: None,
    )


@pytest.mark.asyncio
async def test_admitted_messages_survive_model_fence_without_spinning(
    gateway: AgentTurnHandler, monkeypatch: pytest.MonkeyPatch
) -> None:
    entry = await gateway._ensure_session(CHAT)
    entry.auto_continue_paused = True
    entered, release, worker_release, worker_entered = (asyncio.Event() for _ in range(4))
    original_worker = gateway._worker
    ensure = AsyncMock(wraps=gateway._ensure_session)

    async def worker(chat_ref: ChatRef) -> None:
        await worker_release.wait()
        worker_entered.set()
        await original_worker(chat_ref)

    async def cleanup() -> None:
        entered.set()
        await release.wait()

    monkeypatch.setattr(gateway, "_worker", worker)
    monkeypatch.setattr(gateway, "_ensure_session", ensure)
    monkeypatch.setattr(entry.runtime.agent, "cleanup", cleanup)
    await gateway.handle(CHAT, inbound("first admitted message"))
    await gateway.handle(CHAT, inbound("second admitted message"))
    queue = gateway._queues[CHAT.key]
    queued_worker = gateway._workers[CHAT.key]
    rebuild = asyncio.create_task(gateway._apply_model(CHAT, entry, "anthropic/claude-haiku-4-5-20251001"))
    try:
        await asyncio.wait_for(entered.wait(), 10)
        owner = gateway._operations[CHAT.key]
        ensures = ensure.await_count
        worker_release.set()
        await asyncio.wait_for(worker_entered.wait(), 10)
        for _ in range(10):
            await asyncio.sleep(0)
        assert ensure.await_count == ensures
        assert not queued_worker.done()
        assert gateway._queues[CHAT.key] is queue
        assert entry.auto_continue_paused
        with pytest.raises(ChatLifecycleIncomplete):
            await gateway.handle(CHAT, inbound("not admitted"))
        release.set()
        await asyncio.wait_for(rebuild, 10)
        assert owner.released.is_set()
        adapter = cast(RecordingAdapter, gateway._adapter)
        assert await wait_for(lambda: adapter.sent.count(("42", "done")) == 2)
        await wait_for_turn_done(gateway, CHAT)
        agent = entry.runtime.agent
        assert agent is not None
        texts = [block.text for message in agent.history for block in message.content]
        assert texts[-4:] == ["first admitted message", "done", "second admitted message", "done"]
        assert not entry.auto_continue_paused
        assert gateway._workers[CHAT.key] is queued_worker
        assert not queued_worker.done()
    finally:
        release.set()
        worker_release.set()
        await asyncio.gather(rebuild, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("blocked_at", ["task_entry", "turn_lock"])
@pytest.mark.parametrize("background", [False, True], ids=["normal", "background"])
async def test_admitted_turn_rechecks_fence_after_scheduling_and_lock_acquisition(
    gateway: AgentTurnHandler, monkeypatch: pytest.MonkeyPatch, blocked_at: str, background: bool
) -> None:
    entry = await gateway._ensure_session(CHAT)
    entered, release, turn_entered, turn_release = (asyncio.Event() for _ in range(4))
    original_turn = gateway._run_owned_turn

    async def turn(payload: ChatSession, chat_ref: ChatRef, message: InboundMessage | None) -> None:
        turn_entered.set()
        if blocked_at == "task_entry":
            await turn_release.wait()
        await original_turn(payload, chat_ref, message)

    async def cleanup(_owner: Any) -> None:
        entered.set()
        await release.wait()

    async def deliver(_agent: Any) -> None:
        entry.completions.append({"text": "admitted report"})

    monkeypatch.setattr(gateway, "_run_owned_turn", turn)
    lock_held = False
    if blocked_at == "turn_lock":
        await entry.turn_lock.acquire()
        lock_held = True
    try:
        if background:
            entry.auto_continue_paused = True
            entry.background_jobs = background_jobs(deliver)
            gateway._wake_background(entry)
        else:
            await gateway.handle(CHAT, inbound("admitted before fence"))
        await asyncio.wait_for(turn_entered.wait(), 10)
        owner = gateway._start_operation(CHAT, "retire", cleanup, cleaning=True)
        await asyncio.wait_for(entered.wait(), 10)
        if blocked_at == "turn_lock":
            entry.turn_lock.release()
            lock_held = False
        turn_release.set()
        for _ in range(10):
            await asyncio.sleep(0)
        assert entry.turn_task is not None and not entry.turn_task.done()
        adapter = cast(RecordingAdapter, gateway._adapter)
        reply = ("42", "admitted report" if background else "done")
        assert reply not in adapter.sent
        release.set()
        assert await wait_for(lambda: owner.released.is_set())
        assert await wait_for(lambda: reply in adapter.sent)
        await wait_for_turn_done(gateway, CHAT)
        assert not entry.background_queued
    finally:
        if lock_held:
            entry.turn_lock.release()
        release.set()
        turn_release.set()


@pytest.mark.asyncio
@pytest.mark.parametrize("cleanup_fails", [False, True])
async def test_queued_background_wake_survives_retirement_timeout_and_failed_fence(
    gateway: AgentTurnHandler, monkeypatch: pytest.MonkeyPatch, cleanup_fails: bool
) -> None:
    entered, release = asyncio.Event(), asyncio.Event()
    entry = await gateway._ensure_session(CHAT)
    monkeypatch.setattr(sessions, "LIFECYCLE_WAIT_SECONDS", 0.01)
    presentation = entry.usage_presentation
    assert presentation is not None
    entry.auto_continue_paused = True
    calls = 0

    async def cleanup() -> None:
        entered.set()
        await release.wait()
        if cleanup_fails:
            raise RuntimeError("retired cleanup failed")

    async def deliver(_agent: Any) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            entry.completions.append({"text": "first report"})
            gateway._wake_background(entry)
        elif calls == 2:
            entry.completions.append({"text": "second report"})

    retired = SimpleNamespace(cleanup=AsyncMock(side_effect=cleanup))
    entry.retired_agents.append(retired)
    jobs = background_jobs(deliver)
    entry.background_jobs = jobs
    gateway._wake_background(entry)
    worker = gateway._workers[CHAT.key]
    try:
        await asyncio.wait_for(entered.wait(), 10)
        owner = gateway._operations[CHAT.key]
        assert await wait_for(lambda: entry.turn_task is None)
        for _ in range(10):
            await asyncio.sleep(0)
        assert not worker.done()
        assert entry.background_queued
        assert calls == 1
        assert gateway._workers[CHAT.key] is worker
        assert entry.usage_presentation is presentation
        assert not presentation._closing and not presentation._closed and presentation._unsubscribe is not None
        release.set()
        assert owner.task is not None
        await asyncio.wait({owner.task}, timeout=10)
        assert owner.task.done()
        if cleanup_fails:
            assert owner.cleanup_failed
            assert not owner.released.is_set()
            with pytest.raises(ChatLifecycleIncomplete):
                await gateway.handle(CHAT, inbound("still fenced"))
            gateway._wake_background(entry)
            for _ in range(10):
                await asyncio.sleep(0)
            assert calls == 1 and not worker.done()
            gateway.begin_shutdown()
            with pytest.raises(GatewayShutdownIncomplete):
                await asyncio.wait_for(gateway.wait_closed(), 10)
            assert worker.cancelled()
            assert retired.cleanup.await_count == 1
            assert gateway._operations[CHAT.key] is owner
            assert not presentation._closing and not presentation._closed and presentation._unsubscribe is not None
        else:
            adapter = cast(RecordingAdapter, gateway._adapter)
            assert await wait_for(lambda: ("42", "second report") in adapter.sent)
            await wait_for_turn_done(gateway, CHAT)
            assert calls == 2
            assert not entry.background_queued
            assert not entry.retired_agents
            assert owner.released.is_set()
            assert retired.cleanup.await_count == 1
            assert gateway._workers[CHAT.key] is worker
            assert entry.usage_presentation is presentation
            assert not presentation._closing and not presentation._closed and presentation._unsubscribe is not None
    finally:
        release.set()


@pytest.mark.asyncio
async def test_shutdown_cancels_fence_waiter_without_interrupting_cleanup(
    gateway: AgentTurnHandler, monkeypatch: pytest.MonkeyPatch
) -> None:
    entered, release = asyncio.Event(), asyncio.Event()
    entry = await gateway._ensure_session(CHAT)
    monkeypatch.setattr(sessions, "LIFECYCLE_WAIT_SECONDS", 0.01)
    monkeypatch.setattr(sessions, "SHUTDOWN_WAIT_SECONDS", 0.01)
    presentation = entry.usage_presentation
    assert presentation is not None

    async def cleanup() -> None:
        entered.set()
        await release.wait()

    async def deliver(_agent: Any) -> None:
        gateway._wake_background(entry)

    retired = SimpleNamespace(cleanup=AsyncMock(side_effect=cleanup))
    entry.retired_agents.append(retired)
    entry.background_jobs = background_jobs(deliver)
    gateway._wake_background(entry)
    worker = gateway._workers[CHAT.key]
    try:
        await asyncio.wait_for(entered.wait(), 10)
        owner = gateway._operations[CHAT.key]
        assert await wait_for(lambda: entry.turn_task is None)
        assert not worker.done()
        with pytest.raises(GatewayShutdownIncomplete):
            await gateway.shutdown()
        assert worker.cancelled()
        assert owner.task is not None and not owner.task.done()
        assert not owner.task.cancelling()
        assert owner.cleaning
        assert not presentation._closing and not presentation._closed and presentation._unsubscribe is not None
        release.set()
        await asyncio.wait_for(gateway.wait_closed(), 10)
        assert retired.cleanup.await_count == 1
        assert not gateway._operations
        assert not gateway._workers
        assert not gateway._queues
        assert presentation._closed and presentation._unsubscribe is None
    finally:
        release.set()


@pytest.mark.asyncio
async def test_cancelled_startup_keeps_usage_publisher_until_partial_agent_cleanup_finishes(
    gateway: AgentTurnHandler, monkeypatch: pytest.MonkeyPatch
) -> None:
    initializing, cleaning, release = (asyncio.Event() for _ in range(3))
    agents: list[_RecorderAwareAgent] = []

    async def initialize() -> list[str]:
        initializing.set()
        await asyncio.Future()
        return []

    class Agent(_RecorderAwareAgent):
        def __init__(self, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            agents.append(self)
            self.tool_collection.initialize.side_effect = initialize

        async def cleanup(self) -> None:
            self.cleanup_calls += 1
            cleaning.set()
            await release.wait()
            assert self.session_recorder is not None
            self.session_recorder.start_turn(Message(role="user", content="late cleanup evidence"))
            usage = normalize_usage({"input_tokens": 3, "output_tokens": 1}, "anthropic", "usage-test")
            self.session_recorder.record_assistant(Message(role="assistant", content="cleanup response", usage=usage))
            self.session_recorder.finish_turn("completed")

    monkeypatch.setattr(sessions, "CoderAgent", Agent)
    request = asyncio.create_task(gateway._ensure_session(CHAT))
    try:
        await asyncio.wait_for(initializing.wait(), 10)
        owner = gateway._operations[CHAT.key]
        presentation = owner.usage_presentation
        assert presentation is not None
        request.cancel()
        with pytest.raises(asyncio.CancelledError):
            await request
        await asyncio.wait_for(cleaning.wait(), 10)
        assert owner.task is not None and not owner.task.done()
        assert owner.cleaning and owner.partial_agent is agents[0]
        assert not presentation._closing and not presentation._closed and presentation._unsubscribe is not None
        cancellations = owner.task.cancelling()
        for _ in range(2):
            await gateway.handle(CHAT, inbound("/stop"))
        assert owner.task.cancelling() == cancellations
        with pytest.raises(ChatLifecycleIncomplete):
            await gateway._ensure_session(CHAT)
        assert gateway._registry.get(CHAT) is None
        release.set()
        done, _ = await asyncio.wait({owner.task}, timeout=10)
        assert done == {owner.task}
        assert owner.task.cancelled()
        assert agents[0].cleanup_calls == 1
        assert presentation._closed and presentation._unsubscribe is None
        assert presentation._worker is not None and presentation._worker.done()
        assert owner.usage_presentation is None
        assert owner.relay is not None and owner.relay._pump_task is None
        assert owner.router is not None and owner.router._task is None
        assert owner.released.is_set()
        assert not gateway._operations
        journal = gateway._store.journal(gateway._chat_sessions[CHAT.key])
        events = await FileSessionEventStore(journal).read(journal.session_id)
        usage = [event for event in events if event.event_type == "usage_updated"][-1].content["usage"]
        assert usage["requests"] == 1
        assert usage["total_tokens"] == 4
    finally:
        release.set()
        await asyncio.gather(request, return_exceptions=True)


@pytest.mark.asyncio
async def test_stop_during_protected_retirement_pauses_late_background_continuation(
    gateway: AgentTurnHandler, monkeypatch: pytest.MonkeyPatch
) -> None:
    entered, release, continued = (asyncio.Event() for _ in range(3))
    entry = await gateway._ensure_session(CHAT)

    async def cleanup() -> None:
        entered.set()
        await release.wait()

    async def continuation() -> AsyncIterator[dict[str, Any]]:
        continued.set()
        yield {"type": "response", "content": "automatic continuation", "complete": True}

    async def deliver(_agent: Any) -> None:
        if jobs.pending_count:
            entry.completions.append({"text": "late completion"})
            jobs.pending_count = 0
            jobs.has_work = False

    jobs = background_jobs(deliver)
    entry.background_jobs = jobs
    retired = SimpleNamespace(cleanup=AsyncMock(side_effect=cleanup))
    entry.retired_agents.append(retired)
    monkeypatch.setattr(entry.runtime.agent, "continue_from_history_stream", continuation, raising=False)
    gateway._wake_background(entry)
    try:
        await asyncio.wait_for(entered.wait(), 10)
        owner = gateway._operations[CHAT.key]
        assert owner.task is not None
        cancellations = owner.task.cancelling()
        jobs.pending_count = 1
        jobs.has_work = True
        gateway._wake_background(entry)
        await gateway.handle(CHAT, inbound("/stop"))
        assert entry.auto_continue_paused
        assert owner.cleaning and not owner.stopping
        assert owner.task.cancelling() == cancellations
        assert not release.is_set()
        release.set()
        adapter = cast(RecordingAdapter, gateway._adapter)
        assert await wait_for(lambda: ("42", "late completion") in adapter.sent)
        await wait_for_turn_done(gateway, CHAT)
        assert entry.auto_continue_paused
        assert not continued.is_set()
        assert retired.cleanup.await_count == 1
        assert not entry.background_queued
    finally:
        release.set()
