"""Headless job lifetime and schedule wakeups, without network or wall-clock sleeps."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from kolega_code.cli import loop as loop_module
from kolega_code.cli.loop import LoopState, parse_schedule_text
from kolega_code.cli.main import (
    _AskBackgroundJobs,
    _run_ask_loop_iterations,
    _run_worker_modes,
    _sleep_until_loop_fire,
)
from kolega_code.session.background import BackgroundAgentManager
from tests.background_job_helpers import HostAgent, Jobs
from tests.cli.test_ask_loop import LoopAskFakeAgent


async def consume(stream: Any) -> None:
    async for _ in stream:
        pass


@pytest.mark.asyncio
async def test_plain_ask_waits_for_future_completion_and_continues_without_user_message() -> None:
    agent = HostAgent()
    jobs = _AskBackgroundJobs(agent)
    await jobs.recover()
    gate = asyncio.Event()
    agent.background_jobs.launch(gate)

    async def drain_peers() -> None:
        await jobs.drain(consume)

    invocation = asyncio.create_task(
        _run_worker_modes(
            agent=agent,  # type: ignore[arg-type]
            loop_state=None,
            goal_state=None,
            prompt="start",
            attachments=[],
            session_recorder=MagicMock(),
            json_mode=True,
            consume_turn=consume,
            drain_tokens=lambda state: None,
            drain_peer_turns=drain_peers,
            background_jobs=jobs,
        )
    )
    await asyncio.wait_for(agent.background_jobs.waiting.wait(), 1)
    assert not invocation.done()
    gate.set()
    assert await asyncio.wait_for(invocation, 1) == 0
    assert agent.continuations == 1
    assert agent.messages == []
    assert agent.history[0]["result"] == "child report"
    await agent.cleanup()


@pytest.mark.asyncio
async def test_real_supervisor_completion_is_inserted_once_before_headless_continuation() -> None:
    manager = BackgroundAgentManager()
    history: list[str] = []
    turns: list[bool] = []
    gate = asyncio.Event()
    started = asyncio.Event()

    async def child() -> str:
        started.set()
        await gate.wait()
        return "real child report"

    async def continuation() -> Any:
        assert len(history) == 1
        assert "real child report" in history[0]
        turns.append(True)
        yield {"type": "response", "content": "report received", "complete": True}

    def get_jobs() -> BackgroundAgentManager:
        if agent.background_jobs is None:
            agent.background_jobs = BackgroundAgentManager()
        return agent.background_jobs

    agent = SimpleNamespace(
        background_jobs=manager,
        session_recorder=None,
        get_background_jobs=get_jobs,
        append_user_message=lambda blocks: history.extend(block.text for block in blocks),
        continue_from_history_stream=continuation,
        clear_history=history.clear,
    )
    adapter = _AskBackgroundJobs(agent)
    await adapter.recover()
    await manager.submit("child-1", child, agent_type="investigation", task="find the result")
    invocation = asyncio.create_task(adapter.drain(consume, wait=True))
    try:
        await asyncio.wait_for(started.wait(), 1)
        assert not invocation.done()
        gate.set()
        await asyncio.wait_for(invocation, 1)
        await adapter.drain(consume, wait=True)
        assert turns == [True]
        assert len(history) == 1
        assert not manager.has_work
        assert manager.snapshot("child-1")[0]["delivered"]
        await adapter.reset_history()
        assert adapter.manager is not manager
        assert not history
        await adapter.manager.submit("child-2", child, agent_type="investigation", task="new epoch")
        await asyncio.wait_for(adapter.drain(consume, wait=True), 1)
        assert turns == [True, True]
        assert len(history) == 1
    finally:
        await adapter.manager.close()
        await manager.close()


@pytest.mark.asyncio
async def test_cancelled_ask_wait_does_not_restart_parent_or_cancel_child_implicitly() -> None:
    agent = HostAgent()
    adapter = _AskBackgroundJobs(agent)
    agent.background_jobs.launch(asyncio.Event())
    invocation = asyncio.create_task(adapter.drain(consume, wait=True))
    await asyncio.wait_for(agent.background_jobs.waiting.wait(), 1)
    invocation.cancel()
    with pytest.raises(asyncio.CancelledError):
        await invocation
    assert agent.background_jobs.active_count == 1
    await agent.cleanup()
    assert agent.continuations == 0
    assert agent.background_jobs.outcomes[0]["status"] == "cancelled"


@pytest.mark.asyncio
async def test_completion_wakes_loop_sleep_without_moving_scheduled_fire(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = {"now": datetime(2026, 10, 15, 12)}
    monkeypatch.setattr(loop_module, "now_local", lambda: clock["now"])
    state = LoopState.create(parse_schedule_text("5m"), "poll", max_iterations=2)
    state.mark_fired()
    state.advance_after_completion()
    deadline = state.next_fire_at
    agent = HostAgent()
    adapter = _AskBackgroundJobs(agent)
    gate = asyncio.Event()
    agent.background_jobs.launch(gate)

    async def consume_and_advance(stream: Any) -> None:
        await consume(stream)
        assert state.next_fire_at == deadline
        assert state.iterations == 1
        clock["now"] += timedelta(minutes=5)

    sleeper = asyncio.create_task(
        _sleep_until_loop_fire(state, MagicMock(), True, background_jobs=adapter, consume_turn=consume_and_advance)
    )
    gate.set()
    await asyncio.wait_for(sleeper, 1)
    assert agent.continuations == 1
    assert state.next_fire_at == deadline
    assert state.iterations == 1
    await agent.cleanup()


@pytest.mark.asyncio
async def test_full_ask_invocation_keeps_services_alive_until_job_continuation(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch, isolated_cli_env: Any
) -> None:
    from kolega_code.cli import main as main_module

    gate = asyncio.Event()
    constructed = asyncio.Event()
    instances: list[Any] = []

    class AskAgent(LoopAskFakeAgent):
        def __init__(self, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            self.background_jobs = Jobs()
            self.active_turns = 0
            self.continuations = 0
            instances.append(self)
            constructed.set()

        def get_background_jobs(self) -> Jobs:
            return self.background_jobs

        async def process_message_stream(self, message: str, attachments: Any = None) -> Any:
            self.background_jobs.launch(gate)
            async for chunk in super().process_message_stream(message, attachments):
                yield chunk

        async def continue_from_history_stream(self) -> Any:
            self.continuations += 1
            yield {"type": "response", "content": "used child report", "complete": True}

        async def cleanup(self) -> None:
            await self.background_jobs.close()
            await super().cleanup()

    monkeypatch.setattr(main_module, "CoderAgent", AskAgent)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("KOLEGA_CODE_PROVIDER", "anthropic")
    monkeypatch.setenv("KOLEGA_CODE_MODEL", "claude-opus-5")
    invocation = asyncio.create_task(
        main_module._run_ask(main_module.parse_args(["ask", "delegate", "--project", str(tmp_path)]))
    )
    await asyncio.wait_for(constructed.wait(), 2)
    agent = instances[0]
    await asyncio.wait_for(agent.background_jobs.waiting.wait(), 2)
    assert not invocation.done()
    assert agent.cleanup_calls == 0
    gate.set()
    assert await asyncio.wait_for(invocation, 2) == 0
    assert agent.continuations == 1
    assert agent.messages == ["delegate"]
    assert agent.cleanup_calls == 1


@pytest.mark.asyncio
async def test_fresh_iteration_settles_old_context_jobs_before_clearing_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = {"now": datetime(2026, 10, 15, 12)}
    monkeypatch.setattr(loop_module, "now_local", lambda: clock["now"])
    state = LoopState.create(parse_schedule_text("5m"), "poll", max_iterations=2, fresh=True)
    monkeypatch.setattr(state, "seconds_until", lambda: 0.0)
    state.mark_fired()
    clock["now"] += timedelta(minutes=5)
    agent = HostAgent()
    adapter = _AskBackgroundJobs(agent)
    gate = asyncio.Event()
    agent.background_jobs.launch(gate)
    cleared: list[bool] = []
    old_jobs = agent.background_jobs

    def clear_history() -> None:
        assert not old_jobs.has_work
        assert old_jobs.closed
        assert agent.continuations == 1
        cleared.append(True)

    agent.clear_history = clear_history  # type: ignore[attr-defined]

    async def drain_peers() -> None:
        await adapter.drain(consume)

    iteration = asyncio.create_task(
        _run_ask_loop_iterations(
            agent=agent,  # type: ignore[arg-type]
            loop_state=state,
            prompt="poll",
            attachments=[],
            session_recorder=MagicMock(),
            json_mode=True,
            consume_turn=consume,
            drain_tokens=lambda state: None,
            drain_peer_turns=drain_peers,
            background_jobs=adapter,
        )
    )
    await asyncio.wait_for(agent.background_jobs.waiting.wait(), 1)
    assert not cleared
    gate.set()
    await asyncio.wait_for(iteration, 1)
    assert cleared == [True]
    assert len(agent.messages) == 1
    await agent.cleanup()
