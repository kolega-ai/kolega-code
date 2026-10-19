"""Gateway idle mailbox delivery shares the existing per-chat turn owner."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from kolega_code.agent.baseagent import BaseAgent
from kolega_code.gateway.adapters.telegram.formatting import chunk_text
from kolega_code.gateway.daemon import ERROR_REPLY
from kolega_code.gateway.redaction import scrub
from kolega_code.llm.models import Message, ToolCall, ToolResult
from tests.background_job_helpers import HostAgent
from tests.gateway.test_sessions import (
    RecordingAdapter,
    ScriptedLLM,
    ScriptedStream,
    all_sent_texts,
    chat_ref,
    inbound,
    make_handler,
    wait_for,
)


class GatedAdapter(RecordingAdapter):
    """Reject oversized sends; optionally block or fail a presentation attempt."""

    def __init__(self, *, block_on: int = 1, fail_on: int | None = None) -> None:
        super().__init__()
        self.block_on: int = block_on
        self.fail_on: int | None = fail_on
        self.blocked: asyncio.Event = asyncio.Event()
        self.release: asyncio.Event = asyncio.Event()
        self.failed: asyncio.Event = asyncio.Event()
        self.attempt_times: list[float] = []

    async def send_text(self, chat_id: str, text: str, *, reply_to_message_id: str | None = None) -> str:
        assert len(text) <= self.capabilities.text_chunk_limit
        if text != ERROR_REPLY:
            self.attempt_times.append(asyncio.get_running_loop().time())
            attempt = len(self.attempt_times)
            if attempt == self.fail_on:
                self.failed.set()
                raise OSError("temporary transport failure")
            if attempt == self.block_on:
                self.blocked.set()
                await self.release.wait()
        return await super().send_text(chat_id, text, reply_to_message_id=reply_to_message_id)


def queue_reports(agent: HostAgent, *texts: str) -> None:
    jobs = agent.get_background_jobs()
    for text in texts:
        jobs.pending.append({"job_id": f"job-{len(jobs.pending)}", "status": "completed", "text": text})
    assert jobs.on_ready is not None
    jobs.on_ready()


def host(
    tmp_path: Path, *, max_sessions: int = 4, adapter: RecordingAdapter | None = None
) -> tuple[Any, Any, list[HostAgent]]:
    handler, adapter, _ = make_handler(tmp_path, max_sessions=max_sessions, adapter=adapter)
    agents: list[HostAgent] = []

    async def builder(record: Any, config: Any, **kwargs: Any) -> HostAgent:
        agent = HostAgent()
        agents.append(agent)
        return agent

    handler._agent_builder = builder
    return handler, adapter, agents


@pytest.mark.asyncio
async def test_idle_completion_wakes_worker_and_continues_once(tmp_path: Path) -> None:
    handler, adapter, agents = host(tmp_path)
    entry = await handler._ensure_session(chat_ref())
    gate = asyncio.Event()
    agents[0].background_jobs.launch(gate)
    gate.set()
    try:
        await asyncio.wait_for(agents[0].continued.wait(), 2)
        assert await wait_for(lambda: entry.turn_task is None)
        assert "child report" in all_sent_texts(adapter)
        assert agents[0].continuations == 1
        assert agents[0].messages == []
        assert agents[0].max_active_turns == 1
    finally:
        await handler.shutdown()


@pytest.mark.asyncio
async def test_completion_waits_for_busy_parent_and_stop_prevents_relaunch(tmp_path: Path) -> None:
    handler, adapter, agents = host(tmp_path)
    entry = await handler._ensure_session(chat_ref())
    agent = agents[0]
    agent.turn_gate = asyncio.Event()
    child_gate = asyncio.Event()
    agent.background_jobs.launch(child_gate)
    try:
        await handler.handle(chat_ref(), inbound("work"))
        await asyncio.wait_for(agent.turn_started.wait(), 1)
        child_gate.set()
        await asyncio.wait_for(agent.background_jobs.changed.wait(), 1)
        assert agent.continuations == 0
        assert agent.background_jobs.pending_count == 1
        await handler._stop_turn(chat_ref())
        await asyncio.wait_for(agent.background_jobs.delivered.wait(), 1)
        assert await wait_for(lambda: entry.turn_task is None)
        assert agent.continuations == 0
        assert agent.history[0]["status"] == "completed"
        assert "child report" in all_sent_texts(adapter)
    finally:
        await handler.shutdown()


@pytest.mark.asyncio
async def test_jobs_pin_chat_against_capacity_eviction_and_shutdown_joins(tmp_path: Path) -> None:
    handler, _, agents = host(tmp_path, max_sessions=1)
    first = await handler._ensure_session(chat_ref("one"))
    first.background_jobs.launch(asyncio.Event())
    second = await handler._ensure_session(chat_ref("two"))
    assert handler._registry.get(chat_ref("one")).payload is first
    assert handler._registry.get(chat_ref("two")).payload is second
    await handler.shutdown()
    assert all(agent.cleaned for agent in agents)
    assert first.background_jobs.active_count == 0
    assert first.background_jobs.outcomes[0]["status"] == "cancelled"


@pytest.mark.asyncio
async def test_model_rebuild_transfers_supervisor_without_closing_live_generation(tmp_path: Path) -> None:
    handler, _, agents = host(tmp_path)
    entry = await handler._ensure_session(chat_ref())
    old = agents[0]
    jobs = old.background_jobs
    gate = asyncio.Event()
    jobs.launch(gate)
    try:
        await handler._apply_model(chat_ref(), entry, "anthropic/claude-opus-5")
        rebuilt = agents[1]
        assert rebuilt.background_jobs is jobs
        assert not old.cleaned
        assert not jobs.closed
        gate.set()
        await asyncio.wait_for(rebuilt.continued.wait(), 1)
        assert await wait_for(lambda: entry.turn_task is None)
        assert rebuilt.continuations == 1
        assert old.continuations == 0
        assert old.cleaned
    finally:
        await handler.shutdown()


@pytest.mark.asyncio
async def test_reset_joins_old_parent_and_jobs_without_injecting_into_new_chat(tmp_path: Path) -> None:
    handler, _, agents = host(tmp_path)
    old_entry = await handler._ensure_session(chat_ref())
    old = agents[0]
    old.turn_gate = asyncio.Event()
    jobs = old.background_jobs
    jobs.launch(asyncio.Event())
    try:
        await handler.handle(chat_ref(), inbound("work"))
        await asyncio.wait_for(old.turn_started.wait(), 1)
        await asyncio.wait_for(handler._reset_session(chat_ref()), 1)
        assert old.cleaned
        assert old.active_turns == 0
        assert not jobs.has_work
        new_entry = await handler._ensure_session(chat_ref())
        assert old_entry.session_id != new_entry.session_id
        assert not agents[1].history
        assert agents[1].continuations == 0
    finally:
        await handler.shutdown()


@pytest.mark.asyncio
async def test_long_report_is_scrubbed_and_chunked_before_one_continuation(tmp_path: Path) -> None:
    adapter = GatedAdapter()
    handler, _, agents = host(tmp_path, adapter=adapter)
    entry = await handler._ensure_session(chat_ref())
    secret = "fake-background-secret"
    handler._secret_values.append(secret)
    report = "child report " * 600 + secret
    expected = chunk_text(scrub(report, [secret]), adapter.capabilities.text_chunk_limit)
    assert len(report) > 5000
    assert len(expected) > 1
    try:
        queue_reports(agents[0], report)
        await asyncio.wait_for(adapter.blocked.wait(), 2)
        assert agents[0].continuations == 0
        assert len(entry.completions) == 1
        adapter.release.set()
        await asyncio.wait_for(agents[0].continued.wait(), 2)
        assert await wait_for(lambda: entry.turn_task is None and not entry.background_queued)
        assert [text for _, text in adapter.sent] == expected + ["continued with child report"]
        assert secret not in all_sent_texts(adapter)
        assert agents[0].continuations == 1
        assert agents[0].max_active_turns == 1
        assert not entry.completions
        assert not entry.completion_chunks
    finally:
        await handler.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("blocked_chunk", [1, 2])
async def test_cancelled_report_send_retains_only_unsent_chunks_without_paused_continuation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, blocked_chunk: int
) -> None:
    monkeypatch.setattr("kolega_code.gateway.sessions.BACKGROUND_RETRY_SECONDS", 0.1)
    adapter = GatedAdapter(block_on=blocked_chunk)
    handler, _, agents = host(tmp_path, adapter=adapter)
    entry = await handler._ensure_session(chat_ref())
    report = "cancelled child report " * 300
    expected = chunk_text(report, adapter.capabilities.text_chunk_limit)
    try:
        queue_reports(agents[0], report)
        await asyncio.wait_for(adapter.blocked.wait(), 2)
        assert await handler._stop_turn(chat_ref())
        assert await wait_for(lambda: entry.turn_task is None and entry.background_retry is not None)
        assert entry.auto_continue_paused
        assert entry.completion_chunks == expected[blocked_chunk - 1 :]
        adapter.release.set()
        assert await wait_for(lambda: not entry.completions and entry.turn_task is None and not entry.background_queued)
        assert [text for _, text in adapter.sent] == expected
        assert len(agents[0].history) == 1
        assert agents[0].continuations == 0
        assert entry.auto_continue_paused
        assert entry.background_retry is None
    finally:
        await handler.shutdown()


@pytest.mark.asyncio
async def test_failed_partial_batch_retries_without_resending_acknowledged_reports_or_chunks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    retry_delay = 0.1
    monkeypatch.setattr("kolega_code.gateway.sessions.BACKGROUND_RETRY_SECONDS", retry_delay)
    adapter = GatedAdapter(fail_on=3, block_on=4)
    handler, _, agents = host(tmp_path, adapter=adapter)
    entry = await handler._ensure_session(chat_ref())
    first = "first acknowledged report"
    second = "second child report " * 300
    expected = chunk_text(second, adapter.capabilities.text_chunk_limit)
    try:
        queue_reports(agents[0], first, second)
        await asyncio.wait_for(adapter.failed.wait(), 2)
        await asyncio.wait_for(adapter.blocked.wait(), 2)
        assert adapter.attempt_times[3] - adapter.attempt_times[2] >= retry_delay
        assert [text for _, text in adapter.sent if text != ERROR_REPLY] == [first, expected[0]]
        assert len(entry.completions) == 1
        assert entry.completion_chunks == expected[1:]
        assert len(agents[0].history) == 2
        assert agents[0].continuations == 0
        adapter.release.set()
        await asyncio.wait_for(agents[0].continued.wait(), 2)
        assert await wait_for(lambda: entry.turn_task is None and not entry.background_queued)
        assert [text for _, text in adapter.sent if text != ERROR_REPLY] == [
            first,
            *expected,
            "continued with child report",
        ]
        assert agents[0].continuations == 1
        assert agents[0].max_active_turns == 1
    finally:
        await handler.shutdown()


@pytest.mark.asyncio
async def test_running_base_agent_consumes_report_and_replies_without_extra_continuation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    llm = ScriptedLLM()
    call = ToolCall(id="tool-1", name="read_file", input={})
    tool_stream = ScriptedStream("")
    monkeypatch.setattr(
        tool_stream,
        "get_final_message",
        AsyncMock(return_value=Message(role="assistant", content=[call], tool_calls=[call], stop_reason="tool_use")),
    )
    llm.stream.side_effect = [tool_stream, ScriptedStream("distinct parent response")]
    handler, adapter, _ = make_handler(tmp_path, llm=llm)
    entry = await handler._ensure_session(chat_ref())
    agent = entry.runtime.agent
    assert isinstance(agent, BaseAgent)
    tool_started = asyncio.Event()
    release_tool = asyncio.Event()
    release_child = asyncio.Event()

    async def process_tool_calls(calls: list[ToolCall]) -> list[ToolResult]:
        assert calls == [call]
        tool_started.set()
        await release_tool.wait()
        return [ToolResult(tool_use_id=call.id, name=call.name, content="tool done", is_error=False)]

    async def child() -> str:
        await release_child.wait()
        return "distinct child report"

    monkeypatch.setattr(agent, "process_tool_calls", AsyncMock(side_effect=process_tool_calls))
    continuation = MagicMock(wraps=agent.continue_from_history_stream)
    monkeypatch.setattr(agent, "continue_from_history_stream", continuation)
    jobs = agent.get_background_jobs()
    try:
        await jobs.submit("job-running", child, agent_type="investigation", task="research")
        await handler.handle(chat_ref(), inbound("work"))
        await asyncio.wait_for(tool_started.wait(), 2)
        release_child.set()
        await asyncio.wait_for(jobs.wait_for_completion(), 2)
        assert jobs.pending_count == 1
        assert "distinct child report" not in all_sent_texts(adapter)
        release_tool.set()
        assert await wait_for(
            lambda: (
                "distinct child report" in all_sent_texts(adapter)
                and entry.turn_task is None
                and not entry.background_queued
            )
        )
        texts = [text for _, text in adapter.sent]
        assert sum("distinct child report" in text for text in texts) == 1
        assert texts.count("distinct parent response") == 1
        assert llm.stream.await_count == 2
        continuation.assert_not_called()
        assert not agent.background_needs_turn
        assert jobs.pending_count == 0
        assert not entry.completions
    finally:
        release_tool.set()
        release_child.set()
        await handler.shutdown()
