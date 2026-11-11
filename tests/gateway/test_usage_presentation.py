from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest

from kolega_code.cli.session_event_store import FileSessionEventStore
from kolega_code.events import AgentEvent
from kolega_code.gateway.sessions import AgentTurnHandler
from kolega_code.llm.models import Message
from kolega_code.llm.usage import normalize_usage
from kolega_code.session.projection import replay
from kolega_code.session.runtime import SessionRuntime
from kolega_code.session.usage import UsagePresentation

from .test_sessions import (
    ScriptedLLM,
    ScriptedStream,
    all_sent_texts,
    chat_ref,
    inbound,
    make_handler,
    wait_for,
    wait_for_turn_done,
)

pytestmark = [pytest.mark.asyncio, pytest.mark.usefixtures("isolated_cli_env")]


class _UsageStream(ScriptedStream):
    async def get_final_message(self) -> Message:
        message = await super().get_final_message()
        message.usage = normalize_usage(
            {"input_tokens": 11, "output_tokens": 4}, "anthropic", "claude-haiku-4-5-20251001"
        )
        return message


class _UsageLLM(ScriptedLLM):
    async def _stream(self, **_kwargs: Any) -> ScriptedStream:
        return _UsageStream(self._text)


async def _events(handler: AgentTurnHandler, session_id: str) -> list[AgentEvent]:
    return await FileSessionEventStore(handler._store.journal(session_id)).read(session_id)


def _snapshots(events: list[AgentEvent]) -> list[AgentEvent]:
    return [event for event in events if event.event_type == "usage_updated"]


@pytest.mark.parametrize("cancel_waiter", [False, True])
async def test_legacy_evidence_starts_before_runtime_survives_rebuild_and_closes_after_shutdown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cancel_waiter: bool
) -> None:
    handler, adapter, _ = make_handler(tmp_path, llm=_UsageLLM())
    start = SessionRuntime.start
    starts: list[UsagePresentation] = []

    async def observed_start(runtime: SessionRuntime) -> Any:
        operation = handler._operations[chat_ref().key]
        if operation.kind == "startup":
            owner = operation.usage_presentation
            assert owner is not None
            starts.append(owner)
            snapshot = _snapshots(await _events(handler, runtime.session_id))[-1]
            assert snapshot.content["usage"]["requests"] == 0
            assert snapshot.content["usage"]["total_tokens"] == 0
        return await start(runtime)

    monkeypatch.setattr(SessionRuntime, "start", observed_start)
    release = asyncio.Event()
    shutting: asyncio.Task[None] | None = None
    try:
        await handler.handle(chat_ref(), inbound("first turn"))
        assert await wait_for(lambda: "hello from the scripted model" in all_sent_texts(adapter))
        await wait_for_turn_done(handler, chat_ref())
        registry_entry = handler._registry.get(chat_ref())
        assert registry_entry is not None
        entry = registry_entry.payload
        owner = entry.usage_presentation
        assert owner is not None and starts == [owner]
        await owner.flush()
        first = _snapshots(await _events(handler, entry.session_id))[-1]
        assert first.content["usage"]["requests"] == 1
        assert first.content["usage"]["input_tokens"] == 11
        assert first.content["usage"]["output_tokens"] == 4
        assert first.content["usage"]["total_tokens"] == 15
        assert first.content["usage"]["coverage"]["full"] is False
        assert first.content["usage"]["complete"] is False
        assert first.content["usage"]["coverage"]["legacy_requests"] == 1

        old_agent = entry.runtime.agent
        await handler.handle(chat_ref(), inbound("/model anthropic/claude-haiku-4-5-20251001", message_id="model"))
        assert entry.runtime.agent is not old_agent
        assert entry.usage_presentation is owner
        assert starts == [owner]
        await handler.handle(chat_ref(), inbound("second turn", message_id="second"))
        assert await wait_for(lambda: all_sent_texts(adapter).count("hello from the scripted model") == 2)
        await wait_for_turn_done(handler, chat_ref())
        await owner.flush()
        events = await _events(handler, entry.session_id)
        last = _snapshots(events)[-1]
        assert last.content["usage"]["requests"] == 2
        assert last.content["usage"]["total_tokens"] == 30
        assert last.content["usage"]["coverage"]["full"] is False
        assert last.content["usage"]["coverage"]["accounted_runs"] == 0
        assert last.content["scope"] == "session"
        assert last.content["semantics"] == "snapshot"
        assert type(last.content["source_seq"]) is int
        assert last.session_id == entry.session_id
        assert last.workspace_id == entry.record.workspace_id
        assert last.thread_id == entry.record.thread_id
        assert last.sub_agent_info is None
        assert replay(events).usage == {"source_seq": last.content["source_seq"], **last.content["usage"]}
        assert not any(
            event.event_type.startswith("llm.") for event in handler._store.journal(entry.session_id).read_events()
        )

        entered = asyncio.Event()
        shutdown, close = entry.runtime.shutdown, owner.aclose
        order: list[str] = []

        async def observed_shutdown() -> None:
            assert not owner._closed
            order.append("runtime-start")
            entered.set()
            await release.wait()
            await shutdown()
            recorder = handler._store.recorder(entry.session_id)
            recorder.start_turn(Message(role="user", content="cleanup evidence"))
            usage = normalize_usage({"input_tokens": 3, "output_tokens": 1}, "anthropic", "usage-test")
            recorder.record_assistant(Message(role="assistant", content="cleanup response", usage=usage))
            recorder.finish_turn("completed")
            order.append("runtime-end")

        async def observed_close() -> None:
            assert order == ["runtime-start", "runtime-end"]
            await close()
            order.append("publisher-close")

        monkeypatch.setattr(entry.runtime, "shutdown", observed_shutdown)
        monkeypatch.setattr(owner, "aclose", observed_close)
        shutting = asyncio.create_task(handler.shutdown())
        await asyncio.wait_for(entered.wait(), 10)
        if cancel_waiter:
            shutting.cancel()
            shutting.cancel()
            with pytest.raises(asyncio.CancelledError):
                await shutting
        assert not owner._closed and owner._unsubscribe is not None
        assert handler._operations[chat_ref().key].entry is entry
        assert handler._shutdown_task is not None and not handler._shutdown_task.done()
        release.set()
        await asyncio.wait_for(handler.wait_closed(), 10)
        if not cancel_waiter:
            await shutting
        assert order == ["runtime-start", "runtime-end", "publisher-close"]
        assert owner._closed and owner._unsubscribe is None
        assert owner._worker is not None and owner._worker.done()
        final = _snapshots(await _events(handler, entry.session_id))[-1]
        assert final.content["usage"]["requests"] == 3
        assert final.content["usage"]["input_tokens"] == 25
        assert final.content["usage"]["output_tokens"] == 9
        assert final.content["usage"]["total_tokens"] == 34
        assert final.content["usage"]["coverage"]["full"] is False
        assert final.content["usage"]["coverage"]["accounted_runs"] == 0
    finally:
        release.set()
        if shutting is not None:
            await asyncio.gather(shutting, return_exceptions=True)
        await handler.wait_closed()
        handler._store.release_session_locks()


@pytest.mark.parametrize("stage", ["publication", "agent", "background"])
async def test_partial_startup_retains_and_drains_publisher_without_publishing_a_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stage: str
) -> None:
    handler, _, _ = make_handler(tmp_path)
    failure = RuntimeError(f"{stage} startup failed")
    captured: dict[str, Any] = {}
    entered, release = asyncio.Event(), asyncio.Event()
    start, close = UsagePresentation.start, UsagePresentation.aclose
    builder = handler._agent_builder
    assert builder is not None

    async def observed_start(owner: UsagePresentation) -> None:
        captured["owner"] = owner
        operation = handler._operations[chat_ref().key]
        assert operation.usage_presentation is owner
        captured["operation"] = operation
        await start(owner)
        if stage == "publication":
            raise failure

    async def observed_builder(*args: Any, **kwargs: Any) -> Any:
        agent = await builder(*args, **kwargs)
        captured["agent"] = agent
        cleanup = AsyncMock(wraps=agent.cleanup)
        monkeypatch.setattr(agent, "cleanup", cleanup)
        captured["cleanup"] = cleanup
        if stage == "agent":
            kwargs["owner"].partial_agent = agent
            raise failure
        return agent

    async def failed_background(*_args: Any) -> None:
        raise failure

    async def blocked_close(owner: UsagePresentation) -> None:
        assert captured["operation"].usage_presentation is owner
        if "cleanup" in captured:
            captured["cleanup"].assert_awaited_once()
        entered.set()
        await release.wait()
        await close(owner)

    monkeypatch.setattr(UsagePresentation, "start", observed_start)
    monkeypatch.setattr(UsagePresentation, "aclose", blocked_close)
    monkeypatch.setattr(handler, "_agent_builder", observed_builder)
    if stage == "background":
        monkeypatch.setattr(handler, "_attach_background", failed_background)
    request = asyncio.create_task(handler._ensure_session(chat_ref()))
    try:
        await asyncio.wait_for(entered.wait(), 10)
        with pytest.raises(RuntimeError) as raised:
            await asyncio.wait_for(request, 10)
        assert raised.value is failure
        owner, operation = captured["owner"], captured["operation"]
        assert handler._operations[chat_ref().key] is operation
        assert operation.cleaning and operation.task is not None and not operation.task.done()
        assert handler._registry.get(chat_ref()) is None
        assert not owner._closed
        cancellations = operation.task.cancelling()
        await handler.handle(chat_ref(), inbound("/stop"))
        assert operation.task.cancelling() == cancellations
        release.set()
        result = await asyncio.wait_for(asyncio.gather(operation.task, return_exceptions=True), 10)
        assert result == [failure]
        assert owner._closed and owner._unsubscribe is None
        assert owner._worker is not None and owner._worker.done()
        assert operation.usage_presentation is None
        assert operation.router._task is None
        assert operation.relay._pump_task is None
        assert chat_ref().key not in handler._operations
        assert handler._registry.get(chat_ref()) is None
    finally:
        release.set()
        await asyncio.gather(request, return_exceptions=True)
        await handler.wait_closed()
        handler._store.release_session_locks()
