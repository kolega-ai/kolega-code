from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio

from kolega_code.agent import CoderAgent, PlanningAgent
from kolega_code.agent.prompts import PLANNING_QUESTION_PROMPT, build_implement_plan_prompt
from kolega_code.gateway import sessions
from kolega_code.gateway.adapters.base import InboundMessage
from kolega_code.gateway.sessions import AgentTurnHandler, ChatLifecycleIncomplete, ChatSession
from kolega_code.llm.client import LLMClient
from kolega_code.llm.models import Message, ToolCall, ToolResult
from kolega_code.session.plans import PLAN_APPROVAL_OPTIONS

from .test_sessions import (
    RecordingAdapter,
    ScriptedLLM,
    ScriptedStream,
    chat_ref,
    inbound,
    make_handler,
    wait_for,
    wait_for_turn_done,
)

pytestmark = pytest.mark.usefixtures("isolated_cli_env")

PLAN = "# Complete plan\n\n1. Inspect the repository.\n2. Implement the change.\n3. Verify every case.\n"


@pytest_asyncio.fixture(autouse=True, params=[False, True], ids=["lazy", "eager"])
async def planning_scheduling(request: pytest.FixtureRequest) -> AsyncIterator[None]:
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


@pytest_asyncio.fixture
async def planning_gateway(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> AsyncIterator[tuple[AgentTurnHandler, RecordingAdapter, ScriptedLLM]]:
    handler, adapter, _ = make_handler(tmp_path)
    handler._agent_builder = None
    handler._load_settings().lsp_enabled = False
    llm = ScriptedLLM(PLAN)
    monkeypatch.setattr(LLMClient, "stream", llm.stream)
    monkeypatch.setattr(LLMClient, "generate", llm.generate)
    monkeypatch.setattr(LLMClient, "count_tokens", llm.count_tokens)
    try:
        yield handler, adapter, llm
    finally:
        await handler.shutdown()


def tap(token: str, action: str) -> InboundMessage:
    return replace(inbound(""), callback_token=token, callback_option=action)


async def live_plan_token(entry: ChatSession, adapter: RecordingAdapter, *, after: int = 0) -> str:
    def current() -> str | None:
        return next(
            (
                token
                for token in reversed(adapter.tokens[after:])
                if (prompt := entry.relay._pending.get(token)) is not None
                and prompt.kind == "plan_approval"
                and entry.runtime.control.is_pending(prompt.request_id)
            ),
            None,
        )

    assert await wait_for(lambda: current() is not None)
    token = current()
    assert token is not None
    return token


async def offer(handler: AgentTurnHandler, adapter: RecordingAdapter, *, text: str = "Please plan this change.") -> Any:
    await handler.handle(chat_ref(), inbound("/mode plan"))
    before = len(adapter.buttons)
    await handler.handle(chat_ref(), inbound(text))
    assert await wait_for(lambda: len(adapter.buttons) > before)
    await wait_for_turn_done(handler, chat_ref())
    entry = handler._registry.get(chat_ref())
    assert entry is not None
    return entry.payload


def epoch_ids(handler: AgentTurnHandler, session_id: str) -> list[str | None]:
    return [
        event.epoch_id
        for event in handler._store.journal(session_id).read_events()
        if event.event_type == "context.epoch_started"
    ]


@pytest.mark.asyncio
async def test_mode_command_composes_real_agents_and_preserves_mode_on_model_rebuild(
    planning_gateway: tuple[AgentTurnHandler, RecordingAdapter, ScriptedLLM],
) -> None:
    handler, adapter, _ = planning_gateway
    await handler.handle(chat_ref(), inbound("/help"))
    assert "/mode [build|plan]" in adapter.sent[-1][1]
    await handler.handle(chat_ref(), inbound("/mode invalid"))
    assert adapter.sent[-1][1] == "Usage: /mode build|plan"
    await handler.handle(chat_ref(), inbound("/mode plan"))
    entry = await handler._ensure_session(chat_ref())
    agent = entry.runtime.agent
    assert agent is not None
    assert isinstance(agent, PlanningAgent)
    assert agent.tool_collection is not None
    assert agent.tool_collection.tool_config.read_only
    assert agent.tool_collection.has_tool("write_plan")
    assert not agent.tool_collection.has_tool("write")
    assert any(prompt.markdown == PLANNING_QUESTION_PROMPT for prompt in agent.prompt_extensions)
    assert handler._store.load(entry.session_id).interaction_mode == "plan"
    await handler.handle(chat_ref(), inbound("/status"))
    assert "Mode: plan" in adapter.sent[-1][1]
    await handler.handle(chat_ref(), inbound("/model anthropic/claude-haiku-4-5-20251001"))
    assert isinstance(entry.runtime.agent, PlanningAgent)
    await handler.handle(chat_ref(), inbound("/mode build"))
    assert isinstance(entry.runtime.agent, CoderAgent)
    assert handler._store.load(entry.session_id).interaction_mode == "build"
    assert not adapter.buttons


@pytest.mark.asyncio
@pytest.mark.parametrize("action", [action for action, _ in PLAN_APPROVAL_OPTIONS])
async def test_full_planning_turn_and_all_choices_persist_and_run_only_real_implementation(
    planning_gateway: tuple[AgentTurnHandler, RecordingAdapter, ScriptedLLM], action: str
) -> None:
    handler, adapter, llm = planning_gateway
    await handler.handle(chat_ref(), inbound("existing build context"))
    assert await wait_for(lambda: any(PLAN in text for _, text in adapter.sent))
    await wait_for_turn_done(handler, chat_ref())
    plan = "  # Full Markdown\n\n" + ("- Preserve every detailed step.\n" * 360) + "\nTail: test-key\n"
    llm._text = plan
    adapter.capabilities = replace(adapter.capabilities, text_chunk_limit=256)
    entry = await offer(handler, adapter)
    assert isinstance(entry.runtime.agent, PlanningAgent)
    assert entry.record.latest_plan_markdown == plan
    assert entry.record.plan_pending and entry.record.plan_reofferable
    saved = handler._store.load(entry.session_id)
    assert saved.latest_plan_markdown == plan and saved.plan_pending and saved.plan_reofferable
    request = entry.runtime.control.pending()[0]
    assert request.kind == "plan_approval"
    assert request.payload == {
        "plan": plan,
        "options": [{"id": value, "label": label} for value, label in PLAN_APPROVAL_OPTIONS],
    }
    assert request.default == {"action": None}
    assert [(button.option_id, button.label) for button in adapter.buttons[-1][2]] == list(PLAN_APPROVAL_OPTIONS)
    assert adapter.buttons[-1][1] == "Implement this plan?"
    assert all(len(text) <= 256 for _, text in adapter.sent if text != PLAN)
    assert "test-key" not in "\n".join(text for _, text in adapter.sent)
    epochs = epoch_ids(handler, entry.session_id)
    calls = llm.stream.await_count
    llm._text = "Implementation completed."
    await handler.handle(chat_ref(), tap(adapter.tokens[-1], action))
    assert await wait_for(lambda: entry.plan_approval_task is None)
    saved = handler._store.load(entry.session_id)
    assert saved.latest_plan_markdown == plan and not saved.plan_pending
    if action == "discuss_plan":
        assert isinstance(entry.runtime.agent, PlanningAgent)
        assert saved.interaction_mode == "plan" and saved.plan_reofferable
        assert llm.stream.await_count == calls
        assert epoch_ids(handler, entry.session_id) == epochs
    else:
        assert isinstance(entry.runtime.agent, CoderAgent)
        assert saved.interaction_mode == "build" and not saved.plan_reofferable
        assert llm.stream.await_count == calls + 1
        texts = [message.get_text_content() for message in entry.runtime.agent.history]
        assert texts.count(build_implement_plan_prompt(plan)) == 1
        assert texts[-1] == "Implementation completed."
        if action == "implement_plan_clear":
            assert "existing build context" not in texts
            assert len(epoch_ids(handler, entry.session_id)) == len(epochs) + 1
        else:
            assert "existing build context" in texts
            assert epoch_ids(handler, entry.session_id) == epochs


@pytest.mark.asyncio
async def test_write_plan_tool_wins_over_the_full_narrative_response(
    planning_gateway: tuple[AgentTurnHandler, RecordingAdapter, ScriptedLLM],
) -> None:
    handler, adapter, llm = planning_gateway

    class PlanStream(ScriptedStream):
        async def get_final_message(self) -> Message:
            call = ToolCall(id="gateway-write-plan", name="write_plan", input={"plan_markdown": PLAN})
            return Message(
                role="assistant",
                content=[call],
                tool_calls=[call],
                stop_reason="tool_use",
            )

    async def stream(**_kwargs: Any) -> PlanStream:
        return PlanStream("A narrative that is not the implementation plan.")

    llm.stream.side_effect = stream
    entry = await offer(handler, adapter)
    assert isinstance(entry.runtime.agent, PlanningAgent)
    assert entry.record.latest_plan_markdown == PLAN.strip()
    assert llm.stream.await_count == 1
    assert any(
        isinstance(block, ToolResult) and block.content == "Plan captured."
        for message in entry.runtime.agent.history
        for block in message.content
    )
    assert handler._store.load(entry.session_id).latest_plan_markdown == PLAN.strip()


@pytest.mark.asyncio
async def test_discussion_and_explicit_reoffer_use_new_authority_without_synthetic_turns(
    planning_gateway: tuple[AgentTurnHandler, RecordingAdapter, ScriptedLLM],
) -> None:
    handler, adapter, llm = planning_gateway
    entry = await offer(handler, adapter)
    old_token = adapter.tokens[-1]
    await handler.handle(chat_ref(), tap(old_token, "discuss_plan"))
    assert await wait_for(lambda: entry.plan_approval_task is None)
    count = llm.stream.await_count
    await handler.handle(chat_ref(), inbound("/mode plan"))
    assert await wait_for(lambda: len(adapter.tokens) == 2)
    assert llm.stream.await_count == count
    await handler.handle(chat_ref(), tap(old_token, "implement_plan"))
    assert llm.stream.await_count == count
    await handler.handle(chat_ref(), tap(adapter.tokens[-1], "discuss_plan"))
    assert await wait_for(lambda: entry.plan_approval_task is None)
    llm._text = "# Revised plan\n\nInclude the requested alternative.\n"
    await handler.handle(chat_ref(), inbound("Discuss the alternative and revise the plan."))
    assert await wait_for(lambda: len(adapter.tokens) == 3)
    await wait_for_turn_done(handler, chat_ref())
    assert isinstance(entry.runtime.agent, PlanningAgent)
    assert entry.record.latest_plan_markdown == llm._text
    assert llm.stream.await_count == count + 1
    texts = [message.get_text_content() for message in entry.runtime.agent.history]
    assert "Discuss the alternative and revise the plan." in texts
    assert not any(text == build_implement_plan_prompt(PLAN) for text in texts)


@pytest.mark.asyncio
async def test_completed_offer_is_reclaimed_before_its_queued_settlement_callback(
    planning_gateway: tuple[AgentTurnHandler, RecordingAdapter, ScriptedLLM],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handler, adapter, llm = planning_gateway
    entry = await offer(handler, adapter)
    owner = entry.plan_approval_task
    assert owner is not None
    history = entry.runtime.agent.dump_message_history()
    epochs = epoch_ids(handler, entry.session_id)
    old_token, tokens = adapter.tokens[-1], len(adapter.tokens)
    old_request = entry.relay._pending[old_token].request_id
    reoffered = asyncio.Event()
    observed: list[tuple[bool, bool, asyncio.Task[None] | None]] = []
    save = handler._store.save

    def reoffer_before_settlement() -> None:
        completed, registered = owner.done(), entry.plan_approval_task is owner
        entry.record.plan_pending = True
        handler._start_plan_approval(entry)
        observed.append((completed, registered, entry.plan_approval_task))
        reoffered.set()

    def save_and_reoffer(record: Any) -> None:
        save(record)
        if asyncio.current_task() is owner:
            asyncio.get_running_loop().call_soon(reoffer_before_settlement)

    monkeypatch.setattr(handler._store, "save", save_and_reoffer)
    await handler.handle(chat_ref(), tap(old_token, "discuss_plan"))
    await asyncio.wait_for(reoffered.wait(), 10)
    completed, registered, replacement = observed[0]
    assert completed and registered
    assert replacement is not None and replacement is not owner
    token = await live_plan_token(entry, adapter, after=tokens)
    assert entry.plan_approval_task is replacement and not replacement.done()
    assert entry.relay._pending[token].request_id != old_request
    await handler.handle(chat_ref(), tap(old_token, "implement_plan_clear"))
    assert not replacement.done()
    await handler.handle(chat_ref(), tap(token, "discuss_plan"))
    assert await wait_for(lambda: entry.plan_approval_task is None)
    assert llm.stream.await_count == 1
    assert entry.runtime.agent.dump_message_history() == history
    assert epoch_ids(handler, entry.session_id) == epochs


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "revision", [PLAN, "# Revised plan\n\nDo not implement the first proposal.\n"], ids=["same", "changed"]
)
async def test_queued_planning_turn_fences_first_offer_before_streaming_and_reoffers_its_own_revision(
    planning_gateway: tuple[AgentTurnHandler, RecordingAdapter, ScriptedLLM],
    revision: str,
) -> None:
    handler, adapter, llm = planning_gateway
    first_release, second_release = asyncio.Event(), asyncio.Event()
    llm.stream.side_effect = [
        ScriptedStream(PLAN, block=first_release),
        ScriptedStream(revision, block=second_release),
        ScriptedStream("Implemented the revised plan."),
    ]
    await handler.handle(chat_ref(), inbound("/mode plan"))
    await handler.handle(chat_ref(), inbound("Plan the first approach."))
    assert await wait_for(lambda: llm.stream.await_count == 1)
    entry = await handler._ensure_session(chat_ref())
    await handler.handle(chat_ref(), inbound("Revise that approach before I approve anything."))
    first_release.set()
    try:
        assert await wait_for(lambda: llm.stream.await_count == 2)
        agent = entry.runtime.agent
        assert agent is not None
        assert isinstance(agent, PlanningAgent)
        epochs = epoch_ids(handler, entry.session_id)
        before = agent.dump_message_history()
        jobs = entry.background_jobs
        pending = [request for request in entry.runtime.control.pending() if request.kind == "plan_approval"]
        first_token = adapter.tokens[-1] if adapter.tokens else None
        if first_token is not None:
            await handler.handle(chat_ref(), tap(first_token, "implement_plan_clear"))
        if pending:
            assert await wait_for(lambda: chat_ref().key in handler._operations)
        second_release.set()
        assert await wait_for(lambda: entry.turn_task is None and not handler._operations)
        assert not pending, "the first offer retained authority while its queued revision streamed"
        assert entry.runtime.agent is agent and entry.background_jobs is jobs
        assert epoch_ids(handler, entry.session_id) == epochs
        assert entry.record.history[: len(before)] == before
        assert entry.record.latest_plan_markdown == revision and entry.record.plan_pending
        saved = handler._store.load(entry.session_id)
        assert saved.interaction_mode == "plan" and saved.plan_pending and saved.plan_reofferable
        assert saved.history == agent.dump_message_history() and saved.history
        assert saved.latest_plan_markdown == revision and llm.stream.await_count == 2
        assert await wait_for(
            lambda: bool(entry.runtime.control.pending()) and bool(adapter.tokens) and adapter.tokens[-1] != first_token
        )
        fresh = entry.runtime.control.pending()[0]
        assert fresh.payload["plan"] == revision
        assert adapter.tokens[-1] != first_token
        if first_token is not None:
            await handler.handle(chat_ref(), tap(first_token, "implement_plan_clear"))
        await handler.handle(chat_ref(), tap(adapter.tokens[-1], "implement_plan"))
        assert await wait_for(lambda: entry.plan_approval_task is None)
        assert entry.runtime.agent is not None
        assert isinstance(entry.runtime.agent, CoderAgent)
        texts = [message.get_text_content() for message in entry.runtime.agent.history]
        assert texts.count(build_implement_plan_prompt(revision)) == 1
        assert texts[-1] == "Implemented the revised plan."
        assert llm.stream.await_count == 3 and epoch_ids(handler, entry.session_id) == epochs
    finally:
        first_release.set()
        second_release.set()


@pytest.mark.asyncio
@pytest.mark.parametrize("resolution", ["unknown", "unanswered", "stop", "error"])
async def test_nonapproval_keeps_plan_pending_for_a_later_fresh_offer(
    planning_gateway: tuple[AgentTurnHandler, RecordingAdapter, ScriptedLLM],
    monkeypatch: pytest.MonkeyPatch,
    resolution: str,
) -> None:
    handler, adapter, llm = planning_gateway
    if resolution == "error":
        await handler.handle(chat_ref(), inbound("/mode plan"))
        entry = await handler._ensure_session(chat_ref())
        monkeypatch.setattr(entry.runtime, "request_plan_approval", AsyncMock(side_effect=RuntimeError("offer failed")))
        await handler.handle(chat_ref(), inbound("Plan the change."))
        assert await wait_for(lambda: any(PLAN in text for _, text in adapter.sent))
        await wait_for_turn_done(handler, chat_ref())
    else:
        entry = await offer(handler, adapter)
        if resolution == "unknown":
            await handler.handle(chat_ref(), tap(adapter.tokens[-1], "allow_once"))
        elif resolution == "unanswered":
            entry.runtime.answer_plan_approval(
                entry.runtime.control.pending()[0].request_id, None, client_id=chat_ref().key
            )
        else:
            await handler.handle(chat_ref(), inbound("/stop"))
            await handler.handle(chat_ref(), tap(adapter.tokens[-1], "implement_plan"))
    assert await wait_for(lambda: entry.plan_approval_task is None)
    assert llm.stream.await_count == 1
    assert isinstance(entry.runtime.agent, PlanningAgent)
    saved = handler._store.load(entry.session_id)
    assert saved.plan_pending and saved.plan_reofferable and saved.latest_plan_markdown == PLAN


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["plan", "build"])
async def test_restore_pending_plan_gets_fresh_control_offer_without_replaying_a_turn(
    planning_gateway: tuple[AgentTurnHandler, RecordingAdapter, ScriptedLLM],
    mode: str,
) -> None:
    handler, adapter, llm = planning_gateway
    entry = await offer(handler, adapter)
    old_token = adapter.tokens[-1]
    old_request_id = entry.runtime.control.pending()[0].request_id
    if mode == "build":
        await handler.handle(chat_ref(), inbound("/mode build"))
    await handler.shutdown()
    adapter2 = RecordingAdapter()
    adapter2.tokens.append("reserve-old-token-namespace")
    resumed = AgentTurnHandler(
        config=handler._config,
        adapter=adapter2,
        store=handler._store,
        settings_store=handler._settings_store,
    )
    try:
        restored = await resumed._ensure_session(chat_ref())
        assert isinstance(restored.runtime.agent, PlanningAgent if mode == "plan" else CoderAgent)
        assert await wait_for(lambda: bool(adapter2.buttons))
        fresh = restored.runtime.control.pending()[0]
        assert fresh.request_id != old_request_id
        assert fresh.payload["plan"] == PLAN
        assert restored.record.plan_pending and restored.record.plan_reofferable
        assert llm.stream.await_count == 1
        await resumed.handle(chat_ref(), tap(old_token, "implement_plan"))
        assert llm.stream.await_count == 1
        await resumed.handle(chat_ref(), tap(adapter2.tokens[-1], "discuss_plan"))
        assert await wait_for(lambda: restored.plan_approval_task is None)
        assert isinstance(restored.runtime.agent, PlanningAgent)
        assert restored.record.interaction_mode == "plan" and restored.record.plan_reofferable
    finally:
        await resumed.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("command", ["/mode build", "/model anthropic/claude-haiku-4-5-20251001", "/new"])
async def test_commands_during_an_offer_do_not_deadlock_or_accept_old_buttons(
    planning_gateway: tuple[AgentTurnHandler, RecordingAdapter, ScriptedLLM], command: str
) -> None:
    handler, adapter, llm = planning_gateway
    entry = await offer(handler, adapter)
    old = adapter.tokens[-1]
    await asyncio.wait_for(handler.handle(chat_ref(), inbound(command)), 10)
    await handler.handle(chat_ref(), tap(old, "implement_plan_clear"))
    assert llm.stream.await_count == 1
    if command != "/new":
        assert entry.record.plan_pending
        assert handler._store.load(entry.session_id).history


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["initialize", "history", "adopt"])
async def test_clear_replacement_failure_preserves_live_persisted_history_mode_and_epoch(
    planning_gateway: tuple[AgentTurnHandler, RecordingAdapter, ScriptedLLM],
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    handler, adapter, llm = planning_gateway
    entry = await offer(handler, adapter)
    old = entry.runtime.agent
    before = old.dump_message_history()
    epochs = epoch_ids(handler, entry.session_id)
    partials: list[CoderAgent] = []

    class FailingBuild(CoderAgent):
        def __init__(self, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            partials.append(self)
            if failure == "initialize":
                assert self.tool_collection is not None
                monkeypatch.setattr(
                    self.tool_collection, "initialize", AsyncMock(side_effect=RuntimeError("initialize failed"))
                )

        def restore_message_history(self, history: list[dict[str, Any]]) -> None:
            if failure == "history":
                raise RuntimeError("history failed")
            super().restore_message_history(history)

        def set_permission_mode(self, mode: Any) -> None:
            if failure == "adopt":
                raise RuntimeError("adopt failed")
            super().set_permission_mode(mode)

    monkeypatch.setattr(sessions, "CoderAgent", FailingBuild)
    await handler.handle(chat_ref(), tap(adapter.tokens[-1], "implement_plan_clear"))
    assert await wait_for(lambda: entry.plan_approval_task is None)
    assert entry.runtime.agent is old
    assert old.dump_message_history() == before and before
    assert entry.record.history == before and entry.record.interaction_mode == "plan"
    assert len(partials) == 1
    assert epoch_ids(handler, entry.session_id) == epochs
    saved = handler._store.load(entry.session_id)
    assert saved.history and saved.plan_pending and saved.interaction_mode == "plan"
    assert saved.latest_plan_markdown == PLAN
    assert llm.stream.await_count == 1
    assert not handler._operations


@pytest.mark.asyncio
@pytest.mark.parametrize("retain_jobs", [False, True], ids=["empty", "retained"])
async def test_failed_clear_then_keep_implementation_admits_and_delivers_real_background_agent(
    planning_gateway: tuple[AgentTurnHandler, RecordingAdapter, ScriptedLLM],
    monkeypatch: pytest.MonkeyPatch,
    retain_jobs: bool,
) -> None:
    handler, adapter, llm = planning_gateway
    entry = await offer(handler, adapter)
    old = entry.runtime.agent
    jobs = entry.background_jobs
    before = old.dump_message_history()
    epochs = epoch_ids(handler, entry.session_id)

    class FailingBuild(CoderAgent):
        def __init__(self, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            assert self.tool_collection is not None
            monkeypatch.setattr(
                self.tool_collection, "initialize", AsyncMock(side_effect=RuntimeError("initialize failed"))
            )

    monkeypatch.setattr(sessions, "CoderAgent", FailingBuild)
    started, cancelled = asyncio.Event(), asyncio.Event()

    async def completed_job() -> str:
        return "Retained old completion."

    async def cancelled_job() -> str:
        started.set()
        try:
            await asyncio.Future()
        finally:
            cancelled.set()
        return "unreachable"

    async with entry.turn_lock:
        if retain_jobs:
            await jobs.submit("old-completed", completed_job, agent_type="general", task="completed old job")
            await asyncio.wait_for(jobs.wait(), 10)
            await jobs.submit("old-cancelled", cancelled_job, agent_type="general", task="cancelled old job")
            await asyncio.wait_for(started.wait(), 10)
        failed_offer = entry.plan_approval_task
        assert failed_offer is not None
        await handler.handle(chat_ref(), tap(adapter.tokens[-1], "implement_plan_clear"))
        assert await wait_for(lambda: chat_ref().key in handler._operations)
    assert await wait_for(lambda: failed_offer.done() and not handler._operations and entry.turn_task is None)
    assert entry.runtime.agent is old and not handler._operations
    restored = old.dump_message_history()
    assert entry.record.history == restored and restored[: len(before)] == before and before
    saved = handler._store.load(entry.session_id)
    assert saved.history == restored and saved.interaction_mode == "plan"
    assert saved.plan_pending and saved.plan_reofferable and saved.latest_plan_markdown == PLAN
    assert epoch_ids(handler, entry.session_id) == epochs and llm.stream.await_count == 1
    recovered = entry.background_jobs
    assert recovered is old.get_background_jobs() and recovered.recorder is old.session_recorder
    if retain_jobs:
        assert cancelled.is_set() and jobs.active_count == jobs.pending_count == 0
        rows = recovered.snapshot()
        assert [(row["job_id"], row["status"], row["delivered"]) for row in rows] == [
            ("old-completed", "completed", True),
            ("old-cancelled", "cancelled", True),
        ]
        texts = [message.get_text_content() for message in old.history]
        assert all(sum(row["completion_id"] in text for text in texts) == 1 for row in rows)
    else:
        assert restored == before
    monkeypatch.setattr(sessions, "CoderAgent", CoderAgent)
    tokens = len(adapter.tokens)
    await handler.handle(chat_ref(), inbound("/mode plan"))
    token = await live_plan_token(entry, adapter, after=tokens)
    llm._text = "Implemented with the original context."
    await handler.handle(chat_ref(), tap(token, "implement_plan"))
    assert await wait_for(lambda: entry.plan_approval_task is None)
    agent = entry.runtime.agent
    assert isinstance(agent, CoderAgent)
    assert epoch_ids(handler, entry.session_id) == epochs
    assert agent.dump_message_history()[: len(restored)] == restored
    await handler.handle(chat_ref(), inbound("/permissions auto"))
    launch, task = "Start a background investigation.", "Report on supervisor recovery."
    child_release, child_started = asyncio.Event(), asyncio.Event()

    class DispatchStream(ScriptedStream):
        async def get_final_message(self) -> Message:
            call = ToolCall(
                id="after-failed-clear",
                name="dispatch_agent",
                input={"agent_type": "general", "task": task, "background": True},
            )
            return Message(role="assistant", content=[call], tool_calls=[call], stop_reason="tool_use")

    async def stream(**kwargs: Any) -> ScriptedStream:
        last = kwargs["messages"][-1].get_text_content()
        if last == launch:
            return DispatchStream("")
        if any(message.get_text_content() == task for message in kwargs["messages"]):
            child_started.set()
            return ScriptedStream("Recovered supervisor child report.", block=child_release)
        return ScriptedStream("Background work acknowledged.")

    llm.stream.side_effect = stream
    try:
        await handler.handle(chat_ref(), inbound(launch))
        assert await wait_for(
            lambda: any(
                isinstance(block, ToolResult) and block.tool_use_id == "after-failed-clear"
                for message in agent.history
                for block in message.content
            )
        )
        result = next(
            block
            for message in agent.history
            for block in message.content
            if isinstance(block, ToolResult) and block.tool_use_id == "after-failed-clear"
        )
        assert not result.is_error, result.content
        assert isinstance(result.content, str)
        receipt = json.loads(result.content)
        assert receipt["state"] == "accepted"
        await asyncio.wait_for(child_started.wait(), 10)
        await wait_for_turn_done(handler, chat_ref())
        assert recovered is entry.background_jobs is agent.get_background_jobs() and recovered is not jobs
        assert recovered.recorder is agent.session_recorder
        assert recovered.snapshot(receipt["job_id"])[0]["status"] == "running"
        child_release.set()
        assert await wait_for(
            lambda: bool(recovered.snapshot(receipt["job_id"])[0]["delivered"]) and entry.turn_task is None
        )
        row = recovered.snapshot(receipt["job_id"])[0]
        assert row["status"] == "completed" and row["result"] == "Recovered supervisor child report."
        assert row["epoch_id"] == epochs[-1]
        texts = [message.get_text_content() for message in agent.history]
        assert sum(row["completion_id"] in text for text in texts) == 1
        assert any("Recovered supervisor child report." in text for _, text in adapter.sent)
        assert handler._store.load(entry.session_id).history == agent.dump_message_history()
        assert not handler._operations and epoch_ids(handler, entry.session_id) == epochs
    finally:
        child_release.set()


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", ["stop", "new", "shutdown", "timeout"])
async def test_cancel_during_clear_preparation_retains_partial_cleanup_and_never_resets_or_implements(
    planning_gateway: tuple[AgentTurnHandler, RecordingAdapter, ScriptedLLM],
    monkeypatch: pytest.MonkeyPatch,
    cancel: str,
) -> None:
    handler, adapter, llm = planning_gateway
    entry = await offer(handler, adapter)
    old = entry.runtime.agent
    before = old.dump_message_history()
    epochs = epoch_ids(handler, entry.session_id)
    initializing, cleaning, release = (asyncio.Event() for _ in range(3))
    partials: list[CoderAgent] = []
    cleanups = 0

    async def initialize() -> list[str]:
        initializing.set()
        await asyncio.Future()
        return []

    class BlockedBuild(CoderAgent):
        def __init__(self, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            partials.append(self)
            assert self.tool_collection is not None
            monkeypatch.setattr(self.tool_collection, "initialize", initialize)

        async def cleanup(self) -> None:
            nonlocal cleanups
            cleanups += 1
            cleaning.set()
            await release.wait()
            await super().cleanup()

    monkeypatch.setattr(sessions, "CoderAgent", BlockedBuild)
    if cancel == "timeout":
        monkeypatch.setattr(sessions, "LIFECYCLE_WAIT_SECONDS", 0.02)
    await handler.handle(chat_ref(), tap(adapter.tokens[-1], "implement_plan_clear"))
    await asyncio.wait_for(initializing.wait(), 10)
    owner = handler._operations[chat_ref().key]
    reset: asyncio.Task[None] | None = None
    if cancel == "stop":
        await handler.handle(chat_ref(), inbound("/stop"))
    elif cancel == "new":
        reset = asyncio.create_task(handler.handle(chat_ref(), inbound("/new")))
    elif cancel == "shutdown":
        handler.begin_shutdown()
    await asyncio.wait_for(cleaning.wait(), 10)
    try:
        assert owner.partial_agent is partials[0] and owner.cleaning
        assert owner.task is not None and not owner.task.done()
        assert entry.runtime.agent is old and old.dump_message_history() == before
        assert epoch_ids(handler, entry.session_id) == epochs
        assert handler._store.load(entry.session_id).history
        assert handler._store.load(entry.session_id).plan_pending
        with pytest.raises(ChatLifecycleIncomplete):
            await handler._ensure_session(chat_ref())
        for _ in range(3):
            if cancel != "shutdown":
                await handler.handle(chat_ref(), inbound("/stop"))
                await handler.handle(chat_ref(), tap(adapter.tokens[-1], "implement_plan"))
        assert not owner.task.done() and cleanups == 1
    finally:
        release.set()
    assert await wait_for(lambda: owner.released.is_set())
    assert llm.stream.await_count == 1
    assert epoch_ids(handler, entry.session_id) == epochs
    if reset is not None:
        await asyncio.wait_for(reset, 10)
        assert handler._registry.get(chat_ref()) is None
    if cancel == "shutdown":
        await handler.wait_closed()
    else:
        assert handler._store.load(entry.session_id).history


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", ["stop", "timeout", "none"])
async def test_clear_owns_and_delivers_old_jobs_before_constructing_or_changing_epoch(
    planning_gateway: tuple[AgentTurnHandler, RecordingAdapter, ScriptedLLM],
    monkeypatch: pytest.MonkeyPatch,
    cancel: str,
) -> None:
    handler, adapter, llm = planning_gateway
    entry = await offer(handler, adapter)
    old = entry.runtime.agent
    epochs = epoch_ids(handler, entry.session_id)
    closing, release = asyncio.Event(), asyncio.Event()
    events: list[str] = []

    async def close() -> None:
        events.append("close")
        closing.set()
        await release.wait()

    async def deliver(agent: Any) -> None:
        assert agent is old
        assert epoch_ids(handler, entry.session_id) == epochs
        events.append("deliver")
        agent.append_user_message("old-context job report")

    jobs = SimpleNamespace(
        close=AsyncMock(side_effect=close),
        deliver=AsyncMock(side_effect=deliver),
        set_callbacks=lambda **_kwargs: None,
        active_count=0,
        pending_count=0,
        has_work=False,
    )
    entry.background_jobs = old.background_jobs = jobs
    if cancel == "timeout":
        monkeypatch.setattr(sessions, "LIFECYCLE_WAIT_SECONDS", 0.02)
    llm._text = "Implemented after job settlement."
    await handler.handle(chat_ref(), tap(adapter.tokens[-1], "implement_plan_clear"))
    await asyncio.wait_for(closing.wait(), 10)
    owner = handler._operations[chat_ref().key]
    try:
        assert owner.cleaning and owner.entry is entry and owner.partial_agent is None
        if cancel == "stop":
            await handler.handle(chat_ref(), inbound("/stop"))
        if cancel != "none":
            assert await wait_for(lambda: entry.plan_approval_task is None)
        assert entry.runtime.agent is old
        assert epoch_ids(handler, entry.session_id) == epochs
        assert handler._operations[chat_ref().key] is owner
    finally:
        release.set()
    assert await wait_for(lambda: owner.released.is_set())
    assert events == ["close", "deliver"]
    assert jobs.close.await_count == jobs.deliver.await_count == 1
    if cancel == "none":
        assert await wait_for(lambda: entry.plan_approval_task is None)
        assert isinstance(entry.runtime.agent, CoderAgent)
        assert entry.background_jobs is not jobs
        assert old.background_jobs is None
        assert len(epoch_ids(handler, entry.session_id)) == len(epochs) + 1
        assert llm.stream.await_count == 2
        assert "old-context job report" not in [message.get_text_content() for message in entry.runtime.agent.history]
    else:
        assert entry.runtime.agent is old
        assert entry.background_jobs is old.background_jobs and entry.background_jobs is not jobs
        assert entry.background_jobs.recorder is old.session_recorder
        assert epoch_ids(handler, entry.session_id) == epochs
        assert llm.stream.await_count == 1
        assert entry.record.plan_pending and entry.record.interaction_mode == "plan"
        assert handler._store.load(entry.session_id).history


@pytest.mark.asyncio
async def test_stop_after_adoption_keeps_retired_cleanup_owned_and_never_starts_implementation(
    planning_gateway: tuple[AgentTurnHandler, RecordingAdapter, ScriptedLLM], monkeypatch: pytest.MonkeyPatch
) -> None:
    handler, adapter, llm = planning_gateway
    entry = await offer(handler, adapter)
    old = entry.runtime.agent
    cleaning, release = asyncio.Event(), asyncio.Event()
    cleanup = old.cleanup

    async def blocked_cleanup() -> None:
        cleaning.set()
        await release.wait()
        await cleanup()

    monkeypatch.setattr(old, "cleanup", blocked_cleanup)
    await handler.handle(chat_ref(), tap(adapter.tokens[-1], "implement_plan"))
    await asyncio.wait_for(cleaning.wait(), 10)
    owner = handler._operations[chat_ref().key]
    try:
        assert entry.runtime.agent is not old and isinstance(entry.runtime.agent, CoderAgent)
        assert owner.cleaning and owner.partial_agent is old
        for _ in range(3):
            await handler.handle(chat_ref(), inbound("/stop"))
        assert await wait_for(lambda: entry.plan_approval_task is None)
        assert owner.task is not None and not owner.task.done() and not owner.task.cancelling()
        assert llm.stream.await_count == 1 and entry.record.plan_pending
    finally:
        release.set()
    assert await wait_for(lambda: owner.released.is_set())
    assert llm.stream.await_count == 1 and handler._store.load(entry.session_id).plan_pending


@pytest.mark.asyncio
@pytest.mark.parametrize("interrupt", ["stop", "input"])
async def test_implementation_turn_remains_owned_through_stop_or_queued_input(
    planning_gateway: tuple[AgentTurnHandler, RecordingAdapter, ScriptedLLM], interrupt: str
) -> None:
    handler, adapter, llm = planning_gateway
    entry = await offer(handler, adapter)
    token = adapter.tokens[-1]
    llm._text = "Implementation is still streaming."
    llm._block = asyncio.Event()
    await handler.handle(chat_ref(), tap(token, "implement_plan"))
    assert await wait_for(lambda: any(text == llm._text for _, text in adapter.sent))
    implementation = entry.plan_approval_task
    assert implementation is not None and not implementation.done() and entry.turn_lock.locked()
    assert isinstance(entry.runtime.agent, CoderAgent)
    assert not entry.record.plan_pending and not entry.record.plan_reofferable
    llm._text, llm._block = "Follow-up completed.", None
    if interrupt == "stop":
        await handler.handle(chat_ref(), inbound("/stop"))
        assert await wait_for(lambda: entry.plan_approval_task is None)
    await handler.handle(chat_ref(), inbound("Follow up in build mode."))
    assert await wait_for(lambda: any(text == "Follow-up completed." for _, text in adapter.sent))
    await wait_for_turn_done(handler, chat_ref())
    assert implementation.done() and implementation.cancelled()
    assert entry.plan_approval_task is None and not entry.turn_lock.locked()
    assert isinstance(entry.runtime.agent, CoderAgent)
    assert llm.stream.await_count == 3
    texts = [message.get_text_content() for message in entry.runtime.agent.history]
    assert texts.count(build_implement_plan_prompt(PLAN)) == 1
    assert texts[-2:] == ["Follow up in build mode.", "Follow-up completed."]
    await handler.handle(chat_ref(), tap(token, "implement_plan_clear"))
    assert llm.stream.await_count == 3 and not handler._operations
