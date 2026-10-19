"""Dispatch boundary regressions through the real registered tool surface."""

import asyncio
import json
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio

from kolega_code.agent.coder import CoderAgent
from kolega_code.agent.generalagent import GeneralAgent
from kolega_code.agent.prompt_provider import AgentMode
from kolega_code.cli.session_store import SessionStore
from kolega_code.config import AgentConfig, ModelConfig, ModelProvider
from kolega_code.events import AgentConnectionManager
from kolega_code.hooks import HookConfig, HookDispatcher, HookEvent, HookMatcher, HookOutcome, HookSpec
from kolega_code.llm.models import ToolCall, ToolResult
from kolega_code.permissions import PermissionMode
from kolega_code.session.background import BackgroundAgentManager


@dataclass
class ChildPlan:
    entered: asyncio.Event = field(default_factory=asyncio.Event)
    release: asyncio.Event = field(default_factory=asyncio.Event)
    child: Any = None


@dataclass
class DispatchBoundary:
    root: CoderAgent
    jobs: BackgroundAgentManager
    store: SessionStore
    session_id: str
    plans: dict[str, ChildPlan]

    async def dispatch(self, task: str, **inputs: Any) -> ToolResult:
        return await asyncio.wait_for(
            self.root.execute_single_tool(
                ToolCall(
                    id=f"provider-{task}",
                    execution_id=f"execution-{task}",
                    name="dispatch_agent",
                    input={"agent_type": "general", "task": task, **inputs},
                )
            ),
            timeout=5,
        )


@pytest_asyncio.fixture
async def boundary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[DispatchBoundary]:
    model = ModelConfig(provider=ModelProvider.ANTHROPIC, model="claude-opus-5", thinking_effort="high")
    config = AgentConfig(
        anthropic_api_key="synthetic-not-a-key",
        long_context_config=model,
        fast_config=model.model_copy(deep=True),
    )
    config.lsp.enabled = False
    config.skills_enabled = False
    config.eval_enabled = False
    store = SessionStore(tmp_path / "state")
    session = store.create(tmp_path, "code", {})
    root = CoderAgent(
        project_path=tmp_path,
        workspace_id="boundary-workspace",
        thread_id=session.thread_id,
        connection_manager=AsyncMock(spec=AgentConnectionManager),
        config=config,
        session_recorder=store.recorder(session.session_id),
        memory_enabled=False,
        agent_mode=AgentMode.CLI,
        permission_mode=PermissionMode.ASK,
    )
    jobs = BackgroundAgentManager(root.session_recorder, max_running=1)
    root.background_jobs = jobs
    plans: dict[str, ChildPlan] = {}

    async def stream(child: GeneralAgent, task: str) -> AsyncIterator[dict[str, Any]]:
        plan = plans[task.split("\n\nWhen finished", 1)[0]]
        plan.child = child
        plan.entered.set()
        await plan.release.wait()
        yield {"type": "response", "content": "child report", "complete": True}

    monkeypatch.setattr(GeneralAgent, "process_message_stream", stream)
    monkeypatch.setattr(GeneralAgent, "recap_agent_outcome", AsyncMock(return_value="child report"))
    try:
        yield DispatchBoundary(root, jobs, store, session.session_id, plans)
    finally:
        for plan in plans.values():
            plan.release.set()
        await asyncio.wait_for(jobs.close(), timeout=5)
        root.background_jobs = None
        await asyncio.wait_for(root.cleanup(), timeout=5)
        store.release_session_locks()


@pytest.mark.asyncio
@pytest.mark.parametrize("background", ["false", "true", 0, 1, None, {}, []])
async def test_registered_dispatch_rejects_non_boolean_before_child_side_effects(
    boundary: DispatchBoundary, background: Any
) -> None:
    plan = boundary.plans["invalid"] = ChildPlan()
    plan.release.set()  # A falsey validation regression must fail, not hang.
    result = await boundary.dispatch("invalid", background=background)
    await asyncio.wait_for(boundary.jobs.wait(), timeout=5)
    assert result.is_error
    assert "background must be a boolean" in result.content
    assert plan.child is None
    assert not plan.entered.is_set()
    assert boundary.jobs.snapshot() == []
    assert boundary.root.tool_collection is not None
    assert not boundary.root.tool_collection.agent_tool.agents
    events = boundary.store.journal(boundary.session_id).read_events()
    assert not any(event.event_type.startswith(("background.", "agent.")) for event in events)
    broadcasts = boundary.root.connection_manager.broadcast_event.call_args_list
    assert not any(getattr(call.args[0], "sub_agent_info", None) for call in broadcasts)


def stop_dispatcher(marker: str) -> HookDispatcher:
    return HookDispatcher(
        HookConfig(
            entries={HookEvent.SUBAGENT_STOP: [(HookMatcher("*"), [HookSpec("agent", 3, "global", prompt=marker)])]}
        )
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["queued", "running"])
async def test_background_stop_policy_context_capabilities_and_evidence_are_launch_bound(
    boundary: DispatchBoundary, monkeypatch: pytest.MonkeyPatch, phase: str
) -> None:
    root = boundary.root
    launch_thread, launch_path = root.thread_id, root.project_path
    launch_model = root.config.long_context_config.model
    launch_dispatcher = root.hook_dispatcher = stop_dispatcher("launch-hook")
    observed: list[tuple[Any, Any]] = []
    verification = boundary.plans["verification"] = ChildPlan()
    verification.release.set()

    async def hook(event: Any, spec: Any, caps: Any) -> HookOutcome:
        assert spec.prompt == "launch-hook"
        assert event.session_id == launch_thread
        assert event.cwd == str(launch_path)
        assert event.permission_mode == PermissionMode.ASK.value
        assert caps.project_path == launch_path
        assert caps.prompt_runner.__self__.config.long_context_config.model == launch_model
        assert caps.prompt_runner.__self__.sub_agent_context is None
        # Exercise the real agent-hook capability, not only its bound identity.
        assert await caps.agent_runner("verification") == "child report"
        assert verification.child.config.long_context_config.model == launch_model
        observed.append((event, caps))
        return HookOutcome(additional_context="launch augmentation", blocked=True, reason="launch reason")

    monkeypatch.setattr("kolega_code.hooks.dispatcher.run_hook", hook)
    blocker = boundary.plans["blocker"] = ChildPlan()
    work = boundary.plans["work"] = ChildPlan()
    if phase == "queued":
        assert not (await boundary.dispatch("blocker", background=True)).is_error
        await asyncio.wait_for(blocker.entered.wait(), timeout=5)
    receipt = await boundary.dispatch("work", background=True)
    assert not receipt.is_error
    assert isinstance(receipt.content, str)
    job_id = json.loads(receipt.content)["job_id"]
    if phase == "running":
        await asyncio.wait_for(work.entered.wait(), timeout=5)
    else:
        assert boundary.jobs.snapshot(job_id)[0]["status"] == "queued"
        assert work.child is None
    # Mutate the original policy container as well as replacing the dispatcher.
    launch_dispatcher.config.entries.clear()
    root.hook_dispatcher = stop_dispatcher("later-hook")
    root.config.long_context_config.model = "later-model"
    root.thread_id = "later-thread"
    root.project_path = launch_path / "later-project"
    root.permission_mode = PermissionMode.AUTO
    root.sub_agent_context = {"agent_id": "later-parent"}
    try:
        blocker.release.set()
        await asyncio.wait_for(work.entered.wait(), timeout=5)
        assert work.child is not None
        assert work.child.hook_dispatcher.config.specs_for(HookEvent.SUBAGENT_STOP)[0].prompt == "launch-hook"
        work.release.set()
        await asyncio.wait_for(boundary.jobs.wait(), timeout=5)
        expected = "child report\n\nlaunch augmentation\n\n[hook] launch reason"
        assert boundary.jobs.snapshot(job_id)[0]["result"] == expected
        work_events = [event for event, _ in observed if event.payload["sub_agent_info"]["agent_id"] == job_id]
        assert len(work_events) == 1
        assert work_events[0].payload["result"] == "child report"
        assert work_events[0].payload["sub_agent_info"]["parent_tool_call_id"] == "execution-work"
        events = boundary.store.journal(boundary.session_id).read_events()
        assert any(
            event.event_type == "background.job.terminal" and event.payload.get("result") == expected
            for event in events
        )
        assert any(
            event.event_type == "agent.completed"
            and event.agent_id == job_id
            and event.payload.get("summary") == expected
            for event in events
        )
    finally:
        root.thread_id, root.project_path = launch_thread, launch_path


@pytest.mark.asyncio
async def test_background_does_not_enable_stop_hooks_after_launch(
    boundary: DispatchBoundary, monkeypatch: pytest.MonkeyPatch
) -> None:
    launch_dispatcher = boundary.root.hook_dispatcher = HookDispatcher(HookConfig())
    hook = AsyncMock(return_value=HookOutcome(additional_context="later-hook"))
    monkeypatch.setattr("kolega_code.hooks.dispatcher.run_hook", hook)
    blocker = boundary.plans["blocker"] = ChildPlan()
    work = boundary.plans["work"] = ChildPlan()
    assert not (await boundary.dispatch("blocker", background=True)).is_error
    await asyncio.wait_for(blocker.entered.wait(), timeout=5)
    receipt = await boundary.dispatch("work", background=True)
    assert not receipt.is_error
    assert isinstance(receipt.content, str)
    job_id = json.loads(receipt.content)["job_id"]
    assert boundary.jobs.snapshot(job_id)[0]["status"] == "queued"
    launch_dispatcher.config.entries = stop_dispatcher("later-hook").config.entries
    boundary.root.hook_dispatcher = stop_dispatcher("replacement-hook")
    blocker.release.set()
    work.release.set()
    await asyncio.wait_for(boundary.jobs.wait(), timeout=5)
    assert boundary.jobs.snapshot(job_id)[0]["result"] == "child report"
    hook.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("inputs", [{}, {"background": False}])
async def test_foreground_keeps_parent_hook_semantics(
    boundary: DispatchBoundary, monkeypatch: pytest.MonkeyPatch, inputs: dict[str, Any]
) -> None:
    boundary.root.hook_dispatcher = stop_dispatcher("launch-hook")
    plan = boundary.plans["foreground"] = ChildPlan()

    async def hook(event: Any, spec: Any, caps: Any) -> HookOutcome:
        return HookOutcome(additional_context=spec.prompt)

    monkeypatch.setattr("kolega_code.hooks.dispatcher.run_hook", hook)
    operation = asyncio.create_task(boundary.dispatch("foreground", **inputs))
    try:
        await asyncio.wait_for(plan.entered.wait(), timeout=5)
        assert not operation.done()
        assert boundary.jobs.snapshot() == []
        boundary.root.hook_dispatcher = stop_dispatcher("later-hook")
        plan.release.set()
        result = await operation
        assert not result.is_error
        assert result.content == "child report\n\nlater-hook"
    finally:
        plan.release.set()
        await asyncio.gather(operation, return_exceptions=True)
