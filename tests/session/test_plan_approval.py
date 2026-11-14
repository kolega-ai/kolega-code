import asyncio
from pathlib import Path
from typing import Any

import pytest

from kolega_code.events import AgentEvent, KnownEventType
from kolega_code.permissions import PermissionKind, PermissionMode, PermissionRule, ProjectPermissionStore
from kolega_code.session.control import ControlChannel, ControlLeaseError
from kolega_code.session.projection import replay
from kolega_code.session.runtime import SessionRuntime


def make_runtime(tmp_path: Path, *, timeout: float = 30.0) -> tuple[SessionRuntime, list[AgentEvent]]:
    events: list[AgentEvent] = []

    async def emit(event: AgentEvent) -> None:
        events.append(event)

    runtime = SessionRuntime(
        session_id="plan-session",
        project_path=tmp_path,
        control=ControlChannel(session_id="plan-session", emit=emit, timeout=timeout),
    )
    return runtime, events


async def pending_id(runtime: SessionRuntime) -> str:
    async def wait() -> str:
        while not runtime.control.pending():
            await asyncio.sleep(0)
        return runtime.control.pending()[0].request_id

    return await asyncio.wait_for(wait(), timeout=10)


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["implement_plan", "implement_plan_clear", "discuss_plan"])
async def test_plan_choices_use_recorded_control_roundtrip(tmp_path: Path, action: Any) -> None:
    runtime, events = make_runtime(tmp_path)
    runtime.control.acquire("editor")
    plan = "## Full plan\n\nLiteral <details> & **text**\n" + "Not truncated. " * 1000
    task = asyncio.create_task(runtime.request_plan_approval(plan))
    request_id = await pending_id(runtime)
    request = runtime.control.pending()[0]
    assert request.kind == "plan_approval"
    assert request.payload == {
        "plan": plan,
        "options": [
            {"id": "implement_plan", "label": "Implement plan"},
            {"id": "implement_plan_clear", "label": "Clear context and implement plan"},
            {"id": "discuss_plan", "label": "Discuss further"},
        ],
    }
    assert request.default == {"action": None}
    assert runtime.answer_plan_approval(request_id, action, client_id="editor")
    assert await task == action
    await asyncio.sleep(0)
    assert [event.event_type for event in events] == [
        KnownEventType.CONTROL_REQUESTED,
        KnownEventType.CONTROL_RESOLVED,
    ]
    assert events[0].content["payload"]["plan"] == plan
    assert events[1].content["response"] == {"action": action}
    assert events[1].content["reason"] == "answered"
    assert not runtime.answer_plan_approval(request_id, action, client_id="editor")


@pytest.mark.asyncio
@pytest.mark.parametrize("settlement", ["no_controller", "timeout", "controller_left", "cancelled"])
async def test_unanswered_plan_never_defaults_to_implementation(tmp_path: Path, settlement: str) -> None:
    runtime, events = make_runtime(tmp_path, timeout=0.01 if settlement == "timeout" else 30)
    if settlement != "no_controller":
        runtime.control.acquire("editor")
    task = asyncio.create_task(runtime.request_plan_approval("Keep this pending."))
    if settlement == "controller_left":
        await pending_id(runtime)
        runtime.control.release("editor")
    elif settlement == "cancelled":
        request_id = await pending_id(runtime)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not runtime.answer_plan_approval(request_id, "implement_plan", client_id="editor")
    if settlement != "cancelled":
        assert await asyncio.wait_for(task, timeout=10) is None
    await asyncio.sleep(0)
    assert not runtime.control.pending()
    assert events[-1].content["reason"] == settlement


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response", [{"action": True}, {"action": []}, {"action": "allow"}, {}, {"allowed": True}, None, [], True]
)
async def test_malformed_generic_response_is_not_plan_authority(tmp_path: Path, response: Any) -> None:
    runtime, _ = make_runtime(tmp_path)
    runtime.control.acquire("editor")
    task = asyncio.create_task(runtime.request_plan_approval("Do not infer approval."))
    request_id = await pending_id(runtime)
    assert runtime.control.respond(request_id, response, client_id="editor")
    assert await task is None


@pytest.mark.asyncio
async def test_readonly_client_and_wrong_prompt_cannot_approve_plan(tmp_path: Path) -> None:
    runtime, _ = make_runtime(tmp_path)
    runtime.control.acquire("editor")
    task = asyncio.create_task(runtime.request_plan_approval("Lease-bound approval."))
    request_id = await pending_id(runtime)
    with pytest.raises(ControlLeaseError):
        runtime.answer_plan_approval(request_id, "implement_plan", client_id="viewer")
    assert runtime.control.is_pending(request_id)
    runtime.answer_plan_approval(request_id, None, client_id="editor")
    assert await task is None
    other = asyncio.create_task(
        runtime.control.request("question", {"question": "Continue?"}, default={"answer": None})
    )
    other_id = await pending_id(runtime)
    assert not runtime.answer_plan_approval(other_id, "implement_plan", client_id="editor")
    runtime.answer_question(other_id, {"answer": "No"}, client_id="editor")
    assert await other == {"answer": "No"}


@pytest.mark.asyncio
async def test_agent_replacement_invalidates_predecessor_approval(tmp_path: Path) -> None:
    runtime, _ = make_runtime(tmp_path)
    runtime.adopt(object())
    runtime.control.acquire("editor")
    task = asyncio.create_task(runtime.request_plan_approval("Old generation."))
    request_id = await pending_id(runtime)
    runtime.adopt(object())
    assert runtime.answer_plan_approval(request_id, "implement_plan", client_id="editor")
    assert await task is None


@pytest.mark.asyncio
async def test_unoffered_or_unknown_action_cannot_settle_a_plan_request(tmp_path: Path) -> None:
    runtime, _ = make_runtime(tmp_path)
    runtime.control.acquire("editor")
    task = asyncio.create_task(
        runtime.control.request(
            "plan_approval",
            {"plan": "Discussion only.", "options": [{"id": "discuss_plan", "label": "Discuss further"}]},
            default={"action": None},
        )
    )
    request_id = await pending_id(runtime)
    assert not runtime.answer_plan_approval(request_id, "implement_plan", client_id="editor")
    unknown: Any = "allow_once"
    assert not runtime.answer_plan_approval(request_id, unknown, client_id="editor")
    assert runtime.control.is_pending(request_id)
    assert runtime.answer_plan_approval(request_id, "discuss_plan", client_id="editor")
    assert await task == {"action": "discuss_plan"}


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", [PermissionMode.ASK, PermissionMode.AUTO])
async def test_permission_policy_cannot_approve_implementation(tmp_path: Path, mode: PermissionMode) -> None:
    runtime, _ = make_runtime(tmp_path)
    runtime.set_permission_mode(mode)
    ProjectPermissionStore(tmp_path).add_rule(
        PermissionRule.create(kind=PermissionKind.COMMAND, tool="exec_command", match_type="prefix", pattern="echo ")
    )
    runtime.control.acquire("editor")
    task = asyncio.create_task(runtime.request_plan_approval("Require a separate human decision."))
    request_id = await pending_id(runtime)
    assert not task.done()
    assert runtime.control.pending()[0].kind == "plan_approval"
    runtime.answer_plan_approval(request_id, None, client_id="editor")
    assert await task is None


@pytest.mark.asyncio
@pytest.mark.parametrize("eager", [False, True])
async def test_immediate_plan_response_is_not_lost(tmp_path: Path, eager: bool) -> None:
    if eager and not hasattr(asyncio, "eager_task_factory"):
        pytest.skip("Python 3.12+ eager tasks")
    loop = asyncio.get_running_loop()
    previous_factory = loop.get_task_factory()
    events: list[AgentEvent] = []

    async def emit(event: AgentEvent) -> None:
        events.append(event)
        if event.event_type == KnownEventType.CONTROL_REQUESTED:
            assert runtime.answer_plan_approval(event.content["request_id"], "implement_plan_clear", client_id="editor")

    runtime = SessionRuntime(
        session_id="immediate",
        project_path=tmp_path,
        control=ControlChannel(session_id="immediate", emit=emit),
    )
    runtime.control.acquire("editor")
    try:
        if eager:
            loop.set_task_factory(getattr(asyncio, "eager_task_factory"))
        task = asyncio.create_task(runtime.request_plan_approval("An immediately answered plan."))
        assert await task == "implement_plan_clear"
        await asyncio.sleep(0)
        assert [event.event_type for event in events] == [
            KnownEventType.CONTROL_REQUESTED,
            KnownEventType.CONTROL_RESOLVED,
        ]
        assert not runtime.control.pending()
    finally:
        loop.set_task_factory(previous_factory)


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_during_announcement", [False, True])
async def test_plan_cancellation_cannot_leave_a_request_owner(tmp_path: Path, cancel_during_announcement: bool) -> None:
    entered = asyncio.Event()
    events: list[AgentEvent] = []

    async def emit(event: AgentEvent) -> None:
        events.append(event)
        if event.event_type == KnownEventType.CONTROL_REQUESTED:
            entered.set()
            await asyncio.Event().wait()

    runtime = SessionRuntime(
        session_id="cancelled",
        project_path=tmp_path,
        control=ControlChannel(session_id="cancelled", emit=emit),
    )
    runtime.control.acquire("editor")
    task = asyncio.create_task(runtime.request_plan_approval("No orphan approval."))
    if cancel_during_announcement:
        await asyncio.wait_for(entered.wait(), timeout=10)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.sleep(0)
    assert not runtime.control.pending()
    if cancel_during_announcement:
        assert events[-1].event_type == KnownEventType.CONTROL_RESOLVED
        assert events[-1].content["response"] == {"action": None}
        assert events[-1].content["reason"] == "cancelled"
        assert not runtime.answer_plan_approval(events[0].content["request_id"], "implement_plan", client_id="editor")
    else:
        assert not events


@pytest.mark.asyncio
@pytest.mark.parametrize("plan", ["", " \n\t"])
async def test_empty_plan_does_not_create_control_request(tmp_path: Path, plan: str) -> None:
    runtime, events = make_runtime(tmp_path)
    with pytest.raises(ValueError, match="nonempty"):
        await runtime.request_plan_approval(plan)
    assert not events
    assert not runtime.control.pending()


@pytest.mark.asyncio
async def test_failed_request_announcement_still_settles_between_turns_idle(tmp_path: Path) -> None:
    events = [
        AgentEvent(
            session_id="plan-session",
            sender="system",
            event_type=KnownEventType.TURN_STARTED,
            content={"turn_id": "planning", "user_text": "Plan the change."},
        ),
        AgentEvent(
            session_id="plan-session",
            sender="system",
            event_type=KnownEventType.TURN_ENDED,
            content={"turn_id": "planning", "status": "completed"},
        ),
    ]
    resolved = asyncio.Event()

    async def emit(event: AgentEvent) -> None:
        if event.event_type == KnownEventType.CONTROL_REQUESTED:
            raise RuntimeError("Request announcement unavailable")
        events.append(event)
        resolved.set()

    runtime = SessionRuntime(
        session_id="plan-session",
        project_path=tmp_path,
        control=ControlChannel(session_id="plan-session", emit=emit),
    )
    assert await runtime.request_plan_approval("Keep the plan pending.") is None
    await asyncio.wait_for(resolved.wait(), timeout=10)
    assert [event.event_type for event in events] == [
        KnownEventType.TURN_STARTED,
        KnownEventType.TURN_ENDED,
        KnownEventType.CONTROL_RESOLVED,
    ]
    state = replay(events)
    assert state.activity == "idle"
    assert not state.prompts
    assert all(turn.ended_ms is not None for turn in state.turns)
