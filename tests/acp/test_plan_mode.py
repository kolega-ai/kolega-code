"""Plan/build mode tests: session modes, mode switching, plan updates."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, cast

import pytest
from acp.exceptions import RequestError
from acp.interfaces import Client
from acp.schema import AgentPlanUpdate, CurrentModeUpdate, SetSessionModeResponse

from kolega_code.acp.agent_factory import MODE_BUILD, MODE_PLAN, agent_class_for
from kolega_code.events import AgentEvent
from kolega_code.acp.plans import plan_entries_from_markdown
from kolega_code.acp.server import AcpAgent

from tests.acp.test_server import StreamingLLM, _FakeConn, _FakeFactory, _make_agent, _make_session, _text_event


def test_agent_class_choice() -> None:
    from kolega_code.agent.planningagent import PlanningAgent

    assert agent_class_for(MODE_PLAN) is PlanningAgent
    assert agent_class_for(MODE_BUILD).__name__ == "CoderAgent"
    assert agent_class_for("anything-else").__name__ == "CoderAgent"


def test_plan_entries_from_markdown_parses_headings() -> None:
    text = "# Big plan\n\n## Step one\nbody\n## Step two\nmore\n### Nested"
    entries = plan_entries_from_markdown(text)
    assert [entry.content for entry in entries] == ["Big plan", "Step one", "Step two", "Nested"]
    assert all(entry.status == "pending" for entry in entries)


def test_plan_entries_from_markdown_falls_back_to_single_entry() -> None:
    entries = plan_entries_from_markdown("A plan without headings, just prose.")
    assert [entry.content for entry in entries] == ["A plan without headings, just prose."]


def test_plan_entries_from_markdown_empty() -> None:
    assert plan_entries_from_markdown("") == []


@pytest.mark.asyncio
async def test_new_session_does_not_advertise_modes_field(tmp_path: Path) -> None:
    session, _cm = _make_session(tmp_path, StreamingLLM())
    agent = AcpAgent(factory=cast(Any, _FakeFactory(session)))
    agent.on_connect(cast(Client, _FakeConn()))

    response = await agent.new_session(cwd=str(tmp_path))

    # Mode switching rides the config-options surface (Zed 1.16 renders either
    # config options or the modes/model selector surface, never both).
    assert response.modes is None


@pytest.mark.asyncio
async def test_set_session_mode_switches_and_updates_client(tmp_path: Path) -> None:
    session, _cm = _make_session(tmp_path, StreamingLLM())
    factory = _FakeFactory(session)
    conn = _FakeConn()
    agent = AcpAgent(factory=cast(Any, factory))
    agent.on_connect(cast(Client, conn))
    new_session = await agent.new_session(cwd=str(tmp_path))

    response = await agent.set_session_mode(new_session.session_id, MODE_PLAN)

    assert isinstance(response, SetSessionModeResponse)
    assert factory.interaction_mode_calls == [(new_session.session_id, MODE_PLAN)]
    updates = conn.updates
    assert any(isinstance(update, CurrentModeUpdate) and update.current_mode_id == MODE_PLAN for update in updates)
    assert not any(isinstance(update, AgentPlanUpdate) for update in updates)  # entering plan clears nothing


@pytest.mark.asyncio
async def test_set_session_mode_build_clears_plan(tmp_path: Path) -> None:
    session, _cm = _make_session(tmp_path, StreamingLLM())
    factory = _FakeFactory(session)
    conn = _FakeConn()
    agent = AcpAgent(factory=cast(Any, factory))
    agent.on_connect(cast(Client, conn))
    new_session = await agent.new_session(cwd=str(tmp_path))

    await agent.set_session_mode(new_session.session_id, MODE_BUILD)

    plans = [update for update in conn.updates if isinstance(update, AgentPlanUpdate)]
    assert len(plans) == 1
    assert plans[0].entries == []


@pytest.mark.asyncio
async def test_set_session_mode_rejects_unknown_mode(tmp_path: Path) -> None:
    session, _cm = _make_session(tmp_path, StreamingLLM())
    agent = AcpAgent(factory=cast(Any, _FakeFactory(session)))
    new_session = await agent.new_session(cwd=str(tmp_path))

    with pytest.raises(RequestError):
        await agent.set_session_mode(new_session.session_id, "vibe")


@pytest.mark.asyncio
async def test_set_session_mode_rejects_active_turn(tmp_path: Path) -> None:
    session, _cm = _make_session(tmp_path, StreamingLLM())
    agent = AcpAgent(factory=cast(Any, _FakeFactory(session)))
    new_session = await agent.new_session(cwd=str(tmp_path))

    async def _forever() -> None:
        await asyncio.Event().wait()

    registered = agent._sessions[new_session.session_id]  # noqa: SLF001
    registered.turn_task = asyncio.create_task(_forever())
    try:
        with pytest.raises(RequestError):
            await agent.set_session_mode(new_session.session_id, MODE_PLAN)
    finally:
        registered.turn_task.cancel()


@pytest.mark.asyncio
async def test_plan_turn_emits_plan_update(tmp_path: Path) -> None:
    plan_text = "## Investigate the repo\nDetails.\n## Propose a design\nMore details."
    session, _cm = _make_session(tmp_path, StreamingLLM(events=[_text_event(plan_text)]))
    session.record.interaction_mode = MODE_PLAN
    agent, conn = _make_agent(session)
    new_session = await agent.new_session(cwd=str(tmp_path))

    await agent.prompt(session_id=new_session.session_id, prompt=[{"type": "text", "text": "make a plan"}])  # pyright: ignore[reportArgumentType]

    plans = [update for update in conn.updates if isinstance(update, AgentPlanUpdate)]
    assert plans
    assert [entry.content for entry in plans[-1].entries] == ["Investigate the repo", "Propose a design"]


def test_task_entries_from_markdown_maps_checkboxes() -> None:
    from kolega_code.acp.plans import task_entries_from_markdown

    entries = task_entries_from_markdown("- [x] Done thing\n- [ ] Pending thing\n- plain item")
    assert [(entry.content, entry.status) for entry in entries] == [
        ("Done thing", "completed"),
        ("Pending thing", "pending"),
        ("plain item", "pending"),
    ]


@pytest.mark.asyncio
async def test_bridge_maps_task_list_update_to_plan_view() -> None:
    from kolega_code.acp.bridge import AcpBridge

    conn = _FakeConn()
    bridge = AcpBridge(cast(Client, conn))
    event = AgentEvent(
        sender="agent",
        event_type="chat_message",
        content={"message_type": "task_list_update", "text": "- [x] a\n- [ ] b", "tool_call_id": "c1"},
        is_streaming=False,
    )
    await bridge.handle_event("s1", event)

    plans = [update for update in conn.updates if isinstance(update, AgentPlanUpdate)]
    assert len(plans) == 1
    assert [(entry.content, entry.status) for entry in plans[0].entries] == [
        ("a", "completed"),
        ("b", "pending"),
    ]


@pytest.mark.asyncio
async def test_plan_approval_implement_switches_to_build(tmp_path: Path) -> None:
    from acp.schema import CurrentModeUpdate

    plan_text = "## Step one\n## Step two"
    session, _cm = _make_session(tmp_path, StreamingLLM(events=[_text_event(plan_text)]))
    session.record.interaction_mode = MODE_PLAN
    holder = {"plan": plan_text}
    setattr(session.agent, "consume_completed_plan", lambda: holder.pop("plan", None))
    factory = _FakeFactory(session)
    conn = _FakeConn()
    agent = AcpAgent(factory=cast(Any, factory))
    agent.on_connect(cast(Client, conn))
    new_session = await agent.new_session(cwd=str(tmp_path))

    await agent.prompt(session_id=new_session.session_id, prompt=[{"type": "text", "text": "make a plan"}])  # pyright: ignore[reportArgumentType]

    assert session.record.plan_pending is False
    assert session.record.latest_plan_markdown == plan_text
    assert factory.interaction_mode_calls == [(new_session.session_id, MODE_BUILD)]
    modes = [u.current_mode_id for u in conn.updates if isinstance(u, CurrentModeUpdate)]
    assert modes[-1] == MODE_BUILD
    plans = [u for u in conn.updates if isinstance(u, AgentPlanUpdate)]
    assert [entry.content for entry in plans[0].entries] == ["Step one", "Step two"]


@pytest.mark.asyncio
async def test_plan_approval_discuss_keeps_mode_and_sets_reofferable(tmp_path: Path) -> None:
    from acp.schema import RequestPermissionResponse

    plan_text = "## Step one\n## Step two"
    session, _cm = _make_session(tmp_path, StreamingLLM(events=[_text_event(plan_text)]))
    session.record.interaction_mode = MODE_PLAN
    holder = {"plan": plan_text}
    setattr(session.agent, "consume_completed_plan", lambda: holder.pop("plan", None))
    factory = _FakeFactory(session)
    conn = _FakeConn()

    async def discuss(*a: Any, **k: Any) -> Any:
        return RequestPermissionResponse.model_validate(
            {"outcome": {"outcome": "selected", "optionId": "discuss_plan"}}
        )

    conn.request_permission = discuss  # type: ignore[method-assign]
    agent = AcpAgent(factory=cast(Any, factory))
    agent.on_connect(cast(Client, conn))
    new_session = await agent.new_session(cwd=str(tmp_path))

    await agent.prompt(session_id=new_session.session_id, prompt=[{"type": "text", "text": "make a plan"}])  # pyright: ignore[reportArgumentType]

    assert session.record.plan_pending is False
    assert session.record.plan_reofferable is True
    assert factory.interaction_mode_calls == []


@pytest.mark.asyncio
async def test_plan_approval_clear_wipes_history(tmp_path: Path) -> None:
    from acp.schema import RequestPermissionResponse

    plan_text = "## Step one"
    session, _cm = _make_session(tmp_path, StreamingLLM(events=[_text_event(plan_text)]))
    session.record.interaction_mode = MODE_PLAN
    holder = {"plan": plan_text}
    setattr(session.agent, "consume_completed_plan", lambda: holder.pop("plan", None))
    factory = _FakeFactory(session)
    conn = _FakeConn()

    async def implement_clear(*a: Any, **k: Any) -> Any:
        return RequestPermissionResponse.model_validate(
            {"outcome": {"outcome": "selected", "optionId": "implement_plan_clear"}}
        )

    conn.request_permission = implement_clear  # type: ignore[method-assign]
    agent = AcpAgent(factory=cast(Any, factory))
    agent.on_connect(cast(Client, conn))
    new_session = await agent.new_session(cwd=str(tmp_path))

    await agent.prompt(session_id=new_session.session_id, prompt=[{"type": "text", "text": "make a plan"}])  # pyright: ignore[reportArgumentType]

    assert session.record.plan_pending is False
    assert not any("make a plan" in msg.get_text_content() for msg in session.agent.history)
    assert sum("Step one" in msg.get_text_content() for msg in session.agent.history) == 1
    assert factory.interaction_mode_calls == [(new_session.session_id, MODE_BUILD)]


@pytest.mark.asyncio
async def test_plan_approval_prompt_contains_full_plan(tmp_path: Path) -> None:
    plan_text = "## Step one\n" + ("body text. " * 80) + "## Step two\n" + ("more body. " * 80)
    assert len(plan_text) > 400
    session, _cm = _make_session(tmp_path, StreamingLLM(events=[_text_event(plan_text)]))
    session.record.interaction_mode = MODE_PLAN
    holder = {"plan": plan_text}
    setattr(session.agent, "consume_completed_plan", lambda: holder.pop("plan", None))
    conn = _FakeConn()
    agent = AcpAgent(factory=cast(Any, _FakeFactory(session)))
    agent.on_connect(cast(Client, conn))
    new_session = await agent.new_session(cwd=str(tmp_path))

    await agent.prompt(session_id=new_session.session_id, prompt=[{"type": "text", "text": "make a plan"}])  # pyright: ignore[reportArgumentType]

    assert conn.permission_calls
    content = conn.permission_calls[0][1].content[0].content
    assert content.type == "text"
    assert content.text == plan_text
