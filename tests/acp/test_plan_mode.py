"""Plan/build mode tests: session modes, mode switching, plan updates."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, cast

import pytest
from acp.exceptions import RequestError
from acp.interfaces import Client
from acp.schema import (
    AgentPlanUpdate,
    ConfigOptionUpdate,
    CurrentModeUpdate,
    SessionConfigOptionSelect,
    SetSessionModeResponse,
)

from kolega_code.acp.agent_factory import MODE_BUILD, MODE_PLAN, agent_class_for
from kolega_code.events import AgentEvent
from kolega_code.acp.server import AcpAgent

from tests.acp.test_server import StreamingLLM, _FakeConn, _FakeFactory, _make_agent, _make_session, _text_event


def test_agent_class_choice() -> None:
    from kolega_code.agent.planningagent import PlanningAgent

    assert agent_class_for(MODE_PLAN) is PlanningAgent
    assert agent_class_for(MODE_BUILD).__name__ == "CoderAgent"
    assert agent_class_for("anything-else").__name__ == "CoderAgent"


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
async def test_set_session_mode_build_keeps_task_list_view(tmp_path: Path) -> None:
    session, _cm = _make_session(tmp_path, StreamingLLM())
    factory = _FakeFactory(session)
    conn = _FakeConn()
    agent = AcpAgent(factory=cast(Any, factory))
    agent.on_connect(cast(Client, conn))
    new_session = await agent.new_session(cwd=str(tmp_path))

    await agent.set_session_mode(new_session.session_id, MODE_BUILD)

    # The plan view mirrors the shared task list, which survives mode switches
    # in the TUI; switching modes must not clear it.
    assert not any(isinstance(update, AgentPlanUpdate) for update in conn.updates)


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
async def test_plan_turn_does_not_populate_task_list_view(tmp_path: Path) -> None:
    plan_text = "## Investigate the repo\nDetails.\n## Propose a design\nMore details."
    session, _cm = _make_session(tmp_path, StreamingLLM(events=[_text_event(plan_text)]))
    session.record.interaction_mode = MODE_PLAN
    holder = {"plan": plan_text}
    setattr(session.agent, "consume_completed_plan", lambda: holder.pop("plan", None))
    agent, conn = _make_agent(session)
    new_session = await agent.new_session(cwd=str(tmp_path))

    await agent.prompt(session_id=new_session.session_id, prompt=[{"type": "text", "text": "make a plan"}])  # pyright: ignore[reportArgumentType]

    # The plan and the task list are distinct in the TUI; the plan must not
    # feed the editor's plan view (which mirrors the task list only). The plan
    # still reaches the user through the approval prompt.
    assert not any(isinstance(update, AgentPlanUpdate) for update in conn.updates)
    assert conn.permission_calls
    assert conn.permission_calls[0][1].title == "Approve implementation plan?"


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
    from kolega_code.acp.agent_factory import CONFIG_MODE, AgentFactory

    plan_text = "## Step one\n## Step two"
    session, _cm = _make_session(tmp_path, StreamingLLM(events=[_text_event(plan_text)]))
    session.record.interaction_mode = MODE_PLAN
    holder = {"plan": plan_text}
    setattr(session.agent, "consume_completed_plan", lambda: holder.pop("plan", None))
    factory = _FakeFactory(session)
    # The fake's config surface reflects the record live, so the pushed
    # option update after the server-side switch shows the new mode.
    factory.config_options_for = lambda s: [AgentFactory._mode_option(s)]  # type: ignore[method-assign]
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
    assert not any(isinstance(u, AgentPlanUpdate) for u in conn.updates)
    # The mode select is a config option in Zed; the approval must push a
    # refreshed option set or the client keeps showing the stale mode.
    option_updates = [u for u in conn.updates if isinstance(u, ConfigOptionUpdate)]
    assert option_updates
    mode_selects = [
        option
        for option in option_updates[-1].config_options
        if isinstance(option, SessionConfigOptionSelect) and option.id == CONFIG_MODE
    ]
    assert mode_selects and mode_selects[0].current_value == MODE_BUILD


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


def test_task_list_volatile_section_mirrors_record(tmp_path: Path) -> None:
    from kolega_code.acp.agent_factory import AgentFactory

    session, _cm = _make_session(tmp_path, StreamingLLM())
    provider = AgentFactory._task_list_volatile_section(session.record)

    assert provider().text == ""

    session.record.task_list_markdown = "- [ ] do the thing\n- [x] done thing"
    section = provider()
    assert section.key == "task_list"
    assert section.text == "- [ ] do the thing\n- [x] done thing"


def test_task_list_extension_matches_tui_split(tmp_path: Path) -> None:
    from kolega_code.agent.tool_definitions import tool_description_asset

    from kolega_code.acp.agent_factory import AgentFactory

    session, _cm = _make_session(tmp_path, StreamingLLM())
    holder: dict[str, Any] = {}
    build = AgentFactory._task_list_extension(session.record, holder, MODE_BUILD)
    plan = AgentFactory._task_list_extension(session.record, holder, MODE_PLAN)

    assert set(build.tools) == {"get_task_list", "update_task_list"}
    assert set(plan.tools) == {"get_task_list"}
    assert build.tool_descriptions["get_task_list"] == tool_description_asset("get_task_list")
    assert plan.tool_descriptions["get_task_list"] == tool_description_asset("get_task_list_readonly")


@pytest.mark.asyncio
async def test_rebuild_agent_persists_live_history_first(tmp_path: Path) -> None:
    from unittest.mock import AsyncMock

    from kolega_code.acp.agent_factory import AgentFactory
    from kolega_code.cli.session_store import SessionStore
    from kolega_code.llm.models import Message, MessageHistory, TextBlock

    session, _cm = _make_session(tmp_path, StreamingLLM())
    # The approval flow rebuilds mid-turn: the live agent has the just-finished
    # plan turn, while the persisted record still holds the pre-turn history.
    session.agent.history = MessageHistory()
    session.agent.history.append(Message(role="user", content=[TextBlock(text="plan turn text")]))
    assert session.record.history == []

    captured: dict[str, Any] = {}

    class _StubFactory(AgentFactory):
        async def _construct_agent(
            self,
            record: Any,
            config: Any,
            *,
            restore: bool,
            permission_callback: Any,
            permission_mode: Any = None,
        ) -> tuple[Any, Any]:
            captured["history"] = record.history
            return session.agent, session.manager

    # The real factory registers records in the store before persisting; do
    # the same here so persist() can save.
    SessionStore().create(Path(tmp_path), "cli", {}, session_id=session.record.session_id)
    session.agent.cleanup = AsyncMock()  # type: ignore[method-assign]
    factory = _StubFactory()
    await factory._rebuild_agent(session, session.config)

    assert captured["history"] == session.record.history
    assert any("plan turn text" in str(msg) for msg in captured["history"])
