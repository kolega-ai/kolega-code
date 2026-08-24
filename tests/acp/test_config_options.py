"""Session config option tests: model select + permission toggle wiring."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, cast

import pytest
from acp.exceptions import RequestError
from acp.schema import SessionConfigOptionBoolean, SessionConfigOptionSelect

from kolega_code.acp.agent_factory import CONFIG_MODE, CONFIG_MODEL, CONFIG_PERMISSION_AUTO, AgentFactory
from kolega_code.acp.server import AcpAgent
from kolega_code.acp.session import AcpSession
from kolega_code.cli.session_store import SessionRecord
from kolega_code.permissions import PermissionMode

from tests.agent.compaction_helpers import build_agent
from tests.acp.test_server import StreamingLLM, _FakeConn, _FakeFactory, _make_session

from acp.interfaces import Client  # noqa: E402


def _options(agent: AcpAgent, session: AcpSession) -> list[Any]:
    return agent._factory.config_options_for(session)  # noqa: SLF001


@pytest.mark.asyncio
async def test_new_session_response_advertises_options(tmp_path: Path) -> None:
    session, _cm = _make_session(tmp_path, StreamingLLM())
    factory = _FakeFactory(session)
    factory.config_options = [
        SessionConfigOptionSelect(type="select", id=CONFIG_MODEL, name="Model", current_value="x/y", options=[]),
        SessionConfigOptionBoolean(
            type="boolean", id=CONFIG_PERMISSION_AUTO, name="Auto-approve tools", current_value=False
        ),
    ]
    agent = AcpAgent(factory=cast(Any, factory))
    agent.on_connect(cast(Client, _FakeConn()))

    response = await agent.new_session(cwd=str(tmp_path))

    assert response.config_options == factory.config_options


@pytest.mark.asyncio
async def test_set_config_option_flips_permission_mode(tmp_path: Path) -> None:
    session, _cm = _make_session(tmp_path, StreamingLLM())
    factory = _FakeFactory(session)
    agent = AcpAgent(factory=cast(Any, factory))
    new_session = await agent.new_session(cwd=str(tmp_path))

    first = await agent.set_config_option(CONFIG_PERMISSION_AUTO, new_session.session_id, "auto")
    second = await agent.set_config_option(CONFIG_PERMISSION_AUTO, new_session.session_id, "ask")
    # Legacy clients that persisted the older boolean values keep working.
    await agent.set_config_option(CONFIG_PERMISSION_AUTO, new_session.session_id, True)
    await agent.set_config_option(CONFIG_PERMISSION_AUTO, new_session.session_id, False)

    assert [mode for _, mode in factory.permission_mode_calls] == [
        PermissionMode.AUTO,
        PermissionMode.ASK,
        PermissionMode.AUTO,
        PermissionMode.ASK,
    ]
    assert first.config_options == factory.config_options
    assert second.config_options == factory.config_options


@pytest.mark.asyncio
async def test_set_config_option_applies_model(tmp_path: Path) -> None:
    session, _cm = _make_session(tmp_path, StreamingLLM())
    factory = _FakeFactory(session)
    agent = AcpAgent(factory=cast(Any, factory))
    new_session = await agent.new_session(cwd=str(tmp_path))

    response = await agent.set_config_option(CONFIG_MODEL, new_session.session_id, "deepseek/deepseek-v4-pro")

    assert factory.model_calls == [(new_session.session_id, "deepseek", "deepseek-v4-pro")]
    assert response.config_options == factory.config_options


@pytest.mark.asyncio
async def test_set_config_option_rejects_malformed_model_value(tmp_path: Path) -> None:
    session, _cm = _make_session(tmp_path, StreamingLLM())
    factory = _FakeFactory(session)
    agent = AcpAgent(factory=cast(Any, factory))
    new_session = await agent.new_session(cwd=str(tmp_path))

    with pytest.raises(RequestError):
        await agent.set_config_option(CONFIG_MODEL, new_session.session_id, "no-slash")
    assert factory.model_calls == []


@pytest.mark.asyncio
async def test_set_config_option_rejects_unknown_option(tmp_path: Path) -> None:
    session, _cm = _make_session(tmp_path, StreamingLLM())
    agent = AcpAgent(factory=cast(Any, _FakeFactory(session)))
    new_session = await agent.new_session(cwd=str(tmp_path))

    with pytest.raises(RequestError):
        await agent.set_config_option("nope", new_session.session_id, True)


@pytest.mark.asyncio
async def test_set_config_option_rejects_unknown_session(tmp_path: Path) -> None:
    session, _cm = _make_session(tmp_path, StreamingLLM())
    agent = AcpAgent(factory=cast(Any, _FakeFactory(session)))

    with pytest.raises(RequestError):
        await agent.set_config_option(CONFIG_PERMISSION_AUTO, "missing", True)


@pytest.mark.asyncio
async def test_set_config_option_permission_mode_allowed_mid_turn(tmp_path: Path) -> None:
    session, _cm = _make_session(tmp_path, StreamingLLM())
    factory = _FakeFactory(session)
    agent = AcpAgent(factory=cast(Any, factory))
    new_session = await agent.new_session(cwd=str(tmp_path))

    async def _forever() -> None:
        await asyncio.Event().wait()

    registered = agent._sessions[new_session.session_id]  # noqa: SLF001
    registered.turn_task = asyncio.create_task(_forever())
    try:
        response = await agent.set_config_option(CONFIG_PERMISSION_AUTO, new_session.session_id, "auto")
    finally:
        registered.turn_task.cancel()

    assert [mode for _, mode in factory.permission_mode_calls] == [PermissionMode.AUTO]
    assert response.config_options == factory.config_options


@pytest.mark.asyncio
async def test_set_config_option_rejects_active_turn(tmp_path: Path) -> None:
    session, _cm = _make_session(tmp_path, StreamingLLM())
    factory = _FakeFactory(session)
    agent = AcpAgent(factory=cast(Any, factory))
    new_session = await agent.new_session(cwd=str(tmp_path))

    async def _forever() -> None:
        await asyncio.Event().wait()

    registered = agent._sessions[new_session.session_id]  # noqa: SLF001
    registered.turn_task = asyncio.create_task(_forever())
    try:
        with pytest.raises(RequestError):
            await agent.set_config_option(CONFIG_MODE, new_session.session_id, "plan")
        with pytest.raises(RequestError):
            await agent.set_config_option(CONFIG_MODEL, new_session.session_id, "x/y")
    finally:
        registered.turn_task.cancel()
    assert factory.interaction_mode_calls == []
    assert factory.model_calls == []


@pytest.mark.asyncio
async def test_real_factory_permission_mode_updates_agent_and_record(tmp_path: Path) -> None:
    agent, _cm = build_agent(tmp_path)
    record = SessionRecord.create(Path(tmp_path), "cli", {})
    session = AcpSession(session_id=record.session_id, record=record, agent=agent, manager=_cm)
    factory = AgentFactory()

    await factory.set_permission_mode(session, PermissionMode.AUTO)

    assert session.agent.permission_mode == PermissionMode.AUTO
    assert session.record.permission_mode == "auto"


@pytest.mark.asyncio
async def test_real_factory_builds_both_options(isolated_cli_env: None, tmp_path: Path) -> None:
    agent, cm = build_agent(tmp_path)
    record = SessionRecord.create(Path(tmp_path), "cli", {})
    session = AcpSession(session_id=record.session_id, record=record, agent=agent, manager=cm)
    agent.set_permission_mode(PermissionMode.ASK)
    factory = AgentFactory()
    factory._load_settings()  # noqa: SLF001

    options = factory.config_options_for(session)

    assert [option.id for option in options] == [CONFIG_MODEL, CONFIG_PERMISSION_AUTO, CONFIG_MODE]
    model_option, permission_option, mode_option = options
    assert isinstance(model_option, SessionConfigOptionSelect)
    assert model_option.current_value == "anthropic/claude-haiku-4-5-20251001"
    assert isinstance(permission_option, SessionConfigOptionSelect)
    assert permission_option.current_value == "ask"
    assert [entry.value for entry in permission_option.options] == ["ask", "auto"]
    assert isinstance(mode_option, SessionConfigOptionSelect)
    assert mode_option.current_value == "build"
    assert [entry.value for entry in mode_option.options] == ["build", "plan"]


@pytest.mark.asyncio
async def test_set_config_option_mode_switches_without_touching_task_list(tmp_path: Path) -> None:
    from acp.schema import AgentPlanUpdate, CurrentModeUpdate

    session, _cm = _make_session(tmp_path, StreamingLLM())
    factory = _FakeFactory(session)
    conn = _FakeConn()
    agent = AcpAgent(factory=cast(Any, factory))
    agent.on_connect(cast(Client, conn))
    new_session = await agent.new_session(cwd=str(tmp_path))

    await agent.set_config_option(CONFIG_MODE, new_session.session_id, "plan")
    await agent.set_config_option(CONFIG_MODE, new_session.session_id, "build")

    assert [mode for _, mode in factory.interaction_mode_calls] == ["plan", "build"]
    modes = [u for u in conn.updates if isinstance(u, CurrentModeUpdate)]
    assert [u.current_mode_id for u in modes] == ["plan", "build"]
    # The plan view mirrors the shared task list, which persists across mode
    # switches in the TUI; a mode change must not clear it.
    assert not any(isinstance(u, AgentPlanUpdate) for u in conn.updates)


@pytest.mark.asyncio
async def test_set_config_option_mode_rejects_unknown_mode(tmp_path: Path) -> None:
    session, _cm = _make_session(tmp_path, StreamingLLM())
    agent = AcpAgent(factory=cast(Any, _FakeFactory(session)))
    new_session = await agent.new_session(cwd=str(tmp_path))

    with pytest.raises(RequestError):
        await agent.set_config_option(CONFIG_MODE, new_session.session_id, "vibe")
