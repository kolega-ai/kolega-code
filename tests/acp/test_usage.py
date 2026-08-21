"""Usage meter tests: ACP usage_update from llm_context_update events."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
from acp.interfaces import Client
from acp.schema import UsageUpdate

from kolega_code.acp.bridge import AcpBridge
from kolega_code.acp.usage import build_usage_update, context_window_for
from kolega_code.events import AgentEvent

from tests.agent.compaction_helpers import build_agent
from tests.acp.test_server import StreamingLLM, _FakeConn, _make_agent, _make_session


def _context_event(input_tokens: int) -> AgentEvent:
    return AgentEvent(
        sender="agent",
        event_type="llm_context_update",
        content={"input_tokens": input_tokens, "model_context_length": 1000},
        is_streaming=False,
    )


@pytest.mark.asyncio
async def test_build_usage_update_uses_agent_window(tmp_path: Path) -> None:
    agent, _cm = build_agent(tmp_path)
    assert context_window_for(agent) == 1000
    update = build_usage_update(agent, 321)
    assert update is not None
    assert update.used == 321
    assert update.size == 1000


@pytest.mark.asyncio
async def test_usage_clamps_to_window(tmp_path: Path) -> None:
    agent, _cm = build_agent(tmp_path)
    update = build_usage_update(agent, 5000)
    assert update is not None
    assert update.used == 1000


@pytest.mark.asyncio
async def test_build_usage_update_skips_without_window(tmp_path: Path) -> None:
    agent, _cm = build_agent(tmp_path)
    setattr(agent, "model_max_input_tokens", None)
    setattr(agent, "model_context_length", None)
    setattr(agent, "primary_model_config", SimpleNamespace(context_window_tokens=None))
    assert context_window_for(agent) is None
    assert build_usage_update(agent, 10) is None


@pytest.mark.asyncio
async def test_bridge_tracks_context_events_and_emits_usage(tmp_path: Path) -> None:
    agent, _cm = build_agent(tmp_path)
    conn = _FakeConn()
    bridge = AcpBridge(cast(Client, conn), agent=agent)

    await bridge.handle_event("s1", _context_event(42))
    await bridge.emit_usage("s1")
    await bridge.handle_event("s1", _context_event(77))
    await bridge.emit_usage("s1")

    usage_updates = [update for update in conn.updates if isinstance(update, UsageUpdate)]
    assert [(update.used, update.size) for update in usage_updates] == [(42, 1000), (77, 1000)]


@pytest.mark.asyncio
async def test_emit_usage_skipped_without_window(tmp_path: Path) -> None:
    agent, _cm = build_agent(tmp_path)
    setattr(agent, "model_max_input_tokens", None)
    setattr(agent, "model_context_length", None)
    setattr(agent, "primary_model_config", SimpleNamespace(context_window_tokens=None))
    conn = _FakeConn()
    bridge = AcpBridge(cast(Client, conn), agent=agent)

    await bridge.handle_event("s1", _context_event(42))
    await bridge.emit_usage("s1")

    assert conn.updates == []


@pytest.mark.asyncio
async def test_new_session_emits_initial_usage(tmp_path: Path) -> None:
    session, _cm = _make_session(tmp_path, StreamingLLM())
    agent, conn = _make_agent(session)

    await agent.new_session(cwd=str(tmp_path))

    usage_updates = [update for update in conn.updates if isinstance(update, UsageUpdate)]
    assert len(usage_updates) == 1
    assert usage_updates[0].used == 0
    assert usage_updates[0].size == 1000


@pytest.mark.asyncio
async def test_prompt_emits_usage_after_turn(tmp_path: Path) -> None:
    session, _cm = _make_session(tmp_path, StreamingLLM())
    agent, conn = _make_agent(session)
    new_session = await agent.new_session(cwd=str(tmp_path))

    await agent.prompt(session_id=new_session.session_id, prompt=[{"type": "text", "text": "hi"}])  # pyright: ignore[reportArgumentType]

    usage_updates = [update for update in conn.updates if isinstance(update, UsageUpdate)]
    assert usage_updates
    assert usage_updates[-1].size == 1000
