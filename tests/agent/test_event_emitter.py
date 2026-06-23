"""AgentEventEmitter tags context/status events with sub_agent_info.

Regression coverage for the bug where a sub-agent's context-window updates stomped the
main agent's context indicator in the TUI. context_update() and llm_status() must carry
sub_agent_info (like chat() already does) so hosts can tell main- and sub-agent updates
apart. The main agent's provider returns None; a dispatched sub-agent returns a dict.
"""

from unittest.mock import AsyncMock

import pytest

from kolega_code.events import AgentConnectionManager, AgentEventEmitter

SUB_AGENT_INFO = {"agent_id": "agent-1", "agent_name": "general-agent", "depth": 1}


def _emitter(provider):
    cm = AsyncMock(spec=AgentConnectionManager)
    emitter = AgentEventEmitter(
        connection_manager=cm,
        workspace_id="ws",
        thread_id="th",
        sender="general-agent",
        sub_agent_info_provider=provider,
    )
    return emitter, cm


def _emitted(cm):
    return cm.broadcast_event.await_args.args[0]


@pytest.mark.asyncio
async def test_context_update_carries_sub_agent_info_when_provided():
    emitter, cm = _emitter(lambda: SUB_AGENT_INFO)

    await emitter.context_update(
        input_tokens=6000,
        model_context_length=200000,
        compression_threshold=0.8,
        alert_level="normal",
        message=None,
    )

    event = _emitted(cm)
    assert event.event_type == "llm_context_update"
    assert event.sub_agent_info == SUB_AGENT_INFO


@pytest.mark.asyncio
async def test_context_update_has_no_sub_agent_info_for_main_agent():
    # The main agent's provider returns None (BaseAgent._sub_agent_info).
    emitter, cm = _emitter(lambda: None)

    await emitter.context_update(
        input_tokens=123456,
        model_context_length=200000,
        compression_threshold=0.8,
        alert_level="normal",
        message=None,
    )

    assert _emitted(cm).sub_agent_info is None


@pytest.mark.asyncio
async def test_llm_status_carries_sub_agent_info_when_provided():
    emitter, cm = _emitter(lambda: SUB_AGENT_INFO)

    await emitter.llm_status("overloaded", "Provider overloaded, retrying")

    event = _emitted(cm)
    assert event.event_type == "llm_status_update"
    assert event.sub_agent_info == SUB_AGENT_INFO


@pytest.mark.asyncio
async def test_llm_status_has_no_sub_agent_info_for_main_agent():
    emitter, cm = _emitter(None)

    await emitter.llm_status("overloaded", "Provider overloaded, retrying")

    assert _emitted(cm).sub_agent_info is None
