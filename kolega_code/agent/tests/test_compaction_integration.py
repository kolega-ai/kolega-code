"""Full agent-loop integration for compaction (hermetic, no network)."""

import pytest

from kolega_code.hooks.events import HookEvent

from .compaction_helpers import FakeLLM, build_agent, context_update_events, long_history


def _hook_spy(agent):
    fired = []
    original = agent.fire_hook

    async def spy(name, payload, **kwargs):
        fired.append(name)
        return await original(name, payload, **kwargs)

    agent.fire_hook = spy
    return fired


@pytest.mark.asyncio
async def test_full_loop_compacts_then_terminates(tmp_path):
    agent, cm = build_agent(tmp_path, llm=FakeLLM(token_script=[900, 300]))
    agent.history = long_history(6)
    fired = _hook_spy(agent)

    async for _chunk in agent.process_message_stream("hello"):
        pass

    assert HookEvent.PRE_COMPACT in fired
    assert HookEvent.POST_COMPACT in fired
    assert fired.index(HookEvent.PRE_COMPACT) < fired.index(HookEvent.POST_COMPACT)
    assert agent.conversation.summary is not None
    assert agent.last_compression_index is not None
    # A context_update carrying the lower post-compaction count was emitted.
    counts = [e.content["input_tokens"] for e in context_update_events(cm)]
    assert 300 in counts


@pytest.mark.asyncio
async def test_full_loop_no_fallback_preserves_summary_when_still_over_budget(tmp_path):
    # Even if still over budget after compaction, there is no destructive fallback:
    # the summary survives and history is never wiped.
    agent, _cm = build_agent(tmp_path, llm=FakeLLM(token_script=[900, 900]))
    agent.history = long_history(6)

    async for _chunk in agent.process_message_stream("hello"):
        pass

    assert agent.conversation.summary is not None
    assert not hasattr(agent, "apply_compression_fallback")


@pytest.mark.asyncio
async def test_full_loop_no_compaction_under_budget(tmp_path):
    agent, _cm = build_agent(tmp_path, llm=FakeLLM(token_script=[100]))
    agent.history = long_history(6)
    fired = _hook_spy(agent)

    async for _chunk in agent.process_message_stream("hello"):
        pass

    assert HookEvent.PRE_COMPACT not in fired
    assert HookEvent.POST_COMPACT not in fired
    assert agent.conversation.summary is None
