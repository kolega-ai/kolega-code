"""Full agent-loop integration for compaction (hermetic, no network).

FakeLLM token-script accounting: the loop counts once per iteration, and each
compaction pass counts three times (the summarization request's own budget
check, then ``compress_history``'s finally recount for the gauge, then the
loop's own post-pass recount on the rebuilt history). A script like
``[900, 800, 1100, 1100, 800, 400]`` therefore reads as: 900 triggers the
ordinary pass, 800 is its summarization request (fits), 1100 is its post-pass
size (twice), 800 the fallback's summarization request, 400 the post-fallback
size. The final scripted value repeats, so trailing reads stay deterministic.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from kolega_code.hooks.events import HookEvent
from kolega_code.llm.exceptions import LLMContextWindowExceededError

from .compaction_helpers import FakeLLM, FakeStream, build_agent, context_update_events, long_history


def _hook_spy(agent):
    fired = []
    original = agent.fire_hook

    async def spy(name, payload, **kwargs):
        fired.append(name)
        return await original(name, payload, **kwargs)

    agent.fire_hook = spy
    return fired


def _hook_payload_spy(agent):
    fired = []
    original = agent.fire_hook

    async def spy(name, payload, **kwargs):
        fired.append((name, payload))
        return await original(name, payload, **kwargs)

    agent.fire_hook = spy
    return fired


def _compaction_triggers(fired):
    return [payload["trigger"] for name, payload in fired if name == HookEvent.PRE_COMPACT]


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
    fake = FakeLLM(token_script=[100])
    agent, _cm = build_agent(tmp_path, llm=fake)
    agent.history = long_history(6)
    fired = _hook_spy(agent)

    async for _chunk in agent.process_message_stream("hello"):
        pass

    assert HookEvent.PRE_COMPACT not in fired
    assert HookEvent.POST_COMPACT not in fired
    assert agent.conversation.summary is None
    fake.stream.assert_awaited_once()  # the primary generation only


@pytest.mark.asyncio
async def test_full_loop_single_pass_when_normal_compaction_fits(tmp_path):
    # Post-compaction size is back under the resolved maximum: exactly one
    # ordinary pass, no zero-tail fallback.
    fake = FakeLLM(token_script=[900, 300])
    agent, _cm = build_agent(tmp_path, llm=fake)
    agent.history = long_history(6)
    fired = _hook_payload_spy(agent)

    async for _chunk in agent.process_message_stream("hello"):
        pass

    assert _compaction_triggers(fired) == ["auto"]
    assert fake.stream.await_count == 2  # one summarization + one primary generation
    assert agent.conversation.summary is not None


@pytest.mark.asyncio
async def test_full_loop_zero_tail_fallback_when_retained_tail_too_large(tmp_path):
    # The six-message retained tail alone still exceeds the resolved maximum
    # (1100 > 1000): one aggressive keep_recent=0 pass folds it and the request
    # proceeds (400 <= 1000).
    fake = FakeLLM(token_script=[900, 800, 1100, 1100, 800, 400])
    agent, _cm = build_agent(tmp_path, llm=fake)
    agent.session_recorder = MagicMock()
    agent.history = long_history(6)
    original = list(agent.history)
    fired = _hook_payload_spy(agent)

    async for _chunk in agent.process_message_stream("hello"):
        pass

    # Two passes with distinct provenance, then one primary generation.
    assert _compaction_triggers(fired) == ["auto", "auto_tail_fallback"]
    # Each successful pass is journaled through the existing compaction record
    # with its own trigger.
    recorded_triggers = [
        call.kwargs["info"]["trigger"] for call in agent.session_recorder.record_compaction.call_args_list
    ]
    assert recorded_triggers == ["auto", "auto_tail_fallback"]
    assert fake.stream.await_count == 3
    # The primary request carried the fully folded view: the summary alone.
    primary_messages = fake.stream.await_args_list[-1].kwargs["messages"]
    assert [m.get_text_content() for m in primary_messages] == ["SUMMARY: condensed older turns."]
    # The raw conversation is intact — only the effective view and the
    # compaction boundary changed. Everything present before the turn is still
    # there, in order, and the boundary covers everything but the reply.
    assert list(agent.history)[: len(original)] == original
    assert agent.conversation.summary is not None
    assert agent.conversation.compacted_through == len(agent.history) - 1


@pytest.mark.asyncio
async def test_full_loop_strict_cap_raises_when_fallback_insufficient(tmp_path):
    # Paired cap: window 1000 - output 200 → resolved max input 800
    # (window_minus_output). The zero-tail pass still leaves 900: fail closed
    # before the primary model is called.
    fake = FakeLLM(token_script=[700, 900, 900, 900])
    agent, _cm = build_agent(tmp_path, llm=fake, strict_budget=(1000, 200))
    assert agent.strict_context_budget is True
    assert agent.model_max_input_tokens == 800
    agent.history = long_history(6)
    fired = _hook_payload_spy(agent)

    with pytest.raises(LLMContextWindowExceededError) as excinfo:
        async for _chunk in agent.process_message_stream("hello"):
            pass

    message = str(excinfo.value)
    assert "900" in message  # measured input tokens
    assert "800" in message  # resolved maximum input
    assert _compaction_triggers(fired) == ["auto", "auto_tail_fallback"]
    assert fake.stream.await_count == 2  # both summarizations; the primary model was never called


@pytest.mark.asyncio
async def test_full_loop_strict_cap_allows_equality_with_max_input(tmp_path):
    # Equality with the resolved maximum input is allowed.
    fake = FakeLLM(token_script=[700, 750, 900, 900, 750, 800])
    agent, _cm = build_agent(tmp_path, llm=fake, strict_budget=(1000, 200))
    assert agent.model_max_input_tokens == 800
    agent.history = long_history(6)

    async for _chunk in agent.process_message_stream("hello"):
        pass

    assert fake.stream.await_count == 3  # two summarizations + the primary generation


@pytest.mark.asyncio
async def test_full_loop_no_aggressive_retry_after_llm_error(tmp_path):
    # The ordinary pass fails at the model: the provider client already applied
    # its retry policy, so no second summarization request is issued — the final
    # size check falls through to ordinary provider dispatch.
    fake = FakeLLM(token_script=[900, 1100, 1100])
    fake.stream = AsyncMock(side_effect=[RuntimeError("boom"), FakeStream()])
    agent, _cm = build_agent(tmp_path, llm=fake)
    agent.history = long_history(6)
    fired = _hook_payload_spy(agent)

    async for _chunk in agent.process_message_stream("hello"):
        pass

    assert _compaction_triggers(fired) == ["auto"]  # no auto_tail_fallback pass
    assert fake.stream.await_count == 2  # the failed summarization + the primary generation
    assert agent.conversation.summary is None  # nothing was folded


@pytest.mark.asyncio
async def test_full_loop_strict_cap_raises_after_llm_error_without_retry(tmp_path):
    # Same llm_error path under the paired cap: still no aggressive retry, and
    # the size check rejects the request before the primary model is called.
    fake = FakeLLM(token_script=[700, 900, 900])
    fake.stream = AsyncMock(side_effect=[RuntimeError("boom")])
    agent, _cm = build_agent(tmp_path, llm=fake, strict_budget=(1000, 200))
    agent.history = long_history(6)

    with pytest.raises(LLMContextWindowExceededError):
        async for _chunk in agent.process_message_stream("hello"):
            pass

    assert fake.stream.await_count == 1  # the failed summarization; no retry, no primary


@pytest.mark.asyncio
async def test_full_loop_no_aggressive_retry_after_too_few(tmp_path):
    # Minimum-history guard: the zero-tail pass must not weaken it. Two stored
    # messages plus the new user turn stay below MIN_MESSAGES_TO_COMPRESS.
    fake = FakeLLM(token_script=[900, 1100, 1100])
    agent, _cm = build_agent(tmp_path, llm=fake)
    agent.history = long_history(1)  # 2 messages
    fired = _hook_payload_spy(agent)

    async for _chunk in agent.process_message_stream("hello"):
        pass

    assert _compaction_triggers(fired) == ["auto"]
    post = [payload for name, payload in fired if name == HookEvent.POST_COMPACT]
    assert post[0]["reason"] == "too_few"
    assert fake.stream.await_count == 1  # the primary generation only — no summarization ran
    assert agent.conversation.summary is None


@pytest.mark.asyncio
async def test_full_loop_strict_cap_raises_after_too_few(tmp_path):
    # too_few under the paired cap: the guard stays intact and the size check
    # rejects the request; no LLM call is made at all.
    fake = FakeLLM(token_script=[700, 900, 900])
    agent, _cm = build_agent(tmp_path, llm=fake, strict_budget=(1000, 200))
    agent.history = long_history(1)  # 2 messages

    with pytest.raises(LLMContextWindowExceededError):
        async for _chunk in agent.process_message_stream("hello"):
            pass

    fake.stream.assert_not_called()


@pytest.mark.asyncio
async def test_full_loop_ordinary_run_dispatches_when_fallback_insufficient(tmp_path):
    # Without the paired cap, the legacy provider-dispatch behavior is preserved
    # even when the zero-tail pass still leaves the request too large.
    fake = FakeLLM(token_script=[900, 1100, 1100, 1200])
    agent, _cm = build_agent(tmp_path, llm=fake)
    agent.history = long_history(6)
    original = list(agent.history)

    async for _chunk in agent.process_message_stream("hello"):
        pass

    assert fake.stream.await_count == 3  # both summarizations + the oversized primary request
    assert agent.conversation.summary is not None
    assert list(agent.history)[: len(original)] == original  # raw history intact
