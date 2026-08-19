"""Host-registered volatile sections (plan handle, task list) and post-compaction re-injection.

The CLI registers the plan artifact handle and the shared task list as volatile-context
sections on the top-level agent. The tracker contract (dedup, change-driven re-send) is
covered by ``test_volatile_context.py``; this file exercises the ``BaseAgent`` wiring:
sections are injected as their own user turn, re-sent when they change, and re-sent after
compaction (auto and manual) and history clears.
"""

import pytest

from kolega_code.agent.volatile_context import VolatileSection

from .compaction_helpers import FakeLLM, build_agent, long_history

PLAN_TEXT = "artifact: /plans/current-plan.md"


def _plan_provider(text: str = PLAN_TEXT):
    return lambda: VolatileSection("plan", text)


@pytest.mark.asyncio
async def test_registered_section_is_injected_as_its_own_user_turn(tmp_path):
    agent, _cm = build_agent(tmp_path, llm=FakeLLM())
    agent.add_volatile_section(_plan_provider())

    _ = [chunk async for chunk in agent.process_message_stream("do the thing")]

    # history = [session context, user message, injected volatile block, assistant response]
    assert agent.history[0] is agent.session_context_message
    assert [message.role for message in agent.history] == ["user", "user", "user", "assistant"]
    injected = agent.history[2]
    text = injected.get_text_content()
    assert 'source="plan"' in text
    assert PLAN_TEXT in text
    # The date section is injected alongside on the first turn.
    assert 'source="date"' in text


@pytest.mark.asyncio
async def test_empty_section_is_absent(tmp_path):
    agent, _cm = build_agent(tmp_path, llm=FakeLLM())
    agent.add_volatile_section(lambda: VolatileSection("plan", ""))

    _ = [chunk async for chunk in agent.process_message_stream("do the thing")]

    text = agent.history[2].get_text_content()
    assert 'source="plan"' not in text
    assert 'source="date"' in text


@pytest.mark.asyncio
async def test_changed_section_is_reinjected_alone(tmp_path):
    agent, _cm = build_agent(tmp_path, llm=FakeLLM())
    state = {"plan": "plan v1"}
    agent.add_volatile_section(lambda: VolatileSection("plan", state["plan"]))

    _ = [chunk async for chunk in agent.process_message_stream("turn one")]
    assert "plan v1" in agent.history[2].get_text_content()

    state["plan"] = "plan v2"
    _ = [chunk async for chunk in agent.process_message_stream("turn two")]

    # The second turn's injected block carries only the changed section.
    injected = agent.history[-2]
    text = injected.get_text_content()
    assert 'source="plan"' in text
    assert "plan v2" in text
    assert "plan v1" not in text
    assert text.count("<system-reminder") == 1


@pytest.mark.asyncio
async def test_auto_compaction_reinjects_sections(tmp_path):
    """The core ask: after in-turn compaction, the next turn re-sends the sections."""
    fake = FakeLLM(token_script=[900, 300, 300, 100])
    agent, _cm = build_agent(tmp_path, llm=fake, model_context_length=1000)
    agent.history = long_history(6)  # 12 messages: comfortably above the compaction floor
    agent.add_volatile_section(_plan_provider())

    # Turn 1: first count (900) is over the 80% budget, so the turn compacts mid-flight.
    _ = [chunk async for chunk in agent.process_message_stream("continue")]
    assert agent.conversation.summary is not None

    # Turn 2: the volatile sent-state was forgotten by compress_history, so the plan
    # handle (and the date) are injected again at the start of the turn.
    _ = [chunk async for chunk in agent.process_message_stream("continue again")]
    injected = agent.history[-2]
    text = injected.get_text_content()
    assert 'source="plan"' in text
    assert PLAN_TEXT in text


@pytest.mark.asyncio
async def test_manual_compress_reinjects_sections(tmp_path):
    """/compact previously never re-sent context; the forget now lives in compress_history."""
    fake = FakeLLM(token_script=[900, 300, 300])
    agent, _cm = build_agent(tmp_path, llm=fake)
    agent.history = long_history(6)
    agent.add_volatile_section(_plan_provider())

    # First injection records sent-state; nothing pending while unchanged.
    assert agent.pending_volatile_context() is not None
    assert agent.pending_volatile_context() is None

    result = await agent.command_processor._handle_compress()
    assert "Compressed history" in result

    # A successful manual compaction forgets the sent-state: the next turn re-sends.
    pending = agent.pending_volatile_context()
    assert pending is not None
    assert 'source="plan"' in pending.text


@pytest.mark.asyncio
async def test_clear_history_resets_sections(tmp_path):
    agent, _cm = build_agent(tmp_path, llm=FakeLLM())
    agent.add_volatile_section(_plan_provider())

    assert agent.pending_volatile_context() is not None
    assert agent.pending_volatile_context() is None

    agent.clear_history()

    assert agent.pending_volatile_context() is not None


@pytest.mark.asyncio
async def test_provider_error_is_skipped(tmp_path, caplog):
    agent, _cm = build_agent(tmp_path, llm=FakeLLM())

    def broken():
        raise RuntimeError("boom")

    agent.add_volatile_section(broken)
    agent.add_volatile_section(_plan_provider())

    _ = [chunk async for chunk in agent.process_message_stream("do the thing")]

    # The turn still ran and the healthy section was injected.
    assert len(agent.history) == 4
    assert 'source="plan"' in agent.history[2].get_text_content()
    assert "Volatile-section provider failed" in caplog.text


@pytest.mark.asyncio
async def test_reset_volatile_context_public_wrapper(tmp_path):
    agent, _cm = build_agent(tmp_path, llm=FakeLLM())
    agent.add_volatile_section(_plan_provider())

    assert agent.pending_volatile_context() is not None
    assert agent.pending_volatile_context() is None

    agent.reset_volatile_context()

    assert agent.pending_volatile_context() is not None
