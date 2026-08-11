"""Compaction state survives a dump/restore round-trip (save + resume)."""

import pytest

from kolega_code.llm.models import Message, TextBlock

from .compaction_helpers import FakeLLM, build_agent, long_history


@pytest.mark.asyncio
async def test_agent_compaction_survives_dump_restore(tmp_path):
    agent, _cm = build_agent(tmp_path, llm=FakeLLM(summary_text="PERSISTED SUMMARY"))
    agent.history = long_history(6)  # 12 messages
    result = await agent.compress_history()
    assert result.ok

    history_dump = agent.dump_message_history()
    compaction_dump = agent.dump_compaction_state()
    pre_effective = [m.get_text_content() for m in agent.get_effective_history_for_llm()]
    assert compaction_dump["summary"] == "PERSISTED SUMMARY"
    assert compaction_dump["compacted_through"] > 0
    assert compaction_dump["compacted_history_length"] == 12  # captured at compaction time

    # Resume into a fresh agent: restore messages, then the compaction boundary.
    fresh, _cm2 = build_agent(tmp_path, llm=FakeLLM())
    fresh.restore_message_history(history_dump)
    fresh.restore_compaction_state(compaction_dump)

    assert fresh.conversation.summary is not None
    assert len(fresh.history) == 12  # full history intact
    assert fresh.conversation.compacted_history_length == 12  # round-tripped
    # The effective view the model sees is identical to before saving.
    assert [m.get_text_content() for m in fresh.get_effective_history_for_llm()] == pre_effective


@pytest.mark.asyncio
async def test_agent_without_compaction_dumps_empty(tmp_path):
    agent, _cm = build_agent(tmp_path, llm=FakeLLM())
    agent.history = long_history(2)  # 4 messages, never compacted
    data = agent.dump_compaction_state()
    assert data == {"summary": "", "compacted_through": 0, "compacted_history_length": 0}

    fresh, _cm2 = build_agent(tmp_path, llm=FakeLLM())
    fresh.restore_message_history(agent.dump_message_history())
    fresh.restore_compaction_state(data)
    assert fresh.conversation.summary is None
    assert fresh.conversation.compacted_through == 0


@pytest.mark.asyncio
async def test_zero_tail_compaction_survives_dump_restore(tmp_path):
    # A fallback pass (trigger auto_tail_fallback, keep_recent=0) folds the whole
    # conversation; the boundary + summary round-trip like an ordinary compaction.
    summary_msg = Message(role="assistant", content=[TextBlock(text="ZERO-TAIL PERSISTED")], stop_reason="end_turn")
    primary_msg = Message(role="assistant", content=[TextBlock(text="PRIMARY RESPONSE")], stop_reason="end_turn")
    fake = FakeLLM(
        token_script=[900, 1100, 1100, 400],
        message_script=[summary_msg, summary_msg, primary_msg],
    )
    agent, _cm = build_agent(tmp_path, llm=fake)
    agent.history = long_history(6)

    async for _chunk in agent.process_message_stream("hello"):
        pass

    # Zero tail: everything but the post-compaction primary reply is folded.
    assert agent.conversation.compacted_through == len(agent.history) - 1
    history_dump = agent.dump_message_history()
    compaction_dump = agent.dump_compaction_state()
    pre_effective = [m.get_text_content() for m in agent.get_effective_history_for_llm()]
    assert compaction_dump["summary"] == "ZERO-TAIL PERSISTED"
    assert pre_effective == ["ZERO-TAIL PERSISTED", "PRIMARY RESPONSE"]

    # Resume into a fresh agent: restore messages, then the compaction boundary.
    fresh, _cm2 = build_agent(tmp_path, llm=FakeLLM())
    fresh.restore_message_history(history_dump)
    fresh.restore_compaction_state(compaction_dump)

    assert len(fresh.history) == len(agent.history)  # full history intact after resume
    assert fresh.conversation.summary is not None
    assert fresh.conversation.compacted_through == agent.conversation.compacted_through
    # The effective view the model sees is identical to before saving.
    assert [m.get_text_content() for m in fresh.get_effective_history_for_llm()] == pre_effective
