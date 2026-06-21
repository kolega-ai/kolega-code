"""Tests for the /compress command: real outcome reporting + context update."""

import pytest

from .compaction_helpers import FakeLLM, build_agent, context_update_events, long_history


@pytest.mark.asyncio
async def test_compress_command_reports_real_outcome(tmp_path):
    agent, _cm = build_agent(tmp_path, llm=FakeLLM(token_script=[900, 300]))
    agent.history = long_history(6)  # 12 messages, >= MIN_MESSAGES_TO_COMPRESS

    result = await agent.command_processor._handle_compress()

    assert "Compressed history" in result
    assert "→" in result and "%" in result  # before -> after percentages
    assert agent.conversation.summary is not None


@pytest.mark.asyncio
async def test_compress_command_nothing_to_compress_under_five(tmp_path):
    agent, _cm = build_agent(tmp_path, llm=FakeLLM(token_script=[100]))
    agent.history = long_history(2)  # 4 messages < MIN_MESSAGES_TO_COMPRESS

    result = await agent.command_processor._handle_compress()

    assert "Nothing to compress" in result
    assert agent.conversation.summary is None
    agent.llm.stream.assert_not_called()


@pytest.mark.asyncio
async def test_compress_command_emits_context_update(tmp_path):
    agent, cm = build_agent(tmp_path, llm=FakeLLM(token_script=[900, 300]))
    agent.history = long_history(6)

    await agent.command_processor._handle_compress()

    counts = [e.content["input_tokens"] for e in context_update_events(cm)]
    assert counts  # the gauge was refreshed
    assert 300 in counts  # reflects the post-compaction (lower) count
