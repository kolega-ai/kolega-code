"""Unit tests for HistoryCompressor (recency-aware, structured-outcome compaction)."""

from unittest.mock import AsyncMock

import pytest

from kolega_code.agent.compression import COMPRESSION_SUMMARY_SYSTEM_PROMPT, CompactionResult, HistoryCompressor
from kolega_code.agent.conversation import Conversation

from .compaction_helpers import FakeLLM, build_agent, long_history, tool_pair, text_msg
from kolega_code.llm.models import MessageHistory


SUMMARIZE_KW = dict(model="claude-haiku-4-5-20251001", temperature=1.0, thinking=None)


def proxy_tokens(messages) -> int:
    return sum(len(m.get_text_content()) for m in messages) // 4


@pytest.mark.parametrize(
    "input_tokens,expected",
    [(799, False), (800, False), (801, True), (0, False)],  # strict ">" at 80% of 1000
)
def test_over_budget_boundary(input_tokens, expected):
    compressor = HistoryCompressor(threshold=0.8)
    assert compressor.over_budget(input_tokens, 1000) is expected


@pytest.mark.asyncio
async def test_summarize_too_few_records_nothing():
    conv = Conversation(list(long_history(2)))  # 4 messages < MIN_MESSAGES_TO_COMPRESS
    llm = FakeLLM()
    result = await HistoryCompressor().summarize(conv, llm=llm, **SUMMARIZE_KW)

    assert isinstance(result, CompactionResult)
    assert result.ok is False
    assert result.reason == "too_few"
    assert conv.summary is None
    assert len(conv.history) == 4
    llm.stream.assert_not_called()


@pytest.mark.asyncio
async def test_summarize_happy_path_records_summary():
    conv = Conversation(list(long_history(6)))  # 12 messages
    llm = FakeLLM(summary_text="SUMMARY: earlier turns")
    result = await HistoryCompressor().summarize(conv, llm=llm, **SUMMARIZE_KW)

    assert result.ok is True
    assert result.reason == "ok"
    assert result.summarized_messages > 0
    assert conv.summary is not None
    assert conv.summary.get_text_content() == "SUMMARY: earlier turns"
    assert conv.last_compression_index is not None
    llm.stream.assert_awaited_once()
    # the summary was streamed with the model/params we passed (capped to a small budget).
    kwargs = llm.stream.await_args.kwargs
    assert kwargs["model"] == SUMMARIZE_KW["model"]
    assert kwargs["max_completion_tokens"] <= HistoryCompressor.SUMMARY_MAX_TOKENS


@pytest.mark.asyncio
async def test_summarize_uses_override_system_prompt_without_changing_user_prompt():
    conv = Conversation(list(long_history(6)))
    llm = FakeLLM(summary_text="SUMMARY: earlier turns")

    result = await HistoryCompressor().summarize(
        conv,
        llm=llm,
        system_prompt_text="Custom compaction system prompt",
        **SUMMARIZE_KW,
    )

    assert result.ok is True
    kwargs = llm.stream.await_args.kwargs
    assert kwargs["system"].get_text_content() == "Custom compaction system prompt"
    assert "user turn 0" in kwargs["messages"].get_markdown_conversation()


@pytest.mark.asyncio
async def test_agent_compaction_override_renders_jinja_context(tmp_path):
    prompt_dir = tmp_path / ".kolega" / "prompts"
    prompt_dir.mkdir(parents=True)
    (prompt_dir / "COMPACTION.md").write_text("Compact project {{ context.project_path }}", encoding="utf-8")
    fake = FakeLLM(summary_text="SUMMARY: earlier turns")
    agent, _ = build_agent(tmp_path, llm=fake)
    agent.history = MessageHistory(list(long_history(6)))

    result = await agent.compress_history()

    assert result.ok is True
    kwargs = fake.stream.await_args.kwargs
    assert kwargs["system"].get_text_content() == f"Compact project {tmp_path}"
    assert "user turn 0" in kwargs["messages"].get_markdown_conversation()


@pytest.mark.asyncio
async def test_agent_compaction_malformed_override_falls_back_to_default(tmp_path, caplog):
    prompt_dir = tmp_path / ".kolega" / "prompts"
    prompt_dir.mkdir(parents=True)
    (prompt_dir / "COMPACTION.md").write_text("{{ missing_variable }}", encoding="utf-8")
    fake = FakeLLM(summary_text="SUMMARY: earlier turns")
    agent, _ = build_agent(tmp_path, llm=fake)
    agent.history = MessageHistory(list(long_history(6)))

    result = await agent.compress_history()

    assert result.ok is True
    kwargs = fake.stream.await_args.kwargs
    assert kwargs["system"].get_text_content() == COMPRESSION_SUMMARY_SYSTEM_PROMPT
    assert "missing_variable" in caplog.text


@pytest.mark.asyncio
async def test_summarize_keeps_recent_messages_verbatim():
    history = long_history(6)  # 12 messages
    conv = Conversation(list(history))
    await HistoryCompressor().summarize(conv, llm=FakeLLM(), **SUMMARIZE_KW)

    effective = list(conv.effective_history())
    # The most recent KEEP_RECENT_MESSAGES survive verbatim after the summary.
    assert list(history)[-1] in effective
    assert list(history)[-HistoryCompressor.KEEP_RECENT_MESSAGES] in effective


@pytest.mark.asyncio
async def test_summarize_reduces_effective_tokens():
    conv = Conversation(list(long_history(6)))
    pre = proxy_tokens(conv.effective_history())
    result = await HistoryCompressor().summarize(conv, llm=FakeLLM(), **SUMMARIZE_KW)
    post = proxy_tokens(conv.effective_history())

    assert result.ok is True
    assert post < pre  # compaction must reduce what the LLM sees


@pytest.mark.asyncio
async def test_summarize_llm_error_is_nondestructive():
    history = long_history(6)
    conv = Conversation(list(history))
    llm = FakeLLM()
    llm.stream = AsyncMock(side_effect=RuntimeError("boom"))
    on_error = AsyncMock()

    result = await HistoryCompressor().summarize(conv, llm=llm, on_error=on_error, **SUMMARIZE_KW)

    assert result.ok is False
    assert result.reason == "llm_error"
    assert "boom" in result.message
    assert conv.summary is None  # state untouched
    assert list(conv.history) == list(history)
    on_error.assert_awaited()


@pytest.mark.asyncio
async def test_summarize_empty_output_is_error():
    conv = Conversation(list(long_history(6)))
    result = await HistoryCompressor().summarize(conv, llm=FakeLLM(summary_text="   "), **SUMMARIZE_KW)
    assert result.ok is False
    assert result.reason == "llm_error"
    assert conv.summary is None


@pytest.mark.asyncio
async def test_summarize_nothing_new_after_compaction():
    conv = Conversation(list(long_history(6)))
    first = await HistoryCompressor().summarize(conv, llm=FakeLLM(), **SUMMARIZE_KW)
    assert first.ok is True
    # No new messages added → nothing further to summarize.
    second = await HistoryCompressor().summarize(conv, llm=FakeLLM(), **SUMMARIZE_KW)
    assert second.ok is False
    assert second.reason == "nothing_to_summarize"


@pytest.mark.asyncio
async def test_summarize_snaps_split_out_of_tool_pair():
    a_tc, u_tr = tool_pair("tool_z")
    history = MessageHistory(
        [
            text_msg("user", "u0"),
            text_msg("assistant", "a0"),
            text_msg("user", "u1"),
            text_msg("assistant", "a1"),
            text_msg("user", "u2"),
            text_msg("assistant", "a2"),
            a_tc,
            u_tr,
        ]  # tool pair at the boundary for KEEP_RECENT default math (8 -> split ~ 0..)
    )
    conv = Conversation(list(history))
    result = await HistoryCompressor(threshold=0.8).summarize(conv, llm=FakeLLM(), **SUMMARIZE_KW)
    # Whether or not it summarized, the effective view is always valid.
    assert conv.is_valid_for_anthropic(list(conv.effective_history())) is True
    assert isinstance(result, CompactionResult)
