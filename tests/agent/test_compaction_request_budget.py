"""Regression: the compaction request itself must fit the run's input budget.

The summarization request carries the aged-out span plus the previous summary
and instructions, so at the compaction trigger it is roughly the size of the
parent context. If it exceeds max input, the budget check rejects it before
dispatch, the pass reads as ``llm_error`` (which the agent loop treats as
final), and compaction stays broken while the conversation grows until the
run aborts.

One proxy counter (wire-format chars // 4) plays the provider renderer on both
paths — the context gauge and the stream-side enforcement — as production
routes both through ``provider.count_tokens``.
"""

import json
from unittest.mock import AsyncMock

import pytest

from kolega_code.agent.compression import HistoryCompressor
from kolega_code.agent.conversation import Conversation
from kolega_code.agent.prompts import (
    COMPRESSION_SUMMARY_SYSTEM_PROMPT,
    build_compression_summary_user_prompt,
)
from kolega_code.hooks.events import HookEvent
from kolega_code.llm.exceptions import LLMContextWindowExceededError
from kolega_code.llm.models import Message, MessageHistory, TextBlock, ToolCall, ToolResult
from kolega_code.llm.providers.models import TokenCount

from .compaction_helpers import FakeLLM, build_agent, text_msg

CONTEXT_WINDOW_TOKENS = 4000
MAX_OUTPUT_TOKENS = 500
MAX_INPUT_TOKENS = CONTEXT_WINDOW_TOKENS - MAX_OUTPUT_TOKENS  # 3500


def _wire_tokens(messages, system=None) -> int:
    """Proxy for the provider renderer's count: wire-format chars // 4."""
    chars = len(system.get_text_content()) if system is not None else 0
    for message in messages:
        if isinstance(message.content, str):
            chars += len(message.content)
            continue
        for block in message.content:
            if isinstance(block, ToolCall):
                chars += len(json.dumps(block.input))
            elif isinstance(block, ToolResult):
                chars += len(block.content) if isinstance(block.content, str) else 0
            else:
                chars += len(getattr(block, "text", "") or "")
    return chars // 4


class BudgetEnforcingFakeLLM(FakeLLM):
    """FakeLLM whose ``stream`` enforces the per-run input cap the way
    ``LLMClient._enforce_run_context_budget`` does."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.stream_input_counts: list[int] = []
        self.count_tokens = AsyncMock(side_effect=self._count_tokens_wire)
        self.stream = AsyncMock(side_effect=self._budgeted_stream)

    async def _count_tokens_wire(self, *args, **kwargs):
        messages = kwargs.get("messages") or (args[0] if args else [])
        return TokenCount(input_tokens=_wire_tokens(messages, kwargs.get("system")))

    async def _budgeted_stream(self, *args, **kwargs):
        input_tokens = _wire_tokens(kwargs.get("messages") or [], kwargs.get("system"))
        requested = kwargs.get("max_completion_tokens")
        effective_output = MAX_OUTPUT_TOKENS if requested is None else min(requested, MAX_OUTPUT_TOKENS)
        max_input = CONTEXT_WINDOW_TOKENS - effective_output
        self.stream_input_counts.append(input_tokens)
        if input_tokens > max_input:
            raise LLMContextWindowExceededError(
                f"Request needs {input_tokens} input tokens, exceeding this run's maximum input of "
                f"{max_input} (context window {CONTEXT_WINDOW_TOKENS} tokens minus "
                f"{effective_output} reserved output tokens).",
                provider="fake",
            )
        return await self._stream(*args, **kwargs)


def _tool_pair(i: int) -> list[Message]:
    call_id = f"call_{i:04d}"
    args = {"path": f"src/file_{i:03d}.py", "note": f"step {i} " + "x" * 110}
    output = f"contents of file_{i:03d} " + "y" * 110
    return [
        Message(role="assistant", content=[ToolCall(id=call_id, name="read_file", input=args)]),
        Message(
            role="user", content=[ToolResult(tool_use_id=call_id, name="read_file", content=output, is_error=False)]
        ),
    ]


def _hook_payload_spy(agent):
    fired = []
    original = agent.fire_hook

    async def spy(name, payload, **kwargs):
        fired.append((name, payload))
        return await original(name, payload, **kwargs)

    agent.fire_hook = spy
    return fired


@pytest.mark.asyncio
async def test_compaction_request_is_bounded_to_run_input_budget(tmp_path):
    # A session that already compacted once and regrew: prior summary stands in
    # for the first 8 messages, then 40 tool exchanges of new work.
    prior_summary = "PRIOR SUMMARY " + "z" * 86
    old_messages = [m for i in range(4) for m in _tool_pair(i)]
    regrown = [m for i in range(4, 44) for m in _tool_pair(i)]

    fake = BudgetEnforcingFakeLLM()
    agent, _cm = build_agent(tmp_path, llm=fake, strict_budget=(CONTEXT_WINDOW_TOKENS, MAX_OUTPUT_TOKENS))
    assert agent.model_max_input_tokens == MAX_INPUT_TOKENS
    agent.history = MessageHistory(old_messages + regrown)
    agent.conversation.apply_compaction(prior_summary, len(old_messages))

    # Preconditions, computed on the exact state the compaction pass will see:
    probe = Conversation()
    probe.history = MessageHistory(list(agent.history) + [text_msg("user", "continue")])
    probe.apply_compaction(prior_summary, len(old_messages))
    parent_tokens = _wire_tokens(list(probe.effective_history()), agent.system_prompt)
    # the parent context is over the trigger but under the cap,
    trigger = MAX_INPUT_TOKENS * agent.history_compression_threshold
    assert trigger < parent_tokens <= MAX_INPUT_TOKENS
    # while the whole span rendered into one summarization request exceeds it.
    split = probe.compaction_split_point(
        keep_recent=HistoryCompressor.KEEP_RECENT_MESSAGES,
        min_prefix=HistoryCompressor.MIN_PREFIX_TO_SUMMARIZE,
    )
    prefix_markdown = MessageHistory(list(probe.history[len(old_messages) : split])).get_markdown_conversation()
    unbounded_request = Message(
        role="user",
        content=[
            TextBlock(text=build_compression_summary_user_prompt(prefix_markdown, previous_summary=prior_summary))
        ],
    )
    comp_system = Message(role="system", content=[TextBlock(text=COMPRESSION_SUMMARY_SYSTEM_PROMPT)])
    assert _wire_tokens([unbounded_request], comp_system) > MAX_INPUT_TOKENS

    fired = _hook_payload_spy(agent)
    async for _chunk in agent.process_message_stream("continue"):
        pass

    # Compaction must bound its own request and succeed, not fail llm_error.
    post = [payload for name, payload in fired if name == HookEvent.POST_COMPACT]
    assert post and post[0]["trigger"] == "auto"
    assert post[0]["ok"] is True, f"compaction failed: {post[0]['reason']}"
    assert agent.conversation.compacted_through > len(old_messages)
    assert agent.conversation.summary is not None
    assert agent.conversation.summary.get_text_content() == "SUMMARY: condensed older turns."
    # Every request this turn dispatched — the summarization included — fit the cap.
    assert fake.stream_input_counts and all(count <= MAX_INPUT_TOKENS for count in fake.stream_input_counts), (
        f"stream inputs {fake.stream_input_counts} exceed max input {MAX_INPUT_TOKENS}"
    )


@pytest.mark.asyncio
async def test_oversized_span_folds_with_middle_omission(tmp_path):
    # A span that dwarfs the budget folds in one pass: whole messages are
    # omitted from the middle, with a notice left in the gap.
    fake = BudgetEnforcingFakeLLM()
    agent, _cm = build_agent(tmp_path, llm=fake, strict_budget=(CONTEXT_WINDOW_TOKENS, MAX_OUTPUT_TOKENS))
    agent.history = MessageHistory([m for i in range(120) for m in _tool_pair(i)])

    result = await agent.compress_history(trigger="manual")

    assert result.ok, result.message
    assert fake.stream_input_counts and all(count <= MAX_INPUT_TOKENS for count in fake.stream_input_counts), (
        f"stream inputs {fake.stream_input_counts} exceed max input {MAX_INPUT_TOKENS}"
    )
    assert fake.stream.await_args is not None
    sent_prompt = fake.stream.await_args.kwargs["messages"][0].get_text_content()
    assert "omitted here so the compaction request fits" in sent_prompt
    assert "file_000" in sent_prompt  # first exchange of the span survives
    assert "file_116" in sent_prompt  # last exchange of the span survives
    assert agent.conversation.compacted_through == len(agent.history) - HistoryCompressor.KEEP_RECENT_MESSAGES


class UndercountingFakeLLM(BudgetEnforcingFakeLLM):
    """Counter reads 25% low, the way an estimating tokenizer can."""

    async def _count_tokens_wire(self, *args, **kwargs):
        messages = kwargs.get("messages") or (args[0] if args else [])
        return TokenCount(input_tokens=_wire_tokens(messages, kwargs.get("system")) * 3 // 4)


@pytest.mark.asyncio
async def test_rejected_dispatch_steps_down_budget_ladder(tmp_path):
    # The enforcing side runs hotter than the counter: rejected dispatches must
    # step down the budget ladder and re-widen the gap until one is accepted.
    fake = UndercountingFakeLLM()
    agent, _cm = build_agent(tmp_path, llm=fake, strict_budget=(CONTEXT_WINDOW_TOKENS, MAX_OUTPUT_TOKENS))
    agent.history = MessageHistory([m for i in range(120) for m in _tool_pair(i)])

    result = await agent.compress_history(trigger="manual")

    assert result.ok, result.message
    assert any(count > MAX_INPUT_TOKENS for count in fake.stream_input_counts)  # at least one rejection
    assert fake.stream_input_counts[-1] <= MAX_INPUT_TOKENS  # the accepted request fit
    assert agent.conversation.summary is not None
