"""Incremental token counting must equal a from-scratch recount.

count_tokens memoizes per-message token counts (keyed by object identity, guarded by a
cheap length fingerprint) so that, per agent-loop iteration, only new or changed messages
are re-encoded instead of the whole history. These tests pin the invariant that the
memoized total always equals a fresh provider's full recount, including across in-place
mutation (oversized tool-result truncation), object replacement (provider adaptation /
compaction), and system + tool definitions. No API key required: count_tokens uses no
client state.
"""

import asyncio
import threading

from kolega_code.llm.models import (
    Message,
    MessageHistory,
    TextBlock,
    ThinkingBlock,
    ToolCall,
    ToolDefinition,
    ToolParameter,
    ToolResult,
)
from kolega_code.llm.providers.openai import OpenAIProvider


def _provider() -> OpenAIProvider:
    # __new__ skips __init__ (no API key / client needed); count_tokens lazily creates memos.
    return OpenAIProvider.__new__(OpenAIProvider)


def _count(provider, messages, *, system=None, tools=None) -> int:
    return asyncio.run(
        provider.count_tokens(messages=messages, system=system, model="gpt-4", tools=tools or [])
    ).input_tokens


def _fresh_count(messages, *, system=None, tools=None) -> int:
    return _count(_provider(), messages, system=system, tools=tools)


def _text_message(role: str, text: str) -> Message:
    return Message(role=role, content=[TextBlock(text=text)])


def _history(n: int) -> MessageHistory:
    return MessageHistory(
        [_text_message("user" if i % 2 == 0 else "assistant", f"message {i} " * 30) for i in range(n)]
    )


def test_repeated_count_is_stable():
    provider = _provider()
    history = _history(20)
    first = _count(provider, history)
    second = _count(provider, history)
    assert first == second == _fresh_count(history)


def test_incremental_matches_fresh_on_append():
    provider = _provider()
    history = _history(20)
    _count(provider, history)  # warm the memo

    history.append(_text_message("user", "a freshly appended follow-up message " * 10))
    incremental = _count(provider, history)

    assert incremental == _fresh_count(history)


def test_fingerprint_invalidates_on_inplace_tool_result_truncation():
    big = "x" * 20_000
    result = ToolResult(tool_use_id="call_1", content=big, name="do", is_error=False)
    history = MessageHistory(
        [
            _text_message("user", "run the tool"),
            Message(role="assistant", content=[ToolCall(id="call_1", name="do", input={})]),
            Message(role="tool", content=[result]),
        ]
    )
    provider = _provider()
    before = _count(provider, history)

    # Simulate _sanitize_oversized_tool_results truncating the result in place.
    result.content = "x" * 100
    after = _count(provider, history)

    assert after != before  # the change was detected, not served stale from the memo
    assert after == _fresh_count(history)


def test_incremental_matches_fresh_after_object_replacement():
    """Provider adaptation / compaction replace message objects; the memo must not drift."""
    provider = _provider()
    history = _history(10)
    _count(provider, history)

    # Replace a message at a fixed position with a new object of different content,
    # as adapt_history_for_provider does when it rewrites a message.
    history[4] = _text_message("assistant", "ADAPTED content that differs in length " * 5)
    incremental = _count(provider, history)

    assert incremental == _fresh_count(history)


def test_incremental_matches_fresh_with_system_and_tools():
    provider = _provider()
    system = _text_message("system", "You are a helpful assistant. " * 20)
    tools = [
        ToolDefinition(
            name="search",
            description="Search the web for a query.",
            parameters=[ToolParameter("query", "string", "the search query", required=True)],
        ),
        ToolDefinition(
            name="read_file",
            description="Read a file from disk.",
            parameters=[ToolParameter("path", "string", "absolute path", required=True)],
        ),
    ]
    history = _history(12)
    _count(provider, history, system=system, tools=tools)  # warm

    history.append(_text_message("assistant", "tool planning text " * 8))
    incremental = _count(provider, history, system=system, tools=tools)

    assert incremental == _fresh_count(history, system=system, tools=tools)


def test_shrinking_history_recounts_correctly():
    """A shorter history (e.g. post-compaction tail) must not over-count from the memo."""
    provider = _provider()
    history = _history(30)
    _count(provider, history)

    shorter = MessageHistory(list(history)[:5])
    assert _count(provider, shorter) == _fresh_count(shorter)


def test_thinking_blocks_are_counted():
    """Reasoning replayed to the same provider (DeepSeek ThinkingBlocks) counts toward
    the input budget. Before this branch existed they scored 0, so the gauge undercounted
    replayed reasoning and auto-compaction fired late while payloads stayed large."""
    reasoning = "a deliberate reasoning step. " * 200
    with_thinking = MessageHistory(
        [Message(role="assistant", content=[ThinkingBlock(thinking=reasoning), TextBlock(text="final answer")])]
    )
    without_thinking = MessageHistory([Message(role="assistant", content=[TextBlock(text="final answer")])])

    delta = _fresh_count(with_thinking) - _fresh_count(without_thinking)
    # The reasoning text alone is hundreds of tokens; require a substantial contribution.
    assert delta > 100


def test_count_tokens_runs_off_the_event_loop_thread():
    """The tiktoken encode must be offloaded to a worker thread so it never blocks the
    Textual/asyncio event loop (the 'freeze between steps' on long DeepSeek sessions)."""
    provider = _provider()
    main_thread = threading.main_thread()
    observed = {}

    original = provider._count_tokens_sync

    def spy(all_messages, tools):
        observed["ran_on_main_thread"] = threading.current_thread() is main_thread
        return original(all_messages, tools)

    provider._count_tokens_sync = spy
    asyncio.run(provider.count_tokens(messages=_history(5), model="gpt-4", tools=[]))

    # asyncio.run drives the loop on the main thread; the encode must run elsewhere.
    assert observed["ran_on_main_thread"] is False
