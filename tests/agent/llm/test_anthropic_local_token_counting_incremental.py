"""Incremental local (tiktoken) token counting for the Anthropic provider must equal a
from-scratch recount.

DeepSeek/moonshot/kimi route through AnthropicProvider with ``use_local_token_counting``,
so ``_count_tokens_local`` runs every agent-loop iteration over the whole history. It now
memoizes per-message counts (identity-keyed, guarded by a recursive length fingerprint) so
only new/changed messages are re-encoded. These tests pin that the memoized total always
equals a fresh provider's full recount — including across in-place tool-result truncation,
object replacement (adaptation/compaction), shrink, and system + tools. No API key needed.
"""

import asyncio

from kolega_code.llm.models import (
    Message,
    MessageHistory,
    TextBlock,
    ToolDefinition,
    ToolParameter,
    ToolResult,
)
from kolega_code.llm.providers.anthropic import AnthropicProvider


def _provider() -> AnthropicProvider:
    # __new__ skips __init__ (no API key / client). Force the local-counting branch.
    provider = AnthropicProvider.__new__(AnthropicProvider)
    provider.use_local_token_counting = True
    return provider


def _count(provider, messages, *, system=None, tools=None) -> int:
    return asyncio.run(
        provider.count_tokens(messages=messages, system=system, model="x", tools=tools or [])
    ).input_tokens


def _fresh_count(messages, *, system=None, tools=None) -> int:
    return _count(_provider(), messages, system=system, tools=tools)


def _text(role: str, text: str) -> Message:
    return Message(role=role, content=[TextBlock(text=text)])


def _history(n: int) -> MessageHistory:
    return MessageHistory([_text("user" if i % 2 == 0 else "assistant", f"message {i} " * 20) for i in range(n)])


def test_repeated_count_is_stable():
    provider = _provider()
    history = _history(15)
    first = _count(provider, history)
    second = _count(provider, history)
    assert first == second == _fresh_count(history)


def test_incremental_matches_fresh_on_append():
    provider = _provider()
    history = _history(15)
    _count(provider, history)
    history.append(_text("user", "appended follow-up " * 12))
    assert _count(provider, history) == _fresh_count(history)


def test_fingerprint_invalidates_on_inplace_tool_result_truncation():
    result = ToolResult(tool_use_id="t1", content="X" * 8000, name="do", is_error=False)
    history = MessageHistory([_text("user", "run it"), Message(role="tool", content=[result])])
    provider = _provider()
    before = _count(provider, history)
    result.content = "X" * 40  # simulate _sanitize_oversized_tool_results
    after = _count(provider, history)
    assert after != before
    assert after == _fresh_count(history)


def test_incremental_matches_fresh_after_object_replacement():
    provider = _provider()
    history = _history(10)
    _count(provider, history)
    history[3] = _text("assistant", "REPLACED content that differs " * 5)
    assert _count(provider, history) == _fresh_count(history)


def test_incremental_matches_fresh_with_system_and_tools():
    provider = _provider()
    system = _text("system", "You are helpful. " * 12)
    tools = [
        ToolDefinition(
            name="search", description="Search the web.", parameters=[ToolParameter("q", "string", "query", True)]
        ),
        ToolDefinition(
            name="read", description="Read a file.", parameters=[ToolParameter("path", "string", "path", True)]
        ),
    ]
    history = _history(8)
    _count(provider, history, system=system, tools=tools)
    history.append(_text("user", "follow-up"))
    assert _count(provider, history, system=system, tools=tools) == _fresh_count(history, system=system, tools=tools)


def test_shrinking_history_recounts_correctly():
    provider = _provider()
    history = _history(30)
    _count(provider, history)
    shorter = MessageHistory(list(history)[:5])
    assert _count(provider, shorter) == _fresh_count(shorter)
