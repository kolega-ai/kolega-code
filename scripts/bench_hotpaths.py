#!/usr/bin/env python3
"""Micro-benchmarks for the TUI streaming hot paths.

These isolate the three recompute-from-scratch paths that turn a 15-30s DeepSeek
turn into a multi-minute event-loop-blocking crawl on slow machines (see the
"freeze / slows to a crawl" investigation):

  1. token counting      - OpenAIProvider.count_tokens over the whole history
  2. stream accumulation  - ConversationEntry text growth per chunk
  3. streaming render      - _format_indented_text over the whole growing buffer per flush

Run it before and after the optimizations and compare the printed tables:

    uv run python scripts/bench_hotpaths.py

The numbers that matter:
  - token counting: the "+1 msg recount" column should collapse toward the cost of
    a single new message once counting is incremental (it tracks the full recount today).
  - accumulation: "concat" should grow ~quadratically vs "list+join" linear.
  - render: "full stream" simulates flushing as the buffer grows; pre-fix it is ~O(n^2).
"""

from __future__ import annotations

import asyncio
import time
from typing import Callable

from kolega_code.cli.tui.transcript import TranscriptRenderingMixin
from kolega_code.llm.models import Message, MessageHistory, TextBlock
from kolega_code.llm.providers.openai import OpenAIProvider


def _best(fn: Callable[[], object], repeat: int = 3) -> float:
    best = float("inf")
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def _ms(seconds: float) -> str:
    return f"{seconds * 1000:9.2f} ms"


# --------------------------------------------------------------------------- #
# 1. Token counting
# --------------------------------------------------------------------------- #
def _make_history(n_messages: int, chars_each: int) -> MessageHistory:
    body = ("token text " * (chars_each // 11 + 1))[:chars_each]
    msgs = []
    for i in range(n_messages):
        role = "user" if i % 2 == 0 else "assistant"
        msgs.append(Message(role=role, content=[TextBlock(text=body)]))
    return MessageHistory(msgs)


async def _bench_token_counting() -> None:
    print("\n== 1. Token counting (OpenAIProvider.count_tokens) ==")
    print(f"{'messages':>9} {'chars/msg':>10} {'full count':>14} {'+1 msg recount':>16}")
    provider = OpenAIProvider.__new__(OpenAIProvider)  # count_tokens uses no __init__ state

    async def count(history: MessageHistory) -> None:
        await provider.count_tokens(messages=history, system=None, model="gpt-4", tools=[])

    for n in (50, 200, 800):
        chars = 2000
        hist = _make_history(n, chars)
        await count(hist)  # warm the memo / encoding

        t0 = time.perf_counter()
        await count(hist)
        t_full = time.perf_counter() - t0

        # Append one new message and recount: this is the per-iteration cost during a turn.
        hist.append(Message(role="user", content=[TextBlock(text="one freshly added message " * 40)]))
        t0 = time.perf_counter()
        await count(hist)
        t_recount = time.perf_counter() - t0

        print(f"{n:>9} {chars:>10} {_ms(t_full):>14} {_ms(t_recount):>16}")


# --------------------------------------------------------------------------- #
# 2. Stream accumulation
# --------------------------------------------------------------------------- #
class _Holder:
    """Mimics ConversationEntry: content lives on an attribute, so the CPython
    in-place str-concat optimization (which needs refcount 1) does not apply."""

    def __init__(self) -> None:
        self.content = ""


def _bench_accumulation() -> None:
    print("\n== 2. Stream accumulation (per-chunk text growth, on an attribute) ==")
    print(f"{'target chars':>12} {'chunks':>8} {'attr +=':>14} {'list+join':>14}")
    chunk = "x" * 50  # DeepSeek-style ~50-char reasoning deltas
    for target in (50_000, 200_000, 1_000_000):
        n = target // len(chunk)

        def concat() -> str:
            holder = _Holder()
            for _ in range(n):
                holder.content += chunk
            return holder.content

        def list_join() -> str:
            holder = _Holder()
            parts: list[str] = []
            for _ in range(n):
                parts.append(chunk)
            holder.content = "".join(parts)
            return holder.content

        print(f"{target:>12} {n:>8} {_ms(_best(concat)):>14} {_ms(_best(list_join)):>14}")


# --------------------------------------------------------------------------- #
# 3. Streaming render
# --------------------------------------------------------------------------- #
def _bench_render() -> None:
    from kolega_code.cli.tui.transcript import _IndentedRenderState

    print("\n== 3. Streaming render (flush-as-it-grows, step=4KB) ==")
    print(f"{'buffer chars':>12} {'OLD reformat-each-flush':>26} {'NEW incremental':>18}")
    mixin = TranscriptRenderingMixin.__new__(TranscriptRenderingMixin)  # methods use no instance state
    line = "a moderately sized line of reasoning output\n"
    step = 4000
    for size in (50_000, 200_000, 1_000_000):
        content = (line * (size // len(line) + 1))[:size]

        # OLD: re-split + rebuild the whole buffer on every flush (O(n^2)).
        def old() -> None:
            pos = 0
            while pos < len(content):
                pos = min(len(content), pos + step)
                mixin._format_indented_text(content[:pos])

        # NEW: append only the newly-arrived slice each flush (O(delta), O(n) total).
        def new() -> None:
            state = _IndentedRenderState(None)
            pos = 0
            while pos < len(content):
                nxt = min(len(content), pos + step)
                mixin._extend_indented(state, content[pos:nxt])
                pos = nxt

        print(f"{size:>12} {_ms(_best(old)):>26} {_ms(_best(new)):>18}")


async def _main() -> None:
    print(f"python hot-path benchmark — {time.strftime('%Y-%m-%d %H:%M:%S')}")
    await _bench_token_counting()
    _bench_accumulation()
    _bench_render()
    print("\ndone.")


if __name__ == "__main__":
    asyncio.run(_main())
