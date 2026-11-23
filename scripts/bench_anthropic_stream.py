#!/usr/bin/env python3
"""Measure Anthropic tool-argument streaming cost.

Anthropic streams tool input as many small ``input_json_delta`` fragments. The
provider read loop consumes them one chunk at a time, so per-fragment work is
per-turn latency: copying the whole accumulated argument JSON on every fragment
is quadratic in the tool call's size, and a large ``write``/patch payload can
stall the loop for seconds.

Run this before and after changing ``AnthropicStreamWrapper``:

    uv run python scripts/bench_anthropic_stream.py

The workload is synthetic but shaped like a real stream: one ``tool_use``
content block start, then fixed-size partial-JSON fragments. All content is
ASCII so byte and character counts coincide.
"""

from __future__ import annotations

import asyncio
import time
import types
from typing import Any, Iterable

from kolega_code.llm.providers.anthropic import AnthropicStreamWrapper

# (label, total argument bytes, chars per fragment)
CASES = (
    ("small write", 20_000, 8),
    ("medium write", 50_000, 8),
    ("patch", 200_000, 8),
    ("large write", 200_000, 4),
)
REPEATS = 3


def _ns(**kwargs: Any) -> types.SimpleNamespace:
    return types.SimpleNamespace(**kwargs)


class _FakeStream:
    def __init__(self, chunks: list[Any]) -> None:
        self._chunks = chunks

    async def __aenter__(self) -> Any:
        async def gen() -> Any:
            for chunk in self._chunks:
                yield chunk

        return gen()

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> bool:
        return False


def _chunks(total_bytes: int, fragment: int) -> list[Any]:
    body = "x" * total_bytes
    start = _ns(
        type="content_block_start",
        index=0,
        content_block=_ns(type="tool_use", id="toolu_write", name="write"),
    )
    deltas = [
        _ns(
            type="content_block_delta",
            index=0,
            delta=_ns(type="input_json_delta", partial_json=body[offset : offset + fragment]),
        )
        for offset in range(0, total_bytes, fragment)
    ]
    return [start, *deltas]


async def _consume(chunks: Iterable[Any]) -> None:
    async with AnthropicStreamWrapper(_FakeStream(list(chunks))) as stream:
        async for _chunk in stream:
            pass


def _best_ms(chunks: list[Any]) -> float:
    samples = []
    for _ in range(REPEATS):
        started = time.perf_counter()
        asyncio.run(_consume(chunks))
        samples.append(time.perf_counter() - started)
    return min(samples) * 1000


def main() -> None:
    print(f"anthropic stream benchmark — {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'workload':>13} {'bytes':>8} {'fragments':>10} {'time':>12}")
    for label, total_bytes, fragment in CASES:
        chunks = _chunks(total_bytes, fragment)
        best = _best_ms(chunks)
        print(f"{label:>13} {total_bytes:>8} {len(chunks) - 1:>10} {best:>9.2f} ms")
    print("\ndone.")


if __name__ == "__main__":
    main()
