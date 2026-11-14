#!/usr/bin/env python3
"""Measure GoogleStreamWrapper text accumulation over a synthetic Gemini stream.

Gemini streams answer text as many small chunks. The wrapper keeps the full
answer for ``get_final_message()``; accumulating it with ``self.final_content +=
chunk`` copied the whole buffer per chunk (quadratic), while a parts list joined
once at finalize is linear. Run this before and after changes to
``kolega_code/llm/providers/google.py``:

    uv run python scripts/bench_google_stream.py
"""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace
from typing import Any, Callable, Coroutine, List

from kolega_code.llm.providers.google import GoogleStreamWrapper

# (chunks, chars per chunk): small deltas over a long answer.
CASES = ((5_000, 8), (10_000, 8), (25_000, 8), (50_000, 8))


class _AsyncStream:
    def __init__(self, chunks: List[Any]) -> None:
        self._chunks = chunks
        self._index = 0

    def __aiter__(self) -> "_AsyncStream":
        return self

    async def __anext__(self) -> Any:
        if self._index >= len(self._chunks):
            raise StopAsyncIteration
        chunk = self._chunks[self._index]
        self._index += 1
        return chunk

    async def aclose(self) -> None:
        pass


def _chunks(count: int, size: int) -> List[Any]:
    candidate = SimpleNamespace(content=None, finish_reason=None)
    text = "x" * size
    return [
        SimpleNamespace(text=text, usage_metadata=None, model_version=None, candidates=[candidate])
        for _ in range(count)
    ]


async def _consume(chunks: List[Any]) -> None:
    wrapper = GoogleStreamWrapper(_AsyncStream(chunks), model="gemini-3.5-flash")
    async with wrapper as stream:
        async for _ in stream:
            pass
        await stream.get_final_message()


def _best(run: Callable[[], Coroutine[Any, Any, None]], repeat: int = 3) -> float:
    best = float("inf")
    for _ in range(repeat):
        started = time.perf_counter()
        asyncio.run(run())
        best = min(best, time.perf_counter() - started)
    return best


def main() -> None:
    print(f"{'chunks':>8} {'chars/chunk':>12} {'total chars':>12} {'best':>10}")
    for count, size in CASES:
        chunks = _chunks(count, size)

        async def run() -> None:
            await _consume(chunks)

        print(f"{count:>8} {size:>12} {count * size:>12} {_best(run) * 1000:>9.2f}ms")


if __name__ == "__main__":
    main()
