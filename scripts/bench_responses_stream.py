#!/usr/bin/env python3
"""Measure Responses streaming accumulation cost.

``ResponsesStreamWrapper`` stores the answer, the raw reasoning and each tool
call's argument JSON while the stream is consumed. Appending a delta to a string
held on the wrapper copies the whole accumulated text, so a long stream is
quadratic in the number of deltas: a large answer, an extended reasoning
segment, or a big tool call (a patch, a heredoc) stalls the provider read loop.

Run this before and after changing the accumulators:

    uv run python scripts/bench_responses_stream.py

The workload is synthetic: fixed-size deltas of the shape providers send
(``response.output_text.delta``, ``response.reasoning_text.delta``,
``response.function_call_arguments.delta``), a terminal ``response.completed``
event, and one ``get_final_message()`` to materialize the result. All content is
ASCII so byte and character counts coincide.
"""

from __future__ import annotations

import asyncio
import time
import types
from typing import Any, Iterable

from kolega_code.llm.providers.responses_common import ResponsesStreamWrapper

# (label, event type, deltas, chars per delta)
CASES = (
    ("answer", "response.output_text.delta", 5_000, 50),
    ("answer", "response.output_text.delta", 20_000, 50),
    ("answer", "response.output_text.delta", 50_000, 20),
    ("reasoning", "response.reasoning_text.delta", 5_000, 50),
    ("reasoning", "response.reasoning_text.delta", 20_000, 50),
    ("tool arguments", "response.function_call_arguments.delta", 5_000, 50),
    ("tool arguments", "response.function_call_arguments.delta", 20_000, 50),
)
REPEATS = 3


def _ns(**kwargs: Any) -> types.SimpleNamespace:
    return types.SimpleNamespace(**kwargs)


class _FakeStream:
    def __init__(self, events: list[Any]) -> None:
        self._events = events

    def __aiter__(self) -> Any:
        async def gen() -> Any:
            for event in self._events:
                yield event

        return gen()


def _deltas(event_type: str, count: int, chunk: int) -> list[Any]:
    body = "x" * chunk
    if event_type == "response.function_call_arguments.delta":
        # A parseable payload keeps get_final_message() off its fallback-log path,
        # so the measurement stays in the accumulator.
        return [
            _ns(type=event_type, item_id="call_1", delta='{"text": "'),
            *(_ns(type=event_type, item_id="call_1", delta=body) for _ in range(count)),
            _ns(type=event_type, item_id="call_1", delta='"}'),
        ]
    return [_ns(type=event_type, delta=body) for _ in range(count)]


def _completed() -> Any:
    return _ns(
        type="response.completed",
        response=_ns(status="completed", usage=None, output=[]),
    )


async def _consume(events: Iterable[Any]) -> None:
    wrapper = ResponsesStreamWrapper(_FakeStream([*events, _completed()]))
    async with wrapper as stream:
        async for _chunk in stream:
            pass
    await wrapper.get_final_message()


def _best_ms(events: list[Any]) -> float:
    samples = []
    for _ in range(REPEATS):
        started = time.perf_counter()
        asyncio.run(_consume(events))
        samples.append(time.perf_counter() - started)
    return min(samples) * 1000


def main() -> None:
    print(f"responses stream benchmark — {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'accumulator':>15} {'deltas':>7} {'chars':>10} {'time':>12}")
    for label, event_type, count, chunk in CASES:
        events = _deltas(event_type, count, chunk)
        best = _best_ms(events)
        chars = sum(len(getattr(event, "delta", "")) for event in events)
        print(f"{label:>15} {count:>7} {chars:>10} {best:>9.2f} ms")
    print("\ndone.")


if __name__ == "__main__":
    main()
