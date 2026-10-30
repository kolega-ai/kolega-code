#!/usr/bin/env python3
"""Measure replay cost for terminal-heavy sessions.

``fold`` appends one chunk per ``terminal_output`` event. Rebuilding the whole
retained buffer on every append makes replay quadratic in session length, so a
long build log stalls the web state endpoint, the replay player manifest, and
ACP session loads. Run this before and after changes to the terminal buffer:

    uv run python scripts/bench_projection_replay.py
"""

from __future__ import annotations

import time

from kolega_code.events import AgentEvent, KnownEventType
from kolega_code.session.projection import TERMINAL_BUFFER_CHARS, replay

# (events, chars per chunk): PTY reads are up to 64 KiB; 512-4096 bytes is the
# burst pattern of a verbose build once the reader coalesces.
CASES = ((200, 8_192), (500, 8_192), (1_000, 8_192), (3_000, 2_048), (10_000, 512), (20_000, 256))


def _events(count: int, chunk: int) -> list[AgentEvent]:
    body = "x" * chunk
    return [
        AgentEvent(
            session_id="bench",
            sender="agent",
            event_type=KnownEventType.TERMINAL_OUTPUT,
            content={"output": body, "display_output": body},
            seq=index + 1,
            elapsed_ms=index,
        )
        for index in range(count)
    ]


def _best(events: list[AgentEvent], repeat: int = 3) -> float:
    best = float("inf")
    for _ in range(repeat):
        started = time.perf_counter()
        state = replay(events)
        best = min(best, time.perf_counter() - started)
        assert len(state.terminal) <= TERMINAL_BUFFER_CHARS
    return best


def main() -> None:
    print(f"projection replay benchmark — cap {TERMINAL_BUFFER_CHARS:,} chars")
    print(f"{'events':>7} {'chunk':>7} {'fed chars':>11} {'replay ms':>10}")
    for count, chunk in CASES:
        print(f"{count:>7} {chunk:>7} {count * chunk:>11,} {_best(_events(count, chunk)) * 1000:>10.2f}")


if __name__ == "__main__":
    main()
