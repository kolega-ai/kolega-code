"""Flush-cadence pacing for the TUI.

Every buffered output surface (terminal pane, log pane, transcript, modal
inspectors) coalesces writes on one base cadence. :class:`FlushPacer` stretches
that cadence when the event loop is measurably running late, so sustained load
degrades to a lower frame rate instead of freezing the loop. Content is never
dropped or hidden — only batched into fewer, larger flushes.
"""

from __future__ import annotations

from typing import Callable, Optional

# 20 Hz — the healthy-loop cadence for every buffered surface.
FLUSH_BASE_INTERVAL = 0.05
# The watchdog's first histogram edge: the floor of human-visible chop. Below it
# the pacer returns exactly the base interval.
PACING_DEADBAND = 0.10
# 2 Hz floor — the UI stays visibly alive in the worst backoff.
PACING_CEILING = 0.50
# The watchdog beat interval, so (1 + excess/normalizer) is the loop's measured
# slowdown factor: stretch flushes by how late the loop actually runs.
PACING_NORMALIZER = 0.20

# Coalesce less aggressively as the live streaming entry grows: each flush still costs
# Textual an O(height) re-measure of an auto-height widget, so for very large reasoning
# streams we trade a little update latency for far fewer full re-measures. Sizes are
# characters of the live entry; see transcript._invalidate_conversation.
RENDER_COALESCE_INTERVAL_MEDIUM = 0.12
RENDER_COALESCE_INTERVAL_LARGE = 0.25
RENDER_COALESCE_MEDIUM_CHARS = 40_000
RENDER_COALESCE_LARGE_CHARS = 200_000


class FlushPacer:
    """Maps recent event-loop lateness to a flush-timer interval.

    Deliberately stateless: the step-up/decay dynamics live in the signal itself
    (``ResponsivenessWatchdog.recent_excess``, a decaying peak-hold), so this is
    a pure mapping, every arming site sees the same value at the same instant,
    and tests only need a fake lateness callable.
    """

    def __init__(self, lateness: Optional[Callable[[], float]] = None) -> None:
        self._lateness = lateness

    def attach(self, lateness: Callable[[], float]) -> None:
        self._lateness = lateness

    def interval(self) -> float:
        # Diagnostics init can fail (the app nulls the watchdog); unattached
        # means "assume healthy" and the TUI runs at the fixed base cadence.
        if self._lateness is None:
            return FLUSH_BASE_INTERVAL
        excess = self._lateness()
        if excess < PACING_DEADBAND:
            return FLUSH_BASE_INTERVAL
        return min(PACING_CEILING, FLUSH_BASE_INTERVAL * (1.0 + excess / PACING_NORMALIZER))
