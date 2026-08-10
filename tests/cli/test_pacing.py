import pytest

from kolega_code.cli.tui.pacing import (
    FLUSH_BASE_INTERVAL,
    PACING_CEILING,
    PACING_DEADBAND,
    FlushPacer,
)


def test_unattached_pacer_returns_base_interval() -> None:
    assert FlushPacer().interval() == FLUSH_BASE_INTERVAL


def test_healthy_loop_returns_exactly_base_interval() -> None:
    assert FlushPacer(lambda: 0.0).interval() == FLUSH_BASE_INTERVAL
    assert FlushPacer(lambda: PACING_DEADBAND - 0.01).interval() == FLUSH_BASE_INTERVAL


def test_interval_scales_with_loop_lateness() -> None:
    # (1 + excess/normalizer) is the loop's measured slowdown factor.
    assert FlushPacer(lambda: 0.2).interval() == pytest.approx(0.10)
    assert FlushPacer(lambda: 1.0).interval() == pytest.approx(0.30)


def test_interval_is_capped_at_ceiling() -> None:
    assert FlushPacer(lambda: 60.0).interval() == PACING_CEILING


def test_interval_is_monotonic_in_lateness() -> None:
    samples = [0.0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
    intervals = [FlushPacer(lambda value=value: value).interval() for value in samples]
    assert intervals == sorted(intervals)


def test_attach_switches_the_lateness_source() -> None:
    pacer = FlushPacer()
    pacer.attach(lambda: 1.0)
    assert pacer.interval() == pytest.approx(0.30)
