"""The CLI must not hold the terminal open when shutdown cannot finish."""

from __future__ import annotations

import threading
import time

from kolega_code.cli.main import _arm_exit_watchdog


def test_exit_watchdog_forces_exit_when_shutdown_is_wedged() -> None:
    exits: list[int] = []
    fired = threading.Event()

    def fake_exit(code: int) -> None:
        exits.append(code)
        fired.set()

    _arm_exit_watchdog(0.01, force_exit=fake_exit)

    assert fired.wait(2.0), "watchdog never fired"
    assert exits == [0]


def test_exit_watchdog_stays_quiet_while_shutdown_completes() -> None:
    exits: list[int] = []
    _arm_exit_watchdog(30.0, force_exit=exits.append)

    time.sleep(0.05)

    assert exits == []
