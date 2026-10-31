#!/usr/bin/env python3
"""Measure the import work `kolega-code` pays before it handles any command.

Every `kolega-code` invocation (``--help``, a one-shot command, the TUI) imports
``kolega_code.cli.main`` first, so anything that module imports eagerly is
startup latency. This measures that import in a subprocess, using both wall time
and the cumulative ``-X importtime`` figure so interpreter startup does not
dilute the change being measured:

    uv run python scripts/bench_cli_startup.py

The first run is discarded to warm the OS file cache; the result reports the
median and best of the remaining runs.
"""

from __future__ import annotations

import statistics
import subprocess
import sys
import time

TARGET = "kolega_code.cli.main"
RUNS = 7


def _run() -> tuple[float, float]:
    """Return (subprocess wall seconds, target import seconds)."""
    started = time.perf_counter()
    proc = subprocess.run(
        [sys.executable, "-X", "importtime", "-c", f"import {TARGET}"],
        capture_output=True,
        text=True,
        check=True,
    )
    wall = time.perf_counter() - started
    import_seconds = 0.0
    for line in proc.stderr.splitlines():
        if not line.startswith("import time:"):
            continue
        _, cumulative, name = line.split("|")
        if name.strip() == TARGET:
            import_seconds = int(cumulative.strip()) / 1_000_000
    return wall, import_seconds


def main() -> None:
    _run()
    samples = [_run() for _ in range(RUNS)]
    walls = [wall for wall, _ in samples]
    imports = [imported for _, imported in samples]
    print(f"{TARGET} over {RUNS} runs")
    print(f"  import    median {statistics.median(imports) * 1000:7.1f} ms   best {min(imports) * 1000:7.1f} ms")
    print(f"  + python  median {statistics.median(walls) * 1000:7.1f} ms   best {min(walls) * 1000:7.1f} ms")


if __name__ == "__main__":
    main()
