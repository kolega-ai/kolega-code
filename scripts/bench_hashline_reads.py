#!/usr/bin/env python3
"""Measure Hashline read-tag hashing cost.

Every ``read`` on the Hashline protocol renders one ``LINE#ID`` anchor per shown
line, and an edit re-verifies the anchors it touches. The ID is an xxHash32 port
in pure Python, so per-line hashing is a measurable share of a 50 KiB read, and
a re-read of the same file repeats all of it.

Run this before and after changing ``kolega_code/agent/tool_backend/hashline_v2.py``:

    uv run python scripts/bench_hashline_reads.py
    uv run python scripts/bench_hashline_reads.py --repeat 15

"first read" clears the line-hash memo before each sample, so it measures a file
the session has not seen; "re-read" keeps it warm, which is what reading a file
again after editing it costs. Content is synthetic but shaped like source: lines
up to the read tool's 50 KiB display budget, with the repetition real code has.
"""

from __future__ import annotations

import argparse
import asyncio
import random
import statistics
import tempfile
import time
from pathlib import Path
from typing import Callable

from kolega_code.agent.tool_backend import hashline_v2
from kolega_code.agent.tool_backend.hashline_v2 import format_hash_lines
from kolega_code.agent.tool_backend.read_file_tool import ReadFileTool
from kolega_code.services.file_system import LocalFileSystem

DISPLAY_BUDGET_BYTES = 50 * 1024  # ReadFileTool.MAX_BYTES
DISTINCT_LINES = 1200
STATEMENTS = (
    "return self._value",
    "if not items:",
    'raise ValueError(f"unexpected {name!r}")',
    'logger.debug("%s finished in %.2f ms", name, elapsed)',
    "self._cache[path] = baseline",
    "for index, entry in enumerate(entries):",
    'assert definition.input_kind == "json"',
    'with open(path, "r", encoding="utf-8") as handle:',
    "result.append((score, entry))",
    "else:",
)


def _content(*, seed: int = 7) -> str:
    """Source-shaped lines drawn from a fixed pool so the mix is reproducible."""
    rng = random.Random(seed)
    pool = [f"{STATEMENTS[index % len(STATEMENTS)]}  # {index:04d}" for index in range(DISTINCT_LINES)]
    rng.shuffle(pool)
    lines: list[str] = []
    size = 0
    while size < DISPLAY_BUDGET_BYTES:
        line = pool[rng.randrange(len(pool))]
        lines.append(line)
        size += len(line.encode("utf-8")) + 1
    return "\n".join(lines) + "\n"


def _median_ms(action: Callable[[], object], repeat: int) -> float:
    samples = []
    for _ in range(repeat):
        started = time.perf_counter()
        action()
        samples.append(time.perf_counter() - started)
    return statistics.median(samples) * 1000.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repeat", type=int, default=9)
    args = parser.parse_args()

    text = _content()
    print(
        f"hashline read hashing — {text.count(chr(10)):,} lines, "
        f"{len(set(text.splitlines())):,} distinct, {len(text.encode()) / 1024:.0f} KiB, "
        f"median of {args.repeat} runs"
    )

    def clear() -> None:
        # Absent on a checkout that predates the memo, so the same script
        # reports both sides of the change.
        cached = getattr(hashline_v2, "_cached_line_hash", None)
        if cached is not None:
            cached.cache_clear()

    def tags_first_read() -> None:
        clear()
        format_hash_lines(text)

    def tags_reread() -> None:
        format_hash_lines(text)

    root = Path(tempfile.mkdtemp(prefix="kolega-bench-hashline-"))
    (root / "module.py").write_text(text, encoding="utf-8")
    tool = ReadFileTool.__new__(ReadFileTool)  # read() touches only the filesystem
    tool.filesystem = LocalFileSystem(root)
    loop = asyncio.new_event_loop()

    def read_once() -> None:
        loop.run_until_complete(tool.read("module.py", line_formatter=format_hash_lines))

    def read_first() -> None:
        clear()
        read_once()

    read_once()  # warm the OS file cache and the read path

    print(f"\n{'workload':>26} {'first read':>13} {'re-read':>10}")
    print(
        f"{'tag rendering':>26} {_median_ms(tags_first_read, args.repeat):11.2f}ms {_median_ms(tags_reread, args.repeat):8.2f}ms"
    )
    print(
        f"{'read tool end to end':>26} {_median_ms(read_first, args.repeat):11.2f}ms {_median_ms(read_once, args.repeat):8.2f}ms"
    )
    loop.close()


if __name__ == "__main__":
    main()
