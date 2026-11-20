#!/usr/bin/env python3
"""Measure @-mention ranking cost for a cached workspace index.

Every keystroke inside an ``@`` query rescores the cached index, so this runs on
the TUI event loop: ``cached_search`` must stay well under a frame or typing in a
large checkout stutters. Each rescore folds every path and scans it twice for a
subsequence, so a change to that work shows up as keystroke latency.

The workload below builds an in-memory index of ``--entries`` paths (no filesystem
walk) and types a few queries one prefix at a time, which is what the composer
does. Run it before and after a change to ``kolega_code/cli/file_index.py``:

    uv run python scripts/bench_mention_ranking.py
    uv run python scripts/bench_mention_ranking.py --entries 20000 --repeat 15
"""

from __future__ import annotations

import argparse
import statistics
import time
from typing import Callable, Iterable

from kolega_code.cli.file_index import IndexEntry, WorkspaceFileIndex

TYPED_QUERIES = ("unit_0123", "pkg_03/src", "mod_007", "readme", "widgetzz")
PATHS_PER_PACKAGE = 125
DOC_ENTRIES = 64


def _entries(count: int) -> tuple[IndexEntry, ...]:
    package_entries = max(0, count - DOC_ENTRIES)
    entries = [
        IndexEntry(
            path=f"pkg_{index // PATHS_PER_PACKAGE:03d}/src/mod_{index // PATHS_PER_PACKAGE:02d}/"
            f"unit_{index % PATHS_PER_PACKAGE:04d}_module.py",
            is_dir=False,
        )
        for index in range(package_entries)
    ]
    entries.extend(
        IndexEntry(path=f"docs/readme_{index:03d}.md", is_dir=False)
        for index in range(min(DOC_ENTRIES, count - len(entries)))
    )
    return tuple(entries)


def _median_ms(action: Callable[[], object], repeat: int) -> float:
    samples = []
    for _ in range(repeat):
        started = time.perf_counter()
        action()
        samples.append(time.perf_counter() - started)
    return statistics.median(samples) * 1000.0


def _index(entries: Iterable[IndexEntry]) -> WorkspaceFileIndex:
    index = WorkspaceFileIndex.__new__(WorkspaceFileIndex)  # ranking reads only the snapshot
    index._entries = tuple(entries)
    index._cached_ranking = None
    return index


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entries", type=int, default=5_000)
    parser.add_argument("--repeat", type=int, default=9)
    args = parser.parse_args()

    entries = _entries(args.entries)
    print(f"mention ranking — {len(entries):,} indexed paths, median of {args.repeat} runs")

    print(f"\n{'typed query':>14} {'full scan':>12} {'prefix typing':>16}")
    for typed in TYPED_QUERIES:
        index = _index(entries)
        index.cached_search(typed)  # warm

        def full_scan() -> None:
            index._cached_ranking = None
            index.cached_search(typed)

        def type_query() -> None:
            index._cached_ranking = None
            for length in range(1, len(typed) + 1):
                index.cached_search(typed[:length])

        full = _median_ms(full_scan, args.repeat)
        typed_ms = _median_ms(type_query, args.repeat)
        print(f"{typed:>14} {full:11.2f}ms {typed_ms:15.2f}ms")


if __name__ == "__main__":
    main()
