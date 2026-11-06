#!/usr/bin/env python3
"""Measure session-diff checkpoint cost in a repo full of untracked files.

Every TUI turn captures a checkpoint, and every turn end refreshes the diff.
Both scan ``git status -uall``, so a checkout with many untracked, non-ignored
files pays for every one of them each turn (#899). Run this before and after
changes to ``kolega_code/cli/tui/session_diff.py``:

    uv run python scripts/bench_session_diff.py
    uv run python scripts/bench_session_diff.py --files 50000 --keep

All content is synthetic. The repo and the checkpoint spool live in a temp
directory that is deleted afterwards unless ``--keep`` is given.
"""

from __future__ import annotations

import argparse
import os
import random
import shutil
import statistics
import subprocess
import tempfile
import time
import tracemalloc
from pathlib import Path
from typing import Callable


def _git(root: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=root, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _build_repo(root: Path, *, files: int, large_mb: int, large_count: int, binaries: int) -> None:
    root.mkdir(parents=True)
    _git(root, "init", "-q", "-b", "main")
    _git(root, "config", "user.email", "bench@example.com")
    _git(root, "config", "user.name", "Bench")
    _git(root, "config", "commit.gpgsign", "false")
    src = root / "src"
    src.mkdir()
    for index in range(200):
        (src / f"module_{index}.py").write_text(f"VALUE = {index}\n" * 40, encoding="utf-8")
    _git(root, "add", ".")
    _git(root, "commit", "-q", "-m", "initial")
    for index in range(20):
        with (src / f"module_{index}.py").open("a", encoding="utf-8") as handle:
            handle.write("# uncommitted edit\n")

    rng = random.Random(899)
    per_dir = 1_000
    for index in range(files):
        directory = root / "artifacts" / f"run_{index // per_dir:03d}"
        if index % per_dir == 0:
            directory.mkdir(parents=True)
        body = '{"step": %d, "values": [%s]}\n' % (index, ", ".join(str(rng.random()) for _ in range(180)))
        (directory / f"item_{index:06d}.json").write_text(body, encoding="utf-8")

    recordings = root / "recordings"
    recordings.mkdir()
    line = "✓ frame " + "x" * 90 + "\n"
    for index in range(large_count):
        with (recordings / f"session_{index}.cast").open("w", encoding="utf-8") as handle:
            for _ in range(large_mb * 1024 * 1024 // len(line.encode("utf-8"))):
                handle.write(line)

    media = root / "media"
    media.mkdir()
    for index in range(binaries):
        (media / f"image_{index}.png").write_bytes(b"\x89PNG\x00" + rng.randbytes(1024 * 1024))


def _spool_bytes(state_dir: Path) -> int:
    total = 0
    for dirpath, _dirs, names in os.walk(state_dir):
        for name in names:
            try:
                total += os.path.getsize(os.path.join(dirpath, name))
            except OSError:
                pass
    return total


def _timed(action: Callable[[], object]) -> float:
    started = time.perf_counter()
    action()
    return time.perf_counter() - started


def _median_seconds(action: Callable[[], object], repeat: int) -> float:
    return statistics.median(_timed(action) for _ in range(repeat))


def _close(tracker: object) -> None:
    close = getattr(tracker, "close", None)
    if callable(close):
        close()


def _touch_small_files(root: Path, turn: int, count: int) -> None:
    for index in range(count):
        path = root / "artifacts" / "run_000" / f"item_{index:06d}.json"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(f"// turn {turn}\n")


def _timing_pass(root: Path) -> list[tuple[str, float]]:
    from kolega_code.cli.tui.session_diff import GitSessionDiffTracker

    tracker = GitSessionDiffTracker.create(root)
    assert tracker is not None
    rows = [("startup baseline", _timed(lambda: tracker.capture_baseline("start")))]
    turn = iter(range(1, 1_000))
    rows.append(("turn capture, no changes", _median_seconds(lambda: tracker.capture_checkpoint(f"t{next(turn)}"), 5)))

    def changed_capture() -> None:
        number = next(turn)
        _touch_small_files(root, number, 10)
        tracker.capture_checkpoint(f"t{number}")

    rows.append(("turn capture, 10 files changed", _median_seconds(changed_capture, 3)))
    rows.append(("end-of-turn refresh", _median_seconds(lambda: tracker.refresh(), 3)))
    _close(tracker)
    return rows


def _memory_pass(root: Path, captures: int) -> tuple[float, float]:
    from kolega_code.cli.tui.session_diff import GitSessionDiffTracker

    tracemalloc.start()
    tracker = GitSessionDiffTracker.create(root)
    assert tracker is not None
    tracker.capture_baseline("start")
    after_baseline = tracemalloc.get_traced_memory()[0]
    for number in range(1, captures + 1):
        _touch_small_files(root, 10_000 + number, 3)
        tracker.capture_checkpoint(f"m{number}")
    after_captures = tracemalloc.get_traced_memory()[0]
    _close(tracker)
    tracemalloc.stop()
    return after_baseline / 1e6, after_captures / 1e6


def _issue_repro(work: Path) -> list[tuple[int, float]]:
    """The #899 reproduction: one ~2 MB non-ASCII file appended to every turn."""
    from kolega_code.cli.tui.session_diff import GitSessionDiffTracker

    root = work / "repro"
    root.mkdir()
    _git(root, "init", "-q")
    notes = root / "notes.log"
    notes.write_text("✓ start\n" + ("x" * 99 + "\n") * 20_000, encoding="utf-8")
    tracemalloc.start()
    tracker = GitSessionDiffTracker.create(root)
    assert tracker is not None
    tracker.capture_baseline("b")
    samples: list[tuple[int, float]] = []
    for turn in range(1, 61):
        with notes.open("a", encoding="utf-8") as handle:
            handle.write(f"turn {turn}\n")
        tracker.capture_checkpoint(f"t{turn}")
        if turn in (1, 10, 25, 50, 60):
            samples.append((turn, tracemalloc.get_traced_memory()[0] / 1e6))
    _close(tracker)
    tracemalloc.stop()
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--files", type=int, default=20_000, help="small untracked files (default 20000)")
    parser.add_argument("--large-mb", type=int, default=30, help="size of each large .cast file in MB")
    parser.add_argument("--large-count", type=int, default=3, help="number of large .cast files")
    parser.add_argument("--binaries", type=int, default=20, help="number of 1 MB binary files")
    parser.add_argument("--captures", type=int, default=60, help="captures in the memory pass")
    parser.add_argument("--keep", action="store_true", help="keep the temp repo and spool")
    args = parser.parse_args()

    work = Path(tempfile.mkdtemp(prefix="kolega-bench-session-diff-"))
    state_dir = work / "state"
    os.environ["KOLEGA_CODE_STATE_DIR"] = str(state_dir)
    root = work / "repo"
    try:
        started = time.perf_counter()
        _build_repo(
            root, files=args.files, large_mb=args.large_mb, large_count=args.large_count, binaries=args.binaries
        )
        on_disk = sum(path.stat().st_size for path in root.rglob("*") if path.is_file() and ".git" not in path.parts)
        print(
            f"repo: {args.files:,} small untracked files, {args.large_count} x {args.large_mb} MB .cast, "
            f"{args.binaries} x 1 MB binaries, {on_disk / 1e6:,.0f} MB untracked "
            f"(built in {time.perf_counter() - started:.1f}s)"
        )
        status = _median_seconds(
            lambda: subprocess.run(
                ["git", "--no-optional-locks", "status", "--porcelain", "-z", "-uall"],
                cwd=root,
                check=True,
                stdout=subprocess.DEVNULL,
            ),
            3,
        )
        print(f"{'git status -uall':<34} {status:>8.3f}s")
        for label, seconds in _timing_pass(root):
            print(f"{label:<34} {seconds:>8.3f}s")
        baseline_mb, captures_mb = _memory_pass(root, args.captures)
        print(f"{'traced memory after baseline':<34} {baseline_mb:>8.1f} MB")
        print(f"{f'traced memory after {args.captures} captures':<34} {captures_mb:>8.1f} MB")
        print(f"{'spool on disk after close':<34} {_spool_bytes(state_dir) / 1e6:>8.1f} MB")
        print("issue repro (2 MB non-ASCII file appended each turn):")
        for turn, megabytes in _issue_repro(work):
            print(f"  turn {turn:>2}: {megabytes:>6.0f} MB")
    finally:
        if args.keep:
            print(f"kept {work}")
        else:
            shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
