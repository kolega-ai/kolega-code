"""Bounded, cancellable local workspace traversal.

Agent tools must not call recursive ``pathlib``/``os`` scans on the asyncio event
loop.  This module keeps the synchronous walk small and cooperative so callers can
run it in a worker thread without losing cancellation or allowing an accidentally
broad project root to consume unbounded time and memory.
"""

from __future__ import annotations

import asyncio
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Optional

import pathspec


DEFAULT_SCAN_TIMEOUT_SECONDS = 5.0
DEFAULT_SCAN_MAX_ENTRIES = 50_000
_OUTER_TIMEOUT_GRACE_SECONDS = 1.0


@dataclass(frozen=True)
class ScanLimits:
    timeout_seconds: float = DEFAULT_SCAN_TIMEOUT_SECONDS
    max_entries: int = DEFAULT_SCAN_MAX_ENTRIES
    max_results: Optional[int] = None


@dataclass(frozen=True)
class ScannedPath:
    path: str
    is_dir: bool
    size: int = 0
    modified_time: int = 0


@dataclass
class ScanOutcome:
    paths: list[ScannedPath] = field(default_factory=list)
    visited_entries: int = 0
    elapsed_seconds: float = 0.0
    complete: bool = True
    stop_reason: Optional[str] = None


def compile_workspace_glob(pattern: str) -> pathspec.GitIgnoreSpec:
    normalized = pattern.replace("\\", "/").lstrip("/") or "*"
    # Leading slash anchors patterns without ``**`` to the workspace root,
    # matching Path.glob rather than gitignore's basename-at-any-depth behavior.
    return pathspec.GitIgnoreSpec.from_lines([f"/{normalized}"])


def _fixed_prefix(pattern: str) -> PurePosixPath:
    """Return the directory prefix before the first wildcard component."""
    normalized = pattern.replace("\\", "/").lstrip("/")
    parts = PurePosixPath(normalized).parts
    fixed: list[str] = []
    for part in parts:
        if any(char in part for char in ("*", "?", "[")):
            break
        fixed.append(part)
    if len(fixed) == len(parts) and fixed:
        fixed.pop()  # exact path: scan its parent so the path itself is tested
    return PurePosixPath(*fixed) if fixed else PurePosixPath(".")


def scan_workspace_sync(
    root: Path,
    *,
    pattern: str = "**/*",
    include_files: bool = True,
    include_directories: bool = False,
    exclude_directories: frozenset[str] = frozenset(),
    skip_hidden_directories: bool = False,
    binary_extensions: frozenset[str] = frozenset(),
    max_file_size: Optional[int] = None,
    collect_metadata: bool = True,
    ignore_spec: Optional[pathspec.PathSpec] = None,
    limits: ScanLimits = ScanLimits(),
    cancel_event: Optional[threading.Event] = None,
) -> ScanOutcome:
    """Walk *root* within explicit time/entry/result limits.

    The function is intentionally synchronous.  Async callers should use
    :func:`scan_workspace`, which runs it in a worker thread.
    """
    started = time.monotonic()
    deadline = started + max(0.0, limits.timeout_seconds)
    root = Path(root).resolve()
    matcher = compile_workspace_glob(pattern)
    outcome = ScanOutcome()

    def stop(reason: str) -> bool:
        outcome.complete = False
        outcome.stop_reason = reason
        outcome.elapsed_seconds = time.monotonic() - started
        return True

    def should_stop() -> bool:
        if cancel_event is not None and cancel_event.is_set():
            return stop("cancelled")
        if time.monotonic() >= deadline:
            return stop("deadline")
        if outcome.visited_entries >= limits.max_entries:
            return stop("entry_limit")
        return False

    prefix = _fixed_prefix(pattern)
    if any(part in exclude_directories for part in prefix.parts):
        outcome.elapsed_seconds = time.monotonic() - started
        return outcome
    if skip_hidden_directories and any(part.startswith(".") for part in prefix.parts):
        outcome.elapsed_seconds = time.monotonic() - started
        return outcome
    start_dir = root if str(prefix) == "." else root.joinpath(*prefix.parts)
    if not start_dir.exists():
        outcome.elapsed_seconds = time.monotonic() - started
        return outcome

    if start_dir.is_file():
        scan_stack: list[tuple[Path, PurePosixPath]] = [(start_dir.parent, prefix.parent)]
    else:
        scan_stack = [(start_dir, prefix)]

    while scan_stack:
        if should_stop():
            return outcome
        directory, relative_directory = scan_stack.pop()
        pending_stop_reason: Optional[str] = None
        try:
            discovered = []
            with os.scandir(directory) as iterator:
                for entry in iterator:
                    if cancel_event is not None and cancel_event.is_set():
                        pending_stop_reason = "cancelled"
                        break
                    if time.monotonic() >= deadline:
                        pending_stop_reason = "deadline"
                        break
                    if outcome.visited_entries >= limits.max_entries:
                        pending_stop_reason = "entry_limit"
                        break
                    outcome.visited_entries += 1
                    discovered.append(entry)
        except (FileNotFoundError, NotADirectoryError, PermissionError, OSError):
            continue

        child_directories: list[tuple[Path, PurePosixPath]] = []
        for entry in sorted(discovered, key=lambda item: item.name.casefold()):
            if cancel_event is not None and cancel_event.is_set():
                stop("cancelled")
                return outcome
            if time.monotonic() >= deadline:
                stop("deadline")
                return outcome
            relative = relative_directory / entry.name if str(relative_directory) != "." else PurePosixPath(entry.name)
            relative_text = relative.as_posix()
            try:
                is_directory = entry.is_dir(follow_symlinks=False)
                is_file = entry.is_file(follow_symlinks=False)
            except OSError:
                continue

            if is_directory:
                if entry.name in exclude_directories or (skip_hidden_directories and entry.name.startswith(".")):
                    continue
                if ignore_spec is not None and ignore_spec.match_file(relative_text + "/"):
                    continue
                child_directories.append((Path(entry.path), relative))
                if not include_directories or not matcher.match_file(relative_text + "/"):
                    continue
                scanned = ScannedPath(path=relative_text, is_dir=True)
            elif is_file:
                if not include_files or not matcher.match_file(relative_text):
                    continue
                if ignore_spec is not None and ignore_spec.match_file(relative_text):
                    continue
                if Path(entry.name).suffix.lower() in binary_extensions:
                    continue
                if collect_metadata or max_file_size is not None:
                    try:
                        stat_result = entry.stat(follow_symlinks=False)
                    except OSError:
                        continue
                    if max_file_size is not None and stat_result.st_size > max_file_size:
                        continue
                    scanned = ScannedPath(
                        path=relative_text,
                        is_dir=False,
                        size=int(stat_result.st_size) if collect_metadata else 0,
                        modified_time=int(stat_result.st_mtime) if collect_metadata else 0,
                    )
                else:
                    scanned = ScannedPath(path=relative_text, is_dir=False)
            else:
                continue

            outcome.paths.append(scanned)
            if limits.max_results is not None and len(outcome.paths) >= limits.max_results:
                stop("result_limit")
                return outcome

        if pending_stop_reason is not None:
            stop(pending_stop_reason)
            return outcome

        # Stack is LIFO; reverse to visit alphabetically first.
        scan_stack.extend(reversed(child_directories))

    outcome.elapsed_seconds = time.monotonic() - started
    return outcome


async def scan_workspace(root: Path, **kwargs) -> ScanOutcome:
    """Run :func:`scan_workspace_sync` off-loop with prompt cancellation."""
    limits = kwargs.get("limits")
    if not isinstance(limits, ScanLimits):
        limits = ScanLimits()
        kwargs["limits"] = limits
    cancel_event = threading.Event()
    kwargs["cancel_event"] = cancel_event
    task = asyncio.create_task(asyncio.to_thread(scan_workspace_sync, root, **kwargs))
    try:
        return await asyncio.wait_for(
            asyncio.shield(task),
            timeout=limits.timeout_seconds + _OUTER_TIMEOUT_GRACE_SECONDS,
        )
    except asyncio.CancelledError:
        cancel_event.set()
        task.cancel()
        raise
    except asyncio.TimeoutError:
        cancel_event.set()
        task.cancel()
        return ScanOutcome(
            elapsed_seconds=limits.timeout_seconds + _OUTER_TIMEOUT_GRACE_SECONDS,
            complete=False,
            stop_reason="deadline",
        )
