from __future__ import annotations

import asyncio
import threading
import time
from pathlib import Path

import pytest

from kolega_code.services import workspace_scan
from kolega_code.services.workspace_scan import ScanLimits, ScanOutcome, scan_workspace, scan_workspace_sync


def _paths(outcome: ScanOutcome) -> list[str]:
    return [entry.path for entry in outcome.paths]


def test_scan_preserves_rooted_pathlib_glob_semantics(tmp_path: Path) -> None:
    (tmp_path / "main.py").write_text("", encoding="utf-8")
    (tmp_path / "src" / "pkg").mkdir(parents=True)
    (tmp_path / "src" / "app.py").write_text("", encoding="utf-8")
    (tmp_path / "src" / "pkg" / "nested.py").write_text("", encoding="utf-8")

    root_only = scan_workspace_sync(tmp_path, pattern="*.py")
    recursive = scan_workspace_sync(tmp_path, pattern="**/*.py")
    under_src = scan_workspace_sync(tmp_path, pattern="src/**/*.py")

    assert _paths(root_only) == ["main.py"]
    assert _paths(recursive) == ["main.py", "src/app.py", "src/pkg/nested.py"]
    assert _paths(under_src) == ["src/app.py", "src/pkg/nested.py"]


def test_scan_prunes_excluded_directory_before_descent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "node_modules" / "pkg").mkdir(parents=True)
    (tmp_path / "node_modules" / "pkg" / "index.js").write_text("", encoding="utf-8")
    (tmp_path / "app.js").write_text("", encoding="utf-8")
    real_scandir = workspace_scan.os.scandir

    def guarded_scandir(path):
        if Path(path).name == "node_modules":
            raise AssertionError("excluded directory was traversed")
        return real_scandir(path)

    monkeypatch.setattr(workspace_scan.os, "scandir", guarded_scandir)
    outcome = scan_workspace_sync(
        tmp_path,
        exclude_directories=frozenset({"node_modules"}),
    )

    assert _paths(outcome) == ["app.js"]
    assert outcome.complete is True


def test_scan_stops_after_extra_result_proves_truncation(tmp_path: Path) -> None:
    for index in range(10):
        (tmp_path / f"f{index:02}.txt").write_text("", encoding="utf-8")

    outcome = scan_workspace_sync(tmp_path, limits=ScanLimits(max_results=4))

    assert _paths(outcome) == ["f00.txt", "f01.txt", "f02.txt", "f03.txt"]
    assert outcome.complete is False
    assert outcome.stop_reason == "result_limit"


def test_scan_entry_budget_returns_observed_partial_results(tmp_path: Path) -> None:
    for index in range(10):
        (tmp_path / f"f{index:02}.txt").write_text("", encoding="utf-8")

    outcome = scan_workspace_sync(tmp_path, limits=ScanLimits(max_entries=3))

    assert len(outcome.paths) == 3
    assert set(_paths(outcome)).issubset({f"f{index:02}.txt" for index in range(10)})
    assert outcome.visited_entries == 3
    assert outcome.complete is False
    assert outcome.stop_reason == "entry_limit"


@pytest.mark.asyncio
async def test_async_scan_does_not_block_event_loop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    started = threading.Event()
    release = threading.Event()

    def slow_scan(*args, **kwargs):
        started.set()
        release.wait(timeout=2)
        return ScanOutcome()

    monkeypatch.setattr(workspace_scan, "scan_workspace_sync", slow_scan)
    scan_task = asyncio.create_task(scan_workspace(tmp_path))
    assert await asyncio.to_thread(started.wait, 1)

    marker = asyncio.create_task(asyncio.sleep(0.01, result="responsive"))
    assert await asyncio.wait_for(marker, timeout=0.2) == "responsive"
    release.set()
    await scan_task


@pytest.mark.asyncio
async def test_async_scan_cancellation_signals_worker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    started = threading.Event()
    stopped = threading.Event()

    def cancellable_scan(*args, cancel_event: threading.Event, **kwargs):
        started.set()
        while not cancel_event.is_set():
            time.sleep(0.005)
        stopped.set()
        return ScanOutcome(complete=False, stop_reason="cancelled")

    monkeypatch.setattr(workspace_scan, "scan_workspace_sync", cancellable_scan)
    task = asyncio.create_task(scan_workspace(tmp_path))
    assert await asyncio.to_thread(started.wait, 1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert await asyncio.to_thread(stopped.wait, 0.5)
