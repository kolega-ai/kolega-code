import threading
import time
from pathlib import Path

import pytest

from kolega_code.cli import app as cli_app_module
from kolega_code.events import AgentEvent

from ._app_test_utils import _build_sub_agent_test_app, settle_changes_inspector
from .test_app_changes_inspector import _file_edit_preview_event, _init_git_project


def _terminal_output_event(text: str) -> AgentEvent:
    return AgentEvent(event_type="terminal_output", sender="coder", content={"output": text})


async def _wait_for_thread_event(pilot, event: threading.Event, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if event.is_set():
            return
        await pilot.pause(0.01)
    raise AssertionError("Timed out waiting for refresh worker")


@pytest.mark.asyncio
async def test_session_diff_dirty_marks_do_no_git_work_when_inspector_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = _build_sub_agent_test_app(tmp_path, monkeypatch)
    _init_git_project(app.project_path)

    async with app.run_test():
        tracker = app._session_diff_tracker
        assert tracker is not None
        calls = 0

        def refresh(event_paths=()):
            nonlocal calls
            calls += 1
            return []

        monkeypatch.setattr(tracker, "refresh", refresh)

        app._render_event(_terminal_output_event("one"))
        app._render_event(_terminal_output_event("two"))
        app._render_event(_file_edit_preview_event("src/a.py", tool_call_id="a1"))
        app._render_event(_file_edit_preview_event("src/b.py", tool_call_id="b1"))

        assert calls == 0
        assert app._session_diff_dirty is True
        assert app._session_diff_timer is None


@pytest.mark.asyncio
async def test_open_changes_runs_one_background_refresh_and_populates_diff(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = _build_sub_agent_test_app(tmp_path, monkeypatch)
    _init_git_project(app.project_path)

    async with app.run_test() as pilot:
        (app.project_path / "src" / "a.py").write_text("new a\n", encoding="utf-8")
        tracker = app._session_diff_tracker
        assert tracker is not None
        original_refresh = tracker.refresh
        calls = []

        def refresh(event_paths=()):
            calls.append(tuple(event_paths))
            return original_refresh(event_paths)

        monkeypatch.setattr(tracker, "refresh", refresh)

        app.action_open_changes()
        await settle_changes_inspector(app, pilot)

        assert len(calls) == 1
        assert {change.path: change.status for change in app._session_diff_files} == {"src/a.py": "modified"}
        assert app._session_diff_refresh_running is False


@pytest.mark.asyncio
async def test_dirty_mark_during_in_flight_refresh_schedules_trailing_refresh(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli_app_module, "SESSION_DIFF_REFRESH_INTERVAL", 0.05)
    app = _build_sub_agent_test_app(tmp_path, monkeypatch)
    _init_git_project(app.project_path)

    async with app.run_test() as pilot:
        (app.project_path / "src" / "a.py").write_text("new a\n", encoding="utf-8")
        tracker = app._session_diff_tracker
        assert tracker is not None
        original_refresh = tracker.refresh
        started = threading.Event()
        release = threading.Event()
        calls = []

        def refresh(event_paths=()):
            calls.append(tuple(event_paths))
            if len(calls) == 1:
                started.set()
                if not release.wait(timeout=5.0):
                    raise AssertionError("Timed out waiting to release refresh")
            return original_refresh(event_paths)

        monkeypatch.setattr(tracker, "refresh", refresh)

        app.action_open_changes()
        await _wait_for_thread_event(pilot, started)
        app._render_event(_terminal_output_event("dirty while refreshing"))
        release.set()
        await settle_changes_inspector(app, pilot)

        assert len(calls) == 2
        assert app._session_diff_refresh_running is False


@pytest.mark.asyncio
async def test_session_diff_debounce_coalesces_rapid_dirty_marks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli_app_module, "SESSION_DIFF_REFRESH_INTERVAL", 0.05)
    app = _build_sub_agent_test_app(tmp_path, monkeypatch)
    _init_git_project(app.project_path)

    async with app.run_test() as pilot:
        (app.project_path / "src" / "a.py").write_text("new a\n", encoding="utf-8")
        tracker = app._session_diff_tracker
        assert tracker is not None
        original_refresh = tracker.refresh
        calls = []

        def refresh(event_paths=()):
            calls.append(tuple(event_paths))
            return original_refresh(event_paths)

        monkeypatch.setattr(tracker, "refresh", refresh)

        app.action_open_changes()
        await settle_changes_inspector(app, pilot)
        assert len(calls) == 1

        for _ in range(10):
            app._mark_session_diff_dirty()
        await settle_changes_inspector(app, pilot)

        assert len(calls) == 2


@pytest.mark.asyncio
async def test_session_diff_refresh_exception_resets_running_and_can_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = _build_sub_agent_test_app(tmp_path, monkeypatch)
    _init_git_project(app.project_path)

    async with app.run_test() as pilot:
        (app.project_path / "src" / "a.py").write_text("new a\n", encoding="utf-8")
        tracker = app._session_diff_tracker
        assert tracker is not None
        original_refresh = tracker.refresh
        calls = 0

        def refresh(event_paths=()):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise RuntimeError("refresh failed")
            return original_refresh(event_paths)

        monkeypatch.setattr(tracker, "refresh", refresh)

        app.action_open_changes()
        await settle_changes_inspector(app, pilot)

        assert calls == 1
        assert app._session_diff_files == []
        assert app._session_diff_refresh_running is False

        app._start_session_diff_refresh()
        await settle_changes_inspector(app, pilot)

        assert calls == 2
        assert {change.path for change in app._session_diff_files} == {"src/a.py"}
        assert app._session_diff_refresh_running is False


@pytest.mark.asyncio
async def test_start_session_diff_refresh_runs_with_inspector_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = _build_sub_agent_test_app(tmp_path, monkeypatch)
    _init_git_project(app.project_path)

    async with app.run_test() as pilot:
        (app.project_path / "src" / "a.py").write_text("new a\n", encoding="utf-8")
        tracker = app._session_diff_tracker
        assert tracker is not None
        original_refresh = tracker.refresh
        calls = []

        def refresh(event_paths=()):
            calls.append(tuple(event_paths))
            return original_refresh(event_paths)

        monkeypatch.setattr(tracker, "refresh", refresh)

        assert app._changes_inspector is None
        app._start_session_diff_refresh()
        await settle_changes_inspector(app, pilot)

        assert len(calls) == 1
        assert app._changes_inspector is None
        assert {change.path for change in app._session_diff_files} == {"src/a.py"}
