import threading
import time
from pathlib import Path

import pytest

from kolega_code.cli import app as cli_app_module
from kolega_code.events import AgentEvent

from ._app_test_utils import _build_sub_agent_test_app, settle_changes_inspector, wait_for_session_diff_baseline
from .test_app_changes_inspector import _file_edit_preview_event, _init_git_project


pytestmark = pytest.mark.usefixtures("hermetic_git_config")


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

        def refresh(event_paths=(), *, checkpoint_id=None):
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

        def refresh(event_paths=(), *, checkpoint_id=None):
            calls.append(tuple(event_paths))
            return original_refresh(event_paths, checkpoint_id=checkpoint_id)

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

        def refresh(event_paths=(), *, checkpoint_id=None):
            calls.append(tuple(event_paths))
            if len(calls) == 1:
                started.set()
                if not release.wait(timeout=5.0):
                    raise AssertionError("Timed out waiting to release refresh")
            return original_refresh(event_paths, checkpoint_id=checkpoint_id)

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

        def refresh(event_paths=(), *, checkpoint_id=None):
            calls.append(tuple(event_paths))
            return original_refresh(event_paths, checkpoint_id=checkpoint_id)

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

        def refresh(event_paths=(), *, checkpoint_id=None):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise RuntimeError("refresh failed")
            return original_refresh(event_paths, checkpoint_id=checkpoint_id)

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
async def test_diff_scope_is_resolved_off_the_ui_thread(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """scope() shells out to git, so it must never run on the event loop."""
    app = _build_sub_agent_test_app(tmp_path, monkeypatch)
    _init_git_project(app.project_path)
    ui_thread = threading.get_ident()

    async with app.run_test() as pilot:
        tracker = app._session_diff_tracker
        assert tracker is not None
        original_scope = tracker.scope
        scope_threads: list[int] = []

        def scope(*, checkpoint_id=None):
            scope_threads.append(threading.get_ident())
            return original_scope(checkpoint_id=checkpoint_id)

        monkeypatch.setattr(tracker, "scope", scope)

        (app.project_path / "src" / "a.py").write_text("new a\n", encoding="utf-8")
        app.action_open_changes()
        await settle_changes_inspector(app, pilot)

        assert scope_threads
        assert ui_thread not in scope_threads
        assert app._session_diff_scope is not None
        assert app._session_diff_scope.branch == "main"


@pytest.mark.asyncio
async def test_startup_scope_probe_populates_scope_without_opening_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = _build_sub_agent_test_app(tmp_path, monkeypatch)
    _init_git_project(app.project_path)

    async with app.run_test() as pilot:
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and app._session_diff_scope is None:
            await pilot.pause(0.01)

        assert app._changes_inspector is None
        assert app._session_diff_scope is not None
        assert app._session_diff_scope.branch == "main"
        assert app._session_diff_scope.linked_worktree is False


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

        def refresh(event_paths=(), *, checkpoint_id=None):
            calls.append(tuple(event_paths))
            return original_refresh(event_paths, checkpoint_id=checkpoint_id)

        monkeypatch.setattr(tracker, "refresh", refresh)

        assert app._changes_inspector is None
        app._start_session_diff_refresh()
        await settle_changes_inspector(app, pilot)

        assert len(calls) == 1
        assert app._changes_inspector is None
        assert {change.path for change in app._session_diff_files} == {"src/a.py"}


@pytest.mark.asyncio
async def test_checkpoint_capture_runs_under_session_diff_lock(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    app = _build_sub_agent_test_app(tmp_path, monkeypatch)
    _init_git_project(app.project_path)

    async with app.run_test():
        await wait_for_session_diff_baseline(app)
        tracker = app._session_diff_tracker
        assert tracker is not None
        original = tracker.capture_checkpoint
        lock_held: list[bool] = []

        def capture(label):
            lock_held.append(app._session_diff_lock.locked())
            return original(label)

        monkeypatch.setattr(tracker, "capture_checkpoint", capture)

        await app._process_message("hello")

        assert lock_held == [True]


@pytest.mark.asyncio
async def test_startup_baseline_runs_off_ui_thread(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The baseline reads every dirty file, so it must never run on the event loop."""
    from kolega_code.cli.tui.session_diff import GitSessionDiffTracker

    ui_thread = threading.get_ident()
    baseline_threads: list[int] = []
    original = GitSessionDiffTracker.capture_baseline

    def capture_baseline(self, label=""):
        baseline_threads.append(threading.get_ident())
        return original(self, label)

    monkeypatch.setattr(GitSessionDiffTracker, "capture_baseline", capture_baseline)
    app = _build_sub_agent_test_app(tmp_path, monkeypatch)
    _init_git_project(app.project_path)

    async with app.run_test() as pilot:
        # The scope probe only starts after the baseline worker succeeds, so a
        # populated scope proves the ordering as well as the thread.
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and app._session_diff_scope is None:
            await pilot.pause(0.01)

        assert baseline_threads
        assert ui_thread not in baseline_threads
        assert app._session_diff_scope is not None


@pytest.mark.asyncio
async def test_checkpoint_skipped_on_unbaselined_tracker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A turn racing the startup baseline must not fabricate checkpoint 0."""
    from kolega_code.cli.tui.session_diff import GitSessionDiffTracker

    app = _build_sub_agent_test_app(tmp_path, monkeypatch)
    _init_git_project(app.project_path)

    async with app.run_test():
        fresh = GitSessionDiffTracker.create(app.project_path)
        assert fresh is not None
        original = fresh.capture_checkpoint
        captures: list[str] = []

        def capture(label):
            captures.append(label)
            return original(label)

        monkeypatch.setattr(fresh, "capture_checkpoint", capture)
        app._session_diff_tracker = fresh

        await app._record_turn_checkpoint("racing turn")
        assert captures == []
        assert fresh.checkpoints() == []

        # Once the baseline lands, the same path captures normally again.
        fresh.capture_baseline()
        await app._record_turn_checkpoint("later turn")
        assert len(captures) == 2  # capture_baseline() itself + the turn checkpoint
        assert [checkpoint.label for checkpoint in fresh.checkpoints()] == ["", "later turn"]
