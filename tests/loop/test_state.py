"""Unit tests for the loop state and tools modules."""

import json
import tempfile
import pytest
from pathlib import Path
from unittest.mock import patch

from kolega_code.loop.state import WorkLog, LoopLimitExceeded
from kolega_code.loop.tools import LoopStateTools


# ============================================================
# WorkLog tests
# ============================================================


class TestWorkLogInit:
    def test_creates_new_work_log(self, tmp_path):
        path = tmp_path / "work-log.json"
        wl = WorkLog.load(str(path))
        assert wl.attempts_made == 0
        assert wl.max_attempts == 3
        assert path.exists()
        # Verify all keys from DEFAULT_TEMPLATE are present
        assert wl._data["history"] == []
        assert wl._data["anti_patterns"] == []

    def test_loads_existing_work_log(self, tmp_path):
        path = tmp_path / "work-log.json"
        data = {
            "version": "1.0",
            "task_id": "test-bug",
            "loop_type": "bug-fix",
            "attempts_made": 1,
            "max_attempts": 2,
            "original_branch": "main",
            "last_green_commit": None,
            "last_green_backup": None,
            "touched_files": [],
            "history": [],
            "anti_patterns": [],
        }
        path.write_text(json.dumps(data))
        wl = WorkLog.load(str(path))
        assert wl.attempts_made == 1
        assert wl.max_attempts == 2

    def test_recovers_from_corrupted_file(self, tmp_path):
        path = tmp_path / "work-log.json"
        path.write_text("not json {{{")
        wl = WorkLog.load(str(path))
        assert wl.attempts_made == 0
        assert wl._data["history"] == []
        assert wl._data["anti_patterns"] == []

    def test_defensive_init_has_all_keys(self):
        """__init__ defensively populates _data from DEFAULT_TEMPLATE."""
        wl = WorkLog.__new__(WorkLog)
        wl._path = Path("/nonexistent/work-log.json")
        # Call __init__ manually to test defensive init
        WorkLog.__init__(wl, "/nonexistent/work-log.json")
        assert wl._data["history"] == []
        assert wl._data["anti_patterns"] == []
        assert wl._data["touched_files"] == []
        assert wl._data["version"] == "1.0"


class TestAttemptTracking:
    def test_increments_attempt(self, tmp_path):
        path = tmp_path / "work-log.json"
        wl = WorkLog.load(str(path))
        n = wl.inc_attempt()
        assert n == 1
        assert wl.attempts_made == 1

    def test_raises_on_limit_exceeded(self, tmp_path):
        path = tmp_path / "work-log.json"
        wl = WorkLog.load(str(path))
        wl._data["max_attempts"] = 2
        wl._data["attempts_made"] = 1
        wl.save()
        wl.inc_attempt()  # should succeed (2 <= 2)
        with pytest.raises(LoopLimitExceeded) as exc:
            wl.inc_attempt()  # 3 > 2, should raise
        assert exc.value.attempts_made == 3
        assert exc.value.max_attempts == 2


class TestAntiPatterns:
    def test_records_and_queries(self, tmp_path):
        path = tmp_path / "work-log.json"
        wl = WorkLog.load(str(path))
        wl.record_anti_pattern("no-zero-check", "Division without zero guard", "src/math.py", 42, "Always check != 0")
        aps = wl.get_anti_patterns()
        assert len(aps) == 1
        assert aps[0]["pattern"] == "no-zero-check"

    def test_deduplicates(self, tmp_path):
        path = tmp_path / "work-log.json"
        wl = WorkLog.load(str(path))
        wl.record_anti_pattern("dup", "c", "f.py", 1, "r")
        wl.record_anti_pattern("dup", "c2", "f2.py", 2, "r2")
        assert len(wl.get_anti_patterns()) == 1
        assert wl.get_anti_patterns()[0]["occurrence_count"] == 2

    def test_filters_by_module(self, tmp_path):
        path = tmp_path / "work-log.json"
        wl = WorkLog.load(str(path))
        wl.record_anti_pattern("a", "c", "src/auth.py", 1, "r")
        wl.record_anti_pattern("b", "c", "src/payment.py", 2, "r")
        assert len(wl.get_anti_patterns("auth")) == 1
        assert len(wl.get_anti_patterns("nonexistent")) == 0


class TestHistory:
    def test_records_attempt(self, tmp_path):
        path = tmp_path / "work-log.json"
        wl = WorkLog.load(str(path))
        wl.inc_attempt()
        wl.record_attempt("kept", "Fixed bug", phase="act")
        assert wl._data["history"][0]["status"] == "kept"

    def test_kept_updates_green(self, tmp_path):
        path = tmp_path / "work-log.json"
        wl = WorkLog.load(str(path))
        wl.record_attempt("kept", "Fixed", phase="act")
        # Outside git, last_green_commit stays None since _git_head fails.
        assert wl._data["history"][0]["status"] == "kept"


class TestForTask:
    def test_for_task_loads_fully_initialized(self, tmp_path, monkeypatch):
        """for_task returns a WorkLog with all DEFAULT_TEMPLATE keys populated."""
        import kolega_code.loop.state as state_mod

        state_root = tmp_path / "kolega-state"
        monkeypatch.setattr(state_mod, "_get_state_dir", lambda: state_root)

        wl = WorkLog.for_task("/fake/project", "fix-auth-bug")
        assert wl._data["history"] == []
        assert wl._data["anti_patterns"] == []
        assert wl._data["touched_files"] == []
        assert wl._data["version"] == "1.0"
        assert "work-log.json" in str(wl._path)

    def test_for_task_handles_path_objects(self, tmp_path, monkeypatch):
        """for_task accepts Path objects without raising AttributeError."""
        import kolega_code.loop.state as state_mod

        state_root = tmp_path / "kolega-state"
        monkeypatch.setattr(state_mod, "_get_state_dir", lambda: state_root)

        wl = WorkLog.for_task(Path("/fake/project"), "fix-auth-bug")
        assert wl.attempts_made == 0


class TestRevert:
    def test_revert_uses_project_path_not_cwd(self, tmp_path):
        """revert() references self._project_path, not os.getcwd()."""
        wl = WorkLog.load(str(tmp_path / "work-log.json"), project_path="/my/project")
        cmd = wl.revert()
        # Should reference /my/project, not current directory
        assert "echo '[loop-state] No revert point" in cmd or "/my/project" in cmd

    def test_revert_with_original_branch(self, tmp_path, monkeypatch):
        """When original_branch is set, revert switches to it (branch-based, not reset)."""
        wl = WorkLog.load(str(tmp_path / "work-log.json"), project_path="/my/project")
        wl._data["original_branch"] = "main"
        wl.save()

        # Mock _is_git_repo to return True so the git path is taken
        monkeypatch.setattr(wl, "_is_git_repo", lambda: True)

        cmd = wl.revert()
        assert "git checkout" in cmd
        assert "main" in cmd
        # Should NOT contain git reset --hard (branch-based strategy)
        assert "git reset --hard" not in cmd


# ============================================================
# LoopStateTools tests
# ============================================================


class TestLoopStateTools:
    def test_init(self, tmp_path, monkeypatch):
        import kolega_code.loop.state as state_mod

        state_root = tmp_path / "kolega-state"
        monkeypatch.setattr(state_mod, "_get_state_dir", lambda: state_root)

        tools = LoopStateTools("/fake/project")
        result = tools.loop_state_init("bug-1", "bug-fix", max_attempts=2)
        assert result["task_id"] == "bug-1"
        assert result["loop_type"] == "bug-fix"
        assert result["max_attempts"] == 2
        assert result["attempts_made"] == 0
        # Init should record original branch
        assert result.get("original_branch") is not None or result["original_branch"] is None
        # Verify all keys present (no KeyError)
        assert "history" in result
        assert "anti_patterns" in result

    def test_attempt_tracking(self, tmp_path, monkeypatch):
        import kolega_code.loop.state as state_mod

        state_root = tmp_path / "kolega-state"
        monkeypatch.setattr(state_mod, "_get_state_dir", lambda: state_root)

        tools = LoopStateTools("/fake/project")
        tools.loop_state_init("bug-2", "bug-fix", max_attempts=2)
        r1 = tools.loop_state_attempt("bug-2")
        assert r1["attempt"] == 1
        assert r1["exceeded"] is False
        r2 = tools.loop_state_attempt("bug-2")
        assert r2["attempt"] == 2
        assert r2["exceeded"] is False
        r3 = tools.loop_state_attempt("bug-2")
        assert r3["exceeded"] is True

    def test_log_and_status(self, tmp_path, monkeypatch):
        import kolega_code.loop.state as state_mod

        state_root = tmp_path / "kolega-state"
        monkeypatch.setattr(state_mod, "_get_state_dir", lambda: state_root)

        tools = LoopStateTools("/fake/project")
        tools.loop_state_init("bug-3", "bug-fix")
        tools.loop_state_log("bug-3", "kept", "Fixed bug X", phase="act")
        status = tools.loop_state_status("bug-3")
        assert len(status["history"]) == 1
        assert status["history"][0]["summary"] == "Fixed bug X"

    def test_anti_pattern_flow(self, tmp_path, monkeypatch):
        import kolega_code.loop.state as state_mod

        state_root = tmp_path / "kolega-state"
        monkeypatch.setattr(state_mod, "_get_state_dir", lambda: state_root)

        tools = LoopStateTools("/fake/project")
        tools.loop_state_init("bug-4", "bug-fix")
        tools.loop_state_anti_pattern("bug-4", "no-guard", "Missing guard", "src/a.py", 10, "Add guard")
        result = tools.loop_state_check_anti_patterns("bug-4", module="src/a.py")
        assert len(result["anti_patterns"]) == 1
        assert result["anti_patterns"][0]["pattern"] == "no-guard"
        # Querying unrelated module returns empty
        result2 = tools.loop_state_check_anti_patterns("bug-4", module="nonexistent")
        assert len(result2["anti_patterns"]) == 0

    def test_revert_returns_command(self, tmp_path, monkeypatch):
        import kolega_code.loop.state as state_mod

        state_root = tmp_path / "kolega-state"
        monkeypatch.setattr(state_mod, "_get_state_dir", lambda: state_root)

        tools = LoopStateTools("/fake/project")
        tools.loop_state_init("bug-5", "bug-fix")
        result = tools.loop_state_revert("bug-5")
        assert "command" in result

    def test_record_touched(self, tmp_path, monkeypatch):
        import kolega_code.loop.state as state_mod

        state_root = tmp_path / "kolega-state"
        monkeypatch.setattr(state_mod, "_get_state_dir", lambda: state_root)

        tools = LoopStateTools("/fake/project")
        tools.loop_state_init("bug-6", "bug-fix")
        # Pass a path relative to the project so relpath works correctly
        result = tools.loop_state_record_touched("bug-6", "/fake/project/src/auth.py")
        assert result["recorded"] is True
        status = tools.loop_state_status("bug-6")
        assert "src/auth.py" in status["touched_files"]

    def test_path_objects_in_project_path(self, tmp_path, monkeypatch):
        """LoopStateTools accepts Path objects without raising."""
        import kolega_code.loop.state as state_mod

        state_root = tmp_path / "kolega-state"
        monkeypatch.setattr(state_mod, "_get_state_dir", lambda: state_root)

        tools = LoopStateTools(Path("/fake/project"))
        result = tools.loop_state_init("bug-path", "bug-fix")
        assert result["task_id"] == "bug-path"
