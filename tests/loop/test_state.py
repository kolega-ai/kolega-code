"""Unit tests for the loop state and tools modules."""

import json
import tempfile
import pytest
from pathlib import Path

# Patch local_state before other imports since WorkLog depends on it
import kolega_code.local_state as _ls


class _FakeLocalState:
    @staticmethod
    def get_state_dir():
        return Path(tempfile.gettempdir()) / "kolega-test-state"


_ls.get_state_dir = _FakeLocalState.get_state_dir  # noqa: E402

from kolega_code.loop.state import WorkLog, LoopLimitExceeded  # noqa: E402
from kolega_code.loop.tools import LoopStateTools  # noqa: E402


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

    def test_loads_existing_work_log(self, tmp_path):
        path = tmp_path / "work-log.json"
        data = {
            "version": "1.0",
            "task_id": "test-bug",
            "loop_type": "bug-fix",
            "attempts_made": 1,
            "max_attempts": 2,
            "last_green_commit": None,
            "last_green_backup": None,
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
        wl.inc_attempt()
        with pytest.raises(LoopLimitExceeded) as exc:
            wl.inc_attempt()
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
        # In a git repo this would set last_green_commit;
        # outside git it stays None.
        assert "kept" == wl._data["history"][0]["status"]


class TestForTask:
    def test_creates_state_dir_path(self):
        wl = WorkLog.for_task("/home/user/my-project", "fix-auth-bug")
        assert "fix-auth-bug" in str(wl._path)
        assert "work-log.json" in str(wl._path)
        assert "loops" in str(wl._path)


# ============================================================
# LoopStateTools tests
# ============================================================


class TestLoopStateTools:
    def test_init(self):
        tools = LoopStateTools("/fake/project")
        result = tools.loop_state_init("bug-1", "bug-fix", max_attempts=2)
        assert result["task_id"] == "bug-1"
        assert result["loop_type"] == "bug-fix"
        assert result["max_attempts"] == 2
        assert result["attempts_made"] == 0

    def test_attempt_tracking(self):
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

    def test_log_and_status(self):
        tools = LoopStateTools("/fake/project")
        tools.loop_state_init("bug-3", "bug-fix")
        tools.loop_state_log("bug-3", "kept", "Fixed bug X", phase="act")
        status = tools.loop_state_status("bug-3")
        assert len(status["history"]) == 1
        assert status["history"][0]["summary"] == "Fixed bug X"

    def test_anti_pattern_flow(self):
        tools = LoopStateTools("/fake/project")
        tools.loop_state_init("bug-4", "bug-fix")
        tools.loop_state_anti_pattern("bug-4", "no-guard", "Missing guard", "src/a.py", 10, "Add guard")
        result = tools.loop_state_check_anti_patterns("bug-4", module="src/a.py")
        assert len(result["anti_patterns"]) == 1
        assert result["anti_patterns"][0]["pattern"] == "no-guard"
        # Querying unrelated module returns empty
        result2 = tools.loop_state_check_anti_patterns("bug-4", module="nonexistent")
        assert len(result2["anti_patterns"]) == 0

    def test_revert_returns_command(self):
        tools = LoopStateTools("/fake/project")
        tools.loop_state_init("bug-5", "bug-fix")
        result = tools.loop_state_revert("bug-5")
        assert "command" in result


# ============================================================
# Guard tests
# ============================================================


class TestGuard:
    def test_no_active_loop_returns_none(self):
        from kolega_code.loop.guard import check_loop_limit
        import asyncio

        result = asyncio.run(check_loop_limit("/fake/project"))
        assert result is None

    def test_exceeded_limit_returns_block(self, tmp_path):
        from kolega_code.loop.guard import check_loop_limit
        import asyncio

        # Create a work-log with exceeded attempts
        wl = WorkLog.load(str(tmp_path / "work-log.json"))
        wl._data["attempts_made"] = 3
        wl._data["max_attempts"] = 2
        wl.save()
        # Mock WorkLog.for_task to return this work-log
        import kolega_code.loop.guard as guard_mod

        original = guard_mod.WorkLog.for_task
        guard_mod.WorkLog.for_task = lambda p, t: wl
        try:
            result = asyncio.run(check_loop_limit("/fake/project", "test-task"))
            assert result is not None
            assert result["exceeded"] is True
            assert result["attempts_made"] == 3
            assert result["max_attempts"] == 2
        finally:
            guard_mod.WorkLog.for_task = original
