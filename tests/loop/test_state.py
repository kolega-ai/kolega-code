"""Unit tests for the loop state module.

These tests verify the WorkLog class behavior without requiring
the full kolega-code runtime. The state module only depends on stdlib
and kolega_code.local_state, so it can be tested with minimal setup.
"""

import json
import os
import tempfile
import pytest
from pathlib import Path

# Import the module directly to avoid package-level deps
import importlib.util

_SPEC = importlib.util.spec_from_file_location(
    "state", Path(__file__).parent.parent.parent / "kolega_code" / "loop" / "state.py"
)
state = importlib.util.module_from_spec(_SPEC)


# We need to mock get_state_dir before executing the module.
# Patch it at the module level.
class _FakeLocalState:
    @staticmethod
    def get_state_dir():
        return Path(tempfile.gettempdir()) / "kolega-test-state"


import kolega_code.local_state as _ls

_ls.get_state_dir = _FakeLocalState.get_state_dir


_SPEC.loader.exec_module(state)

WorkLog = state.WorkLog
LoopLimitExceeded = state.LoopLimitExceeded
DEFAULT_TEMPLATE = state.DEFAULT_TEMPLATE


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
        assert path.exists()


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
        wl.inc_attempt()  # 2, at limit
        with pytest.raises(LoopLimitExceeded) as exc:
            wl.inc_attempt()  # 3, exceeds
        assert exc.value.attempts_made == 3
        assert exc.value.max_attempts == 2


class TestAntiPatterns:
    def test_records_and_queries(self, tmp_path):
        path = tmp_path / "work-log.json"
        wl = WorkLog.load(str(path))
        wl.record_anti_pattern(
            "no-zero-check",
            "Division without zero guard",
            "src/math.py",
            42,
            "Always check divisor != 0",
        )
        aps = wl.get_anti_patterns()
        assert len(aps) == 1
        assert aps[0]["pattern"] == "no-zero-check"
        assert aps[0]["occurrence_count"] == 1

    def test_deduplicates_by_pattern(self, tmp_path):
        path = tmp_path / "work-log.json"
        wl = WorkLog.load(str(path))
        wl.record_anti_pattern("dup", "cause", "f.py", 1, "rule")
        wl.record_anti_pattern("dup", "cause2", "f2.py", 2, "rule2")
        aps = wl.get_anti_patterns()
        assert len(aps) == 1
        assert aps[0]["occurrence_count"] == 2

    def test_filters_by_module(self, tmp_path):
        path = tmp_path / "work-log.json"
        wl = WorkLog.load(str(path))
        wl.record_anti_pattern("a", "c", "src/auth.py", 1, "r")
        wl.record_anti_pattern("b", "c", "src/payment.py", 2, "r")
        assert len(wl.get_anti_patterns("auth")) == 1
        assert len(wl.get_anti_patterns("payment")) == 1
        assert len(wl.get_anti_patterns("nonexistent")) == 0


class TestHistory:
    def test_records_attempt_history(self, tmp_path):
        path = tmp_path / "work-log.json"
        wl = WorkLog.load(str(path))
        wl.inc_attempt()
        wl.record_attempt("kept", "Fixed bug", phase="act")
        assert len(wl._data["history"]) == 1
        assert wl._data["history"][0]["status"] == "kept"
        assert wl._data["history"][0]["phase"] == "act"


class TestForTask:
    def test_creates_state_dir_path(self):
        wl = WorkLog.for_task("/home/user/my-project", "fix-auth-bug")
        assert "fix-auth-bug" in str(wl._path)
        assert "work-log.json" in str(wl._path)
        assert "loops" in str(wl._path)
