"""Agent-callable loop state tools.

These tools manage deterministic loop state (attempt tracking, anti-pattern
memory, revert points) for the autonomous bug-fix loop.

The agent calls these to:
- Initialize a loop: loop_state_init("my-bug", "bug-fix", max_attempts=2)
- Track attempts: loop_state_attempt() → {"exceeded": true} when limit hit
- Revert on failure: loop_state_revert() → shell command string
- Record history: loop_state_log("kept", "Fixed bug X")
- Record lessons: loop_state_anti_pattern("pattern", "cause", ...)
- Query past failures: loop_state_check_anti_patterns("module.py")
"""

from __future__ import annotations

from typing import Any, Optional

from kolega_code.loop.state import WorkLog, LoopLimitExceeded


class LoopStateTools:
    """Agent-callable tools for deterministic loop state management.

    These methods are auto-discovered by ToolExtension and exposed as
    tools the coder agent can call during a bug-fix loop.
    """

    def __init__(self, project_path: str):
        self._project_path = project_path
        self._worklog: Optional[WorkLog] = None

    def _get_worklog(self, task_id: str) -> WorkLog:
        """Get or create the WorkLog for a task."""
        if self._worklog is None:
            self._worklog = WorkLog.for_task(self._project_path, task_id)
        return self._worklog

    # ------------------------------------------------------------------
    # Agent-callable tools
    # ------------------------------------------------------------------

    def loop_state_init(
        self,
        task_id: str,
        loop_type: str,
        max_attempts: int = 2,
    ) -> dict[str, Any]:
        """Initialize loop state for a new task.

        Call this at the start of each bug-fix loop. Creates a new work-log
        with the given parameters.

        Args:
            task_id: Short identifier for the bug (e.g., 'div-by-zero')
            loop_type: Type of loop ('bug-fix' or 'new-code')
            max_attempts: Maximum fix attempts (default 2)

        Returns:
            dict with task_id, loop_type, max_attempts, attempts_made
        """
        wl = self._get_worklog(task_id)
        wl._data["task_id"] = task_id
        wl._data["loop_type"] = loop_type
        wl._data["max_attempts"] = max_attempts
        wl._data["attempts_made"] = 0
        wl.save()
        return wl.to_dict()

    def loop_state_attempt(self, task_id: str) -> dict[str, Any]:
        """Increment the attempt counter. Call before each fix attempt.

        If the limit is exceeded, returns {"exceeded": true}. You MUST
        stop the bug-fix loop when exceeded is true. The runtime may also
        enforce this automatically.

        Args:
            task_id: The loop task identifier

        Returns:
            dict with attempt, max, exceeded
        """
        wl = self._get_worklog(task_id)
        try:
            n = wl.inc_attempt()
            return {"attempt": n, "max": wl.max_attempts, "exceeded": False}
        except LoopLimitExceeded as e:
            return {"attempt": e.attempts_made, "max": e.max_attempts, "exceeded": True}

    def loop_state_revert(self, task_id: str) -> dict[str, str]:
        """Return the shell command to revert to last known-good state.

        Pipe the returned command to bash:
            result = loop_state_revert("my-bug")
            exec_command(result["command"])

        Args:
            task_id: The loop task identifier

        Returns:
            dict with "command" key containing the shell command
        """
        wl = self._get_worklog(task_id)
        cmd = wl.revert()
        return {"command": cmd}

    def loop_state_log(
        self,
        task_id: str,
        status: str,
        summary: str,
        phase: str = "",
    ) -> dict[str, Any]:
        """Record an attempt in the loop history.

        Args:
            task_id: The loop task identifier
            status: "kept" if the fix was kept, "reverted" if discarded
            summary: Brief description of what happened
            phase: The loop phase (e.g., "act", "adapt")

        Returns:
            dict with recorded status
        """
        wl = self._get_worklog(task_id)
        wl.record_attempt(status, summary, phase=phase)
        return {"recorded": True, "status": status}

    def loop_state_anti_pattern(
        self,
        task_id: str,
        pattern: str,
        root_cause: str,
        file: str,
        line: int,
        prevention_rule: str,
    ) -> dict[str, Any]:
        """Record an anti-pattern to avoid in future loops.

        Args:
            task_id: The loop task identifier
            pattern: Short name for the anti-pattern (e.g., 'no-zero-check')
            root_cause: Why the bug existed
            file: File where the bug was found
            line: Line number
            prevention_rule: Rule to prevent this class of bug

        Returns:
            dict with recorded status
        """
        wl = self._get_worklog(task_id)
        wl.record_anti_pattern(pattern, root_cause, file, line, prevention_rule)
        return {"recorded": True, "pattern": pattern}

    def loop_state_check_anti_patterns(
        self,
        task_id: str,
        module: str = "",
    ) -> dict[str, Any]:
        """Query past anti-patterns for a module.

        Args:
            task_id: The loop task identifier
            module: Optional module/file name to filter by (substring match)

        Returns:
            dict with "anti_patterns" list
        """
        wl = self._get_worklog(task_id)
        aps = wl.get_anti_patterns(for_module=module if module else None)
        return {"anti_patterns": aps}

    def loop_state_status(self, task_id: str) -> dict[str, Any]:
        """Return the full work-log state.

        Args:
            task_id: The loop task identifier

        Returns:
            Full work-log dict with version, task_id, attempts, history, anti_patterns
        """
        wl = self._get_worklog(task_id)
        return wl.to_dict()

    def loop_state_backup(self, task_id: str) -> dict[str, str]:
        """Snapshot the current working tree for later revert.

        Records the current git HEAD (or a filesystem backup if not a git
        repo) as the revert point.

        Args:
            task_id: The loop task identifier

        Returns:
            dict with "backup" key containing the commit hash or backup path
        """
        wl = self._get_worklog(task_id)
        result = wl.backup_current()
        return {"backup": result}
