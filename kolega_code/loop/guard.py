"""Runtime guard for loop state enforcement.

Provides a hook that checks work-log.json before each agent turn and
blocks the agent if the attempt limit has been exceeded. This is the
deterministic enforcement equivalent of the loop-state CLI's exit code 2.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


async def check_loop_limit(
    project_path: str,
    task_id: Optional[str] = None,
) -> Optional[dict[str, Any]]:
    """Check if the loop attempt limit has been exceeded.

    Call this before each agent turn when a loop is active. If the
    limit is exceeded, returns a blocking result dict. If the limit
    has not been exceeded or no loop is active, returns None.

    Args:
        project_path: Path to the project directory
        task_id: The current loop task ID, if a loop is active

    Returns:
        None if the agent may proceed, or a dict with:
        - exceeded: True
        - attempts_made: int
        - max_attempts: int
        - revert_command: str (shell command to revert)
        - message: str (human-readable explanation)
    """
    if not task_id:
        return None

    from kolega_code.loop.state import WorkLog

    try:
        wl = WorkLog.for_task(project_path, task_id)
    except Exception as exc:
        logger.debug("loop_limit_guard: could not load work-log: %s", exc)
        return None

    if wl.attempts_made <= wl.max_attempts:
        return None

    # Limit exceeded — block the turn
    revert_cmd = wl.revert()

    return {
        "exceeded": True,
        "attempts_made": wl.attempts_made,
        "max_attempts": wl.max_attempts,
        "revert_command": revert_cmd,
        "message": (
            f"Loop limit exceeded: {wl.attempts_made}/{wl.max_attempts} attempts. "
            f"Reverting to last known-good state and handing back to user."
        ),
    }
