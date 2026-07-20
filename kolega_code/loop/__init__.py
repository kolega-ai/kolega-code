"""Loop state management for autonomous bug-fix and new-code engineering.

Provides:
- WorkLog: persistent attempt tracking, anti-pattern memory, deterministic revert
- LoopStateTools: agent-callable tools (loop_state_init, loop_state_attempt, etc.)
- check_loop_limit: runtime guard for hard attempt-limit enforcement
- Schemas: structured JSON schemas for diagnostic reports, check results, adapt decisions
"""

from kolega_code.loop.state import WorkLog, LoopLimitExceeded
from kolega_code.loop.tools import LoopStateTools
from kolega_code.loop.guard import check_loop_limit
from kolega_code.loop.schemas import (
    DIAGNOSTIC_REPORT_SCHEMA,
    CHECK_RESULT_SCHEMA,
    ADAPT_RESULT_SCHEMA,
)

__all__ = [
    "WorkLog",
    "LoopLimitExceeded",
    "LoopStateTools",
    "check_loop_limit",
    "DIAGNOSTIC_REPORT_SCHEMA",
    "CHECK_RESULT_SCHEMA",
    "ADAPT_RESULT_SCHEMA",
]
