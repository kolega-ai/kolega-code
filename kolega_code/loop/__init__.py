"""Loop state management for autonomous bug-fix engineering.

Provides:
- WorkLog: persistent attempt tracking, anti-pattern memory, deterministic revert
- Schemas: structured JSON schemas for diagnostic reports, check results, adapt decisions
"""

from kolega_code.loop.state import WorkLog, LoopLimitExceeded
from kolega_code.loop.schemas import (
    DIAGNOSTIC_REPORT_SCHEMA,
    CHECK_RESULT_SCHEMA,
    ADAPT_RESULT_SCHEMA,
)

__all__ = [
    "WorkLog",
    "LoopLimitExceeded",
    "DIAGNOSTIC_REPORT_SCHEMA",
    "CHECK_RESULT_SCHEMA",
    "ADAPT_RESULT_SCHEMA",
]
