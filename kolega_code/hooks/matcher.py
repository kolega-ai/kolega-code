"""HookMatcher: decide whether a hook entry applies to a tool name / source string.

Mirrors the matching conventions used by permission rules and Claude Code:
- ``""`` or ``"*"`` matches everything.
- A pipe list of plain identifiers (``"Edit|Write"``) is an exact-name OR.
- Anything else is treated as a regular expression (full match).
"""

from __future__ import annotations

import re

_SIMPLE_OR = re.compile(r"[A-Za-z0-9_|]+")


class HookMatcher:
    """Matches a hook entry's ``matcher`` against a target string."""

    def __init__(self, pattern: str | None) -> None:
        self.pattern = (pattern or "").strip()

    def matches(self, target: str) -> bool:
        pattern = self.pattern
        if pattern in ("", "*"):
            return True

        target = target or ""
        if _SIMPLE_OR.fullmatch(pattern):
            return target in pattern.split("|")

        try:
            return re.fullmatch(pattern, target) is not None
        except re.error:
            return False

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"HookMatcher({self.pattern!r})"
