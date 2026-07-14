"""Edit-attempt scoring shared by benchmark execution and tests."""

from __future__ import annotations

from pathlib import Path, PurePosixPath
import re
from typing import Iterable

from .models import TaskSpec, ToolAttempt


_PATCH_FILE = re.compile(r"^\*\*\* (?:Add|Update|Delete) File: (.+?)\s*$", re.MULTILINE)
_PATCH_MOVE = re.compile(r"^\*\*\* Move to: (.+?)\s*$", re.MULTILINE)


def changed_file_paths(task: TaskSpec) -> frozenset[str]:
    """Return every file whose expected content or presence differs."""

    paths = set(task.before_files) | set(task.expected_files)
    return frozenset(path for path in paths if task.before_files.get(path) != task.expected_files.get(path))


def _normalize_path(value: object, workspace: Path | None) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    raw = value.strip()
    if workspace is not None:
        root = workspace.resolve()
        candidate = Path(raw)
        resolved = (
            candidate.resolve(strict=False) if candidate.is_absolute() else (root / candidate).resolve(strict=False)
        )
        try:
            return resolved.relative_to(root).as_posix()
        except ValueError:
            return None
    candidate = PurePosixPath(raw)
    if candidate.is_absolute() or ".." in candidate.parts:
        return None
    normalized = candidate.as_posix()
    return None if normalized in {"", "."} else normalized.removeprefix("./")


def attempt_file_paths(attempt: ToolAttempt, workspace: Path | None = None) -> frozenset[str]:
    """Extract files targeted by one edit call, including multi-file patches."""

    raw = attempt.raw_input
    values: list[object] = []
    if isinstance(raw, dict):
        values.extend(raw.get(key) for key in ("path", "file_path", "rename") if raw.get(key) is not None)
        if isinstance(raw.get("input"), str):
            raw = raw["input"]
    if isinstance(raw, str):
        values.extend(_PATCH_FILE.findall(raw))
        values.extend(_PATCH_MOVE.findall(raw))
    return frozenset(path for value in values if (path := _normalize_path(value, workspace)) is not None)


def score_first_attempts_by_file(
    task: TaskSpec,
    attempts: Iterable[ToolAttempt],
    edit_tool_names: set[str] | frozenset[str],
    *,
    workspace: Path | None = None,
) -> tuple[int, int]:
    """Score the earliest edit call for every file the task must change.

    A later recovery cannot erase an earlier failure for the same file. A
    target file that is never named by an edit call also counts as a failure.
    """

    targets = changed_file_paths(task)
    first_outcomes: dict[str, bool] = {}
    for attempt in attempts:
        if attempt.name not in edit_tool_names:
            continue
        outcome = attempt.parse_ok and attempt.apply_ok
        for path in attempt_file_paths(attempt, workspace):
            first_outcomes.setdefault(path, outcome)
    return sum(first_outcomes.get(path, False) for path in targets), len(targets)
