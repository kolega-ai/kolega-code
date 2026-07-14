"""Tool-neutral mechanical edit recipes and deterministic prompt rendering."""

from __future__ import annotations

from collections import defaultdict
import difflib
from hashlib import sha256
import json
import re
from typing import Iterable

from .models import EditOperationSpec, EditRecipeSpec, FileContent


RECIPE_PROMPT_PREAMBLE = """Apply the exact edits below.

The locations and all new code are supplied. Inspect the named files, copy the
new content verbatim, and do not make any other changes. Line numbers refer to
the original files before any edits in this task.
"""


def text_sha256(text: str) -> str:
    return sha256(text.encode("utf-8")).hexdigest()


def _lines(text: str) -> list[str]:
    return text.splitlines(keepends=True)


def _fence(text: str) -> str:
    longest_run = max((len(match.group()) for match in re.finditer(r"`+", text)), default=0)
    marker = "`" * max(3, longest_run + 1)
    suffix = "" if text.endswith("\n") else "\n"
    return f"{marker}\n{text}{suffix}{marker}"


def _anchor(line: str) -> str:
    return line.rstrip("\r\n")


def render_recipe_prompt(recipe: EditRecipeSpec, before_files: dict[str, FileContent]) -> str:
    """Render a recipe without borrowing syntax from any benchmarked edit tool."""
    sections = [RECIPE_PROMPT_PREAMBLE.rstrip()]
    for index, operation in enumerate(recipe.operations, 1):
        before = before_files.get(operation.path)
        lines = _lines(before.text) if before is not None else []
        heading = f"{index}. `{operation.path}`"
        if operation.kind == "create":
            body = f"Create this file with exactly the following contents:\n\n{_fence(operation.new_text)}"
        elif operation.kind == "insert":
            assert operation.start_line is not None
            if operation.start_line == 0:
                location = "Insert at the very beginning of the file"
            else:
                anchor = _anchor(lines[operation.start_line - 1])
                location = (
                    f"Insert immediately after original line {operation.start_line}, "
                    f"whose text is {json.dumps(anchor, ensure_ascii=False)}"
                )
            body = f"{location}, using exactly this content:\n\n{_fence(operation.new_text)}"
        else:
            assert operation.start_line is not None and operation.end_line is not None
            first = _anchor(lines[operation.start_line - 1])
            last = _anchor(lines[operation.end_line - 1])
            location = (
                f"original lines {operation.start_line}–{operation.end_line}; "
                f"the region begins with {json.dumps(first, ensure_ascii=False)} "
                f"and ends with {json.dumps(last, ensure_ascii=False)}"
            )
            if operation.kind == "delete":
                body = f"Delete {location}."
            else:
                body = f"Replace {location} with exactly:\n\n{_fence(operation.new_text)}"
        sections.append(f"{heading}\n\n{body}")
    return "\n\n".join(sections) + "\n"


def _validate_non_overlapping(operations: Iterable[EditOperationSpec]) -> None:
    occupied: list[tuple[int, int, str]] = []
    insertions: set[int] = set()
    for operation in operations:
        assert operation.start_line is not None
        if operation.kind == "insert":
            if operation.start_line in insertions:
                raise ValueError(f"multiple insertions use the same original line in {operation.path}")
            insertions.add(operation.start_line)
            continue
        assert operation.end_line is not None
        for start, end, identifier in occupied:
            if operation.start_line <= end and start <= operation.end_line:
                raise ValueError(f"operations {identifier} and {operation.id} overlap in {operation.path}")
        occupied.append((operation.start_line, operation.end_line, operation.id))
    for point in insertions:
        for start, end, identifier in occupied:
            if start <= point < end:
                raise ValueError(f"insertion after line {point} overlaps {identifier}")


def apply_recipe(before_files: dict[str, FileContent], recipe: EditRecipeSpec) -> dict[str, FileContent]:
    """Apply operations against original line coordinates and return a full tree."""
    result = {path: content.model_copy(deep=True) for path, content in before_files.items()}
    grouped: dict[str, list[EditOperationSpec]] = defaultdict(list)
    for operation in recipe.operations:
        grouped[operation.path].append(operation)

    for path, operations in grouped.items():
        creates = [operation for operation in operations if operation.kind == "create"]
        if creates:
            if len(operations) != 1 or len(creates) != 1:
                raise ValueError(f"created path has additional operations: {path}")
            if path in result:
                raise ValueError(f"create operation targets an existing file: {path}")
            result[path] = FileContent(text=creates[0].new_text)
            continue
        if path not in result:
            raise ValueError(f"edit recipe targets a missing file: {path}")
        _validate_non_overlapping(operations)
        original = _lines(result[path].text)
        updated = list(original)
        for operation in sorted(operations, key=lambda item: (item.start_line or 0, item.id), reverse=True):
            assert operation.start_line is not None
            if operation.kind == "insert":
                if operation.start_line > len(original):
                    raise ValueError(f"insertion line is outside {path}: {operation.start_line}")
                if operation.before_sha256 is not None:
                    anchor = "" if operation.start_line == 0 else original[operation.start_line - 1]
                    if text_sha256(anchor) != operation.before_sha256:
                        raise ValueError(f"insertion anchor hash does not match {path}:{operation.start_line}")
                updated[operation.start_line : operation.start_line] = _lines(operation.new_text)
                continue
            assert operation.end_line is not None
            if operation.end_line > len(original):
                raise ValueError(f"operation range is outside {path}: {operation.start_line}-{operation.end_line}")
            old_text = "".join(original[operation.start_line - 1 : operation.end_line])
            if text_sha256(old_text) != operation.before_sha256:
                raise ValueError(f"before hash does not match {path}:{operation.start_line}-{operation.end_line}")
            replacement = _lines(operation.new_text) if operation.kind == "replace" else []
            updated[operation.start_line - 1 : operation.end_line] = replacement
        result[path] = FileContent(text="".join(updated))
    return result


def derive_recipe(
    before_files: dict[str, FileContent],
    after_files: dict[str, FileContent],
) -> EditRecipeSpec:
    """Derive deterministic original-coordinate operations from an accepted edit."""
    deleted_paths = sorted(set(before_files) - set(after_files))
    if deleted_paths:
        raise ValueError(f"mechanical core tasks cannot delete files: {', '.join(deleted_paths)}")
    operations: list[EditOperationSpec] = []
    sequence = 1
    for path in sorted(set(before_files) | set(after_files)):
        if path not in before_files:
            operations.append(
                EditOperationSpec(id=f"op-{sequence:03d}", kind="create", path=path, new_text=after_files[path].text)
            )
            sequence += 1
            continue
        if before_files[path] == after_files[path]:
            continue
        before_lines = _lines(before_files[path].text)
        after_lines = _lines(after_files[path].text)
        matcher = difflib.SequenceMatcher(a=before_lines, b=after_lines, autojunk=False)
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "equal":
                continue
            new_text = "".join(after_lines[j1:j2])
            if tag == "insert":
                anchor = "" if i1 == 0 else before_lines[i1 - 1]
                operation = EditOperationSpec(
                    id=f"op-{sequence:03d}",
                    kind="insert",
                    path=path,
                    start_line=i1,
                    before_sha256=text_sha256(anchor),
                    new_text=new_text,
                )
            else:
                old_text = "".join(before_lines[i1:i2])
                operation = EditOperationSpec(
                    id=f"op-{sequence:03d}",
                    kind="delete" if tag == "delete" else "replace",
                    path=path,
                    start_line=i1 + 1,
                    end_line=i2,
                    before_sha256=text_sha256(old_text),
                    new_text=new_text,
                )
            operations.append(operation)
            sequence += 1
    if not operations:
        raise ValueError("accepted edit does not change the workspace")
    recipe = EditRecipeSpec(operations=operations)
    reconstructed = apply_recipe(before_files, recipe)
    if reconstructed != after_files:
        raise ValueError("derived recipe does not reconstruct the accepted after tree")
    return recipe


def line_count(text: str) -> int:
    return len(text.splitlines())


def target_length_bucket(lines: int) -> str:
    if lines < 20:
        raise ValueError("mechanical benchmark targets must contain at least 20 lines")
    if lines < 100:
        return "short"
    if lines < 350:
        return "normal"
    if lines < 900:
        return "medium"
    if lines < 2000:
        return "long"
    if lines <= 6000:
        return "oversized"
    raise ValueError("mechanical benchmark targets cannot exceed 6000 lines")


def payload_line_count(recipe: EditRecipeSpec, before_files: dict[str, FileContent]) -> int:
    """Count the larger side of every edit region, including removed source."""
    total = 0
    for operation in recipe.operations:
        new_lines = line_count(operation.new_text)
        old_lines = (
            operation.end_line - operation.start_line + 1
            if operation.start_line is not None and operation.end_line is not None
            else 0
        )
        total += max(1, old_lines, new_lines)
    return total


def payload_size_bucket(lines: int) -> str:
    if lines <= 5:
        return "tiny"
    if lines <= 20:
        return "small"
    if lines <= 75:
        return "medium"
    if lines <= 200:
        return "large"
    return "very-large"
