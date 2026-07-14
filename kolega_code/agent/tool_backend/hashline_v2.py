"""Oh My Pi Hashline v2 line anchors and in-memory edit application.

The protocol was introduced by Oh My Pi in February 2026 and is used here
under its MIT license (Copyright (c) 2025 Mario Zechner; 2025-2026 Can
Bölük).  This port intentionally implements the original anchor-based v2
surface, without its optional ``replaceText`` or autocorrection modes.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Iterable, Mapping, Sequence


_ALPHABET = "ZPMQVRWSNKTXJBYH"
_TAG_RE = re.compile(r"^\s*[>+\-]*\s*(\d+)\s*#\s*([ZPMQVRWSNKTXJBYH]{2})")
_DISPLAY_PREFIX_RE = re.compile(r"^\s*(?:>>>|>>)?\s*\d+#[0-9A-Za-z]{1,16}:")
_DIFF_PREFIX_RE = re.compile(r"^[+-](?![+-])")
_MASK32 = 0xFFFFFFFF
_PRIME32_1 = 0x9E3779B1
_PRIME32_2 = 0x85EBCA77
_PRIME32_3 = 0xC2B2AE3D
_PRIME32_4 = 0x27D4EB2F
_PRIME32_5 = 0x165667B1


def _rotl32(value: int, count: int) -> int:
    return ((value << count) | (value >> (32 - count))) & _MASK32


def _round32(accumulator: int, value: int) -> int:
    accumulator = (accumulator + value * _PRIME32_2) & _MASK32
    accumulator = _rotl32(accumulator, 13)
    return (accumulator * _PRIME32_1) & _MASK32


def xxhash32(data: bytes, seed: int = 0) -> int:
    """Return the canonical xxHash32 digest used by ``Bun.hash.xxHash32``."""

    length = len(data)
    offset = 0
    if length >= 16:
        v1 = (seed + _PRIME32_1 + _PRIME32_2) & _MASK32
        v2 = (seed + _PRIME32_2) & _MASK32
        v3 = seed & _MASK32
        v4 = (seed - _PRIME32_1) & _MASK32
        limit = length - 16
        while offset <= limit:
            v1 = _round32(v1, int.from_bytes(data[offset : offset + 4], "little"))
            offset += 4
            v2 = _round32(v2, int.from_bytes(data[offset : offset + 4], "little"))
            offset += 4
            v3 = _round32(v3, int.from_bytes(data[offset : offset + 4], "little"))
            offset += 4
            v4 = _round32(v4, int.from_bytes(data[offset : offset + 4], "little"))
            offset += 4
        digest = (_rotl32(v1, 1) + _rotl32(v2, 7) + _rotl32(v3, 12) + _rotl32(v4, 18)) & _MASK32
    else:
        digest = (seed + _PRIME32_5) & _MASK32

    digest = (digest + length) & _MASK32
    while offset + 4 <= length:
        value = int.from_bytes(data[offset : offset + 4], "little")
        digest = (digest + value * _PRIME32_3) & _MASK32
        digest = (_rotl32(digest, 17) * _PRIME32_4) & _MASK32
        offset += 4
    while offset < length:
        digest = (digest + data[offset] * _PRIME32_5) & _MASK32
        digest = (_rotl32(digest, 11) * _PRIME32_1) & _MASK32
        offset += 1

    digest ^= digest >> 15
    digest = (digest * _PRIME32_2) & _MASK32
    digest ^= digest >> 13
    digest = (digest * _PRIME32_3) & _MASK32
    digest ^= digest >> 16
    return digest & _MASK32


def compute_line_hash(line: str) -> str:
    """Return the original v2 two-character ID for a source line."""

    if line.endswith("\r"):
        line = line[:-1]
    normalized = re.sub(r"\s+", "", line)
    value = xxhash32(normalized.encode("utf-8")) & 0xFF
    return _ALPHABET[value >> 4] + _ALPHABET[value & 0x0F]


def format_line_tag(line_number: int, content: str) -> str:
    if line_number == 1 and content.startswith("\ufeff"):
        content = content[1:]
    return f"{line_number}#{compute_line_hash(content)}"


def format_hash_lines(content: str, start_line: int = 1) -> str:
    """Render complete logical lines as ``LINE#ID:CONTENT``."""

    content = content.replace("\r\n", "\n").replace("\r", "\n")
    rendered: list[str] = []
    for index, line in enumerate(content.split("\n")):
        line_number = start_line + index
        if line_number == 1 and line.startswith("\ufeff"):
            line = line[1:]
        rendered.append(f"{format_line_tag(line_number, line)}:{line}")
    return "\n".join(rendered)


@dataclass(frozen=True)
class LineTag:
    line: int
    hash: str


def parse_tag(value: str) -> LineTag:
    if not isinstance(value, str):
        raise ValueError('Line reference must be a string in "LINE#ID" format.')
    match = _TAG_RE.match(value)
    if match is None:
        raise ValueError(f'Invalid line reference "{value}". Expected format "LINE#ID" (for example "5#HK").')
    line = int(match.group(1))
    if line < 1:
        raise ValueError(f'Line number must be at least 1 in "{value}".')
    return LineTag(line=line, hash=match.group(2))


@dataclass(frozen=True)
class HashlineEdit:
    op: str
    content: tuple[str, ...]
    tag: LineTag | None = None
    first: LineTag | None = None
    last: LineTag | None = None
    after: LineTag | None = None
    before: LineTag | None = None


@dataclass(frozen=True)
class HashMismatch:
    line: int
    expected: str
    actual: str


class HashlineMismatchError(ValueError):
    """Raised with fresh tagged context when one or more anchors are stale."""

    def __init__(self, mismatches: Sequence[HashMismatch], file_lines: Sequence[str]) -> None:
        self.mismatches = tuple(mismatches)
        self.file_lines = tuple(file_lines)
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        by_line = {mismatch.line: mismatch for mismatch in self.mismatches}
        display: set[int] = set()
        for mismatch in self.mismatches:
            display.update(range(max(1, mismatch.line - 2), min(len(self.file_lines), mismatch.line + 2) + 1))

        count = len(self.mismatches)
        subject = "line has" if count == 1 else "lines have"
        output = [
            f"{count} {subject} changed since last read. "
            "Use the updated LINE#ID references shown below (>>> marks changed lines).",
            "",
        ]
        previous = -1
        for line_number in sorted(display):
            if previous != -1 and line_number > previous + 1:
                output.append("    ...")
            previous = line_number
            content = self.file_lines[line_number - 1]
            rendered = f"{format_line_tag(line_number, content)}:{content}"
            output.append((">>> " if line_number in by_line else "    ") + rendered)
        return "\n".join(output)


def _strip_copied_prefixes(lines: list[str]) -> list[str]:
    nonempty = [line for line in lines if line]
    if not nonempty:
        return lines
    hash_count = sum(bool(_DISPLAY_PREFIX_RE.match(line)) for line in nonempty)
    diff_count = sum(bool(_DIFF_PREFIX_RE.match(line)) for line in nonempty)
    strip_hash = hash_count > 0 and hash_count >= len(nonempty) * 0.5
    strip_diff = not strip_hash and diff_count > 0 and diff_count >= len(nonempty) * 0.5
    if strip_hash:
        return [_DISPLAY_PREFIX_RE.sub("", line, count=1) for line in lines]
    if strip_diff:
        return [_DIFF_PREFIX_RE.sub("", line, count=1) for line in lines]
    return lines


def parse_content(value: Any, *, allow_null: bool) -> tuple[str, ...]:
    if value is None:
        if allow_null:
            return ()
        raise ValueError("Insert content cannot be null.")
    if isinstance(value, str):
        lines = _strip_copied_prefixes(value.split("\n"))
        if lines and not lines[-1].strip():
            lines = lines[:-1]
        return tuple(lines)
    if isinstance(value, list) and all(isinstance(line, str) for line in value):
        return tuple(value)
    raise ValueError("Edit content must be a string, an array of strings, or null.")


def _require_keys(edit: Mapping[str, Any], required: Iterable[str], allowed: Iterable[str]) -> None:
    missing = [key for key in required if key not in edit]
    if missing:
        raise ValueError(f"{edit.get('op', 'Edit')} requires: {', '.join(missing)}.")
    unexpected = sorted(set(edit) - set(allowed))
    if unexpected:
        raise ValueError(f"Unexpected fields for {edit.get('op', 'edit')}: {', '.join(unexpected)}.")


def parse_edits(raw_edits: Any) -> list[HashlineEdit]:
    if not isinstance(raw_edits, list):
        raise ValueError("edits must be an array.")
    parsed: list[HashlineEdit] = []
    for raw in raw_edits:
        if not isinstance(raw, Mapping):
            raise ValueError("Every edit must be an object.")
        op = raw.get("op")
        if op == "set":
            _require_keys(raw, ("op", "tag", "content"), ("op", "tag", "content"))
            parsed.append(
                HashlineEdit(op=op, tag=parse_tag(raw["tag"]), content=parse_content(raw["content"], allow_null=True))
            )
        elif op == "replace":
            _require_keys(raw, ("op", "first", "last", "content"), ("op", "first", "last", "content"))
            parsed.append(
                HashlineEdit(
                    op=op,
                    first=parse_tag(raw["first"]),
                    last=parse_tag(raw["last"]),
                    content=parse_content(raw["content"], allow_null=True),
                )
            )
        elif op == "append":
            _require_keys(raw, ("op", "content"), ("op", "after", "content"))
            parsed.append(
                HashlineEdit(
                    op=op,
                    after=parse_tag(raw["after"]) if raw.get("after") is not None else None,
                    content=parse_content(raw["content"], allow_null=False),
                )
            )
        elif op == "prepend":
            _require_keys(raw, ("op", "content"), ("op", "before", "content"))
            parsed.append(
                HashlineEdit(
                    op=op,
                    before=parse_tag(raw["before"]) if raw.get("before") is not None else None,
                    content=parse_content(raw["content"], allow_null=False),
                )
            )
        elif op == "insert":
            _require_keys(raw, ("op", "content"), ("op", "after", "before", "content"))
            after = parse_tag(raw["after"]) if raw.get("after") is not None else None
            before = parse_tag(raw["before"]) if raw.get("before") is not None else None
            if after is None and before is None:
                raise ValueError("insert requires at least one of after or before.")
            mapped_op = "insert" if after is not None and before is not None else ("append" if after else "prepend")
            parsed.append(
                HashlineEdit(
                    op=mapped_op,
                    after=after,
                    before=before,
                    content=parse_content(raw["content"], allow_null=False),
                )
            )
        else:
            raise ValueError(f"Invalid Hashline v2 operation: {op!r}.")
    return parsed


def _validate_tag(tag: LineTag, file_lines: Sequence[str], mismatches: list[HashMismatch]) -> bool:
    if tag.line < 1 or tag.line > len(file_lines):
        raise ValueError(f"Line {tag.line} does not exist (file has {len(file_lines)} lines).")
    actual = compute_line_hash(file_lines[tag.line - 1])
    if actual == tag.hash:
        return True
    mismatches.append(HashMismatch(line=tag.line, expected=tag.hash, actual=actual))
    return False


def apply_hashline_edits(content: str, edits: Sequence[HashlineEdit]) -> str:
    """Validate all anchors, then apply edits bottom-up against one snapshot."""

    if not edits:
        return content
    file_lines = content.split("\n")
    mismatches: list[HashMismatch] = []
    for edit in edits:
        if edit.op == "set":
            assert edit.tag is not None
            _validate_tag(edit.tag, file_lines, mismatches)
        elif edit.op == "replace":
            assert edit.first is not None and edit.last is not None
            if edit.first.line > edit.last.line:
                raise ValueError(
                    f"Range start line {edit.first.line} must be less than or equal to end line {edit.last.line}."
                )
            _validate_tag(edit.first, file_lines, mismatches)
            _validate_tag(edit.last, file_lines, mismatches)
        elif edit.op == "append":
            if not edit.content:
                raise ValueError("Append requires non-empty content.")
            if edit.after is not None:
                _validate_tag(edit.after, file_lines, mismatches)
        elif edit.op == "prepend":
            if not edit.content:
                raise ValueError("Prepend requires non-empty content.")
            if edit.before is not None:
                _validate_tag(edit.before, file_lines, mismatches)
        elif edit.op == "insert":
            assert edit.after is not None and edit.before is not None
            if not edit.content:
                raise ValueError("Insert requires non-empty content.")
            if edit.before.line != edit.after.line + 1:
                raise ValueError(
                    f"insert requires adjacent anchors (after {edit.after.line}, before {edit.before.line})."
                )
            _validate_tag(edit.after, file_lines, mismatches)
            _validate_tag(edit.before, file_lines, mismatches)
    if mismatches:
        unique = list({(item.line, item.expected, item.actual): item for item in mismatches}.values())
        raise HashlineMismatchError(unique, file_lines)

    deduplicated: list[tuple[int, HashlineEdit]] = []
    seen: set[tuple[Any, ...]] = set()
    for index, edit in enumerate(edits):
        if edit.op == "set":
            target = (edit.op, edit.tag.line if edit.tag else None)
        elif edit.op == "replace":
            target = (edit.op, edit.first.line if edit.first else None, edit.last.line if edit.last else None)
        elif edit.op == "append":
            target = (edit.op, edit.after.line if edit.after else "EOF")
        elif edit.op == "prepend":
            target = (edit.op, edit.before.line if edit.before else "BOF")
        else:
            target = (edit.op, edit.after.line if edit.after else None, edit.before.line if edit.before else None)
        key = (*target, edit.content)
        if key not in seen:
            seen.add(key)
            deduplicated.append((index, edit))

    def sort_key(item: tuple[int, HashlineEdit]) -> tuple[int, int, int]:
        index, edit = item
        if edit.op == "set":
            return (-(edit.tag.line if edit.tag else 0), 0, index)
        if edit.op == "replace":
            return (-(edit.last.line if edit.last else 0), 0, index)
        if edit.op == "append":
            return (-(edit.after.line if edit.after else len(file_lines) + 1), 1, index)
        if edit.op == "prepend":
            return (-(edit.before.line if edit.before else 0), 2, index)
        return (-(edit.before.line if edit.before else 0), 3, index)

    for _index, edit in sorted(deduplicated, key=sort_key):
        if edit.op == "set":
            assert edit.tag is not None
            file_lines[edit.tag.line - 1 : edit.tag.line] = edit.content
        elif edit.op == "replace":
            assert edit.first is not None and edit.last is not None
            file_lines[edit.first.line - 1 : edit.last.line] = edit.content
        elif edit.op == "append":
            if edit.after is not None:
                file_lines[edit.after.line : edit.after.line] = edit.content
            elif file_lines == [""]:
                file_lines[:] = edit.content
            else:
                file_lines.extend(edit.content)
        elif edit.op == "prepend":
            if edit.before is not None:
                index = edit.before.line - 1
                file_lines[index:index] = edit.content
            elif file_lines == [""]:
                file_lines[:] = edit.content
            else:
                file_lines[0:0] = edit.content
        else:
            assert edit.before is not None
            index = edit.before.line - 1
            file_lines[index:index] = edit.content

    result = "\n".join(file_lines)
    if result == content:
        raise ValueError("No changes made. The edits produced identical content.")
    return result


def strip_hashline_read_output(text: str, *, search: bool) -> str:
    """Remove v2 anchors from a historical read/search result for another protocol."""

    pattern = re.compile(r"^(\s*)(?:>>>\s+)?(\d+)#[ZPMQVRWSNKTXJBYH]{2}:(.*)$")
    output: list[str] = []
    for line in text.splitlines(keepends=True):
        ending = "\n" if line.endswith("\n") else ""
        body = line[:-1] if ending else line
        match = pattern.match(body)
        if match is None:
            output.append(line)
            continue
        indent, line_number, content = match.groups()
        replacement = f"{indent}Line {line_number}: {content}" if search else f"{indent}{content}"
        output.append(replacement + ending)
    return "".join(output)
