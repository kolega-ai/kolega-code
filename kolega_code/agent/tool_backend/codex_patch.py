"""Parser and in-memory applier for Codex's ``apply_patch`` language.

The grammar mirrors OpenAI Codex's Apache-2.0 ``apply_patch.lark`` definition;
the local parser and filesystem integration are Kolega-specific.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional


CODEX_APPLY_PATCH_GRAMMAR = r"""start: begin_patch hunk+ end_patch
begin_patch: "*** Begin Patch" LF
end_patch: "*** End Patch" LF?
hunk: add_hunk | delete_hunk | update_hunk
add_hunk: "*** Add File: " filename LF add_line+
delete_hunk: "*** Delete File: " filename LF
update_hunk: "*** Update File: " filename LF change_move? change?
filename: /(.+)/
add_line: "+" /(.*)/ LF -> line
change_move: "*** Move to: " filename LF
change: (change_context | change_line)+ eof_line?
change_context: ("@@" | "@@ " /(.+)/) LF
change_line: ("+" | "-" | " ") /(.*)/ LF
eof_line: "*** End of File" LF
%import common.LF"""


PatchKind = Literal["add", "delete", "update"]


@dataclass(frozen=True)
class PatchChunk:
    context: Optional[str]
    lines: tuple[tuple[str, str], ...]
    end_of_file: bool = False


@dataclass(frozen=True)
class PatchOperation:
    kind: PatchKind
    path: str
    move_to: Optional[str] = None
    add_lines: tuple[str, ...] = ()
    chunks: tuple[PatchChunk, ...] = ()


class CodexPatchError(ValueError):
    """The patch is malformed or cannot be applied."""


def parse_codex_patch(raw: str) -> list[PatchOperation]:
    """Parse Codex patch text without requiring Lark at runtime."""
    text = raw.replace("\r\n", "\n").replace("\r", "\n")
    lines = text.splitlines()
    if not lines or lines[0].strip() != "*** Begin Patch":
        raise CodexPatchError("Patch must start with '*** Begin Patch'.")
    if len(lines) < 2 or lines[-1].strip() != "*** End Patch":
        raise CodexPatchError("Patch must end with '*** End Patch'.")

    operations: list[PatchOperation] = []
    index = 1
    end = len(lines) - 1
    while index < end:
        marker = lines[index]
        if marker.startswith("*** Add File: "):
            path = _path_after(marker, "*** Add File: ")
            index += 1
            added: list[str] = []
            while index < end and not lines[index].startswith("*** "):
                line = lines[index]
                if not line.startswith("+"):
                    raise CodexPatchError(f"Add File lines must start with '+': {line!r}")
                added.append(line[1:])
                index += 1
            if not added:
                raise CodexPatchError(f"Add File requires at least one '+' line: {path}")
            operations.append(PatchOperation(kind="add", path=path, add_lines=tuple(added)))
            continue

        if marker.startswith("*** Delete File: "):
            operations.append(PatchOperation(kind="delete", path=_path_after(marker, "*** Delete File: ")))
            index += 1
            continue

        if marker.startswith("*** Update File: "):
            path = _path_after(marker, "*** Update File: ")
            index += 1
            move_to: Optional[str] = None
            if index < end and lines[index].startswith("*** Move to: "):
                move_to = _path_after(lines[index], "*** Move to: ")
                index += 1

            chunks: list[PatchChunk] = []
            context: Optional[str] = None
            change_lines: list[tuple[str, str]] = []
            eof = False

            def finish_chunk() -> None:
                nonlocal context, change_lines, eof
                if context is not None or change_lines or eof:
                    chunks.append(PatchChunk(context=context, lines=tuple(change_lines), end_of_file=eof))
                context = None
                change_lines = []
                eof = False

            while index < end and not _is_operation_marker(lines[index]):
                line = lines[index]
                if line == "@@" or line.startswith("@@ "):
                    if eof:
                        raise CodexPatchError("'*** End of File' must be the final line of an update.")
                    finish_chunk()
                    context = line[3:] if line.startswith("@@ ") else None
                elif line == "*** End of File":
                    if eof:
                        raise CodexPatchError("An update may contain only one '*** End of File' marker.")
                    eof = True
                elif line[:1] in {"+", "-", " "}:
                    if eof:
                        raise CodexPatchError("'*** End of File' must be the final line of an update.")
                    change_lines.append((line[0], line[1:]))
                else:
                    raise CodexPatchError(f"Invalid Update File line: {line!r}")
                index += 1
            finish_chunk()
            if not chunks and move_to is None:
                raise CodexPatchError(f"Update File has no changes: {path}")
            operations.append(PatchOperation(kind="update", path=path, move_to=move_to, chunks=tuple(chunks)))
            continue

        raise CodexPatchError(f"Invalid patch operation marker: {marker!r}")

    if not operations:
        raise CodexPatchError("Patch must contain at least one file operation.")
    return operations


def apply_update_chunks(content: str, chunks: tuple[PatchChunk, ...], path: str) -> str:
    """Apply parsed update chunks, using Codex-compatible fuzzy line matching."""
    dominant = _dominant_ending(content)
    normalized = content.replace("\r\n", "\n").replace("\r", "\n")
    had_final_newline = normalized.endswith("\n")
    lines = normalized.split("\n")
    if had_final_newline:
        lines.pop()

    cursor = 0
    for number, chunk in enumerate(chunks, 1):
        hint = cursor
        if chunk.context is not None:
            context_index = _find_line(lines, chunk.context, cursor)
            if context_index is None:
                raise CodexPatchError(f"Update chunk #{number} context does not match {path}: {chunk.context!r}")
            hint = context_index + 1

        old_lines = [text for prefix, text in chunk.lines if prefix in {" ", "-"}]
        new_lines = [text for prefix, text in chunk.lines if prefix in {" ", "+"}]
        if not old_lines:
            start = len(lines) if chunk.end_of_file else hint
            end = start
        else:
            match = _find_sequence(lines, old_lines, hint, at_end=chunk.end_of_file)
            if match is None:
                preview = "\n".join(old_lines[:5])
                raise CodexPatchError(f"Update chunk #{number} does not match {path}:\n{preview}")
            start, end = match
        lines[start:end] = new_lines
        cursor = start + len(new_lines)

    result = "\n".join(lines)
    if had_final_newline:
        result += "\n"
    if dominant != "\n":
        result = result.replace("\n", dominant)
    return result


def _path_after(line: str, prefix: str) -> str:
    path = line[len(prefix) :].strip()
    if not path:
        raise CodexPatchError(f"Missing path after {prefix.strip()!r}.")
    return path


def _is_operation_marker(line: str) -> bool:
    return line.startswith(("*** Add File: ", "*** Delete File: ", "*** Update File: "))


_SMART_TRANSLATION = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u201a": "'",
        "\u201b": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u201e": '"',
        "\u201f": '"',
    }
)


def _forms(value: str) -> tuple[str, str, str, str]:
    return value, value.rstrip(), value.strip(), value.translate(_SMART_TRANSLATION)


def _find_line(lines: list[str], needle: str, start: int) -> Optional[int]:
    needle_forms = _forms(needle)
    for mode in range(4):
        for index in range(start, len(lines)):
            if _forms(lines[index])[mode] == needle_forms[mode]:
                return index
    return None


def _find_sequence(lines: list[str], needle: list[str], start: int, *, at_end: bool) -> Optional[tuple[int, int]]:
    if len(needle) > len(lines):
        return None
    candidates = [len(lines) - len(needle)] if at_end else range(start, len(lines) - len(needle) + 1)
    for mode in range(4):
        wanted = [_forms(line)[mode] for line in needle]
        for index in candidates:
            if index < start:
                continue
            if [_forms(line)[mode] for line in lines[index : index + len(needle)]] == wanted:
                return index, index + len(needle)
        if at_end:
            candidates = [len(lines) - len(needle)]
    return None


def _dominant_ending(content: str) -> str:
    crlf = content.count("\r\n")
    lf = content.count("\n") - crlf
    cr = content.count("\r") - crlf
    if crlf and crlf >= lf and crlf >= cr:
        return "\r\n"
    if cr and cr > lf:
        return "\r"
    return "\n"
