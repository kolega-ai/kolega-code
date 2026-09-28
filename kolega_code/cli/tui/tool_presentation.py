"""Width-aware, literal tool labels and bounded command previews."""

from __future__ import annotations

import ntpath
import re

from rich.cells import cell_len
from rich.console import Console
from rich.markup import escape
from rich.text import Text

from kolega_code.services.lsp import extract_lsp_label
from kolega_code.tool_subjects import build_tool_display

from .. import theme
from ..theme import Color, Glyph
from .state import ConversationEntry, tool_state_presentation


def _head(value: str, width: int) -> str:
    text = Text(value)
    text.truncate(max(0, width), overflow="crop")
    return text.plain.rstrip(" ")


def _tail(value: str, width: int) -> str:
    return _head(value[::-1], width)[::-1]


def _short_filename(value: str, width: int) -> str:
    if width <= 0:
        return ""
    if cell_len(value) <= width:
        return value
    extension = ntpath.splitext(value)[1]
    tail_width = min(width - 1, max(cell_len(extension), (width - 1) // 3))
    return _head(value, width - 1 - tail_width) + "…" + _tail(value, tail_width)


def path_label(path: str, width: int | None = None) -> Text:
    """Keep the filename and nearest parents; elide directories before the name."""
    shown = path
    if width is not None and cell_len(path) > max(0, width):
        if width <= 0:
            return Text()
        separator = "\\" if "\\" in path and "/" not in path else "/"
        parts = path.split(separator)
        filename = parts[-1]
        root = parts[0] + separator
        shown = ""
        # Prefer the root plus as much trailing directory context as fits.
        for start in range(2, len(parts)):
            candidate = root + "…" + separator + separator.join(parts[start:])
            if cell_len(candidate) <= width:
                shown = candidate
                break
        if not shown:
            for start in range(1, len(parts)):
                candidate = "…" + separator + separator.join(parts[start:])
                if cell_len(candidate) <= width:
                    shown = candidate
                    break
        if not shown and cell_len(filename) <= width:
            shown = filename
        if not shown:
            prefix = root + "…" + separator if len(parts) > 1 else ""
            if cell_len(prefix) + 8 > width:
                prefix = "…" + separator if len(parts) > 1 and width >= 10 else ""
            shown = prefix + _short_filename(filename, width - cell_len(prefix))
    split = max(shown.rfind("/"), shown.rfind("\\")) + 1
    text = Text()
    text.append(shown[:split], style="dim")
    text.append(shown[split:], style="bold")
    return text


def edit_previews(preview: dict | None) -> list[dict]:
    if not preview:
        return []
    if preview.get("kind") == "group":
        return [item for item in preview.get("files", []) if isinstance(item, dict)]
    return [preview]


def merge_edit_preview(previous: dict | None, incoming: dict) -> dict:
    """Retain every file of a patch, replacing updates to the same file in place."""
    files = edit_previews(previous)
    for index, item in enumerate(files):
        if item.get("path") == incoming.get("path"):
            files[index] = dict(incoming)
            break
    else:
        files.append(dict(incoming))
    return files[0] if len(files) == 1 else {"kind": "group", "files": files}


def entry_paths(entry: ConversationEntry) -> list[str]:
    paths = entry.tool_display.get("paths")
    previews = edit_previews(entry.edit_preview)
    if isinstance(paths, list) and paths and (len(paths) > 1 or len(previews) <= 1):
        return paths
    if previews:
        return list(dict.fromkeys(str(item.get("path") or "file") for item in previews))
    # Old events lack structured metadata. Only recognized file tools may treat
    # their sanitized subject as a path; never guess from arbitrary output.
    fallback = build_tool_display(
        entry.tool_name or "", {"path": entry.tool_subject, "file_path": entry.tool_subject}
    ).get("paths")
    return fallback if isinstance(fallback, list) else []


def entry_command(entry: ConversationEntry) -> str:
    name = re.split(r"__|[.:/]", (entry.tool_name or "").lower())[-1]
    if name != "exec_command":
        return ""
    command = entry.tool_display.get("command")
    return command if isinstance(command, str) else entry.tool_subject


def tool_title(entry: ConversationEntry, width: int | None = None) -> tuple[Text, bool]:
    """Build a single-row title and say whether the command needs its own preview."""
    state, color = tool_state_presentation(entry.kind)
    header = Text.from_markup(theme.role_header(Glyph.TOOL, escape(entry.tool_name or "tool"), color))
    separator = f" {theme.g(Glyph.BULLET_SEP)} "
    status = Text(separator + state, style="dim")
    suffix = status.copy()
    label = extract_lsp_label(entry.full_content or entry.content)
    if label:
        head, sep, rest = label.partition(" ")
        badge = f"{head} LSP {rest}" if sep else f"LSP {label}"
        badge_color = Color.ERROR if "error" in label else Color.WARNING if "warning" in label else Color.MUTED
        suffix.append(separator + badge, style=badge_color)
    command = entry_command(entry)
    if width is not None:
        width = max(0, width)
        if not width:
            return Text(), bool(command)
        if header.cell_len + suffix.cell_len > width:
            suffix = status
        if suffix.cell_len >= width:
            status = Text(state, style=color)
            status.truncate(width, overflow="ellipsis")
            return status, bool(command)
        header.truncate(width - suffix.cell_len, overflow="ellipsis")
    available = None if width is None else max(0, width - header.cell_len - suffix.cell_len - cell_len(separator))
    separate_command = bool(command) and (
        "\n" in command or "\r" in command or (available is not None and cell_len(command) > available)
    )
    paths = entry_paths(entry)
    if len(paths) == 1:
        subject = path_label(paths[0], available)
    elif len(paths) > 1 or separate_command:
        subject = Text()
    else:
        subject = Text(command or entry.tool_subject, style="dim")
        if available is not None:
            if available:
                subject.truncate(available, overflow="ellipsis")
            else:
                subject = Text()
    if subject:
        header.append(separator, style="dim")
        header.append_text(subject)
    header.append_text(suffix)
    return header, separate_command


def command_preview(command: str, width: int) -> Text:
    """Two visual lines, never a rewritten executable shell command."""
    if width <= 0:
        return Text()
    if width <= 2:
        return Text("…")
    lines = Text(command.expandtabs(4)).wrap(Console(), width - 2, overflow="fold")
    shown = list(lines[:2])
    if len(lines) > 2 and shown:
        shown[-1].truncate(width - 3, overflow="crop")
        shown[-1].append("…")
    result = Text()
    for index, line in enumerate(shown):
        if index:
            result.append("\n")
        result.append("$ " if index == 0 else "  ", style="dim")
        result.append_text(line)
    return result
