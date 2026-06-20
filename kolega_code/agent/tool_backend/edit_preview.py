"""Pure helpers that build small, capped diff/file-head previews for the UI.

These power the inline edit previews shown in the CLI. They are deliberately free
of any rich/textual imports so the agent layer can build the (serializable) preview
payload that rides the UI-only ``file_edit_preview`` event. The payload is pre-capped
here, so the event stays tiny and never carries whole files into the renderer — and,
crucially, never into model context (the change is already in the tool-call argument).

A preview payload is a plain dict::

    {
        "kind": "diff" | "head",
        "path": str,
        "language": str,            # for syntax highlighting of file-heads
        "lines": [[tag, text], ...] # tag in {"add", "del", "meta", "context"}
        "more": int,                # additional lines beyond the cap
        "adds": int, "dels": int,   # totals, for the meta line
    }

The builders return ``None`` when there is nothing worth showing (no-op edit, binary
content, empty file), in which case no preview event should be sent.
"""

import difflib
import os
from typing import Optional

# Agent-side caps. Kept here (not in cli/theme) so the agent layer needn't import the
# CLI. The CLI applies its own, possibly smaller, display caps on top of these.
MAX_DIFF_LINES = 12
MAX_HEAD_LINES = 10
MAX_LINE_CHARS = 240
MAX_DIFF_INPUT_BYTES = 512 * 1024
MAX_DIFF_INPUT_LINES = 8000

_EXT_LANGUAGE = {
    ".py": "python",
    ".pyi": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".json": "json",
    ".md": "markdown",
    ".html": "html",
    ".css": "css",
    ".scss": "scss",
    ".sh": "bash",
    ".bash": "bash",
    ".zsh": "bash",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
    ".ini": "ini",
    ".cfg": "ini",
    ".sql": "sql",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".rb": "ruby",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".hpp": "cpp",
    ".cs": "csharp",
    ".php": "php",
    ".kt": "kotlin",
    ".swift": "swift",
    ".xml": "xml",
    ".txt": "text",
}


def language_for_path(path: str) -> str:
    """Best-effort lexer name from a file extension; ``"text"`` when unknown."""
    return _EXT_LANGUAGE.get(os.path.splitext(path)[1].lower(), "text")


def _looks_binary(text: str) -> bool:
    return "\x00" in text


def _clip(line: str) -> str:
    line = line.rstrip("\n")
    if len(line) > MAX_LINE_CHARS:
        return line[:MAX_LINE_CHARS] + "…"
    return line


def _too_big(*texts: str) -> bool:
    return any(len(t) > MAX_DIFF_INPUT_BYTES or t.count("\n") > MAX_DIFF_INPUT_LINES for t in texts)


def build_head_preview(content: str, path: str) -> Optional[dict]:
    """First few lines of a newly written file, as a syntax-highlightable head."""
    if not content or _looks_binary(content):
        return None
    lines = content.splitlines()
    if not lines:
        return None
    shown = [["context", _clip(line)] for line in lines[:MAX_HEAD_LINES]]
    return {
        "kind": "head",
        "path": path,
        "language": language_for_path(path),
        "lines": shown,
        "more": max(0, len(lines) - MAX_HEAD_LINES),
        "adds": len(lines),
        "dels": 0,
    }


def build_diff_preview(old: str, new: str, path: str) -> Optional[dict]:
    """A capped unified diff of an edit. ``None`` when nothing changed or is renderable."""
    if old == new:
        return None
    if _looks_binary(old) or _looks_binary(new):
        return None
    if _too_big(old, new):
        # Huge edit: fall back to a head of the new content so something still shows.
        return build_head_preview(new, path)

    rows: list[list[str]] = []
    adds = dels = 0
    for raw in difflib.unified_diff(old.splitlines(), new.splitlines(), lineterm="", n=2):
        if raw.startswith("+++") or raw.startswith("---"):
            continue  # bare a/b headers carry no useful path here; the meta line shows it
        if raw.startswith("@@"):
            tag = "meta"
        elif raw.startswith("+"):
            tag = "add"
            adds += 1
        elif raw.startswith("-"):
            tag = "del"
            dels += 1
        else:
            tag = "context"
        rows.append([tag, _clip(raw)])

    if adds == 0 and dels == 0:
        return None
    return {
        "kind": "diff",
        "path": path,
        "language": language_for_path(path),
        "lines": rows[:MAX_DIFF_LINES],
        "more": max(0, len(rows) - MAX_DIFF_LINES),
        "adds": adds,
        "dels": dels,
    }
