"""Parsing and expansion of @path file mentions in CLI prompts."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

MENTION_RE = re.compile(r'(?:(?<=\s)|^)@(?:"([^"]+)"|(\S+))')

TRAILING_PUNCTUATION = ",.;:!?)]}'\""

MAX_ATTACHMENT_LINES = 2000
MAX_ATTACHMENT_BYTES = 48 * 1024
MAX_DIR_ENTRIES = 200
MAX_MENTIONS_PER_MESSAGE = 20


@dataclass(frozen=True)
class Mention:
    raw: str  # token text as typed, without the leading @
    path: str  # candidate path with trailing punctuation stripped


def parse_mentions(text: str) -> List[Mention]:
    """Extract candidate @path tokens; purely syntactic, no filesystem checks."""
    mentions: List[Mention] = []
    for match in MENTION_RE.finditer(text):
        quoted, bare = match.groups()
        if quoted is not None:
            mentions.append(Mention(raw=quoted, path=quoted))
            continue
        path = bare.rstrip(TRAILING_PUNCTUATION)
        if path:
            mentions.append(Mention(raw=bare, path=path))
    return mentions


def build_file_attachments(text: str, project_path: Path) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Resolve @ mentions in ``text`` against ``project_path``.

    Returns (attachments, unresolved) where attachments are
    ``{"type": "file", "path": rel, "content": str, "truncated": bool, "is_dir": bool}``
    payloads and unresolved lists mention tokens that did not match an existing
    path (left in the message as literal text).
    """
    project_path = Path(project_path).resolve()
    attachments: List[Dict[str, Any]] = []
    unresolved: List[str] = []
    seen: set[str] = set()

    for mention in parse_mentions(text)[:MAX_MENTIONS_PER_MESSAGE]:
        rel = _resolve_relative(mention.path, project_path)
        if rel is None:
            unresolved.append(mention.path)
            continue
        if rel in seen:
            continue
        seen.add(rel)

        target = project_path / rel
        if target.is_dir():
            attachments.append(
                {
                    "type": "file",
                    "path": rel,
                    "content": _directory_listing(target),
                    "truncated": False,
                    "is_dir": True,
                }
            )
        else:
            content, truncated = _read_file_content(target)
            attachments.append(
                {
                    "type": "file",
                    "path": rel,
                    "content": content,
                    "truncated": truncated,
                    "is_dir": False,
                }
            )

    return attachments, unresolved


def _resolve_relative(candidate: str, project_path: Path) -> str | None:
    """Map a mention to a project-relative posix path, or None if invalid."""
    raw = Path(candidate.rstrip("/")) if candidate not in ("", "/") else Path(".")
    try:
        if raw.is_absolute():
            rel = raw.resolve().relative_to(project_path)
        else:
            rel = (project_path / raw).resolve().relative_to(project_path)
    except (ValueError, OSError):
        return None
    target = project_path / rel
    if not target.exists():
        return None
    return rel.as_posix()


def _read_file_content(target: Path) -> Tuple[str, bool]:
    if _looks_binary(target):
        return "[binary file - content not attached]", False

    try:
        raw = target.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return f"[could not read file: {exc}]", False

    lines = raw.splitlines(keepends=True)
    total_lines = len(lines)
    truncated = False

    if total_lines > MAX_ATTACHMENT_LINES:
        lines = lines[:MAX_ATTACHMENT_LINES]
        truncated = True
    content = "".join(lines)
    if len(content.encode("utf-8", errors="replace")) > MAX_ATTACHMENT_BYTES:
        content = content.encode("utf-8", errors="replace")[:MAX_ATTACHMENT_BYTES].decode("utf-8", errors="replace")
        truncated = True

    if truncated:
        shown_lines = content.count("\n") + (0 if content.endswith("\n") or not content else 1)
        content += (
            f"\n[truncated: showing first {shown_lines} of {total_lines} lines"
            " - ask the agent to read specific sections for more]"
        )
    return content, truncated


def _looks_binary(target: Path) -> bool:
    try:
        with target.open("rb") as handle:
            chunk = handle.read(8192)
    except OSError:
        return False
    return b"\x00" in chunk


def _directory_listing(target: Path) -> str:
    try:
        children = sorted(target.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
    except OSError as exc:
        return f"[could not list directory: {exc}]"

    names = []
    for child in children[:MAX_DIR_ENTRIES]:
        names.append(child.name + "/" if child.is_dir() else child.name)
    listing = "\n".join(names) if names else "[empty directory]"
    if len(children) > MAX_DIR_ENTRIES:
        listing += f"\n[truncated: showing first {MAX_DIR_ENTRIES} of {len(children)} entries]"
    return listing
