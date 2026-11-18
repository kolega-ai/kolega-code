"""Parsing and expansion of @path file mentions in CLI prompts."""

from __future__ import annotations

import re
from codecs import getincrementaldecoder
from dataclasses import dataclass
from io import IncrementalNewlineDecoder
from pathlib import Path
from typing import Any, BinaryIO, Dict, List, Tuple

from kolega_code.utils.images import MAX_IMAGE_BYTES, encode_image_attachment, image_media_type

MENTION_RE = re.compile(r'(?:(?<=\s)|^)@(?:"([^"]+)"|(\S+))')

TRAILING_PUNCTUATION = ",.;:!?)]}'\""

MAX_ATTACHMENT_LINES = 2000
MAX_ATTACHMENT_BYTES = 48 * 1024
MAX_DIR_ENTRIES = 200
MAX_MENTIONS_PER_MESSAGE = 20
_BINARY_SNIFF_BYTES = 8192
_LINE_ENDINGS = "\n\v\f\x1c\x1d\x1e\x85\u2028\u2029"


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
            media_type = image_media_type(target.name)
            if media_type is not None:
                raw = target.read_bytes()
                if len(raw) > MAX_IMAGE_BYTES:
                    attachments.append(
                        {
                            "type": "file",
                            "path": rel,
                            "content": "[image too large to attach - over 20MB; ask the agent to read it directly]",
                            "truncated": True,
                            "is_dir": False,
                        }
                    )
                else:
                    attachment = encode_image_attachment(raw, media_type, path=rel)
                    attachment["path"] = rel
                    attachments.append(attachment)
                continue
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
    try:
        with target.open("rb", buffering=0) as handle:
            sample = bytearray()
            while len(sample) < _BINARY_SNIFF_BYTES:
                chunk = handle.read(_BINARY_SNIFF_BYTES - len(sample))
                if not chunk:
                    break
                sample.extend(chunk)
            if b"\x00" in sample:
                return "[binary file - content not attached]", False
            return _read_text_prefix(handle, bytes(sample))
    except OSError as exc:
        return f"[could not read file: {exc}]", False


def _read_text_prefix(handle: BinaryIO, sample: bytes) -> tuple[str, bool]:
    decoder = IncrementalNewlineDecoder(getincrementaldecoder("utf-8")("replace"), True)
    remaining_lines = MAX_ATTACHMENT_LINES
    remaining_bytes = MAX_ATTACHMENT_BYTES
    parts: list[str] = []
    chunk = sample
    eof = len(sample) < _BINARY_SNIFF_BYTES
    while True:
        text = decoder.decode(chunk, final=eof)
        lines = text.splitlines(keepends=True)
        extra_lines = len(lines) > remaining_lines
        if extra_lines:
            text = "".join(lines[:remaining_lines])
        encoded = text.encode("utf-8")
        if len(encoded) > remaining_bytes:
            parts.append(encoded[:remaining_bytes].decode("utf-8", errors="ignore"))
            limit = f"{MAX_ATTACHMENT_BYTES // 1024} KiB"
            break
        parts.append(text)
        if extra_lines:
            limit = f"{MAX_ATTACHMENT_LINES:,}-line"
            break
        if eof:
            return "".join(parts), False

        if lines:
            # A partial last line still occupies the same slot in the next chunk.
            remaining_lines -= len(lines) - (text[-1] not in _LINE_ENDINGS)
        remaining_bytes -= len(encoded)
        size = min(_BINARY_SNIFF_BYTES, remaining_bytes + 1) if remaining_lines else 1
        chunk = handle.read(size)
        eof = not chunk

    parts.append(f"\n[truncated at the {limit} attachment limit - ask the agent to read specific sections for more]")
    return "".join(parts), True


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
