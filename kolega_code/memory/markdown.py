"""Owner-private, bounded Markdown project-memory backend."""

from __future__ import annotations

import os
import stat
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from kolega_code.local_state import (
    ensure_private_dir,
    write_private_bytes,
)

from .models import (
    MEMORY_CONTRACT_VERSION,
    MemoryAccessScope,
    MemoryBackendMetadata,
    MemoryBackendStatus,
    MemoryCapability,
    MemoryEntry,
    MemoryEntrySummary,
    MemoryPromptContext,
    MemoryToolBinding,
    MemoryWriteResult,
)

MAX_FILES = 100
MAX_FILE_BYTES = 128 * 1024
MAX_TOTAL_BYTES = 1024 * 1024
MAX_SCAN_ENTRIES = 1000
MAX_SCAN_DIRECTORIES = 100
MAX_DIRECTORY_DEPTH = 32
PROMPT_MAX_LINES = 200
PROMPT_MAX_BYTES = 25 * 1024
INDEX_REFERENCE = "MEMORY.md"


class MemorySafetyError(ValueError):
    pass


@dataclass(frozen=True)
class _ScannedFile:
    reference: str
    data: bytes
    stat_result: os.stat_result


@dataclass
class _ScanBudget:
    entries: int = 0
    directories: int = 1
    total_bytes: int = 0


class MarkdownMemoryBackend:
    metadata = MemoryBackendMetadata(
        backend_id="markdown",
        display_name="Private Markdown",
        contract_version=MEMORY_CONTRACT_VERSION,
        schema_version=1,
        capabilities=frozenset(
            {
                MemoryCapability.PROMPT_CONTEXT,
                MemoryCapability.LIST,
                MemoryCapability.READ,
                MemoryCapability.WRITE,
                MemoryCapability.DELETE,
            }
        ),
    )

    def __init__(self, storage_dir: Path, settings: Mapping[str, Any] | None = None) -> None:
        self.root = Path(storage_dir)
        self.settings = dict(settings or {})

    def status(self) -> MemoryBackendStatus:
        try:
            if not self._root_exists():
                return MemoryBackendStatus(True, False, private_path=str(self.root))
            entries = self.list_entries()
            prompt = self.prepare_prompt_context()
            return MemoryBackendStatus(
                available=True,
                initialized=True,
                entry_count=len(entries),
                total_bytes=sum(item.byte_count for item in entries),
                startup_bytes=prompt.byte_count,
                startup_lines=prompt.line_count,
                startup_truncated=prompt.truncated,
                warnings=prompt.warnings,
                private_path=str(self.root),
            )
        except (OSError, MemorySafetyError) as error:
            return MemoryBackendStatus(False, True, warnings=(str(error),), private_path=str(self.root))

    def initialize(self) -> None:
        """Explicitly initialize private storage; normal mutations do this lazily."""
        self._ensure_root(create=True)

    def prepare_prompt_context(self) -> MemoryPromptContext:
        entry = self.read_entry(INDEX_REFERENCE)
        guidance = (
            "Record stable, reusable facts that will help future work and are not already "
            "authoritative in code or documentation. Good candidates include non-obvious build "
            "or tooling quirks, architectural constraints, recurring failure causes, and "
            "user-confirmed conventions. Before finishing a substantive task, deliberately review "
            "whether any stable, reusable, non-authoritative facts you learned warrant a memory "
            "update. Inspect the already-loaded MEMORY.md first and follow any semantically relevant "
            "topic link with read_memory. If no link is promising, use a targeted list_memory query "
            "before creating a topic. If the fact is already covered, do nothing; rewording is not "
            "a reason to write. Update an existing memory only for materially new, corrected, or "
            "stale information. Keep a short, self-contained fact directly in MEMORY.md; use a flat "
            "topic file only when the memory needs multiple rules, caveats, rationale, or examples. "
            "For a new detailed memory, write the topic first and then add a concise, descriptive "
            "one-line link to MEMORY.md. When deleting a topic, remove its index link before deleting "
            "the topic file. Read existing topic files before overwriting or editing them. Keep "
            "MEMORY.md well under its 200-line prompt budget. Never store secrets, guesses, transient "
            "progress, plans, transcript summaries, or duplicate user instructions.\n"
        )
        if not entry.present:
            return MemoryPromptContext(
                "\nMEMORY.md is currently absent or empty.",
                authoring_guidance=guidance,
            )
        content = entry.content or ""
        bounded, lines, truncated = _bound_prompt(content)
        total_lines = len(content.splitlines())
        note = (
            f"Memory index truncated: showing the first {lines} of {total_lines} lines "
            f"(bounds: {PROMPT_MAX_LINES} lines / {PROMPT_MAX_BYTES // 1024} KiB). "
            "Restructure MEMORY.md into one-line topic links and move details into topic files."
        )
        warning = (note,) if truncated else ()
        body = f"\n### MEMORY.md\n{bounded}"
        if truncated:
            body += f"\n\n[{note}]"
        return MemoryPromptContext(
            body,
            len(bounded.encode()),
            lines,
            truncated=truncated,
            warnings=warning,
            authoring_guidance=guidance,
            recall_guidance=(
                "The MEMORY.md index below is a table of contents, not the full memory. Read any "
                "linked topic relevant to the current task with read_memory before acting on it; "
                "use list_memory to search memory the index does not surface.\n"
            ),
        )

    def list_entries(self, query: str | None = None) -> list[MemoryEntrySummary]:
        entries: list[MemoryEntrySummary] = []
        lowered = query.casefold() if query else None
        for item in self._scan_markdown_files():
            text = item.data.decode("utf-8", errors="ignore")
            if lowered and lowered not in item.reference.casefold() and lowered not in text.casefold():
                continue
            display_name = item.reference
            for line in text.splitlines()[:5]:
                stripped = line.lstrip()
                if stripped.startswith("#"):
                    display_name = stripped.lstrip("#").strip()[:80] or item.reference
                    break
            entries.append(
                MemoryEntrySummary(
                    reference=item.reference,
                    byte_count=len(item.data),
                    modified_ns=item.stat_result.st_mtime_ns,
                    display_name=display_name,
                )
            )
        return sorted(entries, key=lambda item: (item.reference != INDEX_REFERENCE, item.reference.casefold()))

    def read_entry(self, reference: str) -> MemoryEntry:
        parts, normalized = self._reference(reference)
        opened = self._read_path(parts)
        if opened is None:
            return MemoryEntry(normalized, None, 0, present=False)
        data, _ = opened
        content = _decode_utf8(data)
        return MemoryEntry(normalized, content, len(data))

    def write_entry(self, reference: str, content: str) -> MemoryWriteResult:
        candidate = _encode_candidate(content)
        parts, normalized = self._reference(reference)
        return self._commit(parts, normalized, candidate)

    def delete_entry(self, reference: str) -> MemoryWriteResult:
        parts, normalized = self._reference(reference)
        if self._read_path(parts) is None:
            return MemoryWriteResult(False, normalized, error="entry is missing")
        try:
            self.root.joinpath(*parts).unlink()
        except OSError as error:
            raise MemorySafetyError("unable to delete memory entry safely") from error
        return MemoryWriteResult(True, normalized, byte_count=0)

    def tool_bindings(self, scope: MemoryAccessScope) -> tuple[MemoryToolBinding, ...]:
        read = MemoryToolBinding(
            "read_memory",
            {
                "name": "read_memory",
                "description": (
                    "Read a private project-memory file. Use it when the MEMORY.md index in the "
                    "system prompt links a topic relevant to the current task, and before overwriting, "
                    "editing, or deleting an existing topic."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "default": INDEX_REFERENCE,
                            "description": (
                                "Memory file path as shown in MEMORY.md or list_memory, e.g. "
                                "'topics/build.md'. Defaults to the MEMORY.md index."
                            ),
                        }
                    },
                },
            },
            lambda path=INDEX_REFERENCE: self.read_entry(path),
        )
        list_binding = MemoryToolBinding(
            "list_memory",
            {
                "name": "list_memory",
                "description": (
                    "List private project-memory files with sizes and titles. Pass query to filter "
                    "by a case-insensitive substring of path or content. Use it to find relevant "
                    "memory that the MEMORY.md index does not surface, and to review memory before "
                    "reorganizing it."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": (
                                "Optional case-insensitive substring matched against each file's path and content."
                            ),
                        }
                    },
                },
            },
            lambda query=None: self.list_entries(query),
        )
        if not scope.can_mutate:
            return read, list_binding
        write = MemoryToolBinding(
            "write_memory",
            {
                "name": "write_memory",
                "description": (
                    "Create or overwrite one complete private project-memory file. Before creating "
                    "a topic, inspect MEMORY.md and use a targeted list_memory query if needed; do "
                    "nothing when the fact is already covered. Read an existing topic before "
                    "overwriting it. For a new detailed memory, write the topic before adding its "
                    "descriptive one-line link to MEMORY.md."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "content": {
                            "type": "string",
                            "description": "Complete Markdown content for the memory file.",
                        },
                        "path": {
                            "type": "string",
                            "default": INDEX_REFERENCE,
                            "description": (
                                "Memory file path to write, e.g. 'topics/build.md'. Defaults to the MEMORY.md index."
                            ),
                        },
                    },
                    "required": ["content"],
                },
            },
            lambda content, path=INDEX_REFERENCE: self.write_entry(path, content),
            True,
        )
        edit = MemoryToolBinding(
            "edit_memory",
            {
                "name": "edit_memory",
                "description": (
                    "Replace one exact, unique text occurrence in a private project-memory file. "
                    "The edit fails without writing if old_string is empty, missing, or appears "
                    "more than once. MEMORY.md is already loaded in the prompt; read topic files "
                    "before editing them."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "old_string": {
                            "type": "string",
                            "description": "Exact text to replace; it must occur exactly once.",
                        },
                        "new_string": {
                            "type": "string",
                            "description": "Replacement text.",
                        },
                        "path": {
                            "type": "string",
                            "default": INDEX_REFERENCE,
                            "description": (
                                "Memory file path to edit, e.g. 'topics/build.md'. Defaults to the MEMORY.md index."
                            ),
                        },
                    },
                    "required": ["old_string", "new_string"],
                },
            },
            self._edit_tool,
            True,
        )
        delete = MemoryToolBinding(
            "delete_memory",
            {
                "name": "delete_memory",
                "description": (
                    "Delete one private project-memory file by path. Read the topic first. When "
                    "deleting a topic, remove its link from MEMORY.md before deleting the topic file."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Memory file path to delete.",
                        },
                    },
                    "required": ["path"],
                },
            },
            lambda path: self.delete_entry(path),
            True,
        )
        return read, list_binding, write, edit, delete

    def refresh(self) -> None:
        pass

    def close(self) -> None:
        pass

    def _edit_tool(
        self,
        old_string: str,
        new_string: str,
        path: str = INDEX_REFERENCE,
    ) -> MemoryWriteResult:
        entry = self.read_entry(path)
        if not old_string:
            return MemoryWriteResult(False, entry.reference, error="old_string must not be empty")
        content = entry.content or ""
        occurrences = _count_occurrences(content, old_string)
        if occurrences == 0:
            return MemoryWriteResult(False, entry.reference, error="old_string was not found")
        if occurrences > 1:
            return MemoryWriteResult(
                False,
                entry.reference,
                error=f"old_string appears {occurrences} times; provide more context",
            )
        return self.write_entry(entry.reference, content.replace(old_string, new_string, 1))

    def _commit(
        self,
        parts: tuple[str, ...],
        reference: str,
        candidate: bytes,
    ) -> MemoryWriteResult:
        decoded = _decode_utf8(candidate)
        if len(candidate) > MAX_FILE_BYTES:
            return MemoryWriteResult(False, reference, error="memory file exceeds 128 KiB")
        files = self._scan_markdown_files()
        opened = self._read_path(parts)
        existed = opened is not None
        if not existed and len(files) >= MAX_FILES:
            return MemoryWriteResult(False, reference, error="memory file count limit reached")
        target = self.root.joinpath(*parts)
        old_size = len(opened[0]) if opened is not None else 0
        total = sum(len(item.data) for item in files) - old_size + len(candidate)
        if total > MAX_TOTAL_BYTES:
            return MemoryWriteResult(False, reference, error="memory total size exceeds 1 MiB")
        self._ensure_parent(parts[:-1])
        self._reject_symlink_or_unsafe_target(target, allow_missing=True)
        try:
            write_private_bytes(target, candidate)
        except OSError as error:
            raise MemorySafetyError("unable to persist memory entry safely") from error
        line_count = len(decoded.splitlines())
        warnings: tuple[str, ...] = ()
        if reference == INDEX_REFERENCE and (line_count > PROMPT_MAX_LINES or len(candidate) > PROMPT_MAX_BYTES):
            warnings = (
                f"MEMORY.md is now {line_count} lines / {len(candidate):,} bytes; only the first "
                f"{PROMPT_MAX_LINES} lines / {PROMPT_MAX_BYTES // 1024} KiB are injected into prompts. "
                "Move details into topic files and keep the index short.",
            )
        return MemoryWriteResult(True, reference, len(candidate), warnings=warnings)

    def _reference(self, reference: str) -> tuple[tuple[str, ...], str]:
        if not isinstance(reference, str) or not reference or "\0" in reference:
            raise MemorySafetyError("memory path must be a non-empty relative Markdown path")
        if "\\" in reference:
            raise MemorySafetyError("memory path must use normalized '/' separators")
        pure = PurePosixPath(reference)
        if pure.is_absolute() or any(part in ("", ".", "..") for part in pure.parts):
            raise MemorySafetyError("memory path must be normalized and relative")
        normalized = pure.as_posix()
        if normalized != reference or pure.suffix.casefold() != ".md":
            raise MemorySafetyError("memory path must be normalized and end in .md")
        if any(part.casefold() == "manifest.json" for part in pure.parts):
            raise MemorySafetyError("reserved memory path")
        return tuple(pure.parts), normalized

    def _root_exists(self) -> bool:
        if self.root.is_symlink():
            raise MemorySafetyError("memory root must not be a symlink")
        if not self.root.exists():
            return False
        if not self.root.is_dir():
            raise MemorySafetyError("memory root must be a directory")
        return True

    def _ensure_root(self, *, create: bool) -> bool:
        if self._root_exists():
            ensure_private_dir(self.root)
            return True
        if not create:
            return False
        ensure_private_dir(self.root)
        if self.root.is_symlink() or not self.root.is_dir():
            raise MemorySafetyError("memory root must be a private directory")
        return True

    def _ensure_parent(self, parts: tuple[str, ...]) -> None:
        if len(parts) > MAX_DIRECTORY_DEPTH:
            raise MemorySafetyError("memory directory depth exceeds safety limit")
        current = self.root
        for part in parts:
            current /= part
            if current.is_symlink():
                raise MemorySafetyError("symlink below memory root is not allowed")
            if current.exists():
                if not current.is_dir():
                    raise MemorySafetyError("memory path component is not a directory")
                ensure_private_dir(current)
            else:
                ensure_private_dir(current)

    def _read_path(self, parts: tuple[str, ...]) -> tuple[bytes, os.stat_result] | None:
        if not self._ensure_root(create=False):
            return None
        path = self.root
        for part in parts[:-1]:
            path /= part
            if path.is_symlink():
                raise MemorySafetyError("symlink below memory root is not allowed")
            if not path.exists():
                return None
            if not path.is_dir():
                raise MemorySafetyError("memory path component is not a directory")
        target = path / parts[-1]
        if target.is_symlink():
            raise MemorySafetyError("memory target must be a regular non-symlink file")
        if not target.exists():
            return None
        self._reject_symlink_or_unsafe_target(target)
        try:
            info = target.stat()
            if info.st_size > MAX_FILE_BYTES:
                raise MemorySafetyError("memory file exceeds 128 KiB read limit")
            data = target.read_bytes()
        except MemorySafetyError:
            raise
        except OSError as error:
            raise MemorySafetyError("unable to read memory entry safely") from error
        return self._validate_file(data, info), info

    def _reject_symlink_or_unsafe_target(self, path: Path, *, allow_missing: bool = False) -> None:
        if path.is_symlink():
            raise MemorySafetyError("memory target must be a regular non-symlink file")
        if not path.exists():
            if allow_missing:
                return
            raise MemorySafetyError("memory target is missing")
        try:
            mode = path.stat(follow_symlinks=False).st_mode
        except OSError as error:
            raise MemorySafetyError("unable to inspect memory entry safely") from error
        if not stat.S_ISREG(mode):
            raise MemorySafetyError("memory target must be a regular non-symlink file")

    def _validate_file(self, data: bytes, info: os.stat_result) -> bytes:
        if not stat.S_ISREG(info.st_mode):
            raise MemorySafetyError("memory target must be a regular non-symlink file")
        if info.st_size > MAX_FILE_BYTES or len(data) > MAX_FILE_BYTES:
            raise MemorySafetyError("memory file exceeds 128 KiB read limit")
        _decode_utf8(data)
        return data

    def _scan_markdown_files(self, *, enforce_count: bool = True) -> list[_ScannedFile]:
        if not self._ensure_root(create=False):
            return []
        result: list[_ScannedFile] = []
        self._scan_directory(self.root, (), result, _ScanBudget(), enforce_limits=enforce_count)
        return result

    def _scan_directory(
        self,
        directory: Path,
        prefix: tuple[str, ...],
        result: list[_ScannedFile],
        budget: _ScanBudget,
        *,
        enforce_limits: bool,
    ) -> None:
        if len(prefix) > MAX_DIRECTORY_DEPTH:
            raise MemorySafetyError("memory directory depth exceeds safety limit")
        try:
            with os.scandir(directory) as entries:
                for entry in entries:
                    budget.entries += 1
                    if budget.entries > MAX_SCAN_ENTRIES:
                        raise MemorySafetyError("memory directory entry count exceeds scan safety limit")
                    if entry.is_symlink():
                        raise MemorySafetyError("symlink below memory root is not allowed")
                    if entry.is_dir(follow_symlinks=False):
                        budget.directories += 1
                        if budget.directories > MAX_SCAN_DIRECTORIES:
                            raise MemorySafetyError("memory directory count exceeds scan safety limit")
                        self._scan_directory(
                            Path(entry.path),
                            (*prefix, entry.name),
                            result,
                            budget,
                            enforce_limits=enforce_limits,
                        )
                    elif entry.name.casefold().endswith(".md"):
                        self._scan_file(entry, prefix, result, budget, enforce_limits=enforce_limits)
        except MemorySafetyError:
            raise
        except OSError as error:
            raise MemorySafetyError("unable to list memory directory safely") from error

    def _scan_file(
        self,
        entry: os.DirEntry[str],
        prefix: tuple[str, ...],
        result: list[_ScannedFile],
        budget: _ScanBudget,
        *,
        enforce_limits: bool,
    ) -> None:
        if enforce_limits and len(result) >= MAX_FILES:
            raise MemorySafetyError("memory file count exceeds safety limit")
        try:
            info = entry.stat(follow_symlinks=False)
            if not stat.S_ISREG(info.st_mode):
                raise MemorySafetyError("memory target must be a regular non-symlink file")
            if info.st_size > MAX_FILE_BYTES:
                raise MemorySafetyError("memory file exceeds 128 KiB read limit")
            data = Path(entry.path).read_bytes()
        except MemorySafetyError:
            raise
        except OSError as error:
            raise MemorySafetyError("unable to read memory entry safely") from error
        data = self._validate_file(data, info)
        if enforce_limits and budget.total_bytes + len(data) > MAX_TOTAL_BYTES:
            raise MemorySafetyError("memory total size exceeds safety limit")
        if enforce_limits:
            budget.total_bytes += len(data)
        result.append(_ScannedFile("/".join((*prefix, entry.name)), data, info))


def _encode_candidate(content: Any) -> bytes:
    if not isinstance(content, str):
        raise TypeError("memory content must be text")
    return content.encode("utf-8")


def _decode_utf8(content: bytes) -> str:
    try:
        return content.decode("utf-8")
    except UnicodeDecodeError as error:
        raise MemorySafetyError("memory file is not valid UTF-8") from error


def _count_occurrences(content: str, old_string: str) -> int:
    """Count exact matches, including matches that overlap."""
    count = 0
    start = 0
    while (match := content.find(old_string, start)) != -1:
        count += 1
        start = match + 1
    return count


def _bound_prompt(content: str) -> tuple[str, int, bool]:
    chunks: list[str] = []
    used = 0
    truncated = False
    source_lines = content.splitlines(keepends=True)
    for index, line in enumerate(source_lines):
        if index >= PROMPT_MAX_LINES:
            truncated = True
            break
        encoded = line.encode()
        remaining = PROMPT_MAX_BYTES - used
        if len(encoded) > remaining:
            partial = encoded[:remaining].decode("utf-8", errors="ignore")
            if partial:
                chunks.append(partial)
            truncated = True
            break
        chunks.append(line)
        used += len(encoded)
    if len(source_lines) > len(chunks):
        truncated = True
    bounded = "".join(chunks)
    return bounded, len(chunks), truncated
