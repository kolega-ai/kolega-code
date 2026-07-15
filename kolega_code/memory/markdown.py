"""Owner-private, bounded Markdown project-memory backend."""

from __future__ import annotations

import hashlib
import os
import stat
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Generator

from filelock import FileLock

from kolega_code.local_state import (
    PRIVATE_FILE_MODE,
    ensure_private_dir,
    write_private_bytes,
)

from .models import (
    MEMORY_CONTRACT_VERSION,
    MISSING_REVISION,
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
LOCK_NAME = ".memory.lock"


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
                MemoryCapability.BROWSE,
                MemoryCapability.READ,
                MemoryCapability.APPEND,
                MemoryCapability.REPLACE,
                MemoryCapability.DELETE,
                MemoryCapability.CLEAR,
                MemoryCapability.SEARCH,
            }
        ),
    )

    def __init__(self, storage_dir: Path, settings: Mapping[str, Any] | None = None) -> None:
        self.root = Path(storage_dir)
        self.settings = dict(settings or {})
        self._lock_path = self.root / LOCK_NAME

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
        with self._mutation_lock():
            pass

    def prepare_prompt_context(self) -> MemoryPromptContext:
        entry = self.read_entry(INDEX_REFERENCE)
        guidance = (
            "Record only stable reusable facts explicitly during work. Keep MEMORY.md concise and "
            "link topic files; use read_memory before compare-and-swap replacement/deletion. Correct "
            "stale facts. Never store secrets, guesses, transient progress, plans, transcript "
            "summaries, or duplicate user instructions.\n"
        )
        if not entry.present:
            return MemoryPromptContext(
                "\nMEMORY.md is currently absent or empty.",
                authoring_guidance=guidance,
            )
        content = entry.content or ""
        bounded, lines, truncated = _bound_prompt(content)
        warning = ("Memory index truncated; use read_memory for linked topics.",) if truncated else ()
        body = f"\n### MEMORY.md\n{bounded}"
        if truncated:
            body += "\n\n[Memory index truncated; use read_memory for linked topics.]"
        return MemoryPromptContext(
            body,
            len(bounded.encode()),
            lines,
            truncated=truncated,
            warnings=warning,
            authoring_guidance=guidance,
        )

    def list_entries(self, query: str | None = None) -> list[MemoryEntrySummary]:
        entries: list[MemoryEntrySummary] = []
        lowered = query.casefold() if query else None
        for item in self._scan_markdown_files():
            if (
                lowered
                and lowered not in item.reference.casefold()
                and lowered not in item.data.decode("utf-8").casefold()
            ):
                continue
            entries.append(
                MemoryEntrySummary(
                    item.reference,
                    len(item.data),
                    item.stat_result.st_mtime_ns,
                    _sha(item.data),
                    item.reference,
                )
            )
        return sorted(entries, key=lambda item: (item.reference != INDEX_REFERENCE, item.reference.casefold()))

    def read_entry(self, reference: str) -> MemoryEntry:
        parts, normalized = self._reference(reference)
        opened = self._read_path(parts)
        if opened is None:
            return MemoryEntry(normalized, None, 0, MISSING_REVISION, present=False)
        data, _ = opened
        content = _decode_utf8(data)
        revision = _sha(data)
        return MemoryEntry(normalized, content, len(data), revision)

    def append_entry(self, reference: str, content: str) -> MemoryWriteResult:
        fragment = _encode_candidate(content)
        parts, normalized = self._reference(reference)
        with self._mutation_lock():
            opened = self._read_path(parts)
            previous = opened[0] if opened else b""
            return self._commit(parts, normalized, previous + fragment, existed=opened is not None)

    def replace_entry(self, reference: str, content: str, expected_revision: str) -> MemoryWriteResult:
        candidate = _encode_candidate(content)
        parts, normalized = self._reference(reference)
        with self._mutation_lock():
            opened = self._read_path(parts)
            current = opened[0] if opened else b""
            current_revision = _sha(current) if opened else MISSING_REVISION
            if expected_revision != current_revision:
                return MemoryWriteResult(
                    False,
                    normalized,
                    error="stale revision",
                    current_revision=current_revision,
                    byte_count=len(current) if opened else 0,
                )
            return self._commit(parts, normalized, candidate, existed=opened is not None)

    def delete_entry(self, reference: str, expected_revision: str) -> MemoryWriteResult:
        parts, normalized = self._reference(reference)
        with self._mutation_lock():
            opened = self._read_path(parts)
            current_revision = _sha(opened[0]) if opened else MISSING_REVISION
            if current_revision != expected_revision:
                return MemoryWriteResult(False, normalized, error="stale revision", current_revision=current_revision)
            if opened is None:
                return MemoryWriteResult(False, normalized, error="entry is missing")
            try:
                self.root.joinpath(*parts).unlink()
            except OSError as error:
                raise MemorySafetyError("unable to delete memory entry safely") from error
            return MemoryWriteResult(True, normalized, revision=MISSING_REVISION, byte_count=0)

    def clear(self) -> int:
        with self._mutation_lock():
            files = self._scan_markdown_files(enforce_count=False)
            try:
                for item in files:
                    self.root.joinpath(*PurePosixPath(item.reference).parts).unlink()
            except OSError as error:
                raise MemorySafetyError("unable to clear memory entry safely") from error
            return len(files)

    def tool_bindings(self, scope: MemoryAccessScope) -> tuple[MemoryToolBinding, ...]:
        read = MemoryToolBinding(
            "read_memory",
            {
                "name": "read_memory",
                "description": "Read private project memory.",
                "input_schema": {
                    "type": "object",
                    "properties": {"path": {"type": "string", "default": INDEX_REFERENCE}},
                },
            },
            lambda path=INDEX_REFERENCE: self.read_entry(path),
        )
        if not scope.can_mutate:
            return (read,)
        write = MemoryToolBinding(
            "write_memory",
            {
                "name": "write_memory",
                "description": "Append or compare-and-swap replace private project memory.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "memory_content": {"type": "string"},
                        "path": {"type": "string", "default": INDEX_REFERENCE},
                        "mode": {"enum": ["append", "replace"], "default": "append"},
                        "expected_sha256": {"type": ["string", "null"]},
                    },
                    "required": ["memory_content"],
                },
            },
            self._write_tool,
            True,
        )
        delete = MemoryToolBinding(
            "delete_memory",
            {
                "name": "delete_memory",
                "description": "Delete private memory using its current revision.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "expected_sha256": {"type": "string"},
                    },
                    "required": ["path", "expected_sha256"],
                },
            },
            lambda path, expected_sha256: self.delete_entry(path, expected_sha256),
            True,
        )
        return read, write, delete

    def refresh(self) -> None:
        pass

    def close(self) -> None:
        pass

    def _write_tool(
        self,
        memory_content: str,
        path: str = INDEX_REFERENCE,
        mode: str = "append",
        expected_sha256: str | None = None,
    ) -> MemoryWriteResult:
        if mode == "append":
            return self.append_entry(path, memory_content)
        if mode == "replace":
            if expected_sha256 is None:
                return MemoryWriteResult(False, path, error="replace requires expected_sha256")
            return self.replace_entry(path, memory_content, expected_sha256)
        return MemoryWriteResult(False, path, error="mode must be append or replace")

    def _commit(
        self,
        parts: tuple[str, ...],
        reference: str,
        candidate: bytes,
        *,
        existed: bool,
    ) -> MemoryWriteResult:
        _decode_utf8(candidate)
        if len(candidate) > MAX_FILE_BYTES:
            return MemoryWriteResult(False, reference, error="memory file exceeds 128 KiB")
        files = self._scan_markdown_files()
        if not existed and len(files) >= MAX_FILES:
            return MemoryWriteResult(False, reference, error="memory file count limit reached")
        target = self.root.joinpath(*parts)
        old_size = len(next((item.data for item in files if item.reference == reference), b""))
        total = sum(len(item.data) for item in files) - old_size + len(candidate)
        if total > MAX_TOTAL_BYTES:
            return MemoryWriteResult(False, reference, error="memory total size exceeds 1 MiB")
        self._ensure_parent(parts[:-1])
        self._reject_symlink_or_unsafe_target(target, allow_missing=True)
        try:
            write_private_bytes(target, candidate)
        except OSError as error:
            raise MemorySafetyError("unable to persist memory entry safely") from error
        return MemoryWriteResult(True, reference, _sha(candidate), len(candidate))

    @contextmanager
    def _mutation_lock(self) -> Generator[None, None, None]:
        self._ensure_root(create=True)
        self._reject_symlink_or_unsafe_target(self._lock_path, allow_missing=True)
        lock = FileLock(str(self._lock_path), mode=PRIVATE_FILE_MODE)
        try:
            with lock:
                os.chmod(self._lock_path, PRIVATE_FILE_MODE)
                yield
        except OSError as error:
            raise MemorySafetyError("unable to acquire memory lock safely") from error

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
        if any(part.casefold() in {LOCK_NAME.casefold(), "manifest.json"} for part in pure.parts):
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


def _sha(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


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
            chunks.append(encoded[:remaining].decode("utf-8", errors="ignore"))
            truncated = True
            break
        chunks.append(line)
        used += len(encoded)
    if len(source_lines) > len(chunks):
        truncated = True
    bounded = "".join(chunks)
    return bounded, min(len(source_lines), PROMPT_MAX_LINES), truncated
