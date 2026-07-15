"""Versioned, backend-neutral project-memory models."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Callable, Mapping

MEMORY_CONTRACT_VERSION = 1
MISSING_REVISION = "missing"


class MemoryCapability(StrEnum):
    PROMPT_CONTEXT = "prompt_context"
    BROWSE = "browse"
    READ = "read"
    APPEND = "append"
    REPLACE = "replace"
    DELETE = "delete"
    CLEAR = "clear"
    SEARCH = "search"
    RICH_RECALL = "rich_recall"
    AUTOMATIC_RETENTION = "automatic_retention"


class MemoryAccessScope(StrEnum):
    TOP_LEVEL = "top_level"
    SUBAGENT = "subagent"

    @property
    def can_mutate(self) -> bool:
        return self is MemoryAccessScope.TOP_LEVEL


@dataclass(frozen=True, slots=True)
class MemoryBackendMetadata:
    backend_id: str
    display_name: str
    contract_version: int
    schema_version: int
    capabilities: frozenset[MemoryCapability]


@dataclass(frozen=True, slots=True)
class MemoryEntrySummary:
    reference: str
    byte_count: int
    modified_ns: int | None = None
    revision: str | None = None
    display_name: str | None = None
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class MemoryEntry:
    reference: str
    content: str | None
    byte_count: int
    revision: str
    present: bool = True
    withheld: bool = False
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class MemoryWriteResult:
    ok: bool
    reference: str
    revision: str | None = None
    byte_count: int | None = None
    error: str | None = None
    current_revision: str | None = None


@dataclass(frozen=True, slots=True)
class MemoryPromptContext:
    text: str
    byte_count: int = 0
    line_count: int = 0
    truncated: bool = False
    withheld: bool = False
    warnings: tuple[str, ...] = ()
    authoring_guidance: str = ""


@dataclass(frozen=True, slots=True)
class MemoryBackendStatus:
    available: bool
    initialized: bool
    entry_count: int = 0
    total_bytes: int = 0
    startup_bytes: int = 0
    startup_lines: int = 0
    startup_truncated: bool = False
    startup_withheld: bool = False
    warnings: tuple[str, ...] = ()
    private_path: str | None = None


@dataclass(frozen=True, slots=True)
class ProjectMemoryStatus:
    enabled: bool
    backend_id: str
    identity_kind: str
    available: bool
    manifest_exists: bool
    backend: MemoryBackendStatus | None = None
    diagnostic: str | None = None
    display_path: str | None = None


@dataclass(frozen=True, slots=True)
class MemoryToolBinding:
    """A backend-owned model tool; ``definition`` is deliberately opaque to the host."""

    name: str
    definition: Mapping[str, Any]
    handler: Callable[..., Any]
    mutating: bool = False
