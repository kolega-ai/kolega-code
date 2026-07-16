"""Pluggable, private project memory."""

from .identity import ProjectIdentity, resolve_project_identity
from .manager import (
    MemoryAccessError,
    MemoryUnavailableError,
    ProjectMemoryManager,
    default_registry,
)
from .markdown import MarkdownMemoryBackend, MemorySafetyError
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
    ProjectMemoryStatus,
)
from .protocol import MemoryBackend
from .registry import MemoryBackendFactory, MemoryBackendRegistry

__all__ = [
    "MEMORY_CONTRACT_VERSION",
    "MarkdownMemoryBackend",
    "MemoryAccessError",
    "MemoryAccessScope",
    "MemoryBackend",
    "MemoryBackendFactory",
    "MemoryBackendMetadata",
    "MemoryBackendRegistry",
    "MemoryBackendStatus",
    "MemoryCapability",
    "MemoryEntry",
    "MemoryEntrySummary",
    "MemoryPromptContext",
    "MemorySafetyError",
    "MemoryToolBinding",
    "MemoryUnavailableError",
    "MemoryWriteResult",
    "ProjectIdentity",
    "ProjectMemoryManager",
    "ProjectMemoryStatus",
    "default_registry",
    "resolve_project_identity",
]
