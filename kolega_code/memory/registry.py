"""Explicit memory backend registry (no package/entry-point discovery)."""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from .models import MEMORY_CONTRACT_VERSION, MemoryBackendMetadata, MemoryCapability
from .protocol import MemoryBackend

MemoryBackendFactory = Callable[[Path, Mapping[str, Any]], MemoryBackend]
_BACKEND_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,99}\Z")


def validate_backend_id(backend_id: str) -> str:
    """Validate a stable backend ID before using it as a storage component."""
    if not isinstance(backend_id, str) or _BACKEND_ID_PATTERN.fullmatch(backend_id) is None:
        raise ValueError(
            "backend ID must be 1-100 ASCII letters, digits, dots, underscores, or hyphens "
            "and start with a letter or digit"
        )
    return backend_id


class MemoryBackendRegistry:
    def __init__(self) -> None:
        self._factories: dict[str, MemoryBackendFactory] = {}

    def register(self, backend_id: str, factory: MemoryBackendFactory, *, replace: bool = False) -> None:
        validate_backend_id(backend_id)
        if backend_id in self._factories and not replace:
            raise ValueError(f"memory backend already registered: {backend_id}")
        self._factories[backend_id] = factory

    def unregister(self, backend_id: str) -> None:
        self._factories.pop(backend_id, None)

    def create(self, backend_id: str, storage_dir: Path, settings: Mapping[str, Any]) -> MemoryBackend | None:
        factory = self._factories.get(backend_id)
        if factory is None:
            return None
        backend = factory(storage_dir, settings)
        try:
            self._validate_backend(backend_id, backend)
        except Exception:
            try:
                close = getattr(backend, "close", None)
                if callable(close):
                    close()
            except Exception:
                pass
            raise
        return backend

    @staticmethod
    def _validate_backend(backend_id: str, backend: Any) -> None:
        if not isinstance(backend, MemoryBackend):
            raise TypeError(f"memory backend {backend_id!r} does not implement the host contract")
        metadata = backend.metadata
        if not isinstance(metadata, MemoryBackendMetadata):
            raise TypeError(f"memory backend {backend_id!r} returned invalid metadata")
        if metadata.backend_id != backend_id:
            raise ValueError(f"memory backend {backend_id!r} declared a different backend ID: {metadata.backend_id!r}")
        if not isinstance(metadata.display_name, str) or not metadata.display_name.strip():
            raise ValueError(f"memory backend {backend_id!r} must declare a display name")
        if type(metadata.contract_version) is not int:
            raise TypeError(f"memory backend {backend_id!r} declared an invalid contract version")
        if metadata.contract_version != MEMORY_CONTRACT_VERSION:
            raise ValueError(
                f"memory backend {backend_id!r} uses unsupported contract version "
                f"{metadata.contract_version}; expected {MEMORY_CONTRACT_VERSION}"
            )
        if type(metadata.schema_version) is not int or metadata.schema_version < 1:
            raise ValueError(f"memory backend {backend_id!r} declared an invalid schema version")
        if not isinstance(metadata.capabilities, frozenset) or any(
            not isinstance(capability, MemoryCapability) for capability in metadata.capabilities
        ):
            raise TypeError(f"memory backend {backend_id!r} declared invalid capabilities")

    def available(self, backend_id: str) -> bool:
        return backend_id in self._factories

    @property
    def backend_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._factories))
