"""Schema-v1 common project-memory manifest."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from kolega_code.local_state import write_private_text

from .identity import ProjectIdentity
from .registry import validate_backend_id

MANIFEST_SCHEMA_VERSION = 1
DEFAULT_BACKEND_ID = "markdown"
MAX_SETTINGS_BYTES = 16 * 1024


@dataclass(slots=True)
class MemoryManifest:
    identity: str
    identity_kind: str
    display_path: str
    enabled: bool = True
    backend_id: str = DEFAULT_BACKEND_ID
    backend_settings: dict[str, dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def defaults(cls, identity: ProjectIdentity) -> "MemoryManifest":
        return cls(identity.identity, identity.kind, identity.display_path)

    def settings_for(self, backend_id: str) -> Mapping[str, Any]:
        value = self.backend_settings.get(backend_id, {})
        return value if isinstance(value, dict) else {}

    def to_json(self) -> str:
        payload = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "identity": {"kind": self.identity_kind, "value": self.identity},
            "display_path": self.display_path,
            "enabled": self.enabled,
            "backend_id": self.backend_id,
            "backend_settings": self.backend_settings,
        }
        encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
        if len(encoded.encode()) > MAX_SETTINGS_BYTES:
            raise ValueError("memory manifest/settings exceed size limit")
        return encoded


def load_manifest(path: Path, identity: ProjectIdentity) -> tuple[MemoryManifest, str | None]:
    try:
        if path.is_symlink():
            raise ValueError("manifest must be a regular non-symlink file")
        if not path.exists():
            return MemoryManifest.defaults(identity), None
        if not path.is_file():
            raise ValueError("manifest must be a regular non-symlink file")
        raw = path.read_bytes()
        if len(raw) > MAX_SETTINGS_BYTES:
            raise ValueError("manifest exceeds size limit")
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            raise ValueError("manifest root must be an object")
        if payload.get("schema_version") != MANIFEST_SCHEMA_VERSION:
            raise ValueError("unsupported manifest schema")
        ident = payload.get("identity", {})
        if not isinstance(ident, dict):
            raise ValueError("manifest identity must be an object")
        if ident.get("value") != identity.identity or ident.get("kind") != identity.kind:
            raise ValueError("manifest identity mismatch")
        backend_id = payload.get("backend_id")
        enabled = payload.get("enabled")
        settings = payload.get("backend_settings", {})
        if not isinstance(backend_id, str):
            raise ValueError("invalid backend ID")
        validate_backend_id(backend_id)
        if not isinstance(enabled, bool) or not isinstance(settings, dict):
            raise ValueError("invalid manifest values")
        for settings_backend_id, backend_config in settings.items():
            validate_backend_id(settings_backend_id)
            if not isinstance(backend_config, dict):
                raise ValueError("backend settings must be objects")
        manifest = MemoryManifest(
            identity=identity.identity,
            identity_kind=identity.kind,
            display_path=str(payload.get("display_path") or identity.display_path),
            enabled=enabled,
            backend_id=backend_id,
            backend_settings=settings,
        )
        return manifest, None
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError) as error:
        # Corrupt/foreign configuration must never unexpectedly expose memory.
        safe = MemoryManifest.defaults(identity)
        safe.enabled = False
        return safe, f"invalid memory manifest: {error}"


def save_manifest(path: Path, manifest: MemoryManifest) -> None:
    write_private_text(path, manifest.to_json())
