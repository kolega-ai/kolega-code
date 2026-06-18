"""Persistent CLI settings for provider/model selection and API keys."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .session_store import default_state_dir

SETTINGS_SCHEMA_VERSION = 2


class SettingsStoreError(RuntimeError):
    """Raised when CLI settings cannot be loaded or saved."""


@dataclass
class CliSettings:
    active_provider: Optional[str] = None
    active_model: Optional[str] = None
    active_thinking_effort: Optional[str] = None
    active_theme: Optional[str] = None
    api_keys: dict[str, str] = field(default_factory=dict)
    # Resolved project paths whose .kolega/hooks.json the user has opted to trust.
    trusted_hook_projects: list[str] = field(default_factory=list)
    schema_version: int = SETTINGS_SCHEMA_VERSION

    @classmethod
    def from_dict(cls, data: dict) -> "CliSettings":
        schema_version = data.get("schema_version")
        if schema_version not in {1, SETTINGS_SCHEMA_VERSION}:
            raise SettingsStoreError(f"Unsupported settings schema version: {data.get('schema_version')}")
        api_keys = data.get("api_keys") or {}
        trusted = data.get("trusted_hook_projects") or []
        return cls(
            schema_version=SETTINGS_SCHEMA_VERSION,
            active_provider=data.get("active_provider"),
            active_model=data.get("active_model"),
            active_thinking_effort=data.get("active_thinking_effort")
            if schema_version == SETTINGS_SCHEMA_VERSION
            else None,
            # Additive optional field; safe to read from any schema version
            # (absent in older files -> None -> default theme is applied).
            active_theme=data.get("active_theme"),
            api_keys={str(provider): str(key) for provider, key in api_keys.items() if key},
            trusted_hook_projects=[str(path) for path in trusted if path],
        )

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "active_provider": self.active_provider,
            "active_model": self.active_model,
            "active_thinking_effort": self.active_thinking_effort,
            "active_theme": self.active_theme,
            "api_keys": self.api_keys,
            "trusted_hook_projects": self.trusted_hook_projects,
        }

    def get_api_key(self, provider: str) -> Optional[str]:
        return self.api_keys.get(provider)

    def set_api_key(self, provider: str, api_key: str) -> None:
        if api_key:
            self.api_keys[provider] = api_key

    def has_api_key(self, provider: str) -> bool:
        return bool(self.get_api_key(provider))

    def is_hook_project_trusted(self, project_path) -> bool:
        return str(Path(project_path).resolve()) in self.trusted_hook_projects

    def trust_hook_project(self, project_path) -> None:
        resolved = str(Path(project_path).resolve())
        if resolved not in self.trusted_hook_projects:
            self.trusted_hook_projects.append(resolved)


class SettingsStore:
    """Filesystem-backed CLI settings store."""

    def __init__(self, root: Optional[Path] = None) -> None:
        self.root = (root or default_state_dir()).expanduser()
        self.path = self.root / "settings.json"

    def load(self) -> CliSettings:
        if not self.path.exists():
            return CliSettings()
        try:
            return CliSettings.from_dict(json.loads(self.path.read_text(encoding="utf-8")))
        except json.JSONDecodeError as exc:
            raise SettingsStoreError(f"Settings file is not valid JSON: {self.path}") from exc

    def save(self, settings: CliSettings) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(settings.to_dict(), indent=2, sort_keys=True)
        temp = self.path.with_suffix(".json.tmp")
        temp.write_text(payload + "\n", encoding="utf-8")
        _chmod_private(temp)
        temp.replace(self.path)
        _chmod_private(self.path)


def _chmod_private(path: Path) -> None:
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass
