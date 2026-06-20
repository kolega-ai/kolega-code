"""Persistent CLI settings for provider/model selection and API keys."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .session_store import default_state_dir

SETTINGS_SCHEMA_VERSION = 3
_SUPPORTED_SCHEMA_VERSIONS = {1, 2, 3}
# Keys read from each saved per-agent-role model entry.
_AGENT_MODEL_KEYS = ("provider", "model", "thinking_effort")
# Reserved keys in the shared ``api_keys`` dict used for cloud web-search backends.
WEB_SEARCH_KEY_NAMES = ("firecrawl", "tavily")


class SettingsStoreError(RuntimeError):
    """Raised when CLI settings cannot be loaded or saved."""


def _coerce_agent_models(raw: object) -> dict[str, dict]:
    """Normalize a stored agent_models mapping (role -> {provider, model, ...}).

    Tolerant of partial/legacy data: entries without a provider or model are
    dropped so a malformed file can never crash startup."""
    if not isinstance(raw, dict):
        return {}
    result: dict[str, dict] = {}
    for role, entry in raw.items():
        if not isinstance(entry, dict):
            continue
        provider = entry.get("provider")
        model = entry.get("model")
        if not provider or not model:
            continue
        normalized = {key: entry[key] for key in _AGENT_MODEL_KEYS if entry.get(key) is not None}
        result[str(role)] = normalized
    return result


def _coerce_oauth_tokens(raw: object) -> dict[str, dict]:
    """Normalize a stored oauth_tokens mapping (provider -> token dict).

    Tolerant of partial/legacy data: an entry missing the access/refresh token
    is dropped so a malformed file can never crash startup."""
    if not isinstance(raw, dict):
        return {}
    result: dict[str, dict] = {}
    for provider, entry in raw.items():
        if not isinstance(entry, dict):
            continue
        if not entry.get("access_token") or not entry.get("refresh_token"):
            continue
        result[str(provider)] = dict(entry)
    return result


@dataclass
class CliSettings:
    active_provider: Optional[str] = None
    active_model: Optional[str] = None
    active_thinking_effort: Optional[str] = None
    active_theme: Optional[str] = None
    api_keys: dict[str, str] = field(default_factory=dict)
    # Per-agent-role model overrides, keyed by AgentRole value (e.g. "investigation"),
    # each value a {provider, model, thinking_effort} dict. Empty = every role uses
    # the active model.
    agent_models: dict[str, dict] = field(default_factory=dict)
    # Resolved project paths whose .kolega/hooks.json the user has opted to trust.
    trusted_hook_projects: list[str] = field(default_factory=list)
    # Web search backend selection. Additive optional fields (absent -> None -> the
    # "duckduckgo" default is applied downstream), so no schema bump is needed — same
    # convention as active_theme. Cloud backend API keys live in api_keys under the
    # WEB_SEARCH_KEY_NAMES keys.
    web_search_backend: Optional[str] = None
    web_search_base_url: Optional[str] = None
    # ChatGPT-subscription OAuth credentials, keyed by provider value (e.g.
    # "openai_chatgpt"), each a serialized OAuthTokens dict. Additive optional
    # field — absent in older files -> empty mapping, no schema bump (same
    # convention as web_search_backend / active_theme).
    oauth_tokens: dict[str, dict] = field(default_factory=dict)
    schema_version: int = SETTINGS_SCHEMA_VERSION

    @classmethod
    def from_dict(cls, data: dict) -> "CliSettings":
        schema_version = data.get("schema_version")
        if schema_version not in _SUPPORTED_SCHEMA_VERSIONS:
            raise SettingsStoreError(f"Unsupported settings schema version: {data.get('schema_version')}")
        api_keys = data.get("api_keys") or {}
        trusted = data.get("trusted_hook_projects") or []
        return cls(
            schema_version=SETTINGS_SCHEMA_VERSION,
            active_provider=data.get("active_provider"),
            active_model=data.get("active_model"),
            active_thinking_effort=data.get("active_thinking_effort") if schema_version >= 2 else None,
            # Additive optional field; safe to read from any schema version
            # (absent in older files -> None -> default theme is applied).
            active_theme=data.get("active_theme"),
            api_keys={str(provider): str(key) for provider, key in api_keys.items() if key},
            # Additive optional field; absent in pre-v3 files -> empty mapping.
            agent_models=_coerce_agent_models(data.get("agent_models")),
            trusted_hook_projects=[str(path) for path in trusted if path],
            # Additive optional fields; absent in older files -> None -> default backend.
            web_search_backend=data.get("web_search_backend"),
            web_search_base_url=data.get("web_search_base_url"),
            # Additive optional field; absent in older files -> empty mapping.
            oauth_tokens=_coerce_oauth_tokens(data.get("oauth_tokens")),
        )

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "active_provider": self.active_provider,
            "active_model": self.active_model,
            "active_thinking_effort": self.active_thinking_effort,
            "active_theme": self.active_theme,
            "api_keys": self.api_keys,
            "agent_models": self.agent_models,
            "trusted_hook_projects": self.trusted_hook_projects,
            "web_search_backend": self.web_search_backend,
            "web_search_base_url": self.web_search_base_url,
            "oauth_tokens": self.oauth_tokens,
        }

    def get_api_key(self, provider: str) -> Optional[str]:
        return self.api_keys.get(provider)

    def get_agent_model(self, role: str) -> Optional[dict]:
        """Return the saved override for a role, or None when it inherits."""
        return self.agent_models.get(role)

    def set_agent_model(self, role: str, provider: str, model: str, thinking_effort: Optional[str] = None) -> None:
        """Record a per-role model override (provider and model are required)."""
        entry = {"provider": provider, "model": model}
        if thinking_effort:
            entry["thinking_effort"] = thinking_effort
        self.agent_models[role] = entry

    def clear_agent_model(self, role: str) -> None:
        """Remove a per-role override so the role inherits the active model."""
        self.agent_models.pop(role, None)

    def set_api_key(self, provider: str, api_key: str) -> None:
        if api_key:
            self.api_keys[provider] = api_key

    def has_api_key(self, provider: str) -> bool:
        return bool(self.get_api_key(provider))

    def get_oauth_token(self, provider: str) -> Optional[dict]:
        """Return the stored OAuth token dict for a provider, or None."""
        return self.oauth_tokens.get(provider)

    def set_oauth_token(self, provider: str, token: dict) -> None:
        """Persist an OAuth token dict (serialized OAuthTokens) for a provider."""
        if token:
            self.oauth_tokens[provider] = dict(token)

    def clear_oauth_token(self, provider: str) -> None:
        """Remove stored OAuth credentials for a provider (sign out)."""
        self.oauth_tokens.pop(provider, None)

    def has_oauth_token(self, provider: str) -> bool:
        return bool(self.get_oauth_token(provider))

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
