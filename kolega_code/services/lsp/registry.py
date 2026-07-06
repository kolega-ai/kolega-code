"""Load and merge language presets from bundled YAML + user configuration."""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Optional

import yaml

from .config import LanguageServerSpec, LanguageSpec, LspConfig, platform_key

logger = logging.getLogger(__name__)

# Path to the bundled preset registry relative to this file.
_PRESETS_PATH = Path(__file__).resolve().parent / "registry.yaml"


class LspRegistry:
    """Immutable registry of language → server mappings, built from presets + user config."""

    def __init__(self, config: Optional[LspConfig] = None) -> None:
        self._languages: dict[str, LanguageSpec] = {}
        self._ext_to_lang: dict[str, str] = {}
        self._filename_to_lang: dict[str, str] = {}
        self._initialization_options: dict[str, dict] = (
            dict(config.initialization_options) if isinstance(config, LspConfig) else {}
        )

        self._load_presets()
        if config:
            self._apply_user_config(config)

        # Build reverse indexes
        for lang_id, spec in self._languages.items():
            if lang_id in (config.disabled_languages if config else []):
                continue
            for ext in spec.extensions:
                if ext not in self._ext_to_lang:
                    self._ext_to_lang[ext] = lang_id
            for fname in spec.filename_map:
                self._filename_to_lang[fname] = lang_id

    # -- public API --------------------------------------------------------

    @property
    def languages(self) -> dict[str, LanguageSpec]:
        """All registered languages (including user overrides)."""
        return dict(self._languages)

    def get(self, language_id: str) -> Optional[LanguageSpec]:
        return self._languages.get(language_id)

    def language_for_extension(self, ext: str) -> Optional[str]:
        """Return the ``language_id`` for a file extension (lowercased, dot-prefixed)."""
        return self._ext_to_lang.get(ext.lower())

    def language_for_filename(self, filename: str) -> Optional[str]:
        """Return the ``language_id`` for an exact filename match."""
        return self._filename_to_lang.get(filename)

    def initialization_options_for(self, server_name: str) -> dict:
        """Return ``initializationOptions`` for *server_name*, or ``{}`` if unset."""
        return dict(self._initialization_options.get(server_name, {}))

    # -- loading -----------------------------------------------------------

    def _load_presets(self) -> None:
        """Load the bundled ``registry.yaml``."""
        try:
            raw = _PRESETS_PATH.read_text(encoding="utf-8")
            data = yaml.safe_load(raw)
        except Exception:
            logger.exception("Failed to load LSP presets from %s", _PRESETS_PATH)
            return

        if not isinstance(data, dict):
            return

        for lang_id, raw_spec in data.items():
            try:
                spec = _parse_language_spec(lang_id, raw_spec)
                if spec:
                    self._languages[lang_id] = spec
            except Exception:
                logger.exception("Failed to parse LSP preset for '%s'", lang_id)

    def _apply_user_config(self, config: LspConfig) -> None:
        """Merge ``LspConfig`` overrides: preferences, custom servers, disabled languages."""
        # Guard against mock objects in tests
        if not isinstance(config, LspConfig):
            return

        # 1. Register custom servers so preferences can reference them
        for server_name, raw_spec in config.custom_servers.items():
            if not isinstance(raw_spec, dict):
                continue
            try:
                custom = _parse_server_spec(server_name, raw_spec)
            except Exception:
                logger.exception("Failed to parse custom server '%s'", server_name)
                continue

            # Determine which languages to attach this server to.
            # If the user specifies a language list via "languages" key, attach there.
            # Otherwise attach wherever the user's preferences reference it.
            target_lang_ids: list[str] = []
            if "languages" in raw_spec:
                langs = raw_spec["languages"]
                target_lang_ids = langs if isinstance(langs, list) else [str(langs)]
            else:
                # Attach to any language where this server is the preferred one.
                for lid, pref_name in config.preferences.items():
                    if pref_name == server_name and lid in self._languages:
                        target_lang_ids.append(lid)

            for lid in target_lang_ids:
                if lid not in self._languages:
                    continue
                # Prepend so the user's custom server is checked first
                existing = list(self._languages[lid].language_servers)
                # Avoid duplicates
                if not any(s.name == server_name for s in existing):
                    existing.insert(0, custom)
                    self._languages[lid] = _replace_servers(self._languages[lid], existing)

        # 2. Reorder language_servers per user preferences
        for lid, pref_name in config.preferences.items():
            spec = self._languages.get(lid)
            if not spec:
                continue
            servers = list(spec.language_servers)
            # Find the preferred server
            pref_idx = next((i for i, s in enumerate(servers) if s.name == pref_name), None)
            if pref_idx is not None and pref_idx > 0:
                # Move to front
                preferred = servers.pop(pref_idx)
                servers.insert(0, preferred)
                self._languages[lid] = _replace_servers(spec, servers)

        # 3. Prune disabled languages
        for lid in config.disabled_languages:
            self._languages.pop(lid, None)

    def resolve_server(
        self, language_id: str, *, auto_fallback: bool = True
    ) -> tuple[Optional[str], Optional[LanguageServerSpec], list[LanguageServerSpec]]:
        """Find the best available server for a language.

        Returns:
            A tuple of ``(resolved_bin, matched_spec, all_candidates)``.

            - ``resolved_bin`` is the PATH-resolved binary path, or ``None``.
            - ``matched_spec`` is the ``LanguageServerSpec`` whose ``bin`` resolved, or ``None``.
            - ``all_candidates`` are all ``LanguageServerSpec`` entries for the language
              (used to generate missing-server prompts with alternatives).
        """
        spec = self._languages.get(language_id)
        if not spec or not spec.language_servers:
            return None, None, []

        # Resolve family chain (e.g. typescript → javascript)
        resolved_spec = spec
        visited: set[str] = set()
        while resolved_spec.family and resolved_spec.family not in visited:
            visited.add(resolved_spec.id)
            parent = self._languages.get(resolved_spec.family)
            if parent and parent.language_servers:
                resolved_spec = parent
            else:
                break

        all_candidates = list(resolved_spec.language_servers)

        for server in all_candidates:
            resolved = shutil.which(server.bin)
            if resolved:
                return resolved, server, all_candidates

        # Nothing found. If auto_fallback and the original spec had its own servers
        # that differ from the resolved family, try those too.
        if auto_fallback and resolved_spec.id != spec.id and spec.language_servers:
            for server in spec.language_servers:
                resolved = shutil.which(server.bin)
                if resolved:
                    return resolved, server, list(spec.language_servers)

        return None, None, all_candidates

    def install_commands_for_platform(self, server: LanguageServerSpec) -> list[str]:
        """Return install commands for the current platform, falling back to any platform."""
        pk = platform_key()
        cmds = server.install_commands.get(pk)
        if cmds:
            return cmds
        # Try any available platform
        for cmds in server.install_commands.values():
            if cmds:
                return cmds
        return []


# ---------------------------------------------------------------------------
# internal parsers
# ---------------------------------------------------------------------------


def _parse_language_spec(lang_id: str, raw: dict) -> Optional[LanguageSpec]:
    if not isinstance(raw, dict):
        return None

    servers = []
    for raw_srv in raw.get("language_servers", []) or []:
        try:
            srv = _parse_server_spec(raw_srv.get("name", ""), raw_srv)
            if srv:
                servers.append(srv)
        except Exception as exc:
            logger.warning("Failed to parse server for '%s': %s", lang_id, exc)

    return LanguageSpec(
        id=lang_id,
        display_name=raw.get("display_name", lang_id),
        config_files=list(raw.get("config_files", []) or []),
        extensions=list(raw.get("extensions", []) or []),
        filename_map=dict(raw.get("filename_map", {}) or {}),
        language_servers=servers,
        family=raw.get("family"),
    )


def _parse_server_spec(name: str, raw: dict) -> LanguageServerSpec:
    install = raw.get("install_commands", {}) or {}
    return LanguageServerSpec(
        name=name,
        bin=raw.get("bin", name),
        args=list(raw.get("args", []) or []),
        install_commands={k: (list(v) if isinstance(v, list) else [str(v)]) for k, v in install.items()},
        env=dict(raw.get("env", {}) or {}),
    )


def _replace_servers(spec: LanguageSpec, servers: list[LanguageServerSpec]) -> LanguageSpec:
    """Return a copy of *spec* with *servers* replacing ``language_servers``."""
    return LanguageSpec(
        id=spec.id,
        display_name=spec.display_name,
        config_files=list(spec.config_files),
        extensions=list(spec.extensions),
        filename_map=dict(spec.filename_map),
        language_servers=list(servers),
        family=spec.family,
    )


# ---------------------------------------------------------------------------
# project-level overrides (.kolega/lsp.json)
# ---------------------------------------------------------------------------


def load_project_lsp_config(project_path: str | Path) -> Optional[LspConfig]:
    """Load a per-project LSP configuration from ``.kolega/lsp.json``.

    Returns ``None`` if the file doesn't exist or can't be parsed.
    """
    config_path = Path(project_path) / ".kolega" / "lsp.json"
    if not config_path.exists():
        return None

    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        logger.warning("Failed to load project LSP config from %s", config_path)
        return None

    if not isinstance(data, dict):
        return None

    return LspConfig(
        enabled=data.get("enabled", True),
        auto_diagnostics_on_edit=data.get("auto_diagnostics_on_edit", True),
        max_diagnostics=data.get("max_diagnostics", 20),
        auto_fallback=data.get("auto_fallback", True),
        prompt_on_missing=data.get("prompt_on_missing", True),
        disabled_languages=list(data.get("disabled_languages", []) or []),
        preferences=dict(data.get("preferences", {}) or {}),
        custom_servers=dict(data.get("servers", {}) or {}),
        initialization_options=dict(data.get("initialization_options", {}) or {}),
        diagnostic_servers=list(data.get("diagnostic_servers", []) or []),
    )
