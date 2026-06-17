"""Hook configuration: the on-disk schema, parsing, scope merging, and trust gate.

Config lives in two JSON files, both ``{"schema_version": 1, "hooks": {...}}``:
- global / user scope: ``<state_dir>/hooks.json`` (always trusted)
- project scope: ``<project>/.kolega/hooks.json`` (loaded only when the project is trusted)

The ``hooks`` map is ``{EventName: [{"matcher": "...", "hooks": [<spec>, ...]}]}``.
Lists concatenate across scopes (global first, project last), matching Claude Code.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from .events import TOOL_EVENTS, HookEvent
from .matcher import HookMatcher

HOOKS_SCHEMA_VERSION = 1
HOOKS_RELATIVE_PATH = Path(".kolega") / "hooks.json"
GLOBAL_HOOKS_FILENAME = "hooks.json"

VALID_TYPES = frozenset({"command", "python", "prompt", "agent"})
DEFAULT_TIMEOUTS = {"command": 60, "python": 30, "prompt": 30, "agent": 60}
# Required field for each hook type.
REQUIRED_FIELD = {"command": "command", "python": "callable", "prompt": "prompt", "agent": "prompt"}


class HookConfigError(RuntimeError):
    """Raised when a hooks file cannot be loaded or is malformed."""


def _opt_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


@dataclass(frozen=True)
class HookSpec:
    """One configured hook handler."""

    type: str
    timeout: int
    scope: str
    command: Optional[str] = None
    callable: Optional[str] = None
    prompt: Optional[str] = None
    model: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Any, *, scope: str) -> "HookSpec":
        if not isinstance(data, dict):
            raise HookConfigError("hook entry must be an object")

        htype = str(data.get("type") or "").strip()
        if htype not in VALID_TYPES:
            raise HookConfigError(f"unknown hook type {data.get('type')!r} (expected one of {sorted(VALID_TYPES)})")

        spec = cls(
            type=htype,
            timeout=int(data.get("timeout") or DEFAULT_TIMEOUTS[htype]),
            scope=scope,
            command=_opt_str(data.get("command")),
            callable=_opt_str(data.get("callable")),
            prompt=_opt_str(data.get("prompt")),
            model=_opt_str(data.get("model")),
        )

        required = REQUIRED_FIELD[htype]
        if getattr(spec, required) is None:
            raise HookConfigError(f"{htype} hook requires a '{required}' field")
        return spec


@dataclass
class HookConfig:
    """Parsed, merged hook configuration plus any load-time diagnostics."""

    entries: dict[HookEvent, list[tuple[HookMatcher, list[HookSpec]]]] = field(default_factory=dict)
    diagnostics: list[str] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not self.entries

    def specs_for(self, event: HookEvent, target: str = "") -> list[HookSpec]:
        """All hook specs whose matcher applies to ``target`` for ``event``."""
        matched: list[HookSpec] = []
        for matcher, specs in self.entries.get(event, []):
            if matcher.matches(target):
                matched.extend(specs)
        return matched


def project_hooks_present(project_path: Path | str) -> bool:
    """True when the project defines a hooks file (used to decide whether to prompt for trust)."""
    return (Path(project_path) / HOOKS_RELATIVE_PATH).exists()


def _read_hooks_file(path: Path) -> Optional[dict[str, Any]]:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise HookConfigError(f"could not read hooks file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise HookConfigError(f"hooks file is not valid JSON: {path}") from exc

    if not isinstance(data, dict):
        raise HookConfigError(f"hooks file must be a JSON object: {path}")
    version = data.get("schema_version")
    if version != HOOKS_SCHEMA_VERSION:
        raise HookConfigError(f"unsupported hooks schema version {version!r} in {path}")
    return data


def _parse_scope(
    blob: dict[str, Any],
    *,
    scope: str,
    diagnostics: list[str],
) -> dict[HookEvent, list[tuple[HookMatcher, list[HookSpec]]]]:
    parsed: dict[HookEvent, list[tuple[HookMatcher, list[HookSpec]]]] = {}
    hooks_map = blob.get("hooks") or {}
    if not isinstance(hooks_map, dict):
        diagnostics.append(f"[{scope}] 'hooks' must be an object — ignored")
        return parsed

    for event_name, raw_entries in hooks_map.items():
        try:
            event = HookEvent(event_name)
        except ValueError:
            diagnostics.append(f"[{scope}] unknown event '{event_name}' — ignored")
            continue

        for entry in raw_entries or []:
            if not isinstance(entry, dict):
                diagnostics.append(f"[{scope}] {event_name}: entry must be an object — ignored")
                continue
            matcher = HookMatcher(entry.get("matcher"))
            specs: list[HookSpec] = []
            for raw_spec in entry.get("hooks") or []:
                try:
                    spec = HookSpec.from_dict(raw_spec, scope=scope)
                except HookConfigError as exc:
                    diagnostics.append(f"[{scope}] {event_name}: {exc} — skipped")
                    continue
                if spec.type == "agent" and event in TOOL_EVENTS:
                    diagnostics.append(
                        f"[{scope}] {event_name}: 'agent' hooks are not allowed on tool events "
                        "(recursion risk) — skipped"
                    )
                    continue
                specs.append(spec)
            if specs:
                parsed.setdefault(event, []).append((matcher, specs))
    return parsed


def load_hook_config(
    project_path: Path | str,
    state_dir: Path | str | None,
    *,
    project_trusted: bool = False,
) -> HookConfig:
    """Load and merge global + project hook configuration.

    Global/user hooks are always loaded. Project hooks are loaded only when
    ``project_trusted`` is True; otherwise their presence is reported as a
    diagnostic so the host can offer to trust the project.
    """
    diagnostics: list[str] = []
    merged: dict[HookEvent, list[tuple[HookMatcher, list[HookSpec]]]] = {}

    def absorb(blob: Optional[dict[str, Any]], scope: str) -> None:
        if not blob:
            return
        for event, entries in _parse_scope(blob, scope=scope, diagnostics=diagnostics).items():
            merged.setdefault(event, []).extend(entries)

    if state_dir is not None:
        global_path = Path(state_dir).expanduser() / GLOBAL_HOOKS_FILENAME
        try:
            absorb(_read_hooks_file(global_path), "global")
        except HookConfigError as exc:
            diagnostics.append(str(exc))

    project_file = Path(project_path) / HOOKS_RELATIVE_PATH
    if project_file.exists():
        if not project_trusted:
            diagnostics.append(f"project hooks at {project_file} are disabled until this project is trusted")
        else:
            try:
                absorb(_read_hooks_file(project_file), "project")
            except HookConfigError as exc:
                diagnostics.append(str(exc))

    return HookConfig(entries=merged, diagnostics=diagnostics)
