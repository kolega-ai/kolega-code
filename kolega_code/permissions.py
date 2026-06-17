"""Project-local permission rules for agent tool execution."""

from __future__ import annotations

import json
import shlex
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Optional


class PermissionMode(str, Enum):
    AUTO = "auto"
    ASK = "ask"


class PermissionKind(str, Enum):
    COMMAND = "command"
    EDIT = "edit"


class PermissionStoreError(RuntimeError):
    """Raised when project-local permission rules cannot be loaded."""


COMMAND_PERMISSION_TOOLS = frozenset(
    {
        "execute_terminal_command",
        "run_command",
        "run_command_tracked",
    }
)
EDIT_PERMISSION_TOOLS = frozenset(
    {
        "apply_patch",
        "create_file",
        "replace_entire_file",
        "replace_lines",
        "search_and_replace",
    }
)

PERMISSIONS_SCHEMA_VERSION = 1
PERMISSIONS_RELATIVE_PATH = Path(".kolega") / "permissions.json"


@dataclass(frozen=True)
class PermissionRequest:
    """A host approval request for a gated tool call."""

    kind: PermissionKind
    tool_name: str
    inputs: dict[str, Any]
    command: str = ""
    path: str = ""

    @property
    def summary(self) -> str:
        if self.kind == PermissionKind.COMMAND:
            return self.command
        if self.path:
            return f"{self.tool_name} {self.path}"
        return self.tool_name


@dataclass(frozen=True)
class PermissionRule:
    """A persisted allow rule."""

    id: str
    kind: PermissionKind
    tool: str
    match_type: str
    pattern: str
    created_at: str

    @classmethod
    def create(cls, *, kind: PermissionKind, tool: str, match_type: str, pattern: str) -> "PermissionRule":
        return cls(
            id=uuid.uuid4().hex,
            kind=kind,
            tool=tool,
            match_type=match_type,
            pattern=pattern,
            created_at=datetime.now(timezone.utc).isoformat(),
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PermissionRule":
        try:
            return cls(
                id=str(data["id"]),
                kind=PermissionKind(str(data["kind"])),
                tool=str(data.get("tool") or ""),
                match_type=str(data["match_type"]),
                pattern=str(data["pattern"]),
                created_at=str(data["created_at"]),
            )
        except (KeyError, ValueError, TypeError) as exc:
            raise PermissionStoreError("Permission rule is malformed.") from exc

    def to_dict(self) -> dict[str, str]:
        return {
            "id": self.id,
            "kind": self.kind.value,
            "tool": self.tool,
            "match_type": self.match_type,
            "pattern": self.pattern,
            "created_at": self.created_at,
        }

    def matches(self, request: PermissionRequest) -> bool:
        if self.kind != request.kind:
            return False
        if self.tool and self.tool != "*" and self.tool != request.tool_name:
            return False

        if self.kind == PermissionKind.COMMAND:
            return _matches_command(self.match_type, self.pattern, request.command)

        return _matches_edit(self, request)


@dataclass(frozen=True)
class PermissionDecision:
    """The host's decision for a permission request."""

    allowed: bool
    reason: str = ""
    rule: Optional[PermissionRule] = None


@dataclass(frozen=True)
class PermissionRuleOption:
    """A rule the UI can offer for an always-allow decision."""

    label: str
    description: str
    rule: PermissionRule


class ProjectPermissionStore:
    """Filesystem-backed permission rule store rooted at a project directory."""

    def __init__(self, project_path: Path) -> None:
        self.project_path = project_path
        self.path = project_path / PERMISSIONS_RELATIVE_PATH

    def load(self) -> list[PermissionRule]:
        if not self.path.exists():
            return []

        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except OSError as exc:
            raise PermissionStoreError(f"Could not read permissions file: {self.path}") from exc
        except json.JSONDecodeError as exc:
            raise PermissionStoreError(f"Permissions file is not valid JSON: {self.path}") from exc

        if data.get("schema_version") != PERMISSIONS_SCHEMA_VERSION:
            raise PermissionStoreError(f"Unsupported permissions schema version: {data.get('schema_version')}")

        raw_rules = data.get("rules")
        if not isinstance(raw_rules, list):
            raise PermissionStoreError("Permissions file must contain a rules array.")
        return [PermissionRule.from_dict(rule) for rule in raw_rules]

    def save(self, rules: Iterable[PermissionRule]) -> None:
        payload = {
            "schema_version": PERMISSIONS_SCHEMA_VERSION,
            "rules": [rule.to_dict() for rule in rules],
        }
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temp = self.path.with_suffix(".json.tmp")
            temp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            temp.replace(self.path)
        except OSError as exc:
            raise PermissionStoreError(f"Could not write permissions file: {self.path}") from exc

    def add_rule(self, rule: PermissionRule) -> None:
        rules = self.load()
        if not any(_same_rule(existing, rule) for existing in rules):
            rules.append(rule)
        self.save(rules)

    def first_match(self, request: PermissionRequest) -> Optional[PermissionRule]:
        for rule in self.load():
            if rule.matches(request):
                return rule
        return None


def permission_request_for_tool(tool_name: str, inputs: dict[str, Any]) -> Optional[PermissionRequest]:
    if tool_name in COMMAND_PERMISSION_TOOLS:
        command = str(inputs.get("command") or "").strip()
        return PermissionRequest(
            kind=PermissionKind.COMMAND,
            tool_name=tool_name,
            inputs=inputs,
            command=command,
        )

    if tool_name in EDIT_PERMISSION_TOOLS:
        path = _path_from_edit_inputs(tool_name, inputs)
        return PermissionRequest(
            kind=PermissionKind.EDIT,
            tool_name=tool_name,
            inputs=inputs,
            path=path,
        )

    return None


def allow_rule_options(request: PermissionRequest) -> list[PermissionRuleOption]:
    if request.kind == PermissionKind.COMMAND:
        return _command_rule_options(request)
    return _edit_rule_options(request)


def normalize_permission_mode(value: str | PermissionMode | None, *, default: PermissionMode) -> PermissionMode:
    if isinstance(value, PermissionMode):
        return value
    if value is None:
        return default
    try:
        return PermissionMode(str(value).lower())
    except ValueError as exc:
        valid = ", ".join(mode.value for mode in PermissionMode)
        raise ValueError(f"Unsupported permission mode '{value}'. Valid modes: {valid}") from exc


async def auto_allow_permission_callback(request: PermissionRequest) -> PermissionDecision:
    return PermissionDecision(allowed=True)


def _matches_command(match_type: str, pattern: str, command: str) -> bool:
    if match_type == "exact":
        return command == pattern
    if match_type == "prefix":
        return command == pattern or command.startswith(pattern + " ")
    if match_type == "executable":
        return _first_shell_token(command) == pattern
    return False


def _matches_edit(rule: PermissionRule, request: PermissionRequest) -> bool:
    if rule.match_type == "tool":
        return rule.pattern in {"*", request.tool_name}
    if rule.match_type == "path":
        return bool(request.path) and request.path == rule.pattern
    return False


def _path_from_edit_inputs(tool_name: str, inputs: dict[str, Any]) -> str:
    if tool_name == "apply_patch":
        return ""
    value = inputs.get("relative_path")
    return str(value).strip() if value is not None else ""


def _command_rule_options(request: PermissionRequest) -> list[PermissionRuleOption]:
    command = request.command
    if not command:
        return []

    options = [
        PermissionRuleOption(
            label="Always allow this exact command",
            description=f"Allow `{command}`.",
            rule=PermissionRule.create(
                kind=PermissionKind.COMMAND,
                tool="*",
                match_type="exact",
                pattern=command,
            ),
        )
    ]

    tokens = _shell_tokens(command)
    if len(tokens) >= 2:
        prefix = " ".join(tokens[:2])
        options.append(
            PermissionRuleOption(
                label=f"Always allow commands starting with `{prefix}`",
                description=f"Allow commands whose first words are `{prefix}`.",
                rule=PermissionRule.create(
                    kind=PermissionKind.COMMAND,
                    tool="*",
                    match_type="prefix",
                    pattern=prefix,
                ),
            )
        )

    executable = tokens[0] if tokens else ""
    if executable and executable != command:
        options.append(
            PermissionRuleOption(
                label=f"Always allow `{executable}` commands",
                description=f"Allow commands whose executable is `{executable}`.",
                rule=PermissionRule.create(
                    kind=PermissionKind.COMMAND,
                    tool="*",
                    match_type="executable",
                    pattern=executable,
                ),
            )
        )

    return _dedupe_options(options)


def _edit_rule_options(request: PermissionRequest) -> list[PermissionRuleOption]:
    options: list[PermissionRuleOption] = []
    if request.path:
        options.append(
            PermissionRuleOption(
                label=f"Always allow `{request.tool_name}` on `{request.path}`",
                description="Allow this edit tool for this path.",
                rule=PermissionRule.create(
                    kind=PermissionKind.EDIT,
                    tool=request.tool_name,
                    match_type="path",
                    pattern=request.path,
                ),
            )
        )
    options.append(
        PermissionRuleOption(
            label=f"Always allow `{request.tool_name}` edits",
            description="Allow this edit tool for any path.",
            rule=PermissionRule.create(
                kind=PermissionKind.EDIT,
                tool=request.tool_name,
                match_type="tool",
                pattern=request.tool_name,
            ),
        )
    )
    return options


def _shell_tokens(command: str) -> list[str]:
    try:
        return shlex.split(command)
    except ValueError:
        return command.split()


def _first_shell_token(command: str) -> str:
    tokens = _shell_tokens(command)
    return tokens[0] if tokens else ""


def _same_rule(left: PermissionRule, right: PermissionRule) -> bool:
    return (
        left.kind == right.kind
        and left.tool == right.tool
        and left.match_type == right.match_type
        and left.pattern == right.pattern
    )


def _dedupe_options(options: list[PermissionRuleOption]) -> list[PermissionRuleOption]:
    deduped: list[PermissionRuleOption] = []
    seen: set[tuple[PermissionKind, str, str, str]] = set()
    for option in options:
        key = (option.rule.kind, option.rule.tool, option.rule.match_type, option.rule.pattern)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(option)
    return deduped
