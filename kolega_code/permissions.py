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

from kolega_code.local_state import ensure_private_dir, write_private_text


class PermissionMode(str, Enum):
    AUTO = "auto"
    ASK = "ask"


class PermissionKind(str, Enum):
    COMMAND = "command"
    EDIT = "edit"
    MCP = "mcp"


class PermissionStoreError(RuntimeError):
    """Raised when project-local permission rules cannot be loaded."""


COMMAND_PERMISSION_TOOLS = frozenset(
    {
        "execute_terminal_command",
        "exec_command",
    }
)
EDIT_PERMISSION_TOOLS = frozenset(
    {
        "edit",
        "apply_patch",
        "lsp_edit",
        "multi_edit",
        "write",
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
    mcp_server: str = ""
    mcp_tool: str = ""

    @property
    def summary(self) -> str:
        if self.kind == PermissionKind.COMMAND:
            return self.command
        if self.kind == PermissionKind.MCP:
            return f"{self.mcp_server}/{self.mcp_tool}" if self.mcp_server else self.tool_name
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
        if self.kind == PermissionKind.MCP:
            return _matches_mcp(self, request)

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
            ensure_private_dir(self.path.parent)
            write_private_text(self.path, json.dumps(payload, indent=2, sort_keys=True) + "\n")
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
        path = _path_from_edit_inputs(inputs, tool_name=tool_name)
        return PermissionRequest(
            kind=PermissionKind.EDIT,
            tool_name=tool_name,
            inputs=inputs,
            path=path,
        )

    mcp_target = _mcp_target_from_tool_name(tool_name)
    if mcp_target is not None:
        server_id, mcp_tool = mcp_target
        return PermissionRequest(
            kind=PermissionKind.MCP,
            tool_name=tool_name,
            inputs=inputs,
            mcp_server=server_id,
            mcp_tool=mcp_tool,
        )

    return None


def allow_rule_options(request: PermissionRequest) -> list[PermissionRuleOption]:
    if request.kind == PermissionKind.COMMAND:
        return _command_rule_options(request)
    if request.kind == PermissionKind.MCP:
        return _mcp_rule_options(request)
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


def _matches_mcp(rule: PermissionRule, request: PermissionRequest) -> bool:
    if rule.match_type == "tool":
        # MCP tool rules are exact exposed-tool rules (`mcp__server__tool`).
        # Whole-server allows are represented separately by match_type="server".
        return rule.pattern in {"*", request.tool_name}
    if rule.match_type == "server":
        return bool(request.mcp_server) and request.mcp_server == rule.pattern
    return False


def _mcp_target_from_tool_name(tool_name: str) -> Optional[tuple[str, str]]:
    prefix = "mcp__"
    separator = "__"
    if not tool_name.startswith(prefix):
        return None
    rest = tool_name[len(prefix) :]
    if separator not in rest:
        return None
    server_id, mcp_tool = rest.split(separator, 1)
    if not server_id or not mcp_tool:
        return None
    return server_id, mcp_tool


def _path_from_edit_inputs(inputs: dict[str, Any], *, tool_name: str = "") -> str:
    if tool_name == "apply_patch":
        raw = inputs.get("input")
        if not isinstance(raw, str):
            return ""
        try:
            from kolega_code.agent.tool_backend.codex_patch import parse_codex_patch

            operations = parse_codex_patch(raw)
        except (ValueError, TypeError):
            return ""
        paths = list(
            dict.fromkeys(
                path for operation in operations for path in (operation.path, operation.move_to) if path is not None
            )
        )
        # A path-scoped rule is safe only when the patch affects that one path.
        return paths[0] if len(paths) == 1 else ""
    value = inputs.get("path")
    path = str(value).strip() if value is not None else ""
    if inputs.get("operation") == "rename_file" and inputs.get("new_path"):
        return f"{path} -> {str(inputs['new_path']).strip()}"
    return path


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


def _mcp_rule_options(request: PermissionRequest) -> list[PermissionRuleOption]:
    options = [
        PermissionRuleOption(
            label=f"Always allow MCP tool `{request.tool_name}`",
            description="Allow this exact MCP tool.",
            rule=PermissionRule.create(
                kind=PermissionKind.MCP,
                tool=request.tool_name,
                match_type="tool",
                pattern=request.tool_name,
            ),
        )
    ]
    if request.mcp_server:
        options.append(
            PermissionRuleOption(
                label=f"Always allow MCP server `{request.mcp_server}`",
                description="Allow all verified MCP tools from this server.",
                rule=PermissionRule.create(
                    kind=PermissionKind.MCP,
                    tool="*",
                    match_type="server",
                    pattern=request.mcp_server,
                ),
            )
        )
    return _dedupe_options(options)


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
