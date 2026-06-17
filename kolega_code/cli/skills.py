"""Agent Skills discovery and activation helpers for the CLI."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Optional

import yaml

from kolega_code.agent import PromptExtension, ToolExtension
from kolega_code.agent.prompts import build_skill_catalog_prompt
from kolega_code.llm.models import Message
from kolega_code.agent.prompt_provider import AgentMode


PROJECT_SKILLS_DIR = Path(".agents") / "skills"
USER_SKILLS_DIR = Path(".agents") / "skills"
MAX_RESOURCE_FILES = 100
MAX_RESOURCE_READ_CHARS = 100_000
SKILL_NAME_RE = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,62}[a-z0-9])?$")
SKILL_CONTENT_RE = re.compile(r'<skill_content name="([^"]+)">')


@dataclass(frozen=True)
class SkillDiagnostic:
    severity: str
    message: str
    path: Path

    def format(self) -> str:
        return f"{self.severity}: {self.message} ({self.path})"


@dataclass(frozen=True)
class SkillRecord:
    name: str
    description: str
    skill_dir: Path
    skill_file: Path
    scope: str


@dataclass
class SkillCatalog:
    skills: dict[str, SkillRecord] = field(default_factory=dict)
    diagnostics: list[SkillDiagnostic] = field(default_factory=list)

    def has_skills(self) -> bool:
        return bool(self.skills)

    def get(self, name: str) -> Optional[SkillRecord]:
        return self.skills.get(name)

    def format_catalog(self, *, include_diagnostics: bool = True) -> str:
        lines: list[str] = []
        if self.skills:
            lines.append("# Available Skills")
            lines.append("")
            for record in self.skills.values():
                lines.append(f"- `/{record.name}` ({record.scope}): {record.description}")
        else:
            lines.append("No Agent Skills found.")

        if include_diagnostics and self.diagnostics:
            lines.append("")
            lines.append("# Skill Diagnostics")
            for diagnostic in self.diagnostics:
                lines.append(f"- {diagnostic.format()}")

        return "\n".join(lines)

    def prompt_catalog(self) -> str:
        catalog_lines = [
            f"- `{record.name}` ({record.scope}): {record.description}" for record in self.skills.values()
        ]
        return build_skill_catalog_prompt("\n".join(catalog_lines))

    def activation_content(self, name: str, *, active_names: Optional[set[str]] = None) -> str:
        record = self._require_skill(name)
        active_names = active_names or set()
        if record.name in active_names:
            return f"Skill `{record.name}` is already active in this conversation. Continue using its instructions."

        _metadata, body = _parse_skill_file(record.skill_file)
        resources, truncated = self.resource_paths(record.name)
        resource_lines = [f"  <file>{resource}</file>" for resource in resources]
        if truncated:
            resource_lines.append(f"  <truncated>Only the first {MAX_RESOURCE_FILES} resources are listed.</truncated>")
        resource_listing = "\n".join(resource_lines)

        return (
            f'<skill_content name="{record.name}">\n'
            f"{body}\n\n"
            f"Skill directory: {record.skill_dir}\n"
            "Relative paths in this skill are relative to the skill directory.\n"
            "<skill_resources>\n"
            f"{resource_listing}\n"
            "</skill_resources>\n"
            "</skill_content>"
        )

    def resource_paths(self, name: str, *, max_files: int = MAX_RESOURCE_FILES) -> tuple[list[str], bool]:
        record = self._require_skill(name)
        root = record.skill_dir
        paths: list[str] = []

        for path in sorted(root.rglob("*")):
            if len(paths) >= max_files:
                return paths, True
            if not path.is_file():
                continue
            if path.name == "SKILL.md":
                continue
            if any(part in {".git", "node_modules", "__pycache__"} for part in path.relative_to(root).parts):
                continue
            try:
                resolved = path.resolve()
                resolved.relative_to(root.resolve())
            except ValueError:
                continue
            paths.append(path.relative_to(root).as_posix())

        return paths, False

    def read_resource(self, name: str, relative_path: str, *, max_chars: int = MAX_RESOURCE_READ_CHARS) -> str:
        record = self._require_skill(name)
        clean_relative_path = relative_path.strip()
        if not clean_relative_path:
            raise ValueError("relative_path must not be empty.")
        requested = Path(clean_relative_path)
        if requested.is_absolute() or ".." in requested.parts:
            raise ValueError("Skill resource path must stay inside the skill directory.")

        root = record.skill_dir.resolve()
        path = (record.skill_dir / requested).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ValueError("Skill resource path must stay inside the skill directory.") from exc

        if not path.is_file():
            raise ValueError(f"Skill resource not found: {clean_relative_path}")

        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(f"Skill resource is not UTF-8 text: {clean_relative_path}") from exc

        if len(content) <= max_chars:
            return content
        return f"{content[:max_chars]}\n\n[truncated to first {max_chars} characters]"

    def _require_skill(self, name: str) -> SkillRecord:
        skill_name = name.strip()
        record = self.skills.get(skill_name)
        if record is None:
            raise ValueError(f"Skill not found: {skill_name}")
        return record


def discover_skills(project_path: Path, *, user_home: Optional[Path] = None) -> SkillCatalog:
    """Discover project and user Agent Skills."""
    user_home = user_home or Path.home()
    catalog = SkillCatalog()
    scan_roots = [
        ("user", user_home / USER_SKILLS_DIR),
        ("project", project_path / PROJECT_SKILLS_DIR),
    ]

    for scope, root in scan_roots:
        for skill_file in _iter_skill_files(root):
            record, diagnostics = _load_skill(skill_file, scope)
            catalog.diagnostics.extend(diagnostics)
            if record is None:
                continue

            existing = catalog.skills.get(record.name)
            if existing is None:
                catalog.skills[record.name] = record
                continue

            if existing.scope == "user" and record.scope == "project":
                catalog.diagnostics.append(
                    SkillDiagnostic(
                        severity="warning",
                        message=f"Project skill `{record.name}` overrides user skill at {existing.skill_dir}.",
                        path=record.skill_file,
                    )
                )
                catalog.skills[record.name] = record
                continue

            catalog.diagnostics.append(
                SkillDiagnostic(
                    severity="warning",
                    message=f"Duplicate skill `{record.name}` ignored; already loaded from {existing.skill_dir}.",
                    path=record.skill_file,
                )
            )

    catalog.skills = dict(sorted(catalog.skills.items()))
    return catalog


def build_skill_prompt_extension(catalog: SkillCatalog) -> Optional[PromptExtension]:
    if not catalog.has_skills():
        return None
    return PromptExtension(
        id="cli-agent-skills",
        title="Agent Skills",
        markdown=catalog.prompt_catalog(),
        modes=[AgentMode.CLI],
    )


def build_skill_tool_extension(
    catalog: SkillCatalog,
    history_provider: Callable[[], Iterable[Message]],
) -> Optional[ToolExtension]:
    if not catalog.has_skills():
        return None

    async def list_skills() -> str:
        """
        Return Agent Skills available in this CLI session.

        Use this when choosing whether a specialized workflow is available.

        Returns:
            A Markdown list of skill names, scopes, and descriptions.
        """
        return catalog.format_catalog()

    async def activate_skill(name: str) -> str:
        """
        Load the full instructions for an Agent Skill.

        Call this before using a skill's specialized workflow. The returned content explains where skill resources live
        and lists bundled resources that can be read with `read_skill_resource`.

        Args:
            name: The skill name to activate, without a leading slash.

        Returns:
            The activated skill instructions, or a note if the skill is already active.
        """
        return catalog.activation_content(name, active_names=activated_skill_names(history_provider()))

    async def read_skill_resource(name: str, relative_path: str) -> str:
        """
        Read a text resource bundled with an activated Agent Skill.

        Args:
            name: The skill name, without a leading slash.
            relative_path: Path relative to the skill directory.

        Returns:
            UTF-8 text content from the requested skill resource, capped for context size.
        """
        return catalog.read_resource(name, relative_path)

    return ToolExtension(
        name="cli-agent-skills",
        tools={
            "list_skills": list_skills,
            "activate_skill": activate_skill,
            "read_skill_resource": read_skill_resource,
        },
        tool_groups={
            "planning_tools": ["list_skills", "activate_skill", "read_skill_resource"],
            "cli_skill_tools": ["list_skills", "activate_skill", "read_skill_resource"],
        },
    )


def activated_skill_names(history: Iterable[Message]) -> set[str]:
    names: set[str] = set()
    for message in history or []:
        for text in _message_text_parts(message):
            names.update(SKILL_CONTENT_RE.findall(text))
    return names


def skill_names_in_text(text: str) -> list[str]:
    return SKILL_CONTENT_RE.findall(text)


def _iter_skill_files(root: Path) -> Iterable[Path]:
    if not root.is_dir():
        return []
    return [path / "SKILL.md" for path in sorted(root.iterdir()) if path.is_dir() and (path / "SKILL.md").is_file()]


def _load_skill(skill_file: Path, scope: str) -> tuple[Optional[SkillRecord], list[SkillDiagnostic]]:
    diagnostics: list[SkillDiagnostic] = []
    try:
        metadata, _body = _parse_skill_file(skill_file)
    except Exception as exc:
        return None, [
            SkillDiagnostic(
                severity="error",
                message=f"Could not parse SKILL.md: {exc}",
                path=skill_file,
            )
        ]

    name = str(metadata.get("name") or "").strip()
    description = str(metadata.get("description") or "").strip()

    if not name:
        diagnostics.append(SkillDiagnostic("error", "Skill is missing required `name`.", skill_file))
        return None, diagnostics
    if not description:
        diagnostics.append(SkillDiagnostic("error", f"Skill `{name}` is missing required `description`.", skill_file))
        return None, diagnostics

    parent_name = skill_file.parent.name
    if name != parent_name:
        diagnostics.append(
            SkillDiagnostic(
                "warning",
                f"Skill name `{name}` does not match directory `{parent_name}`.",
                skill_file,
            )
        )
    if len(name) > 64 or not SKILL_NAME_RE.match(name) or "--" in name:
        diagnostics.append(
            SkillDiagnostic(
                "warning",
                f"Skill name `{name}` does not follow the Agent Skills name convention.",
                skill_file,
            )
        )

    if len(description) > 1024:
        diagnostics.append(
            SkillDiagnostic(
                "warning",
                f"Skill `{name}` description exceeds 1024 characters.",
                skill_file,
            )
        )

    return (
        SkillRecord(
            name=name,
            description=description,
            skill_dir=skill_file.parent.resolve(),
            skill_file=skill_file.resolve(),
            scope=scope,
        ),
        diagnostics,
    )


def _parse_skill_file(skill_file: Path) -> tuple[dict, str]:
    text = skill_file.read_text(encoding="utf-8")
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        raise ValueError("missing YAML frontmatter")

    closing_index = None
    for index, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            closing_index = index
            break
    if closing_index is None:
        raise ValueError("missing closing YAML frontmatter delimiter")

    frontmatter = "\n".join(lines[1:closing_index])
    metadata = yaml.safe_load(frontmatter) or {}
    if not isinstance(metadata, dict):
        raise ValueError("YAML frontmatter must be a mapping")

    body = "\n".join(lines[closing_index + 1 :]).strip()
    return metadata, body


def _message_text_parts(message: Message) -> Iterable[str]:
    content = getattr(message, "content", None)
    return _content_text_parts(content)


def _content_text_parts(content: object) -> list[str]:
    if isinstance(content, str):
        return [content]
    if not isinstance(content, list):
        return []

    parts: list[str] = []
    for block in content:
        if hasattr(block, "text"):
            parts.append(str(block.text))
        elif hasattr(block, "content"):
            block_content = block.content
            if isinstance(block_content, str):
                parts.append(block_content)
            else:
                parts.extend(_content_text_parts(block_content))
    return parts
