"""Agent Skills discovery and activation helpers for the CLI."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence

import yaml

from kolega_code.agent import PromptExtension, ToolExtension
from kolega_code.agent.prompts import build_skill_catalog_prompt
from kolega_code.agent.prompt_provider import AgentMode
from kolega_code.llm.models import Message
from kolega_code.llm.specs import get_model_specs


PROJECT_SKILLS_DIR = Path(".agents") / "skills"
USER_SKILLS_DIR = Path(".agents") / "skills"
BUILTIN_SKILLS_DIR = Path(__file__).parent.parent / "agent" / "prompt_templates" / "extensions" / "skills"
MAX_RESOURCE_FILES = 100
MAX_RESOURCE_READ_CHARS = 100_000
DEFAULT_SKILL_METADATA_CHAR_BUDGET = 8_000
MAX_SKILL_METADATA_CHAR_BUDGET = 48_000
SKILL_METADATA_CONTEXT_WINDOW_PERCENT = 2
APPROX_CHARS_PER_TOKEN = 4
SKILL_DESCRIPTION_TRUNCATION_SUFFIX = "..."
LIST_SKILLS_DEFAULT_MAX_RESULTS = 50
LIST_SKILLS_MAX_RESULTS = 100
SKILL_NAME_RE = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,62}[a-z0-9])?$")
SKILL_CONTENT_RE = re.compile(r'<skill_content name="([^"]+)">')


@dataclass(frozen=True)
class SkillCatalogBudget:
    max_chars: int
    source: str = "explicit"

    @classmethod
    def for_context_window(cls, context_window_tokens: Optional[int]) -> "SkillCatalogBudget":
        if context_window_tokens and context_window_tokens > 0:
            budget_tokens = max(1, context_window_tokens * SKILL_METADATA_CONTEXT_WINDOW_PERCENT // 100)
            return cls(
                max_chars=min(
                    MAX_SKILL_METADATA_CHAR_BUDGET,
                    budget_tokens * APPROX_CHARS_PER_TOKEN,
                ),
                source=f"{SKILL_METADATA_CONTEXT_WINDOW_PERCENT}% of context window",
            )
        return cls(max_chars=DEFAULT_SKILL_METADATA_CHAR_BUDGET, source="default")

    def normalized(self) -> "SkillCatalogBudget":
        return SkillCatalogBudget(max_chars=max(1, int(self.max_chars)), source=self.source)


@dataclass(frozen=True)
class SkillCatalogRenderReport:
    total_count: int
    included_count: int
    omitted_count: int
    truncated_description_count: int = 0
    truncated_description_chars: int = 0


@dataclass(frozen=True)
class SkillCatalogRenderResult:
    markdown: str
    report: SkillCatalogRenderReport


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

    def render_prompt_catalog(self, budget: Optional[SkillCatalogBudget] = None) -> SkillCatalogRenderResult:
        render = _render_skill_metadata_records(
            list(self.skills.values()),
            budget or SkillCatalogBudget.for_context_window(None),
        )
        markdown = _append_skill_metadata_notes(render.markdown, render.report)
        return SkillCatalogRenderResult(markdown=markdown, report=render.report)

    def prompt_catalog(
        self,
        *,
        context_window_tokens: Optional[int] = None,
        budget: Optional[SkillCatalogBudget] = None,
    ) -> str:
        effective_budget = budget or SkillCatalogBudget.for_context_window(context_window_tokens)
        return build_skill_catalog_prompt(self.render_prompt_catalog(effective_budget).markdown)

    def format_model_catalog(
        self,
        *,
        query: str = "",
        max_results: int = LIST_SKILLS_DEFAULT_MAX_RESULTS,
        budget: Optional[SkillCatalogBudget] = None,
    ) -> str:
        """Return a bounded model-facing skill list with names and descriptions only."""
        records = _filter_skill_records(list(self.skills.values()), query)
        clean_query = query.strip()
        if not records:
            if clean_query:
                return f"No Agent Skills matched query `{clean_query}`."
            return "No Agent Skills found."

        max_results = _clamp_max_results(max_results)
        visible_records = records[:max_results]
        render = _render_skill_metadata_records(
            visible_records,
            budget or SkillCatalogBudget.for_context_window(None),
        )

        if clean_query:
            lines = [f"Available Agent Skills matching `{clean_query}` ({len(records)} total):"]
        else:
            lines = [f"Available Agent Skills ({len(records)} total):"]
        if render.markdown:
            lines.extend(["", render.markdown])

        if render.report.truncated_description_count:
            lines.extend(["", "Skill descriptions were shortened to fit the skill metadata budget."])

        omitted_count = len(records) - render.report.included_count
        if omitted_count > 0:
            lines.extend(
                [
                    "",
                    (
                        f"{omitted_count} matching skills were not shown. "
                        "Call `list_skills(query=...)` with a narrower query to inspect more."
                    ),
                ]
            )

        return "\n".join(lines)

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


def _render_skill_metadata_records(
    records: Sequence[SkillRecord],
    budget: SkillCatalogBudget,
) -> SkillCatalogRenderResult:
    budget = budget.normalized()
    records = list(records)
    total_count = len(records)
    if not records:
        return SkillCatalogRenderResult(
            markdown="",
            report=SkillCatalogRenderReport(total_count=0, included_count=0, omitted_count=0),
        )

    full_lines = [_skill_metadata_line(record, _clean_description(record.description)) for record in records]
    if _joined_len(full_lines) <= budget.max_chars:
        return SkillCatalogRenderResult(
            markdown="\n".join(full_lines),
            report=SkillCatalogRenderReport(total_count=total_count, included_count=total_count, omitted_count=0),
        )

    minimum_lines = [_skill_metadata_line(record, "") for record in records]
    if _joined_len(minimum_lines) <= budget.max_chars:
        lines, truncated_count, truncated_chars = _largest_description_limited_lines(records, budget.max_chars)
        return SkillCatalogRenderResult(
            markdown="\n".join(lines),
            report=SkillCatalogRenderReport(
                total_count=total_count,
                included_count=total_count,
                omitted_count=0,
                truncated_description_count=truncated_count,
                truncated_description_chars=truncated_chars,
            ),
        )

    included_lines: list[str] = []
    included_records: list[SkillRecord] = []
    for record in records:
        line = _skill_metadata_line(record, "")
        if not _line_fits(included_lines, line, budget.max_chars):
            break
        included_lines.append(line)
        included_records.append(record)

    return SkillCatalogRenderResult(
        markdown="\n".join(included_lines),
        report=SkillCatalogRenderReport(
            total_count=total_count,
            included_count=len(included_lines),
            omitted_count=total_count - len(included_lines),
            truncated_description_count=sum(1 for record in included_records if _clean_description(record.description)),
            truncated_description_chars=sum(len(_clean_description(record.description)) for record in included_records),
        ),
    )


def _largest_description_limited_lines(records: Sequence[SkillRecord], max_chars: int) -> tuple[list[str], int, int]:
    max_description_len = max((len(_clean_description(record.description)) for record in records), default=0)
    best_lines = [_skill_metadata_line(record, "") for record in records]
    best_truncated_count = sum(1 for record in records if _clean_description(record.description))
    best_truncated_chars = sum(len(_clean_description(record.description)) for record in records)
    low = 0
    high = max_description_len

    while low <= high:
        mid = (low + high) // 2
        lines, truncated_count, truncated_chars = _description_limited_lines(records, mid)
        if _joined_len(lines) <= max_chars:
            best_lines = lines
            best_truncated_count = truncated_count
            best_truncated_chars = truncated_chars
            low = mid + 1
        else:
            high = mid - 1

    return best_lines, best_truncated_count, best_truncated_chars


def _description_limited_lines(records: Sequence[SkillRecord], description_limit: int) -> tuple[list[str], int, int]:
    lines: list[str] = []
    truncated_count = 0
    truncated_chars = 0
    for record in records:
        description = _clean_description(record.description)
        rendered_description = _truncate_description(description, description_limit)
        if rendered_description != description:
            truncated_count += 1
            truncated_chars += max(0, len(description) - len(rendered_description))
        lines.append(_skill_metadata_line(record, rendered_description))
    return lines, truncated_count, truncated_chars


def _truncate_description(description: str, limit: int) -> str:
    if limit <= 0:
        return ""
    if len(description) <= limit:
        return description
    if limit <= len(SKILL_DESCRIPTION_TRUNCATION_SUFFIX):
        return description[:limit].rstrip()
    body_limit = max(0, limit - len(SKILL_DESCRIPTION_TRUNCATION_SUFFIX))
    return f"{description[:body_limit].rstrip()}{SKILL_DESCRIPTION_TRUNCATION_SUFFIX}"


def _skill_metadata_line(record: SkillRecord, description: str) -> str:
    if description:
        return f"- `{record.name}`: {description}"
    return f"- `{record.name}`:"


def _clean_description(description: str) -> str:
    return " ".join(description.split())


def _joined_len(lines: Sequence[str]) -> int:
    return len("\n".join(lines))


def _line_fits(existing_lines: Sequence[str], new_line: str, max_chars: int) -> bool:
    if not existing_lines:
        return len(new_line) <= max_chars
    return _joined_len([*existing_lines, new_line]) <= max_chars


def _append_skill_metadata_notes(markdown: str, report: SkillCatalogRenderReport) -> str:
    notes: list[str] = []
    if report.truncated_description_count:
        notes.append("Skill descriptions were shortened to fit the skill metadata budget.")
    if report.omitted_count:
        notes.append(f"{report.omitted_count} additional skills were omitted to fit the skill metadata budget.")
    if not notes:
        return markdown
    note_block = "\n".join(f"- {note}" for note in notes)
    if markdown:
        return f"{markdown}\n\n{note_block}"
    return note_block


def _filter_skill_records(records: Sequence[SkillRecord], query: str) -> list[SkillRecord]:
    clean_query = query.strip().lower()
    if not clean_query:
        return list(records)
    return [
        record
        for record in records
        if clean_query in record.name.lower() or clean_query in _clean_description(record.description).lower()
    ]


def _clamp_max_results(max_results: int) -> int:
    try:
        value = int(max_results)
    except (TypeError, ValueError):
        value = LIST_SKILLS_DEFAULT_MAX_RESULTS
    return max(1, min(value, LIST_SKILLS_MAX_RESULTS))


def discover_skills(
    project_path: Path,
    *,
    user_home: Optional[Path] = None,
    include_builtin: bool = True,
) -> SkillCatalog:
    """Discover built-in, project, and user Agent Skills.

    Priority (lowest to highest): builtin < user < project.
    Built-in skills are shipped with kolega-code and can be overridden.

    Set ``include_builtin=False`` to skip built-in skills (useful in tests).
    """
    user_home = user_home or Path.home()
    catalog = SkillCatalog()
    scan_roots: list[tuple[str, Path]] = []
    if include_builtin:
        scan_roots.append(("builtin", BUILTIN_SKILLS_DIR))
    scan_roots.extend([
        ("user", user_home / USER_SKILLS_DIR),
        ("project", project_path / PROJECT_SKILLS_DIR),
    ])

    for scope, root in scan_roots:
        if not root.exists():
            continue
        for skill_file in _iter_skill_files(root):
            record, diagnostics = _load_skill(skill_file, scope)
            catalog.diagnostics.extend(diagnostics)
            if record is None:
                continue

            existing = catalog.skills.get(record.name)
            if existing is None:
                catalog.skills[record.name] = record
                continue

            # Project overrides user, user overrides builtin
            priority = {"builtin": 0, "user": 1, "project": 2}
            if priority.get(scope, 0) > priority.get(existing.scope, 0):
                catalog.diagnostics.append(
                    SkillDiagnostic(
                        severity="warning",
                        message=f"{scope.capitalize()} skill `{record.name}` overrides {existing.scope} skill at {existing.skill_dir}.",
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


def context_window_tokens_for_skill_budget(config: object, agent_name: Optional[str]) -> Optional[int]:
    try:
        model_config = config.model_config_for_agent(agent_name)  # type: ignore[attr-defined]
        specs = get_model_specs(model_config.provider, model_config.model)
        context_length = int(specs.get("context_length") or 0)
    except Exception:
        return None
    return context_length if context_length > 0 else None


def build_skill_prompt_extension(
    catalog: SkillCatalog,
    *,
    context_window_tokens: Optional[int] = None,
) -> Optional[PromptExtension]:
    if not catalog.has_skills():
        return None
    return PromptExtension(
        id="cli-agent-skills",
        title="Agent Skills",
        markdown=catalog.prompt_catalog(context_window_tokens=context_window_tokens),
        modes=[AgentMode.CLI],
    )


def build_skill_tool_extension(
    catalog: SkillCatalog,
    history_provider: Callable[[], Iterable[Message]],
) -> Optional[ToolExtension]:
    if not catalog.has_skills():
        return None

    async def list_skills(query: str = "", max_results: int = LIST_SKILLS_DEFAULT_MAX_RESULTS) -> str:
        """
        Return Agent Skills available in this CLI session.

        Use this when choosing whether a specialized workflow is available. Pass a query to search by skill name or
        description when the default result set is too broad.

        Args:
            query: Optional case-insensitive search text matched against skill names and descriptions.
            max_results: Maximum number of matching skills to inspect. Values are clamped to a safe limit.

        Returns:
            A bounded Markdown list of skill names and descriptions.
        """
        return catalog.format_model_catalog(query=query, max_results=max_results)

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
