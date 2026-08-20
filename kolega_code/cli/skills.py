"""Agent Skills discovery and activation helpers for the CLI."""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence

import yaml

from kolega_code.agent import PromptExtension, ToolExtension
from kolega_code.agent.prompts import build_skill_catalog_prompt
from kolega_code.agent.tool_definitions import tool_description_asset
from kolega_code.agent.prompt_provider import AgentMode
from kolega_code.llm.models import Message
from kolega_code.llm.specs import get_model_specs
from kolega_code.tools import ToolError


PROJECT_SKILLS_DIR = Path(".agents") / "skills"
USER_SKILLS_DIR = Path(".agents") / "skills"
BUNDLED_SKILLS_DIR = Path(__file__).resolve().parents[1] / "_bundled_skills" / "skills"
SKILL_SCOPE_PRECEDENCE = {
    "bundled": 0,
    "user": 1,
    "project": 2,
}
MAX_RESOURCE_FILES = 100
DEFAULT_SKILL_METADATA_CHAR_BUDGET = 8_000
MAX_SKILL_METADATA_CHAR_BUDGET = 48_000
SKILL_METADATA_CONTEXT_WINDOW_PERCENT = 2
APPROX_CHARS_PER_TOKEN = 4
SKILL_DESCRIPTION_TRUNCATION_SUFFIX = "..."
SKILL_NAME_RE = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,62}[a-z0-9])?$")
SKILL_CONTENT_RE = re.compile(r'<skill_content name="([^"]+)">')
SKILL_SUGGESTION_MAX = 5
SKILL_SUGGESTION_CUTOFF = 0.5


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

    def _require_skill(self, name: str) -> SkillRecord:
        skill_name = name.strip()
        record = self.skills.get(skill_name)
        if record is None:
            raise ValueError(_skill_not_found_message(skill_name, self.skills.values()))
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
            report=SkillCatalogRenderReport(total_count=0),
        )

    full_lines = [_skill_metadata_line(record, _clean_description(record.description)) for record in records]
    if _joined_len(full_lines) <= budget.max_chars:
        return SkillCatalogRenderResult(
            markdown="\n".join(full_lines),
            report=SkillCatalogRenderReport(total_count=total_count),
        )

    minimum_lines = [_skill_metadata_line(record, "") for record in records]
    if _joined_len(minimum_lines) <= budget.max_chars:
        lines, truncated_count, truncated_chars = _largest_description_limited_lines(records, budget.max_chars)
        return SkillCatalogRenderResult(
            markdown="\n".join(lines),
            report=SkillCatalogRenderReport(
                total_count=total_count,
                truncated_description_count=truncated_count,
                truncated_description_chars=truncated_chars,
            ),
        )

    # Even name-only lines exceed the budget: list every name anyway. The roster
    # must never hide a skill name; description text is the only thing shed.
    return SkillCatalogRenderResult(
        markdown="\n".join(minimum_lines),
        report=SkillCatalogRenderReport(
            total_count=total_count,
            truncated_description_count=sum(1 for record in records if _clean_description(record.description)),
            truncated_description_chars=sum(len(_clean_description(record.description)) for record in records),
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


def _append_skill_metadata_notes(markdown: str, report: SkillCatalogRenderReport) -> str:
    if not report.truncated_description_count:
        return markdown
    note = "Skill descriptions were shortened to fit the skill metadata budget."
    if markdown:
        return f"{markdown}\n\n- {note}"
    return f"- {note}"


def _skill_not_found_message(name: str, records: Iterable[SkillRecord]) -> str:
    message = f"Skill not found: {name}"
    matches = close_skill_matches(name, records)
    if matches:
        suggestions = ", ".join(f"`{match}`" for match in matches)
        message += f". Did you mean: {suggestions}?"
    return message


def close_skill_matches(
    name: str,
    records: Iterable[SkillRecord],
    *,
    max_matches: int = SKILL_SUGGESTION_MAX,
) -> list[str]:
    """Rank catalog names for a mistyped activation: prefix matches first, then fuzzy similarity."""
    clean_name = name.strip().lower()
    if not clean_name:
        return []
    catalog_names = [record.name for record in records]
    exact = {candidate for candidate in catalog_names if candidate.lower() == clean_name}
    prefix = [
        candidate for candidate in catalog_names if candidate not in exact and candidate.lower().startswith(clean_name)
    ]
    fuzzy = difflib.get_close_matches(clean_name, catalog_names, n=max_matches, cutoff=SKILL_SUGGESTION_CUTOFF)
    ranked: list[str] = []
    for candidate in prefix + fuzzy:
        if candidate not in ranked:
            ranked.append(candidate)
    return ranked[:max_matches]


def discover_skills(
    project_path: Path,
    *,
    user_home: Optional[Path] = None,
    bundled_root: Optional[Path] = BUNDLED_SKILLS_DIR,
) -> SkillCatalog:
    """Discover bundled, user, and project Agent Skills."""
    user_home = user_home or Path.home()
    catalog = SkillCatalog()
    scan_roots: list[tuple[str, Path]] = []
    if bundled_root is not None:
        scan_roots.append(("bundled", bundled_root))
    scan_roots.extend(
        [
            ("user", user_home / USER_SKILLS_DIR),
            ("project", project_path / PROJECT_SKILLS_DIR),
        ]
    )

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

            existing_precedence = SKILL_SCOPE_PRECEDENCE.get(existing.scope, -1)
            record_precedence = SKILL_SCOPE_PRECEDENCE.get(record.scope, -1)
            if record_precedence > existing_precedence:
                catalog.diagnostics.append(
                    SkillDiagnostic(
                        severity="warning",
                        message=(
                            f"{record.scope.capitalize()} skill `{record.name}` overrides "
                            f"{existing.scope} skill at {existing.skill_dir}."
                        ),
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
        # Both the interactive TUI (CLI) and the non-interactive `ask` command (ASK)
        # get skills; only the hosted code/vibe/fix modes are excluded.
        modes=[AgentMode.CLI, AgentMode.ASK],
    )


def build_skill_tool_extension(
    catalog: SkillCatalog,
    history_provider: Callable[[], Iterable[Message]],
) -> Optional[ToolExtension]:
    if not catalog.has_skills():
        return None

    async def skill(name: str) -> str:
        try:
            return catalog.activation_content(name, active_names=activated_skill_names(history_provider()))
        except ValueError as exc:
            raise ToolError(str(exc)) from None

    return ToolExtension(
        name="cli-agent-skills",
        tools={"skill": skill},
        tool_descriptions={"skill": tool_description_asset("skill")},
        tool_schemas={
            "skill": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "The skill name to activate, without a leading slash.",
                    }
                },
                "required": ["name"],
            }
        },
        tool_groups={
            "planning_tools": ["skill"],
            "cli_skill_tools": ["skill"],
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


def skill_activation_label(*names: str) -> str:
    """One-line transcript label for a skill activation (live and restored)."""
    listed = ", ".join(f"`/{name}`" for name in names)
    return f"Activated skill {listed}."


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
