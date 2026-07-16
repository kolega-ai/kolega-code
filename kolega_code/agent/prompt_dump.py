"""Utilities for creating, listing, and validating project prompt override files."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import platform as platform_module
from typing import Iterable, Optional

from kolega_code.services.file_system import LocalFileSystem

from .prompt_overrides import (
    BROWSER_PROMPT_FILE,
    CODER_PROMPT_FILE,
    COMPACTION_PROMPT_FILE,
    GENERAL_PROMPT_FILE,
    INVESTIGATION_PROMPT_FILE,
    PLANNING_PROMPT_FILE,
    PROMPT_OVERRIDE_DIR,
    ProjectPromptOverrides,
    PromptOverrideDiagnostic,
    format_prompt_override_error,
)
from .prompt_provider import AgentMode, AgentType, PromptContext, PromptProvider
from .prompts import COMPRESSION_SUMMARY_SYSTEM_PROMPT


@dataclass(frozen=True)
class PromptDumpSpec:
    key: str
    filename: str
    description: str

    @property
    def relative_path(self) -> str:
        return f"{PROMPT_OVERRIDE_DIR}/{self.filename}"


PROMPT_DUMP_SPECS: tuple[PromptDumpSpec, ...] = (
    PromptDumpSpec("coder", CODER_PROMPT_FILE, "CoderAgent system prompt (all coder modes)"),
    PromptDumpSpec("planning", PLANNING_PROMPT_FILE, "PlanningAgent system prompt"),
    PromptDumpSpec("general", GENERAL_PROMPT_FILE, "GeneralAgent sub-agent system prompt"),
    PromptDumpSpec("investigation", INVESTIGATION_PROMPT_FILE, "InvestigationAgent sub-agent system prompt"),
    PromptDumpSpec("browser", BROWSER_PROMPT_FILE, "BrowserAgent sub-agent system prompt"),
    PromptDumpSpec("compaction", COMPACTION_PROMPT_FILE, "Conversation compaction system prompt"),
)


@dataclass(frozen=True)
class PromptFileStatus:
    key: str
    description: str
    path: Path
    exists: bool


@dataclass
class PromptDumpResult:
    project_path: Path
    written: list[Path] = field(default_factory=list)
    skipped: list[Path] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


@dataclass(frozen=True)
class PromptListResult:
    project_path: Path
    files: tuple[PromptFileStatus, ...]

    @property
    def existing(self) -> tuple[PromptFileStatus, ...]:
        return tuple(item for item in self.files if item.exists)

    @property
    def missing(self) -> tuple[PromptFileStatus, ...]:
        return tuple(item for item in self.files if not item.exists)


@dataclass(frozen=True)
class PromptValidationResult:
    project_path: Path
    files: tuple[PromptFileStatus, ...]
    diagnostics: tuple[PromptOverrideDiagnostic, ...]

    @property
    def ok(self) -> bool:
        return not self.diagnostics

    @property
    def checked(self) -> tuple[PromptFileStatus, ...]:
        return tuple(item for item in self.files if item.exists)


def _valid_prompt_selector_message() -> str:
    keys = ", ".join(spec.key for spec in PROMPT_DUMP_SPECS)
    return f"Valid prompt selectors: {keys}, all (filename aliases such as CODER.md are also accepted)."


def select_prompt_dump_specs(selectors: Iterable[str] | None = None) -> tuple[PromptDumpSpec, ...]:
    """Normalize prompt dump selectors into canonical specs.

    Selectors are matched case-insensitively against prompt keys, filenames, and
    filename stems. Repeated selectors are de-duplicated while preserving the
    canonical ``PROMPT_DUMP_SPECS`` order.
    """
    raw_selectors = [selector.strip() for selector in selectors or () if selector.strip()]
    if not raw_selectors:
        return PROMPT_DUMP_SPECS

    aliases: dict[str, PromptDumpSpec] = {}
    for spec in PROMPT_DUMP_SPECS:
        aliases[spec.key.lower()] = spec
        aliases[spec.filename.lower()] = spec
        aliases[Path(spec.filename).stem.lower()] = spec

    selected_keys: set[str] = set()
    all_requested = False
    for selector in raw_selectors:
        normalized = selector.lower()
        if normalized == "all":
            all_requested = True
            continue
        spec = aliases.get(normalized)
        if spec is None:
            raise ValueError(f"Unknown prompt selector: {selector}. {_valid_prompt_selector_message()}")
        selected_keys.add(spec.key)

    if all_requested:
        return PROMPT_DUMP_SPECS
    return tuple(spec for spec in PROMPT_DUMP_SPECS if spec.key in selected_keys)


def placeholder_dump_context(project_path: str | Path, base_context: Optional[PromptContext] = None) -> PromptContext:
    """Build context for starter prompts with Jinja tags instead of concrete values."""
    return PromptContext(
        system_name="{{ context.system_name }}",
        project_path="{{ context.project_path }}",
        is_git_repo="{{ context.is_git_repo }}",  # type: ignore[arg-type]
        platform="{{ context.platform }}",
        date_today="{{ context.date_today }}",
        model_name="{{ context.model_name }}",
        available_ports="{{ context.available_ports }}",
        workspace_id="{{ context.workspace_id }}",
        workspace_environment_variables={},
        project_guidance="",
        project_guidance_file="",
        memories=[],
    )


def _prompt_dump_content_for_spec(
    spec: PromptDumpSpec,
    *,
    project_path: str | Path,
    base_context: Optional[PromptContext],
    prompt_provider: PromptProvider,
) -> str:
    context = placeholder_dump_context(project_path, base_context=base_context)
    if spec.filename == CODER_PROMPT_FILE:
        return (
            prompt_provider.get_system_prompt(
                agent_type=AgentType.CODER,
                mode=AgentMode.CLI,
                context=context,
                prompt_extensions=[],
            ).strip()
            + "\n"
        )
    if spec.filename == PLANNING_PROMPT_FILE:
        return (
            prompt_provider.get_system_prompt(
                agent_type=AgentType.PLANNING,
                context=context,
                prompt_extensions=[],
            ).strip()
            + "\n"
        )
    if spec.filename == GENERAL_PROMPT_FILE:
        return (
            prompt_provider.get_system_prompt(
                agent_type=AgentType.GENERAL,
                context=context,
                prompt_extensions=[],
            ).strip()
            + "\n"
        )
    if spec.filename == INVESTIGATION_PROMPT_FILE:
        return (
            prompt_provider.get_system_prompt(
                agent_type=AgentType.INVESTIGATION,
                context=context,
                prompt_extensions=[],
            ).strip()
            + "\n"
        )
    if spec.filename == BROWSER_PROMPT_FILE:
        return (
            prompt_provider.get_system_prompt(
                agent_type=AgentType.BROWSER,
                context=context,
                prompt_extensions=[],
            ).strip()
            + "\n"
        )
    if spec.filename == COMPACTION_PROMPT_FILE:
        return COMPRESSION_SUMMARY_SYSTEM_PROMPT.strip() + "\n"
    raise ValueError(f"Unsupported prompt dump spec: {spec.key}")


def prompt_dump_contents(
    project_path: str | Path,
    *,
    selectors: Iterable[str] | None = None,
    base_context: Optional[PromptContext] = None,
    prompt_provider: Optional[PromptProvider] = None,
) -> dict[str, str]:
    """Render starter contents for selected supported prompt override files."""
    provider = prompt_provider or PromptProvider()
    specs = select_prompt_dump_specs(selectors)
    return {
        spec.filename: _prompt_dump_content_for_spec(
            spec,
            project_path=project_path,
            base_context=base_context,
            prompt_provider=provider,
        )
        for spec in specs
    }


def dump_prompt_overrides(
    project_path: str | Path,
    *,
    force: bool = False,
    selectors: Iterable[str] | None = None,
    base_context: Optional[PromptContext] = None,
    prompt_provider: Optional[PromptProvider] = None,
) -> PromptDumpResult:
    """Create selected prompt override starter files under ``.kolega/prompts``."""
    specs = select_prompt_dump_specs(selectors)
    project = Path(project_path).expanduser().resolve()
    result = PromptDumpResult(project_path=project)
    target_dir = project / PROMPT_OVERRIDE_DIR
    try:
        target_dir.mkdir(parents=True, exist_ok=True)
    except Exception as exc:  # noqa: BLE001 - return structured CLI/TUI-friendly errors
        result.errors.append(f"Could not create {target_dir}: {exc}")
        return result

    try:
        contents = prompt_dump_contents(
            project,
            selectors=[spec.key for spec in specs],
            base_context=base_context,
            prompt_provider=prompt_provider,
        )
    except Exception as exc:  # noqa: BLE001
        result.errors.append(f"Could not render prompt templates: {exc}")
        return result

    for spec in specs:
        path = project / spec.relative_path
        if path.exists() and not force:
            result.skipped.append(path)
            continue
        try:
            path.write_text(contents[spec.filename], encoding="utf-8")
        except Exception as exc:  # noqa: BLE001
            result.errors.append(f"Could not write {path}: {exc}")
            continue
        result.written.append(path)
    return result


def list_prompt_overrides(project_path: str | Path) -> PromptListResult:
    """List supported prompt override files and whether they exist."""
    project = Path(project_path).expanduser().resolve()
    return PromptListResult(
        project_path=project,
        files=tuple(
            PromptFileStatus(
                key=spec.key,
                description=spec.description,
                path=project / spec.relative_path,
                exists=(project / spec.relative_path).is_file(),
            )
            for spec in PROMPT_DUMP_SPECS
        ),
    )


def _read_first_existing_project_file(project: Path, filenames: tuple[str, ...]) -> tuple[str, str]:
    for filename in filenames:
        path = project / filename
        if not path.exists():
            continue
        try:
            return filename, path.read_text(encoding="utf-8")
        except Exception:  # noqa: BLE001 - match BaseAgent's non-fatal context loading
            return filename, ""
    return "", ""


def standalone_validation_context(project_path: str | Path) -> PromptContext:
    """Build offline prompt context for prompt validation commands."""
    project = Path(project_path).expanduser().resolve()
    project_guidance_file, project_guidance = _read_first_existing_project_file(project, ("AGENTS.md", "KOLEGA.md"))
    return PromptContext(
        system_name="Kolega Code",
        project_path=str(project),
        is_git_repo=(project / ".git").is_dir(),
        platform=platform_module.system(),
        date_today=datetime.now().strftime("%Y-%m-%d"),
        model_name="",
        available_ports="9001-9999",
        project_guidance=project_guidance,
        project_guidance_file=project_guidance_file,
        kolega_md=project_guidance,
        workspace_id="",
        workspace_environment_variables={},
        memories=[],
    )


def validate_prompt_overrides(
    project_path: str | Path,
    *,
    context: Optional[PromptContext] = None,
    mode: AgentMode | str | None = AgentMode.CLI,
    project_template_slug: Optional[str] = None,
    prompt_provider: Optional[PromptProvider] = None,
) -> PromptValidationResult:
    """Validate existing supported project prompt override templates offline."""
    project = Path(project_path).expanduser().resolve()
    prompt_context = context or standalone_validation_context(project)
    overrides = ProjectPromptOverrides(LocalFileSystem(project))
    diagnostics = overrides.validate_all(
        context=prompt_context,
        mode=mode,
        project_template_slug=project_template_slug,
        prompt_provider=prompt_provider,
    )
    return PromptValidationResult(
        project_path=project,
        files=list_prompt_overrides(project).files,
        diagnostics=tuple(diagnostics),
    )


def format_prompt_dump_result(result: PromptDumpResult) -> str:
    """Human-readable summary for TUI/CLI output."""
    lines = [f"Prompt override directory: `{result.project_path / PROMPT_OVERRIDE_DIR}`"]
    if result.written:
        lines.append("\nWritten:")
        lines.extend(f"- `{path}`" for path in result.written)
    if result.skipped:
        lines.append("\nSkipped existing files:")
        lines.extend(f"- `{path}`" for path in result.skipped)
    if result.errors:
        lines.append("\nErrors:")
        lines.extend(f"- {error}" for error in result.errors)
    if not result.written and not result.skipped and not result.errors:
        lines.append("\nNo prompt files changed.")
    return "\n".join(lines)


def format_prompt_list_result(result: PromptListResult) -> str:
    """Human-readable prompt override status."""
    lines = [f"Prompt override files for `{result.project_path}`:", ""]
    for item in result.files:
        marker = "present" if item.exists else "missing"
        lines.append(f"- `{item.path}` — {marker}; {item.description}")
    return "\n".join(lines)


def format_prompt_validation_result(result: PromptValidationResult) -> str:
    """Human-readable prompt override validation result."""
    lines = [f"Prompt override validation for `{result.project_path}`:"]
    checked = result.checked
    if not checked:
        lines.append(
            f"\nNo supported prompt override files found under `{result.project_path / PROMPT_OVERRIDE_DIR}`; nothing to validate."
        )
        return "\n".join(lines)

    lines.append("\nChecked:")
    lines.extend(f"- `{item.path}`" for item in checked)

    if result.diagnostics:
        lines.append("\nErrors:")
        lines.extend(
            f"- {format_prompt_override_error(diagnostic.path, diagnostic.message)}"
            for diagnostic in result.diagnostics
        )
    else:
        lines.append("\nAll existing prompt overrides are valid.")
    return "\n".join(lines)
