"""Utilities for creating and listing project prompt override files."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .prompt_overrides import (
    BROWSER_PROMPT_FILE,
    CODER_PROMPT_FILE,
    COMPACTION_PROMPT_FILE,
    GENERAL_PROMPT_FILE,
    INVESTIGATION_PROMPT_FILE,
    PLANNING_PROMPT_FILE,
    PROMPT_OVERRIDE_DIR,
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
        agent_memory="",
        agent_memory_file="",
        memories=[],
    )


def prompt_dump_contents(
    project_path: str | Path,
    *,
    base_context: Optional[PromptContext] = None,
    prompt_provider: Optional[PromptProvider] = None,
) -> dict[str, str]:
    """Render starter contents for every supported prompt override file."""
    provider = prompt_provider or PromptProvider()
    context = placeholder_dump_context(project_path, base_context=base_context)
    return {
        CODER_PROMPT_FILE: provider.get_system_prompt(
            agent_type=AgentType.CODER,
            mode=AgentMode.CLI,
            context=context,
            prompt_extensions=[],
        ).strip()
        + "\n",
        PLANNING_PROMPT_FILE: provider.get_system_prompt(
            agent_type=AgentType.PLANNING,
            context=context,
            prompt_extensions=[],
        ).strip()
        + "\n",
        GENERAL_PROMPT_FILE: provider.get_system_prompt(
            agent_type=AgentType.GENERAL,
            context=context,
            prompt_extensions=[],
        ).strip()
        + "\n",
        INVESTIGATION_PROMPT_FILE: provider.get_system_prompt(
            agent_type=AgentType.INVESTIGATION,
            context=context,
            prompt_extensions=[],
        ).strip()
        + "\n",
        BROWSER_PROMPT_FILE: provider.get_system_prompt(
            agent_type=AgentType.BROWSER,
            context=context,
            prompt_extensions=[],
        ).strip()
        + "\n",
        COMPACTION_PROMPT_FILE: COMPRESSION_SUMMARY_SYSTEM_PROMPT.strip() + "\n",
    }


def dump_prompt_overrides(
    project_path: str | Path,
    *,
    force: bool = False,
    base_context: Optional[PromptContext] = None,
    prompt_provider: Optional[PromptProvider] = None,
) -> PromptDumpResult:
    """Create prompt override starter files under ``.kolega/prompts``."""
    project = Path(project_path).expanduser().resolve()
    result = PromptDumpResult(project_path=project)
    target_dir = project / PROMPT_OVERRIDE_DIR
    try:
        target_dir.mkdir(parents=True, exist_ok=True)
    except Exception as exc:  # noqa: BLE001 - return structured CLI/TUI-friendly errors
        result.errors.append(f"Could not create {target_dir}: {exc}")
        return result

    try:
        contents = prompt_dump_contents(project, base_context=base_context, prompt_provider=prompt_provider)
    except Exception as exc:  # noqa: BLE001
        result.errors.append(f"Could not render prompt templates: {exc}")
        return result

    for spec in PROMPT_DUMP_SPECS:
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
