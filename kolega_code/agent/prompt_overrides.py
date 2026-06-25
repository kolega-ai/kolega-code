"""Project-local prompt override discovery.

Overrides are Markdown-with-Jinja templates under ``.kolega/prompts``. The
filenames are fixed and uppercase so a cloned repository cannot steer the loader
to arbitrary paths.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, Optional

import jinja2
from jinja2.sandbox import SandboxedEnvironment

from kolega_code.services.file_system import FileSystem

from .prompt_provider import AgentMode, AgentType, PromptContext, PromptProvider

logger = logging.getLogger(__name__)

PROMPT_OVERRIDE_DIR = ".kolega/prompts"
MAX_PROMPT_OVERRIDE_BYTES = 128 * 1024

CODER_PROMPT_FILE = "CODER.md"
PLANNING_PROMPT_FILE = "PLANNING.md"
GENERAL_PROMPT_FILE = "GENERAL.md"
INVESTIGATION_PROMPT_FILE = "INVESTIGATION.md"
BROWSER_PROMPT_FILE = "BROWSER.md"
COMPACTION_PROMPT_FILE = "COMPACTION.md"

AGENT_PROMPT_FILENAMES: dict[str, str] = {
    AgentType.CODER.value: CODER_PROMPT_FILE,
    AgentType.PLANNING.value: PLANNING_PROMPT_FILE,
    AgentType.GENERAL.value: GENERAL_PROMPT_FILE,
    AgentType.INVESTIGATION.value: INVESTIGATION_PROMPT_FILE,
    AgentType.BROWSER.value: BROWSER_PROMPT_FILE,
}


@dataclass(frozen=True)
class PromptOverride:
    """A loaded project-local prompt override."""

    path: str
    content: str


_OVERRIDE_ENV = SandboxedEnvironment(
    loader=None,
    undefined=jinja2.StrictUndefined,
    trim_blocks=True,
    lstrip_blocks=True,
    keep_trailing_newline=True,
)


def render_prompt_override_source(
    source: str,
    *,
    context: PromptContext,
    mode: AgentMode | str | None,
    project_template_slug: Optional[str],
    prompt_provider: Optional[PromptProvider] = None,
) -> str:
    """Render a project-local prompt override using a limited Jinja context."""
    provider = prompt_provider or PromptProvider()
    template = _OVERRIDE_ENV.from_string(source)
    return template.render(
        **provider.template_vars(
            context=context,
            mode=mode,
            template_slug=project_template_slug,
            prompt_extensions=[],
        )
    ).strip()


class ProjectPromptOverrides:
    """Load fixed prompt override files from a project filesystem."""

    def __init__(self, filesystem: FileSystem) -> None:
        self.filesystem = filesystem

    def load_agent_system_prompt(self, agent_type: AgentType | str) -> Optional[PromptOverride]:
        """Return the Markdown template override for ``agent_type``, if present."""
        agent_type_value = agent_type.value if isinstance(agent_type, AgentType) else str(agent_type)
        filename = AGENT_PROMPT_FILENAMES.get(agent_type_value)
        if filename is None:
            return None
        return self._load(filename)

    def load_compaction_system_prompt(self) -> Optional[PromptOverride]:
        """Return the Markdown template override for conversation compaction, if present."""
        return self._load(COMPACTION_PROMPT_FILE)

    def _load(self, filename: str) -> Optional[PromptOverride]:
        path = f"{PROMPT_OVERRIDE_DIR}/{filename}"
        try:
            if not self._has_exact_filename(filename):
                return None
            if not self.filesystem.exists(path):
                return None
            if not self.filesystem.is_file(path):
                logger.warning("Prompt override %s exists but is not a file; ignoring it.", path)
                return None
            size = self._size(path)
            if size is not None and size > MAX_PROMPT_OVERRIDE_BYTES:
                logger.warning(
                    "Prompt override %s is too large (%s bytes > %s bytes); ignoring it.",
                    path,
                    size,
                    MAX_PROMPT_OVERRIDE_BYTES,
                )
                return None
            content = self.filesystem.read_text(path)
        except Exception as exc:  # noqa: BLE001 - overrides should never break agent startup
            logger.warning("Could not read prompt override %s: %s", path, exc)
            return None

        if len(content.encode("utf-8")) > MAX_PROMPT_OVERRIDE_BYTES:
            logger.warning(
                "Prompt override %s is too large after reading (> %s bytes); ignoring it.",
                path,
                MAX_PROMPT_OVERRIDE_BYTES,
            )
            return None
        return PromptOverride(path=path, content=content)

    def _has_exact_filename(self, filename: str) -> bool:
        """Return True only when the override directory lists the exact uppercase name."""
        try:
            if not self.filesystem.is_dir(PROMPT_OVERRIDE_DIR):
                return False
            names = {self.filesystem.get_name(item) for item in self.filesystem.listdir(PROMPT_OVERRIDE_DIR)}
        except Exception:
            return False
        return filename in names

    def _size(self, path: str) -> Optional[int]:
        try:
            stat = self.filesystem.stat(path)
        except Exception:
            return None
        value: Any = stat.get("size")
        try:
            return int(value)
        except (TypeError, ValueError):
            return None
