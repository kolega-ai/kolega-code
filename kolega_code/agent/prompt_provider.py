from typing import Any, Optional, List, Dict, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import jinja2
import logging

logger = logging.getLogger(__name__)

_base_dir = Path(__file__).parent / "prompt_templates"


class MissingPromptTemplateError(RuntimeError):
    """Raised when a host-only agent mode has no bundled or host-provided prompt template."""


class AgentType(Enum):
    CODER = "coder"
    PLANNING = "planning"
    INVESTIGATION = "investigation"
    BROWSER = "browser"
    GENERAL = "general"


class AgentMode(Enum):
    CLI = "cli"
    VIBE = "vibe"
    CODE = "code"
    FIX = "fix"


@dataclass(frozen=True)
class PromptExtension:
    """Host-provided prompt section rendered into matching agent prompts."""

    id: str
    title: str
    markdown: str
    agent_types: Optional[List[AgentType | str]] = None
    modes: Optional[List[AgentMode | str]] = None
    # Whether this section is inherited by sub-agents. Guidance that only applies
    # to the single top-level agent (task list, planning questions, the gigacode
    # authoring guide) should not bloat or mislead sub-agent prompts.
    propagate_to_sub_agents: bool = True


@dataclass
class PromptContext:
    """Context information for prompt generation."""

    system_name: str = "Kolega Code"
    project_path: str = ""
    is_git_repo: bool = False
    platform: str = ""
    date_today: str = ""
    model_name: str = ""
    available_ports: str = "9001-9999"
    project_guidance: str = ""
    project_guidance_file: str = ""
    private_memory: str = ""
    kolega_md: str = ""
    workspace_id: str = ""
    workspace_environment_variables: Dict[str, str] = field(default_factory=dict)

    # Workspace memories
    memories: List[str] = field(default_factory=list)

    # Bug investigation context — when set, the investigation agent renders
    # the two-pass methodology with bug-specific parameters.
    investigation: Optional[Dict[str, str]] = None

    def __post_init__(self) -> None:
        """Keep the legacy KOLEGA.md field usable for older callers."""
        if self.project_guidance and not self.kolega_md:
            self.kolega_md = self.project_guidance
            if not self.project_guidance_file:
                self.project_guidance_file = "AGENTS.md"
        elif self.kolega_md and not self.project_guidance:
            self.project_guidance = self.kolega_md
            if not self.project_guidance_file:
                self.project_guidance_file = "KOLEGA.md"


class PromptProvider:
    """
    Centralized prompt provider using Jinja2 templates.
    Generates system prompts based on agent type, mode, template, and host-provided extensions.
    """

    def __init__(
        self,
        template_dirs: Optional[Sequence[str | Path]] = None,
        *,
        include_builtin_templates: bool = True,
    ):
        loaders = [jinja2.FileSystemLoader(str(Path(template_dir))) for template_dir in template_dirs or []]
        if include_builtin_templates:
            loaders.append(jinja2.FileSystemLoader(str(_base_dir)))
        self._jinja_env = jinja2.Environment(
            loader=jinja2.ChoiceLoader(loaders),
            trim_blocks=True,
            lstrip_blocks=True,
        )

    def template_vars(
        self,
        *,
        context: PromptContext,
        mode: Optional[AgentMode | str],
        template_slug: Optional[str],
        prompt_extensions: Optional[List[PromptExtension]] = None,
    ) -> Dict[str, Any]:
        """Return template variables shared by bundled and project override prompts."""
        mode_value = mode.value if isinstance(mode, AgentMode) else mode
        return {
            "context": context,
            "mode": mode_value,
            "project_template_slug": template_slug,
            "prompt_extensions": prompt_extensions or [],
            "investigation": context.investigation,
            "system_name": context.system_name,
            "project_path": context.project_path,
            "is_git_repo": context.is_git_repo,
            "platform": context.platform,
            "date_today": context.date_today,
            "model_name": context.model_name,
            "available_ports": context.available_ports,
            "workspace_id": context.workspace_id,
        }

    def get_system_prompt(
        self,
        agent_type: AgentType,
        mode: Optional[AgentMode] = None,
        template_slug: Optional[str] = None,
        prompt_extensions: Optional[List[PromptExtension]] = None,
        context: Optional[PromptContext] = None,
    ) -> str:
        """
        Generate a complete system prompt based on provided parameters.

        Args:
            agent_type: Type of shared coding agent (coder, investigation, browser)
            mode: Agent mode (cli/vibe/code/fix) - only applies to coder agents
            template_slug: Project template slug identifier (e.g., 'mern-stack-template')
            prompt_extensions: Host-provided prompt sections to render for matching agents
            context: Environment and project context

        Returns:
            Complete system prompt string
        """
        context = context or PromptContext()
        matching_extensions = self._filter_prompt_extensions(agent_type, mode, prompt_extensions or [])

        # Get the appropriate Jinja2 template
        template_name = self._get_template_name(agent_type, mode)
        logger.debug(f"Using prompt template: {template_name}")

        try:
            jinja_template = self._jinja_env.get_template(template_name)
        except jinja2.TemplateNotFound:
            message = self._missing_template_message(agent_type, mode, template_name)
            logger.error(message)
            raise MissingPromptTemplateError(message)

        # Prepare template variables. The ``context`` object is the canonical
        # surface; planning.md.j2 also has legacy top-level names for backward
        # compatibility with tests and host templates.
        template_vars = self.template_vars(
            context=context,
            mode=mode,
            template_slug=template_slug,
            prompt_extensions=matching_extensions,
        )

        # Render the prompt
        try:
            return jinja_template.render(**template_vars)
        except Exception as e:
            logger.error(f"Error rendering template {template_name}: {e}")
            raise

    def render_dynamic_sections(
        self,
        agent_type: AgentType,
        mode: Optional[AgentMode],
        prompt_extensions: Optional[List[PromptExtension]],
        context: PromptContext,
    ) -> str:
        """Render dynamic project/runtime sections appended to raw overrides.

        Bundled agent templates already include these sections themselves (except
        for the planning template, whose caller appends this output). Project
        override files are Markdown templates, so they receive the same dynamic
        information through this centralized suffix.
        """
        sections: List[str] = []
        matching_extensions = self._filter_prompt_extensions(agent_type, mode, prompt_extensions or [])

        if context.memories:
            body = "\n".join(f"- {memory}" for memory in context.memories)
            sections.append(
                f"## Workspace Memories\n\nYou have access to the following information about this workspace:\n\n{body}"
            )

        if matching_extensions:
            body_parts = ["## Additional Context"]
            for extension in matching_extensions:
                body_parts.append(f"### {extension.title}\n\n{extension.markdown}")
            sections.append("\n\n".join(body_parts))

        if context.project_guidance:
            sections.append(
                "## Project Instructions\n\n"
                f"The project directory contains `{context.project_guidance_file}`. "
                "Treat it as local project guidance:\n\n"
                "```markdown\n"
                f"{context.project_guidance}\n"
                "```"
            )

        if context.private_memory:
            sections.append(context.private_memory)

        return "\n\n".join(section.strip() for section in sections if section.strip())

    def _filter_prompt_extensions(
        self,
        agent_type: AgentType,
        mode: Optional[AgentMode],
        prompt_extensions: List[PromptExtension],
    ) -> List[PromptExtension]:
        """Return host prompt sections that apply to the current agent and mode."""
        agent_type_value = agent_type.value if isinstance(agent_type, AgentType) else str(agent_type)
        mode_value = mode.value if isinstance(mode, AgentMode) else mode

        matching_extensions = []
        for extension in prompt_extensions:
            if extension.agent_types:
                allowed_agent_types = {
                    item.value if isinstance(item, AgentType) else str(item) for item in extension.agent_types
                }
                if agent_type_value not in allowed_agent_types:
                    continue

            if extension.modes:
                allowed_modes = {item.value if isinstance(item, AgentMode) else str(item) for item in extension.modes}
                if mode_value not in allowed_modes:
                    continue

            matching_extensions.append(extension)

        return matching_extensions

    def _get_template_name(self, agent_type: AgentType, mode: Optional[AgentMode]) -> str:
        """Get the Jinja2 template filename for the agent type and mode."""
        mode_value = mode.value if isinstance(mode, AgentMode) else mode
        if agent_type == AgentType.CODER:
            if mode_value == AgentMode.CLI.value:
                return "system/agents/coder_cli.md.j2"
            elif mode_value == AgentMode.VIBE.value:
                return "system/agents/coder_vibe.md.j2"
            elif mode_value == AgentMode.CODE.value:
                return "system/agents/coder_code.md.j2"
            elif mode_value == AgentMode.FIX.value:
                return "system/agents/coder_fix.md.j2"
            else:
                raise ValueError(
                    f"CODER agent requires a valid mode ('cli', 'vibe', 'code', or 'fix'), got: {mode_value}"
                )
        if agent_type == AgentType.PLANNING:
            return "system/agents/planning.md.j2"

        return f"system/agents/{agent_type.value}.md.j2"

    def _missing_template_message(
        self,
        agent_type: AgentType,
        mode: Optional[AgentMode],
        template_name: str,
    ) -> str:
        mode_value = mode.value if isinstance(mode, AgentMode) else mode
        if agent_type == AgentType.CODER and mode_value in {
            AgentMode.CODE.value,
            AgentMode.VIBE.value,
            AgentMode.FIX.value,
        }:
            return (
                f"Prompt template '{template_name}' is required for coder {mode_value!r} mode, "
                "but kolega-code does not ship hosted-mode prompts. Pass a PromptProvider "
                "configured with host-owned template_dirs."
            )
        return f"Prompt template not found: {template_name}"
