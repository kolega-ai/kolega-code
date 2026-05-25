from typing import Optional, List, Dict
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import jinja2
import logging

logger = logging.getLogger(__name__)

# Set up directory paths and Jinja2 environment at module level
# This ensures the environment is created only once, improving performance
_base_dir = Path(__file__).parent / "prompt_templates"
_jinja_env = jinja2.Environment(
    loader=jinja2.FileSystemLoader(str(_base_dir)), trim_blocks=True, lstrip_blocks=True
)


class AgentType(Enum):
    CODER = "coder"
    INVESTIGATION = "investigation"
    BROWSER = "browser"


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
    kolega_md: str = ""
    workspace_id: str = ""
    workspace_environment_variables: Dict[str, str] = field(default_factory=dict)

    # Workspace memories
    memories: List[str] = field(default_factory=list)


class PromptProvider:
    """
    Centralized prompt provider using Jinja2 templates.
    Generates system prompts based on agent type, mode, template, and host-provided extensions.
    """

    def __init__(self):
        pass

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
            jinja_template = _jinja_env.get_template(template_name)
        except jinja2.TemplateNotFound:
            logger.error(f"Template not found: {template_name}")
            raise

        # Prepare template variables
        template_vars = {
            "context": context,
            "mode": mode.value if isinstance(mode, AgentMode) else mode,
            "project_template_slug": template_slug,
            "prompt_extensions": matching_extensions,
        }

        # Render the prompt
        try:
            return jinja_template.render(**template_vars)
        except Exception as e:
            logger.error(f"Error rendering template {template_name}: {e}")
            raise

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
                return f"agents/{agent_type.value}_cli_mode.j2"
            elif mode_value == AgentMode.VIBE.value:
                return f"agents/{agent_type.value}_vibe_mode.j2"
            elif mode_value == AgentMode.CODE.value:
                return f"agents/{agent_type.value}_code_mode.j2"
            elif mode_value == AgentMode.FIX.value:
                return f"agents/{agent_type.value}_fix_mode.j2"
            else:
                raise ValueError(
                    f"CODER agent requires a valid mode ('cli', 'vibe', 'code', or 'fix'), got: {mode_value}"
                )

        return f"agents/{agent_type.value}.j2"
