"""Discovery and runtime support for user- and project-defined custom agents."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import yaml

from kolega_code.config import AgentConfig, ModelConfig, ModelProvider, RateLimitConfig
from kolega_code.events import AgentConnectionManager
from kolega_code.llm.models import Message, TextBlock
from kolega_code.llm.specs import get_model_specs, normalize_thinking_effort

from .baseagent import BaseAgent
from .prompt_provider import AgentMode, AgentType, PromptExtension, PromptProvider
from .tools import ToolCollection, ToolCollectionConfig


PROJECT_CUSTOM_AGENTS_DIR = Path(".kolega") / "agents"
USER_CUSTOM_AGENTS_DIR = Path("agents")
MAX_CUSTOM_AGENT_FILE_BYTES = 128 * 1024
MAX_CUSTOM_AGENT_DESCRIPTION_CHARS = 1024
CUSTOM_AGENT_NAME_RE = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,62}[a-z0-9])?$")
RESERVED_CUSTOM_AGENT_NAMES = frozenset(
    {
        "browser-agent",
        "coder",
        "general-agent",
        "investigation-agent",
        "planning-agent",
    }
)
CUSTOM_AGENT_FIELDS = frozenset(
    {
        "name",
        "description",
        "mode",
        "tools",
        "model",
        "thinking_effort",
        "max_iterations",
    }
)


@dataclass(frozen=True)
class CustomAgentDiagnostic:
    severity: str
    message: str
    path: Path

    def format(self) -> str:
        return f"{self.severity}: {self.message} ({self.path})"


@dataclass(frozen=True)
class CustomAgentDefinition:
    name: str
    description: str
    prompt: str
    source_path: Path
    scope: str
    mode: str = "build"
    tools: Optional[tuple[str, ...]] = None
    model: Optional[str] = None
    thinking_effort: Optional[str] = None
    max_iterations: Optional[int] = None

    def resolve_model_config(self, config: AgentConfig) -> ModelConfig:
        """Resolve the definition's model/effort over the configured General role."""
        inherited = config.model_config_for_agent("general-agent")
        provider = inherited.provider
        model = inherited.model

        if self.model is not None:
            provider_value, model = self.model.split("/", 1)
            provider = ModelProvider(provider_value)

        # Revalidate here because an effort-only definition depends on the active
        # General model, which is not known during filesystem discovery.
        get_model_specs(provider, model)
        if self.thinking_effort is not None:
            effort_source = self.thinking_effort
        elif self.model is None:
            effort_source = inherited.thinking_effort
        else:
            effort_source = None
        thinking_effort = normalize_thinking_effort(provider, model, effort_source)
        return ModelConfig(
            provider=provider,
            model=model,
            rate_limits=RateLimitConfig(**inherited.rate_limits.model_dump()),
            thinking_effort=thinking_effort,
        )


@dataclass
class CustomAgentCatalog:
    agents: dict[str, CustomAgentDefinition] = field(default_factory=dict)
    diagnostics: list[CustomAgentDiagnostic] = field(default_factory=list)

    def has_agents(self) -> bool:
        return bool(self.agents)

    def get(self, name: str) -> Optional[CustomAgentDefinition]:
        return self.agents.get(name)

    def names(self) -> list[str]:
        return list(self.agents)

    def for_mode(self, mode: str) -> "CustomAgentCatalog":
        """Return definitions enabled for a Build or Plan parent agent."""
        if mode not in {"build", "plan"}:
            raise ValueError(f"Unknown custom-agent mode: {mode}")
        return CustomAgentCatalog(
            agents={name: definition for name, definition in self.agents.items() if definition.mode in {mode, "all"}},
            diagnostics=list(self.diagnostics),
        )

    def model_catalog(self, max_chars: int = 8_000) -> str:
        """Return bounded routing metadata for the dispatch tool description."""
        lines: list[str] = []
        omitted = 0
        for definition in self.agents.values():
            description = " ".join(definition.description.split())
            line = f"- `{definition.name}`: {description}"
            candidate = "\n".join([*lines, line])
            if len(candidate) > max_chars:
                omitted += 1
                continue
            lines.append(line)
        if omitted:
            lines.append(f"- {omitted} additional agent(s) omitted from descriptions; their names remain selectable.")
        return "\n".join(lines)

    def format_catalog(self, *, include_diagnostics: bool = True) -> str:
        lines: list[str] = []
        if self.agents:
            lines.extend(["# Custom Agents", ""])
            for definition in self.agents.values():
                model = definition.model or "inherit general"
                tools = "inherit caller" if definition.tools is None else ", ".join(definition.tools) or "none"
                lines.append(
                    f"- `{definition.name}` ({definition.scope}, {definition.mode}): {definition.description} "
                    f"[model: {model}; tools: {tools}; source: {definition.source_path}]"
                )
        else:
            lines.append("No custom agents found.")

        if include_diagnostics and self.diagnostics:
            lines.extend(["", "# Custom Agent Diagnostics"])
            lines.extend(f"- {diagnostic.format()}" for diagnostic in self.diagnostics)
        return "\n".join(lines)

    def has_errors(self) -> bool:
        return any(diagnostic.severity == "error" for diagnostic in self.diagnostics)


def discover_custom_agents(project_path: Path, state_dir: Path) -> CustomAgentCatalog:
    """Discover effective custom agents, with project definitions overriding user definitions."""
    catalog = CustomAgentCatalog()
    scan_roots = [
        ("user", Path(state_dir) / USER_CUSTOM_AGENTS_DIR),
        ("project", Path(project_path) / PROJECT_CUSTOM_AGENTS_DIR),
    ]

    for scope, root in scan_roots:
        for definition_file in _iter_definition_files(root):
            definition, diagnostics = _load_definition(definition_file, scope)
            catalog.diagnostics.extend(diagnostics)
            if definition is None:
                continue

            existing = catalog.agents.get(definition.name)
            if existing is None:
                catalog.agents[definition.name] = definition
                continue

            if existing.scope == "user" and definition.scope == "project":
                catalog.diagnostics.append(
                    CustomAgentDiagnostic(
                        "warning",
                        f"Project custom agent `{definition.name}` overrides user definition at {existing.source_path}.",
                        definition.source_path,
                    )
                )
                catalog.agents[definition.name] = definition
                continue

            catalog.diagnostics.append(
                CustomAgentDiagnostic(
                    "warning",
                    f"Duplicate custom agent `{definition.name}` ignored; already loaded from {existing.source_path}.",
                    definition.source_path,
                )
            )

    catalog.agents = dict(sorted(catalog.agents.items()))
    return catalog


def validate_custom_agent_models(catalog: CustomAgentCatalog, config: AgentConfig) -> CustomAgentCatalog:
    """Remove definitions incompatible with the session's inherited model configuration."""
    validated = CustomAgentCatalog(agents=dict(catalog.agents), diagnostics=list(catalog.diagnostics))
    for name, definition in list(validated.agents.items()):
        try:
            definition.resolve_model_config(config)
        except ValueError as exc:
            validated.diagnostics.append(
                CustomAgentDiagnostic(
                    "error",
                    f"Custom agent `{name}` has an invalid model configuration: {exc}",
                    definition.source_path,
                )
            )
            del validated.agents[name]
    return validated


def _iter_definition_files(root: Path) -> Iterable[Path]:
    if not root.is_dir():
        return []
    return sorted(path for path in root.rglob("*.md") if path.is_file())


def _load_definition(
    definition_file: Path,
    scope: str,
) -> tuple[Optional[CustomAgentDefinition], list[CustomAgentDiagnostic]]:
    diagnostics: list[CustomAgentDiagnostic] = []
    try:
        if definition_file.stat().st_size > MAX_CUSTOM_AGENT_FILE_BYTES:
            raise ValueError(f"definition exceeds {MAX_CUSTOM_AGENT_FILE_BYTES // 1024} KiB")
        metadata, body = _parse_definition_file(definition_file)
        definition = _definition_from_metadata(metadata, body, definition_file, scope)
    except (OSError, UnicodeError, ValueError, yaml.YAMLError) as exc:
        return None, [CustomAgentDiagnostic("error", f"Could not load custom agent: {exc}", definition_file)]

    if definition_file.stem != definition.name:
        diagnostics.append(
            CustomAgentDiagnostic(
                "warning",
                f"Custom agent name `{definition.name}` does not match filename `{definition_file.stem}`.",
                definition_file,
            )
        )
    return definition, diagnostics


def _parse_definition_file(definition_file: Path) -> tuple[dict[str, Any], str]:
    text = definition_file.read_text(encoding="utf-8")
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        raise ValueError("missing YAML frontmatter")

    closing_index = next((index for index, line in enumerate(lines[1:], start=1) if line.strip() == "---"), None)
    if closing_index is None:
        raise ValueError("missing closing YAML frontmatter delimiter")

    metadata = yaml.safe_load("\n".join(lines[1:closing_index])) or {}
    if not isinstance(metadata, dict):
        raise ValueError("YAML frontmatter must be a mapping")
    return metadata, "\n".join(lines[closing_index + 1 :]).strip()


def _definition_from_metadata(
    metadata: dict[str, Any],
    body: str,
    source_path: Path,
    scope: str,
) -> CustomAgentDefinition:
    unknown_fields = sorted(set(metadata) - CUSTOM_AGENT_FIELDS)
    if unknown_fields:
        raise ValueError(f"unknown frontmatter field(s): {', '.join(unknown_fields)}")

    name = metadata.get("name")
    if not isinstance(name, str) or not CUSTOM_AGENT_NAME_RE.fullmatch(name):
        raise ValueError("`name` must be a lowercase kebab-case identifier of at most 64 characters")
    if name in RESERVED_CUSTOM_AGENT_NAMES:
        raise ValueError(f"`name` is reserved by the built-in `{name}` agent")

    description = metadata.get("description")
    if not isinstance(description, str) or not description.strip():
        raise ValueError("`description` must be a non-empty string")
    description = description.strip()
    if len(description) > MAX_CUSTOM_AGENT_DESCRIPTION_CHARS:
        raise ValueError(f"`description` exceeds {MAX_CUSTOM_AGENT_DESCRIPTION_CHARS} characters")
    if not body:
        raise ValueError("Markdown prompt body must not be empty")

    mode = metadata.get("mode", "build")
    if not isinstance(mode, str) or mode not in {"build", "plan", "all"}:
        raise ValueError("`mode` must be one of: build, plan, all")

    tools_value = metadata.get("tools")
    tools: Optional[tuple[str, ...]] = None
    if tools_value is not None:
        if not isinstance(tools_value, list) or any(
            not isinstance(tool, str) or not tool.strip() for tool in tools_value
        ):
            raise ValueError("`tools` must be a list of non-empty tool names")
        normalized_tools = [tool.strip() for tool in tools_value]
        if len(normalized_tools) != len(set(normalized_tools)):
            raise ValueError("`tools` must not contain duplicate names")
        tools = tuple(normalized_tools)

    model = metadata.get("model")
    if model is not None:
        if not isinstance(model, str) or "/" not in model:
            raise ValueError("`model` must use `<provider>/<model-id>` format")
        provider_value, model_id = model.split("/", 1)
        if not provider_value or not model_id:
            raise ValueError("`model` must use `<provider>/<model-id>` format")
        try:
            provider = ModelProvider(provider_value)
            get_model_specs(provider, model_id)
        except ValueError as exc:
            raise ValueError(str(exc)) from exc

    thinking_effort = metadata.get("thinking_effort")
    if thinking_effort is not None and (not isinstance(thinking_effort, str) or not thinking_effort.strip()):
        raise ValueError("`thinking_effort` must be a non-empty string")
    if isinstance(thinking_effort, str):
        thinking_effort = thinking_effort.strip().lower()
        if model is not None:
            provider_value, model_id = model.split("/", 1)
            try:
                thinking_effort = normalize_thinking_effort(ModelProvider(provider_value), model_id, thinking_effort)
            except ValueError as exc:
                raise ValueError(str(exc)) from exc

    max_iterations = metadata.get("max_iterations")
    if max_iterations is not None and (
        isinstance(max_iterations, bool) or not isinstance(max_iterations, int) or max_iterations < 1
    ):
        raise ValueError("`max_iterations` must be a positive integer")

    return CustomAgentDefinition(
        name=name,
        description=description,
        prompt=body,
        source_path=source_path.resolve(),
        scope=scope,
        mode=mode,
        tools=tools,
        model=model,
        thinking_effort=thinking_effort,
        max_iterations=max_iterations,
    )


class CustomAgent(BaseAgent):
    """A fresh subagent whose prompt and constraints come from a Markdown definition."""

    agent_name = "custom-agent"

    def __init__(
        self,
        project_path: str | Path,
        workspace_id: str,
        thread_id: str,
        connection_manager: AgentConnectionManager,
        config: AgentConfig,
        *,
        definition: CustomAgentDefinition,
        allowed_tools: Sequence[str],
        sub_agent: bool = True,
        filesystem=None,
        terminal_manager=None,
        browser_manager=None,
        langfuse_client=None,
        user_id: Optional[str] = None,
        user_email: Optional[str] = None,
        project_template_slug: Optional[str] = None,
        protected_files: Optional[List[str]] = None,
        agent_mode: Optional[AgentMode] = None,
        workspace_env_var_descriptions: Optional[Dict[str, str]] = None,
        workspace_memories: Optional[List[str]] = None,
        prompt_provider: Optional[PromptProvider] = None,
        prompt_extensions: Optional[List[PromptExtension]] = None,
        tool_extensions: Optional[List[Any]] = None,
        permission_mode: Optional[Any] = None,
        permission_callback: Optional[Any] = None,
        usage_recorder: Optional[Any] = None,
        sub_agent_recorder: Optional[Any] = None,
        hook_dispatcher: Optional[Any] = None,
        max_iterations: Optional[int] = None,
    ) -> None:
        self.definition = definition
        self.agent_name = definition.name
        resolved_model = definition.resolve_model_config(config)
        custom_config = config.model_copy(update={"long_context_config": resolved_model})

        super().__init__(
            project_path,
            workspace_id,
            thread_id,
            connection_manager,
            custom_config,
            sub_agent=sub_agent,
            filesystem=filesystem,
            terminal_manager=terminal_manager,
            browser_manager=browser_manager,
            langfuse_client=langfuse_client,
            user_id=user_id,
            user_email=user_email,
            project_template_slug=project_template_slug,
            protected_files=protected_files,
            agent_mode=agent_mode,
            workspace_env_var_descriptions=workspace_env_var_descriptions,
            workspace_memories=workspace_memories,
            prompt_provider=prompt_provider,
            prompt_extensions=prompt_extensions,
            tool_extensions=tool_extensions,
            permission_mode=permission_mode,
            permission_callback=permission_callback,
            usage_recorder=usage_recorder,
            sub_agent_recorder=sub_agent_recorder,
            hook_dispatcher=hook_dispatcher,
            max_iterations=definition.max_iterations if definition.max_iterations is not None else max_iterations,
        )

        self.tool_collection = ToolCollection(
            self.project_path,
            self.workspace_id,
            self.thread_id,
            self.connection_manager,
            self.config,
            caller=self,
            tool_config=ToolCollectionConfig(
                allowed_tools=list(allowed_tools),
                tool_exclusions=[*ToolCollection.agent_dispatch_tools, *ToolCollection.orchestration_tools],
            ),
            filesystem=self.filesystem,
            terminal_manager=self.terminal_manager,
            browser_manager=self.browser_manager,
            langfuse_client=self.langfuse_client,
            tool_extensions=self.tool_extensions,
        )
        self._initialize_system_prompt()

    def _initialize_system_prompt(self) -> None:
        context = self.build_prompt_context()
        dynamic = self.prompt_provider.render_dynamic_sections(
            AgentType.GENERAL,
            self.agent_mode,
            self.prompt_extensions,
            context,
        )
        prompt = "\n\n".join(part for part in (self.definition.prompt, dynamic) if part)
        self.system_prompt = Message(role="system", content=[TextBlock(text=prompt)])
