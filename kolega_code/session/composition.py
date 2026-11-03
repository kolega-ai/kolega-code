"""UI-independent composition of a session's top-level agent."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from kolega_code.agent import AgentConfig, CoderAgent, PlanningAgent, PromptExtension, ToolExtension
from kolega_code.agent.baseagent import BaseAgent
from kolega_code.agent.custom_agents import (
    CustomAgentCatalog,
    discover_custom_agents,
    validate_custom_agent_models,
)
from kolega_code.agent.prompt_provider import AgentMode
from kolega_code.cli.skills import (
    SkillCatalog,
    build_skill_prompt_extension,
    build_skill_tool_extension,
    context_window_tokens_for_skill_budget,
    discover_skills,
)
from kolega_code.extensions import (
    ExtensionSelection,
    KolegaExtensionBundle,
    KolegaExtensionHost,
    KolegaExtensionLoadError,
    create_extension_bundle,
)
from kolega_code.llm.models import Message
from kolega_code.mcp.config import LoadedMCPConfig
from kolega_code.mcp.tools import build_mcp_tool_extension

if TYPE_CHECKING:
    from kolega_code.cli.session_journal import SessionRecorder
    from kolega_code.events import AgentConnectionManager
    from kolega_code.hooks import HookDispatcher
    from kolega_code.llm.ledger import UsageLedger
    from kolega_code.memory.manager import ProjectMemoryManager
    from kolega_code.permissions import PermissionDecision, PermissionMode, PermissionRequest
    from kolega_code.services.base import BrowserManager


class MCPBuilder(Protocol):
    def __call__(
        self,
        project_path: Path,
        state_dir: Path,
        *,
        project_trusted: bool,
        loaded_config: LoadedMCPConfig | None = None,
    ) -> ToolExtension | None: ...


@dataclass
class HostExtensions:
    prompts: list[PromptExtension]
    tools: list[ToolExtension]
    scratchpad: PromptExtension | None = None


@dataclass
class CompositionHost:
    """Host-specific capabilities and lifecycle ownership for composition."""

    extensions: Callable[[], HostExtensions]
    questions: Callable[[], tuple[PromptExtension, ToolExtension]]
    additional_prompts: Callable[[], list[PromptExtension]]
    history: Callable[[], Iterable[Message]]
    set_catalogs: Callable[[SkillCatalog, CustomAgentCatalog], None]
    mcp_trusted: Callable[[], bool]
    mcp_diagnostic: Callable[[str], None]
    mcp_names: Callable[[LoadedMCPConfig], None]
    own_bundle: Callable[[KolegaExtensionBundle], None]
    hook_dispatcher: Callable[[], HookDispatcher]
    mcp_builder: MCPBuilder = build_mcp_tool_extension


@dataclass
class CompositionRequest:
    config: AgentConfig
    project_path: Path
    active_project_path: Path
    state_dir: Path
    workspace_id: str
    thread_id: str
    interaction_mode: str
    agent_mode: AgentMode
    agent_class: type[BaseAgent]
    connection_manager: AgentConnectionManager
    browser_manager: BrowserManager
    memory_manager: ProjectMemoryManager
    permission_mode: PermissionMode
    permission_callback: Callable[[PermissionRequest], Awaitable[PermissionDecision]]
    session_recorder: SessionRecorder | None
    usage_ledger: UsageLedger
    extension_selection: ExtensionSelection | None


@dataclass
class PreparedComposition:
    prompts: list[PromptExtension]
    tools: list[ToolExtension]
    custom_agents: CustomAgentCatalog
    scratchpad: PromptExtension | None


def select_agent_class(
    interaction_mode: str,
    *,
    build_agent_class: type[BaseAgent] = CoderAgent,
    planning_agent_class: type[BaseAgent] = PlanningAgent,
) -> type[BaseAgent]:
    return planning_agent_class if interaction_mode == "plan" else build_agent_class


def prepare_composition(request: CompositionRequest, host: CompositionHost) -> PreparedComposition:
    skills = discover_skills(request.active_project_path) if request.config.skills_enabled else SkillCatalog()
    custom_agents = validate_custom_agent_models(
        discover_custom_agents(request.active_project_path, request.state_dir),
        request.config,
    ).for_mode(request.interaction_mode)
    host.set_catalogs(skills, custom_agents)
    extensions = host.extensions()
    prompts, tools = extensions.prompts, extensions.tools
    if request.config.skills_enabled:
        skill_prompt = build_skill_prompt_extension(
            skills,
            context_window_tokens=context_window_tokens_for_skill_budget(
                request.config, getattr(request.agent_class, "agent_name", None)
            ),
        )
        skill_tool = build_skill_tool_extension(skills, host.history)
        if skill_prompt is not None:
            prompts.append(skill_prompt)
        if skill_tool is not None:
            tools.append(skill_tool)
    question_prompt, question_tool = host.questions()
    prompts.append(question_prompt)
    tools.append(question_tool)

    mcp_config = getattr(request.config, "mcp_config", None)
    if mcp_config is not None:
        for diagnostic in getattr(mcp_config, "diagnostics", []) or []:
            host.mcp_diagnostic(diagnostic)
    mcp_extension = host.mcp_builder(
        request.active_project_path,
        request.state_dir,
        project_trusted=host.mcp_trusted(),
        loaded_config=mcp_config if request.active_project_path == request.project_path else None,
    )
    if mcp_extension is not None:
        tools.append(mcp_extension)
        if mcp_config is not None and request.active_project_path == request.project_path:
            host.mcp_names(mcp_config)
    prompts.extend(host.additional_prompts())
    return PreparedComposition(prompts, tools, custom_agents, extensions.scratchpad)


def construct_agent(
    request: CompositionRequest, host: CompositionHost, prepared: PreparedComposition
) -> tuple[BaseAgent, KolegaExtensionBundle | None]:
    bundle = None
    prompts = list(prepared.prompts)
    tools = list(prepared.tools)
    if request.extension_selection is not None:
        extension_host = KolegaExtensionHost(
            project_path=request.active_project_path,
            workspace_id=request.workspace_id,
            thread_id=request.thread_id,
            config=request.config,
            agent_mode=request.agent_mode,
        )
        bundle = create_extension_bundle(request.extension_selection, extension_host)
        host.own_bundle(bundle)
        prompts.extend(bundle.prompt_extensions)
        tools.extend(bundle.tool_extensions)
    try:
        agent = request.agent_class(
            project_path=request.active_project_path,
            workspace_id=request.workspace_id,
            thread_id=request.thread_id,
            connection_manager=request.connection_manager,
            config=request.config,
            browser_manager=request.browser_manager,
            agent_mode=request.agent_mode,
            prompt_extensions=prompts,
            tool_extensions=tools,
            memory_manager=request.memory_manager,
            permission_mode=request.permission_mode,
            permission_callback=request.permission_callback,
            session_recorder=request.session_recorder,
            hook_dispatcher=host.hook_dispatcher(),
            custom_agent_catalog=prepared.custom_agents,
            usage_ledger=request.usage_ledger,
            llm_trace_sink=bundle.llm_trace_sink if bundle else None,
        )
        return agent, bundle
    except ValueError as exc:
        if bundle is not None:
            raise KolegaExtensionLoadError(str(exc)) from exc
        raise
