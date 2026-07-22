"""AgentContext: everything an agent needs, grouped by concern.

Replaces the long flat constructor signature on BaseAgent. Hosts build one
AgentContext and hand it to any agent class; the legacy keyword signature
remains supported on BaseAgent and converts to an AgentContext internally.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    # Imported for type hints only; langfuse is a heavy (~tens of MB) optional
    # dependency, so we avoid importing it at module load. The runtime only ever
    # receives a Langfuse *instance* via Telemetry.langfuse_client.
    from langfuse import Langfuse

from kolega_code.config import AgentConfig
from kolega_code.events import AgentConnectionManager
from kolega_code.hooks import NO_OP_DISPATCHER, HookDispatcher
from kolega_code.llm.client import LLMClient
from kolega_code.llm.instrumented_client import InstrumentedLLMClient
from kolega_code.permissions import PermissionMode, auto_allow_permission_callback
from .prompt_provider import AgentMode, PromptExtension, PromptProvider
from kolega_code.services.base import BrowserManager, TerminalManager
from kolega_code.services.browser import PlaywrightBrowserManager
from kolega_code.services.file_system import FileSystem, LocalFileSystem
from kolega_code.services.terminal import LocalTerminalManager


@dataclass
class WorkspaceInfo:
    """Identity and content of the workspace the agent operates in."""

    project_path: Path
    workspace_id: str
    thread_id: str
    project_template_slug: Optional[str] = None
    protected_files: List[str] = field(default_factory=list)
    env_var_descriptions: Dict[str, str] = field(default_factory=dict)
    memories: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if isinstance(self.project_path, str):
            self.project_path = Path(self.project_path)


@dataclass
class AgentServices:
    """The environment abstractions the agent works through."""

    filesystem: FileSystem
    terminal_manager: TerminalManager
    browser_manager: BrowserManager
    memory_manager: Optional[Any] = None

    @classmethod
    def local(cls, workspace: WorkspaceInfo, connection_manager: AgentConnectionManager) -> "AgentServices":
        """Default local-machine services rooted at the workspace project path."""
        return cls(
            filesystem=LocalFileSystem(root_path=workspace.project_path),
            terminal_manager=LocalTerminalManager(
                workspace.workspace_id, workspace.thread_id, connection_manager, default_workdir=workspace.project_path
            ),
            browser_manager=PlaywrightBrowserManager(),
        )


@dataclass
class Telemetry:
    """Observability and usage-recording hooks provided by the host."""

    langfuse_client: Optional["Langfuse"] = None
    user_id: Optional[str] = None
    user_email: Optional[str] = None
    usage_recorder: Optional[Any] = None
    sub_agent_recorder: Optional[Any] = None


@dataclass
class AgentContext:
    """Everything an agent needs to run, grouped by concern."""

    workspace: WorkspaceInfo
    config: AgentConfig
    connection_manager: AgentConnectionManager
    services: AgentServices
    telemetry: Telemetry = field(default_factory=Telemetry)
    agent_mode: Optional[AgentMode] = None
    prompt_provider: Optional[PromptProvider] = None
    prompt_extensions: List[PromptExtension] = field(default_factory=list)
    tool_extensions: List[Any] = field(default_factory=list)
    permission_mode: PermissionMode = PermissionMode.AUTO
    permission_callback: Any = auto_allow_permission_callback
    # Lifecycle-hook dispatcher. Defaults to a stateless no-op so hosts that do
    # not configure hooks (and every existing caller/test) are unaffected.
    hook_dispatcher: HookDispatcher = NO_OP_DISPATCHER

    def create_llm_client(self, agent_name: str) -> LLMClient:
        """Create the LLM client, instrumented when a Langfuse client is available.

        The client is bound to the provider for this agent's role (which may differ
        from the global model when a per-agent override is configured); the model id
        itself is still passed per call.
        """
        model_config = self.config.model_config_for_agent(agent_name)

        if self.telemetry.langfuse_client:
            return InstrumentedLLMClient(
                provider=model_config.provider,
                api_key=self.config.get_api_key(model_config.provider) or "",
                max_retries=model_config.rate_limits.max_retries,
                requests_per_minute=model_config.rate_limits.requests_per_minute,
                tokens_per_minute=model_config.rate_limits.tokens_per_minute,
                langfuse_client=self.telemetry.langfuse_client,
                workspace_id=self.workspace.workspace_id,
                thread_id=self.workspace.thread_id,
                agent_type=agent_name,
                environment=self.config.environment,
                user_id=self.telemetry.user_id,
                user_email=self.telemetry.user_email,
                usage_recorder=self.telemetry.usage_recorder,
                token_manager=self.config.get_chatgpt_token_manager(),
            )

        return LLMClient(
            provider=model_config.provider,
            api_key=self.config.get_api_key(model_config.provider) or "",
            max_retries=model_config.rate_limits.max_retries,
            requests_per_minute=model_config.rate_limits.requests_per_minute,
            tokens_per_minute=model_config.rate_limits.tokens_per_minute,
            token_manager=self.config.get_chatgpt_token_manager(),
        )
