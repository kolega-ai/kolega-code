"""AgentContext: everything an agent needs, grouped by concern.

Replaces the long flat constructor signature on BaseAgent. Hosts build one
AgentContext and hand it to any agent class; the legacy keyword signature
remains supported on BaseAgent and converts to an AgentContext internally.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from langfuse import Langfuse

from .config import AgentConfig
from .connection_manager import AgentConnectionManager
from .llm.client import LLMClient
from .llm.instrumented_client import InstrumentedLLMClient
from .prompt_provider import AgentMode, PromptExtension
from .services.base import BrowserManager, TerminalManager
from .services.browser import PlaywrightBrowserManager
from .services.file_system import FileSystem, LocalFileSystem
from .services.terminal import LocalTerminalManager


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

    @classmethod
    def local(cls, workspace: WorkspaceInfo, connection_manager: AgentConnectionManager) -> "AgentServices":
        """Default local-machine services rooted at the workspace project path."""
        return cls(
            filesystem=LocalFileSystem(root_path=workspace.project_path),
            terminal_manager=LocalTerminalManager(workspace.workspace_id, workspace.thread_id, connection_manager),
            browser_manager=PlaywrightBrowserManager(),
        )


@dataclass
class Telemetry:
    """Observability and usage-recording hooks provided by the host."""

    langfuse_client: Optional[Langfuse] = None
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
    prompt_extensions: List[PromptExtension] = field(default_factory=list)
    tool_extensions: List[Any] = field(default_factory=list)

    def create_llm_client(self, agent_name: str) -> LLMClient:
        """Create the LLM client, instrumented when a Langfuse client is available."""
        model_config = self.config.long_context_config

        if self.telemetry.langfuse_client:
            return InstrumentedLLMClient(
                provider=model_config.provider,
                api_key=self.config.get_api_key(model_config.provider),
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
            )

        return LLMClient(
            provider=model_config.provider,
            api_key=self.config.get_api_key(model_config.provider),
            max_retries=model_config.rate_limits.max_retries,
            requests_per_minute=model_config.rate_limits.requests_per_minute,
            tokens_per_minute=model_config.rate_limits.tokens_per_minute,
        )
