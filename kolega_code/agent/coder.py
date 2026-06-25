from pathlib import Path
from typing import Any, Dict, List, Optional

from .baseagent import BaseAgent
from .common import LogMixin
from kolega_code.config import AgentConfig
from kolega_code.events import AgentConnectionManager
from kolega_code.llm.models import Message, TextBlock
from .prompt_provider import AgentType, AgentMode, PromptExtension, PromptProvider
from .tools import ToolCollection, ToolCollectionConfig
from .utils.commands import CommandProcessor


@CommandProcessor.process_commands
class CoderAgent(BaseAgent, LogMixin):
    """
    An AI coding agent that operates within a workspace to assist with programming tasks.

    The agent has access to the project filesystem and can perform coding operations
    like reading, analyzing, and modifying code files.
    """

    agent_name = "coder"

    def __init__(
        self,
        project_path: str | Path,
        workspace_id: str,
        thread_id: str,
        connection_manager: AgentConnectionManager,
        config: AgentConfig,
        sub_agent: bool = False,
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
        """
        Initialize a new CoderAgent instance.

        Args:
            project_path: File system path to the project root directory
            workspace_id: Identifier for the workspace
            thread_id: Identifier for the thread
            connection_manager: Manager for handling agent connections
            config: Agent configuration settings
            sub_agent: Whether this is a sub-agent (default: False)
            filesystem: File system implementation (optional)
            terminal_manager: Terminal manager implementation (optional)
            browser_manager: Browser manager implementation (optional)
            langfuse_client: Optional Langfuse client for LLM observability
            user_id: Optional ID of user who created this job
            user_email: Optional email of user who created this job
            project_template_slug: Optional slug of the project template being used
            protected_files: Optional list of file basenames protected from edits in vibe mode
            agent_mode: Optional agent mode (CLI, VIBE, CODE, or FIX)
            workspace_env_var_descriptions: Optional mapping of workspace environment variable descriptions
            workspace_memories: Optional list of workspace memories to inject into prompts
            prompt_provider: Optional host-configured prompt provider
            prompt_extensions: Host-provided prompt sections for app-specific context
            tool_extensions: Host-provided tool providers for app-specific tools
            usage_recorder: Optional callback for recording normalized LLM usage
            sub_agent_recorder: Optional callback for persisting sub-agent conversation state
        """
        # Call parent constructor
        super().__init__(
            project_path,
            workspace_id,
            thread_id,
            connection_manager,
            config,
            sub_agent,
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
            max_iterations=max_iterations,
        )

        # Configure tool collection with custom coder agent tools
        tool_exclusions = [
            "read_memory",
            "write_memory",
            "execute_terminal_command",
            "replace_lines",
            "get_tool_list",
            "log_error",
            "log_info",
            # Exclude task-specific dispatch tools since coder shouldn't call itself or other agents
            "dispatch_coding_agent",
        ]
        mode_value = self.agent_mode.value if isinstance(self.agent_mode, AgentMode) else self.agent_mode
        if mode_value == AgentMode.CLI.value:
            tool_exclusions.extend(["build_backend", "build_frontend"])
        if sub_agent:
            # A dispatched coder must not fan out into further sub-agents
            tool_exclusions.append("dispatch_general_agent")

        tool_config = ToolCollectionConfig(
            custom_tool_groups=["coder_agent_tools"],
            tool_exclusions=tool_exclusions,
        )

        self.tool_collection = ToolCollection(
            self.project_path,
            self.workspace_id,
            self.thread_id,
            self.connection_manager,
            self.config,
            caller=self,
            tool_config=tool_config,
            filesystem=self.filesystem,
            terminal_manager=self.terminal_manager,
            browser_manager=self.browser_manager,
            langfuse_client=self.langfuse_client,
            tool_extensions=self.tool_extensions,
        )

        self._initialize_system_prompt()

    def _initialize_system_prompt(self):
        """Initialize system prompt using PromptProvider and project overrides."""
        prompt_text = self.build_agent_system_prompt(AgentType.CODER, self.agent_mode)
        self.system_prompt = Message(role="system", content=[TextBlock(text=prompt_text)])
