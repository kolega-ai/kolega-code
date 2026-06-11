from pathlib import Path
from typing import Any, Dict, List, Optional

from .baseagent import BaseAgent
from kolega_code.config import AgentConfig
from kolega_code.events import AgentConnectionManager
from kolega_code.llm.models import Message, TextBlock
from .prompt_provider import AgentMode, AgentType, PromptExtension
from .tools import ToolCollection, ToolCollectionConfig


class GeneralAgent(BaseAgent):
    """
    A general-purpose sub-agent with the full workspace toolset.

    Dispatched by a parent agent to complete one self-contained task autonomously
    (multiple GeneralAgents may run concurrently on independent tasks). It cannot
    spawn further sub-agents, and its final message is returned to the parent as
    the task report.
    """

    agent_name = "general-agent"

    def __init__(
        self,
        project_path: str | Path,
        workspace_id: str,
        thread_id: str,
        connection_manager: AgentConnectionManager,
        config: AgentConfig,
        sub_agent: bool = True,
        filesystem=None,
        terminal_manager=None,
        browser_manager=None,
        langfuse_client=None,
        user_id: Optional[str] = None,
        user_email: Optional[str] = None,
        project_template_slug: Optional[str] = None,
        protected_files: Optional[List[str]] = None,
        agent_mode: Optional["AgentMode"] = None,
        workspace_env_var_descriptions: Optional[Dict[str, str]] = None,
        workspace_memories: Optional[List[str]] = None,
        prompt_extensions: Optional[List[PromptExtension]] = None,
        tool_extensions: Optional[List[Any]] = None,
        usage_recorder: Optional[Any] = None,
        sub_agent_recorder: Optional[Any] = None,
    ) -> None:
        """
        Initialize a new GeneralAgent instance.

        Args:
            project_path: File system path to the project root directory
            workspace_id: Identifier for the workspace
            thread_id: Identifier for the thread
            connection_manager: Manager for handling agent connections
            config: Agent configuration
            sub_agent: Whether this agent is a sub-agent of another agent
            filesystem: Optional filesystem implementation
            terminal_manager: Optional terminal manager implementation
            browser_manager: Optional browser manager implementation
            langfuse_client: Optional Langfuse client for LLM observability
            user_id: Optional ID of user who created this job
            user_email: Optional email of user who created this job
            project_template_slug: Optional slug of the project template being used
            protected_files: Optional list of file basenames protected from edits in vibe mode
            agent_mode: Optional agent mode inherited from the dispatching agent
            workspace_env_var_descriptions: Optional mapping of workspace environment variable descriptions
            workspace_memories: Optional list of workspace memories to inject into prompts
            prompt_extensions: Host-provided prompt sections for app-specific context
            tool_extensions: Host-provided tool providers for app-specific tools
            usage_recorder: Optional callback for recording normalized LLM usage
            sub_agent_recorder: Optional callback for persisting sub-agent conversation state
        """
        super().__init__(
            project_path,
            workspace_id,
            thread_id,
            connection_manager,
            config,
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
            prompt_extensions=prompt_extensions,
            tool_extensions=tool_extensions,
            usage_recorder=usage_recorder,
            sub_agent_recorder=sub_agent_recorder,
        )

        # Full coder-style toolset, minus sub-agent dispatch (a dispatched agent
        # may not fan out further) and build tools in CLI mode.
        tool_exclusions = [
            "read_memory",
            "write_memory",
            "execute_terminal_command",
            "replace_lines",
            "apply_patch",
            "edit_file",
            "get_tool_list",
            "log_error",
            "log_info",
            "run_command",  # Disabled: unreliable completion detection, use run_command_tracked instead
            # Recursion guard: a general sub-agent may not spawn further sub-agents
            *ToolCollection.agent_dispatch_tools,
        ]
        mode_value = self.agent_mode.value if isinstance(self.agent_mode, AgentMode) else self.agent_mode
        if mode_value == AgentMode.CLI.value:
            tool_exclusions.extend(["build_backend", "build_frontend"])

        tool_config = ToolCollectionConfig(tool_exclusions=tool_exclusions)

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
        """Initialize system prompt using PromptProvider."""
        prompt_text = self.prompt_provider.get_system_prompt(
            agent_type=AgentType.GENERAL,
            mode=self.agent_mode,
            template_slug=self.project_template_slug,
            prompt_extensions=self.prompt_extensions,
            context=self.build_prompt_context(),
        )
        self.system_prompt = Message(role="system", content=[TextBlock(text=prompt_text)])
