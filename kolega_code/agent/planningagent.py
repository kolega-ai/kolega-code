from pathlib import Path
from typing import Any, Dict, List, Optional

from .baseagent import BaseAgent
from kolega_code.config import AgentConfig
from kolega_code.events import AgentConnectionManager
from kolega_code.llm.models import Message, TextBlock
from .prompt_provider import AgentMode, AgentType, PromptExtension, PromptProvider
from .tools import ToolCollection, ToolCollectionConfig, ToolExtension
from .utils.commands import CommandProcessor


@CommandProcessor.process_commands
class PlanningAgent(BaseAgent):
    """Standalone planning agent with read-only repository tools and plan-specific state."""

    agent_name = "planning-agent"
    completion_log_message = "Planning complete"

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
        session_recorder: Optional[Any] = None,
        hook_dispatcher: Optional[Any] = None,
        max_iterations: Optional[int] = None,
        custom_agent_catalog: Optional[Any] = None,
        memory_manager: Optional[Any] = None,
    ) -> None:
        if custom_agent_catalog is not None:
            custom_agent_catalog = custom_agent_catalog.for_mode("plan")

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
            prompt_provider=prompt_provider,
            prompt_extensions=prompt_extensions,
            tool_extensions=tool_extensions,
            permission_mode=permission_mode,
            permission_callback=permission_callback,
            usage_recorder=usage_recorder,
            sub_agent_recorder=sub_agent_recorder,
            session_recorder=session_recorder,
            hook_dispatcher=hook_dispatcher,
            max_iterations=max_iterations,
            custom_agent_catalog=custom_agent_catalog,
            memory_manager=memory_manager,
        )

        self._completed_plan: Optional[str] = None

        planning_tools = ToolExtension(
            name="planning-agent-tools",
            tools={
                "write_plan": self.write_plan,
            },
            tool_groups={"planning_tools": ["write_plan"]},
            # write_plan belongs to the top-level planning agent only; sub-agents
            # (e.g. gigacode investigation agents in plan mode) must not capture plans.
            propagate_to_sub_agents=False,
        )
        self.tool_extensions = [*self.tool_extensions, planning_tools]
        self.tool_collection = ToolCollection(
            self.project_path,
            self.workspace_id,
            self.thread_id,
            self.connection_manager,
            self.config,
            caller=self,
            tool_config=ToolCollectionConfig(
                read_only=True,
                include_memory_tools=True,
                memory_write_access=True,
                custom_tool_groups=["planning_tools", "command_tools", "custom_agent_tools"],
            ),
            filesystem=self.filesystem,
            terminal_manager=self.terminal_manager,
            browser_manager=self.browser_manager,
            langfuse_client=self.langfuse_client,
            tool_extensions=self.tool_extensions,
        )

        self._initialize_system_prompt()

    def _initialize_system_prompt(self) -> None:
        prompt = self.build_agent_system_prompt(AgentType.PLANNING, self.agent_mode)
        self.system_prompt = Message(role="system", content=[TextBlock(text=prompt)])

    async def write_plan(self, plan_markdown: str) -> str:
        """
        Submit the final implementation plan.

        Call this only when the plan is complete enough for a build agent to implement without making additional
        product or architecture decisions.

        Args:
            plan_markdown: The complete final implementation plan in markdown.

        Returns:
            A confirmation that the plan was captured.
        """
        self._completed_plan = plan_markdown.strip()
        return "Plan captured."

    def consume_completed_plan(self) -> Optional[str]:
        plan = self._completed_plan
        self._completed_plan = None
        return plan

    async def recap_agent_outcome(self) -> str:
        if self._completed_plan:
            return self._completed_plan
        return self.history[-1].get_text_content()

    def should_stop_after_tools(self) -> bool:
        # End the loop as soon as write_plan has captured a complete plan.
        return self._completed_plan is not None
