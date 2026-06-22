from pathlib import Path
from typing import Any, Dict, List, Optional

from .baseagent import BaseAgent
from kolega_code.config import AgentConfig
from kolega_code.events import AgentConnectionManager
from kolega_code.llm.models import Message, TextBlock
from .prompt_provider import AgentMode, PromptExtension
from .prompts import build_planning_agent_system_prompt
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
        prompt_extensions: Optional[List[PromptExtension]] = None,
        tool_extensions: Optional[List[Any]] = None,
        permission_mode: Optional[Any] = None,
        permission_callback: Optional[Any] = None,
        usage_recorder: Optional[Any] = None,
        sub_agent_recorder: Optional[Any] = None,
        hook_dispatcher: Optional[Any] = None,
    ) -> None:
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
            permission_mode=permission_mode,
            permission_callback=permission_callback,
            usage_recorder=usage_recorder,
            sub_agent_recorder=sub_agent_recorder,
            hook_dispatcher=hook_dispatcher,
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
                read_only=True, custom_tool_groups=["planning_tools", "command_tools"]
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
        prompt = build_planning_agent_system_prompt(
            system_name=context.system_name,
            project_path=context.project_path,
            is_git_repo=context.is_git_repo,
            platform=context.platform,
            date_today=context.date_today,
            model_name=context.model_name,
        )
        if context.project_guidance:
            prompt += (
                "\n\n## Project Instructions\n\n"
                f"The project directory contains `{context.project_guidance_file}`. "
                "Treat it as local project guidance:\n\n"
                f"```markdown\n{context.project_guidance}\n```"
            )
        if context.agent_memory:
            prompt += (
                "\n\n## Agent Memory\n\n"
                f"The project directory contains `{context.agent_memory_file}`. "
                "Treat it as persistent agent memory:\n\n"
                f"```markdown\n{context.agent_memory}\n```"
            )
        if self.prompt_extensions:
            prompt += "\n\n## Additional Context\n"
            for extension in self.prompt_extensions:
                prompt += f"\n### {extension.title}\n\n{extension.markdown}\n"
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
