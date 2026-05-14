import asyncio
import os
import platform
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional

from .baseagent import BaseAgent
from .config import AgentConfig
from .connection_manager import AgentConnectionManager
from .llm.models import Message, MessageHistory, TextBlock, ToolResult
from .models.public import AgentStatus
from .prompt_provider import AgentType, PromptExtension
from .prompts import BROWSER_AGENT_SYSTEM_PROMPT
from .tools import ToolCollection


class BrowserAgent(BaseAgent):
    """
    An AI coding agent that operates within a workspace to assist with programming tasks.

    The agent has access to the project filesystem and can perform coding operations
    like reading, analyzing, and modifying code files.
    """

    agent_name = "browser-agent"

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
        Initialize a new BrowserAgent instance.

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
            agent_mode: Optional agent mode (not used for BrowserAgent)
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

        self.tool_collection = ToolCollection(
            self.project_path,
            self.workspace_id,
            self.thread_id,
            self.connection_manager,
            self.config,
            caller=self,
            browser_only=True,
            filesystem=self.filesystem,
            terminal_manager=self.terminal_manager,
            browser_manager=self.browser_manager,
            langfuse_client=self.langfuse_client,
            tool_extensions=self.tool_extensions,
        )

        self._initialize_system_prompt()

    def _initialize_system_prompt(self):
        """Initialize system prompt using PromptProvider."""
        # Generate prompt using the shared prompt provider
        prompt_text = self.prompt_provider.get_system_prompt(
            agent_type=AgentType.BROWSER,
            mode=self.agent_mode,
            template_slug=self.project_template_slug,
            prompt_extensions=self.prompt_extensions,
            context=self._build_prompt_context(),
        )
        self.system_prompt = Message(role="system", content=[TextBlock(text=prompt_text)])

    async def recap_agent_outcome(self) -> str:
        """
        Convert the conversation history into a markdown representation.

        This method retrieves the text content from the last message in the agent's
        conversation history, which represents a summary or outcome of the investigation.

        Returns:
            str: Markdown formatted conversation history containing the final investigation outcome
        """
        return self.history[-1].get_text_content()

    async def process_message_stream(
        self, message: str, attachments: List[Dict[str, Any]] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Process a message and yield multiple discrete response messages throughout processing.

        Args:
            message: The user message to process
            attachments: Optional list of attachments to include with the message

        Yields:
            Dict containing message content and metadata
        """
        self.append_user_message(message)

        stop_reason = None
        while stop_reason not in ["end_turn", "max_tokens", "stop_sequence"]:

            self._mark_last_message_for_cache()

            # print(self.history)

            try:
                # Token counting logic
                token_count = await self.count_current_context()
                print("Input token count: ")
                print(token_count)

                if token_count.input_tokens > self.model_context_length * self.history_compression_threshold:
                    await self._compress_message_history()

                    # Get new token count after compression
                    token_count = await self.count_current_context()

                    if token_count.input_tokens > self.model_context_length * self.history_compression_threshold:
                        summary_message = self.history[0]
                        self.history = MessageHistory([summary_message])

                    self._mark_last_message_for_cache()

                # Buffer for accumulated response
                current_response = ""
                current_thinking = ""
                thinking_started = False
                # Use the same UUID for each segment of the response
                response_uuid = str(uuid.uuid4())
                thinking_uuid = str(uuid.uuid4())

                # Fix history before sending to LLM to ensure valid tool call sequences
                effective = self.get_effective_history_for_llm()
                fixed_history = MessageHistory(self._fix_incomplete_tool_calls(list(effective)))

                async with await self.llm.stream(
                    system=self.system_prompt,
                    max_completion_tokens=self.model_completion_tokens,
                    temperature=self.model_default_temperature,
                    messages=fixed_history,
                    model=self.config.long_context_config.model,
                    tools=self.tool_collection.get_tool_list(),
                    thinking=self.config.long_context_config.thinking_tokens,
                ) as stream:
                    async for event in stream:
                        if event.type == "text":
                            current_response += event.text

                            # Send periodic updates as the response grows
                            if len(current_response) >= 50:
                                response_data = {
                                    "type": "response",
                                    "content": current_response,
                                    "complete": False,
                                    "uuid": response_uuid,
                                }
                                yield response_data
                                current_response = ""

                        elif event.type == "thinking" and event.thinking:
                            current_thinking += event.thinking

                            if len(current_thinking) >= 50:
                                thinking_started = True
                                yield {
                                    "type": "thinking",
                                    "content": current_thinking,
                                    "complete": False,
                                    "uuid": thinking_uuid,
                                }
                                current_thinking = ""

                message = await stream.get_final_message()
                stop_reason = message.stop_reason

                self.append_assistant_message(message)

                if thinking_started or current_thinking:
                    yield {"type": "thinking", "content": current_thinking, "complete": True, "uuid": thinking_uuid}

                # Send the final message to mark it complete.
                final_response = {
                    "type": "response",
                    "content": current_response,
                    "complete": True,
                    "uuid": response_uuid,
                }
                yield final_response

                if message.tool_calls:
                    # Process the tool calls and get the results
                    await self.log_info(f"Received {len(message.tool_calls)} tool call(s)", sender=self.agent_name)

                    try:
                        tool_responses = await self.process_tool_calls(message.tool_calls)

                        self.append_user_message(tool_responses)
                        # self.history.append(tool_response_message)
                    except Exception as ex:
                        error_message = f"Error processing tool calls: {str(ex)}"
                        await self.log_error(error_message, sender=self.agent_name)

                        # Add error responses to history
                        error_responses = []
                        for tool_call in message.tool_calls:
                            error_responses.append(
                                ToolResult(
                                    tool_use_id=tool_call.id,
                                    content=f"Failed to process tool calls: {str(ex)}",
                                    name=tool_call.name,
                                    is_error=True,
                                )
                            )

                        self.append_user_message(error_responses)
                        # self.history.append({"role": "user", "content": [er.to_anthropic() for er in error_responses]})

            except Exception as ex:
                await self.handle_llm_error(ex)

        # Log completion message
        await self.log_info("Processing complete", sender=self.agent_name)
