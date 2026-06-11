import asyncio
import contextvars
import os
import re
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional

from .common import LogMixin
from .config import AgentConfig, ModelProvider
from .connection_manager import AgentConnectionManager
from .llm.client import LLMClient
from .llm.instrumented_client import InstrumentedLLMClient
from .llm.exceptions import (
    LLMContextWindowExceededError,
    LLMError,
    LLMInternalServerError,
    LLMRateLimitError,
    map_to_llm_error,
)
from .llm.models import ImageBlock, Message, MessageHistory, TextBlock, ToolCall, ToolResult
from .llm.providers.models import TokenCount
from .llm.specs import get_model_specs
from .models.public import AgentEvent, AgentStatus
from .prompt_provider import PromptProvider, AgentType, AgentMode, PromptContext, PromptExtension
from .prompts import (
    COMPRESSION_PROMPT,
    COMPRESSION_SUMMARY_SYSTEM_PROMPT,
    COMPRESSION_SUMMARY_USER_PROMPT_TEMPLATE,
)
from .services.file_system import FileSystem, LocalFileSystem
from .services.base import TerminalManager, BrowserManager
from .services.terminal import LocalTerminalManager
from .services.browser import PlaywrightBrowserManager
from .tools import ToolCollection
from .utils.commands import CommandProcessor
from langfuse import Langfuse


class BaseAgent(LogMixin):
    """
    Base class for all AI agents in the system.

    Provides common functionality for agent operations including history management,
    logging, and communication with the LLM service.
    """

    agent_name = "base-agent"  # you should never see this
    history_compression_threshold = 0.8
    # Cap on concurrently executing tool calls within one batch (each dispatched
    # sub-agent runs its own multi-turn LLM loop, so an unbounded fan-out would
    # multiply token spend and shared-resource pressure).
    PARALLEL_TOOL_LIMIT = 8
    long_content_tool_calls = ["create_file", "replace_entire_file"]
    max_tool_result_chars_in_history = 100_000
    skill_content_pattern = re.compile(r'<skill_content name="[^"]+">')
    deepseek_image_unsupported_message = (
        "DeepSeek V4 Pro does not support image input via the DeepSeek API. "
        "Remove the image or switch to a vision-capable model for this request."
    )

    def __init__(
        self,
        project_path: str | Path,
        workspace_id: str,
        thread_id: str,
        connection_manager: AgentConnectionManager,
        config: AgentConfig,
        sub_agent: bool = False,
        filesystem: Optional[FileSystem] = None,
        terminal_manager: Optional[TerminalManager] = None,
        browser_manager: Optional[BrowserManager] = None,
        langfuse_client: Optional[Langfuse] = None,
        user_id: Optional[str] = None,
        user_email: Optional[str] = None,
        project_template_slug: Optional[str] = None,
        protected_files: Optional[List[str]] = None,
        agent_mode: Optional[AgentMode] = None,
        workspace_env_var_descriptions: Optional[Dict[str, str]] = None,
        workspace_memories: Optional[List[str]] = None,
        prompt_extensions: Optional[List[PromptExtension]] = None,
        tool_extensions: Optional[List[Any]] = None,
        usage_recorder: Optional[Any] = None,
        sub_agent_recorder: Optional[Any] = None,
    ) -> None:
        """
        Initialize a new BaseAgent instance.

        Args:
            project_path: File system path to the project root directory
            workspace_id: Unique identifier for the workspace
            thread_id: Unique identifier for the thread
            connection_manager: Connection manager for agent communication
            config: Agent configuration
            sub_agent: Whether this is a sub-agent
            filesystem: Optional filesystem implementation. If None, creates LocalFileSystem with project_path as root
            terminal_manager: Optional terminal manager implementation. If None, creates LocalTerminalManager
            browser_manager: Optional browser manager implementation. If None, creates PlaywrightBrowserManager
            langfuse_client: Optional Langfuse client for LLM observability
            user_id: Optional ID of user who created this job
            user_email: Optional email of user who created this job
            project_template_slug: Optional slug of the project template being used
            protected_files: Optional list of file basenames protected from edits in vibe mode
            agent_mode: Optional agent mode (e.g., AgentMode.VIBE or AgentMode.CODE for CoderAgent)
            workspace_env_var_descriptions: Optional mapping of workspace env var names to descriptions
            workspace_memories: Optional list of workspace memories to inject into prompts
            prompt_extensions: Host-provided prompt sections for app-specific context
            tool_extensions: Host-provided tool providers for app-specific tools
            usage_recorder: Optional callback for recording normalized LLM usage
            sub_agent_recorder: Optional callback for persisting sub-agent conversation state
        """
        # Convert string path to Path object if needed
        self.project_path = Path(project_path) if isinstance(project_path, str) else project_path

        # Create filesystem instance if not provided
        if filesystem is None:
            self.filesystem = LocalFileSystem(root_path=self.project_path)
        else:
            self.filesystem = filesystem

        # Create terminal manager instance if not provided
        if terminal_manager is None:
            self.terminal_manager = LocalTerminalManager(workspace_id, thread_id, connection_manager)
        else:
            self.terminal_manager = terminal_manager

        # Create browser manager instance if not provided
        if browser_manager is None:
            self.browser_manager = PlaywrightBrowserManager()
        else:
            self.browser_manager = browser_manager

        # Validate that the project path exists and is a directory using the filesystem
        if not self.filesystem.exists("."):
            raise ValueError(f"Project path does not exist: {self.project_path}")
        if not self.filesystem.is_dir("."):
            raise ValueError(f"Project path is not a directory: {self.project_path}")

        self.workspace_id = workspace_id
        self.thread_id = thread_id
        self.connection_manager = connection_manager
        self.config = config
        self.available_ports = "9001-9999"
        self.langfuse_client = langfuse_client
        self.user_id = user_id
        self.user_email = user_email

        # Store prompt-related parameters
        self.project_template_slug = project_template_slug
        self.protected_files = protected_files or []
        self.agent_mode = agent_mode
        self.workspace_env_var_descriptions = workspace_env_var_descriptions or {}
        self.workspace_memories = workspace_memories or []
        self.prompt_extensions = prompt_extensions or []
        self.tool_extensions = tool_extensions or []
        self.usage_recorder = usage_recorder
        self.sub_agent_recorder = sub_agent_recorder

        # Initialize PromptProvider
        self.prompt_provider = PromptProvider()

        self.history = MessageHistory()

        # Compression marker: index of the last message before a summary was appended
        self.last_compression_index = None

        model_specs = get_model_specs(self.config.long_context_config.provider, self.config.long_context_config.model)
        self.model_context_length = model_specs["context_length"]
        self.model_completion_tokens = model_specs["max_completion_tokens"]
        self.model_default_temperature = model_specs.get("default_temperature", 1.0)

        # Create LLM client - use instrumented version if Langfuse is available
        if langfuse_client:
            self.llm = InstrumentedLLMClient(
                provider=self.config.long_context_config.provider,
                api_key=self.config.get_api_key(self.config.long_context_config.provider),
                max_retries=self.config.long_context_config.rate_limits.max_retries,
                requests_per_minute=self.config.long_context_config.rate_limits.requests_per_minute,
                tokens_per_minute=self.config.long_context_config.rate_limits.tokens_per_minute,
                langfuse_client=langfuse_client,
                workspace_id=workspace_id,
                thread_id=thread_id,
                agent_type=self.agent_name,
                environment=config.environment,
                user_id=user_id,
                user_email=user_email,
                usage_recorder=usage_recorder,
            )
        else:
            self.llm = LLMClient(
                provider=self.config.long_context_config.provider,
                api_key=self.config.get_api_key(self.config.long_context_config.provider),
                max_retries=self.config.long_context_config.rate_limits.max_retries,
                requests_per_minute=self.config.long_context_config.rate_limits.requests_per_minute,
                tokens_per_minute=self.config.long_context_config.rate_limits.tokens_per_minute,
            )

        # Tool collection must be initialized by subclass with appropriate configuration
        # (e.g., read_only, browser_only, custom tool_config, etc.)
        self.tool_collection = None

        self.command_processor = CommandProcessor(self)

        self.sub_agent = sub_agent
        # Per-instance ContextVars so concurrent tool executions (asyncio.gather in
        # process_tool_calls) each see their own current tool call IDs. Instance-level
        # (rather than module-level) vars also keep a nested sub-agent, which executes
        # tools within the same asyncio task as the parent's dispatch call, from
        # clobbering the parent's values.
        self._current_tool_call_id_var = contextvars.ContextVar("current_tool_call_id", default=None)
        self._current_tool_execution_id_var = contextvars.ContextVar("current_tool_execution_id", default=None)
        self._current_provider_tool_call_id_var = contextvars.ContextVar("current_provider_tool_call_id", default=None)
        self.parent_tool_call_id = None  # Parent tool call ID when running as sub-agent
        self.conversation_id = None  # Sub-agent conversation ID
        self.sub_agent_context = None  # Dispatch metadata (agent_id, task) set by AgentTool

    @property
    def current_tool_call_id(self):
        """Internal execution ID for UI and sub-agent records (task-local)."""
        return self._current_tool_call_id_var.get()

    @current_tool_call_id.setter
    def current_tool_call_id(self, value):
        self._current_tool_call_id_var.set(value)

    @property
    def current_tool_execution_id(self):
        """App-unique tool execution ID for the tool call currently running (task-local)."""
        return self._current_tool_execution_id_var.get()

    @current_tool_execution_id.setter
    def current_tool_execution_id(self, value):
        self._current_tool_execution_id_var.set(value)

    @property
    def current_provider_tool_call_id(self):
        """Provider-issued tool call ID for the tool call currently running (task-local)."""
        return self._current_provider_tool_call_id_var.get()

    @current_provider_tool_call_id.setter
    def current_provider_tool_call_id(self, value):
        self._current_provider_tool_call_id_var.set(value)

    def build_prompt_context(self) -> PromptContext:
        """Build PromptContext from agent state."""
        import platform

        # Check if it's a git repository
        is_git_repo = self.filesystem.exists(".git") and self.filesystem.is_dir(".git")

        # Load KOLEGA.md content if it exists
        kolega_md_content = ""
        if self.filesystem.exists("KOLEGA.md"):
            try:
                kolega_md_content = self.filesystem.read("KOLEGA.md")
            except Exception:
                # If there's an error reading the file, use empty string
                kolega_md_content = ""

        return PromptContext(
            system_name=os.getenv("KOLEGA_CODE_SYSTEM_NAME", "Kolega Code"),
            project_path=str(self.project_path),
            is_git_repo=is_git_repo,
            platform=platform.system(),
            date_today=datetime.now().strftime("%Y-%m-%d"),
            model_name=self.config.long_context_config.model,
            available_ports=self.available_ports,
            kolega_md=kolega_md_content,
            workspace_id=self.workspace_id,
            workspace_environment_variables=self.workspace_env_var_descriptions,
            memories=self.workspace_memories,
        )

    def _unsupported_attachment_message(self, attachments: Optional[List[Dict[str, Any]]]) -> Optional[str]:
        provider = getattr(
            self.config.long_context_config.provider,
            "value",
            self.config.long_context_config.provider,
        )
        if provider != ModelProvider.DEEPSEEK.value:
            return None

        if any(attachment.get("type") == "image" for attachment in attachments or []):
            return self.deepseek_image_unsupported_message

        return None

    def _attachment_blocks(self, attachments: Optional[List[Dict[str, Any]]]) -> List[Any]:
        """Convert attachment payloads into content blocks for a user message."""
        blocks: List[Any] = []
        for attachment in attachments or []:
            attachment_type = attachment.get("type")
            if attachment_type == "image":
                blocks.append(
                    ImageBlock(
                        image_type="base64",
                        media_type=attachment.get("media_type", "image/png"),
                        data=attachment["data"],
                    )
                )
            elif attachment_type == "file":
                path = attachment.get("path", "")
                content = attachment.get("content", "")
                blocks.append(TextBlock(text=f'<attached-file path="{path}">\n{content}\n</attached-file>'))
        return blocks

    async def process_message_stream(
        self, message: str, attachments: List[Dict[str, Any]] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        raise NotImplementedError

    async def count_current_context(self) -> TokenCount:
        self._sanitize_oversized_tool_results()
        # Fix history before counting to get accurate count for what LLM will see
        effective = self.get_effective_history_for_llm()
        fixed_history = MessageHistory(self.fix_incomplete_tool_calls(list(effective)))
        token_count = await self.llm.count_tokens(
            system=self.system_prompt,
            messages=fixed_history,
            model=self.config.long_context_config.model,
            tools=self.tool_collection.get_tool_list(),
        )

        # Send context update event
        await self._send_context_update(token_count)

        return token_count

    async def _send_context_update(self, token_count: TokenCount) -> None:
        """Send an event to update the UI about current context usage."""
        usage_percentage = (token_count.input_tokens / self.model_context_length) * 100

        # Determine alert level based on usage
        alert_level = "normal"
        message = None

        if usage_percentage >= 60:
            alert_level = "info"
            message = (
                "Longer threads consume more credits. "
                f"Contents will be compressed automatically at {self.history_compression_threshold * 100:.0f}%. "
                "You can start fresh by clicking \"New Thread\" in the sidebar."
            )

        event = AgentEvent(
            event_type="llm_context_update",
            sender=self.agent_name,
            content={
                "input_tokens": token_count.input_tokens,
                "max_tokens": self.model_context_length,
                "usage_percentage": round(usage_percentage, 1),
                "alert_level": alert_level,
                "message": message,
                "compression_threshold": self.history_compression_threshold * 100,  # Convert to percentage
                "will_compress_at": int(self.model_context_length * self.history_compression_threshold),
            },
        )
        await self.connection_manager.broadcast_event(event, self.workspace_id, self.thread_id)

    def dump_message_history(self) -> List[Dict[str, Any]]:
        """Serializes the message history into a list of dictionaries using custom methods."""
        return [message.to_dict() for message in self.history]

    def restore_message_history(self, serialized_history: List[Dict[str, Any]]) -> None:
        """Restores the message history from a list of dictionaries using custom methods."""
        parsed_messages = [Message.from_dict(item) for item in serialized_history]
        # Keep history authentic - no fixing here
        self.history = MessageHistory(parsed_messages)
        self._sanitize_oversized_tool_results()

    def _sanitize_oversized_tool_results(self) -> int:
        sanitized_count = 0
        for message in self.history:
            if not isinstance(message.content, list):
                continue

            for block in message.content:
                if not isinstance(block, ToolResult) or not isinstance(block.content, str):
                    continue

                content_length = len(block.content)
                if content_length <= self.max_tool_result_chars_in_history:
                    continue

                block.content = (
                    f"[Tool result omitted from history because it was {content_length:,} characters, "
                    f"exceeding the {self.max_tool_result_chars_in_history:,} character safety cap. "
                    f"Re-run `{block.name}` with narrower inputs if the content is still needed.]"
                )
                sanitized_count += 1

        return sanitized_count

    def _is_history_valid_for_anthropic(self, messages: List[Message] = None) -> bool:
        """
        Check if the message history is valid for Anthropic API.
        Every tool_use block must be followed by a tool_result block.

        Args:
            messages: Optional list of messages to validate. If None, uses self.history

        Returns:
            bool: True if history is valid, False otherwise
        """
        if messages is None:
            messages = list(self.history)

        for i, message in enumerate(messages):
            if message.role == "assistant" and isinstance(message.content, list):
                # Check if this message contains tool calls
                tool_calls = [block for block in message.content if isinstance(block, ToolCall)]

                if tool_calls:
                    # Check if next message exists and has tool results
                    if i + 1 >= len(messages):
                        return False  # No next message

                    next_message = messages[i + 1]
                    if next_message.role != "user":
                        return False  # Next message should be user role

                    if not isinstance(next_message.content, list):
                        return False  # Should contain list of tool results

                    # Check if all tool calls have corresponding results
                    tool_call_ids = {call.id for call in tool_calls}
                    tool_result_ids = {
                        block.tool_use_id for block in next_message.content if isinstance(block, ToolResult)
                    }

                    if not tool_call_ids.issubset(tool_result_ids):
                        return False  # Missing tool results

        return True

    def fix_incomplete_tool_calls(self, messages: List[Message]) -> List[Message]:
        """
        Fix incomplete tool call sequences by adding placeholder tool_result blocks
        for any orphaned tool_use blocks.

        Args:
            messages: List of messages to validate and fix

        Returns:
            List[Message]: Fixed messages with placeholder tool results added where needed
        """
        if not messages:
            return messages

        fixed_messages = []
        i = 0
        processed_indices = set()  # Track which messages we've already processed

        while i < len(messages):
            if i in processed_indices:
                i += 1
                continue

            current_message = messages[i]

            # Check if this message contains tool calls
            if current_message.role == "assistant" and isinstance(current_message.content, list):
                tool_calls = [block for block in current_message.content if isinstance(block, ToolCall)]

                if tool_calls:
                    # Add the assistant message with tool calls
                    fixed_messages.append(current_message)
                    processed_indices.add(i)

                    # Collect all tool results from the entire remaining conversation
                    tool_call_ids = {call.id for call in tool_calls}
                    all_tool_results = {}
                    other_content_blocks = []  # Non-tool-result content from the next user message

                    # First, check the immediately following message (expected position)
                    next_user_message = None
                    if i + 1 < len(messages) and messages[i + 1].role == "user":
                        next_user_message = messages[i + 1]
                        if isinstance(next_user_message.content, list):
                            for block in next_user_message.content:
                                if isinstance(block, ToolResult) and block.tool_use_id in tool_call_ids:
                                    all_tool_results[block.tool_use_id] = block
                                else:
                                    other_content_blocks.append(block)

                            # If we found some results here, mark this message as processed
                            if all_tool_results:
                                processed_indices.add(i + 1)

                    # Search the entire remaining conversation for any missing tool results
                    missing_ids = tool_call_ids - set(all_tool_results.keys())
                    if missing_ids:
                        for j in range(i + 1, len(messages)):
                            if j in processed_indices:
                                continue

                            msg = messages[j]
                            if msg.role == "user" and isinstance(msg.content, list):
                                remaining_content = []
                                found_any = False

                                for block in msg.content:
                                    if isinstance(block, ToolResult) and block.tool_use_id in missing_ids:
                                        print(
                                            f"Warning: Found tool result {block.tool_use_id} at position {j} instead of expected position {i+1}"
                                        )
                                        all_tool_results[block.tool_use_id] = block
                                        missing_ids.remove(block.tool_use_id)
                                        found_any = True
                                    else:
                                        remaining_content.append(block)

                                # If we found tool results in this message, we need to handle it
                                if found_any:
                                    if remaining_content:
                                        # Message has other content - keep it but remove tool results
                                        updated_msg = Message(
                                            role=msg.role, content=remaining_content, stop_reason=msg.stop_reason
                                        )
                                        messages[j] = updated_msg
                                    else:
                                        # Message only had tool results - mark for skipping
                                        processed_indices.add(j)

                    # Create the complete tool results list in the correct order
                    complete_tool_results = []
                    for tool_call in tool_calls:
                        if tool_call.id in all_tool_results:
                            complete_tool_results.append(all_tool_results[tool_call.id])
                        else:
                            # Add placeholder for truly missing results
                            print(f"Adding placeholder result for missing tool call: {tool_call.id}")
                            complete_tool_results.append(
                                ToolResult(
                                    tool_use_id=tool_call.id,
                                    content="Operation was interrupted. Please retry if needed.",
                                    name=tool_call.name,
                                    is_error=True,
                                )
                            )

                    # Create the user message with all tool results
                    # Include any other content that was in the original next user message
                    all_content = complete_tool_results + other_content_blocks
                    if all_content:  # Only add if there's content
                        complete_user_message = Message(
                            role="user",
                            content=all_content,
                            stop_reason=next_user_message.stop_reason if next_user_message else None,
                        )
                        fixed_messages.append(complete_user_message)

                    i += 1
                else:
                    # No tool calls, just add the message normally
                    fixed_messages.append(current_message)
                    processed_indices.add(i)
                    i += 1
            else:
                # Not an assistant message with tool calls
                # Skip if already processed (was a tool result message we moved)
                if i not in processed_indices:
                    fixed_messages.append(current_message)
                i += 1

        return fixed_messages

    def append_user_message(self, content) -> None:
        """
        Safely append a user message to history, fixing any incomplete tool calls first.

        Args:
            content: Either a string (converted to TextBlock) or list of ContentBlocks
        """
        # Convert string to ContentBlock list if needed
        if isinstance(content, str):
            content_blocks = [TextBlock(text=content)]
        elif isinstance(content, list):
            content_blocks = content
        else:
            # Handle single ContentBlock
            content_blocks = [content]

        # Check if we're providing tool results for pending tool calls
        incoming_tool_result_ids = set()
        if isinstance(content_blocks, list):
            for block in content_blocks:
                if isinstance(block, ToolResult):
                    incoming_tool_result_ids.add(block.tool_use_id)

        # Check what tool calls are pending
        pending_tool_call_ids = set()
        if self._needs_tool_call_fix():
            last_message = self.history[-1]
            if last_message.role == "assistant" and isinstance(last_message.content, list):
                for block in last_message.content:
                    if isinstance(block, ToolCall):
                        pending_tool_call_ids.add(block.id)

        # Only run fix if we have pending tool calls that aren't being provided
        missing_tool_call_ids = pending_tool_call_ids - incoming_tool_result_ids
        if missing_tool_call_ids:
            # We have tool calls without corresponding results in the incoming content
            # Don't fix here - only fix when sending to LLM
            pass

        # Check for tool results in the new content
        if isinstance(content_blocks, list):
            new_tool_results = {}
            other_blocks = []

            for block in content_blocks:
                if isinstance(block, ToolResult):
                    new_tool_results[block.tool_use_id] = block
                else:
                    other_blocks.append(block)

            if new_tool_results:
                # Find and update any existing tool results with the same IDs
                for i, msg in enumerate(self.history):
                    if msg.role == "user" and isinstance(msg.content, list):
                        updated_content = []
                        replaced_any = False

                        for block in msg.content:
                            if isinstance(block, ToolResult) and block.tool_use_id in new_tool_results:
                                new_result = new_tool_results[block.tool_use_id]
                                # Replace if: old is dummy error OR new is success and old is error
                                should_replace = (block.is_error and "Operation was interrupted" in block.content) or (
                                    not new_result.is_error and block.is_error
                                )

                                if should_replace:
                                    updated_content.append(new_result)
                                    replaced_any = True
                                    print(f"Replaced tool result for tool_use_id: {block.tool_use_id}")
                                    # Remove from new_tool_results since we've used it
                                    del new_tool_results[block.tool_use_id]
                                else:
                                    # Keep existing result
                                    updated_content.append(block)
                                    # Remove from new_tool_results to prevent duplicate
                                    if block.tool_use_id in new_tool_results:
                                        print(f"Skipping duplicate tool result for tool_use_id: {block.tool_use_id}")
                                        del new_tool_results[block.tool_use_id]
                            else:
                                # Keep non-tool-result blocks
                                updated_content.append(block)

                        if replaced_any:
                            # Update the message with replaced content
                            self.history[i] = Message(
                                role=msg.role, content=updated_content, stop_reason=msg.stop_reason
                            )

                # Add any remaining new tool results along with other blocks
                content_blocks = list(new_tool_results.values()) + other_blocks

                # If all blocks were handled (replaced or skipped), don't add empty message
                if not content_blocks:
                    return

        # Replace empty content with placeholder
        if not content_blocks or (isinstance(content_blocks, list) and len(content_blocks) == 0):
            print(f"Warning: User message has empty content, replacing with placeholder")
            content_blocks = [TextBlock(text="[User provided no message content]")]

        # Now safe to append
        self.history.append(Message(role="user", content=content_blocks))

    def append_assistant_message(self, message: Message) -> None:
        """
        Safely append an assistant message to history.

        Args:
            message: The assistant message to append
        """
        # Replace empty content with placeholder
        if not message.content or (isinstance(message.content, list) and len(message.content) == 0):
            print(f"Warning: Assistant message has empty content, replacing with placeholder")
            # Create a new message with placeholder content
            message = Message(
                role=message.role,
                content=[TextBlock(text="[Assistant returned no message content]")],
                stop_reason=message.stop_reason,
                tool_calls=message.tool_calls,
            )

        # Assistant messages can be appended directly
        self.history.append(message)

    def get_effective_history_for_llm(self) -> MessageHistory:
        """
        Return the subset of history to send to the LLM:
        - If compressed: [summary] + all messages after the compression boundary (excluding the summary itself)
        - Else: the full history
        """
        if self.last_compression_index is not None and self.history:
            # Summary is the message immediately after the boundary
            summary_idx = self.last_compression_index + 1
            if summary_idx < len(self.history):
                summary_msg = self.history[summary_idx]
                protected = [
                    message for message in self.history[:summary_idx] if self._is_protected_skill_content(message)
                ]
                # Tail starts after the summary
                tail = list(self.history[summary_idx + 1 :]) if summary_idx + 1 < len(self.history) else []
                return MessageHistory(protected + [summary_msg] + tail)

        return MessageHistory(list(self.history))

    def _is_protected_skill_content(self, message: Message) -> bool:
        if message.role != "user":
            return False
        if isinstance(message.content, str):
            return bool(self.skill_content_pattern.search(message.content))
        if not isinstance(message.content, list):
            return False
        return any(
            isinstance(block, TextBlock) and self.skill_content_pattern.search(block.text)
            for block in message.content
        )

    def extend_history(self, messages: List[Message]) -> None:
        """
        Safely extend history with multiple messages, validating the sequence.

        Args:
            messages: List of messages to append
        """
        # First, combine existing history with new messages
        all_messages = list(self.history) + messages

        # Fix any incomplete tool calls in the combined history
        fixed_messages = self.fix_incomplete_tool_calls(all_messages)

        # Replace history with the fixed version
        self.history = MessageHistory(fixed_messages)

    def _needs_tool_call_fix(self) -> bool:
        """
        Check if the last message has incomplete tool calls.

        Returns:
            bool: True if fix is needed, False otherwise
        """
        if not self.history:
            return False

        last_message = self.history[-1]

        # Only need to fix if last message is assistant with tool calls
        if last_message.role != "assistant":
            return False

        if not isinstance(last_message.content, list):
            return False

        # Check if it has tool calls
        return any(isinstance(block, ToolCall) for block in last_message.content)

    async def compress_history(self) -> None:
        """
        Non-destructively summarize the current history and mark a compression boundary.

        Records:
        - last_compression_index: index of the last message before compression
        - last_compression_summary: synthetic summary Message
        """
        if not self.history or len(self.history) < 5:
            return

        await self.log_info("Compressing message history...", sender=self.agent_name)

        try:
            # Build new compression prompts (system + user) per spec
            conversation_markdown = self.history.get_markdown_conversation()
            user_prompt_filled = COMPRESSION_SUMMARY_USER_PROMPT_TEMPLATE.replace("{HISTORY}", conversation_markdown)

            messages = MessageHistory([Message(role="user", content=[TextBlock(text=user_prompt_filled)])])

            system_message = Message(role="system", content=[TextBlock(text=COMPRESSION_SUMMARY_SYSTEM_PROMPT)])

            response = await self.llm.generate(
                messages=messages,
                system=system_message,
                temperature=self.model_default_temperature,
                model=self.config.long_context_config.model,
                max_completion_tokens=self.model_completion_tokens,
                thinking=self.config.long_context_config.thinking_tokens,
            )

            summary = response.get_text_content()

            # Append a single compressed user message at the end
            summary_message = Message(role="user", content=[TextBlock(text=summary)])
            self.history.append(summary_message)
            # Mark boundary as the last original index before the summary
            self.last_compression_index = len(self.history) - 2

            await self.log_info("Message history compressed (non-destructive).", sender=self.agent_name)

        except Exception as e:
            await self.log_error(f"Failed to compress message history: {str(e)}", sender=self.agent_name)

    def mark_cache_checkpoint(self):
        """
        Mark the last message in history for caching and remove cache_control from all other messages.

        This ensures that only the most recent message is cached, preventing redundant caching
        of older messages in the conversation history.

        To avoid exceeding Anthropic's limit of 4 blocks with cache_control, we only add
        cache_control to the very last content element of the last message.
        """
        # Remove cache_checkpoint from all content blocks in all messages
        for message in self.history:
            if hasattr(message, "content") and isinstance(message.content, list):
                for content_block in message.content:
                    if hasattr(content_block, "cache_checkpoint"):
                        content_block.cache_checkpoint = False

        # Set cache_checkpoint to True only for the last content block of the last message if history exists
        if self.history:
            last_message = self.history[-1]
            if hasattr(last_message, "content") and isinstance(last_message.content, list) and last_message.content:
                # Get the last content block only
                last_content_block = last_message.content[-1]
                if hasattr(last_content_block, "cache_checkpoint"):
                    last_content_block.cache_checkpoint = True

    async def execute_single_tool(self, tool_use_block: ToolCall) -> ToolResult:
        """Execute a single tool and return its result with metadata"""
        tool_name = tool_use_block.name
        inputs = tool_use_block.input
        provider_tool_call_id = tool_use_block.id
        tool_execution_id = getattr(tool_use_block, "execution_id", provider_tool_call_id)

        # Keep provider IDs for LLM history while exposing an internal unique ID to app services.
        self.current_provider_tool_call_id = provider_tool_call_id
        self.current_tool_execution_id = tool_execution_id
        self.current_tool_call_id = tool_execution_id

        try:
            available_tools = {tool.name for tool in self.tool_collection.get_tool_list()}
            if tool_name not in available_tools:
                error_message = f"Tool '{tool_name}' is not available in this mode."
                await self.log_error(error_message, sender=self.agent_name)
                await self.send_chat_message(
                    message_type="tool_error",
                    content=error_message,
                    is_streaming=False,
                    tool_description=tool_name,
                    tool_call_id=tool_execution_id,
                )
                return ToolResult(
                    tool_use_id=provider_tool_call_id,
                    content=error_message,
                    name=tool_name,
                    is_error=True,
                    execution_id=tool_execution_id,
                )

            # Log the tool being called
            await self.log_info(f"Executing tool: {tool_name}", sender=self.agent_name)

            # Send tool_call message to indicate we're starting execution
            if not all(
                [
                    self.config.long_context_config.provider == ModelProvider.ANTHROPIC,
                    tool_name in self.long_content_tool_calls,
                ]
            ):
                await self.send_chat_message(
                    message_type="tool_call",
                    content=f"Calling {tool_name}",
                    is_streaming=False,
                    tool_description=tool_name,
                    tool_call_id=tool_execution_id,
                )

            output = await self.tool_collection.__getattribute__(tool_name)(**inputs)

            # Handle the case where the output is a list of ContentBlock objects
            chat_message_content = output
            if isinstance(output, list):
                chat_message_content = "\n\n".join(item.to_markdown() for item in output)

            if tool_name == "write_memory":
                self._initialize_system_prompt()

            # Send tool_result message for successful execution
            await self.send_chat_message(
                message_type="tool_result",
                content=chat_message_content,
                is_streaming=False,
                tool_description=tool_name,
                tool_call_id=tool_execution_id,
            )

            return ToolResult(
                tool_use_id=provider_tool_call_id,
                content=output,
                name=tool_name,
                is_error=False,
                execution_id=tool_execution_id,
            )
        except Exception as ex:
            error_message = str(ex)
            await self.log_error(f"Error executing tool {tool_name}: {error_message}", sender=self.agent_name)

            # Send tool_error message for failed execution
            await self.send_chat_message(
                message_type="tool_error",
                content=error_message,
                is_streaming=False,
                tool_description=tool_name,
                tool_call_id=tool_execution_id,
            )

            return ToolResult(
                tool_use_id=provider_tool_call_id,
                content=error_message,
                name=tool_use_block.name,
                is_error=True,
                execution_id=tool_execution_id,
            )
        finally:
            # Clear current tool call ID after execution
            self.current_tool_call_id = None
            self.current_tool_execution_id = None
            self.current_provider_tool_call_id = None

    async def process_tool_calls(self, tool_use_blocks: List[ToolCall]) -> List[ToolResult]:
        """
        Process multiple tool calls either in parallel or sequentially based on tool types.

        Args:
            tool_use_blocks: List of tool use blocks from the LLM

        Returns:
            List of tool responses with metadata
        """
        # If only one tool call, just execute it directly
        if len(tool_use_blocks) == 1:
            return [await self.execute_single_tool(tool_use_blocks[0])]

        # Read-only tools have no side effects and agent dispatches operate on
        # independent sub-agents, so batches made up entirely of these can run
        # concurrently. Any other tool in the batch forces sequential execution.
        parallel_safe_tools = set(ToolCollection.read_only_tools) | set(ToolCollection.agent_dispatch_tools)
        all_parallel_safe = all(block.name in parallel_safe_tools for block in tool_use_blocks)

        if all_parallel_safe:
            # Execute all tools in parallel
            await self.log_info(
                f"Executing {len(tool_use_blocks)} parallel-safe tool calls in parallel", sender=self.agent_name
            )
            semaphore = asyncio.Semaphore(self.PARALLEL_TOOL_LIMIT)

            async def run_limited(block: ToolCall) -> ToolResult:
                async with semaphore:
                    return await self.execute_single_tool(block)

            # Wait for all tasks to complete; gather preserves input order so
            # tool results stay aligned with their tool calls in history.
            results = await asyncio.gather(*(run_limited(block) for block in tool_use_blocks))
            return list(results)
        else:
            # Execute tools sequentially
            await self.log_info(
                f"Executing {len(tool_use_blocks)} tool calls sequentially (some are not read-only)",
                sender=self.agent_name,
            )
            results = []
            for block in tool_use_blocks:
                result = await self.execute_single_tool(block)
                results.append(result)
            return results

    async def send_chat_message(
        self, message_type: str, content: str, is_streaming: bool = False, tool_description=None, tool_call_id=None
    ) -> None:
        """
        Send a message to the chat interface.

        Args:
            content: The message content to send
            is_streaming: Whether this is part of a streaming message
        """
        timestamp = datetime.now().isoformat()

        # Include sub_agent_info if this is a sub-agent
        sub_agent_info = None
        if self.sub_agent and self.sub_agent_context:
            # Dispatch metadata set by AgentTool (agent_id, task, parent IDs)
            sub_agent_info = dict(self.sub_agent_context)
        elif self.sub_agent and self.parent_tool_call_id:
            sub_agent_info = {
                "agent_name": self.agent_name,
                "conversation_id": self.conversation_id,
                "parent_tool_call_id": self.parent_tool_call_id,
                "depth": 1,  # Can be enhanced to track nested depth
            }

        event = AgentEvent(
            sender=self.agent_name,
            event_type="chat_message",
            content={
                "message_type": message_type,
                "text": content,
                "tool_description": tool_description,
                "tool_call_id": tool_call_id,
            },
            timestamp=timestamp,
            is_streaming=is_streaming,
            sub_agent_info=sub_agent_info,
        )
        await self.connection_manager.broadcast_event(event, self.workspace_id, self.thread_id)

    async def handle_llm_error(self, error: Exception) -> None:
        """
        Handle LLM errors with appropriate retry logic and logging.

        This method provides centralized error handling for all LLM operations:
        - Rate limit errors: Log warning, wait 60 seconds, and allow retry
        - Other LLM errors: Log error and re-raise
        - Non-LLM errors: Re-raise as-is

        Args:
            error: The exception to handle

        Raises:
            LLMError: Re-raises LLM errors after logging
            Exception: Re-raises non-LLM exceptions as-is
        """
        error = map_to_llm_error(error, provider=self.config.long_context_config.provider.value)

        if isinstance(error, LLMRateLimitError):
            await self.log_warning(
                f"Rate limit exceeded: {error}. Waiting for 60 seconds before retrying...", sender=self.agent_name
            )
            await asyncio.sleep(60)
            await self.log_info("Resuming after rate limit wait period.", sender=self.agent_name)
            # Don't re-raise - allow retry

        elif isinstance(error, LLMInternalServerError):
            event = AgentEvent(
                sender=self.agent_name,
                event_type="llm_status_update",
                content={
                    "status": "error",
                    "message": "There is high traffic on our LLM provider right now. Please try again in a few seconds.",
                },
                timestamp=datetime.now().isoformat(),
                is_streaming=False,
            )
            await self.connection_manager.broadcast_event(event, self.workspace_id, self.thread_id)
            raise

        elif isinstance(error, LLMContextWindowExceededError):
            event = AgentEvent(
                sender=self.agent_name,
                event_type="llm_status_update",
                content={
                    "status": "error",
                    "message": (
                        "The conversation context became too large for the model. "
                        "Oversized tool output is trimmed automatically; please retry the message."
                    ),
                },
                timestamp=datetime.now().isoformat(),
                is_streaming=False,
            )
            await self.connection_manager.broadcast_event(event, self.workspace_id, self.thread_id)
            raise

        elif isinstance(error, LLMError):
            await self.log_error(f"LLM error occurred: {error}", sender=self.agent_name)
            raise  # Re-raise to maintain current behavior
        else:
            # Non-LLM error - just re-raise
            raise

    async def cleanup(self) -> None:
        """
        Clean up all agent resources.
        This should be called when the agent is being destroyed.
        """
        # Clean up tool collection resources
        if hasattr(self, "tool_collection"):
            await self.tool_collection.cleanup()

        # Log cleanup
        await self.log_info("Agent cleanup completed", sender=self.agent_name)

    # ------------------------------------------------------------------
    # Deprecated aliases
    #
    # These methods started as private helpers but host applications built
    # subclasses on top of them, so the underscore names are de-facto public.
    # The supported names are the contract going forward; these aliases exist
    # so existing subclasses keep working and will be removed once all hosts
    # have migrated.
    # ------------------------------------------------------------------

    def _warn_deprecated(self, old: str, new: str) -> None:
        warnings.warn(
            f"BaseAgent.{old} is deprecated; use BaseAgent.{new} instead.",
            DeprecationWarning,
            stacklevel=3,
        )

    def _build_prompt_context(self) -> PromptContext:
        self._warn_deprecated("_build_prompt_context", "build_prompt_context")
        return self.build_prompt_context()

    def _mark_last_message_for_cache(self) -> None:
        self._warn_deprecated("_mark_last_message_for_cache", "mark_cache_checkpoint")
        return self.mark_cache_checkpoint()

    async def _compress_message_history(self) -> None:
        self._warn_deprecated("_compress_message_history", "compress_history")
        return await self.compress_history()

    def _fix_incomplete_tool_calls(self, messages: List[Message]) -> List[Message]:
        self._warn_deprecated("_fix_incomplete_tool_calls", "fix_incomplete_tool_calls")
        return self.fix_incomplete_tool_calls(messages)
