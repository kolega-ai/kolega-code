import asyncio
import contextvars
import logging
import os
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional

from .common import LogMixin
from .compression import HistoryCompressor
from kolega_code.config import AgentConfig, ModelProvider
from kolega_code.events import AgentConnectionManager
from .context import AgentContext, AgentServices, Telemetry, WorkspaceInfo
from .conversation import Conversation
from kolega_code.events import AgentEventEmitter
from kolega_code.llm.exceptions import (
    LLMContextWindowExceededError,
    LLMError,
    LLMInternalServerError,
    LLMRateLimitError,
    map_to_llm_error,
)
from kolega_code.llm.models import ImageBlock, Message, MessageHistory, TextBlock, ToolCall, ToolResult
from kolega_code.llm.providers.models import TokenCount
from kolega_code.llm.specs import get_model_specs
from .prompt_provider import PromptProvider, AgentMode, PromptContext, PromptExtension
from kolega_code.services.base import TerminalManager, BrowserManager
from kolega_code.services.file_system import FileSystem
from .tools import ToolCollection
from kolega_code.tools import ToolError
from .utils.commands import CommandProcessor
from langfuse import Langfuse


logger = logging.getLogger(__name__)


class BaseAgent(LogMixin):
    """
    Base class for all AI agents in the system.

    BaseAgent owns the canonical agent loop and composes the pieces that do
    the real work: a Conversation (history and its invariants), a
    HistoryCompressor (context-budget management), and an AgentEventEmitter
    (event construction and broadcast). Subclasses configure tools and the
    system prompt, and customize behavior through the documented hook methods.
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
        project_path: str | Path | None = None,
        workspace_id: Optional[str] = None,
        thread_id: Optional[str] = None,
        connection_manager: Optional[AgentConnectionManager] = None,
        config: Optional[AgentConfig] = None,
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
        prompt_provider: Optional[PromptProvider] = None,
        prompt_extensions: Optional[List[PromptExtension]] = None,
        tool_extensions: Optional[List[Any]] = None,
        usage_recorder: Optional[Any] = None,
        sub_agent_recorder: Optional[Any] = None,
        context: Optional[AgentContext] = None,
    ) -> None:
        """
        Initialize a new BaseAgent instance.

        Preferred: pass a fully-built ``context`` (AgentContext). The flat
        keyword signature remains supported and is converted internally.

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
            prompt_provider: Optional host-configured prompt provider
            prompt_extensions: Host-provided prompt sections for app-specific context
            tool_extensions: Host-provided tool providers for app-specific tools
            usage_recorder: Optional callback for recording normalized LLM usage
            sub_agent_recorder: Optional callback for persisting sub-agent conversation state
            context: Pre-built AgentContext; takes precedence over the flat keywords
        """
        if context is None:
            if project_path is None or workspace_id is None or thread_id is None:
                raise TypeError(
                    "BaseAgent requires either an AgentContext or project_path/workspace_id/thread_id"
                )

            workspace = WorkspaceInfo(
                project_path=Path(project_path) if isinstance(project_path, str) else project_path,
                workspace_id=workspace_id,
                thread_id=thread_id,
                project_template_slug=project_template_slug,
                protected_files=protected_files or [],
                env_var_descriptions=workspace_env_var_descriptions or {},
                memories=workspace_memories or [],
            )

            defaults = AgentServices.local(workspace, connection_manager)
            services = AgentServices(
                filesystem=filesystem or defaults.filesystem,
                terminal_manager=terminal_manager or defaults.terminal_manager,
                browser_manager=browser_manager or defaults.browser_manager,
            )

            context = AgentContext(
                workspace=workspace,
                config=config,
                connection_manager=connection_manager,
                services=services,
                telemetry=Telemetry(
                    langfuse_client=langfuse_client,
                    user_id=user_id,
                    user_email=user_email,
                    usage_recorder=usage_recorder,
                    sub_agent_recorder=sub_agent_recorder,
                ),
                agent_mode=agent_mode,
                prompt_provider=prompt_provider,
                prompt_extensions=prompt_extensions or [],
                tool_extensions=tool_extensions or [],
            )
        elif prompt_provider is not None:
            context.prompt_provider = prompt_provider

        self.context = context

        # Flat attributes kept for compatibility with subclasses, tools, and hosts.
        self.project_path = context.workspace.project_path
        self.workspace_id = context.workspace.workspace_id
        self.thread_id = context.workspace.thread_id
        self.connection_manager = context.connection_manager
        self.config = context.config
        self.filesystem = context.services.filesystem
        self.terminal_manager = context.services.terminal_manager
        self.browser_manager = context.services.browser_manager
        self.langfuse_client = context.telemetry.langfuse_client
        self.user_id = context.telemetry.user_id
        self.user_email = context.telemetry.user_email
        self.project_template_slug = context.workspace.project_template_slug
        self.protected_files = context.workspace.protected_files
        self.agent_mode = context.agent_mode
        self.workspace_env_var_descriptions = context.workspace.env_var_descriptions
        self.workspace_memories = context.workspace.memories
        self.prompt_extensions = context.prompt_extensions
        self.tool_extensions = context.tool_extensions
        self.usage_recorder = context.telemetry.usage_recorder
        self.sub_agent_recorder = context.telemetry.sub_agent_recorder

        self.available_ports = "9001-9999"

        # Validate that the project path exists and is a directory using the filesystem
        if not self.filesystem.exists("."):
            raise ValueError(f"Project path does not exist: {self.project_path}")
        if not self.filesystem.is_dir("."):
            raise ValueError(f"Project path is not a directory: {self.project_path}")

        self.prompt_provider = context.prompt_provider or PromptProvider()

        self.conversation = Conversation(max_tool_result_chars=self.max_tool_result_chars_in_history)
        self.conversation.skill_content_pattern = self.skill_content_pattern
        self.compressor = HistoryCompressor(threshold=self.history_compression_threshold)
        self.emitter = AgentEventEmitter(
            connection_manager=self.connection_manager,
            workspace_id=self.workspace_id,
            thread_id=self.thread_id,
            sender=self.agent_name,
            sub_agent_info_provider=self._sub_agent_info,
        )

        model_specs = get_model_specs(self.config.long_context_config.provider, self.config.long_context_config.model)
        self.model_context_length = model_specs["context_length"]
        self.model_completion_tokens = model_specs["max_completion_tokens"]
        self.model_default_temperature = model_specs.get("default_temperature", 1.0)

        self.llm = context.create_llm_client(agent_name=self.agent_name)

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

    # ------------------------------------------------------------------
    # Conversation delegation
    #
    # self.conversation owns the message history; these wrappers preserve the
    # established BaseAgent surface for subclasses and hosts.
    # ------------------------------------------------------------------

    @property
    def history(self) -> MessageHistory:
        return self.conversation.history

    @history.setter
    def history(self, value) -> None:
        self.conversation.history = value if isinstance(value, MessageHistory) else MessageHistory(list(value))

    @property
    def last_compression_index(self) -> Optional[int]:
        return self.conversation.last_compression_index

    @last_compression_index.setter
    def last_compression_index(self, value: Optional[int]) -> None:
        self.conversation.last_compression_index = value

    def append_user_message(self, content) -> None:
        """
        Safely append a user message to history, fixing any incomplete tool calls first.

        Args:
            content: Either a string (converted to TextBlock) or list of ContentBlocks
        """
        self.conversation.append_user(content)

    def append_assistant_message(self, message: Message) -> None:
        """
        Safely append an assistant message to history.

        Args:
            message: The assistant message to append
        """
        self.conversation.append_assistant(message)

    def extend_history(self, messages: List[Message]) -> None:
        """
        Safely extend history with multiple messages, validating the sequence.

        Args:
            messages: List of messages to append
        """
        self.conversation.extend(messages)

    def get_effective_history_for_llm(self) -> MessageHistory:
        """
        Return the subset of history to send to the LLM:
        - If compressed: [summary] + all messages after the compression boundary (excluding the summary itself)
        - Else: the full history
        """
        return self.conversation.effective_history()

    def fix_incomplete_tool_calls(self, messages: List[Message]) -> List[Message]:
        """
        Fix incomplete tool call sequences by adding placeholder tool_result blocks
        for any orphaned tool_use blocks.

        Args:
            messages: List of messages to validate and fix

        Returns:
            List[Message]: Fixed messages with placeholder tool results added where needed
        """
        return self.conversation.repaired(messages)

    def mark_cache_checkpoint(self) -> None:
        """
        Mark the last message in history for caching and remove cache_control from all other messages.

        This ensures that only the most recent message is cached, preventing redundant caching
        of older messages in the conversation history.
        """
        self.conversation.mark_cache_checkpoint()

    def dump_message_history(self) -> List[Dict[str, Any]]:
        """Serializes the message history into a list of dictionaries using custom methods."""
        return self.conversation.dump()

    def restore_message_history(self, serialized_history: List[Dict[str, Any]]) -> None:
        """Restores the message history from a list of dictionaries using custom methods."""
        self.conversation.restore(serialized_history)

    def _sanitize_oversized_tool_results(self) -> int:
        return self.conversation.sanitize_oversized_tool_results()

    def _is_history_valid_for_anthropic(self, messages: List[Message] = None) -> bool:
        """
        Check if the message history is valid for Anthropic API.
        Every tool_use block must be followed by a tool_result block.
        """
        return self.conversation.is_valid_for_anthropic(messages)

    def _is_protected_skill_content(self, message: Message) -> bool:
        return self.conversation.is_protected(message)

    def _needs_tool_call_fix(self) -> bool:
        """Check if the last message has incomplete tool calls."""
        return self.conversation.needs_tool_call_fix()

    # ------------------------------------------------------------------
    # Prompt context
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Attachments
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Context budget
    # ------------------------------------------------------------------

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

        await self.emitter.context_update(
            input_tokens=token_count.input_tokens,
            model_context_length=self.model_context_length,
            compression_threshold=self.history_compression_threshold,
            alert_level=alert_level,
            message=message,
        )

    async def compress_history(self) -> None:
        """
        Non-destructively summarize the current history and mark a compression boundary.
        """

        async def on_info(message: str) -> None:
            await self.log_info(message, sender=self.agent_name)

        async def on_error(message: str) -> None:
            await self.log_error(message, sender=self.agent_name)

        await self.compressor.summarize(
            self.conversation,
            llm=self.llm,
            model=self.config.long_context_config.model,
            max_completion_tokens=self.model_completion_tokens,
            temperature=self.model_default_temperature,
            thinking=self.config.long_context_config.thinking_effort,
            on_info=on_info,
            on_error=on_error,
        )

    # ------------------------------------------------------------------
    # Tool execution
    # ------------------------------------------------------------------

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
            registry = self.tool_collection.registry()
            if tool_name not in registry:
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

            output = await registry.call(tool_name, **inputs)

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
        except ToolError as ex:
            # Expected tool failure: surface to the model without an
            # internal-error log.
            error_message = str(ex)
            await self.log_warning(f"Tool {tool_name} failed: {error_message}", sender=self.agent_name)

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

        # A batch runs concurrently only when every tool in it is marked
        # parallel-safe (read-only operations and independent sub-agent
        # dispatches); any other tool forces sequential execution.
        registry = self.tool_collection.registry()
        all_parallel_safe = all(
            block.name in registry and registry.get(block.name).parallel_safe for block in tool_use_blocks
        )

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

    # ------------------------------------------------------------------
    # Events
    # ------------------------------------------------------------------

    def _sub_agent_info(self) -> Optional[Dict[str, Any]]:
        """Sub-agent dispatch metadata included in chat events, when applicable."""
        if self.sub_agent and self.sub_agent_context:
            # Dispatch metadata set by AgentTool (agent_id, task, parent IDs)
            return dict(self.sub_agent_context)
        if self.sub_agent and self.parent_tool_call_id:
            return {
                "agent_name": self.agent_name,
                "conversation_id": self.conversation_id,
                "parent_tool_call_id": self.parent_tool_call_id,
                "depth": 1,  # Can be enhanced to track nested depth
            }
        return None

    async def send_chat_message(
        self, message_type: str, content: str, is_streaming: bool = False, tool_description=None, tool_call_id=None
    ) -> None:
        """
        Send a message to the chat interface.

        Args:
            content: The message content to send
            is_streaming: Whether this is part of a streaming message
        """
        await self.emitter.chat(
            message_type,
            content,
            is_streaming=is_streaming,
            tool_description=tool_description,
            tool_call_id=tool_call_id,
        )

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
            await self.emitter.llm_status(
                "error",
                "There is high traffic on our LLM provider right now. Please try again in a few seconds.",
            )
            raise

        elif isinstance(error, LLMContextWindowExceededError):
            await self.emitter.llm_status(
                "error",
                (
                    "The conversation context became too large for the model. "
                    "Oversized tool output is trimmed automatically; please retry the message."
                ),
            )
            raise

        elif isinstance(error, LLMError):
            await self.log_error(f"LLM error occurred: {error}", sender=self.agent_name)
            raise  # Re-raise to maintain current behavior
        else:
            # Non-LLM error - just re-raise
            raise

    # ------------------------------------------------------------------
    # The agent loop
    #
    # process_message_stream is the single canonical loop shared by every
    # agent. Subclasses customize behavior through the hook methods below
    # (build_user_content, apply_compression_fallback, on_tool_use_start,
    # should_stop_after_tools, recap_agent_outcome) rather than overriding
    # the loop itself.
    # ------------------------------------------------------------------

    completion_log_message = "Processing complete"

    async def build_user_content(self, message: str, attachments: Optional[List[Dict[str, Any]]]) -> List[Any]:
        """
        Build the content blocks for an incoming user message.

        Default: the message text plus blocks for any image/file attachments.
        """
        content_blocks: List[Any] = [TextBlock(text=message)]
        content_blocks.extend(self._attachment_blocks(attachments))

        for attachment in attachments or []:
            if attachment.get("type") == "image":
                await self.log_info(
                    f"Received image attachment: {attachment.get('filename', 'unnamed')} ({attachment.get('media_type', 'unknown')})",
                    sender=self.agent_name,
                )
            elif attachment.get("type") == "file":
                await self.log_info(
                    f"Attached file from @ mention: {attachment.get('path', 'unnamed')}",
                    sender=self.agent_name,
                )

        return content_blocks

    def apply_compression_fallback(self) -> None:
        """
        Hard-truncate history when compression alone could not get under budget.

        Default: keep the first message (the original task) plus any protected
        skill-content messages.
        """
        first_message = self.history[0]
        protected = [
            message
            for message in self.history
            if message is not first_message and self._is_protected_skill_content(message)
        ]
        self.history = MessageHistory(protected + [first_message])

    async def on_tool_use_start(self, tool_call_delta: Dict[str, Any]) -> None:
        """
        Called when the provider streams a tool_use_start event (Anthropic only).

        Long-content tools stream large arguments, so announce them as soon as
        they start instead of waiting for the arguments to finish streaming
        (execute_single_tool skips the announcement for these tools).
        """
        tool_name = tool_call_delta.get("name")
        tool_execution_id = tool_call_delta.get("execution_id") or tool_call_delta.get("id")
        if tool_name in self.long_content_tool_calls:
            await self.send_chat_message(
                message_type="tool_call",
                content=f"Calling {tool_name}",
                is_streaming=False,
                tool_description=tool_name,
                tool_call_id=tool_execution_id,
            )

    def should_stop_after_tools(self) -> bool:
        """Return True to end the loop after a successful tool batch (e.g. a plan was written)."""
        return False

    async def recap_agent_outcome(self) -> str:
        """Return the agent's final report: the text of the last message in history."""
        return self.history[-1].get_text_content()

    async def process_message_stream(
        self, message: str, attachments: List[Dict[str, Any]] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Process a user message and yield response/thinking chunks while the agent works.

        Yields dicts of the form
        ``{"type": "response"|"thinking", "content": str, "complete": bool, "uuid": str}``.
        """
        unsupported_attachment_message = self._unsupported_attachment_message(attachments)
        if unsupported_attachment_message:
            yield {
                "type": "response",
                "content": unsupported_attachment_message,
                "complete": True,
                "uuid": str(uuid.uuid4()),
            }
            return

        self.append_user_message(await self.build_user_content(message, attachments))

        stop_reason = None
        while stop_reason not in ["end_turn", "max_tokens", "stop_sequence"]:
            self.mark_cache_checkpoint()

            try:
                token_count = await self.count_current_context()
                logger.debug("Input token count: %s", token_count)

                if self.compressor.over_budget(token_count.input_tokens, self.model_context_length):
                    await self.compress_history()
                    token_count = await self.count_current_context()

                    if self.compressor.over_budget(token_count.input_tokens, self.model_context_length):
                        self.apply_compression_fallback()

                    self.mark_cache_checkpoint()

                current_response = ""
                current_thinking = ""
                thinking_started = False
                # Use the same UUID for each segment of the response
                response_uuid = str(uuid.uuid4())
                thinking_uuid = str(uuid.uuid4())

                # Fix history before sending to LLM to ensure valid tool call sequences
                effective = self.get_effective_history_for_llm()
                fixed_history = MessageHistory(self.fix_incomplete_tool_calls(list(effective)))

                async with await self.llm.stream(
                    system=self.system_prompt,
                    max_completion_tokens=self.model_completion_tokens,
                    temperature=self.model_default_temperature,
                    messages=fixed_history,
                    model=self.config.long_context_config.model,
                    tools=self.tool_collection.get_tool_list(),
                    thinking=self.config.long_context_config.thinking_effort,
                ) as stream:
                    async for event in stream:
                        if event.type == "text":
                            current_response += event.text

                            # Send periodic updates as the response grows
                            if len(current_response) >= 50:
                                yield {
                                    "type": "response",
                                    "content": current_response,
                                    "complete": False,
                                    "uuid": response_uuid,
                                }
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

                        elif event.type == "tool_use_start" and event.tool_call_delta:
                            # Flush accumulated text first so the user doesn't have to wait for it.
                            yield {
                                "type": "response",
                                "content": current_response,
                                "complete": True,
                                "uuid": response_uuid,
                            }
                            current_response = ""

                            await self.on_tool_use_start(event.tool_call_delta)

                assistant_message = await stream.get_final_message()
                stop_reason = assistant_message.stop_reason

                self.append_assistant_message(assistant_message)

                if thinking_started or current_thinking:
                    yield {"type": "thinking", "content": current_thinking, "complete": True, "uuid": thinking_uuid}

                # Send the final message to mark it complete.
                yield {"type": "response", "content": current_response, "complete": True, "uuid": response_uuid}

                if assistant_message.tool_calls:
                    await self.log_info(
                        f"Received {len(assistant_message.tool_calls)} tool call(s)", sender=self.agent_name
                    )

                    try:
                        tool_responses = await self.process_tool_calls(assistant_message.tool_calls)
                        self.append_user_message(tool_responses)

                        if self.should_stop_after_tools():
                            break
                    except Exception as ex:
                        error_message = f"Error processing tool calls: {str(ex)}"
                        await self.log_error(error_message, sender=self.agent_name)

                        error_responses = [
                            ToolResult(
                                tool_use_id=tool_call.id,
                                content=f"Failed to process tool calls: {str(ex)}",
                                name=tool_call.name,
                                is_error=True,
                            )
                            for tool_call in assistant_message.tool_calls
                        ]
                        self.append_user_message(error_responses)

            except Exception as ex:
                await self.handle_llm_error(ex)

        await self.log_info(self.completion_log_message, sender=self.agent_name)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

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
