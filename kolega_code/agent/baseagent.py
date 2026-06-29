import asyncio
import contextvars
import logging
import os
import random
import re
import sys
import time
import uuid
from email.utils import parsedate_to_datetime
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional

from .common import LogMixin
from .compression import CompactionResult, HistoryCompressor
from .errors import MaxAgentIterationsExceeded
from kolega_code.config import AgentConfig, ModelProvider
from kolega_code.events import AgentConnectionManager
from .context import AgentContext, AgentServices, Telemetry, WorkspaceInfo
from .conversation import Conversation, adapt_history_for_provider
from kolega_code.events import AgentEventEmitter
from kolega_code.hooks import (
    NO_OP_DISPATCHER,
    HookCapabilities,
    HookDispatcher,
    HookEvent,
    HookOutcome,
    LifecycleEvent,
)
from kolega_code.llm.exceptions import (
    LLMConnectionError,
    LLMError,
    LLMInternalServerError,
    LLMRateLimitError,
    llm_error_message,
    map_to_llm_error,
)
from kolega_code.llm.models import ImageBlock, Message, MessageHistory, TextBlock, ToolCall, ToolResult
from kolega_code.llm.providers.models import TokenCount
from kolega_code.llm.specs import get_model_specs, supports_vision as model_supports_vision
from kolega_code.permissions import (
    PermissionDecision,
    PermissionMode,
    auto_allow_permission_callback,
    normalize_permission_mode,
    permission_request_for_tool,
)
from .prompt_provider import PromptProvider, AgentMode, AgentType, PromptContext, PromptExtension
from .prompt_overrides import ProjectPromptOverrides, format_prompt_override_error, render_prompt_override_source
from kolega_code.services.base import TerminalManager, BrowserManager
from kolega_code.services.file_system import FileSystem
from .tools import ToolCollection  # noqa: F401 - kept for tests and downstream monkeypatch compatibility
from kolega_code.tools import ToolError
from .utils.commands import CommandProcessor
from langfuse import Langfuse


logger = logging.getLogger(__name__)

PROJECT_GUIDANCE_FILES = ("AGENTS.md", "KOLEGA.md")
AGENT_MEMORY_FILE = "AGENT_MEMORY.md"

# System prompt for `prompt`/`agent` lifecycle hooks: the model's only job is a
# yes/no decision returned as a compact JSON object.
HOOK_DECISION_SYSTEM_PROMPT = (
    "You are a lifecycle hook that makes a single yes/no decision about an AI coding "
    "agent's action. You are given the event data and a question or condition to evaluate. "
    'Respond with ONLY a JSON object: {"ok": true} to allow the action to proceed, or '
    '{"ok": false, "reason": "<short explanation>"} to block it. The reason is shown to the '
    "agent. Output nothing other than the JSON object."
)


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
    # Cap on how many times a Stop hook may force the agent to keep working in one
    # turn, so a misbehaving "don't stop until X" hook cannot loop forever.
    MAX_STOP_HOOK_OVERRIDES = 5
    long_content_tool_calls = ["create_file", "replace_entire_file"]
    max_tool_result_chars_in_history = 100_000
    skill_content_pattern = re.compile(r'<skill_content name="[^"]+">')

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
        permission_mode: Optional[PermissionMode | str] = None,
        permission_callback: Optional[Any] = None,
        usage_recorder: Optional[Any] = None,
        sub_agent_recorder: Optional[Any] = None,
        hook_dispatcher: Optional[HookDispatcher] = None,
        context: Optional[AgentContext] = None,
        max_iterations: Optional[int] = None,
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
                raise TypeError("BaseAgent requires either an AgentContext or project_path/workspace_id/thread_id")

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
                permission_mode=normalize_permission_mode(permission_mode, default=PermissionMode.AUTO),
                permission_callback=permission_callback or auto_allow_permission_callback,
            )
        elif prompt_provider is not None:
            context.prompt_provider = prompt_provider
            if permission_mode is not None:
                context.permission_mode = normalize_permission_mode(permission_mode, default=context.permission_mode)
            if permission_callback is not None:
                context.permission_callback = permission_callback

        # Apply an explicitly-passed hook dispatcher regardless of how the context
        # was built; otherwise the context's default (NO_OP_DISPATCHER) is used.
        if hook_dispatcher is not None:
            context.hook_dispatcher = hook_dispatcher

        if max_iterations is not None and max_iterations < 1:
            raise ValueError("max_iterations must be a positive integer or None")

        self.context = context
        self.max_iterations = max_iterations

        # Flat attributes kept for compatibility with subclasses, tools, and hosts.
        self.project_path = context.workspace.project_path
        self.workspace_id = context.workspace.workspace_id
        self.thread_id = context.workspace.thread_id
        self.connection_manager = context.connection_manager
        self.config = context.config
        # The model this agent runs its main loop on: the per-role override when one
        # is configured for this agent_name, otherwise the global long-context model.
        self.primary_model_config = self.config.model_config_for_agent(self.agent_name)
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
        self.permission_mode = context.permission_mode
        self.permission_callback = context.permission_callback or auto_allow_permission_callback
        self.hook_dispatcher = context.hook_dispatcher or NO_OP_DISPATCHER
        self.usage_recorder = context.telemetry.usage_recorder
        self.sub_agent_recorder = context.telemetry.sub_agent_recorder

        # gigacode (workflow orchestration) opt-in. Off by default; the host toggles
        # it via apply_gigacode(). The run_workflow tool gate reads this live, so
        # toggling takes effect on the next turn without rebuilding the agent.
        self.gigacode_enabled = False

        self.prompt_override_errors: List[str] = []

        self.available_ports = "9001-9999"

        # Validate that the project path exists and is a directory using the filesystem
        if not self.filesystem.exists("."):
            raise ValueError(f"Project path does not exist: {self.project_path}")
        if not self.filesystem.is_dir("."):
            raise ValueError(f"Project path is not a directory: {self.project_path}")

        self.prompt_provider = context.prompt_provider or PromptProvider()
        self.prompt_overrides = ProjectPromptOverrides(self.filesystem)

        self.conversation = Conversation(max_tool_result_chars=self.max_tool_result_chars_in_history)
        self.conversation.skill_content_pattern = self.skill_content_pattern
        # Counts consecutive transient LLM failures (rate-limit / overload) so the turn loop
        # backs off and eventually gives up instead of retrying forever; reset on any good turn.
        self._consecutive_llm_retries = 0
        self.compressor = HistoryCompressor(threshold=self.history_compression_threshold)
        self.emitter = AgentEventEmitter(
            connection_manager=self.connection_manager,
            workspace_id=self.workspace_id,
            thread_id=self.thread_id,
            sender=self.agent_name,
            sub_agent_info_provider=self._sub_agent_info,
        )

        model_specs = get_model_specs(self.primary_model_config.provider, self.primary_model_config.model)
        self.model_context_length = model_specs["context_length"]
        self.model_completion_tokens = model_specs["max_completion_tokens"]
        self.model_default_temperature = model_specs.get("default_temperature", 1.0)
        # Whether this agent's primary model can accept image input. Read by the
        # ToolCollection read_image tool gate (so non-vision models never see the
        # tool) and used by _unsupported_attachment_message to reject image
        # attachments for non-vision models with a clear message.
        self.supports_vision = bool(model_specs.get("supports_vision", False))

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
        # Set by a blocking PostToolUse hook to end the turn after the current tool batch.
        self._hook_end_turn = False

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

    def _history_for_llm(self) -> MessageHistory:
        """Build the message history to send to the LLM for this turn.

        Compaction-aware and tool-call-repaired. For non-vision models, any
        ``ImageBlock`` carried over from earlier turns (user attachments or
        ``read_image`` tool results) is replaced with a text placeholder on this
        request copy only — the stored history is never mutated, so switching
        back to a vision-capable model restores the images.
        """
        effective = self.get_effective_history_for_llm()
        fixed = self.fix_incomplete_tool_calls(list(effective))
        provider = getattr(self.primary_model_config.provider, "value", self.primary_model_config.provider)
        fixed = adapt_history_for_provider(
            fixed,
            target_provider=str(provider),
            target_model=self.primary_model_config.model,
            supports_vision=self.supports_vision,
        )
        return MessageHistory(fixed)

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

    def apply_gigacode(self, enabled: bool, prompt_extension=None) -> None:
        """Enable or disable gigacode workflow orchestration for this session.

        Flips the ``run_workflow`` tool gate and refreshes the system prompt to
        include or drop the authoring guide. Safe to call mid-session; the tool
        registry and the next turn pick up the change.
        """
        self.gigacode_enabled = enabled
        extensions = [ext for ext in (self.prompt_extensions or []) if getattr(ext, "id", None) != "gigacode"]
        if enabled and prompt_extension is not None:
            extensions.append(prompt_extension)
        self.prompt_extensions = extensions
        initialize = getattr(self, "_initialize_system_prompt", None)
        if callable(initialize):
            initialize()

    def restore_message_history(self, serialized_history: List[Dict[str, Any]]) -> None:
        """Restores the message history from a list of dictionaries using custom methods."""
        self.conversation.restore(serialized_history)

    def dump_compaction_state(self) -> Dict[str, Any]:
        """Serialize the compaction boundary (summary + how many leading messages it folds)."""
        return self.conversation.dump_compaction()

    def restore_compaction_state(self, data: Optional[Dict[str, Any]]) -> None:
        """Restore the compaction boundary; must be called after restore_message_history."""
        self.conversation.restore_compaction(data)

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

    def _load_project_guidance(self) -> tuple[str, str]:
        """Return the first project guidance file found and its content."""
        for guidance_file in PROJECT_GUIDANCE_FILES:
            if not self.filesystem.exists(guidance_file):
                continue
            try:
                return guidance_file, self.filesystem.read_text(guidance_file)
            except Exception:
                return guidance_file, ""
        return "", ""

    def _load_agent_memory(self) -> tuple[str, str]:
        """Return agent memory content when AGENT_MEMORY.md exists."""
        if not self.filesystem.exists(AGENT_MEMORY_FILE):
            return "", ""
        try:
            return AGENT_MEMORY_FILE, self.filesystem.read_text(AGENT_MEMORY_FILE)
        except Exception:
            return AGENT_MEMORY_FILE, ""

    def build_prompt_context(self) -> PromptContext:
        """Build PromptContext from agent state."""
        import platform

        # Check if it's a git repository
        is_git_repo = self.filesystem.exists(".git") and self.filesystem.is_dir(".git")

        project_guidance_file, project_guidance = self._load_project_guidance()
        agent_memory_file, agent_memory = self._load_agent_memory()

        return PromptContext(
            system_name=os.getenv("KOLEGA_CODE_SYSTEM_NAME", "Kolega Code"),
            project_path=str(self.project_path),
            is_git_repo=is_git_repo,
            platform=platform.system(),
            date_today=datetime.now().strftime("%Y-%m-%d"),
            model_name=self.primary_model_config.model,
            available_ports=self.available_ports,
            project_guidance=project_guidance,
            project_guidance_file=project_guidance_file,
            agent_memory=agent_memory,
            agent_memory_file=agent_memory_file,
            kolega_md=project_guidance,
            workspace_id=self.workspace_id,
            workspace_environment_variables=self.workspace_env_var_descriptions,
            memories=self.workspace_memories,
        )

    def _prompt_override_error_message(self, path: str, detail: object) -> str:
        return format_prompt_override_error(path, detail)

    def _add_prompt_override_error(self, message: str, *, emit_stderr: bool = False) -> None:
        if message in self.prompt_override_errors:
            return
        self.prompt_override_errors.append(message)
        logger.warning(message)
        if emit_stderr:
            print(f"kolega-code: {message}", file=sys.stderr)

    def _report_prompt_override_render_error(self, path: str, exc: Exception) -> None:
        self._add_prompt_override_error(
            self._prompt_override_error_message(path, exc),
            emit_stderr=True,
        )

    def validate_prompt_overrides(
        self,
        *,
        context: Optional[PromptContext] = None,
        mode: Optional[AgentMode] = None,
    ) -> None:
        """Validate all supported project prompt override files and collect diagnostics."""
        prompt_context = context or self.build_prompt_context()
        for diagnostic in self.prompt_overrides.validate_all(
            context=prompt_context,
            mode=mode or self.agent_mode,
            project_template_slug=self.project_template_slug,
            prompt_provider=self.prompt_provider,
        ):
            self._add_prompt_override_error(self._prompt_override_error_message(diagnostic.path, diagnostic.message))

    def build_agent_system_prompt(self, agent_type: AgentType, mode: Optional[AgentMode] = None) -> str:
        """Build the final system prompt for an agent, honoring project overrides."""
        context = self.build_prompt_context()
        override = self.prompt_overrides.load_agent_system_prompt(agent_type)
        if override is not None:
            try:
                base = render_prompt_override_source(
                    override.content,
                    context=context,
                    mode=mode,
                    project_template_slug=self.project_template_slug,
                    prompt_provider=self.prompt_provider,
                )
            except Exception as exc:  # noqa: BLE001 - bad project prompts should fall back safely
                self._report_prompt_override_render_error(override.path, exc)
            else:
                dynamic = self.prompt_provider.render_dynamic_sections(
                    agent_type,
                    mode,
                    self.prompt_extensions,
                    context,
                )
                self.validate_prompt_overrides(context=context, mode=mode)
                return "\n\n".join(part for part in (base, dynamic) if part)

        prompt = self.prompt_provider.get_system_prompt(
            agent_type=agent_type,
            mode=mode,
            template_slug=self.project_template_slug,
            prompt_extensions=self.prompt_extensions,
            context=context,
        )
        # The bundled planning template is intentionally base-only; append the
        # same dynamic sections that override prompts receive.
        if agent_type == AgentType.PLANNING:
            dynamic = self.prompt_provider.render_dynamic_sections(
                agent_type,
                mode,
                self.prompt_extensions,
                context,
            )
            prompt = "\n\n".join(part for part in (prompt.strip(), dynamic) if part)
        self.validate_prompt_overrides(context=context, mode=mode)
        return prompt

    def refresh_system_prompt(self) -> None:
        """Refresh this agent's system prompt after prompt files or extensions change."""
        initialize = getattr(self, "_initialize_system_prompt", None)
        if callable(initialize):
            initialize()

    # ------------------------------------------------------------------
    # Attachments
    # ------------------------------------------------------------------

    def _unsupported_attachment_message(self, attachments: Optional[List[Dict[str, Any]]]) -> Optional[str]:
        if not any(attachment.get("type") == "image" for attachment in attachments or []):
            return None

        provider = getattr(
            self.primary_model_config.provider,
            "value",
            self.primary_model_config.provider,
        )
        if model_supports_vision(provider, self.primary_model_config.model):
            return None

        return (
            f"{self.primary_model_config.model} does not support image input. "
            "Your message was not sent to the model. "
            "Remove the image attachment or switch to a vision-capable model with /model."
        )

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
        # History sent to the LLM (and to token counting): tool-call-repaired and,
        # for non-vision models, stripped of image blocks from earlier turns.
        fixed_history = self._history_for_llm()
        token_count = await self.llm.count_tokens(
            system=self.system_prompt,
            messages=fixed_history,
            model=self.primary_model_config.model,
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
                'You can start fresh by clicking "New Thread" in the sidebar.'
            )

        await self.emitter.context_update(
            input_tokens=token_count.input_tokens,
            model_context_length=self.model_context_length,
            compression_threshold=self.history_compression_threshold,
            alert_level=alert_level,
            message=message,
        )

    async def compress_history(self) -> CompactionResult:
        """
        Non-destructively summarize the current history, keeping recent turns
        verbatim. Emits a compaction_status event around the work so the UI can
        show progress, then recounts + emits a context_update so the gauge
        refreshes. Returns the structured outcome.
        """

        async def on_info(message: str) -> None:
            await self.log_info(message, sender=self.agent_name)

        async def on_error(message: str) -> None:
            await self.log_error(message, sender=self.agent_name)

        await self.emitter.compaction_status("started", "Compacting conversation…")
        try:
            compaction_system_prompt = None
            compaction_override = self.prompt_overrides.load_compaction_system_prompt()
            if compaction_override is not None:
                try:
                    compaction_system_prompt = render_prompt_override_source(
                        compaction_override.content,
                        context=self.build_prompt_context(),
                        mode=self.agent_mode,
                        project_template_slug=self.project_template_slug,
                        prompt_provider=self.prompt_provider,
                    )
                except Exception as exc:  # noqa: BLE001 - fall back to the bundled compaction prompt
                    self._report_prompt_override_render_error(compaction_override.path, exc)
            result = await self.compressor.summarize(
                self.conversation,
                llm=self.llm,
                model=self.primary_model_config.model,
                temperature=self.model_default_temperature,
                # No extended thinking: a bounded summary doesn't need it, and the
                # thinking budget could otherwise exceed the small summary max_tokens.
                thinking=None,
                on_info=on_info,
                on_error=on_error,
                system_prompt_text=compaction_system_prompt,
            )
        finally:
            # Recount + emit so the context gauge reflects post-compaction reality
            # (even on a no-op the UI may have been stale).
            await self.count_current_context()

        phase = "finished" if result.ok else "error"
        summary_text = self.conversation.summary.get_text_content() if result.ok and self.conversation.summary else ""
        await self.emitter.compaction_status(phase, result.message, summary=summary_text)
        return result

    def clear_history(self) -> None:
        """Drop all history and reset compaction state."""
        self.conversation.clear()

    # ------------------------------------------------------------------
    # Tool execution
    # ------------------------------------------------------------------

    def set_permission_mode(self, permission_mode: PermissionMode | str) -> None:
        """Update the active permission mode without rebuilding the agent."""
        self.permission_mode = normalize_permission_mode(permission_mode, default=self.permission_mode)
        self.context.permission_mode = self.permission_mode

    def set_permission_callback(self, permission_callback: Any) -> None:
        """Update the host callback used when permission mode is ask."""
        self.permission_callback = permission_callback or auto_allow_permission_callback
        self.context.permission_callback = self.permission_callback

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

            permission_request = permission_request_for_tool(tool_name, inputs)
            if permission_request is not None and self.permission_mode == PermissionMode.ASK:
                await self.fire_hook(
                    HookEvent.NOTIFICATION,
                    {
                        "notification_type": "permission_prompt",
                        "message": f"Permission requested for {tool_name}",
                        "tool_name": tool_name,
                    },
                    target="permission_prompt",
                )
                try:
                    decision = await self.permission_callback(permission_request)
                    if not isinstance(decision, PermissionDecision):
                        raise TypeError("permission callback must return PermissionDecision")
                except Exception as ex:
                    error_message = f"Permission check failed for {tool_name}: {ex}"
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

                if not decision.allowed:
                    reason = decision.reason or "The user denied permission for this action."
                    error_message = f"Permission denied for {tool_name}: {reason}"
                    await self.log_warning(error_message, sender=self.agent_name)
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

            # PreToolUse hooks: may deny the call (returned to the model as a tool
            # error, like a permission denial) or rewrite the tool inputs.
            pre = await self.fire_hook(
                HookEvent.PRE_TOOL_USE,
                {"tool_name": tool_name, "tool_input": inputs, "tool_use_id": tool_execution_id},
                target=tool_name,
            )
            if pre.blocked:
                return await self._blocked_tool_result(tool_name, provider_tool_call_id, tool_execution_id, pre.reason)
            if pre.updated_input is not None:
                inputs = pre.updated_input

            # Send tool_call message to indicate we're starting execution
            if not all(
                [
                    self.primary_model_config.provider == ModelProvider.ANTHROPIC,
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

            # PostToolUse hooks: may replace the output or append context the model sees.
            post = await self.fire_hook(
                HookEvent.POST_TOOL_USE,
                {
                    "tool_name": tool_name,
                    "tool_input": inputs,
                    "tool_output": self._hook_text(output),
                    "is_error": False,
                },
                target=tool_name,
            )
            output = self._apply_post_tool_hook(output, post)

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
            error_message = await self._post_tool_error_hook(tool_name, inputs, error_message)

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
            error_message = await self._post_tool_error_hook(tool_name, inputs, error_message)

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

    # ------------------------------------------------------------------
    # Lifecycle hooks
    # ------------------------------------------------------------------

    async def fire_hook(self, name: HookEvent, payload: Dict[str, Any], *, target: str = "") -> HookOutcome:
        """Build a LifecycleEvent and dispatch it. Returns an empty outcome when no
        hooks are configured (the common case) or when already inside a hook."""
        # Hot path: skip building the event/capabilities when no hooks exist.
        if not self.hook_dispatcher.is_active:
            return HookOutcome.empty()
        event = LifecycleEvent(
            name=name,
            payload=payload,
            session_id=self.thread_id,
            cwd=str(self.project_path),
            permission_mode=self.permission_mode.value if self.permission_mode else None,
        )
        return await self.hook_dispatcher.dispatch(event, target=target, caps=self._hook_capabilities())

    def _hook_capabilities(self) -> HookCapabilities:
        """Capabilities passed to hook backends: project cwd, an LLM prompt runner,
        a sub-agent runner (for `agent` hooks), and a log sink."""
        return HookCapabilities(
            project_path=self.project_path,
            prompt_runner=self._run_hook_prompt,
            agent_runner=self._run_hook_agent,
            log=self._log_hook_message,
        )

    async def _log_hook_message(self, message: str) -> None:
        await self.log_warning(message, sender=self.agent_name)

    async def _run_hook_prompt(self, prompt_text: str, model_hint: Optional[str]) -> str:
        """Run a `prompt` hook: a single completion on a chosen model slot.

        ``model_hint`` selects a configured slot ("fast" (default), "long", or
        "thinking"); arbitrary model ids are not used here to keep provider/API-key
        pairing correct across kolega's multi-provider setup.
        """
        slot = (model_hint or "fast").lower()
        if slot in ("long", "main", "long_context"):
            model_config = self.config.long_context_config
        elif slot == "thinking":
            model_config = self.config.thinking_config
        else:
            model_config = self.config.fast_config

        from kolega_code.llm.client import LLMClient

        client = LLMClient(
            provider=model_config.provider.value,
            api_key=self.config.get_api_key(model_config.provider),
            max_retries=model_config.rate_limits.max_retries,
            requests_per_minute=model_config.rate_limits.requests_per_minute,
            tokens_per_minute=model_config.rate_limits.tokens_per_minute,
            token_manager=self.config.get_chatgpt_token_manager(),
        )
        response = await client.generate(
            model=model_config.model,
            max_completion_tokens=512,
            system=Message(role="system", content=[TextBlock(text=HOOK_DECISION_SYSTEM_PROMPT)]),
            messages=MessageHistory([Message(role="user", content=[TextBlock(text=prompt_text)])]),
            temperature=0.0,
        )
        return response.get_text_content() or ""

    async def _run_hook_agent(self, task: str) -> str:
        """Run an `agent` hook: dispatch a full-tool sub-agent to verify a condition.

        Runs under the dispatcher's re-entrancy guard, so the sub-agent's own tool
        calls do not re-fire tool hooks.
        """
        if self.tool_collection is None or not hasattr(self.tool_collection, "agent_tool"):
            raise RuntimeError("agent hooks require a tool collection with sub-agent dispatch")
        instruction = (
            f"{task}\n\nWhen finished, end your reply with a single JSON object on its own line: "
            '{"ok": true} if the condition holds, or {"ok": false, "reason": "<why>"} if it does not.'
        )
        return await self.tool_collection.agent_tool.dispatch_general_agent(instruction)

    @staticmethod
    def _hook_text(output: Any) -> str:
        """Stringify a tool output (which may be a list of content blocks) for hook input."""
        if isinstance(output, list):
            return "\n\n".join(getattr(item, "to_markdown", lambda: str(item))() for item in output)
        return output if isinstance(output, str) else str(output)

    async def _blocked_tool_result(
        self, tool_name: str, provider_tool_call_id: str, tool_execution_id: str, reason: str
    ) -> ToolResult:
        """Build the is_error ToolResult a blocked tool produces — identical to the
        permission-deny path so the model and UI handle a hook block the same way."""
        error_message = (
            f"Permission denied for {tool_name}: {reason}" if reason else f"Permission denied for {tool_name}."
        )
        await self.log_warning(error_message, sender=self.agent_name)
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

    def _apply_post_tool_hook(self, output: Any, outcome: HookOutcome) -> Any:
        """Fold a PostToolUse outcome into the tool output the model will see."""
        if outcome.is_empty:
            return output
        if outcome.updated_output is not None:
            output = outcome.updated_output
        if outcome.additional_context:
            output = f"{self._hook_text(output)}\n\n{outcome.additional_context}"
        if outcome.blocked or outcome.end_turn:
            # End the turn after this tool batch; surface the reason as a warning line.
            self._hook_end_turn = True
            if outcome.reason:
                output = f"{self._hook_text(output)}\n\n[hook] {outcome.reason}"
        return output

    async def _post_tool_error_hook(self, tool_name: str, inputs: dict, error_message: str) -> str:
        """Fire PostToolUse for a failed tool. On error the hook may only append
        context (it must not mask or rewrite a genuine tool failure)."""
        post = await self.fire_hook(
            HookEvent.POST_TOOL_USE,
            {"tool_name": tool_name, "tool_input": inputs, "tool_output": error_message, "is_error": True},
            target=tool_name,
        )
        if post.additional_context:
            return f"{error_message}\n\n{post.additional_context}"
        return error_message

    async def _fire_stop_hook(self, stop_reason: Optional[str]) -> Optional[str]:
        """Fire the Stop event. Returns a 'keep working' instruction when a hook
        blocks the stop (ok:false), otherwise None."""
        try:
            last_message = await self.recap_agent_outcome()
        except Exception:  # noqa: BLE001 - recap is best-effort context for the hook
            last_message = ""
        outcome = await self.fire_hook(
            HookEvent.STOP,
            {"stop_reason": stop_reason, "last_message": last_message},
        )
        if outcome.blocked:
            return outcome.reason or "Continue working; the stop condition is not yet satisfied."
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

    @staticmethod
    def _parse_retry_after(error: Exception) -> Optional[float]:
        """Best-effort retry-after (seconds) from a RAW provider exception.

        Must be called before map_to_llm_error, which discards the response headers.
        Handles both the integer-seconds and HTTP-date forms; returns None on any miss.
        """
        raw = None
        response = getattr(error, "response", None)
        headers = getattr(response, "headers", None)
        if headers is not None:
            try:
                raw = headers.get("retry-after")
            except Exception:
                raw = None
        if raw is None:
            raw = getattr(error, "retry_after", None)
        if raw is None:
            return None
        try:
            return max(0.0, float(raw))
        except (TypeError, ValueError):
            pass
        try:
            retry_dt = parsedate_to_datetime(str(raw))
            return max(0.0, (retry_dt - datetime.now(retry_dt.tzinfo)).total_seconds())
        except Exception:
            return None

    async def handle_llm_error(self, error: Exception) -> None:
        """Centralized handling for LLM errors raised in the turn loop.

        Transient failures (rate-limit, overload/5xx, connection drops) are retried with
        bounded exponential backoff + jitter — honoring retry-after when present — up to
        ``loop_max_retries`` *consecutive* attempts, then surfaced cleanly. This is the
        fallback for failures the SDK's own retries didn't absorb (budget exhausted, or the
        failure happened mid-stream, which the SDK does not retry). Returning (not raising)
        lets the turn loop re-issue the identical request. Other LLM errors and non-LLM
        errors are terminal.
        """
        # Extract retry-after + HTTP status from the raw exception before mapping strips them.
        retry_after = self._parse_retry_after(error)
        status_code = getattr(error, "status_code", None) or getattr(error, "status", None)
        raw_type = type(error).__name__
        error = map_to_llm_error(error, provider=self.primary_model_config.provider.value)

        # Diagnostic-only structured record (the CLI persists it; the UI still uses
        # llm_status for the user-facing message). Makes a failed turn debuggable.
        try:
            await self.emitter.llm_error(
                provider=self.primary_model_config.provider.value,
                model=self.primary_model_config.model,
                endpoint=self.llm.provider.base_url,
                http_status=status_code,
                error_type=type(error).__name__,
                raw_type=raw_type,
                attempt=self._consecutive_llm_retries + 1,
                message=str(error)[:1000],
            )
        except Exception:
            pass

        # Retry transient failures: rate limits, provider 5xx/overload, and transport-layer
        # failures (LLMConnectionError, incl. its LLMTimeout subclass — e.g. a stalled
        # streaming read hitting the per-request timeout, or a dropped connection).
        if isinstance(error, (LLMRateLimitError, LLMInternalServerError, LLMConnectionError)):
            cap = self.primary_model_config.rate_limits.loop_max_retries
            self._consecutive_llm_retries += 1
            if self._consecutive_llm_retries > cap:
                await self.emitter.llm_status(
                    "error",
                    llm_error_message(error, model=self.primary_model_config.model),
                )
                raise error
            if retry_after is not None:
                delay = min(retry_after, 60.0)
            else:
                # Full jitter on capped exponential backoff de-correlates concurrent agents.
                backoff = min(30.0, 2.0 * (2 ** (self._consecutive_llm_retries - 1)))
                delay = random.uniform(0, backoff)
            await self.log_warning(
                f"Transient LLM error ({error}); retry {self._consecutive_llm_retries}/{cap} in {delay:.1f}s.",
                sender=self.agent_name,
            )
            # Surface the retry in the UI so a stalled stream (which now fails fast on the
            # per-request timeout instead of hanging) reads as "retrying", not a freeze.
            await self.emitter.llm_status(
                "info",
                f"Connection issue — retrying ({self._consecutive_llm_retries}/{cap}) in {delay:.0f}s…",
            )
            await asyncio.sleep(delay)
            await self.log_info("Resuming after backoff.", sender=self.agent_name)
            # Don't re-raise - the turn loop re-issues the request.

        elif isinstance(error, LLMError):
            await self.emitter.llm_status(
                "error",
                llm_error_message(error, model=self.primary_model_config.model),
            )
            raise error
        else:
            # Non-LLM error - just re-raise
            raise

    # ------------------------------------------------------------------
    # The agent loop
    #
    # process_message_stream is the single canonical loop shared by every
    # agent. Subclasses customize behavior through the hook methods below
    # (build_user_content, on_tool_use_start, should_stop_after_tools,
    # recap_agent_outcome) rather than overriding the loop itself.
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

        content_blocks = await self.build_user_content(message, attachments)

        # UserPromptSubmit hooks (skipped for sub-agents, whose task is machine-
        # generated, not a user prompt). May end the turn or inject extra context.
        if not self.sub_agent:
            submit = await self.fire_hook(HookEvent.USER_PROMPT_SUBMIT, {"user_message": message})
            if submit.blocked or submit.end_turn:
                reason = submit.reason or "A hook blocked this prompt."
                yield {"type": "response", "content": reason, "complete": True, "uuid": str(uuid.uuid4())}
                return
            if submit.additional_context:
                content_blocks.append(TextBlock(text=submit.additional_context))

        self.append_user_message(content_blocks)

        stop_reason = None
        stop_overrides = 0
        iterations = 0
        while stop_reason not in ["end_turn", "max_tokens", "stop_sequence"]:
            iterations += 1
            if self.max_iterations is not None and iterations > self.max_iterations:
                raise MaxAgentIterationsExceeded(
                    f"Agent '{self.agent_name}' exceeded max_iterations={self.max_iterations} "
                    "without reaching a terminal stop reason"
                )

            self.mark_cache_checkpoint()

            try:
                token_count = await self.count_current_context()
                logger.debug("Input token count: %s", token_count)

                if self.compressor.over_budget(token_count.input_tokens, self.model_context_length):
                    before_tokens = token_count.input_tokens
                    # PreCompact hooks (advisory): observe before history is compacted.
                    await self.fire_hook(
                        HookEvent.PRE_COMPACT,
                        {
                            "trigger": "auto",
                            "input_tokens": before_tokens,
                            "model_context_length": self.model_context_length,
                        },
                    )
                    result = await self.compress_history()
                    token_count = await self.count_current_context()

                    # PostCompact hooks (advisory): observe the outcome. There is no
                    # destructive fallback — a bounded summary plus capped tool results
                    # and a small verbatim tail keep us under budget; if somehow not, we
                    # send as-is rather than wipe history.
                    await self.fire_hook(
                        HookEvent.POST_COMPACT,
                        {
                            "trigger": "auto",
                            "ok": result.ok,
                            "reason": result.reason,
                            "summarized_messages": result.summarized_messages,
                            "input_tokens_before": before_tokens,
                            "input_tokens_after": token_count.input_tokens,
                            "model_context_length": self.model_context_length,
                        },
                    )

                    self.mark_cache_checkpoint()

                current_response = ""
                current_thinking = ""
                thinking_started = False
                # Use the same UUID for each segment of the response
                response_uuid = str(uuid.uuid4())
                thinking_uuid = str(uuid.uuid4())

                # History sent to the LLM: tool-call-repaired and, for non-vision
                # models, stripped of image blocks carried over from earlier turns.
                fixed_history = self._history_for_llm()

                # Diagnostics: bracket the request so a stall is visible in the timeline
                # (start↔end, or start↔llm_error if it fails) with the actual endpoint.
                _req_start = time.monotonic()
                await self.emitter.llm_request(
                    "start",
                    provider=self.primary_model_config.provider.value,
                    model=self.primary_model_config.model,
                    endpoint=self.llm.provider.base_url,
                )

                async with await self.llm.stream(
                    system=self.system_prompt,
                    max_completion_tokens=self.model_completion_tokens,
                    temperature=self.model_default_temperature,
                    messages=fixed_history,
                    model=self.primary_model_config.model,
                    tools=self.tool_collection.get_tool_list(),
                    thinking=self.primary_model_config.thinking_effort,
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
                await self.emitter.llm_request(
                    "end",
                    provider=self.primary_model_config.provider.value,
                    model=self.primary_model_config.model,
                    elapsed_s=round(time.monotonic() - _req_start, 2),
                    stop_reason=stop_reason,
                )

                self.append_assistant_message(assistant_message)
                # A clean stream resets the transient-failure budget, so the cap measures
                # only consecutive failures, not lifetime failures across the turn.
                self._consecutive_llm_retries = 0

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
                        if self._hook_end_turn:
                            # A blocking PostToolUse hook asked to end the turn.
                            self._hook_end_turn = False
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

                # Stop hooks (main agent only). On a natural turn end, a hook may
                # keep the agent working by blocking the stop and returning a reason.
                if stop_reason in ["end_turn", "max_tokens", "stop_sequence"] and not self.sub_agent:
                    keep_working = await self._fire_stop_hook(stop_reason)
                    if keep_working is not None and stop_overrides < self.MAX_STOP_HOOK_OVERRIDES:
                        stop_overrides += 1
                        self.append_user_message([TextBlock(text=keep_working)])
                        stop_reason = None

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
