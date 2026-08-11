import asyncio
import contextlib
import contextvars
from dataclasses import dataclass
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
from typing import TYPE_CHECKING, Any, AsyncGenerator, Awaitable, Callable, Dict, List, Optional, Sequence, Tuple, cast
from contextlib import AbstractAsyncContextManager
from collections.abc import Coroutine

from .common import LogMixin
from .compression import CompactionResult, HistoryCompressor
from .errors import MaxAgentIterationsExceeded
from .goal import GoalVerdict, build_goal_verifier_instruction, parse_goal_verdict
from kolega_code.config import AgentConfig, EditProtocol, ModelProvider
from kolega_code.events import AgentConnectionManager
from .context import AgentContext, AgentServices, Telemetry, WorkspaceInfo
from .conversation import Conversation, adapt_history_for_provider
from .volatile_context import VolatileContextTracker, VolatileSection
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
from kolega_code.llm.instrumented_client import get_output_tokens
from kolega_code.llm.ledger import HISTORY_ORIGIN, LlmCallOrigin, helper_origin, llm_call_origin
from kolega_code.llm.models import (
    ImageBlock,
    Message,
    MessageHistory,
    TextBlock,
    ToolCall,
    ToolResult,
    WebSearchCallBlock,
)
from kolega_code.llm.providers.models import TokenCount
from kolega_code.llm.specs import (
    deepseek_output_token_cap,
    get_model_specs,
    is_deepseek_model,
    supports_vision as model_supports_vision,
)
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
from kolega_code.services.file_system import FileSystem, LocalFileSystem
from .tools import ToolCollection, ToolExtension  # noqa: F401 - ToolCollection kept for downstream monkeypatching
from kolega_code.tools import ToolError
from .utils.commands import CommandProcessor

if TYPE_CHECKING:
    # Type hints only — langfuse is a heavy optional dependency kept off the
    # startup import path. Agents receive a Langfuse *instance* via the context.
    from langfuse import Langfuse
    from .orchestration.accounting import AgentReservation, WorkflowRunAccounting


logger = logging.getLogger(__name__)

PROJECT_GUIDANCE_FILES = ("AGENTS.md", "KOLEGA.md")

# System prompt for `prompt`/`agent` lifecycle hooks: the model's only job is a
# yes/no decision returned as a compact JSON object.
HOOK_DECISION_SYSTEM_PROMPT = (
    "You are a lifecycle hook that makes a single yes/no decision about an AI coding "
    "agent's action. You are given the event data and a question or condition to evaluate. "
    'Respond with ONLY a JSON object: {"ok": true} to allow the action to proceed, or '
    '{"ok": false, "reason": "<short explanation>"} to block it. The reason is shown to the '
    "agent. Output nothing other than the JSON object."
)


@dataclass
class QueuedUserInput:
    text: str
    attachments: Optional[List[Dict[str, Any]]] = None


def _image_payloads(blocks: List[Any]) -> List[Tuple[str, str]]:
    """``(media_type, base64_data)`` for the base64 images among ``blocks``.

    A tool that returns pictures — reading an image, a browser screenshot —
    describes them in markdown for the transcript, but the bytes themselves
    otherwise reach only the model. Lifting them out here is what lets the event
    stream carry the picture, and therefore what lets a replay show it.
    """
    payloads: List[Tuple[str, str]] = []
    for block in blocks:
        if not isinstance(block, ImageBlock) or block.image_type != "base64":
            continue
        if block.media_type and block.data:
            payloads.append((block.media_type, block.data))
    return payloads


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
    # The active system prompt Message; populated by ``_initialize_system_prompt``
    # (overridden by subclasses, e.g. CoderAgent) which wraps the prompt text in a
    # ``Message(role="system", ...)``. Declared here for type checkers; the base
    # ``__init__`` leaves subclasses to populate it.
    system_prompt: Message = Message(role="system", content=[TextBlock(text="")])
    # Tool collection; subclasses initialize this in ``__init__`` with the
    # appropriate tool configuration. ``None`` until then.
    tool_collection: Optional[ToolCollection] = None
    history_compression_threshold = 0.8
    # Cap on concurrently executing tool calls within one batch (each dispatched
    # sub-agent runs its own multi-turn LLM loop, so an unbounded fan-out would
    # multiply token spend and shared-resource pressure).
    PARALLEL_TOOL_LIMIT = 8
    # Cap on how many times a Stop hook may force the agent to keep working in one
    # turn, so a misbehaving "don't stop until X" hook cannot loop forever.
    MAX_STOP_HOOK_OVERRIDES = 5
    # Cap on consecutive re-prompts after a "silent turn": the model ended its
    # turn with no tool call and no visible text (empty or reasoning-only).
    # Finalizing such a turn reports nothing — in non-interactive modes that
    # reads as success with empty output — so the loop nudges the model to act
    # instead, and after the cap terminates the turn and surfaces the failure.
    # Runtime half of the coder_base.md.j2 invariant ("Thinking is invisible
    # and changes nothing on its own").
    MAX_SILENT_TURN_NUDGES = 3
    # Cap on consecutive continuation prompts after a truncated turn: the model
    # hit the output-token limit mid-reply (honest "max_tokens" stop, visible
    # text, no tool call). Unlike a silent turn there IS partial content, so
    # exhausting the cap finalizes the turn with what was produced instead of
    # reporting no result.
    MAX_TRUNCATED_TURN_NUDGES = 3
    # A DeepSeek response whose output lands within this many tokens of the wire
    # output cap without an honest "max_tokens" stop was likely truncated
    # server-side: DeepSeek reports its own cutoff as a clean finish (probed
    # 2026-08-03; see DEEPSEEK_WIRE_OUTPUT_CAP in llm/specs/accessors.py).
    DEEPSEEK_OUTPUT_CAP_SLACK = 1024
    long_content_tool_calls = ["write"]
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
        langfuse_client: Optional["Langfuse"] = None,
        user_id: Optional[str] = None,
        user_email: Optional[str] = None,
        project_template_slug: Optional[str] = None,
        protected_files: Optional[List[str]] = None,
        agent_mode: Optional[AgentMode] = None,
        workspace_env_var_descriptions: Optional[Dict[str, str]] = None,
        workspace_memories: Optional[List[str]] = None,
        prompt_provider: Optional[PromptProvider] = None,
        prompt_extensions: Optional[List[PromptExtension]] = None,
        tool_extensions: Optional[List[ToolExtension]] = None,
        permission_mode: Optional[PermissionMode | str] = None,
        permission_callback: Optional[Any] = None,
        usage_recorder: Optional[Any] = None,
        sub_agent_recorder: Optional[Any] = None,
        usage_ledger: Optional[Any] = None,
        llm_trace_sink: Optional[Any] = None,
        session_recorder: Optional[Any] = None,
        hook_dispatcher: Optional[HookDispatcher] = None,
        context: Optional[AgentContext] = None,
        max_iterations: Optional[int] = None,
        custom_agent_catalog: Optional[Any] = None,
        memory_manager: Optional[Any] = None,
        memory_project_path: Optional[Path] = None,
        memory_enabled: bool = True,
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
            session_recorder: Optional durable recorder for the top-level CLI session
            context: Pre-built AgentContext; takes precedence over the flat keywords
        """
        if context is None:
            if project_path is None or workspace_id is None or thread_id is None:
                raise TypeError("BaseAgent requires either an AgentContext or project_path/workspace_id/thread_id")
            assert config is not None, "config is required to build an AgentContext from flat kwargs"
            assert connection_manager is not None, (
                "connection_manager is required to build an AgentContext from flat kwargs"
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
                memory_manager=memory_manager,
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
                    usage_ledger=usage_ledger,
                    llm_trace_sink=llm_trace_sink,
                ),
                agent_mode=agent_mode,
                prompt_provider=prompt_provider,
                prompt_extensions=prompt_extensions or [],
                tool_extensions=tool_extensions or [],
                permission_mode=normalize_permission_mode(permission_mode, default=PermissionMode.AUTO),
                permission_callback=permission_callback or auto_allow_permission_callback,
            )
        else:
            if memory_manager is not None:
                context.services.memory_manager = memory_manager
            if prompt_provider is not None:
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
        resolved_edit_protocol = self.config.resolve_edit_protocol(self.primary_model_config)
        self.edit_protocol = (
            resolved_edit_protocol if isinstance(resolved_edit_protocol, EditProtocol) else EditProtocol.SEARCH_REPLACE
        )
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
        self.usage_ledger = context.telemetry.usage_ledger
        self.llm_trace_sink = context.telemetry.llm_trace_sink
        # Subagents start unrecorded; AgentTool assigns a scoped child recorder
        # post-construction when the parent session is being recorded.
        self.session_recorder = None if sub_agent else session_recorder
        self.custom_agent_catalog = custom_agent_catalog
        self.memory_manager = context.services.memory_manager
        self._owns_memory_manager = False
        if self.memory_manager is not None and sub_agent:
            from kolega_code.memory import MemoryAccessScope

            self.memory_manager = self.memory_manager.with_scope(MemoryAccessScope.SUBAGENT)
        elif self.memory_manager is None and not sub_agent:
            from kolega_code.cli.session_store import default_state_dir
            from kolega_code.memory import ProjectMemoryManager

            project = memory_project_path if memory_project_path is not None else self.project_path
            # "Memory off" has one representation everywhere: a present-but-disabled
            # manager, never a None. Memory is live only for a local project with the
            # per-run opt-out unset; a non-local/sandbox host (whose storage would be a
            # local state dir, disconnected from the remote project) or an explicit
            # `--no-memory-tools` opt-out gets an inert, storage-inert manager that
            # writes nothing. Dispatched sub-agents inherit whichever it is.
            if memory_enabled and isinstance(self.filesystem, LocalFileSystem):
                self.memory_manager = ProjectMemoryManager(project, default_state_dir())
            else:
                self.memory_manager = ProjectMemoryManager.disabled(project, default_state_dir())
            self.context.services.memory_manager = self.memory_manager
            self._owns_memory_manager = True

        # Agent-created Git worktrees live under .kolega/worktrees. Keep that
        # runtime-only subtree out of Git status through the clone-local shared
        # info/exclude file; linked worktrees inherit the same rule.
        if not sub_agent and isinstance(self.filesystem, LocalFileSystem):
            from kolega_code.worktrees import ensure_worktree_dir_ignored

            ensure_worktree_dir_ignored(self.project_path)

        # Session scratchpad: a per-session throwaway working directory under the
        # OS temp dir. The TUI injects the prompt extension itself (keyed by its
        # session id); other local hosts get it here, keyed by thread id.
        # Sandbox/E2B hosts are unchanged, as are sub-agents, which inherit the
        # section and the resolved directory from their parent.
        self.scratchpad_dir: Optional[Path] = None
        if not sub_agent and isinstance(self.filesystem, LocalFileSystem):
            from kolega_code.scratchpad import SCRATCHPAD_PROMPT_EXTENSION_ID, ensure_scratchpad_dir

            extensions = self.prompt_extensions or []
            if not any(getattr(ext, "id", None) == SCRATCHPAD_PROMPT_EXTENSION_ID for ext in extensions):
                try:
                    scratchpad_path = ensure_scratchpad_dir(self.project_path, self.thread_id)
                except (OSError, ValueError):
                    scratchpad_path = None
                if scratchpad_path is not None:
                    from .prompts import build_scratchpad_prompt

                    self.prompt_extensions = [
                        *extensions,
                        PromptExtension(
                            id=SCRATCHPAD_PROMPT_EXTENSION_ID,
                            title="Session Scratchpad",
                            markdown=build_scratchpad_prompt(scratchpad_path),
                            # No mode restriction: the fallback serves any local host.
                            propagate_to_sub_agents=True,
                        ),
                    ]
                    self.scratchpad_dir = scratchpad_path

        # gigacode (workflow orchestration) opt-in. Off by default; the host toggles
        # it via apply_gigacode(). The run_workflow tool gate reads this live, so
        # toggling takes effect on the next turn without rebuilding the agent.
        self.gigacode_enabled = False

        # Active autonomous goal condition (set/cleared by the host via apply_goal),
        # or None when no goal is active. Kept on the agent so delegated sub-agents
        # and the system prompt can stay goal-aware across turns.
        self.active_goal_condition: Optional[str] = None

        # Whether a scheduled ``/loop`` iteration drives the current turns (set by
        # the host via apply_loop). Mirrors ``active_goal_condition``: it exists so
        # the system prompt can tell the model the user is probably away.
        self.loop_active: bool = False

        self.prompt_override_errors: List[str] = []

        self.available_ports = "9001-9999"

        # Validate that the project path exists and is a directory using the filesystem
        if not self.filesystem.exists("."):
            raise ValueError(f"Project path does not exist: {self.project_path}")
        if not self.filesystem.is_dir("."):
            raise ValueError(f"Project path is not a directory: {self.project_path}")

        self.prompt_provider = context.prompt_provider or PromptProvider()
        self.prompt_overrides = ProjectPromptOverrides(self.filesystem)

        # Memory, repository guidance, and the date are injected into the conversation when they
        # change instead of being rendered into the system prompt, which keeps the cached prefix
        # stable for the whole session. See agent/volatile_context.py.
        self._volatile_context = VolatileContextTracker()
        # Host-registered volatile-context sections (e.g. a plan handle or the shared task
        # list). Providers run on every turn; an empty text (or None) means the section is
        # currently absent. See add_volatile_section.
        self._extra_volatile_sections: List[Callable[[], Optional[VolatileSection]]] = []

        self.conversation = Conversation(max_tool_result_chars=self.max_tool_result_chars_in_history)
        self.conversation.skill_content_pattern = self.skill_content_pattern
        # Counts consecutive transient LLM failures (rate-limit / overload) so the turn loop
        # backs off and eventually gives up instead of retrying forever; reset on any good turn.
        self._consecutive_llm_retries = 0
        # Host-configured compression threshold (fraction of the context window) shadows
        # the class default when the config carries one; every read below and in
        # _send_context_update goes through the instance attribute.
        config_threshold = getattr(self.config, "history_compression_threshold", None)
        if config_threshold is not None:
            self.history_compression_threshold = config_threshold
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
        # The clamped cap DeepSeek requests carry on the wire (None for other
        # models). Used to flag probable silent server-side truncation — see the
        # at-ceiling warning in process_message_stream.
        self.deepseek_wire_output_cap = (
            deepseek_output_token_cap(
                self.primary_model_config.provider,
                self.primary_model_config.model,
                self.model_completion_tokens,
            )
            if is_deepseek_model(self.primary_model_config.model)
            else None
        )
        # Whether this agent's primary model can accept image input. Read by the
        # ToolCollection read_image tool gate (so non-vision models never see the
        # tool) and used by _unsupported_attachment_message to reject image
        # attachments for non-vision models with a clear message.
        self.supports_vision = bool(model_specs.get("supports_vision", False))
        # Web tool mode: whether the provider's hosted (server-side) web_search
        # tool is requested and whether the client-side web_search/web_fetch
        # tools are registered. Resolved from config.web_search_mode plus the
        # model's supports_hosted_web_search catalog flag; the ToolCollection
        # gate reads client_web_tools_enabled, the stream call reads
        # hosted_web_search_active.
        self.supports_hosted_web_search = bool(model_specs.get("supports_hosted_web_search", False))
        self.web_search_mode = getattr(self.config, "web_search_mode", "auto") or "auto"
        self._apply_web_search_state()

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
        # Private workflow lifecycle state. These references never enter
        # sub_agent_context or any serialized agent metadata.
        self._workflow_accounting: Optional["WorkflowRunAccounting"] = None
        self._accounting_reservation: Optional["AgentReservation"] = None
        self.total_tokens_used: int = 0
        # Hosted web-search accounting side-channel. Searched/fetched content is
        # injected into the model's context SERVER-side (restored on replay of
        # web_search_call items, billed as input) and never reaches the client,
        # so tiktoken-over-history can't see it. The residual — billed input
        # minus the client-side count of the request actually sent — measures it
        # and is added to the gauge in count_current_context. Updated by
        # _update_hosted_search_residual; zeroed by compress_history.
        self._hosted_injected_tokens: int = 0
        self._last_raw_context_count: Optional[int] = None
        # Set by a blocking PostToolUse hook to end the turn after the current tool batch.
        self._hook_end_turn = False
        self.queued_input_provider: Optional[Callable[[], Awaitable[List[QueuedUserInput]]]] = None

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

    async def _record_context_user_message(self, text: str) -> None:
        """Journal and append a runtime-injected user message (not typed by the user)."""
        if self.session_recorder is not None:
            await asyncio.to_thread(
                self.session_recorder.record_context_message,
                Message(role="user", content=[TextBlock(text=text)]),
            )
        self.append_user_message([TextBlock(text=text)])

    async def _record_synthetic_notice(self, text: str, notice_code: str) -> None:
        """Journal a deterministic assistant notice that never enters history."""
        if self.session_recorder is not None:
            await asyncio.to_thread(
                self.session_recorder.record_synthetic_assistant,
                text,
                notice_code=notice_code,
            )

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
        return self._finalize_history_for_llm(self.get_effective_history_for_llm())

    async def _history_for_llm_async(self) -> MessageHistory:
        """``_history_for_llm`` with the repair+adapt pass off the event loop.

        Adapting history images for Anthropic decodes and re-encodes
        screenshots with PIL. Run inline on the event loop this froze the TUI
        for seconds per request, once per concurrently running (sub-)agent.
        The effective-history snapshot is taken on the loop so the worker
        thread never observes a concurrent history mutation.
        """
        effective = self.get_effective_history_for_llm()
        return await asyncio.to_thread(self._finalize_history_for_llm, effective)

    def _finalize_history_for_llm(self, effective: MessageHistory) -> MessageHistory:
        fixed = self.fix_incomplete_tool_calls(list(effective))
        provider = getattr(self.primary_model_config.provider, "value", self.primary_model_config.provider)
        fixed = adapt_history_for_provider(
            fixed,
            target_provider=str(provider),
            target_model=self.primary_model_config.model,
            supports_vision=self.supports_vision,
            target_edit_protocol=self.edit_protocol,
        )
        return MessageHistory(fixed)

    def _normalize_freeform_tool_calls(self, message: Message) -> None:
        """Normalize JSON fallback envelopes before calls enter stored history."""
        if not message.tool_calls or self.tool_collection is None:
            return
        registry = self.tool_collection.registry()
        for call in message.tool_calls:
            if call.name not in registry:
                continue
            definition = registry.get(call.name).definition
            if definition.input_kind != "freeform":
                continue
            if isinstance(call.input, dict):
                raw = call.input.get("input")
                if isinstance(raw, str):
                    call.input = raw
            call.input_kind = "freeform"

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

    def _apply_web_search_state(self) -> None:
        """Resolve web_search_mode + model capability into the two live gates.

        ``hosted_web_search_active`` — request the provider's server-side
        web_search tool on main-loop stream calls. ``client_web_tools_enabled``
        — register the client-side web_search/web_fetch tools (read by the
        ToolCollection gate). Hosted active always excludes the client tools:
        two search paths confuse the model and waste schema tokens.
        """
        mode = (self.web_search_mode or "auto").lower()
        supported = self.supports_hosted_web_search
        if mode == "off":
            hosted, client = False, False
        elif mode == "client":
            hosted, client = False, True
        elif mode == "hosted":
            if not supported:
                provider = self.primary_model_config.provider
                logger.warning(
                    "web_search_mode=hosted, but %s/%s has no hosted web search; falling back to the client tools",
                    getattr(provider, "value", provider),
                    self.primary_model_config.model,
                )
            hosted, client = supported, not supported
        else:  # auto
            hosted, client = supported, not supported
        self.hosted_web_search_active = hosted
        self.client_web_tools_enabled = client

    def apply_web_search_mode(self, mode: str) -> None:
        """Change the web tool mode mid-session (TUI ``/web-search``).

        Mirrors :meth:`apply_gigacode`: the tool registry rebuilds per call and
        the next stream call reads the hosted gate, so no cache invalidation is
        needed. No prompt work — the tools carry their own affordances.
        """
        self.web_search_mode = (mode or "auto").lower()
        self._apply_web_search_state()

    async def _emit_hosted_tool_call(self, delta: Dict[str, Any]) -> None:
        """Render a completed hosted web_search call as a tool_call/tool_result pair.

        The call executed on the provider's servers; there is no local tool
        invocation and the searched content never reaches the client (it is
        injected into the model's context server-side). The shared item id
        correlates the pair in every transcript consumer.
        """
        block = WebSearchCallBlock(
            item_id=delta.get("id") or None,
            status=delta.get("status"),
            action=delta.get("action") or {},
        )
        tool_call_id = block.item_id or str(uuid.uuid4())
        await self.send_chat_message(
            message_type="tool_call",
            content=f"Calling {WebSearchCallBlock.TOOL_LABEL}: {block.label()}",
            is_streaming=False,
            tool_description=WebSearchCallBlock.TOOL_LABEL,
            tool_call_id=tool_call_id,
        )
        await self.send_chat_message(
            message_type="tool_result",
            content=block.result_summary(),
            is_streaming=False,
            tool_description=WebSearchCallBlock.TOOL_LABEL,
            tool_call_id=tool_call_id,
        )

    def apply_goal(self, condition: Optional[str], prompt_extension: Optional["PromptExtension"] = None) -> None:
        """Set, replace, or clear the active autonomous goal for this session.

        Mirrors :meth:`apply_gigacode`: swaps the ``cli-active-goal`` prompt
        extension and refreshes the system prompt so the next turn is (or stops
        being) goal-aware. Safe to call mid-session.
        """
        self.active_goal_condition = condition
        extensions = [ext for ext in (self.prompt_extensions or []) if getattr(ext, "id", None) != "cli-active-goal"]
        if condition and prompt_extension is not None:
            extensions.append(prompt_extension)
        self.prompt_extensions = extensions
        initialize = getattr(self, "_initialize_system_prompt", None)
        if callable(initialize):
            initialize()

    def apply_loop(self, active: bool, prompt_extension: Optional["PromptExtension"] = None) -> None:
        """Start or stop scheduled-loop awareness for this session.

        Mirrors :meth:`apply_goal`: swaps the ``cli-active-loop`` prompt extension
        and refreshes the system prompt so the next turn knows it was started by a
        timer rather than a person. Safe to call mid-session.
        """
        self.loop_active = active
        extensions = [ext for ext in (self.prompt_extensions or []) if getattr(ext, "id", None) != "cli-active-loop"]
        if active and prompt_extension is not None:
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

    def _is_history_valid_for_anthropic(self, messages: Optional[List[Message]] = None) -> bool:
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

    def build_prompt_context(self) -> PromptContext:
        """Build PromptContext from agent state."""
        import platform

        # Check if it's a git repository
        is_git_repo = self.filesystem.exists(".git") and self.filesystem.is_dir(".git")

        project_guidance_file, project_guidance = self._load_project_guidance()
        # ``private_memory`` (policy + the memory itself) is still populated for callers that
        # render their own prompts, but the system prompt renders only ``memory_policy``: the
        # memory body changes as the agent writes, and volatile content in the system prompt
        # invalidates the whole conversation's cached prefix. See agent/volatile_context.py.
        private_memory = ""
        memory_policy = ""
        if self.memory_manager is not None:
            try:
                memory_context = self.memory_manager.prompt_context()
                private_memory = memory_context.text
                memory_policy = memory_context.policy
            except Exception:
                # Disabled/unavailable memory is intentionally invisible to the model.
                private_memory = ""
                memory_policy = ""

        return PromptContext(
            system_name=os.getenv("KOLEGA_CODE_SYSTEM_NAME", "Kolega Code"),
            project_path=str(self.project_path),
            is_git_repo=is_git_repo,
            platform=platform.system(),
            date_today=datetime.now().strftime("%Y-%m-%d"),
            model_name=self.primary_model_config.model,
            model_supports_vision=self.supports_vision,
            available_ports=self.available_ports,
            project_guidance=project_guidance,
            project_guidance_file=project_guidance_file,
            kolega_md=project_guidance,
            workspace_id=self.workspace_id,
            workspace_environment_variables=self.workspace_env_var_descriptions,
            memories=self.workspace_memories,
            private_memory=private_memory,
            memory_policy=memory_policy,
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

    @staticmethod
    def system_prompt_message(text: str) -> Message:
        """Wrap system prompt text with a long-lived cache breakpoint.

        The prompt is byte-stable for the whole session now that volatile content is injected
        into the conversation instead, which makes a breakpoint of its own worth placing: it
        renders after the tools and before every message, so without one the prompt is re-billed
        on any miss further down. The 1h TTL matches the tool list — an interactive session
        routinely idles longer than the default five minutes.
        """
        return Message(role="system", content=[TextBlock(text=text, cache_checkpoint=True, cache_ttl="1h")])

    def _initialize_system_prompt(self) -> None:
        """Rebuild ``self.system_prompt`` from the current prompt configuration.

        Subclasses with a real prompt (e.g. ``CoderAgent``) override this to
        populate ``system_prompt`` via ``build_agent_system_prompt``. The base
        implementation is a no-op so the attribute is always declared on the
        class for type checkers and ``getattr``-based callers stay safe.
        """
        return None

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

    async def count_current_context(self, fixed_history: Optional[MessageHistory] = None) -> TokenCount:
        """Count the current context's tokens and emit a context-budget update.

        Building the request history (``_history_for_llm``: tool-call repair +
        provider adaptation, and image stripping for non-vision models) is an
        O(history) pass that runs on the event loop. The agent loop already builds
        that history for the request, so it passes it in here as ``fixed_history`` to
        avoid rebuilding it a second time per iteration. Callers that only need a
        count (``/context``, ``/compact``) omit it and the history is built here.
        """
        if fixed_history is None:
            self._sanitize_oversized_tool_results()
            # History sent to the LLM (and to token counting): tool-call-repaired and,
            # for non-vision models, stripped of image blocks from earlier turns.
            fixed_history = await self._history_for_llm_async()
        assert self.tool_collection is not None, "tool_collection must be initialized before counting context"
        token_count = await self.llm.count_tokens(
            system=self.system_prompt,
            messages=fixed_history,
            model=self.primary_model_config.model,
            tools=self.tool_collection.get_tool_list(),
        )
        # Hosted web-search content lives only in the server's copy of the
        # context (restored on replay of the web_search_call items, billed as
        # input) — the client history has nothing to count for it. Add the
        # residual measured from billed usage so the gauge AND the compaction
        # check both see the true context size. Nonzero only while un-compacted
        # web_search_call blocks exist (see _update_hosted_search_residual).
        self._last_raw_context_count = token_count.input_tokens
        if self._hosted_injected_tokens > 0:
            token_count.input_tokens += self._hosted_injected_tokens

        # Send context update event
        await self._send_context_update(token_count)

        return token_count

    # Provisional residual bump per hosted web-search call, applied while the
    # search-bearing response's own billing is telescoped (a response that
    # searched is several internal invocations billed cumulatively, so its
    # billed−counted residual overstates the persistent injected size). The next
    # clean response replaces the estimate with the measured residual. Probe
    # calibration 2026-08-04 (findings/probes/hosted_web_search_probe.py):
    # one-call persistent overhead ≈850 (deepseek) / ≈5000 (openai) tokens —
    # 6000 overestimates safely, and overestimating only compacts early.
    HOSTED_SEARCH_PROVISIONAL_TOKENS = 6000

    def _update_hosted_search_residual(self, assistant_message: Message) -> None:
        """Track server-injected hosted-search content from billed usage.

        Search-bearing responses bump the estimate by a provisional per-call
        constant (their own billing telescopes internal rounds, so the direct
        residual would overshoot and could trip compaction spuriously). Clean
        responses — single invocation, so ``billed − counted`` is exact —
        replace it with the measurement. No-ops for sessions that never
        searched, keeping the known ±tokenizer-drift band out of the gauge.
        """
        content = assistant_message.content if isinstance(assistant_message.content, list) else []
        new_calls = sum(1 for block in content if isinstance(block, WebSearchCallBlock))
        if new_calls:
            self._hosted_injected_tokens += self.HOSTED_SEARCH_PROVISIONAL_TOKENS * new_calls
            return
        if self._hosted_injected_tokens <= 0 and not self._history_contains_hosted_search():
            return
        billed = (assistant_message.usage_metadata or {}).get("prompt_tokens")
        counted = self._last_raw_context_count
        if not isinstance(billed, int) or counted is None:
            return
        self._hosted_injected_tokens = max(0, billed - counted)

    def _history_contains_hosted_search(self) -> bool:
        for message in self.history:
            content = getattr(message, "content", None)
            if isinstance(content, list) and any(isinstance(block, WebSearchCallBlock) for block in content):
                return True
        return False

    async def _send_context_update(self, token_count: TokenCount) -> None:
        """Send an event to update the UI about current context usage."""
        usage_percentage = (token_count.input_tokens / self.model_context_length) * 100

        # Determine alert level based on usage
        alert_level = "normal"
        message = None

        if usage_percentage >= 60:
            alert_level = "info"
            message = f"Contents will be compressed automatically at {self.history_compression_threshold * 100:.0f}%."

        await self.emitter.context_update(
            input_tokens=token_count.input_tokens,
            model_context_length=self.model_context_length,
            compression_threshold=self.history_compression_threshold,
            alert_level=alert_level,
            message=message,
        )

    async def compress_history(
        self, *, trigger: str = "manual", tokens_before: Optional[int] = None
    ) -> CompactionResult:
        """
        Non-destructively summarize the current history, keeping recent turns
        verbatim. Emits a compaction_status event around the work so the UI can
        show progress, then recounts + emits a context_update so the gauge
        refreshes. Returns the structured outcome.

        ``trigger``/``tokens_before`` are journaled as compaction provenance.
        """
        # Compacted-away web_search_call blocks stop being replayed, so the
        # server stops restoring their content. Reset the hosted-search residual
        # rather than carrying a stale figure; if search blocks survive in the
        # verbatim tail, the next clean response re-measures it.
        self._hosted_injected_tokens = 0

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
            if result.ok:
                # Compaction summarizes the injected volatile-context blocks along with
                # everything else, so memory, guidance, and any host sections (plan handle,
                # task list) may survive only in lossy summary form. Forget the sent-state
                # so the next turn re-sends them. Done here — not at the call sites — so
                # the manual /compact command gets the same re-injection as auto-compaction.
                self._volatile_context.forget()
                if self.session_recorder is not None:
                    await asyncio.to_thread(
                        self.session_recorder.record_compaction,
                        self.dump_compaction_state(),
                        info={
                            "trigger": trigger,
                            "reason": result.reason,
                            "summarized_messages": result.summarized_messages,
                            "input_tokens_before": tokens_before,
                        },
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
        if self.session_recorder is not None:
            self.session_recorder.start_epoch("agent_clear_command")
        self.conversation.clear()
        # The model no longer holds any injected context; re-send the current sections
        # (memory, guidance, plan handle, task list) on the next turn.
        self.reset_volatile_context()

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

    def set_queued_input_provider(self, provider: Callable[[], Awaitable[List[QueuedUserInput]]]) -> None:
        """Set the host-supplied async callable that drains queued UI messages.

        It drains user messages queued while a turn runs and is wired on the
        main agent only, never on sub-agents.
        """
        self.queued_input_provider = provider

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
        inputs = (
            {"input": tool_use_block.input}
            if tool_use_block.input_kind == "freeform" and isinstance(tool_use_block.input, str)
            else tool_use_block.input
        )
        if not isinstance(inputs, dict):
            inputs = {"input": str(inputs)}
        provider_tool_call_id = tool_use_block.id
        tool_execution_id = getattr(tool_use_block, "execution_id", provider_tool_call_id)

        # Keep provider IDs for LLM history while exposing an internal unique ID to app services.
        self.current_provider_tool_call_id = provider_tool_call_id
        self.current_tool_execution_id = tool_execution_id
        self.current_tool_call_id = tool_execution_id

        assert self.tool_collection is not None, "tool_collection must be initialized before executing tools"
        try:
            registry = self.tool_collection.registry()
            if tool_name not in registry:
                short_message = f"Tool '{tool_name}' is not available in this mode."
                # The model gets the available-tool list so it can self-correct;
                # the transcript row keeps the short form.
                error_message = f"{short_message} Available tools: {', '.join(sorted(registry.names()))}."
                await self.log_error(short_message, sender=self.agent_name)
                await self.send_chat_message(
                    message_type="tool_error",
                    content=short_message,
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
            images: List[Tuple[str, str]] = []
            if isinstance(output, list):
                chat_message_content = "\n\n".join(item.to_markdown() for item in output)
                images = _image_payloads(output)

            # Send tool_result message for successful execution
            await self.send_chat_message(
                message_type="tool_result",
                content=chat_message_content,
                is_streaming=False,
                tool_description=tool_name,
                tool_call_id=tool_execution_id,
                images=images,
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
        # Exclusive session-control tools must run in their own model round.
        # Reject the complete batch before hooks or handlers run so another
        # call cannot mutate state before the control operation.
        if len(tool_use_blocks) > 1:
            assert self.tool_collection is not None, "tool_collection must be initialized before processing tool calls"
            exclusive_tools = getattr(self.tool_collection, "exclusive_tools", frozenset())
            batched_exclusive = sorted({block.name for block in tool_use_blocks if block.name in exclusive_tools})
            if batched_exclusive:
                names = ", ".join(f"`{name}`" for name in batched_exclusive)
                error_message = (
                    f"Exclusive tool {names} must be called by itself. No tools in this batch were executed. "
                    "Call the exclusive tool in a separate tool round, then make any other calls afterward."
                )
                await self.log_warning(error_message, sender=self.agent_name)
                results: list[ToolResult] = []
                for block in tool_use_blocks:
                    execution_id = block.execution_id or f"tool_{block.id}"
                    await self.send_chat_message(
                        message_type="tool_error",
                        content=error_message,
                        is_streaming=False,
                        tool_description=block.name,
                        tool_call_id=execution_id,
                    )
                    results.append(
                        ToolResult(
                            tool_use_id=block.id,
                            content=error_message,
                            name=block.name,
                            is_error=True,
                            execution_id=execution_id,
                            input_kind=block.input_kind,
                        )
                    )
                return results

        # If only one tool call, just execute it directly
        if len(tool_use_blocks) == 1:
            result = await self.execute_single_tool(tool_use_blocks[0])
            result.input_kind = tool_use_blocks[0].input_kind
            return [result]

        assert self.tool_collection is not None, "tool_collection must be initialized before processing tool calls"
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
                    result = await self.execute_single_tool(block)
                    result.input_kind = block.input_kind
                    return result

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
                result.input_kind = block.input_kind
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

    def _main_loop_origin(self) -> LlmCallOrigin:
        """Attribution for this agent's main-loop LLM calls.

        HISTORY marks exactly the responses the session recorder journals as
        ``assistant.message`` (the same ``session_recorder is not None``
        condition guards ``record_assistant``), so the persistence sink skips
        them and no response is ever journaled twice.
        """
        if self.session_recorder is not None:
            if self.sub_agent:
                # Scoped subagent recording journals these responses too; keep
                # the history dedup rule but preserve the agent lineage for
                # failure attribution and usage breakdowns.
                stamp = self.session_recorder.agent_stamp
                return LlmCallOrigin(
                    kind="history",
                    agent_name=stamp.agent_name,
                    agent_id=stamp.agent_id,
                    parent_tool_call_id=stamp.parent_tool_call_id,
                    depth=stamp.depth,
                )
            return HISTORY_ORIGIN
        if self.sub_agent:
            context = getattr(self, "sub_agent_context", None) or {}
            return LlmCallOrigin(
                kind="sub_agent",
                agent_name=self.agent_name,
                agent_id=context.get("agent_id"),
                parent_tool_call_id=context.get("parent_tool_call_id") or getattr(self, "parent_tool_call_id", None),
                depth=context.get("depth"),
            )
        return LlmCallOrigin(kind="primary", agent_name=self.agent_name)

    async def _run_hook_prompt(self, prompt_text: str, model_hint: Optional[str]) -> str:
        """Run a `prompt` hook: a single completion on a chosen model slot.

        ``model_hint`` selects a configured slot ("fast" (default) or "long");
        arbitrary model ids are not used here to keep provider/API-key
        pairing correct across kolega's multi-provider setup.
        """
        slot = (model_hint or "fast").lower()
        if slot in ("long", "main", "long_context"):
            model_config = self.config.long_context_config
        else:
            model_config = self.config.fast_config

        from kolega_code.llm.client import LLMClient

        client = LLMClient(
            provider=model_config.provider.value,
            api_key=self.config.get_api_key(model_config.provider) or "",
            model=model_config.model,
            max_retries=model_config.rate_limits.max_retries,
            requests_per_minute=model_config.rate_limits.requests_per_minute,
            tokens_per_minute=model_config.rate_limits.tokens_per_minute,
            token_manager=self.config.get_chatgpt_token_manager(),
            usage_ledger=self.usage_ledger,
            trace_sink=self.llm_trace_sink,
        )
        with llm_call_origin(helper_origin("hook_prompt")):
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

    async def evaluate_goal_condition(self, condition: str) -> GoalVerdict:
        """Dispatch a read-only investigation sub-agent to verify whether the goal is met.

        The verifier runs on the configured long-context model (the investigation
        role inherits ``long_context_config``) and can read files, search, and run
        commands/tests, but cannot edit — so it cannot game an autonomous goal. It
        inspects the current codebase state fresh each call (stateless across
        evaluations). A dispatch failure or malformed verdict is reported as
        not-met rather than raising, so the goal loop keeps running safely.
        """
        if self.tool_collection is None or not hasattr(self.tool_collection, "agent_tool"):
            raise RuntimeError("goal evaluation requires a tool collection with sub-agent dispatch")
        instruction = build_goal_verifier_instruction(condition)
        try:
            result = await self.tool_collection.agent_tool.dispatch_investigation_agent(instruction)
        except Exception as exc:  # noqa: BLE001 - a verifier failure must not crash the loop
            return GoalVerdict(met=False, reason=f"verifier error: {exc}")
        return parse_goal_verdict(result)

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
        self,
        message_type: str,
        content: str,
        is_streaming: bool = False,
        tool_description=None,
        tool_call_id=None,
        images: Optional[Sequence[Tuple[str, str]]] = None,
    ) -> None:
        """
        Send a message to the chat interface.

        Args:
            content: The message content to send
            is_streaming: Whether this is part of a streaming message
            images: ``(media_type, base64_data)`` pairs a tool produced, so a
                frontend can show the picture rather than a description of one
        """
        await self.emitter.chat(
            message_type,
            content,
            is_streaming=is_streaming,
            tool_description=tool_description,
            tool_call_id=tool_call_id,
            images=images,
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

    @staticmethod
    def _is_silent_turn(message: Message) -> bool:
        """True when an assistant message has no tool calls and no visible text.

        ThinkingBlock carries ``.thinking`` rather than ``.text``, so a
        reasoning-only message reads as empty here. Evaluate the streamed
        message, not history: Conversation.append_assistant replaces empty
        content with a placeholder TextBlock.
        """
        return not message.tool_calls and not message.get_text_content().strip()

    async def _deliver_queued_user_inputs(self) -> None:
        """Append user inputs drained from the host while the current turn is running."""
        if self.queued_input_provider is None:
            return

        inputs = await self.queued_input_provider()
        if not inputs:
            return

        for item in inputs:
            blocks = await self.build_user_content(item.text, item.attachments)
            if self.session_recorder is not None:
                await asyncio.to_thread(
                    self.session_recorder.record_context_message,
                    Message(role="user", content=blocks),
                )
            self.append_user_message(blocks)

        await self.log_info(
            f"Delivered {len(inputs)} queued user message(s)",
            sender=self.agent_name,
        )

    async def process_message_stream(
        self, message: str, attachments: Optional[List[Dict[str, Any]]] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Run one durable top-level turn and record its terminal outcome.

        Chunks yielded to the caller are also mirrored onto the event stream.
        Assistant prose and reasoning historically travelled *only* through this
        generator, which meant the event stream — the thing every other frontend
        and the replay player render from — contained tool activity and status but
        no actual conversation. Mirroring here rather than at each ``yield`` keeps
        one choke point that cannot drift out of sync with the yields.
        """
        turn = self._run_recorded_turn(self._process_message_stream_impl(message, attachments), user_text=message)
        async with contextlib.aclosing(turn):
            async for chunk in turn:
                yield chunk

    async def continue_from_history_stream(self) -> AsyncGenerator[Dict[str, Any], None]:
        """Continue the ordinary agent loop without inserting a user message.

        Requires an already valid conversation ending at a point from which the
        assistant should act — commonly an assistant tool call followed by its
        appended matching tool result. ``restore_message_history``,
        ``restore_compaction_state``, and ``append_user_message`` (with a
        ``ToolResult`` matched to the restored ``ToolCall``) are the public
        preparation surface.

        Runs the same compaction checks, request construction, model streaming,
        tool execution, iteration limits, stop handling, retries, session
        recording, usage accounting, and cancellation/terminal bookkeeping as a
        normal turn, and yields the same chunk format as
        ``process_message_stream``. Adds no user text, attachment,
        volatile-context turn, or prompt-submit hook.
        """
        if not self.get_effective_history_for_llm():
            raise ValueError(
                "continue_from_history_stream requires a non-empty conversation; "
                "restore message history (and any compaction state) first"
            )
        turn = self._run_recorded_turn(self._continue_from_history_impl(), user_text=None)
        async with contextlib.aclosing(turn):
            async for chunk in turn:
                yield chunk

    async def _continue_from_history_impl(self) -> AsyncGenerator[Dict[str, Any], None]:
        # No memory refresh: its only model-visible effect is volatile-context
        # injection, which a continuation deliberately never performs.
        if self.session_recorder is not None:
            await asyncio.to_thread(self.session_recorder.start_continuation_turn)
            await asyncio.to_thread(
                self.session_recorder.record_system_context,
                self.system_prompt.get_text_content(),
            )
            if self.tool_collection is not None:
                await asyncio.to_thread(
                    self.session_recorder.record_tool_definitions,
                    [tool.to_public_dict() for tool in self.tool_collection.get_tool_list()],
                )
        loop_stream = self._agent_loop_stream()
        async with contextlib.aclosing(loop_stream):
            async for chunk in loop_stream:
                yield chunk

    async def _run_recorded_turn(
        self, impl: AsyncGenerator[Dict[str, Any], None], *, user_text: Optional[str]
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Drive one recorded turn: boundaries, chunk mirroring, terminal bookkeeping.

        ``aclosing`` propagates an early close of this generator into ``impl``
        synchronously — an abandoned turn must not leave the inner loop (and the
        provider stream context it holds) waiting for garbage-collection
        finalization.
        """
        turn_id = str(uuid.uuid4())
        await self._emit_turn_boundary("started", turn_id=turn_id, user_text=user_text)
        try:
            async with contextlib.aclosing(impl):
                async for chunk in impl:
                    await self._mirror_stream_chunk(chunk)
                    yield chunk
        except asyncio.CancelledError:
            await self._finish_recorded_turn("cancelled")
            await self._emit_turn_boundary("ended", turn_id=turn_id, status="cancelled")
            raise
        except Exception as exc:
            await self._finish_recorded_turn("failed", error=str(exc))
            await self._emit_turn_boundary("ended", turn_id=turn_id, status="failed")
            raise
        else:
            await self._finish_recorded_turn("completed")
            await self._emit_turn_boundary("ended", turn_id=turn_id, status="completed")

    async def _mirror_stream_chunk(self, chunk: Dict[str, Any]) -> None:
        """Re-emit one generator chunk as a stream event. Never raises."""
        kind = chunk.get("type")
        if kind not in ("response", "thinking"):
            return
        complete = bool(chunk.get("complete"))
        text = str(chunk.get("content") or "")
        if not text and not complete:
            return
        try:
            await self.emitter.assistant_delta(
                text,
                complete=complete,
                stream_uuid=str(chunk.get("uuid") or uuid.uuid4()),
                thinking=kind == "thinking",
            )
        except Exception:
            # Mirroring is observability for other frontends; it must never break
            # the turn that the local caller is already consuming.
            pass

    async def _emit_turn_boundary(
        self,
        phase: str,
        *,
        turn_id: str,
        status: Optional[str] = None,
        user_text: Optional[str] = None,
    ) -> None:
        """Emit a turn boundary, swallowing everything including cancellation.

        Called from ``except asyncio.CancelledError`` blocks, where any further
        await can itself raise, so this must not let that mask the original
        exception it is about to re-raise.
        """
        try:
            await self.emitter.turn_boundary(phase, turn_id=turn_id, status=status, user_text=user_text)
        except BaseException:
            pass

    async def _finish_recorded_turn(self, status: str, *, error: Optional[str] = None) -> None:
        recorder = self.session_recorder
        if recorder is None or recorder.current_turn_id is None:
            return
        await asyncio.to_thread(recorder.finish_turn, status, error=error)

    def refresh_memory_context(self) -> None:
        """Re-read project memory so the next turn picks up any change.

        There is no system prompt to rebuild: the memory body is injected into the conversation
        when it changes rather than rendered into the prompt, so a write costs one appended block
        instead of invalidating the whole conversation's cached prefix.
        """
        if self.memory_manager is None:
            return
        self.memory_manager.refresh()

    def _volatile_context_sections(self) -> List[VolatileSection]:
        """Snapshot the context that changes mid-session and is injected rather than rendered."""
        guidance_file, guidance = self._load_project_guidance()
        memory_body = ""
        if self.memory_manager is not None:
            try:
                memory_body = self.memory_manager.prompt_context().body
            except Exception:
                # Disabled/unavailable memory is intentionally invisible to the model.
                memory_body = ""
        sections = [
            VolatileSection("memory", memory_body),
            VolatileSection("guidance", guidance, guidance_file),
            VolatileSection("date", f"Today's date: {datetime.now().strftime('%Y-%m-%d')}"),
        ]
        for provider in self._extra_volatile_sections:
            try:
                section = provider()
            except Exception:
                # A broken host provider must not kill turns; its section is skipped.
                logger.exception("Volatile-section provider failed; skipping its section")
                continue
            if section is not None:
                sections.append(section)
        return sections

    def add_volatile_section(self, provider: Callable[[], Optional[VolatileSection]]) -> None:
        """Register a host-provided volatile-context section.

        ``provider`` is called on every turn and must return the section's current text;
        ``None`` or an empty-text :class:`VolatileSection` means "not currently present".
        Sections are injected as ``<system-reminder>`` user messages, deduplicated by
        fingerprint, and re-sent after compaction or history clears (see
        ``reset_volatile_context``). The CLI registers the plan handle and the shared task
        list this way so they survive conversation summarization.
        """
        self._extra_volatile_sections.append(provider)

    def reset_volatile_context(self) -> None:
        """Drop the volatile-context sent-state so the next turn re-sends every section.

        Used after operations that wipe or summarize history (``/clear``, compaction),
        which otherwise leave the model without context it was already shown.
        """
        self._volatile_context.forget()

    def pending_volatile_context(self) -> Optional[TextBlock]:
        """Volatile context the model has not been shown yet, or ``None`` if nothing changed."""
        return self._volatile_context.pending_block(self._volatile_context_sections())

    async def _process_message_stream_impl(
        self, message: str, attachments: Optional[List[Dict[str, Any]]] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Process a user message and yield response/thinking chunks while the agent works.

        Yields dicts of the form
        ``{"type": "response"|"thinking", "content": str, "complete": bool, "uuid": str}``.
        """
        if not self.sub_agent:
            try:
                self.refresh_memory_context()
            except Exception:
                # Keep the last valid prompt if memory or prompt refresh fails.
                pass

        unsupported_attachment_message = self._unsupported_attachment_message(attachments)
        if unsupported_attachment_message:
            await self._record_synthetic_notice(unsupported_attachment_message, "unsupported_attachment")
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
                await self._record_synthetic_notice(reason, "hook_blocked")
                yield {"type": "response", "content": reason, "complete": True, "uuid": str(uuid.uuid4())}
                return
            if submit.additional_context:
                content_blocks.append(TextBlock(text=submit.additional_context))

        if self.session_recorder is not None:
            await asyncio.to_thread(
                self.session_recorder.start_turn,
                Message(role="user", content=content_blocks),
            )
            # Journal the rendered provider-facing system context and the
            # available tool schemas; both are deduplicated by fingerprint
            # inside the recorder so only changes are recorded.
            await asyncio.to_thread(
                self.session_recorder.record_system_context,
                self.system_prompt.get_text_content(),
            )
            if self.tool_collection is not None:
                await asyncio.to_thread(
                    self.session_recorder.record_tool_definitions,
                    [tool.to_public_dict() for tool in self.tool_collection.get_tool_list()],
                )

        self.append_user_message(content_blocks)

        # Volatile context is its own user turn rather than extra blocks on the user's message: it
        # is operator context, not something the user said, and keeping it separate leaves the
        # recorded turn showing only what the user actually typed.
        #
        # It follows the user's message, and is journalled after ``start_turn``, so that the
        # in-memory history and the journal agree — a resumed session then replays the same order
        # the live session sent. Journalling it before ``start_turn`` would instead attribute it to
        # the *previous* turn, since ``record_context_message`` stamps the current turn id.
        #
        # Deliberately not gated on ``sub_agent``: sub-agent prompts no longer carry memory or
        # repository guidance either, so gating it would silently starve them of both.
        volatile_context = self.pending_volatile_context()
        if volatile_context is not None:
            if self.session_recorder is not None:
                await asyncio.to_thread(
                    self.session_recorder.record_context_message,
                    Message(role="user", content=[volatile_context]),
                )
            self.append_user_message([volatile_context])

        loop_stream = self._agent_loop_stream()
        async with contextlib.aclosing(loop_stream):
            async for chunk in loop_stream:
                yield chunk

    async def _agent_loop_stream(self) -> AsyncGenerator[Dict[str, Any], None]:
        """The shared agent loop: compaction, model streaming, tool execution.

        Message-independent by construction: user-content handling (attachment
        guards, prompt-submit hooks, user-message insertion, volatile context)
        happens before this generator is entered.
        """
        stop_reason = None
        stop_overrides = 0
        silent_turn_nudges = 0
        truncated_turn_nudges = 0
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
                # Build the request history once and reuse it for both the token
                # count and the request below — the repair+adapt+image-strip pass is
                # O(history) and ran twice per iteration before.
                self._sanitize_oversized_tool_results()
                fixed_history = await self._history_for_llm_async()
                token_count = await self.count_current_context(fixed_history)
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
                    result = await self.compress_history(trigger="auto", tokens_before=before_tokens)
                    # compress_history forgets the volatile-context sent-state on success,
                    # so the next turn re-sends memory, guidance, and any host sections
                    # (plan handle, task list) the summary may have folded away.
                    # Rebuild after compaction (history changed) and re-mark the cache
                    # checkpoint before re-counting so the reused history reflects it.
                    self.mark_cache_checkpoint()
                    self._sanitize_oversized_tool_results()
                    fixed_history = await self._history_for_llm_async()
                    token_count = await self.count_current_context(fixed_history)

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

                current_response = ""
                current_thinking = ""
                thinking_started = False
                # Use the same UUID for each segment of the response
                response_uuid = str(uuid.uuid4())
                thinking_uuid = str(uuid.uuid4())

                # Diagnostics: bracket the request so a stall is visible in the timeline
                # (start↔end, or start↔llm_error if it fails) with the actual endpoint.
                _req_start = time.monotonic()
                await self.emitter.llm_request(
                    "start",
                    provider=self.primary_model_config.provider.value,
                    model=self.primary_model_config.model,
                    endpoint=self.llm.provider.base_url,
                )

                assert self.tool_collection is not None, "tool_collection must be initialized before streaming"
                # ``LLMClient.stream`` is typed as ``AsyncContextManager | Coroutine[..., AsyncContextManager]]``
                # because providers may define ``stream`` either way; every concrete
                # provider is ``async def``, so the call is always a coroutine that we
                # await into the context manager. Cast to the coroutine form for the type
                # checker rather than awaiting the union directly.
                #
                # The origin must stay active for the whole request — entry into
                # the stream context manager, event iteration, and
                # get_final_message() — because providers may sample and emit
                # their trace record at any of those points (the native Tinker
                # provider does all of it in __aenter__), not at stream().
                with llm_call_origin(self._main_loop_origin()):
                    stream_cm = await cast(
                        Coroutine[Any, Any, AbstractAsyncContextManager[Any]],
                        self.llm.stream(
                            system=self.system_prompt,
                            max_completion_tokens=self.model_completion_tokens,
                            temperature=self.model_default_temperature,
                            messages=fixed_history,
                            model=self.primary_model_config.model,
                            tools=self.tool_collection.get_tool_list(),
                            thinking=self.primary_model_config.thinking_effort,
                            hosted_web_search=self.hosted_web_search_active,
                        ),
                    )
                    async with stream_cm as stream:
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

                            elif event.type == "hosted_tool_call" and event.tool_call_delta:
                                # A server-side web_search call completed on the
                                # provider's infrastructure — no local execution.
                                # Surface it as a normal tool_call/tool_result pair
                                # so every transcript view renders it.
                                #
                                # Close the current thinking/response segment first
                                # and rotate both uuids: transcript consumers key
                                # streamed entries by uuid, so without rotation the
                                # post-search reasoning folds into the bubble above
                                # these rows instead of opening a new one below
                                # them. Guards mirror the end-of-stream flushes: an
                                # unopened thinking segment yields nothing, while
                                # the response flush is unconditional (an empty
                                # complete response chunk is a documented no-op for
                                # every consumer) so a partially streamed response
                                # entry is never stranded incomplete by rotation.
                                if thinking_started or current_thinking:
                                    yield {
                                        "type": "thinking",
                                        "content": current_thinking,
                                        "complete": True,
                                        "uuid": thinking_uuid,
                                    }
                                    current_thinking = ""
                                    thinking_started = False
                                    thinking_uuid = str(uuid.uuid4())
                                yield {
                                    "type": "response",
                                    "content": current_response,
                                    "complete": True,
                                    "uuid": response_uuid,
                                }
                                current_response = ""
                                response_uuid = str(uuid.uuid4())
                                await self._emit_hosted_tool_call(event.tool_call_delta)

                    assistant_message = await stream.get_final_message()
                self._normalize_freeform_tool_calls(assistant_message)
                self._update_hosted_search_residual(assistant_message)
                assistant_message.usage_metadata["edit_protocol"] = self.edit_protocol.value
                response_output_tokens = get_output_tokens(
                    assistant_message.usage_metadata,
                    self.primary_model_config.provider.value,
                )
                self.total_tokens_used += response_output_tokens
                if self._accounting_reservation is not None:
                    self._accounting_reservation.report_total(self.total_tokens_used)
                stop_reason = assistant_message.stop_reason
                await self.emitter.llm_request(
                    "end",
                    provider=self.primary_model_config.provider.value,
                    model=self.primary_model_config.model,
                    elapsed_s=round(time.monotonic() - _req_start, 2),
                    stop_reason=stop_reason,
                )
                # DeepSeek reports a SERVER-side output cutoff as a clean finish;
                # the clamped wire cap should always fire first as an honest
                # "max_tokens" stop. A response landing at/near the cap without
                # one was likely truncated silently — surface it in the
                # trajectory (log_message event) since the stop reason cannot be
                # recovered.
                if (
                    self.deepseek_wire_output_cap is not None
                    and stop_reason != "max_tokens"
                    and response_output_tokens >= self.deepseek_wire_output_cap - self.DEEPSEEK_OUTPUT_CAP_SLACK
                ):
                    await self.log_warning(
                        f"DeepSeek response used {response_output_tokens} output tokens — at or near "
                        f"the {self.deepseek_wire_output_cap}-token output cap — but reported "
                        f"stop_reason={stop_reason!r}; the output was likely truncated server-side.",
                        sender=self.agent_name,
                    )

                if self.session_recorder is not None:
                    await asyncio.to_thread(
                        self.session_recorder.record_assistant,
                        assistant_message,
                        reasoning_effort=self.primary_model_config.thinking_effort,
                    )
                self.append_assistant_message(assistant_message)
                # A clean stream resets the transient-failure budget, so the cap measures
                # only consecutive failures, not lifetime failures across the turn.
                self._consecutive_llm_retries = 0
                # Likewise, a response with a tool call or visible text resets the
                # silent-turn budget: the cap measures only consecutive silence.
                if not self._is_silent_turn(assistant_message):
                    silent_turn_nudges = 0
                # And any response that was not cut off at the output limit
                # resets the truncation budget: the cap measures only
                # consecutive truncations.
                if stop_reason != "max_tokens":
                    truncated_turn_nudges = 0

                if thinking_started or current_thinking:
                    yield {"type": "thinking", "content": current_thinking, "complete": True, "uuid": thinking_uuid}

                # Send the final message to mark it complete.
                yield {"type": "response", "content": current_response, "complete": True, "uuid": response_uuid}

                if assistant_message.tool_calls:
                    await self.log_info(
                        f"Received {len(assistant_message.tool_calls)} tool call(s)", sender=self.agent_name
                    )

                    tools_succeeded = True
                    try:
                        tool_responses = await self.process_tool_calls(assistant_message.tool_calls)
                    except Exception as ex:
                        tools_succeeded = False
                        error_message = f"Error processing tool calls: {str(ex)}"
                        await self.log_error(error_message, sender=self.agent_name)

                        error_responses = [
                            ToolResult(
                                tool_use_id=tool_call.id,
                                content=f"Failed to process tool calls: {str(ex)}",
                                name=tool_call.name,
                                is_error=True,
                                execution_id=tool_call.execution_id,
                                input_kind=tool_call.input_kind,
                            )
                            for tool_call in assistant_message.tool_calls
                        ]
                        tool_responses = error_responses

                    # Persistence is deliberately outside the tool-execution handler:
                    # a journal failure is terminal and must not be mistaken for a tool
                    # failure that the model can continue past.
                    if self.session_recorder is not None:
                        tool_responses = await asyncio.to_thread(
                            self.session_recorder.record_tool_results,
                            tool_responses,
                        )
                    self.append_user_message(tool_responses)

                    if tools_succeeded and self.should_stop_after_tools():
                        break
                    if tools_succeeded and self._hook_end_turn:
                        # A blocking PostToolUse hook asked to end the turn.
                        self._hook_end_turn = False
                        break

                    # The loop will continue — deliver any user messages queued in the
                    # host UI so the next model request sees them.
                    await self._deliver_queued_user_inputs()

                # Silent-turn guard: the model ended this iteration with no tool
                # call and no visible text (reasoning-only or empty). Finalizing
                # would report nothing, so re-prompt with an escalating reminder;
                # after the cap, terminate the turn and surface the failure. Runs
                # before the Stop hook: a nudged iteration is not a turn end, and
                # clearing stop_reason makes the stop-hook block skip itself.
                if stop_reason in ["end_turn", "max_tokens", "stop_sequence"] and self._is_silent_turn(
                    assistant_message
                ):
                    from .prompts import SILENT_TURN_EXHAUSTED_NOTICE, build_silent_turn_nudge

                    if silent_turn_nudges < self.MAX_SILENT_TURN_NUDGES:
                        silent_turn_nudges += 1
                        await self.emitter.llm_status(
                            "info",
                            "Model returned no output — asking it to act "
                            f"({silent_turn_nudges}/{self.MAX_SILENT_TURN_NUDGES})…",
                        )
                        await self._record_context_user_message(build_silent_turn_nudge(silent_turn_nudges))
                        stop_reason = None
                    else:
                        await self.log_warning(
                            f"Model ended {self.MAX_SILENT_TURN_NUDGES + 1} consecutive responses with "
                            "no tool call and no visible output; terminating the turn.",
                            sender=self.agent_name,
                        )
                        await self.emitter.llm_status(
                            "warning",
                            "Model returned no output after repeated reminders — ending the turn with no result.",
                        )
                        await self._record_context_user_message(SILENT_TURN_EXHAUSTED_NOTICE)
                        silent_exhausted_text = (
                            "[no output] The model ended its turn with no tool call and no visible "
                            f"message {self.MAX_SILENT_TURN_NUDGES + 1} times in a row despite "
                            "reminders. The turn was terminated without a result."
                        )
                        await self._record_synthetic_notice(silent_exhausted_text, "silent_turn_exhausted")
                        yield {
                            "type": "response",
                            "content": silent_exhausted_text,
                            "complete": True,
                            "uuid": str(uuid.uuid4()),
                        }
                        break

                # Truncated-turn guard: an honest "max_tokens" stop with visible
                # text and no tool call means the reply was cut off mid-message
                # (the silent-turn guard above owns the no-visible-output case
                # and cleared stop_reason if it fired). Finalizing would deliver
                # a partial answer, so ask the model to continue; after the cap,
                # finalize with the partial output — unlike a silent turn there
                # IS content worth delivering.
                if stop_reason == "max_tokens" and not assistant_message.tool_calls:
                    from .prompts import build_truncated_turn_nudge

                    if truncated_turn_nudges < self.MAX_TRUNCATED_TURN_NUDGES:
                        truncated_turn_nudges += 1
                        await self.emitter.llm_status(
                            "info",
                            "Response hit the output limit — asking the model to continue "
                            f"({truncated_turn_nudges}/{self.MAX_TRUNCATED_TURN_NUDGES})…",
                        )
                        await self._record_context_user_message(build_truncated_turn_nudge(truncated_turn_nudges))
                        stop_reason = None
                    else:
                        await self.log_warning(
                            f"Model hit the output-token limit {self.MAX_TRUNCATED_TURN_NUDGES + 1} times "
                            "in a row; finalizing the turn with the partial output.",
                            sender=self.agent_name,
                        )
                        await self.emitter.llm_status(
                            "warning",
                            "Response still truncated after repeated continuation prompts — "
                            "delivering the partial output.",
                        )

                # Stop hooks (main agent only). On a natural turn end, a hook may
                # keep the agent working by blocking the stop and returning a reason.
                if stop_reason in ["end_turn", "max_tokens", "stop_sequence"] and not self.sub_agent:
                    keep_working = await self._fire_stop_hook(stop_reason)
                    if keep_working is not None and stop_overrides < self.MAX_STOP_HOOK_OVERRIDES:
                        stop_overrides += 1
                        await self._record_context_user_message(keep_working)
                        stop_reason = None

            except Exception as ex:
                if getattr(ex, "session_persistence_error", False):
                    raise
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
        try:
            if self.tool_collection is not None:
                await self.tool_collection.cleanup()
        finally:
            if self._owns_memory_manager and self.memory_manager is not None:
                self.memory_manager.close()
                self.memory_manager = None

        # Log cleanup
        await self.log_info("Agent cleanup completed", sender=self.agent_name)
