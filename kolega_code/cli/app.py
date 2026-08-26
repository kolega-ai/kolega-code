"""Textual application for Kolega Code."""

from __future__ import annotations

import asyncio
import os
import platform
import subprocess
import sys
import threading
import time
import traceback
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, TypeVar

from rich.console import Group
from rich.text import Text
from textual import events
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.filter import LineFilter
from textual.screen import Screen
from textual.timer import Timer
from textual.worker import WorkerState
from textual.widgets import (
    Button,
    Footer,
    OptionList,
    Static,
    TabbedContent,
    TabPane,
    TextArea,
)
from textual.widgets.option_list import Option

from kolega_code.agent import AgentConfig, AgentEvent, KnownEventType
from kolega_code.extensions import ExtensionSelection, KolegaExtensionLoadError
from kolega_code.session.inbox import (
    SHARED_INBOX_REGISTRY,
    MAX_QUEUED_PEER_MESSAGES,
    InboxRegistration,
    InboxRegistry,
    PeerMessage,
    PeerRepeatGuard,
    PeerSocketServer,
    bind_session_socket,
    deliver_inbound,
    InboundNotice,
    InboundNoticeKind,
    peer_model_text,
    peer_origin,
)
from kolega_code.session.recording import RecordingConnectionManager
from kolega_code.session.runtime import SessionRuntime, control_channel_for
from kolega_code.agent.prompt_dump import list_prompt_overrides
from kolega_code.agent.prompt_provider import AgentMode
from kolega_code.agent.prompts import (
    build_implement_plan_prompt,
    build_mode_switch_notice,
)
from kolega_code.hooks import HookDispatcher, HookEvent
from kolega_code.llm.models import Message, MessageHistory, TextBlock
from kolega_code.memory import ProjectMemoryManager
from kolega_code.mcp.config import load_mcp_config, mcp_secret_values
from kolega_code.mcp.state import MCPOAuthTokenStore
from kolega_code.permissions import (
    PermissionMode,
    normalize_permission_mode,
)

from . import messages, theme
from .config import CliConfigOverrides, active_model_override_message, key_status
from .connection import CliConnectionManager
from .session_event_store import FileArtifactStore, FileSessionEventStore
from kolega_code.llm.ledger import UsageLedger

from .goal import GoalState
from .session_usage import SessionUsageSink
from .loop import LOOP_TICK_SECONDS, LoopState
from .diagnostics import DiagnosticsLog, ResponsivenessWatchdog
from .file_index import WorkspaceFileIndex
from .mentions import build_file_attachments
from .prompt_history import load_prompt_history, save_prompt_history
from .provider_registry import default_ui_thinking_effort
from .session_journal import RewindOutcome, TurnSummary
from .session_store import SessionRecord, SessionStore, resolve_active_project
from .settings import CliSettings, SettingsStore
from kolega_code.agent.custom_agents import CustomAgentCatalog, discover_custom_agents
from kolega_code.services.terminal import LocalTerminalManager
from kolega_code.tools import ToolError
from kolega_code.worktrees import WorktreeError, resolve_worktree
from .skills import (
    SkillCatalog,
    discover_skills,
)
from .slash_commands import (
    THREAD_RESET_COMMANDS,
    SlashCommandEntry,
    search_commands,
)
from .theme import Color, Glyph
from .updater import check_for_update, current_version, update_status_message
from .tui import constants as tui_constants
from .tui import agent_runtime as tui_agent_runtime
from .tui import changes_screen as tui_changes
from .tui import command_handlers as tui_command_handlers
from .tui import loop_runtime as tui_loop_runtime
from .tui import memory_screen as tui_memory
from .tui import prompt_flows as tui_prompt_flows
from .tui import onboarding_screen as tui_onboarding
from .tui import settings_panel as tui_settings_panel
from .tui import settings_screen as tui_settings_screen
from .tui import pacing as tui_pacing
from .tui import session_diff as tui_session_diff
from .tui import status_dashboard as tui_status_dashboard
from .tui import state as tui_state
from .tui import sub_agent_screen as tui_sub_agents
from .tui import terminal_display as tui_terminal_display
from .tui import transcript as tui_transcript
from .tui import widgets as tui_widgets

CLI_AGENT_MODE = AgentMode.CLI.value
LOG_MAX_LINES = 2_000
TERMINAL_MAX_LINES = 2_000
ScreenResultT = TypeVar("ScreenResultT")
SESSION_DIFF_REFRESH_INTERVAL = 1.0
# Immediate-flush triggers are memory bounds, not cadence: they intentionally
# bypass flush pacing so buffered output can never grow without limit.
TERMINAL_IMMEDIATE_FLUSH_CHARS = 64 * 1024
LOG_IMMEDIATE_FLUSH_ITEMS = 100


class KolegaCodeApp(
    tui_settings_panel.SettingsPanelMixin,
    tui_command_handlers.CommandHandlersMixin,
    tui_agent_runtime.AgentRuntimeMixin,
    tui_loop_runtime.LoopRuntimeMixin,
    tui_status_dashboard.StatusDashboardMixin,
    tui_prompt_flows.PromptFlowMixin,
    tui_transcript.TranscriptRenderingMixin,
    App,
):
    """Interactive terminal UI for Kolega Code."""

    CSS_PATH = "tui/styles.tcss"
    # kolega-code uses its own / slash-command system, so disable Textual's
    # command palette. Its default binding is ctrl+p, which collides with the
    # toggle_permission_mode binding below and made "Ctrl+P Permissions" render
    # twice in the footer.
    ENABLE_COMMAND_PALETTE = False

    BINDINGS = [
        Binding(
            "shift+tab", "toggle_interaction_mode", "Plan/Build", show=True, key_display="Shift+Tab", priority=True
        ),
        Binding("ctrl+p", "toggle_permission_mode", "Permissions", show=True, key_display="Ctrl+P", priority=True),
        Binding("ctrl+o", "toggle_sidebar", "Sidebar", show=True, key_display="Ctrl+O", priority=True),
        Binding("ctrl+g", "open_sub_agent", "Agents", show=True, key_display="Ctrl+G", priority=True),
        Binding("ctrl+r", "open_changes", "Changes", show=True, key_display="Ctrl+R", priority=True),
        Binding("ctrl+c", "cancel_generation", "Cancel", show=True, key_display="Ctrl+C"),
        Binding("escape", "cancel_generation", "Cancel", show=False),
        Binding("ctrl+q", "quit", "Quit", show=True, key_display="Ctrl+Q"),
    ]

    def __init__(
        self,
        project_path: Path,
        mode: str,
        store: SessionStore,
        session: SessionRecord,
        config: Optional[AgentConfig] = None,
        settings_store: Optional[SettingsStore] = None,
        overrides: Optional[CliConfigOverrides] = None,
        permission_mode: Optional[str] = None,
        browser_visible: bool = False,
        check_for_updates: bool = False,
        show_logs: bool = False,
        startup_config_error: Optional[str] = None,
        extension_selection: Optional[ExtensionSelection] = None,
        resuming_session: bool = False,
        inbox_registry: Optional[InboxRegistry] = None,
    ) -> None:
        super().__init__()
        self._terminal_control_filter = tui_terminal_display.TerminalControlFilter()
        self.project_path = project_path.resolve()
        self.config = config
        self.mode = CLI_AGENT_MODE
        self.store = store
        self._session_recorder = store.recorder(session.session_id)
        self.session = store.load(session.session_id)
        self.session.mode = CLI_AGENT_MODE
        self.active_project_path, workspace_warning = resolve_active_project(
            self.session, self.store, self.project_path
        )
        self._startup_workspace_warning = workspace_warning or ""
        self.interaction_mode = self._validated_interaction_mode(self.session.interaction_mode)
        self.session.interaction_mode = self.interaction_mode
        self.permission_mode = normalize_permission_mode(
            permission_mode or self.session.permission_mode,
            default=PermissionMode.ASK,
        )
        self.session.permission_mode = self.permission_mode.value
        self.settings_store = settings_store or SettingsStore(store.root)
        self.memory_manager = ProjectMemoryManager(
            self.project_path,
            state_root=self.settings_store.root,
        )
        self.overrides = overrides or CliConfigOverrides()
        self.settings: CliSettings = CliSettings()
        self.skills_enabled = config.skills_enabled if config is not None else True
        self.skill_catalog: SkillCatalog = (
            discover_skills(self.active_project_path) if self.skills_enabled else SkillCatalog()
        )
        self.custom_agent_catalog: CustomAgentCatalog = discover_custom_agents(
            self.active_project_path,
            self.settings_store.root,
        )
        self.file_index = WorkspaceFileIndex(self.active_project_path)
        self._file_index_refreshing = False
        # Local diagnostics (constructed on mount, once settings/secrets are loaded).
        self._diag: Optional[DiagnosticsLog] = None
        self._watchdog: Optional[ResponsivenessWatchdog] = None
        self.browser_visible = browser_visible
        self.sidebar_visible = True
        self.check_for_updates = check_for_updates
        self.show_logs = show_logs
        self.startup_config_error = startup_config_error
        self._resuming_session = resuming_session
        # Two managers, one stream. The TUI reads live events off the CLI queue;
        # the agent broadcasts into the recording wrapper, which assigns each
        # event a sequence number and persists it before forwarding. That is what
        # makes a session replayable and followable rather than only visible now.
        self.connection_manager = CliConnectionManager()
        _journal = store.journal(session.session_id)
        self.recording_connection_manager = RecordingConnectionManager(
            self.connection_manager,
            FileSessionEventStore(_journal),
            session_id=session.session_id,
            artifact_store=FileArtifactStore(_journal),
        )
        self._recording_primed = False
        # Started on demand by /share; nothing listens until the user asks.
        self._share_server = None
        # Interactions that need a human answer go through the control channel, so
        # this client answers prompts exactly the way a browser or a remote client
        # would. The channel announces on the recording transport, which is what
        # puts decision points into the session's replayable history.
        self.control_channel = control_channel_for(
            session.session_id,
            self.recording_connection_manager.broadcast_event,
            workspace_id=session.workspace_id,
            thread_id=session.thread_id,
        )
        self.session_runtime = SessionRuntime(
            session_id=session.session_id,
            # Saved permission rules are launch-scoped like trust: they are
            # matched and persisted in the session's immutable launch checkout,
            # not the active worktree, so they survive worktree removal and
            # apply across workspace switches.
            project_path=self.project_path,
            control=self.control_channel,
            permission_mode=self.permission_mode,
            on_notice=lambda text: self._notify_user(text, severity="warning"),
        )
        # A single client holds control; viewers of a shared session may watch a
        # prompt appear but never answer it.
        self.control_channel.acquire(tui_constants.TUI_CLIENT_ID)
        # Cross-session messaging (phase 1): this session's entry in the
        # process-wide peer inbox. Hosts running isolated fleets inject their
        # own registry; the shared default serves the ordinary CLI.
        self.inbox_registry = inbox_registry if inbox_registry is not None else SHARED_INBOX_REGISTRY
        # Phase 2: this session's cross-process socket. Bound on mount when the
        # feature is enabled; None means in-process-only.
        self._inbox_socket: Optional[PeerSocketServer] = None
        self.messaging_socket_path: Optional[Path] = None
        # Loop protection: identical repeats from one sender inside the window
        # are dropped before they can bounce two agents into a forever loop.
        self._peer_repeat_guard = PeerRepeatGuard()
        self._hook_dispatcher: Optional[HookDispatcher] = None
        self._session_started = False
        #: Set when action_quit saves the session cleanly; main.py prints the
        #: resume-with-session-ID hint only when this is True.
        self._quit_cleanly = False
        self.extension_selection = extension_selection
        self._extension_bundle = None
        self.agent = None
        self.agent_worker = None
        self.conversation_entries: list[tui_state.ConversationEntry] = []
        self._stream_entries: dict[str, tui_state.ConversationEntry] = {}
        self._tool_entries: dict[str, tui_state.ConversationEntry] = {}
        self._tool_stream_buffers: dict[str, str] = {}
        self._sub_agent_activities: dict[str, tui_state.SubAgentActivity] = {}
        self._sub_agent_by_tool_call: dict[str, str] = {}
        self._sub_agent_seq = 0
        self._session_file_changes: list[tui_state.SessionFileChange] = []
        self._pending_workspace_switch: Optional[tui_state.PendingWorkspaceSwitch] = None
        self._session_diff_tracker: Optional[tui_session_diff.SessionDiffTrackerBase] = None
        self._session_diff_generation = 0
        self._session_diff_files: list[tui_session_diff.SessionDiffFile] = []
        self._session_diff_scope: Optional[tui_session_diff.DiffScope] = None
        # Trackers are not safe for concurrent use; this serializes the refresh
        # worker against the one-shot scope probe.
        self._session_diff_lock = threading.Lock()
        self._session_diff_dirty = False
        self._session_diff_refresh_running = False
        self._session_diff_timer: Optional[Timer] = None
        self._session_diff_baseline_id: Optional[int] = None
        self._rewind_running = False
        self._workflow_activities: dict[str, tui_state.WorkflowActivity] = {}
        self._render_pending = False
        self._conversation_anchor_pending = False
        self._transcript_window: Optional[tui_widgets.ScrollbackWindow] = None
        self._transcript_sync_pending = False
        self._rendered_entry_count = 0
        self._dirty_entry_ids: set[str] = set()
        self._active_progress_entry: Optional[tui_state.ConversationEntry] = None
        self._turn_active = False
        self._latest_plan: Optional[str] = self.session.latest_plan_markdown or None
        self._plan_pending: bool = bool(self._latest_plan and self.session.plan_pending)
        self._plan_reofferable: bool = bool(self._latest_plan and (self.session.plan_reofferable or self._plan_pending))
        self._plan_decision_active = False
        self._gigacode_enabled = bool(self.session.gigacode_enabled)
        # Session web tool mode override (None -> config-resolved mode applies).
        self._web_search_mode: Optional[str] = self.session.web_search_mode or None
        # Process-wide LLM usage accounting; every agent build (including
        # rebuilds on model/settings changes) shares this one ledger. The
        # journal sink is attached in on_mount, once the UI loop exists.
        self._usage_ledger = UsageLedger()
        self._usage_sink: Optional[SessionUsageSink] = None
        self._goal: Optional[GoalState] = GoalState.from_dict(self.session.goal) if self.session.goal else None
        # Checkpoint for goal token accounting: drains advance it, so a resumed
        # goal continues from the persisted tokens_spent without recounting.
        self._goal_usage_mark = self._usage_ledger.snapshot()
        self._scheduled_loop: Optional[LoopState] = (
            LoopState.from_dict(self.session.loop) if self.session.loop else None
        )
        self._loop_iteration_active = False
        self._loop_scheduler_timer: Optional[Timer] = None
        self._pending_question: Optional[tui_state.PendingQuestion] = None
        self._pending_approval: Optional[tui_state.PendingApproval] = None
        self._pending_image_attachments: list[dict] = []
        self._queued_messages: list[tui_state.QueuedMessage] = []
        self._queued_message_seq = 0
        # Dedup flag: one vision-mismatch system message per non-vision model
        # session. Reset in _switch_model so a new model gets a fresh warning.
        self._vision_warning_shown = False
        self._settings_screen: Optional[tui_settings_screen.SettingsScreen] = None
        self._memory_screen: Optional[tui_memory.MemoryScreen] = None
        self._onboarding_screen: Optional[tui_onboarding.OnboardingScreen] = None
        self._onboarding_skipped = False
        self._permission_lock = asyncio.Lock()
        self._persistence_lock = asyncio.Lock()
        self._pending_model_selection: Optional[tui_state.PendingModelSelection] = None
        self._pending_effort_selection: Optional[tui_state.PendingEffortSelection] = None
        self._pending_theme_selection: Optional[tui_state.PendingThemeSelection] = None
        # Saved per-agent model/effort awaiting the provider->model cascade that
        # restores them (keyed by the row's model/effort select id). See
        # _populate_agent_model_rows for why the cascade, not direct assignment, applies them.
        self._pending_agent_models: dict[str, str] = {}
        self._pending_agent_efforts: dict[str, str] = {}
        provider, model = self._startup_model()
        self._status_state = tui_state.StatusDashboardState(
            provider=provider,
            model=model,
            mode=self.interaction_mode,
            permission_mode=self.permission_mode.value,
            gigacode_enabled=self._gigacode_enabled,
        )
        self._turn_started_at: Optional[float] = None
        self._turn_finished_duration: Optional[float] = None
        self._turn_timer: Optional[Timer] = None
        self._turn_status_text = ""
        self._turn_final_text = ""
        self._turn_final_state = tui_state.TurnState.IDLE
        self._last_turn_status_content: Optional[str] = None
        self._spinner_frame = 0
        self._last_sub_agent_tick = 0.0
        self._sub_agent_inspector: Optional[tui_sub_agents.SubAgentInspectorScreen] = None
        self._changes_inspector: Optional[tui_changes.ChangesInspectorScreen] = None
        self._terminal_has_content = False
        self._terminal_output_buffer: list[str] = []
        self._terminal_output_buffer_chars = 0
        self._terminal_flush_timer: Optional[Timer] = None
        self._terminal_display_normalizer = tui_terminal_display.TerminalDisplayNormalizer()
        self._log_output_buffer: list[Any] = []
        self._log_flush_timer: Optional[Timer] = None
        self._flush_pacer = tui_pacing.FlushPacer()

    def get_line_filters(self) -> Sequence[LineFilter]:
        """Apply the terminal-control boundary after Textual's style filters."""
        return (*super().get_line_filters(), self._terminal_control_filter)

    def compose(self) -> ComposeResult:
        with Horizontal(id="body"):
            with Vertical(id="conversation_panel"):
                yield Static(
                    self._meta_content(),
                    classes="meta",
                    id="session_meta",
                )
                yield tui_widgets.ConversationView(id="conversation")
                yield tui_widgets.JumpToBottomBar(
                    f"{theme.g(Glyph.DOWN)} More output below — click to jump to the latest",
                    id="jump_to_bottom",
                )
                yield tui_widgets.ActionList(id="plan_actions")
                yield tui_widgets.PromptPanel(
                    id="question_prompt",
                    actions_id="question_actions",
                    title=f"{theme.g(Glyph.QUESTION)} Question",
                )
                yield tui_widgets.PromptPanel(
                    id="approval_prompt",
                    actions_id="approval_actions",
                    title=f"{theme.g(Glyph.QUESTION)} Permission",
                )
                yield tui_widgets.ActionList(id="model_actions")
                yield tui_widgets.ActionList(id="effort_actions")
                yield tui_widgets.ActionList(id="theme_actions")
                yield Static("", id="turn_status", markup=True)
                yield Static("", id="queued_messages", markup=False)
                with Horizontal(id="composer_hint_row"):
                    yield Static("", id="composer_hint", markup=False)
                    yield Button(theme.g(Glyph.CROSS), id="detach_btn", classes="hint-detach")
                yield tui_widgets.CompletionDropdown(id="completion_dropdown")
                yield tui_widgets.ChatComposer(placeholder=messages.COMPOSER_PLACEHOLDER, id="composer")
            with Vertical(id="side_panel"):
                with TabbedContent(id="events"):
                    with TabPane("Status", id="status_pane"):
                        with VerticalScroll(id="status_form"):
                            with Vertical(classes="status-section", id="status_summary_section") as status_section:
                                status_section.border_title = "Status"
                                yield Static("", id="status_dashboard", markup=True)
                            with Vertical(classes="status-section", id="status_usage_section") as usage_section:
                                usage_section.border_title = "Usage"
                                yield Static("", id="status_usage", markup=True)
                            with Vertical(classes="status-section", id="status_task_list_section") as task_section:
                                task_section.border_title = "Task List"
                                yield tui_widgets.PlanningMarkdown(
                                    messages.TASK_LIST_EMPTY_MESSAGE,
                                    id="status_task_list_markdown",
                                    empty_source=messages.TASK_LIST_EMPTY_MESSAGE,
                                )
                    if self.show_logs:
                        with TabPane("Logs", id="logs_pane"):
                            yield tui_widgets.LogOutputLog(
                                id="logs",
                                wrap=True,
                                markup=True,
                                max_lines=LOG_MAX_LINES,
                            )
                    with TabPane("Terminal", id="terminal_pane"):
                        # Sidebar-rendered history is bounded for UI performance;
                        # command output returned to the agent is unaffected.
                        yield tui_widgets.TerminalOutputLog(
                            id="terminal",
                            wrap=True,
                            markup=False,
                            max_lines=TERMINAL_MAX_LINES,
                        )
                    with TabPane("Plan", id="planning_pane"):
                        with VerticalScroll(id="planning_form"):
                            with Vertical(classes="planning-section", id="planning_plan") as plan_section:
                                plan_section.border_title = "Plan"
                                yield tui_widgets.PlanningMarkdown(
                                    messages.PLAN_EMPTY_MESSAGE,
                                    id="planning_plan_markdown",
                                    empty_source=messages.PLAN_EMPTY_MESSAGE,
                                )
                    with TabPane("Settings", id="settings_pane"):
                        with Vertical(id="settings_summary_panel"):
                            with Vertical(classes="settings-section", id="settings_summary_section") as summary_section:
                                summary_section.border_title = "Settings"
                                yield Static("", id="settings_summary")
                                yield Button(
                                    "Open Settings →",
                                    id="open_settings",
                                    classes="quiet",
                                )
                                yield Static("", id="settings_summary_status")
        yield Footer()

    def _diagnostics_header(self) -> dict:
        """One-shot environment/config snapshot for the diagnostics timeline."""
        header: dict = {
            "kolega_version": current_version(),
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "term": os.environ.get("TERM", ""),
            "term_program": os.environ.get("TERM_PROGRAM", ""),  # captures ghostty/iTerm/etc.
            "interaction_mode": self.interaction_mode,
            "permission_mode": self.permission_mode.value,
            "gigacode_enabled": self._gigacode_enabled,
        }
        try:
            if self.config is not None:
                lc = self.config.long_context_config
                header["provider"] = getattr(lc.provider, "value", str(lc.provider))
                header["model"] = lc.model
                header["thinking_effort"] = getattr(lc, "thinking_effort", None)
        except Exception:
            pass
        try:
            header["providers_with_keys"] = sorted(k for k, v in self.settings.api_keys.items() if v)
        except Exception:
            pass
        return header

    # ------------------------------------------------------------------
    # Cross-session messaging (phase 1): the in-process peer inbox.
    # ------------------------------------------------------------------

    def _peer_status(self) -> str:
        return "busy" if (self._turn_active or self.agent_worker is not None) else "idle"

    def _register_with_inbox(self) -> None:
        """Publish this session's live entry so sibling sessions can see and
        message it. Callables, not snapshots: every query reads current state."""
        self.inbox_registry.register(
            InboxRegistration(
                session_id=self.session.session_id,
                describe_title=lambda: self.session.title,
                describe_project_path=lambda: str(self.active_project_path),
                describe_status=self._peer_status,
                deliver_message=self._deliver_peer_message,
            )
        )

    def _unregister_from_inbox(self) -> None:
        self.inbox_registry.unregister(self.session.session_id)

    async def _start_inbox_socket(self) -> None:
        """Bind this session's cross-process inbox socket.

        Never breaks startup: any bind failure (unwritable state dir,
        path-length limits) degrades to in-process-only messaging with a
        notice, not a failed launch.
        """
        server = await bind_session_socket(
            store_root=self.store.root,
            session_id=self.session.session_id,
            describe_status=self._peer_status,
            deliver_message=self._deliver_peer_message,
            on_unavailable=lambda reason: self._log_status(
                messages.MESSAGING_SOCKET_UNAVAILABLE.format(reason=reason), "info"
            ),
        )
        if server is not None:
            self._inbox_socket = server
            self.messaging_socket_path = server.path

    async def _stop_inbox_socket(self) -> None:
        server, self._inbox_socket = self._inbox_socket, None
        self.messaging_socket_path = None
        if server is not None:
            try:
                await server.stop()
            except Exception:  # noqa: BLE001 — shutdown must always complete
                pass

    def _enqueue_peer_message(self, message: PeerMessage) -> None:
        """Queue a peer message on the recipient's own rhythm.

        The transcript keeps the raw text; the model receives the provenance
        preamble followed by the text. The existing queue machinery provides
        the delivery guarantees — between tool calls during a turn, a fresh
        turn when idle, never interrupting a running tool.
        """
        self._queue_user_message(message.text, origin=peer_origin(message), model_text=peer_model_text(message))
        self._schedule_maybe_start_queued_message()

    async def _emit_peer_event(self, event_type: str, message: PeerMessage) -> None:
        """Record a peer-message lifecycle event on this session's stream.

        Rides the same recording transport as everything else, so replays show
        arrival and acceptance. Observability only: a transport failure must
        never break delivery itself.
        """
        event = AgentEvent(
            sender="peer-inbox",
            event_type=event_type,
            session_id=self.session.session_id,
            content=message.event_content(),
        )
        try:
            await self.recording_connection_manager.broadcast_event(
                event,
                self.session.workspace_id,
                self.session.thread_id,
            )
        except Exception:
            pass

    def _render_peer_notice(self, notice: InboundNotice) -> None:
        sender = notice.message.sender_title
        if notice.kind is InboundNoticeKind.REPEAT_DROPPED:
            self._log_status(messages.PEER_REPEAT_DROPPED.format(sender=sender), "info")
        elif notice.kind is InboundNoticeKind.POLICY_DROPPED:
            # Silent toward the sender (it still reports success); the user
            # sees why nothing happened.
            self._log_status(messages.PEER_MESSAGE_REFUSED.format(sender=sender), "info")
        else:
            self._log_status(messages.PEER_QUEUE_FULL.format(limit=MAX_QUEUED_PEER_MESSAGES), "info")

    async def _deliver_peer_message(self, message: PeerMessage) -> str:
        """Inbound hook registered with the inbox. The recipient decides.

        All sequencing lives in :func:`deliver_inbound`; this host only says
        where events and queued messages go, how notices render, and that a
        held message can be asked about interactively.
        """
        return await deliver_inbound(
            message,
            policy=self.settings.get_cross_session_inbound(),
            recipient_mode=self.permission_mode.value,
            repeat_guard=self._peer_repeat_guard,
            queued_count=lambda: sum(1 for item in self._queued_messages if item.is_peer),
            record_event=lambda event_type: self._emit_peer_event(event_type, message),
            enqueue=self._enqueue_peer_message,
            notify=self._render_peer_notice,
            hold_for_review=self._schedule_hold_approval,
        )

    def _schedule_hold_approval(self, message: PeerMessage) -> None:
        """Park a held message behind an Accept/Drop prompt that expires.

        Rides the control channel's question flow, so expiry resolves to the
        default (Drop) via the channel's timeout machinery — a held message can
        never park silently forever.
        """
        expiry_seconds = self.settings.get_dialog_expiry()

        async def ask() -> None:
            response = await self.control_channel.request(
                "question",
                {
                    "question": messages.PEER_HOLD_QUESTION.format(sender=message.sender_title),
                    "options": ["Accept", "Drop"],
                    "descriptions": [messages.PEER_HOLD_ACCEPT_DESC, messages.PEER_HOLD_DROP_DESC],
                },
                default={"answer": "Drop"},
                timeout=expiry_seconds,
            )
            answered = str((response or {}).get("answer") or "").strip().lower()
            if answered == "accept":
                self._log_status(messages.PEER_MESSAGE_ACCEPTED.format(sender=message.sender_title), "info")
                self._enqueue_peer_message(message)
                await self._emit_peer_event(KnownEventType.PEER_MESSAGE_DELIVERED, message)
            else:
                self._log_status(messages.PEER_MESSAGE_DROPPED.format(sender=message.sender_title), "info")

        self.run_worker(ask(), name="peer-hold-approval", group="peer-messaging")

    async def on_mount(self) -> None:
        self.settings = self.settings_store.load()
        self._register_with_inbox()
        await self._start_inbox_socket()
        self._restore_prompt_history()
        # Attach usage persistence before any turn can run: the sink journals
        # every ledger-settled non-history response, and its start marker must
        # precede the first covered turn so coverage boundaries are exact.
        usage_sink = SessionUsageSink(
            self.store.journal(self.session.session_id),
            self._session_recorder,
            self._usage_ledger,
            mode="tui",
        )
        self._usage_sink = usage_sink
        self._usage_ledger.observer = usage_sink
        await usage_sink.start()
        # Local diagnostics: a per-turn timeline + a responsiveness watchdog that dumps the
        # blocking stack if the UI goes unresponsive. Local-only (shared only via /bug);
        # never let diagnostics setup break mount.
        try:
            secret_values = [v for v in getattr(self.settings, "api_keys", {}).values() if v]
            mcp_config = getattr(self.config, "mcp_config", None)
            if mcp_config is None:
                mcp_config = load_mcp_config(
                    self.project_path,
                    self.settings_store.root,
                    project_trusted=self.settings.is_mcp_project_trusted(self.project_path),
                )
            secret_values.extend(mcp_secret_values(mcp_config))
            secret_values.extend(MCPOAuthTokenStore(self.settings_store.root).secret_values())
            self._diag = DiagnosticsLog(self.store.root, self.session.session_id, secret_values=secret_values)
            self._diag.record("session_start", **self._diagnostics_header())
            self._watchdog = ResponsivenessWatchdog(self._diag)
            self._watchdog.start()
            # The beat doubles as the loop-latency probe, so it ticks several times a
            # second: chop below the stack-capture threshold is only visible if a beat
            # lands inside the stalled window. The callback is O(1).
            self.set_interval(self._watchdog.beat_interval, self._watchdog.beat)
            self._flush_pacer.attach(self._watchdog.recent_excess)
        except Exception:
            self._diag, self._watchdog = None, None
        # Register all themes and apply the persisted one before the first paint,
        # so the splash and settings controls render already themed. In non-truecolor
        # terminals (e.g. macOS Terminal.app) the chrome is neutralized to gray so it
        # doesn't quantize to a saturated cube color.
        truecolor = theme.supports_truecolor(self.console)
        for textual_theme in theme.build_textual_themes(truecolor=truecolor):
            self.register_theme(textual_theme)
        theme.apply_theme(self.settings.active_theme)
        try:
            self.theme = theme.textual_theme_name(self.settings.active_theme)
        except Exception:
            pass
        self._update_settings_status()
        if self._startup_workspace_warning:
            self._add_conversation_entry(
                tui_state.ConversationEntry(kind="system", content=self._startup_workspace_warning)
            )
            self._notify_user(self._startup_workspace_warning, severity="warning")
        self._initialize_session_diff_tracker()
        self._refresh_status_dashboard()
        self._restore_plan_action_visibility()
        self._set_question_actions_visible(False)
        self._set_approval_actions_visible(False)
        self._set_model_actions_visible(False)
        self._set_effort_actions_visible(False)
        self._refresh_planning_sidebar()
        self._ensure_startup_entry()
        self._update_detach_button()
        if self.check_for_updates:
            self.run_worker(self._check_for_update_on_startup(), name="kolega-update-check", group="updates")
        # Warm the @-mention file index off the event loop so the first mention has
        # results without blocking the UI on os.walk (slow on iCloud/large trees).
        self._maybe_refresh_file_index()
        self._schedule_conversation_bottom_anchor()
        self.run_worker(self._consume_events(), name="kolega-events", group="events")
        # Always-on scheduler tick. It is a cheap no-op with no loop armed, and
        # only re-renders the dashboard when the coarse countdown label changes.
        self._loop_scheduler_timer = self.set_interval(LOOP_TICK_SECONDS, self._loop_tick, name="loop-scheduler")
        try:
            if self.config is not None:
                await self._build_agent(self.config)
                self._set_chat_enabled(True)
                self._schedule_primary_focus_restore()
            else:
                await self._ensure_agent_from_settings()
        except KolegaExtensionLoadError as exc:
            # An unloadable --extension refuses to start rather than running
            # without the requested extension.
            self.exit(return_code=1, message=str(exc))
            return
        await self._restore_loop_on_startup()
        if self.config is None and not self._onboarding_skipped:
            await self.action_open_onboarding()

    @property
    def _conversation(self) -> tui_widgets.ConversationView:
        return self.query_one("#conversation", tui_widgets.ConversationView)

    @property
    def _logs(self) -> tui_widgets.LogOutputLog:
        return self.query_one("#logs", tui_widgets.LogOutputLog)

    @property
    def _terminal(self) -> tui_widgets.TerminalOutputLog:
        return self.query_one("#terminal", tui_widgets.TerminalOutputLog)

    def _format_terminal_command(self, command: str) -> Text:
        """Accent prompt glyph plus the command in bold."""
        return Text.assemble(
            (theme.g(Glyph.USER) + " ", Color.ACCENT),
            (command, "bold"),
        )

    def _queue_terminal_output(self, output: str) -> None:
        """Buffer display-safe terminal chunks so high-volume output renders in batches."""
        output = self._terminal_display_normalizer.feed(output)
        if not output:
            return

        buffer_was_empty = not self._terminal_output_buffer
        self._terminal_output_buffer.append(output)
        self._terminal_output_buffer_chars += len(output)
        self._terminal_has_content = True

        if buffer_was_empty:
            self._mark_tab_activity("terminal_pane")

        if self._terminal_output_buffer_chars >= TERMINAL_IMMEDIATE_FLUSH_CHARS:
            self._flush_terminal_output()
            return

        if self._terminal_flush_timer is None:
            self._terminal_flush_timer = self.set_timer(
                self._flush_pacer.interval(),
                self._flush_terminal_output,
                name="terminal-output-flush",
            )

    def _flush_terminal_output(self) -> None:
        """Write any buffered terminal output as a single sticky-follow append."""
        if self._terminal_flush_timer is not None:
            self._terminal_flush_timer.stop()
            self._terminal_flush_timer = None

        if not self._terminal_output_buffer:
            return

        output = "".join(self._terminal_output_buffer)
        self._terminal_output_buffer.clear()
        self._terminal_output_buffer_chars = 0

        try:
            terminal = self._terminal
        except Exception:
            return
        terminal.write_terminal(output)

    def _write_terminal_command(self, command: str) -> None:
        self._terminal_display_normalizer.reset()
        self._flush_terminal_output()
        try:
            terminal = self._terminal
        except Exception:
            return
        if self._terminal_has_content:
            terminal.write_terminal("")
        terminal.write_terminal(self._format_terminal_command(command))
        self._terminal_has_content = True
        self._mark_tab_activity("terminal_pane")

    def _clear_runtime_output(self) -> None:
        """Clear runtime-only terminal/log sidebar output for thread resets."""
        if self._terminal_flush_timer is not None:
            self._terminal_flush_timer.stop()
            self._terminal_flush_timer = None
        if self._log_flush_timer is not None:
            self._log_flush_timer.stop()
            self._log_flush_timer = None

        self._terminal_display_normalizer.reset()
        self._terminal_output_buffer.clear()
        self._terminal_output_buffer_chars = 0
        self._terminal_has_content = False
        self._log_output_buffer.clear()

        try:
            self._terminal.clear_output()
        except Exception:
            pass

        if self.show_logs:
            try:
                self._logs.clear_output()
            except Exception:
                pass

        self._clear_tab_activity("terminal_pane")
        self._clear_tab_activity("logs_pane")

    def _format_log_line(self, text: str, level: str = "info") -> Text:
        """One log line: muted HH:MM:SS, a level-colored glyph, then the text."""
        body_style = Color.MUTED if level == "debug" else ""
        return Text.assemble(
            (time.strftime("%H:%M:%S") + " ", Color.MUTED),
            (theme.g(Glyph.STATUS) + " ", theme.log_level_color(level)),
            (text, body_style),
        )

    def _write_log(self, text: str, level: str = "info") -> None:
        """Single write path into the optional Logs tab."""
        if not self.show_logs:
            return
        self._queue_log_output(self._format_log_line(text, level))

    def _queue_log_output(self, renderable: object) -> None:
        buffer_was_empty = not self._log_output_buffer
        self._log_output_buffer.append(renderable)

        if buffer_was_empty:
            self._mark_tab_activity("logs_pane")

        if len(self._log_output_buffer) >= LOG_IMMEDIATE_FLUSH_ITEMS:
            self._flush_log_output()
            return

        if self._log_flush_timer is None:
            self._log_flush_timer = self.set_timer(
                self._flush_pacer.interval(),
                self._flush_log_output,
                name="log-output-flush",
            )

    def _flush_log_output(self) -> None:
        """Write any buffered log lines as one RichLog append."""
        if self._log_flush_timer is not None:
            self._log_flush_timer.stop()
            self._log_flush_timer = None

        if not self._log_output_buffer:
            return

        entries = list(self._log_output_buffer)
        self._log_output_buffer.clear()
        try:
            logs = self._logs
        except Exception:
            return
        logs.write_log(entries[0] if len(entries) == 1 else Group(*entries))

    def _mark_tab_activity(self, pane_id: str) -> None:
        """Add an activity dot to a background tab's label."""
        base = tui_constants.TAB_BASE_LABELS.get(pane_id)
        if base is None:
            return
        try:
            tabs = self.query_one("#events", TabbedContent)
            if tabs.active == pane_id:
                return
            tab = tabs.get_tab(pane_id)
            label = f"{base} {theme.g(Glyph.STATUS)}"
            if str(tab.label) != label:
                tab.label = label
        except Exception:
            return

    def _clear_tab_activity(self, pane_id: str) -> None:
        base = tui_constants.TAB_BASE_LABELS.get(pane_id)
        if base is None:
            return
        try:
            tab = self.query_one("#events", TabbedContent).get_tab(pane_id)
            if str(tab.label) != base:
                tab.label = base
        except Exception:
            return

    def on_tabbed_content_tab_activated(self, event: TabbedContent.TabActivated) -> None:
        tabbed_content = getattr(event, "tabbed_content", None)
        if tabbed_content is None or tabbed_content.id != "events":
            return
        pane_id = getattr(event.pane, "id", None)
        if pane_id in tui_constants.TAB_BASE_LABELS:
            self._clear_tab_activity(pane_id)

    def _log_status(self, text: str, level: str = "info") -> None:
        """Write a status line to the Logs tab with the semantic palette."""
        self._write_log(text, level)

    def _notify_user(self, message: str, *, severity: str = "information", title: Optional[str] = None) -> None:
        """Record a user-facing notice in the Logs tab without showing a transient popup."""
        level = {"information": "ok", "warning": "warn", "error": "error"}.get(severity, "info")
        self._log_status(message, level)

    async def _check_for_update_on_startup(self) -> None:
        result = await asyncio.to_thread(check_for_update)
        message = update_status_message(result)
        if not message:
            return
        self._add_conversation_entry(tui_state.ConversationEntry(kind="system", content=message))
        self._notify_user(message)

    def _validated_interaction_mode(self, interaction_mode: str) -> str:
        if interaction_mode in {tui_constants.BUILD_INTERACTION_MODE, tui_constants.PLAN_INTERACTION_MODE}:
            return interaction_mode
        return tui_constants.BUILD_INTERACTION_MODE

    def _sync_planning_state_to_session(self) -> None:
        self.session.interaction_mode = self.interaction_mode
        self.session.permission_mode = self.permission_mode.value
        self.session.gigacode_enabled = self._gigacode_enabled
        self.session.web_search_mode = self._web_search_mode
        self.session.latest_plan_markdown = self._latest_plan or ""
        self.session.plan_pending = bool(self._latest_plan and self._plan_pending)
        self.session.plan_reofferable = bool(self._latest_plan and self._plan_reofferable)

    def _session_snapshot_locked(self) -> SessionRecord:
        """Return a detached session snapshot for background persistence.

        Call only while ``_persistence_lock`` is held. The snapshot gets shallow
        copies of mutable payloads so ``SessionStore.save`` never serializes the
        live ``self.session`` object on a worker thread.
        """
        self._sync_planning_state_to_session()
        record = self.session
        return SessionRecord(
            schema_version=record.schema_version,
            session_id=record.session_id,
            project_path=record.project_path,
            workspace_id=record.workspace_id,
            thread_id=record.thread_id,
            mode=record.mode,
            title=record.title,
            created_at=record.created_at,
            updated_at=record.updated_at,
            config=dict(record.config),
            history=list(record.history),
            compaction=dict(record.compaction),
            task_list_markdown=record.task_list_markdown,
            latest_plan_markdown=record.latest_plan_markdown,
            plan_pending=record.plan_pending,
            plan_reofferable=record.plan_reofferable,
            interaction_mode=record.interaction_mode,
            permission_mode=record.permission_mode,
            gigacode_enabled=record.gigacode_enabled,
            web_search_mode=record.web_search_mode,
            goal=dict(record.goal),
            loop=dict(record.loop),
            usage=dict(record.usage),
            active_project_path=record.active_project_path,
        )

    async def _save_session_async(self) -> None:
        """Persist lightweight session state without blocking Textual's loop."""
        async with self._persistence_lock:
            snapshot = self._session_snapshot_locked()
            await asyncio.to_thread(self.store.save, snapshot)
            self.session.updated_at = snapshot.updated_at

    async def _save_workspace_switch_async(
        self,
        *,
        old_root: Path,
        new_root: Path,
        old_label: str,
        new_label: str,
        old_branch: str,
        new_branch: str,
    ) -> None:
        """Atomically persist switch metadata and its durable journal boundary."""
        async with self._persistence_lock:
            snapshot = self._session_snapshot_locked()
            await asyncio.to_thread(
                self.store.save_workspace_switch,
                snapshot,
                self._session_recorder,
                old_root=str(old_root),
                new_root=str(new_root),
                old_label=old_label,
                new_label=new_label,
                old_branch=old_branch,
                new_branch=new_branch,
            )
            self.session.updated_at = snapshot.updated_at

    async def _save_session_history_async(self) -> None:
        """Flush metadata and refresh the in-memory projection off the UI loop.

        The history dump is not written by ``SessionStore.save``; canonical history
        has already been appended at semantic turn boundaries. Keeping this local
        projection current avoids replaying the entire JSONL log after every turn.
        """
        async with self._persistence_lock:
            agent = self.agent
            if agent is None:
                return
            snapshot = self._session_snapshot_locked()

            def dump_and_save() -> tuple[list[dict], dict, str]:
                history = agent.dump_message_history()
                compaction = agent.dump_compaction_state()
                self.store.save(snapshot)
                return history, compaction, snapshot.updated_at

            history, compaction, updated_at = await asyncio.to_thread(dump_and_save)
            self.session.history = history
            self.session.compaction = compaction
            self.session.updated_at = updated_at

    def _restore_plan_action_visibility(self) -> None:
        # A pending plan always gets the full decision set on restore:
        # `_plan_decision_active` is in-memory only, so gating on it hid
        # "Clear context and implement plan" and "Discuss further" whenever a
        # session with a pending plan was resumed.
        self._set_plan_actions_visible(
            self.interaction_mode == tui_constants.PLAN_INTERACTION_MODE and self._plan_pending,
            allow_discuss=self._plan_pending,
        )

    def _restore_prompt_history(self) -> None:
        """Seed the composer's recall history from per-project local state.

        History is project-scoped (shell-like), so it crosses sessions and app
        runs; a missing or corrupt file simply starts recall empty.
        """
        try:
            composer = self.query_one("#composer", tui_widgets.ChatComposer)
        except Exception:
            return
        composer.restore_prompt_history(load_prompt_history(self.store.root))

    def _persist_prompt_history(self, composer: tui_widgets.ChatComposer) -> None:
        """Write the composer's recall history through to local state.

        Runs after ChatComposer.action_submit has already recorded the entry,
        so dumping the widget's list is the single source of truth. Persistence
        must never break a submit.
        """
        try:
            save_prompt_history(self.store.root, composer.prompt_history)
        except Exception:
            pass

    async def on_chat_composer_submitted(self, event: tui_widgets.ChatComposer.Submitted) -> None:
        text = event.value
        stripped_text = text.strip()
        self._persist_prompt_history(event.composer)
        if stripped_text.lower() in THREAD_RESET_COMMANDS:
            if self._turn_active or self.agent_worker is not None:
                self._show_composer_hint(messages.BLOCK_STOP_BEFORE_RESET)
                self._notify_user(messages.BLOCK_STOP_BEFORE_RESET, severity="warning")
                return
            event.composer.load_text("")
            await self._reset_current_thread()
            return

        if await self._handle_tui_slash_command(stripped_text, event.composer):
            return

        if self._pending_model_selection is not None:
            if not stripped_text:
                self._set_composer_status(messages.MODEL_PLACEHOLDER)
                return
            event.composer.load_text("")
            await self._answer_model_selection(stripped_text)
            return

        if self._pending_effort_selection is not None:
            if not stripped_text:
                self._set_composer_status(messages.EFFORT_PLACEHOLDER)
                return
            event.composer.load_text("")
            await self._answer_effort_selection(stripped_text)
            return

        if self._pending_theme_selection is not None:
            if not stripped_text:
                self._set_composer_status(messages.THEME_PLACEHOLDER)
                return
            event.composer.load_text("")
            await self._answer_theme_selection(stripped_text)
            return

        if await self._handle_skill_slash_command(stripped_text, event.composer):
            return

        if self._pending_question is not None:
            if not stripped_text:
                self._set_composer_status(messages.QUESTION_PLACEHOLDER)
                return
            event.composer.load_text("")
            await self._answer_pending_question(stripped_text)
            return

        if self._pending_approval is not None:
            self._set_composer_status(messages.APPROVAL_PLACEHOLDER)
            self._notify_user(messages.BLOCK_PENDING_APPROVAL, severity="warning")
            return

        if self._plan_decision_active:
            self._set_composer_status(messages.PLAN_READY_PLACEHOLDER)
            self._notify_user(messages.BLOCK_PLAN_DECISION, severity="warning")
            return

        if not stripped_text or self.agent is None:
            if stripped_text:
                self._set_settings_status(messages.SETTINGS_REQUIRED, tone="warning")
                if self.config is None:
                    self._set_composer_status(messages.DISCONNECTED_COMPOSER_PLACEHOLDER)
                    self._show_composer_hint(messages.DISCONNECTED_ACTIVITY, tone="warning")
            return
        # Build attachments first (without clearing pending) so the vision gate
        # can block before we consume the composer text and pending images.
        attachments = self._build_mention_attachments(text)
        if self._pending_image_attachments:
            attachments = (attachments or []) + self._pending_image_attachments
        # Pre-send vision gate: block when the current model can't see images.
        # Catches @file.png mentions (which bypass add_pending_image_attachment)
        # and serves as a final gate for all attachment paths. When blocked, the
        # composer text and pending attachments are PRESERVED so the user can
        # remove the image (/detach or edit the @mention) or switch model and
        # resend — nothing is added to the transcript because nothing was sent.
        # History images are NOT blocked (stripped to placeholders by
        # _history_for_llm, send proceeds normally).
        if attachments and not self._model_supports_vision():
            has_image = any(a.get("type") == "image" for a in attachments)
            if has_image:
                self._add_vision_mismatch_system_message(context="attachment")
                self._show_composer_hint(messages.MODEL_NON_VISION_IMAGE_BLOCKED, tone="warning")
                return
        # Safe to consume — clear the composer, pending attachments, and the
        # attach hint (which would otherwise linger during generation or queueing).
        event.composer.load_text("")
        self._pending_image_attachments.clear()
        self._clear_composer_hint()
        if self._turn_active or self.agent_worker is not None:
            self._queue_user_message(text, attachments)
            return
        self._add_conversation_entry(tui_state.ConversationEntry(kind="user", content=text))
        self.agent_worker = self.run_worker(
            self._process_message(text, attachments), name="kolega-turn", group="turns", exclusive=True
        )

    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        if event.text_area.id == "composer":
            self._refresh_completion_dropdown()

    def _refresh_completion_dropdown(self) -> None:
        try:
            dropdown = self.query_one("#completion_dropdown", tui_widgets.CompletionDropdown)
            composer = self.query_one("#composer", tui_widgets.ChatComposer)
        except Exception:
            return
        slash = composer.active_slash_query()
        if slash is not None:
            commands = search_commands(slash[0], self.skill_catalog, limit=8, skills_enabled=self.skills_enabled)
            if not commands:
                dropdown.close()
                return
            dropdown.open_with([tui_widgets.command_completion_item(entry) for entry in commands])
            return
        active = composer.active_mention_query()
        if active is None:
            dropdown.close()
            return
        # Search the cached snapshot only — never block the keystroke on os.walk. A
        # stale/empty cache triggers a background refresh that re-runs this when done.
        self._maybe_refresh_file_index()
        entries = self.file_index.cached_search(active[0], limit=8)
        if not entries:
            dropdown.close()
            return
        dropdown.open_with([tui_widgets.file_completion_item(entry) for entry in entries])

    def _maybe_refresh_file_index(self) -> None:
        """Refresh the file index on a worker thread if it's stale and not already running."""
        if self._file_index_refreshing or not self.file_index.is_stale():
            return
        self._file_index_refreshing = True
        self.run_worker(self._refresh_file_index(), name="kolega-file-index", group="file-index")

    async def _refresh_file_index(self) -> None:
        try:
            await asyncio.to_thread(self.file_index.refresh)
        finally:
            self._file_index_refreshing = False
        # Surface freshly-walked results if an @-mention is still being typed.
        self._refresh_completion_dropdown()

    def on_descendant_focus(self, event: events.DescendantFocus) -> None:
        # Fires after screen.focused settles. Catches AUTO_FOCUS landing on the
        # conversation transcript (resume/resize) and any other stray focus while a
        # prompt is shown, pulling focus back to the active option list.
        self._heal_prompt_focus()

    def on_descendant_blur(self, event: events.DescendantBlur) -> None:
        # A background click does set_focus(None) and emits no DescendantFocus, so
        # the focus hook alone would miss it. Re-assert after the refresh settles, so
        # we run after any AUTO_FOCUS/_reset_focus the same blur triggered.
        self.call_after_refresh(self._heal_prompt_focus)

    def on_app_focus(self, event: events.AppFocus) -> None:
        # When a terminal window regains OS focus, the next likely action is typing
        # a prompt. Defer so this runs after Textual's resume/auto-focus churn, while
        # still respecting active prompt lists and disabled composer states.
        self._schedule_primary_focus_restore()

    async def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        if event.option_list.id == "question_actions":
            event.stop()
            await self._answer_question_option(event.option_index)
            return
        if event.option_list.id == "approval_actions":
            event.stop()
            await self._answer_approval_option(event.option_index)
            return
        if event.option_list.id == "model_actions":
            event.stop()
            await self._answer_model_option(event.option_index)
            return
        if event.option_list.id == "effort_actions":
            event.stop()
            await self._answer_effort_option(event.option_index)
            return
        if event.option_list.id == "theme_actions":
            event.stop()
            await self._answer_theme_option(event.option_index)
            return
        if event.option_list.id == "plan_actions":
            event.stop()
            if event.option_id == "implement_plan":
                await self._implement_pending_plan()
            elif event.option_id == "implement_plan_clear":
                await self._implement_pending_plan(clear_context=True)
            elif event.option_id == "discuss_plan":
                await self._discuss_pending_plan()
            return
        if event.option_list.id != "completion_dropdown":
            return
        event.stop()
        try:
            dropdown = self.query_one("#completion_dropdown", tui_widgets.CompletionDropdown)
            composer = self.query_one("#composer", tui_widgets.ChatComposer)
        except Exception:
            return
        entry = dropdown.entry_at(event.option_index)
        if entry is not None:
            composer.apply_completion(entry)
            if isinstance(entry, SlashCommandEntry) or not entry.is_dir:
                dropdown.close()
        composer.focus()

    def _build_mention_attachments(self, text: str) -> list[dict] | None:
        """Expand @path mentions in a prompt into file attachments."""
        try:
            attachments, unresolved = build_file_attachments(text, self.active_project_path)
        except Exception:
            return None
        if unresolved:
            joined = ", ".join(f"@{path}" for path in unresolved)
            self._show_composer_hint(messages.MENTIONS_NOT_FOUND.format(mentions=joined))
        return attachments or None

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "detach_btn":
            await self._command_detach("")
            return
        if event.button.id == "open_settings":
            if self.config is None:
                await self.action_open_onboarding()
            else:
                self.action_open_settings()
            return
        if event.button.id == "save_settings":
            await self._save_settings_from_ui()
            return
        if event.button.id == "provider_chatgpt_login":
            await self._settings_login_chatgpt()
            return
        if event.button.id == "provider_chatgpt_logout":
            self._settings_logout_chatgpt()
            return
        if event.button.id == "provider_remove_api_key":
            self._settings_remove_api_key()
            return
        if event.button.id == "provider_test_connection":
            await self._test_settings_connection()
            return
        if event.button.id and event.button.id.startswith("mcp_"):
            if await self._handle_mcp_settings_button(event.button.id):
                return

    def copy_to_clipboard(self, text: str) -> None:
        super().copy_to_clipboard(text)
        if sys.platform != "darwin":
            return

        try:
            subprocess.run(["pbcopy"], input=text, text=True, check=True)
        except (OSError, subprocess.CalledProcessError):
            try:
                self._notify_user(messages.COPY_MACOS_FAILED, severity="warning")
            except Exception:
                pass

    def _mode_switch_blocked(self) -> bool:
        if self._pending_approval is not None:
            self._set_composer_status(messages.APPROVAL_PLACEHOLDER)
            self._notify_user(messages.BLOCK_PENDING_APPROVAL, severity="warning")
            return True
        if self._turn_active or self.agent_worker is not None:
            self._show_composer_hint(messages.BLOCK_STOP_BEFORE_MODE_SWITCH)
            self._notify_user(messages.BLOCK_STOP_BEFORE_MODE_SWITCH, severity="warning")
            return True
        if self._plan_decision_active:
            self._set_composer_status(messages.PLAN_READY_PLACEHOLDER)
            self._notify_user(messages.BLOCK_PLAN_DECISION_MODE_SWITCH, severity="warning")
            return True
        return False

    def _permission_mode_switch_blocked(self) -> bool:
        if self._pending_approval is not None:
            self._set_composer_status(messages.APPROVAL_PLACEHOLDER)
            self._notify_user(messages.BLOCK_PENDING_APPROVAL_MODE_SWITCH, severity="warning")
            return True
        return False

    async def action_toggle_interaction_mode(self) -> None:
        if self._mode_switch_blocked():
            return

        target = (
            tui_constants.PLAN_INTERACTION_MODE
            if self.interaction_mode == tui_constants.BUILD_INTERACTION_MODE
            else tui_constants.BUILD_INTERACTION_MODE
        )
        await self._set_interaction_mode(target)

    async def action_toggle_permission_mode(self) -> None:
        if self._permission_mode_switch_blocked():
            return
        target = PermissionMode.AUTO if self.permission_mode == PermissionMode.ASK else PermissionMode.ASK
        await self._set_permission_mode(target)

    async def action_toggle_sidebar(self) -> None:
        self._set_sidebar_visible(not self.sidebar_visible)
        message = messages.SIDEBAR_SHOWN if self.sidebar_visible else messages.SIDEBAR_HIDDEN
        self._notify_user(message)

    @property
    def _modal_cover_active(self) -> bool:
        """A full-screen modal screen currently covers the root conversation screen."""
        try:
            return len(self.screen_stack) > 1
        except Exception:
            return False

    def _push_fullscreen_modal(self, screen: Screen[ScreenResultT]) -> None:
        """Push a full-screen modal, re-syncing the deferred transcript when it closes.

        While a modal covers the root screen, transcript widget updates are
        deferred (see transcript._flush_conversation_render); the dismiss
        callback fires however the screen closes and triggers the catch-up.
        """
        self.push_screen(screen, callback=self._on_fullscreen_modal_dismissed)

    def _on_fullscreen_modal_dismissed(self, _result: object = None) -> None:
        # Promote an unflushed invalidation to a deferred full sync before
        # neutralizing its timer; the rebuild below subsumes that flush.
        if self._render_pending:
            self._transcript_sync_pending = True
        self._render_pending = False

        def resync() -> None:
            if self._modal_cover_active or not self._transcript_sync_pending:
                return
            self._transcript_sync_pending = False
            self._render_conversation()

        try:
            # Re-check after the stack pop has fully settled (nested modals keep
            # the cover active, so their dismissal must not resync early).
            self.call_after_refresh(resync)
        except Exception:
            resync()

    def action_open_settings(self, category: str = "model") -> None:
        """Open the full-screen settings editor."""
        if self._settings_screen is not None or self._onboarding_screen is not None:
            return
        if self._pending_approval is not None or self._pending_question is not None or self._plan_decision_active:
            self._notify_user("Resolve the active prompt before opening Settings.", severity="warning")
            return
        screen = tui_settings_screen.SettingsScreen(self, category=category)
        self._settings_screen = screen
        self._push_fullscreen_modal(screen)

    def action_open_memory(self, *, inspect_disabled: bool = False) -> None:
        """Open the backend-neutral project-memory browser and editor."""
        if self._memory_screen is not None or self._onboarding_screen is not None:
            return
        if self._pending_approval is not None or self._pending_question is not None or self._plan_decision_active:
            self._notify_user("Resolve the active prompt before opening Memory.", severity="warning")
            return
        screen = tui_memory.MemoryScreen(self, inspect_disabled=inspect_disabled)
        self._memory_screen = screen
        self._push_fullscreen_modal(screen)

    def _memory_mutation_blocked(self) -> bool:
        if not self._turn_active and self.agent_worker is None:
            return False
        self._notify_user(
            "Wait for the active agent turn to finish, or cancel it, before changing memory.",
            severity="warning",
        )
        return True

    async def _refresh_agent_memory(self) -> None:
        """Best-effort prompt refresh after an already committed memory mutation."""
        try:
            agent = self.agent
            refresh_agent = getattr(agent, "refresh_memory_context", None)
            if callable(refresh_agent):
                await asyncio.to_thread(refresh_agent)
            else:
                await asyncio.to_thread(self.memory_manager.refresh)
        except Exception:
            self._notify_user(
                "Memory was updated, but the active prompt could not be refreshed. "
                "It will be reloaded before the next top-level turn.",
                severity="warning",
            )

    async def _apply_memory_enabled(self, enabled: bool) -> None:
        """Commit the enabled flag, then refresh the idle agent prompt."""
        await asyncio.to_thread(self.memory_manager.set_enabled, enabled)
        await self._refresh_agent_memory()

    def _close_memory_manager(self) -> None:
        self.memory_manager.close()

    async def action_open_onboarding(self) -> None:
        """Open the independent connection wizard."""
        if self.config is not None or self._onboarding_screen is not None or self._settings_screen is not None:
            return
        screen = tui_onboarding.OnboardingScreen(self)
        self._onboarding_screen = screen
        await self.push_screen(screen, callback=self._on_fullscreen_modal_dismissed)

    def action_open_sub_agent(self, key: Optional[str] = None) -> None:
        """Open the full-screen sub-agent inspector (mission control)."""
        if self._sub_agent_inspector is not None:
            return
        if not self._sub_agent_activities:
            self._notify_user(messages.SUB_AGENT_INSPECTOR_EMPTY, severity="information")
            return
        if key is None or key not in self._sub_agent_activities:
            key = self._default_sub_agent_key()
        if key is None:
            return
        screen = tui_sub_agents.SubAgentInspectorScreen(self, key)
        self._sub_agent_inspector = screen
        self._push_fullscreen_modal(screen)

    def action_open_changes(self, path: Optional[str] = None, *, baseline_id: Optional[int] = None) -> None:
        """Open the full-screen session changes inspector."""
        if not self._changes_available():
            return
        if baseline_id is not None:
            self._set_changes_baseline(baseline_id)
        if self._changes_inspector is not None:
            return
        paths = {change.path for change in self._session_diff_files}
        if path is None or path not in paths:
            path = self._default_changes_path() or ""
        screen = tui_changes.ChangesInspectorScreen(self, path)
        self._changes_inspector = screen
        self._push_fullscreen_modal(screen)
        self._start_session_diff_refresh()

    def _initialize_session_diff_tracker(self, baseline_label: str = "") -> None:
        """Set up git-only net diff tracking for the Changes inspector.

        Tracker creation is one fast ``git rev-parse``; the baseline capture
        can read every dirty file, so it runs in a worker. Until it finishes,
        the tracker has no checkpoint 0 and turn checkpoints skip themselves.
        """
        self._session_diff_generation += 1
        self._session_diff_tracker = tui_session_diff.GitSessionDiffTracker.create(self.active_project_path)
        self._session_diff_files = []
        self._session_diff_scope = None
        if self._session_diff_tracker is None:
            return
        try:
            self.run_worker(
                self._session_diff_baseline_worker(
                    self._session_diff_tracker, self._session_diff_generation, baseline_label
                ),
                name="kolega-diff-baseline",
                group="session-diff-baseline",
            )
        except Exception:
            self._session_diff_tracker = None
            self._session_diff_files = []

    async def _session_diff_baseline_worker(
        self, tracker: tui_session_diff.SessionDiffTrackerBase, generation: int, label: str
    ) -> None:
        try:
            started = time.monotonic()
            await asyncio.to_thread(self._tracker_capture_baseline, tracker, label)
            if self.show_logs:
                self._write_log(f"Session diff baseline captured in {time.monotonic() - started:.2f}s", "debug")
        except Exception:
            if generation == self._session_diff_generation and tracker is self._session_diff_tracker:
                self._session_diff_tracker = None
                self._session_diff_files = []
            return
        if generation != self._session_diff_generation or tracker is not self._session_diff_tracker:
            return
        self._start_scope_probe()

    def _tracker_capture_baseline(self, tracker: tui_session_diff.SessionDiffTrackerBase, label: str) -> None:
        """Capture the session baseline under the tracker lock. Worker threads only."""
        with self._session_diff_lock:
            tracker.capture_baseline(label)

    async def _confirm_worktree_switch(self, new_root: Path, branch: str, old_root: Path) -> tuple[bool, str]:
        """Ask the user to approve an agent-initiated workspace switch.

        Returns ``(approved, answer)``. Only the exact approve label approves:
        free text the user typed instead is a decline, and is handed back to the
        model so it can react. An unanswered prompt — no control lease, or the
        channel's timeout — is also a decline, which is the safe direction for a
        switch nobody asked for.
        """
        if self._pending_question is not None:
            raise ToolError("A question is already waiting for an answer. Resolve it before switching worktrees.")

        branch_note = f" on branch `{branch}`" if branch else ""
        question = messages.WORKTREE_SWITCH_CONFIRM_QUESTION.format(
            old_root=old_root,
            new_root=new_root,
            branch_note=branch_note,
        )
        try:
            answer = await self._ask_user_choice(
                question,
                [messages.WORKTREE_SWITCH_CONFIRM_APPROVE, messages.WORKTREE_SWITCH_CONFIRM_DECLINE],
                [
                    messages.WORKTREE_SWITCH_CONFIRM_APPROVE_DESCRIPTION,
                    messages.WORKTREE_SWITCH_CONFIRM_DECLINE_DESCRIPTION.format(old_root=old_root),
                ],
            )
        except ToolError:
            # Nobody could answer (no controller, or the request timed out).
            return False, ""
        if answer == messages.WORKTREE_SWITCH_CONFIRM_APPROVE:
            return True, answer
        if answer == messages.WORKTREE_SWITCH_CONFIRM_DECLINE:
            return False, ""
        return False, answer

    async def _switch_active_worktree(self, path: str) -> str:
        """Durably commit a switch of the active workspace; it applies at turn end."""
        agent = self.agent
        if agent is None:
            raise ToolError("The active agent workspace is not ready yet.")
        pending = self._pending_workspace_switch
        if pending is not None:
            raise ToolError(
                f"A workspace switch to `{pending.new_root}` is already committed for this turn. "
                "End your turn to complete it."
            )

        try:
            target = await asyncio.to_thread(resolve_worktree, self.project_path, path)
        except WorktreeError as exc:
            raise ToolError(str(exc)) from exc

        new_root = target.path.resolve()
        old_root = self.active_project_path.resolve()
        if new_root == old_root:
            label = f" ({target.branch})" if target.branch else ""
            return f"Workspace is already active at `{new_root}`{label}; the Changes baseline was not reset."

        terminal_manager = getattr(agent, "terminal_manager", None)
        if isinstance(terminal_manager, LocalTerminalManager) and terminal_manager.has_running_sessions:
            raise ToolError(
                "Cannot switch worktrees while terminal sessions are running. Stop them with `kill_command`, "
                "then call `switch_worktree` again by itself."
            )

        # Asked after the cheap refusals (no prompt for a no-op or for a switch
        # that is going to be refused anyway) and before anything durable is
        # written. The agent may only switch when the user asked for it, so the
        # user gets the final say regardless of permission mode or goal loop.
        approved, answer = await self._confirm_worktree_switch(new_root, target.branch, old_root)
        if not approved:
            declined = messages.WORKTREE_SWITCH_DECLINED.format(old_root=old_root)
            if answer:
                declined += messages.WORKTREE_SWITCH_DECLINED_ANSWER.format(answer=answer)
            return declined

        try:
            old_info = await asyncio.to_thread(resolve_worktree, self.project_path, old_root)
        except WorktreeError:
            old_info = None
        old_branch = old_info.branch if old_info is not None else ""

        previous_active_metadata = self.session.active_project_path
        self.session.active_project_path = None if new_root == self.project_path else str(new_root)
        try:
            # The boundary is durable before the model is told the switch
            # happened: a crash between now and the rebuild resumes in the new
            # workspace, which is what the transcript claims.
            await self._save_workspace_switch_async(
                old_root=old_root,
                new_root=new_root,
                old_label=old_root.name,
                new_label=new_root.name,
                old_branch=old_branch,
                new_branch=target.branch,
            )
        except Exception as exc:
            self.session.active_project_path = previous_active_metadata
            raise ToolError(
                f"Could not persist the workspace switch; the active workspace is unchanged: {exc}"
            ) from exc

        self._pending_workspace_switch = tui_state.PendingWorkspaceSwitch(
            new_root=new_root,
            old_root=old_root,
            branch=target.branch,
            old_branch=old_branch,
            previous_active_metadata=previous_active_metadata,
        )
        branch_note = f" on branch `{target.branch}`" if target.branch else ""
        return (
            f"Committed a switch of the active workspace to `{new_root}`{branch_note}. It takes effect when "
            "this turn ends, so end your turn now — further tool calls this turn would still run in the "
            "previous workspace. You will be prompted to continue in the new workspace."
        )

    async def _apply_pending_workspace_switch(self) -> Optional[str]:
        """Rebuild the workspace for a committed switch once the stream is over.

        Returns the agent-facing continuation prompt when a switch was applied,
        so the turn driver can hand control back in the new workspace.
        """
        pending = self._pending_workspace_switch
        if pending is None:
            return None
        self._pending_workspace_switch = None
        branch_note = f" on branch `{pending.branch}`" if pending.branch else ""
        worktree_label = pending.branch or pending.new_root.name
        try:
            await self._activate_workspace(pending.new_root, baseline_label=f'Workspace switched · "{worktree_label}"')
        except Exception as exc:
            await self._restore_previous_workspace(pending, exc)
            return None

        notice = (
            f"Workspace switched to `{pending.new_root}`{branch_note}. Changes and Rewind now use a fresh "
            "“Workspace switched” baseline; prior-workspace checkpoints are no longer shown."
        )
        self._add_conversation_entry(tui_state.ConversationEntry(kind="system", content=notice))
        return (
            f"The active workspace is now `{pending.new_root}`{branch_note}; the previous turn ended to "
            "complete the switch. Continue the work that prompted it, or reply briefly if none remains."
        )

    async def _restore_previous_workspace(self, pending: tui_state.PendingWorkspaceSwitch, error: Exception) -> None:
        """Recommit and rebuild the pre-switch workspace after a failed rebuild."""
        self.session.active_project_path = pending.previous_active_metadata
        try:
            # Append the reverse boundary so the journal records what actually
            # happened rather than silently contradicting the earlier event.
            await self._save_workspace_switch_async(
                old_root=pending.new_root,
                new_root=pending.old_root,
                old_label=pending.new_root.name,
                new_label=pending.old_root.name,
                old_branch=pending.branch,
                new_branch=pending.old_branch,
            )
        except Exception:
            pass
        try:
            await self._activate_workspace(pending.old_root, baseline_label="")
        except Exception as fallback_error:
            self._set_chat_enabled(False)
            message = (
                f"Could not rebuild the workspace at `{pending.new_root}` ({error}), and rebuilding the "
                f"previous workspace `{pending.old_root}` also failed ({fallback_error}). "
                "Restart or resume the session."
            )
            self._add_conversation_entry(tui_state.ConversationEntry(kind="system", content=message))
            self._notify_user(message, severity="error")
            return
        message = (
            f"Could not switch the active workspace to `{pending.new_root}` ({error}); "
            f"continuing in `{pending.old_root}`."
        )
        self._add_conversation_entry(tui_state.ConversationEntry(kind="system", content=message))
        self._notify_user(message, severity="warning")

    async def _activate_workspace(self, root: Path, *, baseline_label: str) -> None:
        """Point the TUI at ``root`` and rebuild the agent there.

        The rebuild goes through ``_build_agent`` — the same assembly a resume
        uses — so catalogs, extensions, and tool services cannot drift from
        what a restart in this workspace would produce.
        """
        if self.config is None:
            raise RuntimeError("Agent configuration is not available.")
        self.active_project_path = root
        self.file_index = WorkspaceFileIndex(root)
        self._file_index_refreshing = False
        self._pending_edit_previews = {}
        if self._session_diff_timer is not None:
            self._session_diff_timer.pause()
            self._session_diff_timer = None
        self._session_file_changes = []
        self._session_diff_baseline_id = None
        self._session_diff_dirty = False
        self._initialize_session_diff_tracker(baseline_label)
        await self._build_agent(self.config, rebuild=True, restore_transcript=False, preserve_queued=True)
        self._close_changes_inspector()
        self._maybe_refresh_file_index()
        self._refresh_status_dashboard()
        self._update_mode_chrome()

    def _start_scope_probe(self) -> None:
        """Resolve the diff scope once so the dashboard can name the worktree.

        The scope shells out to git, and waiting for the Changes screen to be
        opened would leave sibling-worktree sessions indistinguishable.
        """
        tracker = self._session_diff_tracker
        if tracker is None:
            return
        generation = self._session_diff_generation
        try:
            self.run_worker(
                self._scope_probe_worker(tracker, generation),
                name="kolega-diff-scope",
                group="session-diff-scope",
            )
        except Exception:
            pass

    async def _scope_probe_worker(self, tracker: tui_session_diff.SessionDiffTrackerBase, generation: int) -> None:
        try:
            scope = await asyncio.to_thread(self._tracker_scope, tracker)
        except Exception:
            return
        if scope is None:
            return
        if generation != self._session_diff_generation or tracker is not self._session_diff_tracker:
            return
        self._session_diff_scope = scope
        self._refresh_status_dashboard()

    def _tracker_scope(
        self, tracker: tui_session_diff.SessionDiffTrackerBase, baseline_id: Optional[int] = None
    ) -> Optional[tui_session_diff.DiffScope]:
        """Read the tracker scope under the tracker lock. Worker threads only."""
        with self._session_diff_lock:
            return tracker.scope(checkpoint_id=baseline_id)

    def _tracker_refresh(
        self,
        tracker: tui_session_diff.SessionDiffTrackerBase,
        event_paths: list[str],
        baseline_id: Optional[int],
    ) -> list[tui_session_diff.SessionDiffFile]:
        """Collect net changes under the tracker lock. Worker threads only."""
        with self._session_diff_lock:
            return tracker.refresh(event_paths, checkpoint_id=baseline_id)

    def _changes_scope_label(self) -> str:
        """Pre-rendered worktree/branch prefix for the Changes screen, or ""."""
        scope = self._session_diff_scope
        if scope is None:
            return ""
        if scope.linked_worktree and scope.root_label:
            label = messages.CHANGES_SCOPE_WORKTREE.format(name=scope.root_label)
            return f"{label} ({scope.branch})" if scope.branch else label
        if scope.branch:
            return messages.CHANGES_SCOPE_BRANCH.format(branch=scope.branch)
        return ""

    def _changes_history_note(self) -> str:
        """One-line note about history the Changes screen deliberately omits."""
        scope = self._session_diff_scope
        if scope is None:
            return ""
        # A non-git project has no commits to report on; branch/worktree
        # presence is what identifies a git-backed scope.
        git_backed = bool(scope.branch or scope.linked_worktree)
        if git_backed and not scope.history_tracked:
            return messages.CHANGES_HISTORY_UNTRACKED
        if scope.history_moved:
            return messages.CHANGES_HISTORY_MOVED
        return ""

    def _initialize_ledger_diff_tracker(self) -> None:
        """Non-git fallback: track agent-recorded edits once the snapshot service exists."""
        if self._session_diff_tracker is not None or self.agent is None or self.agent.tool_collection is None:
            return
        service = getattr(self.agent.tool_collection, "snapshot_service", None)
        if service is None:
            return
        tracker = tui_session_diff.SnapshotLedgerDiffTracker(self.active_project_path, service)
        try:
            tracker.capture_baseline()
        except Exception:
            return
        self._session_diff_tracker = tracker
        self._start_scope_probe()

    def _changes_available(self) -> bool:
        return self._session_diff_tracker is not None

    # ---- rewind --------------------------------------------------------------

    def _changes_baseline_ladder(self) -> list[int]:
        tracker = self._session_diff_tracker
        if tracker is None:
            return []
        return [checkpoint.checkpoint_id for checkpoint in tracker.checkpoints()]

    def _changes_baseline_checkpoint(self) -> Optional[tui_session_diff.TurnCheckpoint]:
        tracker = self._session_diff_tracker
        if tracker is None:
            return None
        if self._session_diff_baseline_id is None:
            checkpoints = tracker.checkpoints()
            return checkpoints[0] if checkpoints else None
        return tracker.checkpoint_for_id(self._session_diff_baseline_id)

    def _changes_baseline_label(self) -> str:
        checkpoint = self._changes_baseline_checkpoint()
        if checkpoint is None:
            return messages.CHANGES_BASELINE_SESSION_START
        if checkpoint.checkpoint_id == 0:
            if not checkpoint.label:
                return messages.CHANGES_BASELINE_SESSION_START
            sep = theme.g(Glyph.BULLET_SEP)
            stamp = time.strftime("%H:%M", time.localtime(checkpoint.created_at))
            return f"{checkpoint.label} {sep} {stamp}"
        sep = theme.g(Glyph.BULLET_SEP)
        stamp = time.strftime("%H:%M", time.localtime(checkpoint.created_at))
        label = f"Turn {checkpoint.checkpoint_id}"
        if checkpoint.label:
            label += f' {sep} "{checkpoint.label}"'
        return f"{label} {sep} {stamp}"

    def _set_changes_baseline(self, baseline_id: Optional[int]) -> None:
        ladder = self._changes_baseline_ladder()
        if not ladder or (baseline_id is not None and baseline_id not in ladder):
            return
        normalized = None if baseline_id == ladder[0] else baseline_id
        if normalized == self._session_diff_baseline_id:
            return
        self._session_diff_baseline_id = normalized
        self._session_diff_dirty = True
        self._invalidate_changes_detail()
        self._start_session_diff_refresh()

    def _shift_changes_baseline(self, delta: int) -> None:
        ladder = self._changes_baseline_ladder()
        if not ladder:
            return
        current = self._session_diff_baseline_id
        index = ladder.index(current) if current in ladder else 0
        self._set_changes_baseline(ladder[max(0, min(len(ladder) - 1, index + delta))])

    def _reset_changes_baseline(self) -> None:
        if self._session_diff_baseline_id is None:
            return
        self._session_diff_baseline_id = None
        self._session_diff_dirty = True
        self._start_session_diff_refresh()

    def _rewind_target_turn(self, baseline_id: Optional[int]) -> Optional[TurnSummary]:
        """The first journal turn at/after the checkpoint: the conversation boundary.

        Checkpoint capture strictly precedes the turn.started event in serial
        code, so the timestamp comparison is an exact join, not a heuristic.
        """
        tracker = self._session_diff_tracker
        if tracker is None:
            return None
        checkpoints = tracker.checkpoints()
        checkpoint = (
            (checkpoints[0] if baseline_id is None else tracker.checkpoint_for_id(baseline_id)) if checkpoints else None
        )
        if checkpoint is None:
            return None
        for turn in self._session_recorder.list_rewindable_turns():
            try:
                started = datetime.fromisoformat(turn.started_at).timestamp()
            except ValueError:
                continue
            if started >= checkpoint.created_at:
                return turn
        return None

    async def _rewind_worker(self, baseline_id: Optional[int], label: str, paths: Optional[set[str]] = None) -> None:
        """Restore files to the checkpoint and (whole-turn only) rewind the conversation."""
        if self._turn_active or self.agent_worker is not None:
            self._notify_user(messages.REWIND_BLOCKED_TURN, severity="warning")
            return
        tracker = self._session_diff_tracker
        agent = self.agent
        if tracker is None or agent is None:
            return
        self._rewind_running = True
        try:
            while self._session_diff_refresh_running:
                await asyncio.sleep(0.05)
            event_paths = [change.path for change in self._session_file_changes]
            try:
                plan = await asyncio.to_thread(
                    lambda: tracker.build_restore_plan(checkpoint_id=baseline_id, event_paths=event_paths, paths=paths)
                )
            except ValueError:
                self._set_changes_baseline(None)  # the checkpoint was evicted; reset the scope
                return

            snapshot_id = ""
            service = getattr(agent.tool_collection, "snapshot_service", None) if agent.tool_collection else None
            if service is not None and plan:
                try:
                    snapshot = await asyncio.to_thread(
                        lambda: service.create_manual_snapshot(
                            paths=[item.display_path for item in plan],
                            reason=f"pre-rewind to {label}",
                        )
                    )
                    snapshot_id = snapshot.snapshot_id
                except Exception:
                    snapshot_id = ""

            try:
                result = await asyncio.to_thread(tracker.apply_restore_plan, plan)
            except tui_session_diff.RewindDriftError:
                self._notify_user(messages.REWIND_DRIFT, severity="warning")
                return
            if result.errors:
                detail = "; ".join(f"{path}: {reason}" for path, reason in result.errors[:3])
                self._notify_user(messages.REWIND_PARTIAL.format(detail=detail), severity="error")
                return

            conversation_rewound = False
            if paths is None:
                target = await asyncio.to_thread(self._rewind_target_turn, baseline_id)
                if target is not None:
                    outcome = await asyncio.to_thread(self._session_recorder.record_rewind, target.turn_id)
                    await self._apply_conversation_rewind(outcome)
                    conversation_rewound = True

            count = len(result.restored) + len(result.deleted)
            note = (
                messages.REWIND_DONE_CONVERSATION.format(count=count, label=label)
                if conversation_rewound
                else messages.REWIND_DONE.format(count=count, label=label)
            )
            if result.skipped:
                detail = "; ".join(f"{path} ({reason})" for path, reason in result.skipped[:3])
                note += messages.REWIND_SKIPPED_NOTE.format(detail=detail)
            if snapshot_id:
                note += messages.REWIND_SAFETY_NOTE.format(snapshot_id=snapshot_id)
            self._add_conversation_entry(tui_state.ConversationEntry(kind="system", content=note))
            self._notify_user(note, severity="warning" if result.skipped else "information")
            if conversation_rewound:
                self._close_changes_inspector()
                self._schedule_primary_focus_restore()
        finally:
            self._rewind_running = False
            self._session_diff_dirty = True
            self._start_session_diff_refresh()

    async def _apply_conversation_rewind(self, outcome: RewindOutcome) -> None:
        """Rebuild agent and transcript state through the resume path."""
        agent = self.agent
        assert agent is not None
        record = await asyncio.to_thread(self.store.load, self.session.session_id)
        agent.restore_message_history(record.history)
        agent.restore_compaction_state(record.compaction)
        self.session.history = record.history
        self.session.compaction = record.compaction
        self._clear_queued_messages()
        self._restore_conversation_history(record.history)
        self._add_conversation_entry(
            tui_state.ConversationEntry(
                kind="system",
                content=messages.REWOUND_MARKER.format(
                    excerpt=tui_agent_runtime.checkpoint_excerpt(outcome.user_message_text)
                ),
            )
        )
        if self._goal is not None and self._goal.is_active:
            await self._pause_goal(messages.REWIND_GOAL_PAUSED)
        if self._scheduled_loop is not None and self._scheduled_loop.is_active:
            await self._stop_loop(messages.REWIND_LOOP_STOPPED, notify=False)
        try:
            composer = self.query_one("#composer", tui_widgets.ChatComposer)
            composer.load_text(outcome.user_message_text)
        except Exception:
            pass
        try:
            await agent.count_current_context()
        except Exception:
            pass
        await self._save_session_async()

    def _mark_session_diff_dirty(self) -> None:
        self._session_diff_dirty = True
        if self._changes_inspector is None:
            return
        self._schedule_session_diff_refresh()

    def _schedule_session_diff_refresh(self) -> None:
        if self._session_diff_timer is not None or self._session_diff_refresh_running:
            return
        try:
            self._session_diff_timer = self.set_timer(
                SESSION_DIFF_REFRESH_INTERVAL,
                self._session_diff_timer_fired,
                name="session-diff-refresh",
            )
        except Exception:
            self._session_diff_timer = None

    def _session_diff_timer_fired(self) -> None:
        self._session_diff_timer = None
        if not self._session_diff_dirty or self._changes_inspector is None:
            return
        self._start_session_diff_refresh()

    def _start_session_diff_refresh(self) -> None:
        tracker = self._session_diff_tracker
        if tracker is None or self._session_diff_refresh_running or self._rewind_running:
            return
        try:
            self._session_diff_refresh_running = True
            self.run_worker(self._session_diff_refresh_worker(), name="kolega-session-diff", group="session-diff")
        except Exception:
            self._session_diff_refresh_running = False

    async def _session_diff_refresh_worker(self) -> None:
        tracker = self._session_diff_tracker
        generation = self._session_diff_generation
        try:
            self._session_diff_dirty = False
            event_paths = [change.path for change in self._session_file_changes]
            baseline_id = self._session_diff_baseline_id
            if baseline_id is not None and tracker is not None and tracker.checkpoint_for_id(baseline_id) is None:
                baseline_id = None
                self._session_diff_baseline_id = None  # the selected checkpoint was evicted
            if tracker is None:
                diffs = []
                scope = None
            else:
                try:
                    # Both shell out to git, so they must stay off the UI thread.
                    diffs = await asyncio.to_thread(lambda: self._tracker_refresh(tracker, event_paths, baseline_id))
                    scope = await asyncio.to_thread(self._tracker_scope, tracker, baseline_id)
                except Exception:
                    diffs = []
                    scope = None
            if generation != self._session_diff_generation or tracker is not self._session_diff_tracker:
                return
            self._session_diff_files = diffs
            if scope is not None:
                self._session_diff_scope = scope
            self._invalidate_changes_detail()
            self._refresh_status_dashboard()
        finally:
            self._session_diff_refresh_running = False
        if self._session_diff_dirty and self._changes_inspector is not None:
            self._schedule_session_diff_refresh()

    def check_action(self, action: str, parameters: tuple[object, ...]) -> bool | None:
        if action == "open_changes":
            return self._changes_available()
        return True

    def _default_sub_agent_key(self) -> Optional[str]:
        """Most-recently-started running agent, else the most recent overall."""
        pool = self._running_sub_agents() or list(self._sub_agent_activities.values())
        if not pool:
            return None
        return max(pool, key=lambda a: a.index).agent_id

    def _default_changes_path(self) -> Optional[str]:
        """Most recent net-changed file in the live TUI session."""
        if not self._session_diff_files:
            return None
        return self._session_diff_files[-1].path

    def _close_sub_agent_inspector(self) -> None:
        screen = self._sub_agent_inspector
        if screen is None:
            return
        self._sub_agent_inspector = None
        try:
            screen.dismiss()
        except Exception:
            pass

    def _close_changes_inspector(self) -> None:
        screen = self._changes_inspector
        if screen is None:
            return
        self._changes_inspector = None
        try:
            screen.dismiss()
        except Exception:
            pass

    def on_sub_agent_entry_widget_pressed(self, message: tui_sub_agents.SubAgentEntryWidget.Pressed) -> None:
        activity = self._sub_agent_activity_for_entry(message.entry)
        if activity is not None:
            self.action_open_sub_agent(activity.agent_id)

    def on_unmount(self) -> None:
        # Stop the watchdog thread on shutdown (also keeps test apps from leaking threads).
        if self._watchdog is not None:
            self._watchdog.stop()
        # Backstop for teardown paths that never reach action_quit; this hook is
        # sync, so the socket closes when the serve task next runs.
        if self._share_server is not None:
            self._share_server.request_stop()
            self._share_server = None
        # Same backstop for the peer inbox: a dead session must not stay
        # discoverable by its siblings. The async stop runs on the loop; the
        # immediate unlink stops new connections even if the loop exits first.
        if self._inbox_socket is not None:
            try:
                asyncio.get_running_loop().create_task(self._stop_inbox_socket())
            except RuntimeError:
                pass
            try:
                self._inbox_socket.path.unlink(missing_ok=True)
            except OSError:
                pass
            self._inbox_socket = None
            self.messaging_socket_path = None
        self._unregister_from_inbox()
        self._close_memory_manager()

    def on_worker_state_changed(self, event) -> None:
        # Capture worker (e.g. agent-turn) crashes that Textual otherwise only logs to stderr.
        try:
            if event.state is WorkerState.ERROR and self._diag is not None:
                worker = event.worker
                err = getattr(worker, "error", None)
                self._diag.record(
                    "worker_error",
                    worker=getattr(worker, "name", None),
                    group=getattr(worker, "group", None),
                    error_type=type(err).__name__ if err else None,
                    traceback="".join(traceback.format_exception(type(err), err, err.__traceback__)) if err else None,
                )
        except Exception:
            pass

    async def action_quit(self) -> None:
        try:
            # Release control first, so any prompt still open resolves to its
            # default instead of leaving a turn waiting on a closing window.
            self.control_channel.release(tui_constants.TUI_CLIENT_ID)
            if self._watchdog is not None:
                self._watchdog.stop()
            # Persist any streaming tail still buffered for coalescing, so a
            # session quit mid-stream still replays what was on screen.
            await self.recording_connection_manager.flush()
            # Never leave a port listening on someone's network after they quit.
            await self._stop_share_server()
            if self.agent is not None:
                fire = getattr(self.agent, "fire_hook", None)
                if fire is not None:
                    try:
                        await fire(HookEvent.SESSION_END, {"reason": "quit"})
                    except Exception:
                        pass
                await self._save_session_history_async()
            # The session is durably saved: the post-quit resume hint may point
            # at it. If any step above raised, the flag stays False and main.py
            # prints nothing (the finally still exits).
            self._quit_cleanly = True
        finally:
            # Generation teardown and the sink drain run even when an earlier
            # step raised; the original failure still propagates.
            try:
                await self._cleanup_agent_generation()
                # After cleanup: cancelled streams settle their failures into
                # the ledger first, so the drain below captures them.
                if self._usage_sink is not None:
                    await self._usage_sink.aclose()
            finally:
                await self._stop_inbox_socket()
                self._unregister_from_inbox()
                self._close_memory_manager()
                self.exit()

    def _set_sidebar_visible(self, visible: bool) -> None:
        self.sidebar_visible = visible
        try:
            side_panel = self.query_one("#side_panel")
            side_panel.display = visible
        except Exception:
            return
        if not visible:
            self._schedule_primary_focus_restore()

    async def _set_interaction_mode(self, interaction_mode: str) -> None:
        if interaction_mode not in {tui_constants.BUILD_INTERACTION_MODE, tui_constants.PLAN_INTERACTION_MODE}:
            raise ValueError(f"Unknown interaction mode: {interaction_mode}")
        if self.interaction_mode == interaction_mode:
            return

        previous_tool_names = self._agent_tool_names()
        had_history = self.agent is not None and bool(getattr(self.agent, "history", None))

        self.interaction_mode = interaction_mode
        self._plan_decision_active = False
        await self._save_session_async()
        self._restore_plan_action_visibility()
        self._refresh_input_area_visibility()
        self._cancel_pending_question()
        self._cancel_pending_approval()
        self._cancel_pending_model_selection()
        self._cancel_pending_effort_selection()
        self._cancel_pending_theme_selection()
        self._clear_queued_messages()

        if self.config is not None:
            await self._build_agent(self.config, rebuild=True, restore_transcript=False)

        if previous_tool_names is not None and had_history:
            await self._inject_mode_switch_notice(previous_tool_names)

        self._update_mode_chrome()
        self._restore_composer_placeholder()
        self._set_chat_enabled(self.agent is not None)
        self._notify_user(messages.SWITCHED_MODE.format(mode=self.interaction_mode))

    def _agent_tool_names(self) -> Optional[set[str]]:
        """Registry names of the live agent, or ``None`` when unavailable (no agent, or a test fake)."""
        collection = getattr(self.agent, "tool_collection", None)
        registry_fn = getattr(collection, "registry", None)
        if not callable(registry_fn):
            return None
        registry: Any = registry_fn()
        return set(registry.names())

    async def _inject_mode_switch_notice(self, previous_tool_names: set[str]) -> None:
        """Tell the model its toolset changed. Only called on a real mode switch with prior history.

        The rebuilt agent carries the old conversation verbatim, so without this the model's
        history shows it using tools that no longer exist (write_plan after plan->build being
        the canonical failure).
        """
        current_tool_names = self._agent_tool_names()
        if current_tool_names is None or self.agent is None:
            return
        removed = sorted(previous_tool_names - current_tool_names)
        added = sorted(current_tool_names - previous_tool_names)
        if not removed and not added:
            return
        notice = build_mode_switch_notice(
            to_plan=self.interaction_mode == tui_constants.PLAN_INTERACTION_MODE,
            removed_tools=removed,
            added_tools=added,
        )
        await asyncio.to_thread(
            self._session_recorder.record_context_message,
            Message(role="user", content=[TextBlock(text=notice)]),
        )
        self.agent.append_user_message([TextBlock(text=notice)])
        await self._save_session_history_async()

    async def _set_permission_mode(self, permission_mode: PermissionMode | str) -> None:
        mode = normalize_permission_mode(permission_mode, default=self.permission_mode)
        if self.permission_mode == mode:
            return

        self.permission_mode = mode
        self.session.permission_mode = mode.value
        self.settings.permission_mode = mode.value
        await self._save_session_async()
        await asyncio.to_thread(self.settings_store.save, self.settings)
        # The runtime holds permission policy, so telling it is what actually
        # changes behaviour; it propagates the mode to the live agent.
        self.session_runtime.set_permission_mode(mode)
        if self.agent is not None:
            self.agent.set_permission_callback(self.session_runtime.permission_callback)
        self._update_mode_chrome()
        self._notify_user(messages.SWITCHED_PERMISSION_MODE.format(mode=mode.value))

    async def _capture_completed_plan(self) -> None:
        if self.interaction_mode != tui_constants.PLAN_INTERACTION_MODE or self.agent is None:
            return
        consume_completed_plan = getattr(self.agent, "consume_completed_plan", None)
        if not callable(consume_completed_plan):
            return

        plan = consume_completed_plan()
        if plan:
            plan_str = str(plan)
            self._latest_plan = plan_str
            self._plan_reofferable = True
            self._ensure_current_plan_artifact(plan_str)
            await self._show_plan_for_decision(plan_str, notification=messages.PLAN_CAPTURED)
            return

        if self._latest_plan and self._plan_reofferable and not self._plan_pending:
            self._ensure_current_plan_artifact(self._latest_plan)
            await self._show_plan_for_decision(self._latest_plan, notification=messages.PLAN_REOFFERED)

    async def _show_plan_for_decision(self, plan: str, *, notification: str) -> None:
        self._plan_pending = True
        self._plan_decision_active = True
        await self._save_session_async()
        self._refresh_planning_sidebar()
        self._add_conversation_entry(tui_state.ConversationEntry(kind="plan", content=plan, complete=True))
        self._set_plan_actions_visible(True, allow_discuss=True)
        self._set_composer_status(messages.PLAN_READY_PLACEHOLDER)
        self._set_chat_enabled(False)
        self._refresh_input_area_visibility()
        self._notify_user(notification)

    async def _implement_pending_plan(self, *, clear_context: bool = False) -> None:
        plan = self._latest_plan
        if not plan or not self._plan_pending or self._turn_active or self.agent_worker is not None:
            return

        # Leave self._latest_plan set so the planning sidebar keeps showing the
        # plan as a read-only reference while it is being built; clearing
        # _plan_pending is what hides the "Implement plan" action so it does not
        # reappear when the user re-enters plan mode.
        plan_artifact_path = self._ensure_current_plan_artifact(plan)
        self._plan_pending = False
        self._plan_reofferable = False
        self._plan_decision_active = False
        if clear_context:
            self._clear_agent_context(reason="implement_plan_with_cleared_context")
        await self._save_session_async()
        await self._set_interaction_mode(tui_constants.BUILD_INTERACTION_MODE)
        self._refresh_planning_sidebar()
        self._set_plan_actions_visible(False)
        self._refresh_input_area_visibility()

        prompt = build_implement_plan_prompt(
            plan,
            gigacode_enabled=self._gigacode_enabled,
            plan_artifact_path=str(plan_artifact_path) if plan_artifact_path is not None else None,
        )
        self._add_conversation_entry(tui_state.ConversationEntry(kind="user", content="Implement the approved plan."))
        self.agent_worker = self.run_worker(
            self._process_message(prompt, turn_label="Implement the approved plan."),
            name="kolega-turn",
            group="turns",
            exclusive=True,
        )

    async def _discuss_pending_plan(self) -> None:
        if not self._latest_plan:
            return

        self._plan_pending = False
        self._plan_reofferable = True
        self._plan_decision_active = False
        await self._save_session_async()
        self._refresh_planning_sidebar()
        self._set_plan_actions_visible(False)
        self._refresh_input_area_visibility()
        self._restore_composer_placeholder()
        self._set_chat_enabled(self.agent is not None)
        self._schedule_primary_focus_restore()
        self._notify_user(messages.PLAN_DISCUSSION_RESUMED)

    def _active_prompt_actions(self) -> Optional[tui_widgets.ActionList]:
        """The option list that must own keyboard focus while a prompt is shown.

        Returns None when no prompt is active (free typing). Keys off the same
        _pending_* / plan flags that gate visibility, plus a .display check, so
        "displayed" and "should be focused" cannot drift. Only one of these is ever
        active at a time; the order is a safety net.
        """
        candidates = [
            (self._pending_approval is not None, "#approval_actions"),
            (self._pending_question is not None, "#question_actions"),
            (self._pending_model_selection is not None, "#model_actions"),
            (self._pending_effort_selection is not None, "#effort_actions"),
            (self._pending_theme_selection is not None, "#theme_actions"),
            (
                self.interaction_mode == tui_constants.PLAN_INTERACTION_MODE and self._plan_pending,
                "#plan_actions",
            ),
        ]
        for active, selector in candidates:
            if not active:
                continue
            try:
                actions = self.query_one(selector, tui_widgets.ActionList)
            except Exception:
                return None
            return actions if actions.display else None
        return None

    def _restore_primary_focus(self) -> None:
        """Focus the highest-priority input target for the current TUI state.

        Prompt/action lists keep keyboard ownership while they are active. In the
        normal chat state, the composer is the primary input target, but only when
        it is enabled and the main screen is visible.
        """
        try:
            if len(self.screen_stack) != 1:
                return
        except Exception:
            return

        if self._active_prompt_actions() is not None:
            self._heal_prompt_focus()
            return

        try:
            composer = self.query_one("#composer", tui_widgets.ChatComposer)
        except Exception:
            return

        if composer.disabled or self.screen.focused is composer:
            return
        self.screen.set_focus(composer)

    def _schedule_primary_focus_restore(self) -> None:
        """Restore focus now and after refresh to beat Textual focus churn.

        The deferred restore is only a re-assertion of the focus we just set. If
        focus moves before the next refresh (for example, a user clicks or a test
        deliberately focuses the transcript), do not yank it back to the composer.
        """
        self._restore_primary_focus()
        try:
            scheduled_focus = self.screen.focused
        except Exception:
            scheduled_focus = None

        def restore_if_unchanged() -> None:
            try:
                current_focus = self.screen.focused
            except Exception:
                return
            if current_focus is None or current_focus is scheduled_focus:
                self._restore_primary_focus()

        try:
            self.call_after_refresh(restore_if_unchanged)
        except Exception:
            pass

    def _focus_active_prompt(self) -> None:
        """Focus the active prompt list now and re-assert after the refresh settles.

        The synchronous set_focus handles the common fast path; the deferred
        re-assert defeats the documented race where compose/resume/disable churn
        resets focus right after we set it (see tui_widgets.PromptPanel.prompt)."""
        actions = self._active_prompt_actions()
        if actions is None:
            return
        if self.screen.focused is not actions:
            self.screen.set_focus(actions)
        self.call_after_refresh(self._heal_prompt_focus)

    def _focus_active_prompt_from_composer(self) -> bool:
        """Move keyboard focus from the composer back to the active option list.

        Planning questions intentionally allow composer focus for custom answers.
        When the user wants to return to the visible options, ChatComposer calls
        this helper from its top-line Up key handling. Returns True only when the
        handoff happened, so the composer can otherwise keep normal cursor motion.
        """
        actions = self._active_prompt_actions()
        if actions is None:
            return False

        try:
            composer = self.query_one("#composer", tui_widgets.ChatComposer)
        except Exception:
            return False

        if composer.disabled or self.screen.focused is not composer:
            return False

        if actions.option_count:
            actions.highlighted = actions.option_count - 1
        self.screen.set_focus(actions)
        return True

    def _focus_composer_from_active_prompt(self, actions: tui_widgets.ActionList) -> bool:
        """Move keyboard focus from the bottom of an active option list to composer."""
        active_actions = self._active_prompt_actions()
        if active_actions is None or active_actions is not actions:
            return False

        try:
            composer = self.query_one("#composer", tui_widgets.ChatComposer)
        except Exception:
            return False

        if composer.disabled or self.screen.focused is not actions:
            return False

        self.screen.set_focus(composer)
        return True

    def _heal_prompt_focus(self) -> None:
        """Re-grab focus for the active prompt list if it has drifted. Idempotent.

        Restores keyboard navigation after focus is lost to nothing (background
        click), to the conversation transcript (AUTO_FOCUS on resume/resize), or to
        any other stray widget. No-op when no prompt is active or the list is already
        focused."""
        actions = self._active_prompt_actions()
        if actions is None or self.screen.focused is actions:
            return
        # Legitimate exception: during a QUESTION the composer is enabled so the user
        # can type a free-form answer; don't fight a deliberate move there. For
        # approvals/plan the composer is disabled and thus never focusable here.
        focused = self.screen.focused
        if (
            self._pending_question is not None
            and isinstance(focused, tui_widgets.ChatComposer)
            and not focused.disabled
        ):
            return
        self.screen.set_focus(actions)

    def _set_plan_actions_visible(self, visible: bool, *, allow_discuss: bool = False) -> None:
        try:
            plan_actions = self.query_one("#plan_actions", tui_widgets.ActionList)
            if visible:
                options = [Option("Implement plan", id="implement_plan")]
                if allow_discuss:
                    options.append(Option("Clear context and implement plan", id="implement_plan_clear"))
                    options.append(Option("Discuss further", id="discuss_plan"))
                plan_actions.show_options(options)
                self._focus_active_prompt()
            else:
                plan_actions.hide()
        except Exception:
            return

    def _set_effort_actions_visible(self, visible: bool) -> None:
        try:
            effort_actions = self.query_one("#effort_actions", tui_widgets.ActionList)
            if visible and self._pending_effort_selection is not None:
                effort_actions.show_options(
                    [
                        Option(
                            self._effort_option_label(index, label, value),
                            id=f"{tui_constants.EFFORT_OPTION_ID_PREFIX}{index}",
                        )
                        for index, (label, value) in enumerate(self._pending_effort_selection.options)
                    ]
                )
            else:
                effort_actions.hide()
        except Exception:
            return

    def _set_model_actions_visible(self, visible: bool) -> None:
        try:
            model_actions = self.query_one("#model_actions", tui_widgets.ActionList)
            if visible and self._pending_model_selection is not None:
                model_actions.show_options(
                    [
                        Option(
                            self._model_option_label(
                                index,
                                label,
                                value,
                                self._pending_model_selection.provider,
                            ),
                            id=f"{tui_constants.MODEL_OPTION_ID_PREFIX}{index}",
                        )
                        for index, (label, value) in enumerate(self._pending_model_selection.options)
                    ]
                )
            else:
                model_actions.hide()
        except Exception:
            return

    def _set_theme_actions_visible(self, visible: bool) -> None:
        try:
            theme_actions = self.query_one("#theme_actions", tui_widgets.ActionList)
            if visible and self._pending_theme_selection is not None:
                theme_actions.show_options(
                    [
                        Option(
                            self._theme_option_label(index, name),
                            id=f"{tui_constants.THEME_OPTION_ID_PREFIX}{index}",
                        )
                        for index, (name, _value) in enumerate(self._pending_theme_selection.options)
                    ]
                )
            else:
                theme_actions.hide()
        except Exception:
            return

    def _meta_content(self) -> str:
        gigacode = "on" if self._gigacode_enabled else "off"
        return (
            f"{self.active_project_path} | session {self.session.session_id} | "
            f"agent {self.mode} | {self.interaction_mode} | permissions {self.permission_mode.value} | "
            f"gigacode {gigacode}"
        )

    def _update_mode_chrome(self) -> None:
        try:
            self.query_one("#session_meta", Static).update(self._meta_content())
        except Exception:
            pass
        self._refresh_status_dashboard()
        self._refresh_planning_sidebar()
        self._ensure_startup_entry()

    def _refresh_planning_sidebar(self) -> None:
        plan_content = self._latest_plan or messages.PLAN_EMPTY_MESSAGE
        task_list_content = self.session.task_list_markdown or messages.TASK_LIST_EMPTY_MESSAGE
        try:
            plan_markdown = self.query_one("#planning_plan_markdown", tui_widgets.PlanningMarkdown)
            task_list_markdown = self.query_one("#status_task_list_markdown", tui_widgets.PlanningMarkdown)
            plan_markdown.update(plan_content)
            task_list_markdown.update(task_list_content)
            plan_markdown.set_class(plan_content == messages.PLAN_EMPTY_MESSAGE, "empty-state")
            task_list_markdown.set_class(task_list_content == messages.TASK_LIST_EMPTY_MESSAGE, "empty-state")
        except Exception:
            pass

    def _refresh_input_area_visibility(self) -> None:
        prompt_or_decision_pending = (
            self._pending_approval is not None or self._pending_question is not None or self._plan_decision_active
        )
        try:
            composer = self.query_one("#composer", tui_widgets.ChatComposer)
            composer.display = True
        except Exception:
            pass
        try:
            queued_panel = self.query_one("#queued_messages")
        except Exception:
            return
        if prompt_or_decision_pending:
            queued_panel.display = False
        else:
            self._refresh_queued_messages_panel()

    def _set_chat_enabled(self, enabled: bool) -> None:
        try:
            composer = self.query_one("#composer", tui_widgets.ChatComposer)
        except Exception:
            # The composer may be unmounted while a turn finalizes (e.g. app
            # teardown); nothing to update then. Guarded like the sibling
            # composer helpers so a late finalize can't raise WorkerFailed.
            return
        composer.disabled = not enabled or self._plan_decision_active or self._pending_approval is not None
        if self.config is None and self.agent is None:
            composer.placeholder = messages.DISCONNECTED_COMPOSER_PLACEHOLDER
        elif enabled and composer.placeholder == messages.DISCONNECTED_COMPOSER_PLACEHOLDER:
            composer.placeholder = messages.COMPOSER_PLACEHOLDER

    def _set_composer_status(self, status: str) -> None:
        try:
            composer = self.query_one("#composer", tui_widgets.ChatComposer)
        except Exception:
            return
        composer.placeholder = status

    def _restore_composer_placeholder(self) -> None:
        try:
            composer = self.query_one("#composer", tui_widgets.ChatComposer)
        except Exception:
            return
        composer.placeholder = messages.COMPOSER_PLACEHOLDER
        self._clear_composer_hint()

    def _show_composer_hint(self, text: str, tone: str = "warning") -> None:
        try:
            hint = self.query_one("#composer_hint", Static)
            row = self.query_one("#composer_hint_row", Horizontal)
        except Exception:
            return
        hint.set_class(tone == "warning", "hint-warning")
        hint.set_class(tone != "warning", "hint-info")
        hint.update(text)
        row.display = bool(text)
        self._update_detach_button()

    def _clear_composer_hint(self) -> None:
        try:
            row = self.query_one("#composer_hint_row", Horizontal)
            hint = self.query_one("#composer_hint", Static)
        except Exception:
            return
        hint.update("")
        row.display = False

    def _update_detach_button(self) -> None:
        """Show the detach × button only when there are pending image attachments."""
        try:
            btn = self.query_one("#detach_btn", Button)
        except Exception:
            return
        btn.display = bool(self._pending_image_attachments)

    def _clear_agent_context(self, *, reason: str = "context_cleared") -> None:
        """Wipe the agent's LLM history so the build agent starts fresh, while leaving
        the visible transcript and the captured plan intact."""
        self._session_recorder.start_epoch(reason)
        if self.agent is not None:
            self.agent.history = MessageHistory()
            self.agent.last_compression_index = None
            # The wiped history took the injected reminders with it; the next turn must
            # re-send memory, guidance, the plan handle, and the task list.
            self.agent.reset_volatile_context()
        self.session.history = []
        self.session.compaction = {}
        self._clear_queued_messages()

    async def _reset_current_thread(self, *, preserve_loop: bool = False) -> None:
        """Clear the conversation thread.

        ``preserve_loop`` keeps the active scheduled loop and its session state
        so a ``/loop --fresh`` iteration can start from a clean context without
        cancelling the loop that asked for it.
        """
        self._close_sub_agent_inspector()
        await asyncio.to_thread(self._session_recorder.start_epoch, "thread_reset")
        if self.agent is not None:
            self.agent.history = MessageHistory()
            # The thread wipe took the injected reminders with it; the next turn re-sends
            # the current sections. The plan and task list are cleared below, so those
            # come back absent — memory and guidance are re-sent.
            self.agent.reset_volatile_context()
        self.session.history = []
        self.session.compaction = {}
        self.session.task_list_markdown = ""
        self.conversation_entries = []
        # Per-session file-edit log used by the diff view. It is appended to on every
        # file-edit preview and otherwise never cleared, so reset it on thread reset to
        # stop it growing for the life of the process.
        self._session_file_changes = []
        self._session_diff_dirty = True
        self._stream_entries = {}
        self._tool_entries = {}
        self._tool_stream_buffers = {}
        self._sub_agent_activities = {}
        self._sub_agent_by_tool_call = {}
        self._sub_agent_seq = 0
        self._workflow_activities = {}
        self._active_progress_entry = None
        self._clear_queued_messages()
        self._latest_plan = None
        self._plan_pending = False
        self._plan_reofferable = False
        self._plan_decision_active = False
        self._goal = None
        self.session.goal = {}
        if self.agent is not None:
            self.agent.apply_goal(None)
        if preserve_loop:
            # A --fresh iteration cleared the thread on the loop's behalf; keep
            # the loop and re-apply its prompt extension, which the agent still
            # carries but which _build_agent-free resets would otherwise drop.
            self._sync_loop_to_session()
        else:
            self._scheduled_loop = None
            self.session.loop = {}
            self._loop_iteration_active = False
            if self.agent is not None:
                self.agent.apply_loop(False)
        self._clear_runtime_output()
        await self._save_session_async()
        self._set_plan_actions_visible(False)
        self._cancel_pending_question()
        self._cancel_pending_approval()
        self._cancel_pending_model_selection()
        self._cancel_pending_effort_selection()
        self._cancel_pending_theme_selection()
        self._refresh_planning_sidebar()
        self._clear_turn_status_strip()
        self._turn_active = False
        self._restore_composer_placeholder()
        self._set_chat_enabled(self.agent is not None)
        self._ensure_startup_entry(render=False)
        self._add_conversation_entry(
            tui_state.ConversationEntry(kind="progress", content=messages.THREAD_RESET_MESSAGE, complete=True)
        )

    def _add_conversation_entry(self, entry: tui_state.ConversationEntry) -> None:
        self.conversation_entries.append(entry)
        if entry.uuid:
            self._stream_entries[entry.uuid] = entry
        if entry.tool_call_id:
            self._tool_entries[entry.tool_call_id] = entry
        self._invalidate_conversation(entry)

    def _ensure_startup_entry(self, *, render: bool = True) -> None:
        existing = next((entry for entry in self.conversation_entries if entry.kind == "startup"), None)
        if existing is None:
            self.conversation_entries.insert(
                0, tui_state.ConversationEntry(kind="startup", content=self._startup_content())
            )
        elif self.conversation_entries[0] is existing:
            existing.content = self._startup_content()
            if render:
                self._invalidate_conversation(existing)
            return
        else:
            existing.content = self._startup_content()
            self.conversation_entries.remove(existing)
            self.conversation_entries.insert(0, existing)
        if render:
            self._render_conversation()

    def _startup_prompt_override_lines(self) -> list[str]:
        lines: list[str] = []
        existing = list_prompt_overrides(self.project_path).existing
        if existing:
            filenames = ", ".join(item.path.name for item in existing)
            lines.append(f"Prompt overrides: {filenames}")
        errors = list(getattr(self.agent, "prompt_override_errors", []) or [])
        if errors:
            lines.append("Prompt override errors:")
            lines.extend(f"- {error}" for error in errors)
        return lines

    def _startup_custom_agent_lines(self) -> list[str]:
        lines: list[str] = []
        if self.custom_agent_catalog.has_agents():
            lines.append(f"Custom agents: {', '.join(self.custom_agent_catalog.names())}")
        if self.custom_agent_catalog.diagnostics:
            lines.append("Custom agent diagnostics:")
            lines.extend(f"- {diagnostic.format()}" for diagnostic in self.custom_agent_catalog.diagnostics)
        return lines

    def _startup_content(self) -> str:
        session_id = str(self.session.session_id)[:8]
        provider, model = self._startup_model()
        model_display = f"{provider}/{model}" if model else provider
        effort = self._startup_thinking_effort() or "not supported"
        api_key = (
            key_status(provider, self.project_path, self.settings)
            if model
            else "not checked until a model is configured"
        )
        override_message = (
            active_model_override_message(self.config, self.project_path, self.overrides, self.settings)
            if self.config is not None
            else None
        )
        startup_lines = [*tui_constants.STARTUP_WORDMARK, ""]
        if self.config is None:
            startup_lines.extend(
                [
                    messages.DISCONNECTED_HEADLINE,
                    "",
                    messages.DISCONNECTED_STARTUP_GUIDANCE,
                    messages.DISCONNECTED_SIDEBAR_GUIDANCE,
                    "",
                ]
            )
        bullet = theme.g(Glyph.BULLET_SEP)
        project_lines = [f"Project: {self.active_project_path}"]
        if self.active_project_path != self.project_path:
            project_lines.append(f"Launch project: {self.project_path}")
        startup_lines.extend(
            [
                *project_lines,
                f"Session: {session_id}",
                f"Mode: {self.mode}",
                f"Interaction: {self.interaction_mode}",
                f"Permissions: {self.permission_mode.value}",
                f"Gigacode: {'on' if self._gigacode_enabled else 'off'}",
                f"Model: {model_display}",
                *([override_message] if override_message else []),
                *self._startup_prompt_override_lines(),
                *self._startup_custom_agent_lines(),
                f"Thinking effort: {effort}",
                f"API key: {api_key}",
                *self._startup_lsp_lines(),
                (
                    f"Enter send {bullet} Shift+Enter/Ctrl+J newline {bullet} "
                    f"Shift+Tab or /plan /build {bullet} Ctrl+P permissions {bullet} Ctrl+O sidebar"
                ),
                (f"Alt+V or /attach image {bullet} Ctrl+C stop turn {bullet} Cmd+C copy selection {bullet} / commands"),
            ]
        )
        if self._running_under_tmux_or_screen():
            startup_lines.extend(["", messages.TMUX_SHORTCUT_HINT])
        return "\n".join(startup_lines)

    @staticmethod
    def _running_under_tmux_or_screen() -> bool:
        """True when the process is nested under tmux or GNU screen.

        Shift-modified keys often never reach the TUI in those multiplexers
        unless extended-keys / CSI-u is configured. Used only for a one-time
        startup hint with portable fallbacks.
        """
        if os.environ.get("TMUX"):
            return True
        term = os.environ.get("TERM", "").strip().lower()
        return term.startswith("screen") or term.startswith("tmux")

    def _startup_lsp_lines(self) -> list[str]:
        """Plain-text LSP status lines for the startup block (above command summary)."""
        agent = self.agent
        if agent is None or agent.tool_collection is None:
            return [""]
        manager = agent.tool_collection.lsp_manager
        if manager is None or not manager.enabled:
            return [""]
        report = manager.report
        if report is None or not report.detected:
            return [""]

        lines = ["", "LSP:"]
        for d in report.detected:
            rl = next((r for r in report.resolved if r.language_id == d.language_id), None)
            if rl:
                lines.append(f"  {d.display_name} \u2192 {rl.server_name}")
            else:
                missing_rl = next((m for m in report.missing if m.language_id == d.language_id), None)
                if missing_rl:
                    install = missing_rl.install_commands[0] if missing_rl.install_commands else "see docs"
                    lines.append(f"  {d.display_name} \u2192 {missing_rl.server_name} (install: {install})")
                    if missing_rl.alternatives:
                        lines.append(f"     Alternatives: {', '.join(missing_rl.alternatives)}")
        lines.append("")  # blank line before command summary
        return lines

    def _startup_model(self) -> tuple[str, str]:
        if self.config is not None:
            return self.config.long_context_config.provider.value, self.config.long_context_config.model

        if self.settings.active_provider and self.settings.active_model:
            return self.settings.active_provider, self.settings.active_model

        return "not configured", ""

    def _startup_thinking_effort(self) -> Optional[str]:
        if self.config is not None:
            return self.config.long_context_config.thinking_effort

        provider, model = self._startup_model()
        if (
            self.settings.active_provider == provider
            and self.settings.active_model == model
            and self.settings.active_thinking_effort
        ):
            return self.settings.active_thinking_effort
        return default_ui_thinking_effort(provider, model)
