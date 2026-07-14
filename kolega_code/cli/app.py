"""Textual application for Kolega Code."""

from __future__ import annotations

import asyncio
import os
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Optional

from rich.console import Group
from rich.text import Text
from textual import events
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
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

from kolega_code.agent import AgentConfig
from kolega_code.agent.prompt_dump import list_prompt_overrides
from kolega_code.agent.prompt_provider import AgentMode
from kolega_code.agent.prompts import (
    build_implement_plan_prompt,
)
from kolega_code.hooks import HookDispatcher, HookEvent
from kolega_code.llm.models import MessageHistory
from kolega_code.mcp.config import load_mcp_config, mcp_secret_values
from kolega_code.mcp.state import MCPOAuthTokenStore
from kolega_code.permissions import (
    PermissionMode,
    normalize_permission_mode,
)

from . import messages, theme
from .config import CliConfigOverrides, active_model_override_message, key_status
from .connection import CliConnectionManager
from .goal import GoalState
from .diagnostics import DiagnosticsLog, ResponsivenessWatchdog
from .file_index import WorkspaceFileIndex
from .mentions import build_file_attachments
from .provider_registry import default_ui_thinking_effort
from .session_store import SessionRecord, SessionStore
from .settings import CliSettings, SettingsStore
from kolega_code.agent.custom_agents import CustomAgentCatalog, discover_custom_agents
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
from .tui import prompt_flows as tui_prompt_flows
from .tui import onboarding_screen as tui_onboarding
from .tui import settings_panel as tui_settings_panel
from .tui import settings_screen as tui_settings_screen
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
TERMINAL_FLUSH_INTERVAL = 0.04
SESSION_DIFF_REFRESH_INTERVAL = 1.0
TERMINAL_IMMEDIATE_FLUSH_CHARS = 64 * 1024
LOG_FLUSH_INTERVAL = 0.05
LOG_IMMEDIATE_FLUSH_ITEMS = 100


class KolegaCodeApp(
    tui_settings_panel.SettingsPanelMixin,
    tui_command_handlers.CommandHandlersMixin,
    tui_agent_runtime.AgentRuntimeMixin,
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
    ) -> None:
        super().__init__()
        self.project_path = project_path
        self.config = config
        self.mode = CLI_AGENT_MODE
        self.store = store
        self.session = session
        self.session.mode = CLI_AGENT_MODE
        self.interaction_mode = self._validated_interaction_mode(self.session.interaction_mode)
        self.session.interaction_mode = self.interaction_mode
        self.permission_mode = normalize_permission_mode(
            permission_mode or self.session.permission_mode,
            default=PermissionMode.ASK,
        )
        self.session.permission_mode = self.permission_mode.value
        self.settings_store = settings_store or SettingsStore(store.root)
        self.overrides = overrides or CliConfigOverrides()
        self.settings: CliSettings = CliSettings()
        self.skill_catalog: SkillCatalog = discover_skills(self.project_path)
        self.custom_agent_catalog: CustomAgentCatalog = discover_custom_agents(
            self.project_path,
            self.settings_store.root,
        )
        self.file_index = WorkspaceFileIndex(self.project_path)
        self._file_index_refreshing = False
        # Local diagnostics (constructed on mount, once settings/secrets are loaded).
        self._diag: Optional[DiagnosticsLog] = None
        self._watchdog: Optional[ResponsivenessWatchdog] = None
        self.browser_visible = browser_visible
        self.sidebar_visible = True
        self.check_for_updates = check_for_updates
        self.show_logs = show_logs
        self.startup_config_error = startup_config_error
        self.connection_manager = CliConnectionManager()
        self._hook_dispatcher: Optional[HookDispatcher] = None
        self._session_started = False
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
        self._session_diff_tracker: Optional[tui_session_diff.GitSessionDiffTracker] = None
        self._session_diff_files: list[tui_session_diff.SessionDiffFile] = []
        self._session_diff_dirty = False
        self._session_diff_refresh_running = False
        self._session_diff_timer: Optional[Timer] = None
        self._workflow_activities: dict[str, tui_state.WorkflowActivity] = {}
        self._render_pending = False
        self._conversation_anchor_pending = False
        self._entry_widgets: dict[str, tui_widgets.ConversationEntryWidget | tui_widgets.ToolEntryWidget] = {}
        self._dirty_entry_ids: set[str] = set()
        self._active_progress_entry: Optional[tui_state.ConversationEntry] = None
        self._turn_active = False
        self._latest_plan: Optional[str] = self.session.latest_plan_markdown or None
        self._plan_pending: bool = bool(self._latest_plan and self.session.plan_pending)
        self._plan_reofferable: bool = bool(self._latest_plan and (self.session.plan_reofferable or self._plan_pending))
        self._plan_decision_active = False
        self._gigacode_enabled = bool(self.session.gigacode_enabled)
        self._goal: Optional[GoalState] = GoalState.from_dict(self.session.goal) if self.session.goal else None
        self._pending_question: Optional[tui_state.PendingQuestion] = None
        self._pending_approval: Optional[tui_state.PendingApproval] = None
        self._pending_image_attachments: list[dict] = []
        self._queued_messages: list[tui_state.QueuedMessage] = []
        self._queued_message_seq = 0
        # Dedup flag: one vision-mismatch system message per non-vision model
        # session. Reset in _switch_model so a new model gets a fresh warning.
        self._vision_warning_shown = False
        self._settings_screen: Optional[tui_settings_screen.SettingsScreen] = None
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

    async def on_mount(self) -> None:
        self.settings = self.settings_store.load()
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
            self.set_interval(1.0, self._watchdog.beat)
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
        if self.config is not None:
            await self._build_agent(self.config)
            self._set_chat_enabled(True)
            self._schedule_primary_focus_restore()
        else:
            await self._ensure_agent_from_settings()
        if self.config is None and not self._onboarding_skipped:
            self.call_after_refresh(self.action_open_onboarding)

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
                TERMINAL_FLUSH_INTERVAL,
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
                LOG_FLUSH_INTERVAL,
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
            goal=dict(record.goal),
        )

    async def _save_session_async(self) -> None:
        """Persist lightweight session state without blocking Textual's loop."""
        async with self._persistence_lock:
            snapshot = self._session_snapshot_locked()
            await asyncio.to_thread(self.store.save, snapshot)
            self.session.updated_at = snapshot.updated_at

    async def _save_session_history_async(self) -> None:
        """Dump agent history and persist the session off the UI loop."""
        async with self._persistence_lock:
            agent = self.agent
            if agent is None:
                return
            snapshot = self._session_snapshot_locked()

            def dump_and_save() -> tuple[list[dict], dict, str]:
                history = agent.dump_message_history()
                compaction = agent.dump_compaction_state()
                snapshot.history = history
                snapshot.compaction = compaction
                self.store.save(snapshot)
                return history, compaction, snapshot.updated_at

            history, compaction, updated_at = await asyncio.to_thread(dump_and_save)
            self.session.history = history
            self.session.compaction = compaction
            self.session.updated_at = updated_at

    def _restore_plan_action_visibility(self) -> None:
        self._set_plan_actions_visible(
            self.interaction_mode == tui_constants.PLAN_INTERACTION_MODE and self._plan_pending,
            allow_discuss=self._plan_decision_active,
        )

    async def on_chat_composer_submitted(self, event: tui_widgets.ChatComposer.Submitted) -> None:
        text = event.value
        stripped_text = text.strip()
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
            commands = search_commands(slash[0], self.skill_catalog, limit=8)
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
            attachments, unresolved = build_file_attachments(text, self.project_path)
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
                self.action_open_onboarding()
            else:
                self.action_open_settings()
            return
        if event.button.id == "save_settings":
            await self._save_settings_from_ui()
            return
        if event.button.id == "settings_chatgpt_login":
            await self._settings_login_chatgpt()
            return
        if event.button.id == "settings_chatgpt_logout":
            self._settings_logout_chatgpt()
            return
        if event.button.id == "settings_remove_api_key":
            self._settings_remove_api_key()
            return
        if event.button.id == "settings_test_connection":
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

    def action_open_settings(self, category: str = "model") -> None:
        """Open the full-screen settings editor."""
        if self._settings_screen is not None or self._onboarding_screen is not None:
            return
        if self._pending_approval is not None or self._pending_question is not None or self._plan_decision_active:
            self._notify_user("Resolve the active prompt before opening Settings.", severity="warning")
            return
        screen = tui_settings_screen.SettingsScreen(self, category=category)
        self._settings_screen = screen
        self.push_screen(screen)

    def action_open_onboarding(self) -> None:
        """Open the independent connection wizard."""
        if self.config is not None or self._onboarding_screen is not None or self._settings_screen is not None:
            return
        screen = tui_onboarding.OnboardingScreen(self)
        self._onboarding_screen = screen
        self.push_screen(screen)

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
        self.push_screen(screen)

    def action_open_changes(self, path: Optional[str] = None) -> None:
        """Open the full-screen git session changes inspector."""
        if not self._changes_available() or self._changes_inspector is not None:
            return
        paths = {change.path for change in self._session_diff_files}
        if path is None or path not in paths:
            path = self._default_changes_path() or ""
        screen = tui_changes.ChangesInspectorScreen(self, path)
        self._changes_inspector = screen
        self.push_screen(screen)
        self._start_session_diff_refresh()

    def _initialize_session_diff_tracker(self) -> None:
        """Set up git-only net diff tracking for the Changes inspector."""
        self._session_diff_tracker = tui_session_diff.GitSessionDiffTracker.create(self.project_path)
        self._session_diff_files = []
        if self._session_diff_tracker is None:
            return
        try:
            self._session_diff_tracker.capture_baseline()
        except Exception:
            self._session_diff_tracker = None
            self._session_diff_files = []

    def _changes_available(self) -> bool:
        return self._session_diff_tracker is not None

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
        if tracker is None or self._session_diff_refresh_running:
            return
        try:
            self._session_diff_refresh_running = True
            self.run_worker(self._session_diff_refresh_worker(), name="kolega-session-diff", group="session-diff")
        except Exception:
            self._session_diff_refresh_running = False

    async def _session_diff_refresh_worker(self) -> None:
        try:
            self._session_diff_dirty = False
            tracker = self._session_diff_tracker
            event_paths = [change.path for change in self._session_file_changes]
            if tracker is None:
                diffs = []
            else:
                try:
                    diffs = await asyncio.to_thread(tracker.refresh, event_paths)
                except Exception:
                    diffs = []
            self._session_diff_files = diffs
            self._invalidate_changes_detail()
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
        if self._watchdog is not None:
            self._watchdog.stop()
        if self.agent is not None:
            fire = getattr(self.agent, "fire_hook", None)
            if fire is not None:
                try:
                    await fire(HookEvent.SESSION_END, {"reason": "quit"})
                except Exception:
                    pass
            await self._save_session_history_async()
            await self.agent.cleanup()
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

        self._update_mode_chrome()
        self._restore_composer_placeholder()
        self._set_chat_enabled(self.agent is not None)
        self._notify_user(messages.SWITCHED_MODE.format(mode=self.interaction_mode))

    async def _set_permission_mode(self, permission_mode: PermissionMode | str) -> None:
        mode = normalize_permission_mode(permission_mode, default=self.permission_mode)
        if self.permission_mode == mode:
            return

        self.permission_mode = mode
        self.session.permission_mode = mode.value
        self.settings.permission_mode = mode.value
        await self._save_session_async()
        await asyncio.to_thread(self.settings_store.save, self.settings)
        if self.agent is not None:
            self.agent.set_permission_mode(mode)
            self.agent.set_permission_callback(self._permission_callback)
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
            self._clear_agent_context()
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
            self._process_message(prompt), name="kolega-turn", group="turns", exclusive=True
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
            f"{self.project_path} | session {self.session.session_id} | "
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
        composer = self.query_one("#composer", tui_widgets.ChatComposer)
        composer.disabled = not enabled or self._plan_decision_active or self._pending_approval is not None
        if self.config is None and self.agent is None:
            composer.placeholder = messages.DISCONNECTED_COMPOSER_PLACEHOLDER
        elif enabled and composer.placeholder == messages.DISCONNECTED_COMPOSER_PLACEHOLDER:
            composer.placeholder = messages.COMPOSER_PLACEHOLDER

    def _set_composer_status(self, status: str) -> None:
        self.query_one("#composer", tui_widgets.ChatComposer).placeholder = status

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

    def _clear_agent_context(self) -> None:
        """Wipe the agent's LLM history so the build agent starts fresh, while leaving
        the visible transcript and the captured plan intact."""
        if self.agent is not None:
            self.agent.history = MessageHistory()
            self.agent.last_compression_index = None
        self.session.history = []
        self.session.compaction = {}
        self._clear_queued_messages()

    async def _reset_current_thread(self) -> None:
        self._close_sub_agent_inspector()
        if self.agent is not None:
            self.agent.history = MessageHistory()
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
        startup_lines.extend(
            [
                f"Project: {self.project_path}",
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
