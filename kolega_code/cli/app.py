"""Textual application for Kolega Code."""

from __future__ import annotations

import asyncio
import subprocess
import sys
import time
from pathlib import Path
from typing import Awaitable, Callable, Optional

from rich.markup import escape
from rich.text import Text
from textual import events
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.css.query import NoMatches
from textual.timer import Timer
from textual.widgets import (
    Button,
    Collapsible,
    Footer,
    Input,
    Label,
    Markdown,
    OptionList,
    RichLog,
    Select,
    Static,
    TabbedContent,
    TabPane,
    TextArea,
)
from textual.widgets.option_list import Option

from kolega_code.agent import AgentConfig, AgentEvent, CoderAgent, PlanningAgent, PromptExtension, ToolExtension
from kolega_code.auth import constants as chatgpt_constants
from kolega_code.auth.chatgpt_oauth import run_login_flow
from kolega_code.agent.prompt_provider import AgentMode
from kolega_code.agent.prompts import (
    build_implement_plan_prompt,
    build_init_agents_prompt,
)
from kolega_code.agent.tool_backend.search_backends import (
    DEFAULT_BACKEND as DEFAULT_WEB_SEARCH_BACKEND,
    SearchBackendError,
    available_backends,
    get_backend_class,
)
from kolega_code.hooks import HookDispatcher, HookEvent, load_hook_config, project_hooks_present
from kolega_code.llm.exceptions import LLMError, llm_error_message
from kolega_code.llm.models import MessageHistory, TextBlock
from kolega_code.permissions import (
    PermissionMode,
    normalize_permission_mode,
)
from kolega_code.services.browser import PlaywrightBrowserManager

from . import messages, theme
from .config import CliConfigError, CliConfigOverrides, build_agent_config, config_summary, key_status
from .connection import CliConnectionManager
from .file_index import WorkspaceFileIndex
from .mentions import build_file_attachments
from .provider_registry import (
    INHERIT_SENTINEL,
    UI_DEFAULT_MODEL,
    UI_DEFAULT_PROVIDER,
    agent_role_options,
    agent_role_provider_options,
    default_ui_thinking_effort,
    get_ui_model,
    ui_model_options,
    ui_provider_options,
    ui_thinking_effort_options,
)
from .session_store import SessionRecord, SessionStore
from .settings import CliSettings, SettingsStore, WEB_SEARCH_KEY_NAMES
from .skills import (
    SkillCatalog,
    activated_skill_names,
    build_skill_prompt_extension,
    build_skill_tool_extension,
    discover_skills,
)
from .slash_commands import (
    SKILLS_LIST_COMMAND,
    THREAD_RESET_COMMANDS,
    TUI_COMMAND_NAMES,
    SlashCommandEntry,
    agent_command_names,
    search_commands,
)
from .theme import Color, Glyph
from .updater import check_for_update, run_self_update, update_status_message
from .tui import constants as tui_constants
from .tui import prompt_flows as tui_prompt_flows
from .tui import state as tui_state
from .tui import sub_agent_screen as tui_sub_agents
from .tui import transcript as tui_transcript
from .tui import widgets as tui_widgets
from .tui.styles import APP_CSS

CLI_AGENT_MODE = AgentMode.CLI.value


class KolegaCodeApp(tui_prompt_flows.PromptFlowMixin, tui_transcript.TranscriptRenderingMixin, App):
    """Interactive terminal UI for Kolega Code."""

    CSS = APP_CSS


    BINDINGS = [
        Binding(
            "shift+tab", "toggle_interaction_mode", "Plan/Build", show=True, key_display="Shift+Tab", priority=True
        ),
        Binding("ctrl+p", "toggle_permission_mode", "Permissions", show=True, key_display="Ctrl+P", priority=True),
        Binding("ctrl+o", "toggle_sidebar", "Sidebar", show=True, key_display="Ctrl+O", priority=True),
        Binding("ctrl+g", "open_sub_agent", "Agents", show=True, key_display="Ctrl+G", priority=True),
        Binding("ctrl+c", "cancel_generation", "Cancel", show=True),
        Binding("escape", "cancel_generation", "Cancel", show=False),
        Binding("ctrl+q", "quit", "Quit", show=True),
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
        self.file_index = WorkspaceFileIndex(self.project_path)
        self.browser_visible = browser_visible
        self.sidebar_visible = True
        self.check_for_updates = check_for_updates
        self.connection_manager = CliConnectionManager()
        self._hook_dispatcher: Optional[HookDispatcher] = None
        self._session_started = False
        self.agent: Optional[CoderAgent | PlanningAgent] = None
        self.agent_worker = None
        self.conversation_entries: list[tui_state.ConversationEntry] = []
        self._stream_entries: dict[str, tui_state.ConversationEntry] = {}
        self._tool_entries: dict[str, tui_state.ConversationEntry] = {}
        self._tool_stream_buffers: dict[str, str] = {}
        self._sub_agent_activities: dict[str, tui_state.SubAgentActivity] = {}
        self._sub_agent_by_tool_call: dict[str, str] = {}
        self._sub_agent_seq = 0
        self._workflow_activities: dict[str, tui_state.WorkflowActivity] = {}
        self._render_pending = False
        self._entry_widgets: dict[str, tui_widgets.ConversationEntryWidget | tui_widgets.ToolEntryWidget] = {}
        self._dirty_entry_ids: set[str] = set()
        self._active_progress_entry: Optional[tui_state.ConversationEntry] = None
        self._turn_active = False
        self._latest_plan: Optional[str] = self.session.latest_plan_markdown or None
        self._plan_pending: bool = bool(self._latest_plan and self.session.plan_pending)
        self._plan_reofferable: bool = bool(
            self._latest_plan and (self.session.plan_reofferable or self._plan_pending)
        )
        self._plan_decision_active = False
        self._gigacode_enabled = False
        self._pending_question: Optional[tui_state.PendingQuestion] = None
        self._pending_approval: Optional[tui_state.PendingApproval] = None
        self._pending_image_attachments: list[dict] = []
        # Dedup flag: one vision-mismatch system message per non-vision model
        # session. Reset in _switch_model so a new model gets a fresh warning.
        self._vision_warning_shown = False
        self._permission_lock = asyncio.Lock()
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
        self._terminal_has_content = False

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
                with Horizontal(id="composer_hint_row"):
                    yield Static("", id="composer_hint", markup=False)
                    yield Button(
                        theme.g(Glyph.CROSS), id="detach_btn", classes="hint-detach"
                    )
                yield tui_widgets.CompletionDropdown(id="completion_dropdown")
                yield tui_widgets.ChatComposer(placeholder=messages.COMPOSER_PLACEHOLDER, id="composer")
            with Vertical(id="side_panel"):
                with TabbedContent(id="events"):
                    with TabPane("Status", id="status_pane"):
                        with Vertical(id="status_container"):
                            yield Static("", id="status_dashboard", markup=True)
                    with TabPane("Logs", id="logs_pane"):
                        yield RichLog(id="logs", wrap=True, markup=True)
                    with TabPane("Terminal", id="terminal_pane"):
                        yield RichLog(id="terminal", wrap=True, markup=False)
                    with TabPane("Planning", id="planning_pane"):
                        with VerticalScroll(id="planning_form"):
                            with Collapsible(title="Plan", collapsed=False, id="planning_plan"):
                                yield Markdown(messages.PLAN_EMPTY_MESSAGE, id="planning_plan_markdown")
                            with Collapsible(title="Task List", collapsed=False, id="planning_task_list"):
                                yield Markdown(messages.TASK_LIST_EMPTY_MESSAGE, id="planning_task_list_markdown")
                    with TabPane("Settings", id="settings_pane"):
                        with VerticalScroll(id="settings_form"):
                            with Vertical(classes="settings-section", id="settings_model") as model_section:
                                model_section.border_title = "Model"
                                yield Label("Provider")
                                yield Select(
                                    ui_provider_options(),
                                    id="provider_select",
                                    allow_blank=False,
                                    value=UI_DEFAULT_PROVIDER,
                                )
                                yield Label("Model")
                                yield Select(
                                    ui_model_options(UI_DEFAULT_PROVIDER),
                                    id="model_select",
                                    allow_blank=False,
                                    value=UI_DEFAULT_MODEL,
                                )
                                yield Label("Thinking effort")
                                yield Select(
                                    ui_thinking_effort_options(UI_DEFAULT_PROVIDER, UI_DEFAULT_MODEL),
                                    id="thinking_effort_select",
                                    allow_blank=True,
                                    value=default_ui_thinking_effort(UI_DEFAULT_PROVIDER, UI_DEFAULT_MODEL),
                                )
                                yield Label("API key")
                                yield Input(password=True, id="api_key_input")
                            with Vertical(classes="settings-section", id="settings_agent_models") as agents_section:
                                agents_section.border_title = "Agent Models"
                                yield Static(
                                    "Give individual agents their own model. "
                                    "Leave a role on “Default” to use the model above.",
                                    classes="settings-hint",
                                )
                                for role_label, role_value in agent_role_options():
                                    with Vertical(classes="agent-model-group"):
                                        yield Static(role_label, classes="agent-model-role")
                                        with Horizontal(classes="agent-model-field"):
                                            yield Label("Provider", classes="agent-model-field-label")
                                            yield Select(
                                                agent_role_provider_options(),
                                                id=f"am_provider_{role_value}",
                                                allow_blank=False,
                                                value=INHERIT_SENTINEL,
                                            )
                                        with Horizontal(classes="agent-model-field"):
                                            yield Label("Model", classes="agent-model-field-label")
                                            yield Select([], id=f"am_model_{role_value}", allow_blank=True, prompt="—")
                                        with Horizontal(classes="agent-model-field"):
                                            yield Label("Effort", classes="agent-model-field-label")
                                            yield Select([], id=f"am_effort_{role_value}", allow_blank=True, prompt="—")
                            with Vertical(classes="settings-section", id="settings_web_search") as web_search_section:
                                web_search_section.border_title = "Web Search"
                                yield Static(
                                    "Backend for the web_search tool. DuckDuckGo and Firecrawl work "
                                    "without a key; add a key for higher rate limits.",
                                    classes="settings-hint",
                                )
                                yield Label("Backend")
                                yield Select(
                                    available_backends(),
                                    id="web_search_backend_select",
                                    allow_blank=False,
                                    value=DEFAULT_WEB_SEARCH_BACKEND,
                                )
                                yield Label("API key", id="web_search_api_key_label")
                                yield Input(password=True, id="web_search_api_key_input")
                                yield Label("SearXNG base URL", id="web_search_base_url_label")
                                yield Input(
                                    id="web_search_base_url_input",
                                    placeholder="https://searxng.example.com",
                                )
                            with Vertical(classes="settings-section", id="settings_appearance") as appearance_section:
                                appearance_section.border_title = "Appearance"
                                yield Label("Theme")
                                yield Select(
                                    [(name, name) for name in theme.available_themes()],
                                    id="theme_select",
                                    allow_blank=False,
                                    value=theme.DEFAULT_THEME_NAME,
                                )
                            with Vertical(id="settings_actions"):
                                yield Button("Save Settings", variant="primary", id="save_settings")
                                yield Static("", id="settings_status")
        yield Footer()

    async def on_mount(self) -> None:
        self.settings = self.settings_store.load()
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
        self._populate_settings_controls()
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
        self._conversation.anchor()
        self.run_worker(self._consume_events(), name="kolega-events", group="events")
        if self.config is not None:
            await self._build_agent(self.config)
            self._set_chat_enabled(True)
            self.query_one("#composer", tui_widgets.ChatComposer).focus()
        else:
            await self._ensure_agent_from_settings()

    @property
    def _conversation(self) -> tui_widgets.ConversationView:
        return self.query_one("#conversation", tui_widgets.ConversationView)

    @property
    def _logs(self) -> RichLog:
        return self.query_one("#logs", RichLog)

    @property
    def _terminal(self) -> RichLog:
        return self.query_one("#terminal", RichLog)

    def _format_terminal_command(self, command: str) -> Text:
        """Accent prompt glyph plus the command in bold."""
        return Text.assemble(
            (theme.g(Glyph.USER) + " ", Color.ACCENT),
            (command, "bold"),
        )

    def _write_terminal_command(self, command: str) -> None:
        try:
            terminal = self._terminal
        except Exception:
            return
        if self._terminal_has_content:
            terminal.write("")
        terminal.write(self._format_terminal_command(command))
        self._terminal_has_content = True
        self._mark_tab_activity("terminal_pane")

    def _format_log_line(self, text: str, level: str = "info") -> Text:
        """One log line: muted HH:MM:SS, a level-colored glyph, then the text."""
        body_style = Color.MUTED if level == "debug" else ""
        return Text.assemble(
            (time.strftime("%H:%M:%S") + " ", Color.MUTED),
            (theme.g(Glyph.STATUS) + " ", theme.log_level_color(level)),
            (text, body_style),
        )

    def _write_log(self, text: str, level: str = "info") -> None:
        """Single write path into the Logs tab."""
        try:
            logs = self._logs
        except Exception:
            return
        logs.write(self._format_log_line(text, level))
        self._mark_tab_activity("logs_pane")

    def _mark_tab_activity(self, pane_id: str) -> None:
        """Add an activity dot to a background tab's label."""
        base = tui_constants.TAB_BASE_LABELS.get(pane_id)
        if base is None:
            return
        try:
            tabs = self.query_one("#events", TabbedContent)
            if tabs.active == pane_id:
                return
            tabs.get_tab(pane_id).label = f"{base} {theme.g(Glyph.STATUS)}"
        except Exception:
            return

    def _clear_tab_activity(self, pane_id: str) -> None:
        base = tui_constants.TAB_BASE_LABELS.get(pane_id)
        if base is None:
            return
        try:
            self.query_one("#events", TabbedContent).get_tab(pane_id).label = base
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

    @property
    def _status_dashboard(self) -> Static:
        return self.query_one("#status_dashboard", Static)

    @property
    def _turn_status(self) -> Static:
        return self.query_one("#turn_status", Static)

    @property
    def _settings_status(self) -> Static:
        return self.query_one("#settings_status", Static)

    def _validated_interaction_mode(self, interaction_mode: str) -> str:
        if interaction_mode in {tui_constants.BUILD_INTERACTION_MODE, tui_constants.PLAN_INTERACTION_MODE}:
            return interaction_mode
        return tui_constants.BUILD_INTERACTION_MODE

    def _sync_planning_state_to_session(self) -> None:
        self.session.interaction_mode = self.interaction_mode
        self.session.permission_mode = self.permission_mode.value
        self.session.latest_plan_markdown = self._latest_plan or ""
        self.session.plan_pending = bool(self._latest_plan and self._plan_pending)
        self.session.plan_reofferable = bool(self._latest_plan and self._plan_reofferable)

    def _save_session(self) -> None:
        self._sync_planning_state_to_session()
        self.store.save(self.session)

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
            self._reset_current_thread()
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
        # attach hint (which would otherwise linger during generation).
        event.composer.load_text("")
        self._pending_image_attachments.clear()
        self._clear_composer_hint()
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
        entries = self.file_index.search(active[0], limit=8)
        if not entries:
            dropdown.close()
            return
        dropdown.open_with([tui_widgets.file_completion_item(entry) for entry in entries])

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
                self._discuss_pending_plan()
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

    async def _process_message(self, message: str, attachments: list[dict] | None = None) -> None:
        if self.agent is None:
            return
        self._begin_turn_progress()
        self._log_status(messages.GENERATING, "ok")
        try:
            stream = (
                self.agent.process_message_stream(message, attachments)
                if attachments
                else self.agent.process_message_stream(message)
            )
            async for chunk in stream:
                if chunk.get("type") == "response":
                    if chunk.get("content"):
                        self._update_progress(messages.READING_RESPONSE, complete=False, state=tui_state.TurnState.GENERATING)
                    self._apply_stream_chunk(chunk, kind="assistant")
                    continue

                content = chunk.get("content")
                if chunk.get("type") == "thinking":
                    self._update_progress(messages.THINKING, complete=False, state=tui_state.TurnState.THINKING)
                    self._apply_stream_chunk(chunk, kind="thinking")
                    if content:
                        self._write_log(content, "debug")
            await self._drain_pending_events()
            self._finalize_sub_agent_activities()
            self._finalize_workflow_activities()
            self._save_session_history()
            self._finish_turn_progress(messages.FINISHED, tui_state.TurnState.IDLE)
            self._capture_completed_plan()
            self._log_status(messages.FINISHED, "ok")
        except asyncio.CancelledError:
            self._cancel_pending_question()
            self._cancel_pending_approval()
            await self._drain_pending_events()
            self._finalize_sub_agent_activities()
            self._finalize_workflow_activities()
            self._save_session_history()
            self._finish_turn_progress(messages.STOPPED_BY_USER, tui_state.TurnState.STOPPED)
            self._log_status(messages.STOPPED_BY_USER, "warn")
        except LLMError as exc:
            self._cancel_pending_question()
            self._cancel_pending_approval()
            await self._drain_pending_events()
            self._finalize_sub_agent_activities()
            self._finalize_workflow_activities()
            self._save_session_history()
            model = self.config.long_context_config.model if self.config is not None else None
            message_text = llm_error_message(exc, model=model)
            self._finish_turn_progress(message_text, tui_state.TurnState.ERROR)
            self._log_status(message_text, "error")
        except Exception as exc:
            self._cancel_pending_question()
            self._cancel_pending_approval()
            await self._drain_pending_events()
            self._finalize_sub_agent_activities()
            self._finalize_workflow_activities()
            self._save_session_history()
            self._finish_turn_progress(messages.STOPPED_WITH_ERROR.format(error=exc), tui_state.TurnState.ERROR)
            self._log_status(messages.STOPPED_WITH_ERROR.format(error=exc), "error")
            raise
        finally:
            self._flush_conversation_render()
            self._active_progress_entry = None
            self._turn_active = False
            self.agent_worker = None
            if self._plan_decision_active:
                self._set_composer_status(messages.PLAN_READY_PLACEHOLDER)
            else:
                self._restore_composer_placeholder()
            self._set_chat_enabled(self.agent is not None and not self._plan_decision_active)

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "detach_btn":
            await self._command_detach("")
            return
        if event.button.id == "save_settings":
            await self._save_settings_from_ui()

    def on_select_changed(self, event: Select.Changed) -> None:
        select_id = event.select.id or ""

        if select_id == "provider_select":
            provider = str(event.value)
            self._repopulate_model_select(provider, "model_select", "thinking_effort_select")
            try:
                api_key_input = self.query_one("#api_key_input", Input)
                api_key_input.placeholder = self._api_key_placeholder(provider)
                # OAuth providers sign in via /login, so the key field is read-only.
                api_key_input.disabled = provider == chatgpt_constants.PROVIDER_KEY
            except NoMatches:
                pass
            return

        if select_id == "model_select":
            try:
                provider = str(self.query_one("#provider_select", Select).value)
            except NoMatches:
                return
            self._set_effort_select_default(provider, str(event.value))
            return

        if select_id == "web_search_backend_select":
            self._update_search_backend_fields(str(event.value))
            return

        if select_id.startswith("am_provider_"):
            role = select_id[len("am_provider_") :]
            provider = str(event.value)
            if provider == INHERIT_SENTINEL:
                # Don't drop pending here: the selects post an initial inherit-valued
                # Changed on mount, which would clear a restore before its real cascade
                # runs. _populate_agent_model_rows clears stale pending per row instead.
                self._clear_model_effort_selects(f"am_model_{role}", f"am_effort_{role}")
            else:
                model_value = self._pending_agent_models.pop(f"am_model_{role}", None)
                self._repopulate_model_select(
                    provider, f"am_model_{role}", f"am_effort_{role}", model_value=model_value
                )
            return

        if select_id.startswith("am_model_"):
            role = select_id[len("am_model_") :]
            try:
                provider = str(self.query_one(f"#am_provider_{role}", Select).value)
            except NoMatches:
                return
            if provider != INHERIT_SENTINEL and event.value is not Select.NULL:
                # A restored effort waits here for the model that hosts it; a manual
                # model change has none pending and falls back to preserve/default.
                preferred = self._pending_agent_efforts.pop(f"am_effort_{role}", None)
                self._set_effort_select_default(provider, str(event.value), f"am_effort_{role}", preferred=preferred)
            return

        if select_id == "theme_select":
            name = str(event.value)
            if name != (self.settings.active_theme or theme.DEFAULT_THEME_NAME):
                self.settings.active_theme = name
                self.settings_store.save(self.settings)
                self._apply_theme(name)

    async def _consume_events(self) -> None:
        while True:
            event = await self.connection_manager.next_event()
            self._render_event(event)

    async def _drain_pending_events(self) -> None:
        while True:
            try:
                event = self.connection_manager.events.get_nowait()
            except asyncio.QueueEmpty:
                return
            self._render_event(event)

    def _render_event(self, event: AgentEvent) -> None:
        text = self._display_text_from_event(event)
        if event.event_type == "log_message":
            level = str(event.content.get("level", "info"))
            self._write_log(text, level)
        elif event.event_type == "terminal_output":
            self._terminal.write(event.content.get("output", ""))
            self._terminal_has_content = True
            self._mark_tab_activity("terminal_pane")
        elif event.event_type == "terminal_command":
            command = str(event.content.get("command") or "")
            self._write_terminal_command(command)
            if command:
                self._update_activity_progress(messages.RUNNING_TERMINAL_COMMAND, state=tui_state.TurnState.RUNNING_TOOL)
        elif event.event_type == "chat_message":
            if event.sub_agent_info:
                self._render_sub_agent_event(event)
                return
            message_text = event.content.get("text", "")
            message_type = event.content.get("message_type", "message")
            if message_type in {"tool_call", "tool_result", "tool_error"}:
                self._add_tool_message(message_type, event.content)
            elif message_type == "workflow_start":
                self._handle_workflow_start(event.content)
            elif message_type == "workflow_phase":
                self._handle_workflow_phase(event.content)
            elif message_type == "workflow_log":
                self._handle_workflow_log(event.content)
            elif message_type == "workflow_end":
                self._handle_workflow_end(event.content)
            elif message_text:
                self._add_conversation_entry(tui_state.ConversationEntry(kind="message", content=message_text))
        elif event.event_type == "tool_streaming_update":
            if event.sub_agent_info:
                self._note_sub_agent_tool_stream(event)
            else:
                self._apply_tool_streaming_update(event.content)
        elif event.event_type == "file_edit_preview":
            # UI-only inline diff/head preview. Sub-agent edits are not shown inline (v1).
            if not event.sub_agent_info:
                self._apply_edit_preview(event.content)
        elif event.event_type == "llm_context_update":
            if event.sub_agent_info:
                self._note_sub_agent_context(event)
            else:
                self._apply_context_status_update(event.content)
        elif event.event_type == "compaction_status":
            # Only the main agent's compaction drives the status dashboard; a
            # sub-agent's compaction must not stomp the main indicator.
            if not event.sub_agent_info:
                self._apply_compaction_status(event.content)
        elif event.event_type in {"llm_status_update", "status_update"}:
            if event.sub_agent_info:
                self._note_sub_agent_status(event)
            elif text:
                self._write_log(text, "info")
                self._update_activity_progress(text)
        else:
            if text:
                self._write_log(f"{event.event_type}: {text}", "info")
            else:
                self._write_log(messages.LOG_IGNORED_EVENT.format(event_type=event.event_type), "debug")

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

    def action_cancel_generation(self) -> None:
        if self.agent_worker is not None:
            self._update_progress(messages.STOP_REQUESTED, complete=False, state=tui_state.TurnState.STOPPING)
            self._cancel_pending_question()
            self._cancel_pending_approval()
            self.agent_worker.cancel()
            self._notify_user(messages.CANCEL_REQUESTED, severity="warning")

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

        target = tui_constants.PLAN_INTERACTION_MODE if self.interaction_mode == tui_constants.BUILD_INTERACTION_MODE else tui_constants.BUILD_INTERACTION_MODE
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

    def _default_sub_agent_key(self) -> Optional[str]:
        """Most-recently-started running agent, else the most recent overall."""
        pool = self._running_sub_agents() or list(self._sub_agent_activities.values())
        if not pool:
            return None
        return max(pool, key=lambda a: a.index).agent_id

    def _close_sub_agent_inspector(self) -> None:
        screen = self._sub_agent_inspector
        if screen is None:
            return
        self._sub_agent_inspector = None
        try:
            screen.dismiss()
        except Exception:
            pass

    def on_sub_agent_entry_widget_pressed(self, message: tui_sub_agents.SubAgentEntryWidget.Pressed) -> None:
        activity = self._sub_agent_activity_for_entry(message.entry)
        if activity is not None:
            self.action_open_sub_agent(activity.agent_id)

    async def action_quit(self) -> None:
        if self.agent is not None:
            fire = getattr(self.agent, "fire_hook", None)
            if fire is not None:
                try:
                    await fire(HookEvent.SESSION_END, {"reason": "quit"})
                except Exception:
                    pass
            self._persist_agent_into_session()
            self._save_session()
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
            try:
                composer = self.query_one("#composer", tui_widgets.ChatComposer)
                if not composer.disabled:
                    composer.focus()
            except Exception:
                return

    def _populate_settings_controls(self) -> None:
        provider_values = {value for _, value in ui_provider_options()}
        provider = (
            self.settings.active_provider if self.settings.active_provider in provider_values else UI_DEFAULT_PROVIDER
        )
        model_options = ui_model_options(provider)
        valid_models = {value for _, value in model_options}
        model = self.settings.active_model if self.settings.active_model in valid_models else None
        if model is None:
            model = model_options[0][1] if model_options else UI_DEFAULT_MODEL
        effort_options = {value for _, value in ui_thinking_effort_options(provider, model)}
        effort = (
            self.settings.active_thinking_effort if self.settings.active_thinking_effort in effort_options else None
        )
        if effort is None:
            effort = default_ui_thinking_effort(provider, model)
        provider_select = self.query_one("#provider_select", Select)
        model_select = self.query_one("#model_select", Select)
        effort_select = self.query_one("#thinking_effort_select", Select)
        api_key_input = self.query_one("#api_key_input", Input)

        provider_select.value = provider
        model_select.set_options(model_options)
        model_select.value = model
        effort_select.set_options(ui_thinking_effort_options(provider, model))
        if effort is not None:
            effort_select.value = effort
        theme_select = self.query_one("#theme_select", Select)
        theme_select.value = (
            self.settings.active_theme
            if self.settings.active_theme in theme.available_themes()
            else theme.DEFAULT_THEME_NAME
        )
        api_key_input.placeholder = self._api_key_placeholder(provider)
        api_key_input.disabled = provider == chatgpt_constants.PROVIDER_KEY
        self._populate_agent_model_rows()
        self._populate_web_search_controls()
        self._update_settings_status()

    def _populate_agent_model_rows(self) -> None:
        """Seed each per-agent row from saved settings (absent role -> inherit).

        A model select that was just given its options can't accept a value until the
        next refresh, and setting the provider value posts a Changed that re-runs the
        cascade afterwards. So we stash the saved model/effort and let that cascade —
        the last writer — apply them, rather than assigning here and being clobbered."""
        provider_values = {value for _, value in ui_provider_options()}
        for _, role in agent_role_options():
            try:
                provider_select = self.query_one(f"#am_provider_{role}", Select)
            except NoMatches:
                continue
            entry = self.settings.get_agent_model(role) or {}
            provider = entry.get("provider")
            self._pending_agent_models.pop(f"am_model_{role}", None)
            self._pending_agent_efforts.pop(f"am_effort_{role}", None)
            if provider not in provider_values:
                provider_select.value = INHERIT_SENTINEL
                self._clear_model_effort_selects(f"am_model_{role}", f"am_effort_{role}")
                continue
            if entry.get("model"):
                self._pending_agent_models[f"am_model_{role}"] = str(entry["model"])
            if entry.get("thinking_effort"):
                self._pending_agent_efforts[f"am_effort_{role}"] = str(entry["thinking_effort"])
            # Triggers on_select_changed, which consumes the pending model/effort.
            provider_select.value = provider

    def _update_search_backend_fields(self, backend: str) -> None:
        """Show only the inputs the selected web-search backend needs.

        Called from on_select_changed (which can fire its initial Changed on mount,
        before the section is fully populated) and from populate, so every query_one
        is guarded against NoMatches."""
        try:
            backend_cls = get_backend_class(backend)
        except SearchBackendError:
            backend_cls = None
        needs_key = bool(backend_cls and backend_cls.accepts_api_key)
        needs_url = bool(backend_cls and backend_cls.requires_base_url)
        for widget_id, visible in (
            ("web_search_api_key_label", needs_key),
            ("web_search_api_key_input", needs_key),
            ("web_search_base_url_label", needs_url),
            ("web_search_base_url_input", needs_url),
        ):
            try:
                self.query_one(f"#{widget_id}").display = visible
            except NoMatches:
                pass
        if needs_key:
            try:
                key_input = self.query_one("#web_search_api_key_input", Input)
            except NoMatches:
                return
            env_var = (backend_cls.env_var if backend_cls else None) or "API"
            if self.settings.has_api_key(backend):
                key_input.placeholder = "Stored API key will be kept if blank"
            elif backend_cls and backend_cls.requires_api_key:
                key_input.placeholder = f"{env_var} key"
            else:
                key_input.placeholder = f"Optional — {env_var} key for higher rate limits"

    def _populate_web_search_controls(self) -> None:
        """Seed the Web Search controls from saved settings (key field stays blank)."""
        valid = {name for _, name in available_backends()}
        backend = self.settings.web_search_backend
        if backend not in valid:
            backend = DEFAULT_WEB_SEARCH_BACKEND
        try:
            self.query_one("#web_search_backend_select", Select).value = backend
            self.query_one("#web_search_base_url_input", Input).value = self.settings.web_search_base_url or ""
            self.query_one("#web_search_api_key_input", Input).value = ""
        except NoMatches:
            pass
        self._update_search_backend_fields(backend)

    def _collect_web_search_from_ui(self) -> None:
        """Write the Web Search controls into settings (keys only when newly typed)."""
        try:
            backend = str(self.query_one("#web_search_backend_select", Select).value)
            base_url_input = self.query_one("#web_search_base_url_input", Input)
            key_input = self.query_one("#web_search_api_key_input", Input)
        except NoMatches:
            return
        self.settings.web_search_backend = backend
        self.settings.web_search_base_url = base_url_input.value.strip() or None
        key = key_input.value.strip()
        if key and backend in WEB_SEARCH_KEY_NAMES:
            self.settings.set_api_key(backend, key)
        key_input.value = ""
        self._update_search_backend_fields(backend)

    def _set_effort_select_default(
        self, provider: str, model: str, effort_id: str = "thinking_effort_select", *, preferred: Optional[str] = None
    ) -> None:
        try:
            effort_select = self.query_one(f"#{effort_id}", Select)
        except Exception:
            return
        # Prefer an explicit value (a restored effort), else keep the current one if it
        # is still valid for this model, else fall back to the model's default. This
        # keeps a restore or a provider switch from clobbering the chosen effort.
        current = effort_select.value
        current = None if current is Select.NULL else str(current)
        effort_options = ui_thinking_effort_options(provider, model)
        valid_efforts = {value for _, value in effort_options}
        effort_select.set_options(effort_options)
        if preferred in valid_efforts:
            chosen = preferred
        elif current in valid_efforts:
            chosen = current
        else:
            chosen = default_ui_thinking_effort(provider, model)
        if chosen is not None:
            effort_select.value = chosen

    def _repopulate_model_select(
        self,
        provider: str,
        model_id: str,
        effort_id: str,
        *,
        model_value: Optional[str] = None,
        effort_value: Optional[str] = None,
    ) -> None:
        """Fill a provider→model→effort trio for a provider.

        Used by the global Model section and each per-agent row. ``model_value`` /
        ``effort_value`` pre-select a model/effort (used while restoring saved
        settings). Otherwise the select's current model is kept when it is still valid
        for ``provider`` (so a restore is not clobbered), falling back to the
        provider's first model."""
        try:
            model_select = self.query_one(f"#{model_id}", Select)
        except NoMatches:
            return
        if model_value is None:
            current = model_select.value
            model_value = None if current is Select.NULL else str(current)
        model_options = ui_model_options(provider)
        model_select.set_options(model_options)
        valid_models = {value for _, value in model_options}
        model = model_value if (model_value and model_value in valid_models) else None
        if model is None:
            model = model_options[0][1] if model_options else UI_DEFAULT_MODEL
        if model_options:
            model_select.value = model
        self._set_effort_select_default(provider, model, effort_id, preferred=effort_value)

    def _clear_model_effort_selects(self, model_id: str, effort_id: str) -> None:
        """Blank a per-agent row's model+effort selects (the role inherits)."""
        for select_id in (model_id, effort_id):
            try:
                select = self.query_one(f"#{select_id}", Select)
            except NoMatches:
                continue
            select.set_options([])
            select.value = Select.NULL

    async def _save_settings_from_ui(self) -> None:
        provider = str(self.query_one("#provider_select", Select).value)
        model = str(self.query_one("#model_select", Select).value)
        effort = str(self.query_one("#thinking_effort_select", Select).value)
        valid_efforts = {value for _, value in ui_thinking_effort_options(provider, model)}
        if effort not in valid_efforts:
            effort = default_ui_thinking_effort(provider, model) or ""
        api_key_input = self.query_one("#api_key_input", Input)
        api_key = api_key_input.value.strip()

        self.settings.active_provider = provider
        self.settings.active_model = model
        self.settings.active_thinking_effort = effort or default_ui_thinking_effort(provider, model)
        self.settings.active_theme = str(self.query_one("#theme_select", Select).value)
        if api_key:
            self.settings.set_api_key(provider, api_key)
        self._collect_agent_models_from_ui()
        self._collect_web_search_from_ui()
        self.settings_store.save(self.settings)
        api_key_input.value = ""
        api_key_input.placeholder = self._api_key_placeholder(provider)

        await self._ensure_agent_from_settings(rebuild=True)
        if self.config is not None:
            self._notify_user(messages.SETTINGS_SAVED)

    def _collect_agent_models_from_ui(self) -> None:
        """Write each per-agent row into settings.agent_models (inherit rows removed)."""
        for _, role in agent_role_options():
            try:
                provider = str(self.query_one(f"#am_provider_{role}", Select).value)
                model_select = self.query_one(f"#am_model_{role}", Select)
                effort_select = self.query_one(f"#am_effort_{role}", Select)
            except NoMatches:
                continue
            if provider == INHERIT_SENTINEL or model_select.value is Select.NULL:
                self.settings.clear_agent_model(role)
                continue
            model = str(model_select.value)
            effort = "" if effort_select.value is Select.NULL else str(effort_select.value)
            valid_efforts = {value for _, value in ui_thinking_effort_options(provider, model)}
            if effort not in valid_efforts:
                effort = default_ui_thinking_effort(provider, model) or ""
            self.settings.set_agent_model(role, provider, model, effort or None)

    async def _ensure_agent_from_settings(self, rebuild: bool = False) -> None:
        try:
            config = build_agent_config(
                self.project_path, self.overrides, settings=self.settings, settings_store=self.settings_store
            )
        except CliConfigError as exc:
            self.config = None
            self._set_chat_enabled(False)
            self._refresh_status_dashboard()
            self._set_settings_status(messages.SETTINGS_INCOMPLETE.format(error=exc), tone="error")
            self._ensure_startup_entry()
            self.query_one("#events", TabbedContent).active = "settings_pane"
            return

        self.config = config
        self.session.config = config_summary(config)
        self._save_session()
        await self._build_agent(config, rebuild=rebuild)
        self._set_chat_enabled(True)
        self._update_settings_status()
        self._ensure_startup_entry()
        self.query_one("#composer", tui_widgets.ChatComposer).focus()

    async def _build_agent(self, config: AgentConfig, rebuild: bool = False) -> None:
        history = self.session.history
        compaction = self.session.compaction
        if self.agent is not None:
            history = self.agent.dump_message_history()
            compaction = self.agent.dump_compaction_state()
            self.session.history = history
            self.session.compaction = compaction
            self._save_session()
            if rebuild:
                await self.agent.cleanup()

        browser_manager = PlaywrightBrowserManager()
        browser_manager.headless = not self.browser_visible
        agent_class = PlanningAgent if self.interaction_mode == tui_constants.PLAN_INTERACTION_MODE else CoderAgent
        self.skill_catalog = discover_skills(self.project_path)
        prompt_extensions: list[PromptExtension] = []
        tool_extensions: list[ToolExtension] = []
        # The shared task list is build-mode execution tracking; plan mode produces
        # a plan via write_plan and does not get the task-list tools.
        if self.interaction_mode == tui_constants.BUILD_INTERACTION_MODE:
            prompt_extensions.append(self._shared_task_list_prompt_extension())
            tool_extensions.append(self._shared_task_list_tool_extension())
        skill_prompt_extension = build_skill_prompt_extension(self.skill_catalog)
        skill_tool_extension = build_skill_tool_extension(
            self.skill_catalog,
            lambda: self.agent.history if self.agent is not None else [],
        )
        if skill_prompt_extension is not None:
            prompt_extensions.append(skill_prompt_extension)
        if skill_tool_extension is not None:
            tool_extensions.append(skill_tool_extension)
        if self.interaction_mode == tui_constants.PLAN_INTERACTION_MODE:
            prompt_extensions.append(self._planning_question_prompt_extension())
            tool_extensions.append(self._planning_question_tool_extension())

        # gigacode applies to any top-level agent and is carried across rebuilds.
        # In plan mode the orchestrating agent is read-only, so its workflow
        # sub-agents are forced read-only too (enforced in the dispatch adapter).
        gigacode_active = self._gigacode_enabled
        if gigacode_active:
            prompt_extensions.append(self._gigacode_prompt_extension())

        self.agent = agent_class(
            project_path=self.project_path,
            workspace_id=self.session.workspace_id,
            thread_id=self.session.thread_id,
            connection_manager=self.connection_manager,
            config=config,
            browser_manager=browser_manager,
            agent_mode=AgentMode(self.mode),
            prompt_extensions=prompt_extensions,
            tool_extensions=tool_extensions,
            permission_mode=self.permission_mode,
            permission_callback=self._permission_callback,
            hook_dispatcher=self._session_hook_dispatcher(),
        )
        self.agent.gigacode_enabled = gigacode_active
        if history:
            self.agent.restore_message_history(history)
            self.agent.restore_compaction_state(compaction)
            self._restore_conversation_history(history)
        self._update_mode_chrome()
        await self._fire_session_start_once()

    def _session_hook_dispatcher(self) -> HookDispatcher:
        """Build (once) the hook dispatcher for this session from global + project config."""
        if self._hook_dispatcher is None:
            trusted = self.settings.is_hook_project_trusted(self.project_path)
            config = load_hook_config(self.project_path, self.settings_store.root, project_trusted=trusted)
            self._hook_dispatcher = HookDispatcher(config)
            self._announce_hook_status(config)
        return self._hook_dispatcher

    def _announce_hook_status(self, config) -> None:
        """Surface hook diagnostics and an untrusted-project notice once at startup."""
        for diagnostic in config.diagnostics:
            self._log_status(f"hooks: {diagnostic}", level="warn")
        if project_hooks_present(self.project_path) and not self.settings.is_hook_project_trusted(self.project_path):
            self._notify_user(
                "This project defines hooks in .kolega/hooks.json, but they are not trusted, so they "
                "are disabled. Global hooks still run. Re-launch with `--trust-hooks` to enable them.",
                severity="warning",
                title="Untrusted project hooks",
            )

    async def _fire_session_start_once(self) -> None:
        if self._session_started or self.agent is None:
            return
        fire = getattr(self.agent, "fire_hook", None)
        if fire is None:
            return
        self._session_started = True
        outcome = await fire(HookEvent.SESSION_START, {"source": "startup"})
        if outcome.additional_context:
            self._add_conversation_entry(tui_state.ConversationEntry(kind="system", content=outcome.additional_context))

    async def _set_interaction_mode(self, interaction_mode: str) -> None:
        if interaction_mode not in {tui_constants.BUILD_INTERACTION_MODE, tui_constants.PLAN_INTERACTION_MODE}:
            raise ValueError(f"Unknown interaction mode: {interaction_mode}")
        if self.interaction_mode == interaction_mode:
            return

        self.interaction_mode = interaction_mode
        self._plan_decision_active = False
        self._save_session()
        self._restore_plan_action_visibility()
        self._cancel_pending_question()
        self._cancel_pending_approval()
        self._cancel_pending_model_selection()
        self._cancel_pending_effort_selection()
        self._cancel_pending_theme_selection()

        if self.config is not None:
            await self._build_agent(self.config, rebuild=True)

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
        self._save_session()
        if self.agent is not None:
            self.agent.set_permission_mode(mode)
            self.agent.set_permission_callback(self._permission_callback)
        self._update_mode_chrome()
        self._notify_user(messages.SWITCHED_PERMISSION_MODE.format(mode=mode.value))

    def _capture_completed_plan(self) -> None:
        if self.interaction_mode != tui_constants.PLAN_INTERACTION_MODE or not isinstance(self.agent, PlanningAgent):
            return

        plan = self.agent.consume_completed_plan()
        if plan:
            self._latest_plan = plan
            self._plan_reofferable = True
            self._show_plan_for_decision(plan, notification=messages.PLAN_CAPTURED)
            return

        if self._latest_plan and self._plan_reofferable and not self._plan_pending:
            self._show_plan_for_decision(self._latest_plan, notification=messages.PLAN_REOFFERED)

    def _show_plan_for_decision(self, plan: str, *, notification: str) -> None:
        self._plan_pending = True
        self._plan_decision_active = True
        self._save_session()
        self._refresh_planning_sidebar()
        self._add_conversation_entry(tui_state.ConversationEntry(kind="plan", content=plan, complete=True))
        self._set_plan_actions_visible(True, allow_discuss=True)
        self._set_composer_status(messages.PLAN_READY_PLACEHOLDER)
        self._set_chat_enabled(False)
        self._notify_user(notification)

    async def _implement_pending_plan(self, *, clear_context: bool = False) -> None:
        plan = self._latest_plan
        if not plan or not self._plan_pending or self._turn_active or self.agent_worker is not None:
            return

        # Leave self._latest_plan set so the planning sidebar keeps showing the
        # plan as a read-only reference while it is being built; clearing
        # _plan_pending is what hides the "Implement plan" action so it does not
        # reappear when the user re-enters plan mode.
        self._plan_pending = False
        self._plan_reofferable = False
        self._plan_decision_active = False
        if clear_context:
            self._clear_agent_context()
        self._save_session()
        await self._set_interaction_mode(tui_constants.BUILD_INTERACTION_MODE)
        self._refresh_planning_sidebar()
        self._set_plan_actions_visible(False)

        prompt = build_implement_plan_prompt(plan, gigacode_enabled=self._gigacode_enabled)
        self._add_conversation_entry(tui_state.ConversationEntry(kind="user", content="Implement the approved plan."))
        self.agent_worker = self.run_worker(
            self._process_message(prompt), name="kolega-turn", group="turns", exclusive=True
        )

    def _discuss_pending_plan(self) -> None:
        if not self._latest_plan:
            return

        self._plan_pending = False
        self._plan_reofferable = True
        self._plan_decision_active = False
        self._save_session()
        self._refresh_planning_sidebar()
        self._set_plan_actions_visible(False)
        self._restore_composer_placeholder()
        self._set_chat_enabled(self.agent is not None)
        self.query_one("#composer", tui_widgets.ChatComposer).focus()
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
        return (
            f"{self.project_path} | session {self.session.session_id} | "
            f"agent {self.mode} | {self.interaction_mode} | permissions {self.permission_mode.value}"
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
            plan_markdown = self.query_one("#planning_plan_markdown", Markdown)
            task_list_markdown = self.query_one("#planning_task_list_markdown", Markdown)
            plan_markdown.update(plan_content)
            task_list_markdown.update(task_list_content)
            plan_markdown.set_class(plan_content == messages.PLAN_EMPTY_MESSAGE, "empty-state")
            task_list_markdown.set_class(task_list_content == messages.TASK_LIST_EMPTY_MESSAGE, "empty-state")
        except Exception:
            pass

    def _set_chat_enabled(self, enabled: bool) -> None:
        composer = self.query_one("#composer", tui_widgets.ChatComposer)
        composer.disabled = not enabled or self._plan_decision_active or self._pending_approval is not None

    def _set_composer_status(self, status: str) -> None:
        self.query_one("#composer", tui_widgets.ChatComposer).placeholder = status

    def _restore_composer_placeholder(self) -> None:
        self.query_one("#composer", tui_widgets.ChatComposer).placeholder = messages.COMPOSER_PLACEHOLDER
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

    def _tui_command_handlers(self) -> dict[str, Callable[[str], Awaitable[None]]]:
        return {
            "/attach": self._command_attach,
            "/detach": self._command_detach,
            "/init": self._command_init,
            "/plan": self._command_plan,
            "/build": self._command_build,
            "/sidebar": self._command_sidebar,
            "/permissions": self._command_permissions,
            "/model": self._command_model,
            "/effort": self._command_effort,
            "/login": self._command_login,
            "/logout": self._command_logout,
            "/gigacode": self._command_gigacode,
            "/theme": self._command_theme,
            "/copy": self._command_copy,
            "/version": self._command_version,
            "/update": self._command_update,
            "/quit": self._command_quit,
            "/exit": self._command_quit,
        }

    async def _handle_tui_slash_command(self, stripped_text: str, composer: tui_widgets.ChatComposer) -> bool:
        if not stripped_text.startswith("/"):
            return False
        command_text, _, args = stripped_text.partition(" ")
        handler = self._tui_command_handlers().get(command_text.lower())
        if handler is None:
            return False
        if command_text.lower() != "/model":
            self._cancel_pending_model_selection()
        if command_text.lower() != "/effort":
            self._cancel_pending_effort_selection()
        if command_text.lower() != "/theme":
            self._cancel_pending_theme_selection()
        composer.load_text("")
        await handler(args.strip())
        return True

    def _model_supports_vision(self) -> bool:
        """Safely check if the current agent's model supports vision input.

        Returns ``False`` when no agent is loaded (conservative: images won't
        work without a model, so a warning is the right default).
        """
        if self.agent is None:
            return False
        return getattr(self.agent, "supports_vision", False)

    def _add_vision_mismatch_system_message(self, *, context: str) -> None:
        """Add a persistent warning to the transcript when images meet a non-vision model.

        ``context`` is ``"attachment"`` (new image attached) or ``"model_switch"``
        (switched to a non-vision model with images in history). Deduplicated per
        model session so repeated attachments don't spam the transcript — the
        composer hint still updates with each attachment (showing all names), but
        the transcript system message appears only once.
        """
        if self._vision_warning_shown:
            return
        model_config = getattr(self.agent, "primary_model_config", None)
        model_name = getattr(model_config, "model", None) or "The current model"
        if context == "attachment":
            message = (
                f"⚠ {model_name} does not support vision. Use /detach to remove image "
                f"attachments or /model to switch to a vision-capable model."
            )
        else:  # model_switch
            message = messages.MODEL_NON_VISION_IMAGE_HISTORY
        self._add_conversation_entry(tui_state.ConversationEntry(kind="system", content=message, tone="warning"))
        self._vision_warning_shown = True

    def add_pending_image_attachment(self, attachment: dict) -> None:
        """Stash a pending image attachment for the next submitted message.

        Centralized vision check: when the current model does not support vision,
        shows a combined warning-tone hint (attachment confirmation + vision
        mismatch) and a persistent system message in the transcript. This is the
        single funnel for clipboard paste (both Ctrl+Shift+V and on_paste) and
        /attach, so the warning is consistent across all three paths.
        """
        self._pending_image_attachments.append(attachment)
        names = ", ".join(a.get("path", "image") for a in self._pending_image_attachments)
        if not self._model_supports_vision():
            self._show_composer_hint(
                f"Attached: {names}. {messages.MODEL_NON_VISION_IMAGE_ATTACHED}",
                tone="warning",
            )
            self._add_vision_mismatch_system_message(context="attachment")
        else:
            self._show_composer_hint(
                f"Attached images: {names} (press Enter to send, × to remove)", tone="info"
            )

    async def _command_attach(self, args: str) -> None:
        arg = args.strip()
        if not arg:
            self._show_composer_hint(
                "Usage: /attach <path-to-image>  (PNG, JPEG, GIF, WebP, BMP)", tone="warning"
            )
            return
        from pathlib import Path as _Path

        from kolega_code.utils.images import encode_image_file

        candidate = _Path(arg)
        if not candidate.is_absolute():
            candidate = (self.project_path / arg).resolve()
        attachment = encode_image_file(candidate)
        if attachment is None:
            self._show_composer_hint(
                f"Could not attach {arg}: not a supported image, missing, or too large (>20MB). Use /attach <path>",
                tone="warning",
            )
            return
        attachment["path"] = arg
        self.add_pending_image_attachment(attachment)

    async def _command_detach(self, args: str) -> None:
        """Remove all pending image attachments (clears the attach queue).

        The user has no other way to discard a pending image once attached,
        especially on a non-vision model where the image can't be sent.
        """
        if not self._pending_image_attachments:
            self._show_composer_hint("No pending image attachments to remove.", tone="info")
            return
        names = ", ".join(a.get("path", "image") for a in self._pending_image_attachments)
        count = len(self._pending_image_attachments)
        self._pending_image_attachments.clear()
        self._show_composer_hint(
            f"Removed {count} image attachment(s): {names}", tone="info"
        )

    async def _paste_clipboard_image_worker(self) -> None:
        from kolega_code.cli.clipboard_image import read_clipboard_image
        from kolega_code.utils.images import encode_image_attachment

        result = await read_clipboard_image()
        if result is None:
            self._show_composer_hint(
                "No image on the clipboard, or your terminal doesn't support image paste. "
                "Use /attach <path> or @image.png instead.",
                tone="warning",
            )
            return
        data, media_type = result
        attachment = encode_image_attachment(data, media_type, path="clipboard")
        # Vision check is centralized in add_pending_image_attachment.
        self.add_pending_image_attachment(attachment)

    async def _command_plan(self, args: str) -> None:
        if self._mode_switch_blocked():
            return
        await self._set_interaction_mode(tui_constants.PLAN_INTERACTION_MODE)

    async def _command_build(self, args: str) -> None:
        if self._mode_switch_blocked():
            return
        await self._set_interaction_mode(tui_constants.BUILD_INTERACTION_MODE)

    async def _command_gigacode(self, args: str) -> None:
        if self.agent is None:
            self._set_settings_status(messages.SETTINGS_REQUIRED, tone="warning")
            return

        clean = args.strip().lower()
        if clean in ("", "toggle"):
            new_state = not self._gigacode_enabled
        elif clean in ("on", "enable", "enabled", "true"):
            new_state = True
        elif clean in ("off", "disable", "disabled", "false"):
            new_state = False
        else:
            self._notify_user("Usage: /gigacode [on|off]", severity="warning")
            return

        self._gigacode_enabled = new_state
        self.agent.apply_gigacode(new_state, self._gigacode_prompt_extension() if new_state else None)

        if new_state:
            note = (
                "gigacode workflow orchestration enabled — I can now author multi-agent "
                "workflows with the run_workflow tool for large fan-out tasks."
            )
            if self.interaction_mode == tui_constants.PLAN_INTERACTION_MODE:
                note += " In plan mode, workflow sub-agents are read-only (parallel research only)."
        else:
            note = "gigacode workflow orchestration disabled."
        self._add_conversation_entry(tui_state.ConversationEntry(kind="system", content=note))
        self._update_mode_chrome()

    def _gigacode_prompt_extension(self) -> PromptExtension:
        from kolega_code.agent.orchestration.guide import GIGACODE_AUTHORING_GUIDE

        return PromptExtension(
            id="gigacode",
            title="gigacode — workflow orchestration",
            markdown=GIGACODE_AUTHORING_GUIDE,
            agent_types=None,
            modes=None,
            # Sub-agents can't run workflows (run_workflow is gated off for them),
            # so the authoring guide is just prompt bloat for a sub-agent.
            propagate_to_sub_agents=False,
        )

    async def _command_sidebar(self, args: str) -> None:
        await self.action_toggle_sidebar()

    async def _command_init(self, args: str) -> None:
        if self._pending_question is not None:
            self._set_composer_status(messages.QUESTION_PLACEHOLDER)
            self._notify_user(messages.BLOCK_PENDING_QUESTION_INIT, severity="warning")
            return

        if self._pending_approval is not None:
            self._set_composer_status(messages.APPROVAL_PLACEHOLDER)
            self._notify_user(messages.BLOCK_PENDING_APPROVAL, severity="warning")
            return

        if self._turn_active or self.agent_worker is not None:
            self._show_composer_hint(messages.BLOCK_STOP_BEFORE_INIT)
            self._notify_user(messages.BLOCK_STOP_BEFORE_INIT, severity="warning")
            return

        if self._plan_decision_active:
            self._set_composer_status(messages.PLAN_READY_PLACEHOLDER)
            self._notify_user(messages.BLOCK_PLAN_DECISION_INIT, severity="warning")
            return

        if self.agent is None:
            self._set_settings_status(messages.SETTINGS_REQUIRED, tone="warning")
            return

        if self.interaction_mode != tui_constants.BUILD_INTERACTION_MODE:
            await self._set_interaction_mode(tui_constants.BUILD_INTERACTION_MODE)

        if self.agent is None:
            self._set_settings_status(messages.SETTINGS_REQUIRED, tone="warning")
            return

        prompt = build_init_agents_prompt(args)
        transcript = "/init" if not args else f"/init {args}"
        self._add_conversation_entry(tui_state.ConversationEntry(kind="user", content=transcript))
        self.agent_worker = self.run_worker(
            self._process_message(prompt), name="kolega-turn", group="turns", exclusive=True
        )

    async def _command_permissions(self, args: str) -> None:
        if self._permission_mode_switch_blocked():
            return

        clean_args = args.strip().lower()
        if not clean_args:
            lines = [
                messages.PERMISSIONS_STATUS.format(mode=self.permission_mode.value),
                messages.PERMISSIONS_SWITCH_HINT,
            ]
            self._add_conversation_entry(tui_state.ConversationEntry(kind="system", content="\n".join(lines)))
            return

        if clean_args == "toggle":
            await self.action_toggle_permission_mode()
            return

        try:
            mode = normalize_permission_mode(clean_args, default=self.permission_mode)
        except ValueError as exc:
            self._notify_user(str(exc), severity="warning")
            return

        await self._set_permission_mode(mode)

    async def _command_model(self, args: str) -> None:
        provider = self.settings.active_provider or UI_DEFAULT_PROVIDER
        model_options = ui_model_options(provider)
        if not args:
            if self._turn_active or self.agent_worker is not None:
                self._show_composer_hint(messages.BLOCK_STOP_BEFORE_MODEL_SWITCH)
                self._notify_user(messages.BLOCK_STOP_BEFORE_MODEL_SWITCH, severity="warning")
                return

            current_provider, current_model = self._startup_model()
            current_effort = self._startup_thinking_effort()
            active_model_line = (
                messages.SETTINGS_ACTIVE_MODEL.format(provider=current_provider, model=current_model)
                if current_model
                else messages.SETTINGS_ACTIVE_MODEL_UNCONFIGURED
            )
            lines = [
                active_model_line,
                messages.SETTINGS_THINKING_EFFORT_LINE.format(effort=current_effort or "not supported"),
                "",
                "Available models:",
                *(f"- `{value}` ({label})" for label, value in model_options),
                "",
                messages.MODEL_SWITCH_HINT,
            ]
            self._add_conversation_entry(tui_state.ConversationEntry(kind="system", content="\n".join(lines)))
            self._pending_model_selection = tui_state.PendingModelSelection(provider=provider, options=model_options)
            self._cancel_pending_effort_selection()
            self._show_model_options()
            self._set_composer_status(messages.MODEL_PLACEHOLDER)
            return

        if self._turn_active or self.agent_worker is not None:
            self._show_composer_hint(messages.BLOCK_STOP_BEFORE_MODEL_SWITCH)
            self._notify_user(messages.BLOCK_STOP_BEFORE_MODEL_SWITCH, severity="warning")
            return

        matched = self._match_model_value(model_options, args)
        if matched is None:
            self._notify_user(messages.MODEL_UNKNOWN.format(model=args, provider=provider), severity="warning")
            return

        await self._switch_model(provider, matched)

    async def _answer_model_option(self, option_index: int) -> None:
        pending = self._pending_model_selection
        if pending is None:
            return
        if option_index < 0 or option_index >= len(pending.options):
            return
        await self._switch_model(pending.provider, pending.options[option_index][1])

    async def _answer_model_selection(self, answer: str) -> None:
        pending = self._pending_model_selection
        if pending is None:
            return

        clean_answer = answer.strip()
        if not clean_answer:
            self._set_composer_status(messages.MODEL_PLACEHOLDER)
            return

        matched = self._match_model_value(pending.options, clean_answer)
        if matched is None:
            self._set_composer_status(messages.MODEL_PLACEHOLDER)
            self._notify_user(
                messages.MODEL_UNKNOWN.format(model=clean_answer, provider=pending.provider),
                severity="warning",
            )
            return

        await self._switch_model(pending.provider, matched)

    async def _switch_model(self, provider: str, model: str) -> None:
        self._cancel_pending_model_selection()
        self._cancel_pending_effort_selection()
        self.settings.active_provider = provider
        self.settings.active_model = model
        self.settings.active_thinking_effort = default_ui_thinking_effort(provider, model)
        self.settings_store.save(self.settings)
        await self._ensure_agent_from_settings(rebuild=True)
        try:
            self._populate_settings_controls()
        except Exception:
            pass
        self._restore_composer_placeholder()
        # Reset the per-session dedup flag so the new model gets a fresh warning.
        self._vision_warning_shown = False
        if self.agent is not None and not self._model_supports_vision():
            conversation = getattr(self.agent, "conversation", None)
            if conversation is not None and conversation.has_image_blocks():
                # Dual-channel: persistent transcript message + ephemeral composer hint.
                self._add_vision_mismatch_system_message(context="model_switch")
                self._show_composer_hint(messages.MODEL_NON_VISION_IMAGE_HISTORY, tone="warning")
        elif self.agent is not None and self._model_supports_vision():
            # Switching to a vision-capable model: clear any stale non-vision warning.
            self._clear_composer_hint()
        self._notify_user(
            messages.MODEL_SWITCHED.format(
                provider=provider,
                model=model,
                effort=self.settings.active_thinking_effort or "not supported",
            )
        )

    def _match_model_value(self, model_options: list[tuple[str, str]], value: str) -> Optional[str]:
        clean_value = value.strip().lower()
        return next((model for _, model in model_options if model.lower() == clean_value), None)

    # Providers the user can sign in to with /login <provider>. Add new targets
    # here as more OAuth integrations land.
    LOGIN_TARGETS: tuple[str, ...] = ("chatgpt",)

    async def _command_login(self, args: str) -> None:
        """Sign in to a provider: ``/login <provider>`` (e.g. ``/login chatgpt``)."""
        target = args.strip().lower()
        targets = ", ".join(self.LOGIN_TARGETS)
        if target == "chatgpt":
            await self._login_chatgpt()
        elif target in ("", "help"):
            self._add_conversation_entry(
                tui_state.ConversationEntry(kind="system", content=messages.LOGIN_USAGE.format(targets=targets))
            )
        else:
            self._notify_user(
                messages.LOGIN_UNKNOWN_TARGET.format(target=target, targets=targets), severity="warning"
            )

    async def _login_chatgpt(self) -> None:
        """Start the browser "Sign in with ChatGPT" flow in a background worker.

        The flow can wait up to a few minutes for the browser round-trip, so it
        runs as a worker to keep the UI responsive.
        """
        if self._turn_active or self.agent_worker is not None:
            self._show_composer_hint(messages.BLOCK_STOP_BEFORE_MODEL_SWITCH)
            self._notify_user(messages.BLOCK_STOP_BEFORE_MODEL_SWITCH, severity="warning")
            return
        self._add_conversation_entry(tui_state.ConversationEntry(kind="system", content=messages.CHATGPT_LOGIN_STARTING))
        self.run_worker(self._do_chatgpt_login(), name="chatgpt-login", group="auth", exclusive=True)

    def _on_login_url(self, url: str) -> None:
        self._add_conversation_entry(
            tui_state.ConversationEntry(kind="system", content=messages.CHATGPT_LOGIN_URL.format(url=url))
        )

    async def _do_chatgpt_login(self) -> None:
        try:
            tokens = await run_login_flow(on_url=self._on_login_url)
        except Exception as exc:  # LoginError / TokenRefreshError / unexpected
            text = messages.CHATGPT_LOGIN_FAILED.format(error=exc)
            self._notify_user(text, severity="error")
            self._add_conversation_entry(tui_state.ConversationEntry(kind="system", content=text, tone="error"))
            return

        self.settings.set_oauth_token(chatgpt_constants.PROVIDER_KEY, tokens.model_dump(mode="json"))
        self.settings_store.save(self.settings)
        self._add_conversation_entry(
            tui_state.ConversationEntry(
                kind="system",
                content=messages.CHATGPT_LOGIN_SUCCESS.format(
                    email=tokens.email or "your ChatGPT account",
                    plan=tokens.plan_type or "subscription",
                ),
            )
        )
        # Switch to the ChatGPT provider so it's usable immediately. The stored
        # token is already saved above, so the agent rebuild inside _switch_model
        # picks it up.
        try:
            await self._switch_model(chatgpt_constants.PROVIDER_KEY, chatgpt_constants.DEFAULT_MODEL)
        except Exception as exc:
            self._notify_user(messages.CHATGPT_LOGIN_SWITCH_FAILED.format(error=exc), severity="warning")

    async def _command_logout(self, args: str) -> None:
        """Sign out of a provider: ``/logout <provider>`` (e.g. ``/logout chatgpt``)."""
        target = args.strip().lower()
        targets = ", ".join(self.LOGIN_TARGETS)
        if target == "chatgpt":
            self._logout_chatgpt()
        elif target in ("", "help"):
            self._add_conversation_entry(
                tui_state.ConversationEntry(kind="system", content=messages.LOGOUT_USAGE.format(targets=targets))
            )
        else:
            self._notify_user(
                messages.LOGOUT_UNKNOWN_TARGET.format(target=target, targets=targets), severity="warning"
            )

    def _logout_chatgpt(self) -> None:
        if not self.settings.has_oauth_token(chatgpt_constants.PROVIDER_KEY):
            self._notify_user(messages.CHATGPT_LOGOUT_NONE, severity="warning")
            return
        self.settings.clear_oauth_token(chatgpt_constants.PROVIDER_KEY)
        self.settings_store.save(self.settings)
        self._notify_user(messages.CHATGPT_LOGOUT_DONE)

    async def _command_effort(self, args: str) -> None:
        provider, model = self._startup_model()
        effort_options = ui_thinking_effort_options(provider, model)
        current_effort = self._startup_thinking_effort()
        if not effort_options:
            self._notify_user(messages.EFFORT_UNSUPPORTED.format(provider=provider, model=model), severity="warning")
            return

        if not args:
            if self._turn_active or self.agent_worker is not None:
                self._show_composer_hint(messages.BLOCK_STOP_BEFORE_EFFORT_SWITCH)
                self._notify_user(messages.BLOCK_STOP_BEFORE_EFFORT_SWITCH, severity="warning")
                return

            active_model_line = (
                messages.SETTINGS_ACTIVE_MODEL.format(provider=provider, model=model)
                if model
                else messages.SETTINGS_ACTIVE_MODEL_UNCONFIGURED
            )
            lines = [
                active_model_line,
                messages.SETTINGS_THINKING_EFFORT_LINE.format(effort=current_effort or "not supported"),
                "",
                "Available thinking efforts:",
                *(f"- `{value}` ({label})" for label, value in effort_options),
                "",
                messages.EFFORT_SWITCH_HINT,
            ]
            self._add_conversation_entry(tui_state.ConversationEntry(kind="system", content="\n".join(lines)))
            self._pending_effort_selection = tui_state.PendingEffortSelection(
                provider=provider,
                model=model,
                options=effort_options,
            )
            self._show_effort_options()
            self._set_composer_status(messages.EFFORT_PLACEHOLDER)
            return

        if self._turn_active or self.agent_worker is not None:
            self._show_composer_hint(messages.BLOCK_STOP_BEFORE_EFFORT_SWITCH)
            self._notify_user(messages.BLOCK_STOP_BEFORE_EFFORT_SWITCH, severity="warning")
            return

        matched = self._match_effort_value(effort_options, args)
        if matched is None:
            self._notify_user(
                messages.EFFORT_UNKNOWN.format(effort=args, provider=provider, model=model),
                severity="warning",
            )
            return

        self._cancel_pending_effort_selection()
        await self._switch_thinking_effort(provider, model, matched)

    async def _answer_effort_option(self, option_index: int) -> None:
        pending = self._pending_effort_selection
        if pending is None:
            return
        if option_index < 0 or option_index >= len(pending.options):
            return
        await self._switch_thinking_effort(pending.provider, pending.model, pending.options[option_index][1])

    async def _answer_effort_selection(self, answer: str) -> None:
        pending = self._pending_effort_selection
        if pending is None:
            return

        clean_answer = answer.strip()
        if not clean_answer:
            self._set_composer_status(messages.EFFORT_PLACEHOLDER)
            return

        matched = self._match_effort_value(pending.options, clean_answer)
        if matched is None:
            self._set_composer_status(messages.EFFORT_PLACEHOLDER)
            self._notify_user(
                messages.EFFORT_UNKNOWN.format(
                    effort=clean_answer,
                    provider=pending.provider,
                    model=pending.model,
                ),
                severity="warning",
            )
            return

        await self._switch_thinking_effort(pending.provider, pending.model, matched)

    async def _switch_thinking_effort(self, provider: str, model: str, effort: str) -> None:
        self._cancel_pending_effort_selection()
        self.settings.active_provider = provider
        self.settings.active_model = model
        self.settings.active_thinking_effort = effort
        self.settings_store.save(self.settings)
        await self._ensure_agent_from_settings(rebuild=True)
        try:
            self._populate_settings_controls()
        except Exception:
            pass
        self._restore_composer_placeholder()
        self._notify_user(messages.EFFORT_SWITCHED.format(effort=effort, provider=provider, model=model))

    def _match_effort_value(self, effort_options: list[tuple[str, str]], value: str) -> Optional[str]:
        clean_value = value.strip().lower()
        return next((effort for _, effort in effort_options if effort.lower() == clean_value), None)

    async def _command_theme(self, args: str) -> None:
        if not args:
            current = self.settings.active_theme or theme.DEFAULT_THEME_NAME
            lines = [
                messages.SETTINGS_ACTIVE_THEME.format(theme=current),
                "",
                "Available themes:",
                *(f"- `{name}`" for name in theme.available_themes()),
                "",
                messages.THEME_SWITCH_HINT,
            ]
            self._add_conversation_entry(tui_state.ConversationEntry(kind="system", content="\n".join(lines)))
            self._pending_theme_selection = tui_state.PendingThemeSelection(
                options=[(name, name) for name in theme.available_themes()]
            )
            self._show_theme_options()
            self._set_composer_status(messages.THEME_PLACEHOLDER)
            return

        matched = self._match_theme_value(args)
        if matched is None:
            self._notify_user(messages.THEME_UNKNOWN.format(theme=args), severity="warning")
            return
        await self._switch_theme(matched)

    async def _answer_theme_option(self, option_index: int) -> None:
        pending = self._pending_theme_selection
        if pending is None:
            return
        if option_index < 0 or option_index >= len(pending.options):
            return
        await self._switch_theme(pending.options[option_index][1])

    async def _answer_theme_selection(self, answer: str) -> None:
        pending = self._pending_theme_selection
        if pending is None:
            return
        clean_answer = answer.strip()
        if not clean_answer:
            self._set_composer_status(messages.THEME_PLACEHOLDER)
            return
        matched = self._match_theme_value(clean_answer)
        if matched is None:
            self._set_composer_status(messages.THEME_PLACEHOLDER)
            self._notify_user(messages.THEME_UNKNOWN.format(theme=clean_answer), severity="warning")
            return
        await self._switch_theme(matched)

    async def _switch_theme(self, name: str) -> None:
        self._cancel_pending_theme_selection()
        self.settings.active_theme = name
        self.settings_store.save(self.settings)
        self._apply_theme(name)
        try:
            self._populate_settings_controls()
        except Exception:
            pass
        self._restore_composer_placeholder()
        self._notify_user(messages.THEME_SWITCHED.format(theme=name))

    def _match_theme_value(self, value: str) -> Optional[str]:
        clean_value = value.strip().lower()
        return next((name for name in theme.available_themes() if name.lower() == clean_value), None)

    def _apply_theme(self, name: Optional[str]) -> None:
        """Apply a theme live: swap Rich roles + Textual CSS, then re-skin the UI."""
        theme.apply_theme(name)
        try:
            self.theme = theme.textual_theme_name(name)
        except Exception:
            pass
        # Already-mounted Rich renderables baked in the old Color strings; rebuild
        # the conversation and dashboard so they pick up the new palette.
        self._render_conversation()
        self._refresh_status_dashboard()

    async def _command_copy(self, args: str) -> None:
        entry = next(
            (entry for entry in reversed(self.conversation_entries) if entry.kind == "assistant" and entry.content),
            None,
        )
        if entry is None:
            self._notify_user(messages.COPY_NOTHING, severity="warning")
            return
        self.copy_to_clipboard(entry.content)
        self._notify_user(messages.COPY_LAST_RESPONSE)

    async def _command_version(self, args: str) -> None:
        result = await asyncio.to_thread(check_for_update)
        lines = [messages.VERSION_INFO.format(version=result.current_version)]
        update_message = update_status_message(result, include_up_to_date=True, include_errors=True)
        if update_message:
            lines.append(update_message)
        self._add_conversation_entry(tui_state.ConversationEntry(kind="system", content="\n".join(lines)))

    async def _command_update(self, args: str) -> None:
        if self._turn_active or self.agent_worker is not None:
            self._show_composer_hint(messages.BLOCK_STOP_BEFORE_UPDATE)
            self._notify_user(messages.BLOCK_STOP_BEFORE_UPDATE, severity="warning")
            return

        self._notify_user(messages.UPDATE_STARTED)
        result = await asyncio.to_thread(run_self_update, capture_output=True)
        severity = "information" if result.returncode == 0 else "error"
        if result.returncode == 0:
            lines = [messages.UPDATE_COMPLETED]
        else:
            lines = [messages.UPDATE_FAILED.format(code=result.returncode)]
            if result.error:
                lines.append(result.error)

        output = "\n".join(part for part in (result.stdout.strip(), result.stderr.strip()) if part)
        if output:
            if len(output) > 4000:
                output = "[output truncated]\n" + output[-4000:]
            lines.extend(["", "Output:", "```text", output, "```"])

        content = "\n".join(lines)
        self._add_conversation_entry(tui_state.ConversationEntry(kind="system", content=content))
        self._notify_user(lines[0], severity=severity)

    async def _command_quit(self, args: str) -> None:
        await self.action_quit()

    async def _handle_skill_slash_command(self, stripped_text: str, composer: tui_widgets.ChatComposer) -> bool:
        command = self._parse_skill_slash_command(stripped_text)
        if command is None:
            return False

        command_name, prompt = command
        composer.load_text("")

        if command_name == "skills":
            self._add_conversation_entry(tui_state.ConversationEntry(kind="system", content=self.skill_catalog.format_catalog()))
            self._log_status(messages.SKILLS_LISTED, "ok")
            return True

        if self._pending_question is not None:
            self._set_composer_status(messages.QUESTION_PLACEHOLDER)
            self._notify_user(messages.BLOCK_PENDING_QUESTION_SKILL, severity="warning")
            return True

        if self._pending_approval is not None:
            self._set_composer_status(messages.APPROVAL_PLACEHOLDER)
            self._notify_user(messages.BLOCK_PENDING_APPROVAL, severity="warning")
            return True

        if self._plan_decision_active:
            self._set_composer_status(messages.PLAN_READY_PLACEHOLDER)
            self._notify_user(messages.BLOCK_PLAN_DECISION_SKILL, severity="warning")
            return True

        if self._turn_active or self.agent_worker is not None:
            self._show_composer_hint(messages.BLOCK_STOP_BEFORE_SKILL)
            self._notify_user(messages.BLOCK_STOP_BEFORE_SKILL, severity="warning")
            return True

        if self.agent is None:
            self._set_settings_status(messages.SETTINGS_REQUIRED_SKILL, tone="warning")
            return True

        activated = self._activate_skill_in_agent(command_name)
        self._add_conversation_entry(tui_state.ConversationEntry(kind="skill", content=activated))
        self._notify_user(messages.SKILL_ACTIVATED.format(name=command_name))

        if prompt:
            attachments = self._build_mention_attachments(prompt)
            self._add_conversation_entry(tui_state.ConversationEntry(kind="user", content=prompt))
            self.agent_worker = self.run_worker(
                self._process_message(prompt, attachments), name="kolega-turn", group="turns", exclusive=True
            )
        else:
            self._save_session_history()
            self._restore_composer_placeholder()
            self._set_chat_enabled(True)

        return True

    def _parse_skill_slash_command(self, stripped_text: str) -> Optional[tuple[str, str]]:
        if not stripped_text.startswith("/"):
            return None

        command_text, _, prompt = stripped_text.partition(" ")
        command = command_text.lower()
        if command == SKILLS_LIST_COMMAND:
            return "skills", prompt.strip()
        if command in agent_command_names() or command in TUI_COMMAND_NAMES:
            return None

        skill_name = command.removeprefix("/")
        if self.skill_catalog.get(skill_name) is None:
            return None

        return skill_name, prompt.strip()

    def _activate_skill_in_agent(self, skill_name: str) -> str:
        if self.agent is None:
            raise RuntimeError("Cannot activate a skill before an agent exists.")

        active_names = activated_skill_names(self.agent.history)
        content = self.skill_catalog.activation_content(skill_name, active_names=active_names)
        if skill_name not in active_names:
            self.agent.append_user_message([TextBlock(text=content)])
        return content


    def _clear_agent_context(self) -> None:
        """Wipe the agent's LLM history so the build agent starts fresh, while leaving
        the visible transcript and the captured plan intact."""
        if self.agent is not None:
            self.agent.history = MessageHistory()
            self.agent.last_compression_index = None
        self.session.history = []
        self.session.compaction = {}

    def _reset_current_thread(self) -> None:
        self._close_sub_agent_inspector()
        if self.agent is not None:
            self.agent.history = MessageHistory()
        self.session.history = []
        self.session.compaction = {}
        self.session.task_list_markdown = ""
        self.conversation_entries = []
        self._stream_entries = {}
        self._tool_entries = {}
        self._tool_stream_buffers = {}
        self._sub_agent_activities = {}
        self._sub_agent_by_tool_call = {}
        self._sub_agent_seq = 0
        self._workflow_activities = {}
        self._active_progress_entry = None
        self._latest_plan = None
        self._plan_pending = False
        self._plan_reofferable = False
        self._plan_decision_active = False
        self._save_session()
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
        self._add_conversation_entry(tui_state.ConversationEntry(kind="progress", content=messages.THREAD_RESET_MESSAGE, complete=True))
        self._notify_user(messages.THREAD_RESET_MESSAGE)

    def _set_settings_status(self, text: str, tone: str = "info") -> None:
        """Update the settings status with a tone glyph in the semantic palette."""
        glyph, style = {
            "ok": (Glyph.CHECK, Color.SUCCESS),
            "error": (Glyph.CROSS, Color.ERROR),
            "warning": (Glyph.STATUS, Color.WARNING),
        }.get(tone, (Glyph.STATUS, Color.MUTED))
        content = Text()
        content.append(theme.g(glyph) + " ", style=style)
        content.append(text)
        try:
            self._settings_status.update(content)
        except Exception:
            return

    def _update_settings_status(self) -> None:
        if not (self.settings.active_provider and self.settings.active_model):
            text = "\n".join(
                [
                    messages.SETTINGS_ACTIVE_MODEL_UNCONFIGURED,
                    messages.SETTINGS_THINKING_EFFORT_LINE.format(effort="not configured"),
                    messages.SETTINGS_API_KEY_LINE.format(status="not checked until a model is configured"),
                ]
            )
            self._set_settings_status(text, "warning")
            self._refresh_status_dashboard()
            return

        provider = self.settings.active_provider
        model = self.settings.active_model
        effort = self.settings.active_thinking_effort or default_ui_thinking_effort(provider, model) or "not supported"
        status = key_status(provider, self.project_path, self.settings)
        tone = "warning" if "missing" in status.lower() else "ok"
        text = "\n".join(
            [
                messages.SETTINGS_ACTIVE_MODEL.format(provider=provider, model=model),
                messages.SETTINGS_THINKING_EFFORT_LINE.format(effort=effort),
                messages.SETTINGS_API_KEY_LINE.format(status=status),
            ]
        )
        self._set_settings_status(text, tone)
        self._refresh_status_dashboard()

    def _api_key_placeholder(self, provider: str) -> str:
        if provider == chatgpt_constants.PROVIDER_KEY:
            # OAuth provider: no API key — the field is informational only.
            if self.settings.has_oauth_token(provider):
                return "Signed in with ChatGPT — run /login chatgpt to switch accounts"
            return "Run /login chatgpt to sign in with your ChatGPT subscription"
        if self.settings.has_api_key(provider):
            return "Stored API key will be kept if blank"
        model = get_ui_model(provider, (ui_model_options(provider) or [("", "")])[0][1])
        return f"{model.provider_label} API key" if model else "API key"

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
            self.conversation_entries.insert(0, tui_state.ConversationEntry(kind="startup", content=self._startup_content()))
        else:
            existing.content = self._startup_content()
            if self.conversation_entries[0] is not existing:
                self.conversation_entries.remove(existing)
                self.conversation_entries.insert(0, existing)
        if render:
            self._render_conversation()

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
        return "\n".join(
            [
                *tui_constants.STARTUP_WORDMARK,
                "",
                f"Project: {self.project_path}",
                f"Session: {session_id}",
                f"Mode: {self.mode}",
                f"Interaction: {self.interaction_mode}",
                f"Permissions: {self.permission_mode.value}",
                f"Model: {model_display}",
                f"Thinking effort: {effort}",
                f"API key: {api_key}",
                "",
                f"Enter send {theme.g(Glyph.BULLET_SEP)} Shift+Enter newline {theme.g(Glyph.BULLET_SEP)} Shift+Tab plan/build {theme.g(Glyph.BULLET_SEP)} Ctrl+P permissions {theme.g(Glyph.BULLET_SEP)} Ctrl+O sidebar",
                f"Ctrl+C stop turn {theme.g(Glyph.BULLET_SEP)} Cmd+C copy selection {theme.g(Glyph.BULLET_SEP)} / commands",
            ]
        )

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

    def _refresh_status_dashboard(self) -> None:
        provider, model = self._startup_model()
        self._status_state.provider = provider
        self._status_state.model = model
        self._status_state.thinking_effort = self._startup_thinking_effort()
        self._status_state.mode = self.interaction_mode
        self._status_state.permission_mode = self.permission_mode.value
        try:
            self._status_dashboard.update(self._format_status_dashboard())
        except Exception:
            return

    def _format_status_dashboard(self) -> str:
        state = self._status_state
        provider_model = f"{state.provider}/{state.model}" if state.model else state.provider
        effort = state.thinking_effort or "not supported"
        mode = state.mode.title()
        permission_mode = state.permission_mode.title()
        turn_style = tui_state.turn_state_color(state.turn_state)
        context_style = self._context_style(state.usage_percentage, state.compression_threshold)

        def label(text: str) -> str:
            return theme.styled(text, Color.MUTED)

        if state.usage_percentage is None:
            context_lines = theme.styled("Waiting for first context count", Color.MUTED)
        else:
            percentage = f"{state.usage_percentage:.1f}%"
            token_line = self._context_token_line(state.input_tokens, state.max_tokens)
            threshold = self._compression_threshold_line(state.compression_threshold)
            context_lines = (
                f"[{context_style}]{self._context_bar(state.usage_percentage)}[/] "
                f"[bold {context_style}]{percentage}[/]\n"
                f"{token_line}\n"
                f"{theme.styled(threshold, Color.MUTED)}"
            )
            if state.context_note:
                note_style = self._context_note_style(state.alert_level)
                context_lines += f"\n[{note_style}]{escape(state.context_note)}[/{note_style}]"

        if state.is_compacting:
            indicator = escape(state.compaction_message or messages.COMPACTING)
            context_lines += f"\n[{Color.ACCENT}]{theme.g(Glyph.RUNNING)} {indicator}[/{Color.ACCENT}]"

        title = theme.role_header(Glyph.STATUS, "Status", Color.ACCENT)
        turn_line = (
            f"{label('Turn')} [{turn_style}]{theme.g(Glyph.STATUS)}[/{turn_style}] "
            f"[bold]{escape(state.turn_state.value)}[/bold]"
        )
        return (
            f"{title}\n\n"
            f"{label('Model')}\n[bold]{escape(provider_model)}[/bold]\n\n"
            f"{label('Thinking effort')} [bold]{escape(effort)}[/bold]\n"
            f"{label('Mode')} [bold]{mode}[/bold]\n"
            f"{label('Permissions')} [bold]{permission_mode}[/bold]\n"
            f"{turn_line}\n\n"
            f"{label('Context')}\n"
            f"{context_lines}\n\n"
            f"{label('Activity')}\n"
            f"{escape(state.activity)}"
        )

    def _context_bar(self, usage_percentage: float) -> str:
        return theme.context_bar(usage_percentage)

    def _context_token_line(self, input_tokens: Optional[int], max_tokens: Optional[int]) -> str:
        if input_tokens is None or max_tokens is None:
            return theme.styled(messages.STATUS_TOKENS_UNKNOWN, Color.MUTED)
        return f"Tokens: {input_tokens:,} / {max_tokens:,}"

    def _compression_threshold_line(self, compression_threshold: Optional[float]) -> str:
        if compression_threshold is None:
            return "Compression threshold unknown"
        return f"Compresses at {compression_threshold:.0f}%"

    def _context_style(self, usage_percentage: Optional[float], compression_threshold: Optional[float]) -> str:
        if usage_percentage is None:
            return Color.SUCCESS
        if compression_threshold is not None and usage_percentage >= compression_threshold:
            return Color.ERROR
        if usage_percentage >= 60:
            return Color.WARNING
        return Color.SUCCESS

    def _context_note_style(self, alert_level: str) -> str:
        if alert_level.lower() in {"error", "critical"}:
            return Color.ERROR
        return Color.WARNING

    def _set_status_activity(self, content: str, *, turn_state: Optional[tui_state.TurnState] = None) -> None:
        if content:
            self._status_state.activity = content
        if turn_state is not None:
            self._status_state.turn_state = turn_state
        self._refresh_status_dashboard()

    def _apply_compaction_status(self, content: dict) -> None:
        """Toggle the 'compaction in progress' indicator and, on finish, drop the
        summary into the transcript as a collapsible the user can expand."""
        phase = str(content.get("phase") or "")
        if phase == "started":
            self._status_state.is_compacting = True
            message = content.get("message")
            self._status_state.compaction_message = (
                message if isinstance(message, str) and message else messages.COMPACTING
            )
        else:  # "finished" | "error"
            self._status_state.is_compacting = False
            self._status_state.compaction_message = ""
            if phase == "finished":
                summary = content.get("summary")
                if isinstance(summary, str) and summary.strip():
                    self._add_conversation_entry(
                        tui_state.ConversationEntry(kind="compaction_summary", content=summary.strip())
                    )
        self._refresh_status_dashboard()

    def _apply_context_status_update(self, content: dict) -> None:
        self._status_state.input_tokens = self._as_optional_int(content.get("input_tokens"))
        self._status_state.max_tokens = self._as_optional_int(content.get("max_tokens"))
        self._status_state.usage_percentage = self._as_optional_float(content.get("usage_percentage"))
        self._status_state.compression_threshold = self._as_optional_float(content.get("compression_threshold"))
        self._status_state.alert_level = str(content.get("alert_level") or "normal")
        message = content.get("message")
        self._status_state.context_note = message if isinstance(message, str) else ""
        self._refresh_status_dashboard()

    def _display_text_from_event(self, event: AgentEvent) -> str:
        for key in ("text", "message"):
            value = event.content.get(key)
            if isinstance(value, str):
                return value
        return ""

    def _as_optional_int(self, value: object) -> Optional[int]:
        try:
            return int(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    def _as_optional_float(self, value: object) -> Optional[float]:
        try:
            return float(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    def _now(self) -> float:
        return time.monotonic()

    def _start_turn_timer(self, status_text: str) -> None:
        if self._turn_timer is not None:
            self._turn_timer.stop()
        self._turn_started_at = self._now()
        self._turn_finished_duration = None
        self._turn_status_text = status_text
        self._turn_final_text = ""
        self._turn_final_state = tui_state.TurnState.IDLE
        self._spinner_frame = 0
        self._turn_timer = self.set_interval(
            theme.SPINNER_INTERVAL, self._refresh_turn_status_strip, name="turn-status"
        )
        self._refresh_turn_status_strip()

    def _complete_turn_timer(self, content: str, state: tui_state.TurnState = tui_state.TurnState.IDLE) -> None:
        if self._turn_timer is not None:
            self._turn_timer.stop()
            self._turn_timer = None
        if self._turn_started_at is None:
            return

        self._turn_finished_duration = max(0.0, self._now() - self._turn_started_at)
        duration = self._format_turn_duration(self._turn_finished_duration)
        self._turn_final_state = state
        if state is tui_state.TurnState.ERROR:
            self._turn_final_text = messages.ERRORED_AFTER.format(duration=duration)
        elif state in {tui_state.TurnState.STOPPED, tui_state.TurnState.STOPPING}:
            self._turn_final_text = messages.STOPPED_AFTER.format(duration=duration)
        else:
            self._turn_final_text = messages.DONE_IN.format(duration=duration)
        self._turn_started_at = None
        self._refresh_turn_status_strip()

    def _clear_turn_status_strip(self) -> None:
        if self._turn_timer is not None:
            self._turn_timer.stop()
            self._turn_timer = None
        self._turn_started_at = None
        self._turn_finished_duration = None
        self._turn_status_text = ""
        self._turn_final_text = ""
        self._turn_final_state = tui_state.TurnState.IDLE
        self._refresh_turn_status_strip()

    def _refresh_turn_status_strip(self) -> None:
        try:
            strip = self._turn_status
        except Exception:
            return

        self._spinner_frame += 1
        content = self._turn_status_content()
        strip.display = bool(content)
        strip.update(content)
        # Tick elapsed time on running sub-agents at most once per second so the
        # faster spinner cadence only touches this cheap status strip.
        now = self._now()
        if now - self._last_sub_agent_tick >= 1.0:
            self._last_sub_agent_tick = now
            self._tick_running_sub_agents()
            self._tick_running_workflows()

    def _turn_status_content(self) -> str:
        if self._turn_started_at is not None:
            elapsed = max(0.0, self._now() - self._turn_started_at)
            status = self._turn_status_text or messages.WORKING
            frames = theme.spinner_frames()
            frame = frames[self._spinner_frame % len(frames)]
            return (
                f"[{Color.ACCENT}]{frame}[/{Color.ACCENT}] {escape(status)} "
                f"[dim]{theme.g(Glyph.BULLET_SEP)} {self._format_turn_duration(elapsed)}[/dim]"
            )
        if self._turn_final_text:
            if self._turn_final_state is tui_state.TurnState.ERROR:
                glyph, color = Glyph.CROSS, Color.ERROR
            elif self._turn_final_state in {tui_state.TurnState.STOPPED, tui_state.TurnState.STOPPING}:
                glyph, color = Glyph.CROSS, Color.WARNING
            else:
                glyph, color = Glyph.CHECK, Color.SUCCESS
            return f"[{color}]{theme.g(glyph)}[/{color}] {escape(self._turn_final_text)}"
        return ""

    def _format_turn_duration(self, seconds: float) -> str:
        total_seconds = max(0, int(seconds))
        minutes, remaining_seconds = divmod(total_seconds, 60)
        if minutes:
            return f"{minutes}m {remaining_seconds:02d}s"
        return f"{remaining_seconds}s"

