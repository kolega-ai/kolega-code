"""Textual application for Kolega Code."""

from __future__ import annotations

import asyncio
import itertools
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Awaitable, Callable, Optional

from rich.console import Group
from rich.markdown import Markdown as RichMarkdown
from rich.markup import escape
from rich.padding import Padding
from rich.segment import Segment
from rich.style import Style
from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.message import Message as TextualMessage
from textual.selection import Selection
from textual.strip import Strip
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
    TabPane,
    TabbedContent,
    TextArea,
)

from kolega_code import __version__ as kolega_code_version
from kolega_code.agent import AgentConfig, AgentEvent, CoderAgent, PlanningAgent, PromptExtension, ToolExtension
from kolega_code.agent.llm.models import Message, MessageHistory, TextBlock, ToolCall, ToolResult
from kolega_code.agent.prompt_provider import AgentMode
from kolega_code.agent.services.browser import PlaywrightBrowserManager

from . import messages
from . import theme
from .config import CliConfigError, CliConfigOverrides, build_agent_config, config_summary, key_status
from .connection import CliConnectionManager
from .file_index import IndexEntry, WorkspaceFileIndex
from .mentions import build_file_attachments
from .theme import Color, Glyph
from .provider_registry import UI_DEFAULT_MODEL, UI_DEFAULT_PROVIDER, get_ui_model, ui_model_options, ui_provider_options
from .session_store import SessionRecord, SessionStore
from .settings import CliSettings, SettingsStore
from .slash_commands import (
    SKILLS_LIST_COMMAND,
    THREAD_RESET_COMMANDS,
    TUI_COMMAND_NAMES,
    SlashCommandEntry,
    agent_command_names,
    search_commands,
)
from .skills import (
    SkillCatalog,
    activated_skill_names,
    build_skill_prompt_extension,
    build_skill_tool_extension,
    discover_skills,
    skill_names_in_text,
)

# Re-exported from theme/messages so existing importers (including tests) keep working.
TOOL_RESULT_PREVIEW_CHARS = theme.TOOL_RESULT_PREVIEW_CHARS
TOOL_STREAM_PREVIEW_CHARS = theme.TOOL_STREAM_PREVIEW_CHARS
SUB_AGENT_TAIL_CHARS = theme.SUB_AGENT_TAIL_CHARS
SUB_AGENT_TASK_PREVIEW_CHARS = theme.SUB_AGENT_TASK_PREVIEW_CHARS
COMPOSER_PLACEHOLDER = messages.COMPOSER_PLACEHOLDER
PLAN_READY_PLACEHOLDER = messages.PLAN_READY_PLACEHOLDER
THREAD_RESET_MESSAGE = messages.THREAD_RESET_MESSAGE
TASK_LIST_EMPTY_MESSAGE = messages.TASK_LIST_EMPTY_MESSAGE
PLAN_EMPTY_MESSAGE = messages.PLAN_EMPTY_MESSAGE
CLI_AGENT_MODE = AgentMode.CLI.value
BUILD_INTERACTION_MODE = "build"
PLAN_INTERACTION_MODE = "plan"
SHARED_TASK_LIST_PROMPT = """The CLI provides a shared Markdown task list through `get_task_list` and `update_task_list`.
Use it to coordinate planning and implementation.

In planning mode, create or update the task list before calling `write_plan`.
In build mode, call `get_task_list` when a shared task list exists or when implementing an approved plan.
After each meaningful task is completed, call `update_task_list` to check off that item by rewriting the full Markdown list.
Do not wait until every TODO is complete to update the shared task list."""
PLANNING_QUESTION_PROMPT = """The CLI provides `ask_user_choice` for important multiple-choice planning decisions.
Use it only when a decision materially changes the plan. Provide concise options; the user can also type a custom answer."""
IMPLEMENT_PLAN_PROMPT = """Implement the approved plan below. Follow it as the source of truth, but still inspect the code before editing and run appropriate checks.

{plan}
"""
QUESTION_TOOL_NAME = "ask_user_choice"
QUESTION_OPTION_BUTTON_PREFIX = "question_option_"
QUESTION_PLACEHOLDER = messages.QUESTION_PLACEHOLDER
STARTUP_WORDMARK = (
    " _  __     _                    ____          _",
    "| |/ /___ | | ___  __ _  __ _ / ___|___   __| | ___",
    "| ' // _ \\| |/ _ \\/ _` |/ _` | |   / _ \\ / _` |/ _ \\",
    "| . \\ (_) | |  __/ (_| | (_| | |__| (_) | (_| |  __/",
    "|_|\\_\\___/|_|\\___|\\__, |\\__,_|\\____\\___/ \\__,_|\\___|",
    "                  |___/",
)


class TurnState(str, Enum):
    """Explicit lifecycle state of the active turn, shown on the status dashboard."""

    IDLE = "Idle"
    GENERATING = "Generating"
    THINKING = "Thinking"
    RUNNING_TOOL = "Running tool"
    RUNNING_SUB_AGENTS = "Running sub-agents"
    WAITING_FOR_USER = "Waiting for input"
    STOPPING = "Stopping"
    STOPPED = "Stopped"
    ERROR = "Error"


TURN_STATE_STYLES = {
    TurnState.IDLE: Color.SUCCESS,
    TurnState.STOPPING: Color.WARNING,
    TurnState.STOPPED: Color.WARNING,
    TurnState.ERROR: Color.ERROR,
}

TOOL_STATE_PRESENTATION = {
    "tool_call": ("running", Color.ACCENT),
    "tool_result": ("done", Color.SUCCESS),
    "tool_error": ("failed", Color.ERROR),
}

TAB_BASE_LABELS = {
    "logs_pane": "Logs",
    "terminal_pane": "Terminal",
}


_ENTRY_ID_COUNTER = itertools.count(1)


def _next_entry_id() -> str:
    return f"entry-{next(_ENTRY_ID_COUNTER)}"


@dataclass
class ConversationEntry:
    kind: str
    content: str
    complete: bool = True
    uuid: Optional[str] = None
    tool_name: Optional[str] = None
    tool_call_id: Optional[str] = None
    tone: Optional[str] = None  # "warning" | "error" styling hint for progress entries
    full_content: str = ""  # untruncated tool output for expand-on-demand (capped)
    entry_id: str = field(default_factory=_next_entry_id)  # UI-only widget key, not persisted


@dataclass
class SubAgentActivity:
    """Live display state for one dispatched sub-agent."""

    agent_id: str
    agent_name: str
    task: str
    index: int  # display ordinal within the turn: #1, #2, ...
    entry: ConversationEntry  # kind="sub_agent", updated in place
    status: str = "running"  # running | completed | failed | stopped
    tool_calls: int = 0
    last_activity: str = ""
    started_at: float = 0.0
    finished_at: Optional[float] = None
    stream_buffers: dict[str, str] = field(default_factory=dict)  # chunk uuid -> accumulated text
    active_stream_uuid: Optional[str] = None


@dataclass
class PendingQuestion:
    question: str
    options: list[str]
    future: asyncio.Future[str]


@dataclass
class StatusDashboardState:
    provider: str = UI_DEFAULT_PROVIDER
    model: str = UI_DEFAULT_MODEL
    mode: str = BUILD_INTERACTION_MODE
    turn_state: TurnState = TurnState.IDLE
    activity: str = "Ready"
    input_tokens: Optional[int] = None
    max_tokens: Optional[int] = None
    usage_percentage: Optional[float] = None
    compression_threshold: Optional[float] = None
    alert_level: str = "normal"
    context_note: str = ""


class ConversationEntryWidget(Static):
    """Displays one ConversationEntry and is updated in place as the entry changes."""

    def __init__(self, entry: ConversationEntry, format_entry: Callable[[ConversationEntry], object]) -> None:
        super().__init__("")
        self.entry = entry
        self._format_entry = format_entry
        self._kind_class = ""
        self._formatted: object = None
        self.refresh_content()

    def refresh_content(self) -> None:
        kind_class = f"entry-{self.entry.kind}"
        if kind_class != self._kind_class:
            if self._kind_class:
                self.remove_class(self._kind_class)
            self.add_class(kind_class)
            self._kind_class = kind_class
        self._formatted = self._format_entry(self.entry)
        self.update(self._formatted)

    def render_line(self, y: int) -> Strip:
        # Tag each segment with its rendered (x, y) offset so the compositor can
        # map mouse positions to text offsets, enabling drag selection over any
        # visual type, including rich renderables such as Markdown.
        strip = super().render_line(y)
        source_x = 0
        selectable_segments: list[Segment] = []
        for segment in strip:
            if segment.control:
                selectable_segments.append(segment)
                continue
            offset_style = Style.from_meta({"offset": (source_x, y)})
            style = segment.style + offset_style if segment.style is not None else offset_style
            selectable_segments.append(Segment(segment.text, style, segment.control))
            source_x += len(segment.text)
        return Strip(selectable_segments, strip.cell_length)

    def get_selection(self, selection: Selection) -> tuple[str, str] | None:
        # Extract from the rendered lines so coordinates match what is on screen,
        # regardless of whether the entry is plain markup or a rich renderable.
        height = self.size.height
        if height <= 0:
            return None
        lines = [super(ConversationEntryWidget, self).render_line(y).text.rstrip() for y in range(height)]
        text = "\n".join(lines)
        if not text.strip():
            return None
        return selection.extract(text), "\n"


class ToolEntryWidget(Vertical):
    """Tool entry rendered as a collapsed-by-default Collapsible with the full output inside."""

    def __init__(self, entry: ConversationEntry, title_factory: Callable[[ConversationEntry], str]) -> None:
        super().__init__()
        self.entry = entry
        self._title_factory = title_factory
        self._collapsible: Optional[Collapsible] = None
        self._body: Optional[Static] = None

    def compose(self) -> ComposeResult:
        self._body = Static("", markup=False, classes="tool-body")
        self._collapsible = Collapsible(self._body, title=self._title_factory(self.entry), collapsed=True)
        yield self._collapsible

    def on_mount(self) -> None:
        self.refresh_content()

    def refresh_content(self) -> None:
        if self._collapsible is None or self._body is None:
            return
        self._collapsible.title = self._title_factory(self.entry)
        self._body.update(self.entry.full_content or self.entry.content)


class ConversationView(VerticalScroll):
    """Scrollable list of per-entry widgets, anchored to the bottom while streaming."""

    def watch_scroll_y(self, old_value: float, new_value: float) -> None:
        super().watch_scroll_y(old_value, new_value)
        try:
            update = getattr(self.app, "_update_jump_button", None)
        except Exception:
            return
        if update is not None:
            update()


class JumpToBottomBar(Static):
    """One-line affordance shown when the conversation is scrolled away from the end."""

    @dataclass
    class Pressed(TextualMessage):
        bar: JumpToBottomBar

    def on_click(self) -> None:
        self.post_message(self.Pressed(self))


@dataclass(frozen=True)
class CompletionItem:
    """One row in the completion dropdown: a display prompt plus the value it completes to."""

    prompt: Text | str
    value: IndexEntry | SlashCommandEntry


def file_completion_item(entry: IndexEntry) -> CompletionItem:
    return CompletionItem(prompt=entry.path, value=entry)


def command_completion_item(entry: SlashCommandEntry) -> CompletionItem:
    prompt = Text.assemble((entry.token, "bold"), "  ", (entry.description, "dim"))
    return CompletionItem(prompt=prompt, value=entry)


class CompletionDropdown(OptionList):
    """Completion list shown above the composer for @ file mentions and / commands."""

    can_focus = False

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._items: list[CompletionItem] = []

    @property
    def is_open(self) -> bool:
        return self.display

    def open_with(self, items: list[CompletionItem]) -> None:
        self._items = list(items)
        self.clear_options()
        self.add_options([item.prompt for item in self._items])
        if self._items:
            self.highlighted = 0
        self.display = True

    def close(self) -> None:
        self.display = False
        self._items = []
        self.clear_options()

    def highlighted_entry(self) -> Optional[IndexEntry | SlashCommandEntry]:
        if self.highlighted is None or not self._items:
            return None
        if 0 <= self.highlighted < len(self._items):
            return self._items[self.highlighted].value
        return None

    def entry_at(self, index: int) -> Optional[IndexEntry | SlashCommandEntry]:
        if 0 <= index < len(self._items):
            return self._items[index].value
        return None


class ChatComposer(TextArea):
    """Multiline chat input that submits on Enter and inserts newlines on Shift+Enter."""

    BINDINGS = [
        *TextArea.BINDINGS,
        Binding("enter", "submit", "Send", priority=True),
        Binding("shift+enter,ctrl+enter,ctrl+j", "insert_newline", "New line", key_display="Shift+Enter", priority=True),
        Binding("up", "mention_prev", "Previous match", show=False, priority=True),
        Binding("down", "mention_next", "Next match", show=False, priority=True),
        Binding("tab", "mention_accept", "Complete path", show=False, priority=True),
        Binding("escape", "mention_dismiss", "Dismiss matches", show=False, priority=True),
    ]

    MENTION_QUERY_RE = re.compile(r"(?:^|(?<=\s))@(\S*)$")
    SLASH_QUERY_RE = re.compile(r"^\s*/([\w-]*)$")
    MENTION_ACTIONS = {"mention_prev", "mention_next", "mention_accept", "mention_dismiss"}

    @dataclass
    class Submitted(TextualMessage):
        composer: ChatComposer
        value: str

        @property
        def control(self) -> ChatComposer:
            return self.composer

    def check_action(self, action: str, parameters: tuple) -> bool | None:
        if action in self.MENTION_ACTIONS:
            dropdown = self.mention_dropdown()
            return dropdown is not None and dropdown.is_open
        return super().check_action(action, parameters)

    def mention_dropdown(self) -> Optional[CompletionDropdown]:
        try:
            return self.screen.query_one("#completion_dropdown", CompletionDropdown)
        except Exception:
            return None

    def active_mention_query(self) -> Optional[tuple[str, int, int]]:
        """Return (query, start_col, end_col) for the @ token under the cursor, if any."""
        row, col = self.cursor_location
        try:
            line = self.document.get_line(row)
        except Exception:
            return None
        match = self.MENTION_QUERY_RE.search(line[:col])
        if match is None:
            return None
        return match.group(1), match.start(), col

    def active_slash_query(self) -> Optional[tuple[str, int, int]]:
        """Return (query, start_col, end_col) for a /command token starting the input, if any.

        Only fires when the cursor is on the first line and the slash is the
        first non-whitespace character, so paths like ``src/foo`` never match.
        """
        row, col = self.cursor_location
        if row != 0:
            return None
        try:
            line = self.document.get_line(0)
        except Exception:
            return None
        match = self.SLASH_QUERY_RE.match(line[:col])
        if match is None:
            return None
        return match.group(1), match.start(1) - 1, col

    def apply_completion(self, entry: IndexEntry | SlashCommandEntry) -> None:
        if isinstance(entry, SlashCommandEntry):
            active = self.active_slash_query()
            if active is None:
                return
            _, start_col, end_col = active
            self.replace(f"{entry.token} ", (0, start_col), (0, end_col), maintain_selection_offset=False)
            return
        active = self.active_mention_query()
        if active is None:
            return
        _, start_col, end_col = active
        row, _ = self.cursor_location
        token = f'@"{entry.path}"' if " " in entry.path else f"@{entry.path}"
        if not entry.is_dir:
            token += " "
        self.replace(token, (row, start_col), (row, end_col), maintain_selection_offset=False)

    def _accept_mention_completion(self) -> bool:
        dropdown = self.mention_dropdown()
        if dropdown is None or not dropdown.is_open:
            return False
        entry = dropdown.highlighted_entry()
        if entry is None:
            return False
        self.apply_completion(entry)
        if isinstance(entry, SlashCommandEntry) or not entry.is_dir:
            dropdown.close()
        return True

    def action_submit(self) -> None:
        if self._accept_mention_completion():
            return
        self.post_message(self.Submitted(self, self.text))

    def action_insert_newline(self) -> None:
        self.insert("\n", maintain_selection_offset=False)

    def action_mention_prev(self) -> None:
        dropdown = self.mention_dropdown()
        if dropdown is not None and dropdown.is_open:
            dropdown.action_cursor_up()

    def action_mention_next(self) -> None:
        dropdown = self.mention_dropdown()
        if dropdown is not None and dropdown.is_open:
            dropdown.action_cursor_down()

    def action_mention_accept(self) -> None:
        self._accept_mention_completion()

    def action_mention_dismiss(self) -> None:
        dropdown = self.mention_dropdown()
        if dropdown is not None:
            dropdown.close()

    def on_blur(self, event) -> None:
        dropdown = self.mention_dropdown()
        if dropdown is not None:
            dropdown.close()


class KolegaCodeApp(App):
    """Interactive terminal UI for Kolega Code."""

    CSS = """
    Screen {
        layout: vertical;
    }

    #body {
        height: 1fr;
    }

    #conversation_panel {
        width: 2fr;
        height: 100%;
    }

    #side_panel {
        width: 1fr;
        min-width: 34;
        height: 100%;
    }

    #conversation, #logs, #terminal {
        height: 1fr;
        border: round $surface;
    }

    ConversationEntryWidget {
        height: auto;
        margin-bottom: 1;
    }

    ToolEntryWidget {
        height: auto;
        margin-bottom: 1;
    }

    ToolEntryWidget Collapsible {
        background: transparent;
        border-top: none;
        padding-bottom: 0;
        padding-left: 0;
    }

    ToolEntryWidget .tool-body {
        color: $text-muted;
    }

    #jump_to_bottom {
        display: none;
        height: 1;
        padding: 0 1;
        background: $surface;
        color: $text-muted;
        text-align: center;
    }

    #status_container {
        height: 1fr;
    }

    #status_dashboard {
        height: 1fr;
        min-height: 15;
        border: round $surface;
        padding: 1;
    }

    #settings_form, #planning_form {
        height: 1fr;
        padding: 1;
    }

    #settings_status {
        margin-top: 1;
    }

    #settings_form Label {
        margin-top: 1;
    }

    #settings_form Button {
        margin-top: 1;
    }

    #planning_form Markdown.empty-state {
        color: $text-muted;
    }

    #composer {
        dock: bottom;
        height: 5;
    }

    #turn_status {
        display: none;
        height: 1;
        padding: 0 1;
        color: $text-muted;
        background: $surface;
    }

    #composer_hint {
        display: none;
        height: 1;
        padding: 0 1;
        background: $surface;
    }

    #completion_dropdown {
        display: none;
        height: auto;
        max-height: 10;
        border: round $surface;
        background: $surface;
    }

    #composer_hint.hint-warning {
        color: $warning;
    }

    #composer_hint.hint-info {
        color: $text-muted;
    }

    #composer:disabled {
        opacity: 0.6;
    }

    #plan_actions, #question_actions {
        display: none;
        height: auto;
        padding: 0 1;
    }

    #plan_actions Button, #question_actions Button {
        margin-right: 1;
    }

    .meta {
        color: $text-muted;
    }
    """

    BINDINGS = [
        Binding("shift+tab", "toggle_interaction_mode", "Plan/Build", show=True, key_display="Shift+Tab", priority=True),
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
        browser_visible: bool = False,
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
        self.settings_store = settings_store or SettingsStore(store.root)
        self.overrides = overrides or CliConfigOverrides()
        self.settings: CliSettings = CliSettings()
        self.skill_catalog: SkillCatalog = discover_skills(self.project_path)
        self.file_index = WorkspaceFileIndex(self.project_path)
        self.browser_visible = browser_visible
        self.connection_manager = CliConnectionManager()
        self.agent: Optional[CoderAgent | PlanningAgent] = None
        self.agent_worker = None
        self.conversation_entries: list[ConversationEntry] = []
        self._stream_entries: dict[str, ConversationEntry] = {}
        self._tool_entries: dict[str, ConversationEntry] = {}
        self._tool_stream_buffers: dict[str, str] = {}
        self._sub_agent_activities: dict[str, SubAgentActivity] = {}
        self._sub_agent_by_tool_call: dict[str, str] = {}
        self._sub_agent_seq = 0
        self._render_pending = False
        self._entry_widgets: dict[str, ConversationEntryWidget | ToolEntryWidget] = {}
        self._dirty_entry_ids: set[str] = set()
        self._active_progress_entry: Optional[ConversationEntry] = None
        self._turn_active = False
        self._latest_plan: Optional[str] = self.session.latest_plan_markdown or None
        self._plan_decision_active = False
        self._pending_question: Optional[PendingQuestion] = None
        provider, model = self._startup_model()
        self._status_state = StatusDashboardState(provider=provider, model=model, mode=self.interaction_mode)
        self._turn_started_at: Optional[float] = None
        self._turn_finished_duration: Optional[float] = None
        self._turn_timer: Optional[Timer] = None
        self._turn_status_text = ""
        self._turn_final_text = ""
        self._turn_final_state = TurnState.IDLE
        self._spinner_frame = 0
        self._last_sub_agent_tick = 0.0
        self._terminal_has_content = False

    def compose(self) -> ComposeResult:
        with Horizontal(id="body"):
            with Vertical(id="conversation_panel"):
                yield Static(
                    self._meta_content(),
                    classes="meta",
                    id="session_meta",
                )
                yield ConversationView(id="conversation")
                yield JumpToBottomBar(
                    f"{theme.g(Glyph.DOWN)} More output below — click to jump to the latest",
                    id="jump_to_bottom",
                )
                with Horizontal(id="plan_actions"):
                    yield Button("Implement plan", variant="primary", id="implement_plan")
                    yield Button("Discuss further", id="discuss_plan")
                with Horizontal(id="question_actions"):
                    pass
                yield Static("", id="turn_status", markup=True)
                yield Static("", id="composer_hint", markup=False)
                yield CompletionDropdown(id="completion_dropdown")
                yield ChatComposer(placeholder=COMPOSER_PLACEHOLDER, id="composer")
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
                                yield Markdown(PLAN_EMPTY_MESSAGE, id="planning_plan_markdown")
                            with Collapsible(title="Task List", collapsed=False, id="planning_task_list"):
                                yield Markdown(TASK_LIST_EMPTY_MESSAGE, id="planning_task_list_markdown")
                    with TabPane("Settings", id="settings_pane"):
                        with Vertical(id="settings_form"):
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
                            yield Label("API key")
                            yield Input(password=True, id="api_key_input")
                            yield Button("Save Settings", variant="primary", id="save_settings")
                            yield Static("", id="settings_status")
        yield Footer()

    async def on_mount(self) -> None:
        self.settings = self.settings_store.load()
        self._populate_settings_controls()
        self._refresh_status_dashboard()
        self._restore_plan_action_visibility()
        self._set_question_actions_visible(False)
        self._refresh_planning_sidebar()
        self._ensure_startup_entry()
        self._conversation.anchor()
        self.run_worker(self._consume_events(), name="kolega-events", group="events")
        if self.config is not None:
            await self._build_agent(self.config)
            self._set_chat_enabled(True)
            self.query_one("#composer", ChatComposer).focus()
        else:
            await self._ensure_agent_from_settings()

    @property
    def _conversation(self) -> ConversationView:
        return self.query_one("#conversation", ConversationView)

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
        base = TAB_BASE_LABELS.get(pane_id)
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
        base = TAB_BASE_LABELS.get(pane_id)
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
        if pane_id in TAB_BASE_LABELS:
            self._clear_tab_activity(pane_id)

    def _log_status(self, text: str, level: str = "info") -> None:
        """Write a status line to the Logs tab with the semantic palette."""
        self._write_log(text, level)

    def _notify_user(self, message: str, *, severity: str = "information", title: Optional[str] = None) -> None:
        """Show a transient toast and keep a copy in the Logs tab."""
        level = {"information": "ok", "warning": "warn", "error": "error"}.get(severity, "info")
        self._log_status(message, level)
        try:
            self.notify(message, severity=severity, title=title)
        except Exception:
            pass

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
        if interaction_mode in {BUILD_INTERACTION_MODE, PLAN_INTERACTION_MODE}:
            return interaction_mode
        return BUILD_INTERACTION_MODE

    def _sync_planning_state_to_session(self) -> None:
        self.session.interaction_mode = self.interaction_mode
        self.session.latest_plan_markdown = self._latest_plan or ""

    def _save_session(self) -> None:
        self._sync_planning_state_to_session()
        self.store.save(self.session)

    def _restore_plan_action_visibility(self) -> None:
        self._set_plan_actions_visible(
            self.interaction_mode == PLAN_INTERACTION_MODE and bool(self._latest_plan),
            allow_discuss=self._plan_decision_active,
        )

    async def on_chat_composer_submitted(self, event: ChatComposer.Submitted) -> None:
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

        if await self._handle_skill_slash_command(stripped_text, event.composer):
            return

        if self._pending_question is not None:
            if not stripped_text:
                self._set_composer_status(QUESTION_PLACEHOLDER)
                return
            event.composer.load_text("")
            await self._answer_pending_question(stripped_text)
            return

        if self._plan_decision_active:
            self._set_composer_status(PLAN_READY_PLACEHOLDER)
            self._notify_user(messages.BLOCK_PLAN_DECISION, severity="warning")
            return

        if not stripped_text or self.agent is None:
            if stripped_text:
                self._set_settings_status(messages.SETTINGS_REQUIRED, tone="warning")
            return
        event.composer.load_text("")
        attachments = self._build_mention_attachments(text)
        self._add_conversation_entry(ConversationEntry(kind="user", content=text))
        self.agent_worker = self.run_worker(
            self._process_message(text, attachments), name="kolega-turn", group="turns", exclusive=True
        )

    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        if event.text_area.id == "composer":
            self._refresh_completion_dropdown()

    def _refresh_completion_dropdown(self) -> None:
        try:
            dropdown = self.query_one("#completion_dropdown", CompletionDropdown)
            composer = self.query_one("#composer", ChatComposer)
        except Exception:
            return
        slash = composer.active_slash_query()
        if slash is not None:
            commands = search_commands(slash[0], self.skill_catalog, limit=8)
            if not commands:
                dropdown.close()
                return
            dropdown.open_with([command_completion_item(entry) for entry in commands])
            return
        active = composer.active_mention_query()
        if active is None:
            dropdown.close()
            return
        entries = self.file_index.search(active[0], limit=8)
        if not entries:
            dropdown.close()
            return
        dropdown.open_with([file_completion_item(entry) for entry in entries])

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        if event.option_list.id != "completion_dropdown":
            return
        event.stop()
        try:
            dropdown = self.query_one("#completion_dropdown", CompletionDropdown)
            composer = self.query_one("#composer", ChatComposer)
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
                        self._update_progress(messages.READING_RESPONSE, complete=False, state=TurnState.GENERATING)
                    self._apply_stream_chunk(chunk, kind="assistant")
                    continue

                content = chunk.get("content")
                if chunk.get("type") == "thinking":
                    self._update_progress(messages.THINKING, complete=False, state=TurnState.THINKING)
                    self._apply_stream_chunk(chunk, kind="thinking")
                    if content:
                        self._write_log(content, "debug")
            await self._drain_pending_events()
            self._finalize_sub_agent_activities()
            self._save_session_history()
            self._finish_turn_progress(messages.FINISHED, TurnState.IDLE)
            self._capture_completed_plan()
            self._log_status(messages.FINISHED, "ok")
        except asyncio.CancelledError:
            self._cancel_pending_question()
            await self._drain_pending_events()
            self._finalize_sub_agent_activities()
            self._save_session_history()
            self._finish_turn_progress(messages.STOPPED_BY_USER, TurnState.STOPPED)
            self._log_status(messages.STOPPED_BY_USER, "warn")
        except Exception as exc:
            self._cancel_pending_question()
            await self._drain_pending_events()
            self._finalize_sub_agent_activities()
            self._save_session_history()
            self._finish_turn_progress(messages.STOPPED_WITH_ERROR.format(error=exc), TurnState.ERROR)
            self._log_status(messages.STOPPED_WITH_ERROR.format(error=exc), "error")
            raise
        finally:
            self._flush_conversation_render()
            self._active_progress_entry = None
            self._turn_active = False
            self.agent_worker = None
            if self._plan_decision_active:
                self._set_composer_status(PLAN_READY_PLACEHOLDER)
            else:
                self._restore_composer_placeholder()
            self._set_chat_enabled(self.agent is not None and not self._plan_decision_active)

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "save_settings":
            await self._save_settings_from_ui()
        elif event.button.id == "implement_plan":
            await self._implement_pending_plan()
        elif event.button.id == "discuss_plan":
            self._discuss_pending_plan()
        elif event.button.id and event.button.id.startswith(QUESTION_OPTION_BUTTON_PREFIX):
            await self._answer_question_option(event.button.id)

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id != "provider_select":
            return
        provider = str(event.value)
        model_select = self.query_one("#model_select", Select)
        model_options = ui_model_options(provider)
        model_select.set_options(model_options)
        if model_options:
            model_select.value = model_options[0][1]
        api_key_input = self.query_one("#api_key_input", Input)
        api_key_input.placeholder = self._api_key_placeholder(provider)

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
                self._update_activity_progress(messages.RUNNING_TERMINAL_COMMAND, state=TurnState.RUNNING_TOOL)
        elif event.event_type == "chat_message":
            if event.sub_agent_info:
                self._render_sub_agent_event(event)
                return
            message_text = event.content.get("text", "")
            message_type = event.content.get("message_type", "message")
            if message_type in {"tool_call", "tool_result", "tool_error"}:
                self._add_tool_message(message_type, event.content)
            elif message_text:
                self._add_conversation_entry(ConversationEntry(kind="message", content=message_text))
        elif event.event_type == "tool_streaming_update":
            if event.sub_agent_info:
                self._note_sub_agent_tool_stream(event)
            else:
                self._apply_tool_streaming_update(event.content)
        elif event.event_type == "llm_context_update":
            self._apply_context_status_update(event.content)
        elif event.event_type in {"llm_status_update", "status_update"}:
            if text:
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
            self._update_progress(messages.STOP_REQUESTED, complete=False, state=TurnState.STOPPING)
            self._cancel_pending_question()
            self.agent_worker.cancel()
            self._notify_user(messages.CANCEL_REQUESTED, severity="warning")

    def _mode_switch_blocked(self) -> bool:
        if self._turn_active or self.agent_worker is not None:
            self._show_composer_hint(messages.BLOCK_STOP_BEFORE_MODE_SWITCH)
            self._notify_user(messages.BLOCK_STOP_BEFORE_MODE_SWITCH, severity="warning")
            return True
        if self._plan_decision_active:
            self._set_composer_status(PLAN_READY_PLACEHOLDER)
            self._notify_user(messages.BLOCK_PLAN_DECISION_MODE_SWITCH, severity="warning")
            return True
        return False

    async def action_toggle_interaction_mode(self) -> None:
        if self._mode_switch_blocked():
            return

        target = PLAN_INTERACTION_MODE if self.interaction_mode == BUILD_INTERACTION_MODE else BUILD_INTERACTION_MODE
        await self._set_interaction_mode(target)

    async def action_quit(self) -> None:
        if self.agent is not None:
            self.session.history = self.agent.dump_message_history()
            self._save_session()
            await self.agent.cleanup()
        self.exit()

    def _populate_settings_controls(self) -> None:
        if not self.settings.active_provider:
            self.settings.active_provider = UI_DEFAULT_PROVIDER
        provider = self.settings.active_provider
        model_options = ui_model_options(provider)
        valid_models = {value for _, value in model_options}
        if not self.settings.active_model or self.settings.active_model not in valid_models:
            self.settings.active_model = model_options[0][1] if model_options else UI_DEFAULT_MODEL
        model = self.settings.active_model
        provider_select = self.query_one("#provider_select", Select)
        model_select = self.query_one("#model_select", Select)
        api_key_input = self.query_one("#api_key_input", Input)

        provider_select.value = provider
        model_select.set_options(model_options)
        model_select.value = model
        api_key_input.placeholder = self._api_key_placeholder(provider)
        self._update_settings_status()

    async def _save_settings_from_ui(self) -> None:
        provider = str(self.query_one("#provider_select", Select).value)
        model = str(self.query_one("#model_select", Select).value)
        api_key_input = self.query_one("#api_key_input", Input)
        api_key = api_key_input.value.strip()

        self.settings.active_provider = provider
        self.settings.active_model = model
        if api_key:
            self.settings.set_api_key(provider, api_key)
        self.settings_store.save(self.settings)
        api_key_input.value = ""
        api_key_input.placeholder = self._api_key_placeholder(provider)

        await self._ensure_agent_from_settings(rebuild=True)
        if self.config is not None:
            self._notify_user(messages.SETTINGS_SAVED)

    async def _ensure_agent_from_settings(self, rebuild: bool = False) -> None:
        try:
            config = build_agent_config(self.project_path, self.overrides, settings=self.settings)
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
        self.query_one("#composer", ChatComposer).focus()

    async def _build_agent(self, config: AgentConfig, rebuild: bool = False) -> None:
        history = self.session.history
        if self.agent is not None:
            history = self.agent.dump_message_history()
            self.session.history = history
            self._save_session()
            if rebuild:
                await self.agent.cleanup()

        browser_manager = PlaywrightBrowserManager()
        browser_manager.headless = not self.browser_visible
        agent_class = PlanningAgent if self.interaction_mode == PLAN_INTERACTION_MODE else CoderAgent
        self.skill_catalog = discover_skills(self.project_path)
        prompt_extensions = [self._shared_task_list_prompt_extension()]
        tool_extensions = [self._shared_task_list_tool_extension()]
        skill_prompt_extension = build_skill_prompt_extension(self.skill_catalog)
        skill_tool_extension = build_skill_tool_extension(
            self.skill_catalog,
            lambda: self.agent.history if self.agent is not None else [],
        )
        if skill_prompt_extension is not None:
            prompt_extensions.append(skill_prompt_extension)
        if skill_tool_extension is not None:
            tool_extensions.append(skill_tool_extension)
        if self.interaction_mode == PLAN_INTERACTION_MODE:
            prompt_extensions.append(self._planning_question_prompt_extension())
            tool_extensions.append(self._planning_question_tool_extension())

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
        )
        if history:
            self.agent.restore_message_history(history)
            self._restore_conversation_history(history)
        self._update_mode_chrome()

    async def _set_interaction_mode(self, interaction_mode: str) -> None:
        if interaction_mode not in {BUILD_INTERACTION_MODE, PLAN_INTERACTION_MODE}:
            raise ValueError(f"Unknown interaction mode: {interaction_mode}")
        if self.interaction_mode == interaction_mode:
            return

        self.interaction_mode = interaction_mode
        self._plan_decision_active = False
        self._save_session()
        self._restore_plan_action_visibility()
        self._cancel_pending_question()

        if self.config is not None:
            await self._build_agent(self.config, rebuild=True)

        self._update_mode_chrome()
        self._restore_composer_placeholder()
        self._set_chat_enabled(self.agent is not None)
        self._notify_user(messages.SWITCHED_MODE.format(mode=self.interaction_mode))

    def _capture_completed_plan(self) -> None:
        if self.interaction_mode != PLAN_INTERACTION_MODE or not isinstance(self.agent, PlanningAgent):
            return

        plan = self.agent.consume_completed_plan()
        if not plan:
            return

        self._latest_plan = plan
        self._plan_decision_active = True
        self._save_session()
        self._refresh_planning_sidebar()
        self._add_conversation_entry(ConversationEntry(kind="plan", content=plan, complete=True))
        self._set_plan_actions_visible(True, allow_discuss=True)
        self._set_composer_status(PLAN_READY_PLACEHOLDER)
        self._set_chat_enabled(False)
        self._notify_user(messages.PLAN_CAPTURED)

    async def _implement_pending_plan(self) -> None:
        plan = self._latest_plan
        if not plan or self._turn_active or self.agent_worker is not None:
            return

        self._plan_decision_active = False
        self._save_session()
        await self._set_interaction_mode(BUILD_INTERACTION_MODE)
        self._refresh_planning_sidebar()
        self._set_plan_actions_visible(False)

        prompt = IMPLEMENT_PLAN_PROMPT.format(plan=plan)
        self._add_conversation_entry(ConversationEntry(kind="user", content="Implement the approved plan."))
        self.agent_worker = self.run_worker(self._process_message(prompt), name="kolega-turn", group="turns", exclusive=True)

    def _discuss_pending_plan(self) -> None:
        if not self._latest_plan:
            return

        self._latest_plan = None
        self._plan_decision_active = False
        self._save_session()
        self._refresh_planning_sidebar()
        self._set_plan_actions_visible(False)
        self._restore_composer_placeholder()
        self._set_chat_enabled(self.agent is not None)
        self.query_one("#composer", ChatComposer).focus()
        self._notify_user(messages.PLAN_DISCUSSION_RESUMED)

    def _set_plan_actions_visible(self, visible: bool, *, allow_discuss: bool = False) -> None:
        try:
            self.query_one("#plan_actions", Horizontal).display = visible
            self.query_one("#implement_plan", Button).display = visible
            self.query_one("#discuss_plan", Button).display = visible and allow_discuss
        except Exception:
            return

    def _meta_content(self) -> str:
        return (
            f"{self.project_path} | session {self.session.session_id} | "
            f"agent {self.mode} | {self.interaction_mode}"
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
        plan_content = self._latest_plan or PLAN_EMPTY_MESSAGE
        task_list_content = self.session.task_list_markdown or TASK_LIST_EMPTY_MESSAGE
        try:
            plan_markdown = self.query_one("#planning_plan_markdown", Markdown)
            task_list_markdown = self.query_one("#planning_task_list_markdown", Markdown)
            plan_markdown.update(plan_content)
            task_list_markdown.update(task_list_content)
            plan_markdown.set_class(plan_content == PLAN_EMPTY_MESSAGE, "empty-state")
            task_list_markdown.set_class(task_list_content == TASK_LIST_EMPTY_MESSAGE, "empty-state")
        except Exception:
            pass

    def _set_chat_enabled(self, enabled: bool) -> None:
        composer = self.query_one("#composer", ChatComposer)
        composer.disabled = not enabled or self._plan_decision_active

    def _set_composer_status(self, status: str) -> None:
        self.query_one("#composer", ChatComposer).placeholder = status

    def _restore_composer_placeholder(self) -> None:
        self.query_one("#composer", ChatComposer).placeholder = COMPOSER_PLACEHOLDER
        self._clear_composer_hint()

    def _show_composer_hint(self, text: str, tone: str = "warning") -> None:
        try:
            hint = self.query_one("#composer_hint", Static)
        except Exception:
            return
        hint.set_class(tone == "warning", "hint-warning")
        hint.set_class(tone != "warning", "hint-info")
        hint.update(text)
        hint.display = bool(text)

    def _clear_composer_hint(self) -> None:
        try:
            hint = self.query_one("#composer_hint", Static)
        except Exception:
            return
        hint.update("")
        hint.display = False

    def _tui_command_handlers(self) -> dict[str, Callable[[str], Awaitable[None]]]:
        return {
            "/plan": self._command_plan,
            "/build": self._command_build,
            "/model": self._command_model,
            "/copy": self._command_copy,
            "/version": self._command_version,
            "/quit": self._command_quit,
        }

    async def _handle_tui_slash_command(self, stripped_text: str, composer: ChatComposer) -> bool:
        if not stripped_text.startswith("/"):
            return False
        command_text, _, args = stripped_text.partition(" ")
        handler = self._tui_command_handlers().get(command_text.lower())
        if handler is None:
            return False
        composer.load_text("")
        await handler(args.strip())
        return True

    async def _command_plan(self, args: str) -> None:
        if self._mode_switch_blocked():
            return
        await self._set_interaction_mode(PLAN_INTERACTION_MODE)

    async def _command_build(self, args: str) -> None:
        if self._mode_switch_blocked():
            return
        await self._set_interaction_mode(BUILD_INTERACTION_MODE)

    async def _command_model(self, args: str) -> None:
        provider = self.settings.active_provider or UI_DEFAULT_PROVIDER
        model_options = ui_model_options(provider)
        if not args:
            current_provider, current_model = self._startup_model()
            lines = [
                messages.SETTINGS_ACTIVE_MODEL.format(provider=current_provider, model=current_model),
                "",
                "Available models:",
                *(f"- `{value}` ({label})" for label, value in model_options),
                "",
                messages.MODEL_SWITCH_HINT,
            ]
            self._add_conversation_entry(ConversationEntry(kind="system", content="\n".join(lines)))
            return

        if self._turn_active or self.agent_worker is not None:
            self._show_composer_hint(messages.BLOCK_STOP_BEFORE_MODEL_SWITCH)
            self._notify_user(messages.BLOCK_STOP_BEFORE_MODEL_SWITCH, severity="warning")
            return

        matched = next((value for _, value in model_options if value.lower() == args.lower()), None)
        if matched is None:
            self._notify_user(messages.MODEL_UNKNOWN.format(model=args, provider=provider), severity="warning")
            return

        self.settings.active_model = matched
        self.settings_store.save(self.settings)
        await self._ensure_agent_from_settings(rebuild=True)
        try:
            self._populate_settings_controls()
        except Exception:
            pass
        self._notify_user(messages.MODEL_SWITCHED.format(provider=provider, model=matched))

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
        self._add_conversation_entry(
            ConversationEntry(kind="system", content=messages.VERSION_INFO.format(version=kolega_code_version))
        )

    async def _command_quit(self, args: str) -> None:
        await self.action_quit()

    async def _handle_skill_slash_command(self, stripped_text: str, composer: ChatComposer) -> bool:
        command = self._parse_skill_slash_command(stripped_text)
        if command is None:
            return False

        command_name, prompt = command
        composer.load_text("")

        if command_name == "skills":
            self._add_conversation_entry(ConversationEntry(kind="system", content=self.skill_catalog.format_catalog()))
            self._log_status(messages.SKILLS_LISTED, "ok")
            return True

        if self._pending_question is not None:
            self._set_composer_status(QUESTION_PLACEHOLDER)
            self._notify_user(messages.BLOCK_PENDING_QUESTION_SKILL, severity="warning")
            return True

        if self._plan_decision_active:
            self._set_composer_status(PLAN_READY_PLACEHOLDER)
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
        self._add_conversation_entry(ConversationEntry(kind="skill", content=activated))
        self._notify_user(messages.SKILL_ACTIVATED.format(name=command_name))

        if prompt:
            attachments = self._build_mention_attachments(prompt)
            self._add_conversation_entry(ConversationEntry(kind="user", content=prompt))
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

    def _shared_task_list_prompt_extension(self) -> PromptExtension:
        return PromptExtension(
            id="cli-shared-task-list",
            title="Shared Task List",
            markdown=SHARED_TASK_LIST_PROMPT,
            modes=[AgentMode.CLI],
        )

    def _shared_task_list_tool_extension(self) -> ToolExtension:
        async def get_task_list() -> str:
            """
            Return the shared CLI task list.

            Use this before planning or implementation work when you need the current task state.

            Returns:
                The current shared task list, or a note that no task list has been set.
            """
            return self.session.task_list_markdown or TASK_LIST_EMPTY_MESSAGE

        async def update_task_list(task_list_markdown: str) -> str:
            """
            Replace the shared CLI task list.

            Format the list as Markdown checkboxes, for example `- [ ] inspect CLI state handling`.
            Use this after completing individual task-list items so progress is visible incrementally; do not wait
            until every TODO is complete before updating the list.

            Args:
                task_list_markdown: The full current shared task list as Markdown.

            Returns:
                A confirmation that the shared task list was updated.
            """
            self.session.task_list_markdown = task_list_markdown.strip()
            self._save_session()
            self._refresh_planning_sidebar()
            return "Task list updated."

        return ToolExtension(
            name="cli-shared-task-list",
            tools={
                "get_task_list": get_task_list,
                "update_task_list": update_task_list,
            },
            tool_groups={
                "planning_tools": ["get_task_list", "update_task_list"],
                "cli_task_list_tools": ["get_task_list", "update_task_list"],
            },
        )

    def _planning_question_prompt_extension(self) -> PromptExtension:
        return PromptExtension(
            id="cli-planning-questions",
            title="Planning Questions",
            markdown=PLANNING_QUESTION_PROMPT,
            modes=[AgentMode.CLI],
        )

    def _planning_question_tool_extension(self) -> ToolExtension:
        async def ask_user_choice(question: str, options: list[str]) -> str:
            """
            Ask the user a multiple-choice planning question and wait for their answer.

            Use this only for planning decisions that materially affect the final plan. The user may either select
            one of the provided options or type a custom free-text answer.

            Args:
                question: The concise question to ask the user.
                options: Two or more concise answer options.

            Returns:
                The selected option text, or the user's custom answer text.
            """
            if self.interaction_mode != PLAN_INTERACTION_MODE:
                raise RuntimeError("ask_user_choice is only available in planning mode.")
            if isinstance(options, str) or not isinstance(options, list):
                raise ValueError("options must be a list of answer strings.")

            clean_question = str(question).strip()
            clean_options = [str(option).strip() for option in options if str(option).strip()]
            if not clean_question:
                raise ValueError("question must not be empty.")
            if len(clean_options) < 2:
                raise ValueError("ask_user_choice requires at least two non-empty options.")
            if self._pending_question is not None:
                raise RuntimeError("A planning question is already waiting for an answer.")

            return await self._ask_user_choice(clean_question, clean_options)

        return ToolExtension(
            name="cli-planning-questions",
            tools={QUESTION_TOOL_NAME: ask_user_choice},
            tool_groups={"planning_tools": [QUESTION_TOOL_NAME]},
        )

    async def _ask_user_choice(self, question: str, options: list[str]) -> str:
        loop = asyncio.get_running_loop()
        future: asyncio.Future[str] = loop.create_future()
        self._pending_question = PendingQuestion(question=question, options=options, future=future)
        self._add_conversation_entry(
            ConversationEntry(kind="question", content=self._format_question_content(question, options))
        )
        await self._show_question_actions(options)
        self._set_composer_status(QUESTION_PLACEHOLDER)
        self._set_chat_enabled(True)
        self._update_activity_progress(messages.WAITING_FOR_ANSWER, state=TurnState.WAITING_FOR_USER)

        try:
            return await future
        finally:
            if self._pending_question is not None and self._pending_question.future is future:
                self._pending_question = None
                self._set_question_actions_visible(False)

    async def _answer_question_option(self, button_id: str) -> None:
        if self._pending_question is None:
            return
        index_text = button_id.removeprefix(QUESTION_OPTION_BUTTON_PREFIX)
        try:
            option_index = int(index_text)
        except ValueError:
            return
        if option_index < 0 or option_index >= len(self._pending_question.options):
            return
        await self._answer_pending_question(self._pending_question.options[option_index])

    async def _answer_pending_question(self, answer: str) -> None:
        pending_question = self._pending_question
        if pending_question is None:
            return

        clean_answer = answer.strip()
        if not clean_answer:
            self._set_composer_status(QUESTION_PLACEHOLDER)
            return

        self._pending_question = None
        self._set_question_actions_visible(False)
        self._add_conversation_entry(ConversationEntry(kind="user", content=clean_answer))
        if not pending_question.future.done():
            pending_question.future.set_result(clean_answer)

        if self._turn_active:
            self._restore_composer_placeholder()
            self._set_chat_enabled(False)
            self._update_progress(messages.WORKING, complete=False, state=TurnState.GENERATING)
        else:
            self._restore_composer_placeholder()
            self._set_chat_enabled(self.agent is not None)

    async def _show_question_actions(self, options: list[str]) -> None:
        try:
            question_actions = self.query_one("#question_actions", Horizontal)
            await question_actions.remove_children()
            buttons = [
                Button(
                    self._question_option_button_label(index, option),
                    id=f"{QUESTION_OPTION_BUTTON_PREFIX}{index}",
                    variant="primary" if index == 0 else "default",
                )
                for index, option in enumerate(options)
            ]
            if buttons:
                await question_actions.mount(*buttons)
            self._set_question_actions_visible(True)
        except Exception:
            return

    def _set_question_actions_visible(self, visible: bool) -> None:
        try:
            self.query_one("#question_actions", Horizontal).display = visible
        except Exception:
            return

    def _cancel_pending_question(self) -> None:
        pending_question = self._pending_question
        if pending_question is not None and not pending_question.future.done():
            pending_question.future.cancel()
        self._pending_question = None
        self._set_question_actions_visible(False)

    def _format_question_content(self, question: str, options: list[str]) -> str:
        option_lines = [f"{index + 1}. {option}" for index, option in enumerate(options)]
        return "\n".join([question, "", *option_lines])

    def _question_option_button_label(self, index: int, option: str) -> str:
        label = f"{index + 1}. {option}"
        if len(label) <= 60:
            return label
        return f"{label[:57]}..."

    def _reset_current_thread(self) -> None:
        if self.agent is not None:
            self.agent.history = MessageHistory()
        self.session.history = []
        self.session.task_list_markdown = ""
        self.conversation_entries = []
        self._stream_entries = {}
        self._tool_entries = {}
        self._tool_stream_buffers = {}
        self._sub_agent_activities = {}
        self._sub_agent_by_tool_call = {}
        self._sub_agent_seq = 0
        self._active_progress_entry = None
        self._latest_plan = None
        self._plan_decision_active = False
        self._save_session()
        self._set_plan_actions_visible(False)
        self._cancel_pending_question()
        self._refresh_planning_sidebar()
        self._clear_turn_status_strip()
        self._turn_active = False
        self._restore_composer_placeholder()
        self._set_chat_enabled(self.agent is not None)
        self._ensure_startup_entry(render=False)
        self._add_conversation_entry(ConversationEntry(kind="progress", content=THREAD_RESET_MESSAGE, complete=True))
        self._notify_user(THREAD_RESET_MESSAGE)

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
        provider = self.settings.active_provider or UI_DEFAULT_PROVIDER
        model = self.settings.active_model or UI_DEFAULT_MODEL
        status = key_status(provider, self.project_path, self.settings)
        tone = "warning" if "missing" in status.lower() else "ok"
        text = "\n".join(
            [
                messages.SETTINGS_ACTIVE_MODEL.format(provider=provider, model=model),
                messages.SETTINGS_API_KEY_LINE.format(status=status),
            ]
        )
        self._set_settings_status(text, tone)
        self._refresh_status_dashboard()

    def _api_key_placeholder(self, provider: str) -> str:
        if self.settings.has_api_key(provider):
            return "Stored API key will be kept if blank"
        model = get_ui_model(provider, (ui_model_options(provider) or [("", "")])[0][1])
        return f"{model.provider_label} API key" if model else "API key"

    def _add_conversation_entry(self, entry: ConversationEntry) -> None:
        self.conversation_entries.append(entry)
        if entry.uuid:
            self._stream_entries[entry.uuid] = entry
        if entry.tool_call_id:
            self._tool_entries[entry.tool_call_id] = entry
        self._invalidate_conversation(entry)

    def _ensure_startup_entry(self, *, render: bool = True) -> None:
        existing = next((entry for entry in self.conversation_entries if entry.kind == "startup"), None)
        if existing is None:
            self.conversation_entries.insert(0, ConversationEntry(kind="startup", content=self._startup_content()))
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
        api_key = key_status(provider, self.project_path, self.settings)
        return "\n".join(
            [
                *STARTUP_WORDMARK,
                "",
                f"Project: {self.project_path}",
                f"Session: {session_id}",
                f"Mode: {self.mode}",
                f"Interaction: {self.interaction_mode}",
                f"Model: {provider}/{model}",
                f"API key: {api_key}",
                "",
                f"Enter send {theme.g(Glyph.BULLET_SEP)} Shift+Enter newline {theme.g(Glyph.BULLET_SEP)} Shift+Tab plan/build",
                f"Ctrl+C stop turn {theme.g(Glyph.BULLET_SEP)} Cmd+C copy selection {theme.g(Glyph.BULLET_SEP)} / commands",
            ]
        )

    def _startup_model(self) -> tuple[str, str]:
        if self.config is not None:
            return self.config.long_context_config.provider.value, self.config.long_context_config.model

        provider = self.settings.active_provider or UI_DEFAULT_PROVIDER
        model = self.settings.active_model
        if model:
            return provider, model

        model_options = ui_model_options(provider)
        if model_options:
            return provider, model_options[0][1]
        return provider, UI_DEFAULT_MODEL

    def _refresh_status_dashboard(self) -> None:
        provider, model = self._startup_model()
        self._status_state.provider = provider
        self._status_state.model = model
        self._status_state.mode = self.interaction_mode
        try:
            self._status_dashboard.update(self._format_status_dashboard())
        except Exception:
            return

    def _format_status_dashboard(self) -> str:
        state = self._status_state
        provider_model = f"{state.provider}/{state.model}"
        mode = state.mode.title()
        turn_style = TURN_STATE_STYLES.get(state.turn_state, Color.ACCENT)
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

        title = theme.role_header(Glyph.STATUS, "Status", Color.ACCENT)
        turn_line = (
            f"{label('Turn')} [{turn_style}]{theme.g(Glyph.STATUS)}[/{turn_style}] "
            f"[bold]{escape(state.turn_state.value)}[/bold]"
        )
        return (
            f"{title}\n\n"
            f"{label('Model')}\n[bold]{escape(provider_model)}[/bold]\n\n"
            f"{label('Mode')} [bold]{mode}[/bold]\n"
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

    def _set_status_activity(self, content: str, *, turn_state: Optional[TurnState] = None) -> None:
        if content:
            self._status_state.activity = content
        if turn_state is not None:
            self._status_state.turn_state = turn_state
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
        self._turn_final_state = TurnState.IDLE
        self._spinner_frame = 0
        self._turn_timer = self.set_interval(theme.SPINNER_INTERVAL, self._refresh_turn_status_strip, name="turn-status")
        self._refresh_turn_status_strip()

    def _complete_turn_timer(self, content: str, state: TurnState = TurnState.IDLE) -> None:
        if self._turn_timer is not None:
            self._turn_timer.stop()
            self._turn_timer = None
        if self._turn_started_at is None:
            return

        self._turn_finished_duration = max(0.0, self._now() - self._turn_started_at)
        duration = self._format_turn_duration(self._turn_finished_duration)
        self._turn_final_state = state
        if state is TurnState.ERROR:
            self._turn_final_text = messages.ERRORED_AFTER.format(duration=duration)
        elif state in {TurnState.STOPPED, TurnState.STOPPING}:
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
        self._turn_final_state = TurnState.IDLE
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
            if self._turn_final_state is TurnState.ERROR:
                glyph, color = Glyph.CROSS, Color.ERROR
            elif self._turn_final_state in {TurnState.STOPPED, TurnState.STOPPING}:
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

    def _restore_conversation_history(self, history: list[dict]) -> None:
        self.conversation_entries = []
        self._stream_entries = {}
        self._tool_entries = {}
        self._tool_stream_buffers = {}
        self._sub_agent_activities = {}
        self._sub_agent_by_tool_call = {}
        self._sub_agent_seq = 0
        self._active_progress_entry = None
        self._plan_decision_active = False
        self._restore_plan_action_visibility()
        self._cancel_pending_question()
        self._refresh_planning_sidebar()
        self._ensure_startup_entry(render=False)
        for item in history:
            try:
                message = Message.from_dict(item)
            except Exception:
                continue
            self.conversation_entries.extend(self._conversation_entries_from_message(message))
        self._render_conversation()

    def _conversation_entries_from_message(self, message: Message) -> list[ConversationEntry]:
        entries: list[ConversationEntry] = []

        if isinstance(message.content, str):
            content = message.content.strip()
            if content:
                entries.append(self._conversation_entry_for_text(message.role, content))
            return entries

        pending_text: list[str] = []

        def flush_text() -> None:
            text = "\n".join(part for part in pending_text if part).strip()
            pending_text.clear()
            if text:
                entries.append(self._conversation_entry_for_text(message.role, text))

        for block in message.content:
            if isinstance(block, TextBlock):
                pending_text.append(block.text)
            elif isinstance(block, ToolCall):
                flush_text()
                entries.append(
                    ConversationEntry(
                        kind="tool_call",
                        content=f"Calling {block.name}",
                        complete=True,
                        tool_name=block.name,
                        tool_call_id=getattr(block, "execution_id", None),
                    )
                )
            elif isinstance(block, ToolResult):
                flush_text()
                text = self._tool_content_to_text(block.content)
                entries.append(
                    ConversationEntry(
                        kind="tool_error" if block.is_error else "tool_result",
                        content=self._truncate_tool_text(text) if block.is_error else self._tool_result_preview(text),
                        tool_name=block.name,
                        tool_call_id=getattr(block, "execution_id", None),
                        full_content=self._capped_tool_text(text),
                    )
                )

        flush_text()
        return entries

    def _conversation_entry_for_text(self, role: str, text: str) -> ConversationEntry:
        names = skill_names_in_text(text)
        if names:
            skill_list = ", ".join(f"`/{name}`" for name in names)
            return ConversationEntry(kind="skill", content=f"Activated skill {skill_list}.")
        return ConversationEntry(kind=self._entry_kind_for_role(role), content=text)

    def _entry_kind_for_role(self, role: str) -> str:
        if role == "assistant":
            return "assistant"
        if role == "user":
            return "user"
        return "system"

    def _tool_content_to_text(self, content: object) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "\n\n".join(
                item.to_markdown() if hasattr(item, "to_markdown") else str(item) for item in content
            )
        return str(content)

    def _apply_stream_chunk(self, chunk: dict, *, kind: str) -> None:
        chunk_uuid = str(chunk.get("uuid") or "")
        content = str(chunk.get("content") or "")
        complete = bool(chunk.get("complete"))

        entry = self._stream_entries.get(chunk_uuid) if chunk_uuid else None
        if entry is None:
            if not content:
                return
            entry = ConversationEntry(kind=kind, content="", complete=complete, uuid=chunk_uuid or None)
            self.conversation_entries.append(entry)
            if chunk_uuid:
                self._stream_entries[chunk_uuid] = entry

        entry.content += content
        entry.complete = complete
        self._invalidate_conversation(entry)

    def _begin_turn_progress(self) -> None:
        self._tool_entries = {}
        self._tool_stream_buffers = {}
        self._sub_agent_activities = {}
        self._sub_agent_by_tool_call = {}
        self._sub_agent_seq = 0
        self._active_progress_entry = None
        self._turn_active = True
        self._set_chat_enabled(False)
        self._start_turn_timer(messages.WORKING)
        self._set_status_activity(messages.WORKING, turn_state=TurnState.GENERATING)
        self._update_progress(messages.WORKING, complete=False, state=TurnState.GENERATING)

    def _update_progress(self, content: str, complete: bool, state: Optional[TurnState] = None) -> None:
        if complete:
            final_state = state or TurnState.IDLE
            self._complete_turn_timer(content, final_state)
            self._set_status_activity(content, turn_state=final_state)
            if final_state is not TurnState.IDLE:
                tone = "error" if final_state is TurnState.ERROR else "warning"
                self._add_conversation_entry(
                    ConversationEntry(kind="progress", content=content, complete=True, tone=tone)
                )
            self._restore_composer_placeholder()
            return
        self._turn_status_text = content
        self._refresh_turn_status_strip()
        self._set_status_activity(content, turn_state=state)

    def _update_activity_progress(self, content: str, state: Optional[TurnState] = None) -> None:
        if self._turn_active:
            self._update_progress(content, complete=False, state=state)

    def _finish_turn_progress(self, content: str, state: TurnState = TurnState.IDLE) -> None:
        self._update_progress(content, complete=True, state=state)

    def _save_session_history(self) -> None:
        if self.agent is None:
            return
        self.session.history = self.agent.dump_message_history()
        self._save_session()

    def _add_tool_message(self, message_type: str, content: dict) -> None:
        tool_name = str(content.get("tool_description") or content.get("tool_name") or "tool")
        tool_call_id = str(content.get("tool_call_id") or "")
        text = str(content.get("text") or "")
        if tool_name == QUESTION_TOOL_NAME and message_type in {"tool_call", "tool_result"}:
            return
        entry = self._find_tool_entry(tool_call_id, tool_name)

        if message_type == "tool_call":
            self._clear_tool_stream_buffer(tool_call_id, tool_name)
            entry_content = text or f"Calling {tool_name}"
            full_content = ""
            complete = False
            self._update_activity_progress(messages.RUNNING_TOOL.format(tool=tool_name), state=TurnState.RUNNING_TOOL)
        elif message_type == "tool_error":
            entry_content = self._truncate_tool_text(text)
            full_content = self._capped_tool_text(text)
            complete = True
            self._clear_tool_stream_buffer(tool_call_id, tool_name)
            self._update_activity_progress(messages.TOOL_FAILED.format(tool=tool_name))
        else:
            entry_content = self._tool_result_preview(text)
            full_content = self._capped_tool_text(text)
            complete = True
            self._clear_tool_stream_buffer(tool_call_id, tool_name)
            self._update_activity_progress(messages.TOOL_DONE.format(tool=tool_name))

        if entry is None:
            self._add_conversation_entry(
                ConversationEntry(
                    kind=message_type,
                    content=entry_content,
                    complete=complete,
                    tool_name=tool_name,
                    tool_call_id=tool_call_id or None,
                    full_content=full_content,
                )
            )
            return

        entry.kind = message_type
        entry.content = entry_content
        entry.complete = complete
        entry.tool_name = tool_name
        entry.full_content = full_content
        entry.tool_call_id = tool_call_id or entry.tool_call_id
        if entry.tool_call_id:
            self._tool_entries[entry.tool_call_id] = entry
        self._invalidate_conversation(entry)

    def _apply_tool_streaming_update(self, content: dict) -> None:
        tool_name = str(content.get("tool_name") or content.get("tool_description") or "tool")
        tool_call_id = str(content.get("tool_call_id") or "")
        text = str(content.get("text") or "")
        is_complete = bool(content.get("is_complete"))
        stream_mode = str(content.get("stream_mode") or "replace")
        entry = self._find_tool_entry(tool_call_id, tool_name)
        buffer_key = self._tool_stream_buffer_key(tool_call_id, tool_name)

        if is_complete:
            self._clear_tool_stream_buffer(tool_call_id, tool_name)
            entry_content = self._tool_result_preview(text)
            full_content = self._capped_tool_text(text)
        elif stream_mode == "append":
            buffer_text = self._tool_stream_buffers.get(buffer_key, "") + text
            self._tool_stream_buffers[buffer_key] = buffer_text
            entry_content = self._tool_stream_preview(buffer_text)
            full_content = self._capped_tool_text(buffer_text)
        else:
            self._tool_stream_buffers[buffer_key] = text
            entry_content = self._truncate_tool_text(text)
            full_content = self._capped_tool_text(text)

        if is_complete:
            self._update_activity_progress(messages.TOOL_DONE.format(tool=tool_name))
        else:
            self._update_activity_progress(messages.RUNNING_TOOL.format(tool=tool_name), state=TurnState.RUNNING_TOOL)

        if entry is None:
            self._add_conversation_entry(
                ConversationEntry(
                    kind="tool_result" if is_complete else "tool_call",
                    content=entry_content or f"Running {tool_name}",
                    complete=is_complete,
                    tool_name=tool_name,
                    tool_call_id=tool_call_id or None,
                    full_content=full_content,
                )
            )
            return

        entry.kind = "tool_result" if is_complete else "tool_call"
        entry.content = entry_content or entry.content
        entry.complete = is_complete
        entry.tool_name = tool_name
        entry.full_content = full_content or entry.full_content
        entry.tool_call_id = tool_call_id or entry.tool_call_id
        if entry.tool_call_id:
            self._tool_entries[entry.tool_call_id] = entry
        self._invalidate_conversation(entry)

    def _tool_stream_buffer_key(self, tool_call_id: str, tool_name: str) -> str:
        return tool_call_id or f"name:{tool_name}"

    def _clear_tool_stream_buffer(self, tool_call_id: str, tool_name: str) -> None:
        if tool_call_id:
            self._tool_stream_buffers.pop(tool_call_id, None)
        self._tool_stream_buffers.pop(f"name:{tool_name}", None)

    def _find_tool_entry(self, tool_call_id: str, tool_name: str) -> Optional[ConversationEntry]:
        if tool_call_id and tool_call_id in self._tool_entries:
            return self._tool_entries[tool_call_id]
        for entry in reversed(self.conversation_entries):
            if entry.kind not in {"tool_call", "tool_result", "tool_error"}:
                continue
            if entry.complete:
                continue
            if entry.tool_name == tool_name:
                return entry
        return None

    def _sub_agent_key(self, event: AgentEvent) -> str:
        info = event.sub_agent_info or {}
        return str(
            info.get("agent_id")
            or info.get("parent_tool_call_id")
            or info.get("agent_name")
            or event.sender
        )

    def _ensure_sub_agent_activity(self, event: AgentEvent) -> SubAgentActivity:
        key = self._sub_agent_key(event)
        activity = self._sub_agent_activities.get(key)
        if activity is None:
            info = event.sub_agent_info or {}
            self._sub_agent_seq += 1
            entry = ConversationEntry(kind="sub_agent", content="", complete=False)
            activity = SubAgentActivity(
                agent_id=key,
                agent_name=str(info.get("agent_name") or event.sender or "sub-agent"),
                task=str(info.get("task") or ""),
                index=self._sub_agent_seq,
                entry=entry,
                started_at=self._now(),
            )
            self._sub_agent_activities[key] = activity
            parent_id = info.get("parent_tool_call_id")
            if parent_id:
                self._sub_agent_by_tool_call[str(parent_id)] = key
            entry.content = self._format_sub_agent_content(activity)
            self._add_conversation_entry(entry)
            self._refresh_sub_agent_activity_status()
        return activity

    def _render_sub_agent_event(self, event: AgentEvent) -> None:
        activity = self._ensure_sub_agent_activity(event)
        content = event.content
        status = content.get("status")
        if status:  # lifecycle event from AgentTool
            if status != "GENERATING":
                message = str(content.get("message") or "")
                failed = status == "ERROR" or message.startswith("Error")
                activity.status = "failed" if failed else "completed"
                activity.finished_at = self._now()
                activity.entry.complete = True
                activity.last_activity = message if failed else ""
                self._refresh_sub_agent_activity_status()
            self._refresh_sub_agent_entry(activity, force=True)
            return

        message_type = content.get("message_type", "message")
        text = str(content.get("text") or "")
        if message_type == "tool_call":
            activity.tool_calls += 1
            activity.last_activity = str(content.get("tool_description") or content.get("tool_name") or "tool")
        elif message_type in {"tool_result", "tool_error"}:
            suffix = "failed" if message_type == "tool_error" else "done"
            tool = str(content.get("tool_description") or content.get("tool_name") or "tool")
            activity.last_activity = f"{tool} {suffix}"
        elif message_type == "thinking":
            activity.last_activity = "thinking"
        else:  # streamed response text - accumulate by chunk uuid
            if event.uuid and text:
                buffer = activity.stream_buffers.get(event.uuid, "") + text
                activity.stream_buffers[event.uuid] = buffer
                activity.active_stream_uuid = event.uuid
        self._refresh_sub_agent_entry(activity)

    def _note_sub_agent_tool_stream(self, event: AgentEvent) -> None:
        activity = self._ensure_sub_agent_activity(event)
        tool_name = str(event.content.get("tool_name") or event.content.get("tool_description") or "tool")
        is_complete = bool(event.content.get("is_complete"))
        activity.last_activity = f"{tool_name} done" if is_complete else f"{tool_name} streaming"
        self._refresh_sub_agent_entry(activity)

    def _refresh_sub_agent_entry(self, activity: SubAgentActivity, *, force: bool = False) -> None:
        activity.entry.content = self._format_sub_agent_content(activity)
        self._invalidate_conversation(activity.entry)
        if force:
            self._flush_conversation_render()

    def _format_sub_agent_content(self, activity: SubAgentActivity) -> str:
        if activity.finished_at is not None:
            elapsed = max(0.0, activity.finished_at - activity.started_at)
        else:
            elapsed = max(0.0, self._now() - activity.started_at)
        duration = self._format_turn_duration(elapsed)

        if activity.status == "running":
            color, state = Color.ACCENT, f"running {theme.g(Glyph.BULLET_SEP)} {duration}"
        elif activity.status == "completed":
            color, state = Color.SUCCESS, f"completed in {duration}"
        elif activity.status == "failed":
            color, state = Color.ERROR, f"failed after {duration}"
        else:
            color, state = Color.WARNING, f"stopped after {duration}"

        header = theme.role_header(
            Glyph.SUB_AGENT,
            escape(activity.agent_name),
            color,
            state=f"#{activity.index} {theme.g(Glyph.BULLET_SEP)} {state}",
        )

        body_lines: list[str] = []
        if activity.task:
            task = activity.task
            if len(task) > SUB_AGENT_TASK_PREVIEW_CHARS:
                task = f"{task[:SUB_AGENT_TASK_PREVIEW_CHARS]}{theme.g(Glyph.ELLIPSIS)}"
            body_lines.append(f"Task: {task}")
        tools_line = f"{activity.tool_calls} tool{'' if activity.tool_calls == 1 else 's'}"
        if activity.last_activity:
            tools_line += f" · last: {activity.last_activity}"
        body_lines.append(tools_line)
        if activity.status == "running" and activity.active_stream_uuid:
            tail = activity.stream_buffers.get(activity.active_stream_uuid, "")
            tail = " ".join(tail.split())
            if tail:
                if len(tail) > SUB_AGENT_TAIL_CHARS:
                    tail = f"…{tail[-SUB_AGENT_TAIL_CHARS:]}"
                body_lines.append(tail)

        body = self._format_inset_content("\n".join(body_lines))
        return f"{header}\n{body}"

    def _running_sub_agents(self) -> list[SubAgentActivity]:
        return [a for a in self._sub_agent_activities.values() if a.status == "running"]

    def _refresh_sub_agent_activity_status(self) -> None:
        running = self._running_sub_agents()
        if running:
            if len(running) == 1:
                text = messages.RUNNING_SUB_AGENT.format(name=running[0].agent_name, index=running[0].index)
            else:
                text = messages.RUNNING_SUB_AGENTS.format(count=len(running))
            self._update_activity_progress(text, state=TurnState.RUNNING_SUB_AGENTS)
        elif self._turn_active:
            self._update_activity_progress(messages.WORKING, state=TurnState.GENERATING)

    def _finalize_sub_agent_activities(self, status: str = "stopped") -> None:
        """Mark still-running sub-agents as finished (no lifecycle event arrives on cancel)."""
        changed = False
        for activity in self._sub_agent_activities.values():
            if activity.status == "running":
                activity.status = status
                activity.finished_at = self._now()
                activity.entry.complete = True
                activity.entry.content = self._format_sub_agent_content(activity)
                self._invalidate_conversation(activity.entry)
                changed = True
        if changed:
            self._flush_conversation_render()

    def _tick_running_sub_agents(self) -> None:
        running = self._running_sub_agents()
        if not running:
            return
        for activity in running:
            activity.entry.content = self._format_sub_agent_content(activity)
            self._invalidate_conversation(activity.entry)

    def _tool_result_preview(self, text: str) -> str:
        # The entry header already conveys completion; the body is just the preview.
        return self._truncate_tool_text(text)

    def _truncate_tool_text(self, text: str) -> str:
        if len(text) <= TOOL_RESULT_PREVIEW_CHARS:
            return text
        return f"{text[:TOOL_RESULT_PREVIEW_CHARS]}{theme.g(Glyph.ELLIPSIS)}"

    def _capped_tool_text(self, text: str) -> str:
        if len(text) <= theme.TOOL_FULL_CONTENT_CAP_CHARS:
            return text
        return f"{text[:theme.TOOL_FULL_CONTENT_CAP_CHARS]}{theme.g(Glyph.ELLIPSIS)}"

    def _tool_stream_preview(self, text: str) -> str:
        if len(text) <= TOOL_STREAM_PREVIEW_CHARS:
            return text
        notice = messages.STREAM_TRUNCATED.format(chars=TOOL_STREAM_PREVIEW_CHARS)
        return f"{notice}\n{text[-TOOL_STREAM_PREVIEW_CHARS:]}"

    def _invalidate_conversation(self, entry: Optional[ConversationEntry] = None) -> None:
        """Mark the conversation dirty and coalesce re-renders.

        Hot paths (stream chunks, tool updates, sub-agent ticks) call this
        instead of rendering directly, so rapid event bursts produce at most
        one flush per coalesce interval, and a flush only touches new or
        changed entry widgets.
        """
        if entry is not None:
            self._dirty_entry_ids.add(entry.entry_id)
        if self._render_pending:
            return
        self._render_pending = True
        try:
            self.set_timer(
                theme.RENDER_COALESCE_INTERVAL,
                self._flush_conversation_render,
                name="conversation-render",
            )
        except Exception:
            # Timers are unavailable before the app is running; render directly.
            self._flush_conversation_render()

    def _flush_conversation_render(self) -> None:
        if not self._render_pending:
            return
        self._render_pending = False
        try:
            view = self._conversation
        except Exception:
            # A coalesced flush can fire after the widget is unmounted (e.g. on exit).
            self._dirty_entry_ids.clear()
            return

        rendered_ids = list(self._entry_widgets)
        current_ids = [entry.entry_id for entry in self.conversation_entries]
        if current_ids[: len(rendered_ids)] != rendered_ids:
            # Entries were removed, replaced, or inserted before the end; rebuild.
            self._render_conversation()
            return

        for entry_id in self._dirty_entry_ids:
            widget = self._entry_widgets.get(entry_id)
            if widget is not None:
                widget.refresh_content()
        self._dirty_entry_ids.clear()

        new_entries = self.conversation_entries[len(rendered_ids):]
        if new_entries:
            widgets = []
            for entry in new_entries:
                widget = self._make_entry_widget(entry)
                self._entry_widgets[entry.entry_id] = widget
                widgets.append(widget)
            view.mount(*widgets)
        self._update_jump_button()

    def _render_conversation(self) -> None:
        """Full rebuild of the conversation view (restore, reset, startup changes)."""
        self._render_pending = False
        self._dirty_entry_ids.clear()
        try:
            view = self._conversation
        except Exception:
            return
        view.remove_children()
        self._entry_widgets = {}
        widgets = []
        for entry in self.conversation_entries:
            widget = self._make_entry_widget(entry)
            self._entry_widgets[entry.entry_id] = widget
            widgets.append(widget)
        if widgets:
            view.mount(*widgets)
        view.anchor()
        self._update_jump_button()

    def _make_entry_widget(self, entry: ConversationEntry) -> ConversationEntryWidget | ToolEntryWidget:
        if entry.kind in {"tool_call", "tool_result", "tool_error"}:
            return ToolEntryWidget(entry, self._tool_entry_title)
        return ConversationEntryWidget(entry, self._format_conversation_entry)

    def _update_jump_button(self) -> None:
        try:
            view = self._conversation
            bar = self.query_one("#jump_to_bottom", JumpToBottomBar)
        except Exception:
            return
        bar.display = view.max_scroll_y > 0 and view.scroll_y < view.max_scroll_y - 1

    def on_jump_to_bottom_bar_pressed(self, message: JumpToBottomBar.Pressed) -> None:
        view = self._conversation
        view.scroll_end(animate=False)
        view.anchor()
        self._update_jump_button()

    def _format_conversation_entry(self, entry: ConversationEntry) -> str | Text | Group:
        """Render an entry using the shared header grammar.

        GRAMMAR: <colored glyph> <bold label> [ · state] — body inset beneath.
        """
        if entry.kind == "startup":
            return self._format_startup_entry(entry)
        streaming = None if entry.complete else theme.g(Glyph.ELLIPSIS)
        if entry.kind == "user":
            header = theme.role_header(Glyph.USER, "You", Color.USER)
            return f"{header}\n{self._format_inset_content(entry.content)}"
        if entry.kind == "assistant":
            header = theme.role_header(Glyph.AGENT, "Agent", Color.AGENT, state=streaming)
            if entry.complete and entry.content.strip():
                return self._markdown_entry(header, entry.content)
            return f"{header}\n{self._format_inset_content(entry.content)}"
        if entry.kind == "thinking":
            header = theme.role_header(Glyph.STATUS, "Thinking", Color.THINKING, label_style="dim italic", state=streaming)
            return f"{header}\n{self._format_inset_content(entry.content, style='italic dim')}"
        if entry.kind == "progress":
            color = Color.ERROR if entry.tone == "error" else Color.WARNING
            header = theme.role_header(Glyph.STATUS, "Status", color, state=streaming)
            return f"{header}\n{self._format_inset_content(entry.content)}"
        if entry.kind == "plan":
            header = theme.role_header(Glyph.PLAN, "Plan", Color.SUCCESS)
            if entry.content.strip():
                return self._markdown_entry(header, entry.content)
            return header
        if entry.kind == "question":
            header = theme.role_header(Glyph.QUESTION, "Question", Color.ACCENT)
            return f"{header}\n{self._format_inset_content(entry.content)}"
        if entry.kind == "skill":
            header = theme.role_header(Glyph.PLAN, "Skill", Color.SUCCESS)
            return f"{header}\n{self._format_inset_content(entry.content)}"
        if entry.kind == "sub_agent":
            return entry.content  # pre-formatted markup, see _format_sub_agent_content
        if entry.kind in TOOL_STATE_PRESENTATION:
            state, color = TOOL_STATE_PRESENTATION[entry.kind]
            return self._format_tool_entry(entry, state=state, color=color)
        if entry.kind == "system":
            return f"[dim]{escape(entry.content)}[/dim]"
        return escape(entry.content)

    def _markdown_entry(self, header: str, content: str) -> Group:
        return Group(
            Text.from_markup(header),
            Padding(
                RichMarkdown(content, code_theme=theme.MARKDOWN_CODE_THEME),
                (0, 0, 0, theme.INSET_WIDTH),
            ),
        )

    def _format_startup_entry(self, entry: ConversationEntry) -> Text:
        lines = entry.content.splitlines()
        try:
            separator = lines.index("")
        except ValueError:
            separator = len(STARTUP_WORDMARK)
        rendered = Text()
        logo = "\n".join(lines[:separator])
        if logo:
            rendered.append(logo, style=f"bold {Color.ACCENT}")
        for line in lines[separator + 1 :]:
            rendered.append("\n")
            label, sep, value = line.partition(": ")
            if sep and label and len(label) <= 12:
                # Aligned two-column key/value line: muted label, normal value.
                rendered.append(f"{label + ':':<13}", style="dim")
                rendered.append(value)
            else:
                rendered.append(line, style="dim")
        return rendered

    def _format_tool_entry(self, entry: ConversationEntry, *, state: str, color: str) -> str:
        tool_name = escape(entry.tool_name or "tool")
        header = theme.role_header(Glyph.TOOL, tool_name, color, state=state)
        if not entry.content:
            return header
        return f"{header}\n{self._format_inset_content(entry.content)}"

    def _tool_entry_title(self, entry: ConversationEntry) -> str:
        state, color = TOOL_STATE_PRESENTATION.get(entry.kind, ("running", Color.ACCENT))
        return theme.role_header(Glyph.TOOL, escape(entry.tool_name or "tool"), color, state=state)

    def _format_inset_content(self, content: str, style: Optional[str] = None) -> str:
        bar = f"[dim]  {theme.g(Glyph.INSET_BAR)}[/dim]"
        lines = content.splitlines() or [""]
        if style:
            return "\n".join(f"{bar} [{style}]{escape(line)}[/{style}]" if line else bar for line in lines)
        return "\n".join(f"{bar} {escape(line)}" if line else bar for line in lines)
