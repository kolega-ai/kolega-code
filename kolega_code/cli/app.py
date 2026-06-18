"""Textual application for Kolega Code."""

from __future__ import annotations

import asyncio
import itertools
import json
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
from textual import events
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.css.query import NoMatches
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
    TabbedContent,
    TabPane,
    TextArea,
)
from textual.widgets.option_list import Option

from kolega_code.agent import AgentConfig, AgentEvent, CoderAgent, PlanningAgent, PromptExtension, ToolExtension
from kolega_code.agent.prompt_provider import AgentMode
from kolega_code.agent.prompts import (
    PLANNING_QUESTION_PROMPT,
    SHARED_TASK_LIST_PROMPT,
    build_implement_plan_prompt,
    build_init_agents_prompt,
)
from kolega_code.hooks import HookDispatcher, HookEvent, load_hook_config, project_hooks_present
from kolega_code.llm.exceptions import LLMError, llm_error_message
from kolega_code.llm.models import Message, MessageHistory, TextBlock, ToolCall, ToolResult
from kolega_code.permissions import (
    PermissionDecision,
    PermissionMode,
    PermissionRequest,
    PermissionRuleOption,
    PermissionStoreError,
    ProjectPermissionStore,
    allow_rule_options,
    normalize_permission_mode,
)
from kolega_code.services.browser import PlaywrightBrowserManager
from kolega_code.tools import ASK_USER_CHOICE_INPUT_SCHEMA, ASK_USER_CHOICE_SHAPE_HINT, ToolError

from . import messages, theme
from .config import CliConfigError, CliConfigOverrides, build_agent_config, config_summary, key_status
from .connection import CliConnectionManager
from .file_index import IndexEntry, WorkspaceFileIndex
from .mentions import build_file_attachments
from .provider_registry import (
    UI_DEFAULT_MODEL,
    UI_DEFAULT_PROVIDER,
    default_ui_thinking_effort,
    get_ui_model,
    ui_model_options,
    ui_provider_options,
    ui_thinking_effort_options,
)
from .session_store import SessionRecord, SessionStore
from .settings import CliSettings, SettingsStore
from .skills import (
    SkillCatalog,
    activated_skill_names,
    build_skill_prompt_extension,
    build_skill_tool_extension,
    discover_skills,
    skill_names_in_text,
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
QUESTION_TOOL_NAME = "ask_user_choice"
QUESTION_OPTION_ID_PREFIX = "question_option_"
APPROVAL_OPTION_ID_PREFIX = "approval_option_"
MODEL_OPTION_ID_PREFIX = "model_option_"
EFFORT_OPTION_ID_PREFIX = "effort_option_"
THEME_OPTION_ID_PREFIX = "theme_option_"
QUESTION_PLACEHOLDER = messages.QUESTION_PLACEHOLDER
APPROVAL_PLACEHOLDER = messages.APPROVAL_PLACEHOLDER
MODEL_PLACEHOLDER = messages.MODEL_PLACEHOLDER
EFFORT_PLACEHOLDER = messages.EFFORT_PLACEHOLDER
THEME_PLACEHOLDER = messages.THEME_PLACEHOLDER
STARTUP_WORDMARK = (
    " _  __     _                   ____          _",
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


# Role colors are stored as Color attribute NAMES (not values) so they resolve
# against the active theme at render time instead of snapshotting it at import;
# see theme.apply_theme(), which reassigns the Color attributes on theme switch.
TURN_STATE_STYLES = {
    TurnState.IDLE: "SUCCESS",
    TurnState.STOPPING: "WARNING",
    TurnState.STOPPED: "WARNING",
    TurnState.ERROR: "ERROR",
}

TOOL_STATE_PRESENTATION = {
    "tool_call": ("running", "ACCENT"),
    "tool_result": ("done", "SUCCESS"),
    "tool_error": ("failed", "ERROR"),
}


def turn_state_color(state: TurnState) -> str:
    """Live role color for a turn state (resolves against the active theme)."""
    return getattr(Color, TURN_STATE_STYLES.get(state, "ACCENT"))


def tool_state_presentation(kind: str) -> tuple[str, str]:
    """(state label, live role color) for a tool-entry kind."""
    label, attr = TOOL_STATE_PRESENTATION.get(kind, ("running", "ACCENT"))
    return label, getattr(Color, attr)


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
    options: list[str]  # selectable option labels; the selected label is the answer
    future: asyncio.Future[str]
    descriptions: Optional[list[str]] = None  # per-option descriptions, parallel to options


@dataclass
class PendingApproval:
    request: PermissionRequest
    future: asyncio.Future[PermissionDecision]
    rule_options: list[PermissionRuleOption]


@dataclass
class PendingModelSelection:
    provider: str
    options: list[tuple[str, str]]


@dataclass
class PendingEffortSelection:
    provider: str
    model: str
    options: list[tuple[str, str]]


@dataclass
class PendingThemeSelection:
    # (display name, value); value == display name for themes.
    options: list[tuple[str, str]]


@dataclass
class StatusDashboardState:
    provider: str = UI_DEFAULT_PROVIDER
    model: str = UI_DEFAULT_MODEL
    thinking_effort: Optional[str] = None
    mode: str = BUILD_INTERACTION_MODE
    permission_mode: str = PermissionMode.ASK.value
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
        super().__init__("", markup=False)
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


class ActionList(OptionList):
    """Vertical list of selectable actions (question options, plan decision).

    Unlike CompletionDropdown this list takes focus itself, so arrow keys and
    Enter work directly. Pressing a digit selects the matching option.
    """

    def show_options(self, options: list[Option]) -> None:
        self.clear_options()
        self.add_options(options)
        if options:
            self.highlighted = 0
        self.display = True

    def hide(self) -> None:
        had_focus = self.has_focus
        self.display = False
        self.clear_options()
        if had_focus:
            composer = self.screen.query_one("#composer", ChatComposer)
            if not composer.disabled:
                composer.focus()

    def on_key(self, event: events.Key) -> None:
        if event.character and event.character.isdigit():
            index = int(event.character) - 1
            if 0 <= index < self.option_count:
                self.highlighted = index
                self.action_select()
                event.stop()


class PromptPanel(Vertical):
    """A bordered panel that pairs a prompt header with its ActionList of options.

    The question (or permission request) is shown as the panel header above the
    selectable options, so the prompt and its answers read as a single unit
    instead of a chat bubble disconnected from a separate option box.
    """

    def __init__(self, *, actions_id: str, title: str, **kwargs) -> None:
        super().__init__(**kwargs)
        self._actions_id = actions_id
        self._title = title

    def compose(self) -> ComposeResult:
        self.border_title = self._title
        yield Static("", classes="prompt-header", markup=True)
        yield ActionList(id=self._actions_id)

    @property
    def actions(self) -> ActionList:
        return self.query_one(ActionList)

    def prompt(self, header: str, options: list[Option]) -> None:
        self.query_one(".prompt-header", Static).update(Text.from_markup(header))
        self.actions.show_options(options)
        self.display = True
        # Focus the options synchronously (not via Widget.focus(), which defers to a
        # later event-loop tick): in a real terminal that deferred focus races with the
        # refresh loop and the composer being disabled, so the options never get focus
        # and arrow/Enter keys do nothing. Matches the other selection lists in this file.
        self.screen.set_focus(self.actions)

    def hide(self) -> None:
        self.display = False
        self.actions.hide()


class ChatComposer(TextArea):
    """Multiline chat input that submits on Enter and inserts newlines on Shift+Enter."""

    BINDINGS = [
        *TextArea.BINDINGS,
        Binding("enter", "submit", "Send", priority=True),
        Binding(
            "shift+enter,ctrl+enter,ctrl+j", "insert_newline", "New line", key_display="Shift+Enter", priority=True
        ),
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

    .settings-section {
        height: auto;
        border: round $surface;
        border-title-color: $text;
        border-title-style: bold;
        padding: 0 1;
        margin-bottom: 1;
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
        border: round $surface;
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

    #plan_actions, #model_actions, #effort_actions, #theme_actions {
        display: none;
        height: auto;
        max-height: 12;
        border: round $surface;
        background: $surface;
    }

    /* Question / approval prompts: the question is folded into the panel header
       (the border title) above its options, so the whole prompt reads as one
       bordered unit. The inner ActionList drops its own border. */
    #question_prompt, #approval_prompt {
        display: none;
        height: auto;
        max-height: 14;
        border: round $surface;
        background: $surface;
        border-title-color: $text;
        border-title-style: bold;
        padding: 0 1;
    }

    #question_prompt > ActionList, #approval_prompt > ActionList {
        border: none;
        background: $surface;
        height: auto;
        max-height: 10;
        padding: 0;
    }

    .prompt-header {
        padding: 0 0 1 0;
        background: $surface;
    }

    /* Neutralize the selected-row highlight on every choice list. Textual paints
       it with $block-cursor-background (= $primary, a saturated brand color),
       which clashes with the otherwise-neutral chrome and looks wrong in 256-color
       Terminal.app. Each theme pins $surface-lighten-2 to a near-neutral gray
       (see theme.build_textual_theme) so the highlight stays subtle across all
       themes — incl. Solarized, whose auto-derived $surface-lighten-2 would
       otherwise quantize to a saturated teal. The OptionList type selector also
       covers its subclasses: ActionList, CompletionDropdown, and the Select's
       SelectOverlay dropdown. */
    OptionList > .option-list--option-highlighted {
        background: $surface-lighten-2;
        color: $text;
    }

    .meta {
        color: $text-muted;
    }

    Footer {
        background: $surface;
    }

    Input {
        border: round $surface;
    }

    Input:focus {
        border: round $surface-lighten-2;
    }

    Select > SelectCurrent {
        border: round $surface;
    }

    Select:focus > SelectCurrent {
        border: round $surface-lighten-2;
    }

    Select > SelectOverlay {
        border: round $surface;
    }
    """

    BINDINGS = [
        Binding(
            "shift+tab", "toggle_interaction_mode", "Plan/Build", show=True, key_display="Shift+Tab", priority=True
        ),
        Binding("ctrl+p", "toggle_permission_mode", "Permissions", show=True, key_display="Ctrl+P", priority=True),
        Binding("ctrl+o", "toggle_sidebar", "Sidebar", show=True, key_display="Ctrl+O", priority=True),
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
        self._pending_approval: Optional[PendingApproval] = None
        self._permission_lock = asyncio.Lock()
        self._pending_model_selection: Optional[PendingModelSelection] = None
        self._pending_effort_selection: Optional[PendingEffortSelection] = None
        self._pending_theme_selection: Optional[PendingThemeSelection] = None
        provider, model = self._startup_model()
        self._status_state = StatusDashboardState(
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
                yield ActionList(id="plan_actions")
                yield PromptPanel(
                    id="question_prompt",
                    actions_id="question_actions",
                    title=f"{theme.g(Glyph.QUESTION)} Question",
                )
                yield PromptPanel(
                    id="approval_prompt",
                    actions_id="approval_actions",
                    title=f"{theme.g(Glyph.QUESTION)} Permission",
                )
                yield ActionList(id="model_actions")
                yield ActionList(id="effort_actions")
                yield ActionList(id="theme_actions")
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
                                yield Button("Save Settings", variant="primary", id="save_settings")
                                yield Static("", id="settings_status")
                            with Vertical(classes="settings-section", id="settings_appearance") as appearance_section:
                                appearance_section.border_title = "Appearance"
                                yield Label("Theme")
                                yield Select(
                                    [(name, name) for name in theme.available_themes()],
                                    id="theme_select",
                                    allow_blank=False,
                                    value=theme.DEFAULT_THEME_NAME,
                                )
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
        if self.check_for_updates:
            self.run_worker(self._check_for_update_on_startup(), name="kolega-update-check", group="updates")
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
        """Record a user-facing notice in the Logs tab without showing a transient popup."""
        level = {"information": "ok", "warning": "warn", "error": "error"}.get(severity, "info")
        self._log_status(message, level)

    async def _check_for_update_on_startup(self) -> None:
        result = await asyncio.to_thread(check_for_update)
        message = update_status_message(result)
        if not message:
            return
        self._add_conversation_entry(ConversationEntry(kind="system", content=message))
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
        if interaction_mode in {BUILD_INTERACTION_MODE, PLAN_INTERACTION_MODE}:
            return interaction_mode
        return BUILD_INTERACTION_MODE

    def _sync_planning_state_to_session(self) -> None:
        self.session.interaction_mode = self.interaction_mode
        self.session.permission_mode = self.permission_mode.value
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

        if self._pending_model_selection is not None:
            if not stripped_text:
                self._set_composer_status(MODEL_PLACEHOLDER)
                return
            event.composer.load_text("")
            await self._answer_model_selection(stripped_text)
            return

        if self._pending_effort_selection is not None:
            if not stripped_text:
                self._set_composer_status(EFFORT_PLACEHOLDER)
                return
            event.composer.load_text("")
            await self._answer_effort_selection(stripped_text)
            return

        if self._pending_theme_selection is not None:
            if not stripped_text:
                self._set_composer_status(THEME_PLACEHOLDER)
                return
            event.composer.load_text("")
            await self._answer_theme_selection(stripped_text)
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

        if self._pending_approval is not None:
            self._set_composer_status(APPROVAL_PLACEHOLDER)
            self._notify_user(messages.BLOCK_PENDING_APPROVAL, severity="warning")
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
            elif event.option_id == "discuss_plan":
                self._discuss_pending_plan()
            return
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
            self._cancel_pending_approval()
            await self._drain_pending_events()
            self._finalize_sub_agent_activities()
            self._save_session_history()
            self._finish_turn_progress(messages.STOPPED_BY_USER, TurnState.STOPPED)
            self._log_status(messages.STOPPED_BY_USER, "warn")
        except LLMError as exc:
            self._cancel_pending_question()
            self._cancel_pending_approval()
            await self._drain_pending_events()
            self._finalize_sub_agent_activities()
            self._save_session_history()
            model = self.config.long_context_config.model if self.config is not None else None
            message_text = llm_error_message(exc, model=model)
            self._finish_turn_progress(message_text, TurnState.ERROR)
            self._log_status(message_text, "error")
        except Exception as exc:
            self._cancel_pending_question()
            self._cancel_pending_approval()
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

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "provider_select":
            provider = str(event.value)
            try:
                model_select = self.query_one("#model_select", Select)
                api_key_input = self.query_one("#api_key_input", Input)
            except NoMatches:
                return
            model_options = ui_model_options(provider)
            model_select.set_options(model_options)
            model = model_options[0][1] if model_options else UI_DEFAULT_MODEL
            if model_options:
                model_select.value = model
            self._set_effort_select_default(provider, model)
            api_key_input.placeholder = self._api_key_placeholder(provider)
            return

        if event.select.id == "model_select":
            try:
                provider = str(self.query_one("#provider_select", Select).value)
            except NoMatches:
                return
            self._set_effort_select_default(provider, str(event.value))
            return

        if event.select.id == "theme_select":
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
            self._cancel_pending_approval()
            self.agent_worker.cancel()
            self._notify_user(messages.CANCEL_REQUESTED, severity="warning")

    def _mode_switch_blocked(self) -> bool:
        if self._pending_approval is not None:
            self._set_composer_status(APPROVAL_PLACEHOLDER)
            self._notify_user(messages.BLOCK_PENDING_APPROVAL, severity="warning")
            return True
        if self._turn_active or self.agent_worker is not None:
            self._show_composer_hint(messages.BLOCK_STOP_BEFORE_MODE_SWITCH)
            self._notify_user(messages.BLOCK_STOP_BEFORE_MODE_SWITCH, severity="warning")
            return True
        if self._plan_decision_active:
            self._set_composer_status(PLAN_READY_PLACEHOLDER)
            self._notify_user(messages.BLOCK_PLAN_DECISION_MODE_SWITCH, severity="warning")
            return True
        return False

    def _permission_mode_switch_blocked(self) -> bool:
        if self._pending_approval is not None:
            self._set_composer_status(APPROVAL_PLACEHOLDER)
            self._notify_user(messages.BLOCK_PENDING_APPROVAL_MODE_SWITCH, severity="warning")
            return True
        return False

    async def action_toggle_interaction_mode(self) -> None:
        if self._mode_switch_blocked():
            return

        target = PLAN_INTERACTION_MODE if self.interaction_mode == BUILD_INTERACTION_MODE else BUILD_INTERACTION_MODE
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

    async def action_quit(self) -> None:
        if self.agent is not None:
            fire = getattr(self.agent, "fire_hook", None)
            if fire is not None:
                try:
                    await fire(HookEvent.SESSION_END, {"reason": "quit"})
                except Exception:
                    pass
            self.session.history = self.agent.dump_message_history()
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
                composer = self.query_one("#composer", ChatComposer)
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
        self._update_settings_status()

    def _set_effort_select_default(self, provider: str, model: str) -> None:
        try:
            effort_select = self.query_one("#thinking_effort_select", Select)
        except Exception:
            return
        effort_options = ui_thinking_effort_options(provider, model)
        effort_select.set_options(effort_options)
        default_effort = default_ui_thinking_effort(provider, model)
        if default_effort is not None:
            effort_select.value = default_effort

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
            permission_mode=self.permission_mode,
            permission_callback=self._permission_callback,
            hook_dispatcher=self._session_hook_dispatcher(),
        )
        if history:
            self.agent.restore_message_history(history)
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
            self._add_conversation_entry(ConversationEntry(kind="system", content=outcome.additional_context))

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
        try:
            self.screen.set_focus(self.query_one("#plan_actions", ActionList))
        except Exception:
            pass
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

        prompt = build_implement_plan_prompt(plan)
        self._add_conversation_entry(ConversationEntry(kind="user", content="Implement the approved plan."))
        self.agent_worker = self.run_worker(
            self._process_message(prompt), name="kolega-turn", group="turns", exclusive=True
        )

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
            plan_actions = self.query_one("#plan_actions", ActionList)
            if visible:
                options = [Option("Implement plan", id="implement_plan")]
                if allow_discuss:
                    options.append(Option("Discuss further", id="discuss_plan"))
                plan_actions.show_options(options)
            else:
                plan_actions.hide()
        except Exception:
            return

    def _set_effort_actions_visible(self, visible: bool) -> None:
        try:
            effort_actions = self.query_one("#effort_actions", ActionList)
            if visible and self._pending_effort_selection is not None:
                effort_actions.show_options(
                    [
                        Option(
                            self._effort_option_label(index, label, value),
                            id=f"{EFFORT_OPTION_ID_PREFIX}{index}",
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
            model_actions = self.query_one("#model_actions", ActionList)
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
                            id=f"{MODEL_OPTION_ID_PREFIX}{index}",
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
            theme_actions = self.query_one("#theme_actions", ActionList)
            if visible and self._pending_theme_selection is not None:
                theme_actions.show_options(
                    [
                        Option(
                            self._theme_option_label(index, name),
                            id=f"{THEME_OPTION_ID_PREFIX}{index}",
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
        composer.disabled = not enabled or self._plan_decision_active or self._pending_approval is not None

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
            "/init": self._command_init,
            "/plan": self._command_plan,
            "/build": self._command_build,
            "/sidebar": self._command_sidebar,
            "/permissions": self._command_permissions,
            "/model": self._command_model,
            "/effort": self._command_effort,
            "/theme": self._command_theme,
            "/copy": self._command_copy,
            "/version": self._command_version,
            "/update": self._command_update,
            "/quit": self._command_quit,
            "/exit": self._command_quit,
        }

    async def _handle_tui_slash_command(self, stripped_text: str, composer: ChatComposer) -> bool:
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

    async def _command_plan(self, args: str) -> None:
        if self._mode_switch_blocked():
            return
        await self._set_interaction_mode(PLAN_INTERACTION_MODE)

    async def _command_build(self, args: str) -> None:
        if self._mode_switch_blocked():
            return
        await self._set_interaction_mode(BUILD_INTERACTION_MODE)

    async def _command_sidebar(self, args: str) -> None:
        await self.action_toggle_sidebar()

    async def _command_init(self, args: str) -> None:
        if self._pending_question is not None:
            self._set_composer_status(QUESTION_PLACEHOLDER)
            self._notify_user(messages.BLOCK_PENDING_QUESTION_INIT, severity="warning")
            return

        if self._pending_approval is not None:
            self._set_composer_status(APPROVAL_PLACEHOLDER)
            self._notify_user(messages.BLOCK_PENDING_APPROVAL, severity="warning")
            return

        if self._turn_active or self.agent_worker is not None:
            self._show_composer_hint(messages.BLOCK_STOP_BEFORE_INIT)
            self._notify_user(messages.BLOCK_STOP_BEFORE_INIT, severity="warning")
            return

        if self._plan_decision_active:
            self._set_composer_status(PLAN_READY_PLACEHOLDER)
            self._notify_user(messages.BLOCK_PLAN_DECISION_INIT, severity="warning")
            return

        if self.agent is None:
            self._set_settings_status(messages.SETTINGS_REQUIRED, tone="warning")
            return

        if self.interaction_mode != BUILD_INTERACTION_MODE:
            await self._set_interaction_mode(BUILD_INTERACTION_MODE)

        if self.agent is None:
            self._set_settings_status(messages.SETTINGS_REQUIRED, tone="warning")
            return

        prompt = build_init_agents_prompt(args)
        transcript = "/init" if not args else f"/init {args}"
        self._add_conversation_entry(ConversationEntry(kind="user", content=transcript))
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
            self._add_conversation_entry(ConversationEntry(kind="system", content="\n".join(lines)))
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
            self._add_conversation_entry(ConversationEntry(kind="system", content="\n".join(lines)))
            self._pending_model_selection = PendingModelSelection(provider=provider, options=model_options)
            self._cancel_pending_effort_selection()
            self._show_model_options()
            self._set_composer_status(MODEL_PLACEHOLDER)
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
            self._set_composer_status(MODEL_PLACEHOLDER)
            return

        matched = self._match_model_value(pending.options, clean_answer)
        if matched is None:
            self._set_composer_status(MODEL_PLACEHOLDER)
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
            self._add_conversation_entry(ConversationEntry(kind="system", content="\n".join(lines)))
            self._pending_effort_selection = PendingEffortSelection(
                provider=provider,
                model=model,
                options=effort_options,
            )
            self._show_effort_options()
            self._set_composer_status(EFFORT_PLACEHOLDER)
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
            self._set_composer_status(EFFORT_PLACEHOLDER)
            return

        matched = self._match_effort_value(pending.options, clean_answer)
        if matched is None:
            self._set_composer_status(EFFORT_PLACEHOLDER)
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
            self._add_conversation_entry(ConversationEntry(kind="system", content="\n".join(lines)))
            self._pending_theme_selection = PendingThemeSelection(
                options=[(name, name) for name in theme.available_themes()]
            )
            self._show_theme_options()
            self._set_composer_status(THEME_PLACEHOLDER)
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
            self._set_composer_status(THEME_PLACEHOLDER)
            return
        matched = self._match_theme_value(clean_answer)
        if matched is None:
            self._set_composer_status(THEME_PLACEHOLDER)
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
        self._add_conversation_entry(ConversationEntry(kind="system", content="\n".join(lines)))

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
        self._add_conversation_entry(ConversationEntry(kind="system", content=content))
        self._notify_user(lines[0], severity=severity)

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

        if self._pending_approval is not None:
            self._set_composer_status(APPROVAL_PLACEHOLDER)
            self._notify_user(messages.BLOCK_PENDING_APPROVAL, severity="warning")
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
        async def ask_user_choice(questions: list[dict]) -> str:
            """
            Ask the user one or more multiple-choice planning questions and wait for their answers.

            Use this only for planning decisions that materially affect the final plan. Each question has a
            short `header`, the `question` text, a `multiSelect` flag, and an `options` array of
            `{label, description}` choices. The user selects one option per question or types a custom
            free-text answer. Questions are asked one at a time, in order.

            Returns:
                A JSON object mapping each question's header (or its text) to the chosen option label
                or the user's custom answer.
            """
            if self.interaction_mode != PLAN_INTERACTION_MODE:
                raise ToolError("ask_user_choice is only available in planning mode.")

            normalized = self._normalize_choice_questions(questions)
            if self._pending_question is not None:
                raise ToolError("A planning question is already waiting for an answer.")

            answers: dict[str, str] = {}
            for clean_question, header, labels, descriptions in normalized:
                answer = await self._ask_user_choice(clean_question, labels, descriptions)
                answers[header or clean_question] = answer
            return json.dumps(answers)

        return ToolExtension(
            name="cli-planning-questions",
            tools={QUESTION_TOOL_NAME: ask_user_choice},
            tool_schemas={QUESTION_TOOL_NAME: ASK_USER_CHOICE_INPUT_SCHEMA},
            tool_groups={"planning_tools": [QUESTION_TOOL_NAME]},
        )

    def _normalize_choice_questions(self, questions: object) -> list[tuple[str, str, list[str], list[str]]]:
        """Validate the structured questions input and return normalized questions.

        Strict: rejects malformed input with an instructive ToolError instead of coercing.
        Each result is (question_text, header, option_labels, option_descriptions).
        """
        if not isinstance(questions, list) or not questions:
            raise ToolError("'questions' must be a non-empty array of question objects. " + ASK_USER_CHOICE_SHAPE_HINT)

        normalized: list[tuple[str, str, list[str], list[str]]] = []
        for question in questions:
            if not isinstance(question, dict):
                raise ToolError("each item in 'questions' must be an object. " + ASK_USER_CHOICE_SHAPE_HINT)

            clean_question = str(question.get("question", "")).strip()
            if not clean_question:
                raise ToolError("each question must include non-empty 'question' text. " + ASK_USER_CHOICE_SHAPE_HINT)

            header = str(question.get("header", "")).strip()

            raw_options = question.get("options")
            if not isinstance(raw_options, list):
                raise ToolError(
                    "each question's 'options' must be an array of {label, description} objects. "
                    + ASK_USER_CHOICE_SHAPE_HINT
                )

            labels: list[str] = []
            descriptions: list[str] = []
            for option in raw_options:
                if not isinstance(option, dict):
                    raise ToolError(
                        "each option must be an object with a 'label' (and ideally a 'description'). "
                        + ASK_USER_CHOICE_SHAPE_HINT
                    )
                label = str(option.get("label", "")).strip()
                if not label:
                    continue
                labels.append(label)
                descriptions.append(str(option.get("description", "")).strip())

            if len(labels) < 2:
                raise ToolError(
                    "each question needs at least two options, each with a non-empty 'label'. "
                    + ASK_USER_CHOICE_SHAPE_HINT
                )

            normalized.append((clean_question, header, labels, descriptions))

        return normalized

    async def _ask_user_choice(
        self, question: str, options: list[str], descriptions: Optional[list[str]] = None
    ) -> str:
        loop = asyncio.get_running_loop()
        future: asyncio.Future[str] = loop.create_future()
        self._pending_question = PendingQuestion(
            question=question, options=options, future=future, descriptions=descriptions
        )
        self._show_question_options(question, options, descriptions)
        self._set_composer_status(QUESTION_PLACEHOLDER)
        self._set_chat_enabled(True)
        self._update_activity_progress(messages.WAITING_FOR_ANSWER, state=TurnState.WAITING_FOR_USER)

        try:
            return await future
        finally:
            if self._pending_question is not None and self._pending_question.future is future:
                self._pending_question = None
                self._set_question_actions_visible(False)

    async def _answer_question_option(self, option_index: int) -> None:
        if self._pending_question is None:
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
        self._add_conversation_entry(ConversationEntry(kind="question", content=pending_question.question))
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

    def _show_question_options(
        self, question: str, options: list[str], descriptions: Optional[list[str]] = None
    ) -> None:
        try:
            panel = self.query_one("#question_prompt", PromptPanel)
        except Exception:
            return
        option_widgets = [
            Option(
                self._question_option_label(index, option, self._option_description(descriptions, index)),
                id=f"{QUESTION_OPTION_ID_PREFIX}{index}",
            )
            for index, option in enumerate(options)
        ]
        panel.prompt(escape(question), option_widgets)

    async def _permission_callback(self, request: PermissionRequest) -> PermissionDecision:
        if self.permission_mode != PermissionMode.ASK:
            return PermissionDecision(allowed=True)

        async with self._permission_lock:
            store = ProjectPermissionStore(self.project_path)
            try:
                matched_rule = store.first_match(request)
            except PermissionStoreError as exc:
                matched_rule = None
                self._notify_user(str(exc), severity="warning")

            if matched_rule is not None:
                return PermissionDecision(allowed=True, reason=f"Allowed by saved rule {matched_rule.id}.")

            return await self._ask_permission(request)

    async def _ask_permission(self, request: PermissionRequest) -> PermissionDecision:
        loop = asyncio.get_running_loop()
        future: asyncio.Future[PermissionDecision] = loop.create_future()
        rule_options = allow_rule_options(request)
        self._pending_approval = PendingApproval(request=request, future=future, rule_options=rule_options)
        self._show_approval_options(rule_options)
        self._set_composer_status(APPROVAL_PLACEHOLDER)
        self._set_chat_enabled(False)
        self._update_activity_progress(messages.WAITING_FOR_PERMISSION, state=TurnState.WAITING_FOR_USER)

        try:
            return await future
        finally:
            if self._pending_approval is not None and self._pending_approval.future is future:
                self._pending_approval = None
                self._set_approval_actions_visible(False)

    def _show_approval_options(self, rule_options: list[PermissionRuleOption]) -> None:
        self._set_approval_actions_visible(True)

    async def _answer_approval_option(self, option_index: int) -> None:
        pending = self._pending_approval
        if pending is None:
            return

        decision: PermissionDecision
        if option_index == 0:
            decision = PermissionDecision(allowed=True, reason="Allowed once by the user.")
            chosen_label = "Allow once"
        elif option_index == 1:
            decision = PermissionDecision(allowed=False, reason="Denied by the user.")
            chosen_label = "Deny"
        else:
            rule_index = option_index - 2
            if rule_index < 0 or rule_index >= len(pending.rule_options):
                return
            rule = pending.rule_options[rule_index].rule
            chosen_label = pending.rule_options[rule_index].label
            try:
                ProjectPermissionStore(self.project_path).add_rule(rule)
            except PermissionStoreError as exc:
                self._notify_user(str(exc), severity="warning")
                decision = PermissionDecision(allowed=True, reason="Allowed once because the rule could not be saved.")
            else:
                decision = PermissionDecision(allowed=True, reason="Allowed by a saved rule.", rule=rule)

        self._pending_approval = None
        self._set_approval_actions_visible(False)
        self._add_conversation_entry(
            ConversationEntry(kind="question", content=self._format_permission_content(pending.request))
        )
        self._add_conversation_entry(ConversationEntry(kind="user", content=chosen_label))
        if not pending.future.done():
            pending.future.set_result(decision)

        if self._turn_active:
            self._restore_composer_placeholder()
            self._set_chat_enabled(False)
            self._update_progress(messages.WORKING, complete=False, state=TurnState.GENERATING)
        else:
            self._restore_composer_placeholder()
            self._set_chat_enabled(self.agent is not None)

    def _format_permission_content(self, request: PermissionRequest) -> str:
        if request.kind.value == "command":
            return "\n".join(["Allow the agent to run this command?", "", request.command])
        target = f" on {request.path}" if request.path else ""
        return f"Allow the agent to run {request.tool_name}{target}?"

    def _show_effort_options(self) -> None:
        self._set_effort_actions_visible(True)
        try:
            effort_actions = self.query_one("#effort_actions", ActionList)
            self.screen.set_focus(effort_actions)
        except Exception:
            return

    def _show_model_options(self) -> None:
        self._set_model_actions_visible(True)
        try:
            model_actions = self.query_one("#model_actions", ActionList)
            self.screen.set_focus(model_actions)
        except Exception:
            return

    def _show_theme_options(self) -> None:
        self._set_theme_actions_visible(True)
        try:
            theme_actions = self.query_one("#theme_actions", ActionList)
            self.screen.set_focus(theme_actions)
        except Exception:
            return

    def _set_question_actions_visible(self, visible: bool) -> None:
        try:
            panel = self.query_one("#question_prompt", PromptPanel)
        except Exception:
            return
        if visible:
            panel.display = True
            panel.actions.display = True
        else:
            panel.hide()

    def _set_approval_actions_visible(self, visible: bool) -> None:
        try:
            panel = self.query_one("#approval_prompt", PromptPanel)
        except Exception:
            return
        if visible and self._pending_approval is not None:
            labels = ["Allow once", "Deny", *(option.label for option in self._pending_approval.rule_options)]
            options = [
                Option(self._question_option_label(index, label), id=f"{APPROVAL_OPTION_ID_PREFIX}{index}")
                for index, label in enumerate(labels)
            ]
            panel.prompt(escape(self._format_permission_content(self._pending_approval.request)), options)
        else:
            panel.hide()

    def _cancel_pending_question(self) -> None:
        pending_question = self._pending_question
        if pending_question is not None and not pending_question.future.done():
            pending_question.future.cancel()
        self._pending_question = None
        self._set_question_actions_visible(False)

    def _cancel_pending_approval(self) -> None:
        pending_approval = self._pending_approval
        if pending_approval is not None and not pending_approval.future.done():
            pending_approval.future.cancel()
        self._pending_approval = None
        self._set_approval_actions_visible(False)

    def _question_option_label(self, index: int, option: str, description: str = "") -> str:
        if description:
            return f"{index + 1}. {option} — {description}"
        return f"{index + 1}. {option}"

    @staticmethod
    def _option_description(descriptions: Optional[list[str]], index: int) -> str:
        if descriptions is not None and 0 <= index < len(descriptions):
            return descriptions[index]
        return ""

    def _cancel_pending_model_selection(self) -> None:
        self._pending_model_selection = None
        self._set_model_actions_visible(False)

    def _model_option_label(self, index: int, label: str, value: str, provider: str) -> str:
        current_provider, current_model = self._startup_model()
        current_suffix = " current" if provider == current_provider and value == current_model else ""
        return f"{index + 1}. {label} ({value}){current_suffix}"

    def _cancel_pending_effort_selection(self) -> None:
        self._pending_effort_selection = None
        self._set_effort_actions_visible(False)

    def _effort_option_label(self, index: int, label: str, value: str) -> str:
        current_suffix = " current" if value == self._startup_thinking_effort() else ""
        return f"{index + 1}. {label} ({value}){current_suffix}"

    def _cancel_pending_theme_selection(self) -> None:
        self._pending_theme_selection = None
        self._set_theme_actions_visible(False)

    def _theme_option_label(self, index: int, name: str) -> str:
        current = self.settings.active_theme or theme.DEFAULT_THEME_NAME
        current_suffix = " current" if name == current else ""
        return f"{index + 1}. {name}{current_suffix}"

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
        model_display = f"{provider}/{model}" if model else provider
        effort = self._startup_thinking_effort() or "not supported"
        api_key = (
            key_status(provider, self.project_path, self.settings)
            if model
            else "not checked until a model is configured"
        )
        return "\n".join(
            [
                *STARTUP_WORDMARK,
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
        turn_style = turn_state_color(state.turn_state)
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
        self._turn_timer = self.set_interval(
            theme.SPINNER_INTERVAL, self._refresh_turn_status_strip, name="turn-status"
        )
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
        self._cancel_pending_model_selection()
        self._cancel_pending_effort_selection()
        self._cancel_pending_theme_selection()
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
            return "\n\n".join(item.to_markdown() if hasattr(item, "to_markdown") else str(item) for item in content)
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
        return str(info.get("agent_id") or info.get("parent_tool_call_id") or info.get("agent_name") or event.sender)

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
        header = Text.from_markup(self._sub_agent_header(activity)).plain
        body = "\n".join(self._sub_agent_body_lines(activity))
        return f"{header}\n{body}" if body else header

    def _format_sub_agent_renderable(self, activity: SubAgentActivity) -> Text | Group:
        return self._entry_renderable(
            self._sub_agent_header(activity),
            "\n".join(self._sub_agent_body_lines(activity)),
        )

    def _sub_agent_header(self, activity: SubAgentActivity) -> str:
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

        return theme.role_header(
            Glyph.SUB_AGENT,
            escape(activity.agent_name),
            color,
            state=f"#{activity.index} {theme.g(Glyph.BULLET_SEP)} {state}",
        )

    def _sub_agent_body_lines(self, activity: SubAgentActivity) -> list[str]:
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

        return body_lines

    def _sub_agent_activity_for_entry(self, entry: ConversationEntry) -> Optional[SubAgentActivity]:
        for activity in self._sub_agent_activities.values():
            if activity.entry is entry:
                return activity
        return None

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
        return f"{text[: theme.TOOL_FULL_CONTENT_CAP_CHARS]}{theme.g(Glyph.ELLIPSIS)}"

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

        new_entries = self.conversation_entries[len(rendered_ids) :]
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

    def _format_conversation_entry(self, entry: ConversationEntry) -> Text | Group:
        """Render an entry using the shared header grammar.

        GRAMMAR: <colored glyph> <bold label> [ · state] — body inset beneath.
        """
        if entry.kind == "startup":
            return self._format_startup_entry(entry)
        streaming = None if entry.complete else theme.g(Glyph.ELLIPSIS)
        if entry.kind == "user":
            header = theme.role_header(Glyph.USER, "You", Color.USER)
            return self._entry_renderable(header, entry.content)
        if entry.kind == "assistant":
            header = theme.role_header(Glyph.AGENT, "Agent", Color.AGENT, state=streaming)
            if entry.complete and entry.content.strip():
                return self._markdown_entry(header, entry.content)
            return self._entry_renderable(header, entry.content)
        if entry.kind == "thinking":
            header = theme.role_header(
                Glyph.STATUS, "Thinking", Color.THINKING, label_style="dim italic", state=streaming
            )
            return self._entry_renderable(header, entry.content, body_style="italic dim")
        if entry.kind == "progress":
            color = Color.ERROR if entry.tone == "error" else Color.WARNING
            header = theme.role_header(Glyph.STATUS, "Status", color, state=streaming)
            return self._entry_renderable(header, entry.content)
        if entry.kind == "plan":
            header = theme.role_header(Glyph.PLAN, "Plan", Color.SUCCESS)
            if entry.content.strip():
                return self._markdown_entry(header, entry.content)
            return Text.from_markup(header)
        if entry.kind == "question":
            header = theme.role_header(Glyph.QUESTION, "Question", Color.ACCENT)
            return self._entry_renderable(header, entry.content)
        if entry.kind == "skill":
            header = theme.role_header(Glyph.PLAN, "Skill", Color.SUCCESS)
            return self._entry_renderable(header, entry.content)
        if entry.kind == "sub_agent":
            activity = self._sub_agent_activity_for_entry(entry)
            if activity is not None:
                return self._format_sub_agent_renderable(activity)
            return Text(entry.content)
        if entry.kind in TOOL_STATE_PRESENTATION:
            state, color = tool_state_presentation(entry.kind)
            return self._format_tool_entry(entry, state=state, color=color)
        if entry.kind == "system":
            return Text(entry.content, style="dim")
        return Text(entry.content)

    def _entry_renderable(
        self,
        header: str,
        body: Optional[str] = None,
        *,
        body_style: Optional[str] = None,
    ) -> Text | Group:
        header_text = Text.from_markup(header)
        if body is None:
            return header_text
        return Group(header_text, self._format_inset_text(body, style=body_style))

    def _markdown_entry(self, header: str, content: str) -> Group:
        return Group(
            Text.from_markup(header),
            Padding(
                RichMarkdown(content, code_theme=theme.markdown_code_theme()),
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
        logo_lines = lines[:separator]
        if logo_lines:
            top, bottom = theme.splash_colors()
            gradient = (
                theme.gradient_hex(top, bottom, len(logo_lines)) if theme.supports_truecolor(self.console) else []
            )
            if gradient:
                # Two-tone vertical gradient: accent at top -> secondary at bottom.
                for index, line in enumerate(logo_lines):
                    if index:
                        rendered.append("\n")
                    rendered.append(line, style=f"bold {gradient[index]}")
            else:
                # 256-color terminal: flat bold primary (matches the primary buttons).
                rendered.append("\n".join(logo_lines), style=f"bold {top}")
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

    def _format_tool_entry(self, entry: ConversationEntry, *, state: str, color: str) -> Text | Group:
        tool_name = escape(entry.tool_name or "tool")
        header = theme.role_header(Glyph.TOOL, tool_name, color, state=state)
        if not entry.content:
            return Text.from_markup(header)
        return self._entry_renderable(header, entry.content)

    def _tool_entry_title(self, entry: ConversationEntry) -> str:
        state, color = tool_state_presentation(entry.kind)
        return theme.role_header(Glyph.TOOL, escape(entry.tool_name or "tool"), color, state=state)

    def _format_inset_text(self, content: str, style: Optional[str] = None) -> Text:
        bar = f"  {theme.g(Glyph.INSET_BAR)}"
        lines = content.splitlines() or [""]
        rendered = Text()
        for index, line in enumerate(lines):
            if index:
                rendered.append("\n")
            rendered.append(bar, style="dim")
            if line:
                rendered.append(" ")
                rendered.append(line, style=style)
        return rendered
