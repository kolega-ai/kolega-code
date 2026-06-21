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
from textual.screen import ModalScreen
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
from textual.widgets._collapsible import CollapsibleTitle
from textual.widgets.option_list import Option

from kolega_code.agent import AgentConfig, AgentEvent, CoderAgent, PlanningAgent, PromptExtension, ToolExtension
from kolega_code.auth import constants as chatgpt_constants
from kolega_code.auth.chatgpt_oauth import run_login_flow
from kolega_code.agent.prompt_provider import AgentMode
from kolega_code.agent.prompts import (
    PLANNING_QUESTION_PROMPT,
    SHARED_TASK_LIST_PROMPT,
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
    edit_preview: Optional[dict] = None  # UI-only structured diff/head preview for edit tools (not persisted)
    entry_id: str = field(default_factory=_next_entry_id)  # UI-only widget key, not persisted


@dataclass
class SubAgentActivity:
    """Live display state for one dispatched sub-agent."""

    agent_id: str
    agent_name: str
    task: str
    index: int  # display ordinal within the turn: #1, #2, ...
    entry: ConversationEntry  # kind="sub_agent", updated in place
    task_full: str = ""  # untruncated task, shown as the first step in the inspector
    status: str = "running"  # running | completed | failed | stopped
    tool_calls: int = 0
    last_activity: str = ""
    started_at: float = 0.0
    finished_at: Optional[float] = None
    stream_buffers: dict[str, str] = field(default_factory=dict)  # chunk uuid -> accumulated text
    active_stream_uuid: Optional[str] = None
    # Full captured trajectory for the inspector. Steps are ConversationEntry objects so
    # the existing entry widgets/renderers display them verbatim. They live on the activity
    # (not in the top-level conversation) so the main transcript stays an append-only summary.
    steps: list[ConversationEntry] = field(default_factory=list)
    tool_steps: dict[str, ConversationEntry] = field(default_factory=dict)  # tool_call_id/name -> step
    stream_steps: dict[str, ConversationEntry] = field(default_factory=dict)  # chunk uuid -> step
    tokens: Optional[int] = None  # cumulative tokens consumed, from lifecycle events
    # Context-window occupancy (how full this sub-agent's own context is right now),
    # from llm_context_update events. Distinct from cumulative `tokens` above.
    context_percentage: Optional[float] = None
    context_input_tokens: Optional[int] = None
    context_max_tokens: Optional[int] = None
    depth: int = 1
    current_action: str = ""  # live "now:" action while running
    workflow_run_id: str = ""  # set when dispatched by a workflow, for card rollup
    workflow_phase: str = ""  # the workflow phase this agent runs under


@dataclass
class PhaseState:
    """One stage of a workflow, shown as a checklist row on the workflow card."""

    title: str
    detail: str = ""
    state: str = "pending"  # pending | active | done | failed
    agents_total: int = 0
    agents_done: int = 0


@dataclass
class WorkflowActivity:
    """Live display state for one running workflow ("gigacode"), shown inline as a card."""

    run_id: str
    name: str
    description: str
    entry: ConversationEntry  # kind="workflow", updated in place
    phases: list[PhaseState] = field(default_factory=list)
    status: str = "running"  # running | completed | failed | stopped
    current_phase: str = ""
    latest_log: str = ""
    agent_count: int = 0
    tokens: int = 0
    started_at: float = 0.0
    finished_at: Optional[float] = None

    def phase_by_title(self, title: str) -> Optional[PhaseState]:
        for phase in self.phases:
            if phase.title == title:
                return phase
        return None


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
    is_compacting: bool = False
    compaction_message: str = ""


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
        strip = _with_selection_style(super().render_line(y), self.text_selection, y, self.selection_style)
        return _with_selection_offsets(strip, y)

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


def _with_selection_offsets(strip: Strip, y: int) -> Strip:
    """Tag rendered segments so Textual can map mouse positions to text offsets."""
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


def _with_selection_style(strip: Strip, selection: Selection | None, y: int, selection_style: Style) -> Strip:
    """Apply the screen selection highlight to Rich renderables that Textual can't style natively.

    Only the selection *background* is applied; each segment keeps its own foreground
    color. The default ``$screen-selection-foreground`` is ``"transparent"``, so adding
    the full selection style would override every segment's foreground and render the
    selected glyphs invisible. Applying the background alone keeps selected text readable
    across every theme while preserving per-role colors (user/agent/code).
    """
    if selection is None:
        return strip

    span = selection.get_span(y)
    if span is None:
        return strip

    start, end = span
    line_length = sum(len(segment.text) for segment in strip if not segment.control)
    if end == -1:
        end = line_length
    start = max(0, min(start, line_length))
    end = max(start, min(end, line_length))
    if start == end:
        return strip

    # Background only: Style.from_color(bgcolor=None) is an empty style, so a missing
    # selection background degrades to "no highlight" rather than blanking the text.
    selection_background = Style.from_color(bgcolor=selection_style.bgcolor)

    selected_segments: list[Segment] = []
    source_x = 0
    for segment in strip:
        if segment.control:
            selected_segments.append(segment)
            continue

        text = segment.text
        segment_start = source_x
        segment_end = source_x + len(text)
        source_x = segment_end

        if segment_end <= start or segment_start >= end:
            selected_segments.append(segment)
            continue

        before_end = max(0, start - segment_start)
        selected_start = before_end
        selected_end = min(len(text), end - segment_start)

        if before_end:
            selected_segments.append(Segment(text[:before_end], segment.style, segment.control))

        selected_text = text[selected_start:selected_end]
        if selected_text:
            style = segment.style + selection_background if segment.style is not None else selection_background
            selected_segments.append(Segment(selected_text, style, segment.control))

        if selected_end < len(text):
            selected_segments.append(Segment(text[selected_end:], segment.style, segment.control))

    return Strip(selected_segments, strip.cell_length)


class SelectableCollapsibleTitle(CollapsibleTitle):
    """Collapsible title that still participates in Textual text selection."""

    ALLOW_SELECT = True

    def render_line(self, y: int) -> Strip:
        return _with_selection_offsets(super().render_line(y), y)


class SelectableCollapsible(Collapsible):
    """Collapsible with a selectable title, used for transcript tool entries."""

    class Contents(Collapsible.Contents):
        """Selectable padding/content wrapper so drags can start in the expanded tool indent."""

        ALLOW_SELECT = True

        def render_line(self, y: int) -> Strip:
            return _with_selection_offsets(super().render_line(y), y)

        def get_selection(self, selection: Selection) -> tuple[str, str] | None:
            return "", ""

    def __init__(
        self,
        *children,
        title: str = "Toggle",
        collapsed: bool = True,
        collapsed_symbol: str = "▶",
        expanded_symbol: str = "▼",
        **kwargs,
    ) -> None:
        super().__init__(
            *children,
            title=title,
            collapsed=collapsed,
            collapsed_symbol=collapsed_symbol,
            expanded_symbol=expanded_symbol,
            **kwargs,
        )
        self._title = SelectableCollapsibleTitle(
            label=title,
            collapsed_symbol=collapsed_symbol,
            expanded_symbol=expanded_symbol,
            collapsed=collapsed,
        )


class ToolEntryWidget(Vertical):
    """Tool entry rendered as a collapsed-by-default Collapsible with the full output inside."""

    def __init__(
        self,
        entry: ConversationEntry,
        title_factory: Callable[[ConversationEntry], str],
        preview_factory: Optional[Callable[[ConversationEntry], object]] = None,
    ) -> None:
        super().__init__()
        self.entry = entry
        self._title_factory = title_factory
        self._preview_factory = preview_factory
        self._collapsible: Optional[Collapsible] = None
        self._body: Optional[Static] = None
        self._preview: Optional[Static] = None

    def compose(self) -> ComposeResult:
        # Always-visible inline preview (diff/file-head) for edit tools; hidden otherwise.
        self._preview = Static("", markup=False, classes="tool-preview")
        self._preview.display = False
        yield self._preview
        self._body = Static("", markup=False, classes="tool-body")
        self._collapsible = SelectableCollapsible(self._body, title=self._title_factory(self.entry), collapsed=True)
        yield self._collapsible

    def on_mount(self) -> None:
        self.refresh_content()

    def refresh_content(self) -> None:
        if self._collapsible is None or self._body is None:
            return
        self._collapsible.title = self._title_factory(self.entry)
        self._body.update(self.entry.full_content or self.entry.content)
        if self._preview is not None:
            renderable = self._preview_factory(self.entry) if self._preview_factory else None
            if renderable is not None:
                self._preview.update(renderable)
                self._preview.display = True
            else:
                self._preview.update("")
                self._preview.display = False


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
        Binding("ctrl+shift+v", "paste_clipboard_image", "Paste image", key_display="Ctrl+Shift+V", priority=True),
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

    def action_paste_clipboard_image(self) -> None:
        app = self.app
        app.run_worker(app._paste_clipboard_image_worker(), name="paste-image", group="turns")

    async def on_paste(self, event: events.Paste) -> None:
        text = event.text or ""
        import base64 as _b64
        from pathlib import Path as _Path

        from kolega_code.utils.images import (
            encode_image_attachment,
            encode_image_file,
            image_media_type,
        )

        stripped = text.strip()
        if stripped.startswith("data:image/") and ";base64," in stripped:
            header, _, b64data = stripped.partition(";base64,")
            media_type = header[len("data:"):]  # e.g. image/png
            try:
                raw = _b64.b64decode(b64data)
            except Exception:
                # Not a valid data-URI image — fall through to default paste.
                return
            self.app.add_pending_image_attachment(
                encode_image_attachment(raw, media_type, path="pasted-data-uri")
            )
            event.prevent_default()
            event.stop()
            return
        if (
            stripped
            and "\n" not in stripped
            and _Path(stripped).exists()
            and image_media_type(stripped) is not None
        ):
            att = encode_image_file(_Path(stripped))
            if att is not None:
                att["path"] = stripped
                self.app.add_pending_image_attachment(att)
                event.prevent_default()
                event.stop()
                return
        # No image detected — let TextArea's default _on_paste insert the text.
        return

    def on_blur(self, event) -> None:
        dropdown = self.mention_dropdown()
        if dropdown is not None:
            dropdown.close()


class SubAgentEntryWidget(ConversationEntryWidget):
    """A sub-agent summary card that opens the inspector when clicked."""

    @dataclass
    class Pressed(TextualMessage):
        entry: ConversationEntry

    def on_click(self) -> None:
        self.post_message(self.Pressed(self.entry))


class SubAgentRosterRow(Static):
    """One selectable row in the inspector's fleet roster."""

    @dataclass
    class Selected(TextualMessage):
        key: str

    def __init__(self, key: str) -> None:
        super().__init__("", markup=False)
        self.key = key

    def on_click(self) -> None:
        self.post_message(self.Selected(self.key))


class SubAgentInspectorScreen(ModalScreen):
    """Full-screen mission-control view of dispatched sub-agents.

    Left: a roster of every sub-agent in the turn (running + finished). Right: the
    selected agent's full trajectory, rendered with the same entry widgets as the main
    transcript (collapsible tool calls, thinking, responses). Switch agents with the
    arrow keys or by clicking a roster row; the view updates live while agents run.
    """

    BINDINGS = [
        Binding("escape", "close", "Close", show=True, priority=True),
        Binding("q", "close", "Close", show=False, priority=True),
        Binding("down", "next_agent", "Next agent", show=True, priority=True),
        Binding("up", "prev_agent", "Prev agent", show=False, priority=True),
        Binding("o", "toggle_follow", "Follow", show=True, priority=True),
        Binding("y", "copy_trajectory", "Copy", show=True, priority=True),
    ]

    def __init__(self, owner: "KolegaCodeApp", selected_key: str) -> None:
        super().__init__()
        self._owner = owner
        self._selected_key = selected_key
        self._follow = True
        self._rows: dict[str, SubAgentRosterRow] = {}
        self._step_widgets: dict[str, ConversationEntryWidget | ToolEntryWidget] = {}
        self._rendered_key: Optional[str] = None
        self._empty_shown = False
        self._spinner_frame = 0
        self._flush_pending = False

    def compose(self) -> ComposeResult:
        with Horizontal(id="inspector_body"):
            yield VerticalScroll(id="inspector_roster")
            with Vertical(id="inspector_main"):
                yield Static("", id="inspector_header", markup=True)
                yield VerticalScroll(id="inspector_trajectory")
        yield Static("", id="inspector_footer", markup=False)

    def on_mount(self) -> None:
        self.border_title = f"{theme.g(Glyph.SUB_AGENT)} Sub-agents"
        self._sync_roster()
        self._refresh_header()
        self._sync_trajectory()
        self._refresh_footer()
        self.set_interval(theme.SPINNER_INTERVAL, self._on_tick)

    def on_unmount(self) -> None:
        # Defensive: the owner clears this on close, but guarantee no dangling reference.
        if self._owner._sub_agent_inspector is self:
            self._owner._sub_agent_inspector = None

    # ---- live updates ---------------------------------------------------------

    def note_activity_updated(self, activity: SubAgentActivity) -> None:
        """Called by the owner when any sub-agent emits an event (coalesced)."""
        self._schedule_flush()

    def _schedule_flush(self) -> None:
        if self._flush_pending:
            return
        self._flush_pending = True
        try:
            self.set_timer(theme.RENDER_COALESCE_INTERVAL, self._flush)
        except Exception:
            self._flush()

    def _flush(self) -> None:
        self._flush_pending = False
        self._sync_roster()
        self._refresh_header()
        self._sync_trajectory()

    def _on_tick(self) -> None:
        self._spinner_frame += 1
        self._sync_roster()
        self._refresh_header()

    # ---- selection ------------------------------------------------------------

    def _ordered_activities(self) -> list[SubAgentActivity]:
        return sorted(self._owner._sub_agent_activities.values(), key=lambda a: a.index)

    def action_next_agent(self) -> None:
        self._move_selection(1)

    def action_prev_agent(self) -> None:
        self._move_selection(-1)

    def _move_selection(self, delta: int) -> None:
        keys = [a.agent_id for a in self._ordered_activities()]
        if not keys:
            return
        if self._selected_key in keys:
            index = (keys.index(self._selected_key) + delta) % len(keys)
        else:
            index = 0
        self._selected_key = keys[index]
        self._select_changed()

    def on_sub_agent_roster_row_selected(self, message: SubAgentRosterRow.Selected) -> None:
        if message.key != self._selected_key:
            self._selected_key = message.key
            self._select_changed()

    def _select_changed(self) -> None:
        self._sync_roster()
        self._refresh_header()
        self._sync_trajectory()
        row = self._rows.get(self._selected_key)
        if row is not None:
            try:
                row.scroll_visible()
            except Exception:
                pass

    # ---- rendering ------------------------------------------------------------

    def _sync_roster(self) -> None:
        try:
            roster = self.query_one("#inspector_roster", VerticalScroll)
        except Exception:
            return
        if not roster.is_attached:
            return
        activities = self._ordered_activities()
        keys = [a.agent_id for a in activities]
        if keys != list(self._rows):
            roster.remove_children()
            self._rows = {}
            rows = []
            for activity in activities:
                row = SubAgentRosterRow(activity.agent_id)
                self._rows[activity.agent_id] = row
                rows.append(row)
            if rows:
                roster.mount(*rows)
        for activity in activities:
            row = self._rows.get(activity.agent_id)
            if row is not None:
                row.update(self._roster_row(activity, selected=activity.agent_id == self._selected_key))

    def _status_glyph(self, activity: SubAgentActivity) -> tuple[str, str]:
        if activity.status == "running":
            frames = theme.spinner_frames()
            return frames[self._spinner_frame % len(frames)], Color.ACCENT
        if activity.status == "completed":
            return theme.g(Glyph.CHECK), Color.SUCCESS
        if activity.status == "failed":
            return theme.g(Glyph.CROSS), Color.ERROR
        return theme.g(Glyph.BULLET_SEP), Color.WARNING

    def _elapsed(self, activity: SubAgentActivity) -> str:
        if activity.finished_at is not None:
            seconds = max(0.0, activity.finished_at - activity.started_at)
        else:
            seconds = max(0.0, self._owner._now() - activity.started_at)
        return self._owner._format_turn_duration(seconds)

    def _roster_row(self, activity: SubAgentActivity, *, selected: bool) -> Text:
        sep = theme.g(Glyph.BULLET_SEP)
        glyph, color = self._status_glyph(activity)
        indent = "  " * max(0, activity.depth - 1)
        row_style = "bold" if selected else ""
        line = Text()
        line.append("> " if selected else "  ", style=row_style)
        line.append(f"{indent}{glyph} ", style=color)
        parts = [f"#{activity.index}", activity.agent_name, self._elapsed(activity), f"{activity.tool_calls}t"]
        if activity.tokens:
            parts.append(f"{self._owner._format_token_count(activity.tokens)}tok")
        if activity.context_percentage is not None:
            parts.append(f"ctx {activity.context_percentage:.0f}%")
        line.append(f"  {sep}  ".join(parts), style=row_style)
        tail = activity.current_action if activity.status == "running" else activity.last_activity
        if tail:
            line.append(f"\n    {theme.g(Glyph.INSET_ELBOW)} ", style="dim")
            line.append(tail[:48], style="dim")
        return line

    def _refresh_header(self) -> None:
        try:
            header = self.query_one("#inspector_header", Static)
        except Exception:
            return
        activity = self._owner._sub_agent_activities.get(self._selected_key)
        if activity is None:
            header.update(messages.SUB_AGENT_INSPECTOR_NO_SELECTION)
            return
        header.update(Text.from_markup(self._header_markup(activity)))

    def _header_markup(self, activity: SubAgentActivity) -> str:
        sep = theme.g(Glyph.BULLET_SEP)
        owner = self._owner
        base = owner._sub_agent_header(activity)
        extras = [f"{activity.tool_calls} tool{'' if activity.tool_calls == 1 else 's'}"]
        if activity.tokens:
            extras.append(f"{owner._format_token_count(activity.tokens)} tok")
        if activity.context_percentage is not None:
            extras.append(f"ctx {activity.context_percentage:.0f}%")
        line = base + theme.styled(f" {sep} " + f" {sep} ".join(extras), "dim")
        if activity.task:
            line += "\n" + theme.styled(escape(f"Task: {activity.task}"), "dim")
        return line

    def _sync_trajectory(self) -> None:
        try:
            view = self.query_one("#inspector_trajectory", VerticalScroll)
        except Exception:
            return
        if not view.is_attached:
            return
        activity = self._owner._sub_agent_activities.get(self._selected_key)
        if activity is None:
            view.remove_children()
            self._step_widgets = {}
            self._rendered_key = None
            self._empty_shown = False
            return
        if self._rendered_key != self._selected_key:
            view.remove_children()
            self._step_widgets = {}
            self._empty_shown = False
            self._rendered_key = self._selected_key
        if not activity.steps:
            if not self._empty_shown:
                view.remove_children()
                self._step_widgets = {}
                view.mount(Static(messages.SUB_AGENT_INSPECTOR_NO_STEPS, classes="inspector-empty"))
                self._empty_shown = True
            return
        if self._empty_shown:
            view.remove_children()
            self._step_widgets = {}
            self._empty_shown = False
        rendered_ids = list(self._step_widgets)
        current_ids = [step.entry_id for step in activity.steps]
        if current_ids[: len(rendered_ids)] != rendered_ids:
            view.remove_children()
            self._step_widgets = {}
            rendered_ids = []
        for widget in self._step_widgets.values():
            widget.refresh_content()
        new_steps = activity.steps[len(rendered_ids) :]
        if new_steps:
            widgets = []
            for step in new_steps:
                widget = self._owner._make_entry_widget(step)
                self._step_widgets[step.entry_id] = widget
                widgets.append(widget)
            view.mount(*widgets)
        if self._follow:
            self._scroll_trajectory_end()

    def _scroll_trajectory_end(self) -> None:
        """Scroll to the newest step after layout settles (heights are auto)."""

        def _do() -> None:
            try:
                self.query_one("#inspector_trajectory", VerticalScroll).scroll_end(animate=False)
            except Exception:
                pass

        try:
            self.call_after_refresh(_do)
        except Exception:
            _do()

    def _refresh_footer(self) -> None:
        try:
            footer = self.query_one("#inspector_footer", Static)
        except Exception:
            return
        sep = theme.g(Glyph.BULLET_SEP)
        follow = "on" if self._follow else "off"
        footer.update(
            f"Esc close  {sep}  Up/Down switch agent  {sep}  Tab+Enter expand tool"
            f"  {sep}  o follow:{follow}  {sep}  y copy"
        )

    # ---- actions --------------------------------------------------------------

    def action_close(self) -> None:
        self._owner._sub_agent_inspector = None
        self.dismiss()

    def action_toggle_follow(self) -> None:
        self._follow = not self._follow
        self._refresh_footer()
        if self._follow:
            self._sync_trajectory()

    def action_copy_trajectory(self) -> None:
        activity = self._owner._sub_agent_activities.get(self._selected_key)
        if activity is None:
            return
        self._owner.copy_to_clipboard(self._trajectory_text(activity))
        try:
            self._owner._notify_user(messages.SUB_AGENT_TRAJECTORY_COPIED, severity="information")
        except Exception:
            pass

    def _trajectory_text(self, activity: SubAgentActivity) -> str:
        lines = [f"{activity.agent_name} #{activity.index} ({activity.status})"]
        full_task = activity.task_full or activity.task
        if full_task:
            lines.append(f"Task: {full_task}")
        lines.append("")
        for step in activity.steps:
            # The full task is already printed above as the header Task line.
            if step.kind == "sub_agent_task":
                continue
            label = step.tool_name or step.kind
            lines.append(f"[{step.kind}] {label}")
            body = step.full_content or step.content
            if body:
                lines.append(body)
            lines.append("")
        return "\n".join(lines).rstrip() + "\n"


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
        padding-bottom: 1;
    }

    ToolEntryWidget {
        height: auto;
        padding-bottom: 1;
    }

    ToolEntryWidget Collapsible {
        background: transparent;
        border-top: none;
        padding-bottom: 0;
        padding-left: 0;
    }

    ToolEntryWidget Collapsible Contents {
        padding-left: 3;
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

    .settings-hint {
        color: $text-muted;
        margin-top: 0;
    }

    .agent-model-group {
        height: auto;
        margin-top: 1;
    }

    .agent-model-role {
        text-style: bold;
        color: $text;
    }

    .agent-model-field {
        height: auto;
        margin-top: 1;
    }

    .agent-model-field-label {
        width: 10;
        margin-top: 1;
        color: $text-muted;
    }

    .agent-model-field Select {
        width: 1fr;
    }

    #settings_actions {
        height: auto;
        padding: 0 1;
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

    SubAgentInspectorScreen {
        align: center middle;
    }

    SubAgentInspectorScreen #inspector_body {
        width: 100%;
        height: 1fr;
    }

    SubAgentInspectorScreen #inspector_roster {
        width: 40;
        height: 100%;
        border: round $surface;
        padding: 0 1;
    }

    SubAgentInspectorScreen #inspector_main {
        width: 1fr;
        height: 100%;
    }

    SubAgentInspectorScreen #inspector_header {
        height: auto;
        border: round $surface;
        padding: 0 1;
    }

    SubAgentInspectorScreen #inspector_trajectory {
        height: 1fr;
        border: round $surface;
        padding: 0 1;
    }

    SubAgentRosterRow {
        height: auto;
        padding: 0 0 1 0;
    }

    SubAgentInspectorScreen .inspector-empty {
        color: $text-muted;
        padding: 1;
    }

    SubAgentInspectorScreen #inspector_footer {
        height: 1;
        padding: 0 1;
        background: $surface;
        color: $text-muted;
    }
    """

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
        self.conversation_entries: list[ConversationEntry] = []
        self._stream_entries: dict[str, ConversationEntry] = {}
        self._tool_entries: dict[str, ConversationEntry] = {}
        self._tool_stream_buffers: dict[str, str] = {}
        self._sub_agent_activities: dict[str, SubAgentActivity] = {}
        self._sub_agent_by_tool_call: dict[str, str] = {}
        self._sub_agent_seq = 0
        self._workflow_activities: dict[str, WorkflowActivity] = {}
        self._render_pending = False
        self._entry_widgets: dict[str, ConversationEntryWidget | ToolEntryWidget] = {}
        self._dirty_entry_ids: set[str] = set()
        self._active_progress_entry: Optional[ConversationEntry] = None
        self._turn_active = False
        self._latest_plan: Optional[str] = self.session.latest_plan_markdown or None
        self._plan_pending: bool = bool(self.session.plan_pending)
        self._plan_decision_active = False
        self._gigacode_enabled = False
        self._pending_question: Optional[PendingQuestion] = None
        self._pending_approval: Optional[PendingApproval] = None
        self._pending_image_attachments: list[dict] = []
        self._permission_lock = asyncio.Lock()
        self._pending_model_selection: Optional[PendingModelSelection] = None
        self._pending_effort_selection: Optional[PendingEffortSelection] = None
        self._pending_theme_selection: Optional[PendingThemeSelection] = None
        # Saved per-agent model/effort awaiting the provider->model cascade that
        # restores them (keyed by the row's model/effort select id). See
        # _populate_agent_model_rows for why the cascade, not direct assignment, applies them.
        self._pending_agent_models: dict[str, str] = {}
        self._pending_agent_efforts: dict[str, str] = {}
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
        self._sub_agent_inspector: Optional[SubAgentInspectorScreen] = None
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
        self.session.plan_pending = self._plan_pending

    def _save_session(self) -> None:
        self._sync_planning_state_to_session()
        self.store.save(self.session)

    def _restore_plan_action_visibility(self) -> None:
        self._set_plan_actions_visible(
            self.interaction_mode == PLAN_INTERACTION_MODE and self._plan_pending,
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
        if self._pending_image_attachments:
            attachments = (attachments or []) + self._pending_image_attachments
            self._pending_image_attachments.clear()
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
            self._finalize_workflow_activities()
            self._save_session_history()
            self._finish_turn_progress(messages.FINISHED, TurnState.IDLE)
            self._capture_completed_plan()
            self._log_status(messages.FINISHED, "ok")
        except asyncio.CancelledError:
            self._cancel_pending_question()
            self._cancel_pending_approval()
            await self._drain_pending_events()
            self._finalize_sub_agent_activities()
            self._finalize_workflow_activities()
            self._save_session_history()
            self._finish_turn_progress(messages.STOPPED_BY_USER, TurnState.STOPPED)
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
            self._finish_turn_progress(message_text, TurnState.ERROR)
            self._log_status(message_text, "error")
        except Exception as exc:
            self._cancel_pending_question()
            self._cancel_pending_approval()
            await self._drain_pending_events()
            self._finalize_sub_agent_activities()
            self._finalize_workflow_activities()
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
                self._update_activity_progress(messages.RUNNING_TERMINAL_COMMAND, state=TurnState.RUNNING_TOOL)
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
                self._add_conversation_entry(ConversationEntry(kind="message", content=message_text))
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
        screen = SubAgentInspectorScreen(self, key)
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

    def on_sub_agent_entry_widget_pressed(self, message: SubAgentEntryWidget.Pressed) -> None:
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
        self.query_one("#composer", ChatComposer).focus()

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
        agent_class = PlanningAgent if self.interaction_mode == PLAN_INTERACTION_MODE else CoderAgent
        self.skill_catalog = discover_skills(self.project_path)
        prompt_extensions: list[PromptExtension] = []
        tool_extensions: list[ToolExtension] = []
        # The shared task list is build-mode execution tracking; plan mode produces
        # a plan via write_plan and does not get the task-list tools.
        if self.interaction_mode == BUILD_INTERACTION_MODE:
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
        if self.interaction_mode == PLAN_INTERACTION_MODE:
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
        self._plan_pending = True
        self._plan_decision_active = True
        self._save_session()
        self._refresh_planning_sidebar()
        self._add_conversation_entry(ConversationEntry(kind="plan", content=plan, complete=True))
        self._set_plan_actions_visible(True, allow_discuss=True)
        self._set_composer_status(PLAN_READY_PLACEHOLDER)
        self._set_chat_enabled(False)
        self._notify_user(messages.PLAN_CAPTURED)

    async def _implement_pending_plan(self, *, clear_context: bool = False) -> None:
        plan = self._latest_plan
        if not plan or self._turn_active or self.agent_worker is not None:
            return

        # Leave self._latest_plan set so the planning sidebar keeps showing the
        # plan as a read-only reference while it is being built; clearing
        # _plan_pending is what hides the "Implement plan" action so it does not
        # reappear when the user re-enters plan mode.
        self._plan_pending = False
        self._plan_decision_active = False
        if clear_context:
            self._clear_agent_context()
        self._save_session()
        await self._set_interaction_mode(BUILD_INTERACTION_MODE)
        self._refresh_planning_sidebar()
        self._set_plan_actions_visible(False)

        prompt = build_implement_plan_prompt(plan, gigacode_enabled=self._gigacode_enabled)
        self._add_conversation_entry(ConversationEntry(kind="user", content="Implement the approved plan."))
        self.agent_worker = self.run_worker(
            self._process_message(prompt), name="kolega-turn", group="turns", exclusive=True
        )

    def _discuss_pending_plan(self) -> None:
        if not self._latest_plan:
            return

        self._latest_plan = None
        self._plan_pending = False
        self._plan_decision_active = False
        self._save_session()
        self._refresh_planning_sidebar()
        self._set_plan_actions_visible(False)
        self._restore_composer_placeholder()
        self._set_chat_enabled(self.agent is not None)
        self.query_one("#composer", ChatComposer).focus()
        self._notify_user(messages.PLAN_DISCUSSION_RESUMED)

    def _active_prompt_actions(self) -> Optional[ActionList]:
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
                self.interaction_mode == PLAN_INTERACTION_MODE and self._plan_pending,
                "#plan_actions",
            ),
        ]
        for active, selector in candidates:
            if not active:
                continue
            try:
                actions = self.query_one(selector, ActionList)
            except Exception:
                return None
            return actions if actions.display else None
        return None

    def _focus_active_prompt(self) -> None:
        """Focus the active prompt list now and re-assert after the refresh settles.

        The synchronous set_focus handles the common fast path; the deferred
        re-assert defeats the documented race where compose/resume/disable churn
        resets focus right after we set it (see PromptPanel.prompt)."""
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
            and isinstance(focused, ChatComposer)
            and not focused.disabled
        ):
            return
        self.screen.set_focus(actions)

    def _set_plan_actions_visible(self, visible: bool, *, allow_discuss: bool = False) -> None:
        try:
            plan_actions = self.query_one("#plan_actions", ActionList)
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
            "/attach": self._command_attach,
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

    def add_pending_image_attachment(self, attachment: dict) -> None:
        """Stash a pending image attachment for the next submitted message."""
        self._pending_image_attachments.append(attachment)
        names = ", ".join(a.get("path", "image") for a in self._pending_image_attachments)
        self._show_composer_hint(f"Attached images: {names} (press Enter to send)", tone="info")

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
        if self.agent is not None and not getattr(self.agent, "supports_vision", False):
            self._show_composer_hint(
                "Pasted an image, but the current model does not support vision. "
                "Switch with /model or attach anyway.",
                tone="warning",
            )
        attachment = encode_image_attachment(data, media_type, path="clipboard")
        self.add_pending_image_attachment(attachment)

    async def _command_plan(self, args: str) -> None:
        if self._mode_switch_blocked():
            return
        await self._set_interaction_mode(PLAN_INTERACTION_MODE)

    async def _command_build(self, args: str) -> None:
        if self._mode_switch_blocked():
            return
        await self._set_interaction_mode(BUILD_INTERACTION_MODE)

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
            if self.interaction_mode == PLAN_INTERACTION_MODE:
                note += " In plan mode, workflow sub-agents are read-only (parallel research only)."
        else:
            note = "gigacode workflow orchestration disabled."
        self._add_conversation_entry(ConversationEntry(kind="system", content=note))
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
                ConversationEntry(kind="system", content=messages.LOGIN_USAGE.format(targets=targets))
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
        self._add_conversation_entry(ConversationEntry(kind="system", content=messages.CHATGPT_LOGIN_STARTING))
        self.run_worker(self._do_chatgpt_login(), name="chatgpt-login", group="auth", exclusive=True)

    def _on_login_url(self, url: str) -> None:
        self._add_conversation_entry(
            ConversationEntry(kind="system", content=messages.CHATGPT_LOGIN_URL.format(url=url))
        )

    async def _do_chatgpt_login(self) -> None:
        try:
            tokens = await run_login_flow(on_url=self._on_login_url)
        except Exception as exc:  # LoginError / TokenRefreshError / unexpected
            text = messages.CHATGPT_LOGIN_FAILED.format(error=exc)
            self._notify_user(text, severity="error")
            self._add_conversation_entry(ConversationEntry(kind="system", content=text, tone="error"))
            return

        self.settings.set_oauth_token(chatgpt_constants.PROVIDER_KEY, tokens.model_dump(mode="json"))
        self.settings_store.save(self.settings)
        self._add_conversation_entry(
            ConversationEntry(
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
                ConversationEntry(kind="system", content=messages.LOGOUT_USAGE.format(targets=targets))
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
            # The task list is owned by the single top-level build agent; sub-agents
            # must not get it or they would race on the shared list.
            propagate_to_sub_agents=False,
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
            # Single-owner, shared mutable state: never hand it to sub-agents.
            propagate_to_sub_agents=False,
        )

    def _planning_question_prompt_extension(self) -> PromptExtension:
        return PromptExtension(
            id="cli-planning-questions",
            title="Planning Questions",
            markdown=PLANNING_QUESTION_PROMPT,
            modes=[AgentMode.CLI],
            # ask_user_choice is a top-level, interactive planning tool; sub-agents
            # should not prompt the user.
            propagate_to_sub_agents=False,
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
            propagate_to_sub_agents=False,
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
        self._focus_active_prompt()

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
        self._focus_active_prompt()

    def _show_model_options(self) -> None:
        self._set_model_actions_visible(True)
        self._focus_active_prompt()

    def _show_theme_options(self) -> None:
        self._set_theme_actions_visible(True)
        self._focus_active_prompt()

    def _set_question_actions_visible(self, visible: bool) -> None:
        try:
            panel = self.query_one("#question_prompt", PromptPanel)
        except Exception:
            return
        if visible:
            panel.display = True
            panel.actions.display = True
            self._focus_active_prompt()
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
            self._focus_active_prompt()
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
        if provider == chatgpt_constants.PROVIDER_KEY:
            # OAuth provider: no API key — the field is informational only.
            if self.settings.has_oauth_token(provider):
                return "Signed in with ChatGPT — run /login chatgpt to switch accounts"
            return "Run /login chatgpt to sign in with your ChatGPT subscription"
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

    def _set_status_activity(self, content: str, *, turn_state: Optional[TurnState] = None) -> None:
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
                        ConversationEntry(kind="compaction_summary", content=summary.strip())
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
        self._workflow_activities = {}
        self._active_progress_entry = None
        self._plan_decision_active = False
        self._restore_plan_action_visibility()
        self._cancel_pending_question()
        self._cancel_pending_model_selection()
        self._cancel_pending_effort_selection()
        self._cancel_pending_theme_selection()
        self._refresh_planning_sidebar()
        self._ensure_startup_entry(render=False)
        # If the restored agent is in a compacted state, mark the boundary with a
        # collapsible summary (between the folded prefix and the verbatim tail).
        summary_entry = self._resume_compaction_entry()
        boundary = None
        if summary_entry is not None:
            through = int((self.session.compaction or {}).get("compacted_through") or 0)
            boundary = min(through, len(history))
        for index, item in enumerate(history):
            if summary_entry is not None and index == boundary:
                self.conversation_entries.append(summary_entry)
            try:
                message = Message.from_dict(item)
            except Exception:
                continue
            self.conversation_entries.extend(self._conversation_entries_from_message(message))
        if summary_entry is not None and boundary is not None and boundary >= len(history):
            self.conversation_entries.append(summary_entry)
        self._render_conversation()

    def _resume_compaction_entry(self) -> Optional[ConversationEntry]:
        """A collapsible summary entry for the restored compaction boundary, or None.

        Built from the session's persisted compaction metadata (the same data the
        agent was restored from), so it does not depend on agent internals.
        """
        data = self.session.compaction or {}
        summary_text = (data.get("summary") or "").strip()
        if not summary_text or int(data.get("compacted_through") or 0) <= 0:
            return None
        return ConversationEntry(kind="compaction_summary", content=summary_text)

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
        self._close_sub_agent_inspector()
        self._tool_entries = {}
        self._tool_stream_buffers = {}
        self._sub_agent_activities = {}
        self._sub_agent_by_tool_call = {}
        self._sub_agent_seq = 0
        self._workflow_activities = {}
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

    def _persist_agent_into_session(self) -> None:
        """Capture the agent's message history and compaction boundary into the session."""
        if self.agent is None:
            return
        self.session.history = self.agent.dump_message_history()
        self.session.compaction = self.agent.dump_compaction_state()

    def _save_session_history(self) -> None:
        if self.agent is None:
            return
        self._persist_agent_into_session()
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
            new_entry = ConversationEntry(
                kind=message_type,
                content=entry_content,
                complete=complete,
                tool_name=tool_name,
                tool_call_id=tool_call_id or None,
                full_content=full_content,
            )
            # A preview event can land before this entry exists; apply any stash now.
            pending = getattr(self, "_pending_edit_previews", None)
            if pending and tool_call_id and tool_call_id in pending:
                new_entry.edit_preview = pending.pop(tool_call_id)
            self._add_conversation_entry(new_entry)
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

    def _apply_edit_preview(self, content: dict) -> None:
        """Attach a UI-only diff/head preview (from a file_edit_preview event) to its tool entry."""
        tool_call_id = str(content.get("tool_call_id") or "")
        tool_name = str(content.get("tool_name") or "")
        if not tool_call_id:
            return
        entry = self._find_tool_entry(tool_call_id, tool_name)
        if entry is None:
            # Preview can arrive before the tool entry exists; stash and apply on creation.
            self.__dict__.setdefault("_pending_edit_previews", {})[tool_call_id] = content
            return
        entry.edit_preview = content
        self._invalidate_conversation(entry)

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
            task_full = str(info.get("task_full") or info.get("task") or "")
            activity = SubAgentActivity(
                agent_id=key,
                agent_name=str(info.get("agent_name") or event.sender or "sub-agent"),
                task=str(info.get("task") or ""),
                index=self._sub_agent_seq,
                entry=entry,
                task_full=task_full,
                workflow_run_id=str(info.get("workflow_run_id") or ""),
                workflow_phase=str(info.get("phase") or ""),
                started_at=self._now(),
            )
            self._sub_agent_activities[key] = activity
            parent_id = info.get("parent_tool_call_id")
            if parent_id:
                self._sub_agent_by_tool_call[str(parent_id)] = key
            if task_full:
                # The full task becomes the first entry in the inspector trajectory; the
                # inline summary keeps showing only the truncated preview.
                activity.steps.append(
                    ConversationEntry(kind="sub_agent_task", content=task_full, full_content=task_full)
                )
            entry.content = self._format_sub_agent_content(activity)
            self._add_conversation_entry(entry)
            self._refresh_sub_agent_activity_status()
            self._note_workflow_sub_agent(activity)
        return activity

    def _render_sub_agent_event(self, event: AgentEvent) -> None:
        activity = self._ensure_sub_agent_activity(event)
        content = event.content
        info = event.sub_agent_info or {}
        depth = info.get("depth")
        if isinstance(depth, int):
            activity.depth = depth
        status = content.get("status")
        if status:  # lifecycle event from AgentTool
            tokens = content.get("total_tokens", content.get("tokens"))
            if isinstance(tokens, int):
                activity.tokens = tokens
            if status != "GENERATING":
                message = str(content.get("message") or "")
                failed = status == "ERROR" or message.startswith("Error")
                activity.status = "failed" if failed else "completed"
                activity.finished_at = self._now()
                activity.entry.complete = True
                activity.current_action = ""
                activity.last_activity = message if failed else ""
                self._refresh_sub_agent_activity_status()
            self._refresh_sub_agent_entry(activity, force=True)
            self._invalidate_sub_agent_detail(activity)
            self._note_workflow_sub_agent(activity)
            return

        message_type = content.get("message_type", "response")
        text = str(content.get("text") or "")
        if message_type == "tool_call":
            activity.tool_calls += 1
            tool = str(content.get("tool_description") or content.get("tool_name") or "tool")
            activity.last_activity = tool
            activity.current_action = tool
            self._record_sub_agent_tool_step(activity, "tool_call", content)
        elif message_type in {"tool_result", "tool_error"}:
            suffix = "failed" if message_type == "tool_error" else "done"
            tool = str(content.get("tool_description") or content.get("tool_name") or "tool")
            activity.last_activity = f"{tool} {suffix}"
            activity.current_action = f"{tool} {suffix}"
            self._record_sub_agent_tool_step(activity, message_type, content)
        elif message_type == "thinking":
            activity.last_activity = "thinking"
            activity.current_action = "thinking"
            self._accumulate_sub_agent_stream(activity, "thinking", event, text)
        else:  # streamed response text - accumulate by chunk uuid
            activity.current_action = "responding"
            if event.uuid and text:
                buffer = activity.stream_buffers.get(event.uuid, "") + text
                activity.stream_buffers[event.uuid] = buffer
                activity.active_stream_uuid = event.uuid
            self._accumulate_sub_agent_stream(activity, "assistant", event, text)
        self._refresh_sub_agent_entry(activity)
        self._invalidate_sub_agent_detail(activity)

    def _record_sub_agent_tool_step(self, activity: SubAgentActivity, message_type: str, content: dict) -> None:
        """Capture a sub-agent tool_call/result/error as a ConversationEntry step,
        pairing call->result by tool_call_id exactly like _add_tool_message.

        Sub-agent tool events always carry a stable tool_call_id (the emitter sets it from
        tool_execution_id). The no-id branch is a defensive fallback: a call always gets its
        own step, and a result/error attaches to the most recent unpaired same-name call so
        distinct executions of the same tool can never collide onto one shared step.
        """
        tool_name = str(content.get("tool_description") or content.get("tool_name") or "tool")
        tool_call_id = str(content.get("tool_call_id") or "").strip()
        text = str(content.get("text") or "")
        if message_type == "tool_call":
            entry_content = text or f"Calling {tool_name}"
            full_content = ""
            complete = False
        elif message_type == "tool_error":
            entry_content = self._truncate_tool_text(text)
            full_content = self._capped_tool_text(text)
            complete = True
        else:  # tool_result
            entry_content = self._tool_result_preview(text)
            full_content = self._capped_tool_text(text)
            complete = True

        step = None
        if tool_call_id:
            step = activity.tool_steps.get(tool_call_id)
        elif message_type != "tool_call":
            step = self._last_unpaired_sub_agent_tool_step(activity, tool_name)
        if step is None:
            step = ConversationEntry(
                kind=message_type,
                content=entry_content,
                complete=complete,
                tool_name=tool_name,
                tool_call_id=tool_call_id or None,
                full_content=full_content,
            )
            activity.steps.append(step)
            if tool_call_id:
                activity.tool_steps[tool_call_id] = step
            return
        step.kind = message_type
        step.content = entry_content
        step.complete = complete
        step.tool_name = tool_name
        step.full_content = full_content or step.full_content

    def _last_unpaired_sub_agent_tool_step(
        self, activity: SubAgentActivity, tool_name: str
    ) -> Optional[ConversationEntry]:
        """Most recent still-running tool step for a tool name (name-based result pairing)."""
        for step in reversed(activity.steps):
            if step.kind == "tool_call" and not step.complete and step.tool_name == tool_name:
                return step
        return None

    def _accumulate_sub_agent_stream(
        self, activity: SubAgentActivity, kind: str, event: AgentEvent, text: str
    ) -> None:
        """Accumulate streamed thinking/response chunks into one step per chunk uuid,
        mirroring the main transcript's _apply_stream_chunk.

        Events normally carry a uuid; the kind-qualified sentinel for the no-uuid case keeps
        consecutive uuid-less chunks of the same kind merged into one step (rather than
        fragmenting) while never merging thinking into response.
        """
        complete = not event.is_streaming
        chunk_uuid = str(event.uuid or "")
        cache_key = chunk_uuid or f"__nouuid__:{kind}"
        step = activity.stream_steps.get(cache_key)
        if step is None:
            if not text and not complete:
                return
            step = ConversationEntry(kind=kind, content="", complete=complete, uuid=chunk_uuid or None)
            activity.steps.append(step)
            activity.stream_steps[cache_key] = step
        step.content += text
        step.complete = complete

    def _note_sub_agent_tool_stream(self, event: AgentEvent) -> None:
        activity = self._ensure_sub_agent_activity(event)
        tool_name = str(event.content.get("tool_name") or event.content.get("tool_description") or "tool")
        is_complete = bool(event.content.get("is_complete"))
        activity.last_activity = f"{tool_name} done" if is_complete else f"{tool_name} streaming"
        self._refresh_sub_agent_entry(activity)

    def _note_sub_agent_context(self, event: AgentEvent) -> None:
        """Record a sub-agent's context-window usage on its own card, so it never
        overwrites the main agent's context indicator on the status dashboard."""
        activity = self._ensure_sub_agent_activity(event)
        content = event.content
        activity.context_percentage = self._as_optional_float(content.get("usage_percentage"))
        activity.context_input_tokens = self._as_optional_int(content.get("input_tokens"))
        activity.context_max_tokens = self._as_optional_int(content.get("max_tokens"))
        self._refresh_sub_agent_entry(activity)
        self._invalidate_sub_agent_detail(activity)

    def _note_sub_agent_status(self, event: AgentEvent) -> None:
        """Surface a sub-agent's status notice (e.g. provider overload) on its card,
        keeping it off the main activity line."""
        activity = self._ensure_sub_agent_activity(event)
        message = str(event.content.get("message") or "").strip()
        if message:
            activity.last_activity = message
            if activity.status == "running":
                activity.current_action = message
        self._refresh_sub_agent_entry(activity)
        self._invalidate_sub_agent_detail(activity)

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
        sep = theme.g(Glyph.BULLET_SEP)
        body_lines: list[str] = []
        if activity.task:
            task = activity.task
            if len(task) > SUB_AGENT_TASK_PREVIEW_CHARS:
                task = f"{task[:SUB_AGENT_TASK_PREVIEW_CHARS]}{theme.g(Glyph.ELLIPSIS)}"
            body_lines.append(f"Task: {task}")
        tools_line = f"{activity.tool_calls} tool{'' if activity.tool_calls == 1 else 's'}"
        if activity.tokens:
            tools_line += f" {sep} {self._format_token_count(activity.tokens)} tok"
        if activity.context_percentage is not None:
            tools_line += f" {sep} ctx {activity.context_percentage:.0f}%"
        if activity.last_activity:
            tools_line += f" {sep} last: {activity.last_activity}"
        body_lines.append(tools_line)
        if activity.status == "running" and activity.current_action:
            body_lines.append(f"{theme.g(Glyph.TOOL)} now: {activity.current_action}")
        if activity.status == "running" and activity.active_stream_uuid:
            tail = activity.stream_buffers.get(activity.active_stream_uuid, "")
            tail = " ".join(tail.split())
            if tail:
                if len(tail) > SUB_AGENT_TAIL_CHARS:
                    tail = f"{theme.g(Glyph.ELLIPSIS)}{tail[-SUB_AGENT_TAIL_CHARS:]}"
                body_lines.append(tail)
        if any(step.kind != "sub_agent_task" for step in activity.steps):
            body_lines.append(messages.SUB_AGENT_INSPECT_HINT)

        return body_lines

    def _format_token_count(self, tokens: int) -> str:
        """Compact token count for cards/roster: 980, 3.1k, 1.2M."""
        if tokens < 1000:
            return str(tokens)
        if tokens < 1_000_000:
            return f"{tokens / 1000:.1f}k".replace(".0k", "k")
        return f"{tokens / 1_000_000:.1f}M".replace(".0M", "M")

    def _invalidate_sub_agent_detail(self, activity: SubAgentActivity) -> None:
        """Refresh the open inspector if it is showing this agent (no-op when closed)."""
        screen = self._sub_agent_inspector
        if screen is not None:
            screen.note_activity_updated(activity)

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

    # ---- workflow cards ("gigacode") ------------------------------------------

    def _handle_workflow_start(self, content: dict) -> None:
        run_id = str(content.get("workflow_run_id") or "")
        if not run_id or run_id in self._workflow_activities:
            return
        phases: list[PhaseState] = []
        for raw in content.get("phases") or []:
            # meta.phases entries may be dicts ({title, detail}) or bare strings;
            # extract_meta only guarantees name+description, so guard the shape.
            if isinstance(raw, dict):
                title = str(raw.get("title") or "").strip()
                detail = str(raw.get("detail") or "").strip()
            else:
                title, detail = str(raw).strip(), ""
            if title:
                phases.append(PhaseState(title=title, detail=detail))
        entry = ConversationEntry(kind="workflow", content="", complete=False)
        activity = WorkflowActivity(
            run_id=run_id,
            name=str(content.get("name") or "workflow"),
            description=str(content.get("description") or ""),
            entry=entry,
            phases=phases,
            started_at=self._now(),
        )
        self._workflow_activities[run_id] = activity
        entry.content = self._format_workflow_content(activity)
        self._add_conversation_entry(entry)

    def _handle_workflow_phase(self, content: dict) -> None:
        title = str(content.get("text") or "").strip()
        activity = self._workflow_for_run(content)
        if title:
            self._update_activity_progress(f"workflow: {title}", state=TurnState.RUNNING_SUB_AGENTS)
        if activity is None or not title:
            return
        # phase() calls are sequential, so a new explicit phase retires the prior one.
        if activity.current_phase and activity.current_phase != title:
            prev = activity.phase_by_title(activity.current_phase)
            if prev is not None and prev.state == "active":
                prev.state = "done"
        phase = activity.phase_by_title(title)
        if phase is None:
            phase = PhaseState(title=title)
            activity.phases.append(phase)
        if phase.state == "pending":
            phase.state = "active"
        activity.current_phase = title
        self._refresh_workflow_entry(activity)

    def _handle_workflow_log(self, content: dict) -> None:
        message = str(content.get("text") or "").strip()
        activity = self._workflow_for_run(content)
        if activity is None or not message:
            return
        activity.latest_log = message
        self._refresh_workflow_entry(activity)

    def _handle_workflow_end(self, content: dict) -> None:
        activity = self._workflow_for_run(content)
        if activity is None:
            return
        status = str(content.get("status") or "completed")
        activity.status = status
        activity.finished_at = self._now()
        activity.current_phase = ""
        for phase in activity.phases:
            if status == "failed":
                # Only an in-flight phase failed; phases never reached stay pending.
                if phase.state == "active":
                    phase.state = "failed"
            elif phase.state in {"active", "pending"}:
                phase.state = "done"
        activity.entry.complete = True
        self._refresh_workflow_entry(activity, force=True)

    def _workflow_for_run(self, content: dict) -> Optional[WorkflowActivity]:
        run_id = str(content.get("workflow_run_id") or "")
        return self._workflow_activities.get(run_id) if run_id else None

    def _note_workflow_sub_agent(self, activity: SubAgentActivity) -> None:
        """Roll a workflow sub-agent's phase/tokens into its workflow card.

        sub_agent_info carries workflow_run_id + phase for every workflow-dispatched
        agent; consuming it here drives per-phase agent counts and marks a phase active
        even when the script used the agent(phase=...) kwarg (which emits no phase event).
        """
        card = self._workflow_activities.get(activity.workflow_run_id) if activity.workflow_run_id else None
        if card is None:
            return
        self._recompute_workflow_rollup(card)
        self._refresh_workflow_entry(card)

    def _recompute_workflow_rollup(self, card: WorkflowActivity) -> None:
        """Idempotently derive agent counts + tokens for a card from its sub-agents."""
        members = [a for a in self._sub_agent_activities.values() if a.workflow_run_id == card.run_id]
        card.agent_count = len(members)
        card.tokens = sum(a.tokens for a in members if isinstance(a.tokens, int))
        by_phase: dict[str, list[SubAgentActivity]] = {}
        for member in members:
            by_phase.setdefault(member.workflow_phase, []).append(member)
        for phase in card.phases:
            members_for_phase = by_phase.get(phase.title, [])
            phase.agents_total = len(members_for_phase)
            phase.agents_done = sum(1 for a in members_for_phase if a.status != "running")
            if members_for_phase and phase.state == "pending":
                phase.state = "active"
        # Phases that exist only via the agent(phase=...) kwarg, never declared in meta.
        for title, members_for_phase in by_phase.items():
            if title and card.phase_by_title(title) is None:
                card.phases.append(
                    PhaseState(
                        title=title,
                        state="active",
                        agents_total=len(members_for_phase),
                        agents_done=sum(1 for a in members_for_phase if a.status != "running"),
                    )
                )

    def _refresh_workflow_entry(self, activity: WorkflowActivity, *, force: bool = False) -> None:
        activity.entry.content = self._format_workflow_content(activity)
        self._invalidate_conversation(activity.entry)
        if force:
            self._flush_conversation_render()

    def _workflow_activity_for_entry(self, entry: ConversationEntry) -> Optional[WorkflowActivity]:
        for activity in self._workflow_activities.values():
            if activity.entry is entry:
                return activity
        return None

    def _tick_running_workflows(self) -> None:
        for activity in self._workflow_activities.values():
            if activity.status == "running":
                activity.entry.content = self._format_workflow_content(activity)
                self._invalidate_conversation(activity.entry)

    def _finalize_workflow_activities(self, status: str = "stopped") -> None:
        """Mark still-running workflow cards as finished (workflow_end never arrives on cancel)."""
        changed = False
        for activity in self._workflow_activities.values():
            if activity.status == "running":
                activity.status = status
                activity.finished_at = self._now()
                activity.current_phase = ""
                if status == "completed":
                    for phase in activity.phases:
                        if phase.state in {"active", "pending"}:
                            phase.state = "done"
                # On a stop/cancel, leave phase glyphs as they were — an interrupted
                # phase stays "active" rather than misreporting as done or pending.
                activity.entry.complete = True
                activity.entry.content = self._format_workflow_content(activity)
                self._invalidate_conversation(activity.entry)
                changed = True
        if changed:
            self._flush_conversation_render()

    def _workflow_phase_glyph(self, phase: PhaseState) -> tuple[str, str]:
        if phase.state == "done":
            return Glyph.CHECK, Color.SUCCESS
        if phase.state == "failed":
            return Glyph.CROSS, Color.ERROR
        if phase.state == "active":
            return Glyph.RUNNING, Color.ACCENT
        return Glyph.PENDING, Color.MUTED

    def _workflow_header(self, activity: WorkflowActivity) -> str:
        if activity.finished_at is not None:
            elapsed = max(0.0, activity.finished_at - activity.started_at)
        else:
            elapsed = max(0.0, self._now() - activity.started_at)
        duration = self._format_turn_duration(elapsed)
        sep = theme.g(Glyph.BULLET_SEP)
        if activity.status == "running":
            color, state = Color.ACCENT, f"running {sep} {duration}"
        elif activity.status == "completed":
            color, state = Color.SUCCESS, f"completed in {duration}"
        elif activity.status == "failed":
            color, state = Color.ERROR, f"failed after {duration}"
        else:
            color, state = Color.WARNING, f"stopped after {duration}"
        return theme.role_header(
            Glyph.PLAN,
            escape(activity.name or "workflow"),
            color,
            state=f"workflow {sep} {state}",
        )

    def _workflow_footer_line(self, activity: WorkflowActivity) -> str:
        sep = theme.g(Glyph.BULLET_SEP)
        bits: list[str] = []
        if activity.status == "running" and activity.current_phase:
            bits.append(f"now: {activity.current_phase}")
        if activity.agent_count:
            bits.append(f"{activity.agent_count} agent{'' if activity.agent_count == 1 else 's'}")
        if activity.tokens:
            bits.append(f"{self._format_token_count(activity.tokens)} tok")
        if activity.latest_log:
            log = activity.latest_log
            if len(log) > SUB_AGENT_TASK_PREVIEW_CHARS:
                log = f"{log[:SUB_AGENT_TASK_PREVIEW_CHARS]}{theme.g(Glyph.ELLIPSIS)}"
            bits.append(log)
        return f" {sep} ".join(bits)

    def _workflow_phase_rows(self, activity: WorkflowActivity) -> list[Text]:
        sep = theme.g(Glyph.BULLET_SEP)
        bar = f"  {theme.g(Glyph.INSET_BAR)}"
        rows: list[Text] = []
        for phase in activity.phases:
            glyph, color = self._workflow_phase_glyph(phase)
            line = Text()
            line.append(bar, style="dim")
            line.append(" ")
            line.append(f"{theme.g(glyph)} ", style=color)
            title_style = "bold" if phase.state == "active" else ("dim" if phase.state == "pending" else "")
            line.append(phase.title, style=title_style)
            if phase.detail:
                line.append(f"  {sep} {phase.detail}", style="dim")
            if phase.agents_total:
                line.append(f"  {sep} {phase.agents_done}/{phase.agents_total} agents", style="dim")
            rows.append(line)
        return rows

    def _format_workflow_renderable(self, activity: WorkflowActivity) -> Group:
        parts: list = [Text.from_markup(self._workflow_header(activity))]
        if activity.description:
            parts.append(self._format_inset_text(activity.description, style="dim"))
        parts.extend(self._workflow_phase_rows(activity))
        footer = self._workflow_footer_line(activity)
        if footer:
            parts.append(self._format_inset_text(footer, style="dim"))
        return Group(*parts)

    def _format_workflow_content(self, activity: WorkflowActivity) -> str:
        header = Text.from_markup(self._workflow_header(activity)).plain
        lines = [header]
        if activity.description:
            lines.append(activity.description)
        for phase in activity.phases:
            glyph, _ = self._workflow_phase_glyph(phase)
            row = f"{theme.g(glyph)} {phase.title}"
            if phase.detail:
                row += f" — {phase.detail}"
            if phase.agents_total:
                row += f" ({phase.agents_done}/{phase.agents_total} agents)"
            lines.append(row)
        footer = self._workflow_footer_line(activity)
        if footer:
            lines.append(footer)
        return "\n".join(lines)

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
        if not view.is_attached:
            # During teardown the view detaches before it is flagged closing, so query_one
            # still resolves it but mounting into it raises MountError.
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
        if not view.is_attached:
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
            return ToolEntryWidget(entry, self._tool_entry_title, self._tool_preview_renderable)
        if entry.kind == "compaction_summary":
            return ToolEntryWidget(entry, self._compaction_summary_title)
        if entry.kind == "sub_agent":
            return SubAgentEntryWidget(entry, self._format_conversation_entry)
        return ConversationEntryWidget(entry, self._format_conversation_entry)

    def _compaction_summary_title(self, entry: ConversationEntry) -> str:
        return theme.role_header(Glyph.STATUS, messages.COMPACTION_SUMMARY_TITLE, Color.ACCENT)

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
        if entry.kind == "workflow":
            wf = self._workflow_activity_for_entry(entry)
            if wf is not None:
                return self._format_workflow_renderable(wf)
            return Text(entry.content)
        if entry.kind == "sub_agent_task":
            header = theme.role_header(Glyph.USER, "Task", Color.USER)
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

    def _tool_preview_renderable(self, entry: ConversationEntry) -> Optional[Group]:
        """Inline diff/file-head preview for an edit tool, or None to hide the preview region."""
        preview = entry.edit_preview
        if not preview:
            return None
        try:
            return self._build_edit_preview(preview)
        except Exception:
            return None

    def _build_edit_preview(self, preview: dict) -> Group:
        kind = str(preview.get("kind") or "")
        path = str(preview.get("path") or "file")
        lines = preview.get("lines") or []
        more = int(preview.get("more") or 0)

        meta = Text()
        meta.append(escape(path), style="bold")
        if kind == "diff":
            meta.append("  ")
            meta.append(f"+{int(preview.get('adds') or 0)}", style=Color.SUCCESS)
            meta.append(" ")
            meta.append(f"-{int(preview.get('dels') or 0)}", style=Color.ERROR)

        if kind == "head":
            code = "\n".join(str(row[1]) for row in lines if isinstance(row, (list, tuple)) and len(row) >= 2)
            body = self._edit_preview_code(code, str(preview.get("language") or "text"))
        else:
            body = self._edit_preview_diff(lines)

        parts: list = [meta, Padding(body, (0, 0, 0, theme.INSET_WIDTH))]
        if more > 0:
            footer = Text(f"{theme.g(Glyph.ELLIPSIS)} +{more} more lines", style="dim")
            parts.append(Padding(footer, (0, 0, 0, theme.INSET_WIDTH)))
        return Group(*parts)

    def _edit_preview_diff(self, lines: list) -> Text:
        bar = f"{theme.g(Glyph.INSET_BAR)} "
        styles = {"add": Color.SUCCESS, "del": Color.ERROR, "meta": "dim", "context": "dim"}
        rendered = Text()
        for index, row in enumerate(lines):
            if isinstance(row, (list, tuple)) and len(row) >= 2:
                tag, text = str(row[0]), str(row[1])
            else:
                tag, text = "context", str(row)
            if index:
                rendered.append("\n")
            rendered.append(bar, style="dim")
            rendered.append(text, style=styles.get(tag))
        return rendered

    def _edit_preview_code(self, code: str, language: str):
        try:
            from rich.syntax import Syntax

            return Syntax(
                code,
                language or "text",
                theme=theme.markdown_code_theme(),
                background_color="default",
                word_wrap=False,
            )
        except Exception:
            # Unknown lexer / pathological content: fall back to plain inset text.
            bar = f"{theme.g(Glyph.INSET_BAR)} "
            rendered = Text()
            for index, line in enumerate(code.split("\n")):
                if index:
                    rendered.append("\n")
                rendered.append(bar, style="dim")
                rendered.append(line)
            return rendered

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
