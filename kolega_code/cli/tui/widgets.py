"""Reusable Textual widgets for the CLI TUI."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Optional

from rich.segment import Segment
from rich.style import Style
from rich.text import Text
from textual import events
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical, VerticalScroll
from textual.message import Message as TextualMessage
from textual.selection import Selection
from textual.strip import Strip
from textual.widgets import Collapsible, OptionList, RichLog, Static, TextArea
from textual.widgets._collapsible import CollapsibleTitle
from textual.widgets.option_list import Option

from ..file_index import IndexEntry
from ..slash_commands import SlashCommandEntry
from .state import ConversationEntry

class ConversationEntryWidget(Static):
    """Displays one ConversationEntry and is updated in place as the entry changes."""

    def __init__(self, entry: ConversationEntry, format_entry: Callable[[ConversationEntry], object]) -> None:
        super().__init__("", markup=False)
        self.entry = entry
        self._format_entry = format_entry
        self._kind_class = ""
        self._formatted: object = None
        self._content_snapshot: tuple[object, ...] | None = None
        self.refresh_content()

    def refresh_content(self) -> None:
        kind_class = f"entry-{self.entry.kind}"
        if kind_class != self._kind_class:
            if self._kind_class:
                self.remove_class(self._kind_class)
            self.add_class(kind_class)
            self._kind_class = kind_class

        snapshot = self._entry_snapshot()
        if snapshot == self._content_snapshot and self._formatted is not None:
            return
        self._content_snapshot = snapshot
        self._formatted = self._format_entry(self.entry)
        self.update(self._formatted)

    def _entry_snapshot(self) -> tuple[object, ...]:
        return (
            self.entry.kind,
            self.entry.content,
            self.entry.complete,
            self.entry.tool_name,
            self.entry.tone,
            self.entry.full_content,
            repr(self.entry.edit_preview),
        )

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
        self._title = ""
        self._body_content: object = None
        self._preview_key = ""
        self._preview_visible = False

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

        title = self._title_factory(self.entry)
        if title != self._title:
            self._collapsible.title = title
            self._title = title

        body_content = self.entry.full_content or self.entry.content
        if body_content != self._body_content:
            self._body.update(body_content)
            self._body_content = body_content

        if self._preview is not None:
            preview_key = repr(self.entry.edit_preview)
            if preview_key == self._preview_key:
                return
            self._preview_key = preview_key
            renderable = self._preview_factory(self.entry) if self._preview_factory else None
            if renderable is not None:
                self._preview.update(renderable)
                if not self._preview_visible:
                    self._preview.display = True
                    self._preview_visible = True
            else:
                if self._preview_visible:
                    self._preview.update("")
                    self._preview.display = False
                    self._preview_visible = False


class ConversationView(VerticalScroll):
    """Scrollable list of per-entry widgets, anchored to the bottom while streaming."""

    bottom_tolerance = 1

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.auto_follow_bottom = True

    def is_at_bottom(self) -> bool:
        """Return whether the current scroll position is effectively at the end."""
        return self.max_scroll_y <= 0 or self.scroll_y >= self.max_scroll_y - self.bottom_tolerance

    def set_auto_follow(self, enabled: bool) -> None:
        """Record whether transcript updates should keep the view pinned to the end."""
        self.auto_follow_bottom = enabled

    def watch_scroll_y(self, old_value: float, new_value: float) -> None:
        super().watch_scroll_y(old_value, new_value)
        self.auto_follow_bottom = self.is_at_bottom()
        try:
            update = getattr(self.app, "_update_jump_button", None)
        except Exception:
            return
        if update is not None:
            update()


class StickyRichLog(RichLog):
    """RichLog with sticky bottom-follow semantics.

    ``RichLog`` defaults to unconditional auto-scroll on every write. That is
    wrong for sidebar streams because manual scrollback snaps back to the newest
    output as soon as another line arrives. This widget keeps the view pinned
    only while the user is already at the bottom.
    """

    bottom_tolerance = 1

    def __init__(self, *args, **kwargs) -> None:
        kwargs["auto_scroll"] = False
        super().__init__(*args, **kwargs)
        self.auto_follow_bottom = True

    def is_at_bottom(self) -> bool:
        """Return whether the current scroll position is effectively at the end."""
        return self.max_scroll_y <= 0 or self.scroll_y >= self.max_scroll_y - self.bottom_tolerance

    def watch_scroll_y(self, old_value: float, new_value: float) -> None:
        super().watch_scroll_y(old_value, new_value)
        self.auto_follow_bottom = self.is_at_bottom()

    def write_sticky(self, content: object) -> None:
        """Append content, following only if the user was at the bottom."""
        should_follow = self.auto_follow_bottom or self.is_at_bottom()
        self.write(content, scroll_end=should_follow)


class TerminalOutputLog(StickyRichLog):
    """Sticky log for terminal output."""

    def write_terminal(self, content: object) -> None:
        self.write_sticky(content)


class LogOutputLog(StickyRichLog):
    """Sticky log for diagnostic messages."""

    def write_log(self, content: object) -> None:
        self.write_sticky(content)


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
        with VerticalScroll(classes="prompt-header-scroll"):
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
