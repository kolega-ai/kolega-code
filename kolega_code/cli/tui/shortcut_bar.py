"""Focus-sensitive hints for the main screen, without changing key handling."""

from __future__ import annotations

from dataclasses import dataclass

from rich.cells import cell_len
from rich.text import Text
from textual import events
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical, VerticalScroll
from textual.screen import ModalScreen, Screen
from textual.widgets import Button, Footer, Static
from textual.widgets._footer import FooterKey

from .widgets import ActionList, ChatComposer


@dataclass(frozen=True)
class Shortcut:
    """A displayed hint; clicks send ``key`` through the normal binding chain."""

    key: str
    key_display: str
    description: str
    action: str = ""

    @property
    def cell_width(self) -> int:
        """Compact FooterKey text plus its two-cell inter-item margin."""
        return cell_len(self.key_display) + 1 + cell_len(self.description) + 2


class ContextFooter(Footer):
    """Main-screen footer. State owners call :meth:`refresh_context` after changes.

    The app supplies an F1 ``shortcut_help`` binding/action, which closes an
    already-open ShortcutHelpScreen or opens one on the root screen. This widget
    only chooses what to advertise: omitted bindings remain untouched.
    """

    DEFAULT_CSS = """
    ContextFooter {
        height: 1;
        padding: 0;
        overflow: hidden hidden;
        background: $background;
        color: $text-muted;
    }
    ContextFooter.-compact FooterKey {
        margin: 0 2 0 0;
        padding: 0;
        border: none;
        background: $background;
        .footer-key--key {
            padding: 0;
            color: $text;
            background: $background;
        }
        .footer-key--description {
            padding: 0 0 0 1;
            color: $text-muted;
            background: $background;
        }
    }
    ContextFooter.-compact FooterKey.shortcut-overflow {
        margin: 0;
    }
    """

    def __init__(self, *, id: str | None = None) -> None:
        super().__init__(id=id, show_command_palette=False, compact=True)
        self._context_pending: bool = False
        self._shown: tuple[Shortcut, ...] = ()

    def context_shortcuts(self) -> tuple[Shortcut, ...]:
        """Return ordered hints for the actual focus, before width fitting."""
        app = self.app
        screen = self.screen
        focused = screen.focused
        bindings = screen.active_bindings
        hints: list[Shortcut] = []

        def add(key: str, label: str, description: str) -> None:
            active = bindings.get(key)
            if active is not None and active.enabled:
                hints.append(Shortcut(key, label, description, active.binding.action))

        working = bool(getattr(app, "_turn_active", False)) or getattr(app, "agent_worker", None) is not None
        active_actions = getattr(app, "_active_prompt_actions", None)
        actions = active_actions() if callable(active_actions) else None
        if isinstance(focused, ActionList) and focused is actions and not focused.disabled:
            add("down", "↑↓", "choose")
            add("enter", "Enter", "select")
        elif isinstance(focused, ChatComposer) and not focused.disabled:
            dropdown = focused.mention_dropdown()
            if dropdown is not None and dropdown.display:
                add("down", "↑↓", "choose")
                add("enter", "Enter", "select")
                add("escape", "Esc", "dismiss")
            else:
                question = getattr(app, "_pending_question", None) is not None
                if question:
                    add("enter", "Enter", "answer")
                elif getattr(app, "agent", None) is not None:
                    add("enter", "Enter", "queue" if working else "send")
                add("shift+enter", "Shift+Enter", "newline")
                if not working and not question:
                    # Slash is text input, not a binding; a literal character
                    # makes simulate_key preserve TextArea's insertion behavior.
                    hints.append(Shortcut("/", "/", "commands"))
                    add("shift+tab", "Shift+Tab", "Plan/Build")
        add("ctrl+o", "Ctrl+O", "sidebar")
        if working:
            add("ctrl+g", "Ctrl+G", "agents")
        return tuple(hints)

    def _fitted_shortcuts(self) -> tuple[Shortcut, ...]:
        width = self.content_size.width
        if width < 1:
            return ()
        remaining = width - 1  # Keep the help ellipsis even at one column.
        fitted: list[Shortcut] = []
        for hint in self.context_shortcuts():
            if hint.cell_width <= remaining:
                fitted.append(hint)
                remaining -= hint.cell_width
        fitted.append(Shortcut("f1", "…", "", "shortcut_help"))
        return tuple(fitted)

    def compose(self) -> ComposeResult:
        for hint in self._shown:
            yield FooterKey(
                hint.key,
                hint.key_display,
                hint.description,
                hint.action,
                tooltip="All shortcuts (F1)" if hint.key == "f1" else "",
                classes="shortcut-overflow" if hint.key == "f1" else "",
            ).data_bind(compact=Footer.compact)

    def refresh_context(self) -> None:
        """Coalesce state invalidations; unchanged hints do not remount widgets."""
        if not self.is_attached or self._context_pending:
            return
        self._context_pending = True
        self.call_next(self._refresh_context)

    def _refresh_context(self) -> None:
        self._context_pending = False
        shown = self._fitted_shortcuts()
        if shown != self._shown:
            self._shown = shown
            self.refresh(recompose=True)

    def bindings_changed(self, screen: Screen) -> None:
        """Footer's subscription covers focus changes and screen-stack resumes."""
        if self.is_attached and screen is self.screen:
            self.refresh_context()

    def on_mount(self) -> None:
        super().on_mount()
        self.refresh_context()

    def on_resize(self, event: events.Resize) -> None:
        self.refresh_context()


class ShortcutHelpScreen(ModalScreen[None]):
    """Local shortcut reference. Never routes through the agent's ``/help``."""

    BINDINGS = [
        Binding("escape,f1", "close", "Close", priority=True),
    ]

    DEFAULT_CSS = """
    ShortcutHelpScreen {
        align: center middle;
        background: $background 70%;
    }
    ShortcutHelpScreen #shortcut-help-dialog {
        width: 76;
        max-width: 100%;
        height: 85%;
        border: round $primary;
        padding: 1 2;
        background: $surface;
    }
    ShortcutHelpScreen #shortcut-help-title {
        height: auto;
        text-style: bold;
        margin-bottom: 1;
    }
    ShortcutHelpScreen #shortcut-help-scroll {
        height: 1fr;
    }
    ShortcutHelpScreen #shortcut-help-content {
        height: auto;
    }
    ShortcutHelpScreen #shortcut-help-close {
        margin-top: 1;
    }
    """

    def _reference(self) -> Text:
        text = Text()
        sections = (
            (
                "Composer",
                "Enter — send; queue while working; answer a question in the text field.\n"
                "Shift+Enter — newline. Ctrl+J is the portable fallback; Ctrl+Enter also works.\n"
                "/ — command completion at the start of a draft; @ — file completion.\n"
                "Up / Down — move the cursor; recall prompt history at the draft boundaries.\n"
                "Ctrl+L / Cmd+A — select all.\n"
                "Ctrl+Shift+V — paste an image. Alt+V is the portable fallback; /attach also works.\n",
            ),
            (
                "Completion",
                "Up / Down — choose a match. Enter / Tab — select. Esc — dismiss.\n"
                "Completion takes precedence over sending or answering a question.\n",
            ),
            (
                "Prompt options",
                "Up / Down — choose. Enter — select. 1–9 — select the numbered option.\n"
                "Down from the last option — enter the composer when it is enabled.\n"
                "Up from the composer's first line — return to the active options.\n",
            ),
            (
                "Local help",
                "F1 / … — open this reference. Esc / F1 — close and return to your draft.\n"
                "/help is an agent command, not this local reference.\n"
                "Extended key combinations depend on the terminal; use the fallbacks above.\n",
            ),
        )
        for title, body in sections:
            text.append(title + "\n", style="bold")
            text.append(body + "\n")

        # Include hidden and width-omitted bindings. Keep focus scopes separate:
        # for example, Escape dismisses completion in the editor, not the turn.
        for title, declarations in (
            ("App key reference", self.app.BINDINGS),
            ("Editor key reference", ChatComposer.BINDINGS),
            ("Option list key reference", ActionList.BINDINGS),
        ):
            all_bindings = {binding.key: binding for binding in Binding.make_bindings(declarations)}
            grouped: dict[tuple[str, str], list[str]] = {}
            for binding in all_bindings.values():
                grouped.setdefault((binding.action, binding.description), []).append(binding.key)
            text.append(title + "\n", style="bold")
            for (_, description), keys in grouped.items():
                text.append(f"{' / '.join(keys)} — {description}\n")
            text.append("\n")
        return text

    def compose(self) -> ComposeResult:
        with Vertical(id="shortcut-help-dialog"):
            yield Static("Keyboard shortcuts", id="shortcut-help-title")
            with VerticalScroll(id="shortcut-help-scroll"):
                yield Static(self._reference(), id="shortcut-help-content")
            yield Button("Close (Esc)", id="shortcut-help-close")

    def action_close(self) -> None:
        # Textual keeps the underlying Screen's focused widget and selection.
        self.dismiss(None)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "shortcut-help-close":
            event.stop()
            self.action_close()
