"""Local, opt-in quick-tip modal."""

from __future__ import annotations

from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, VerticalScroll
from textual.content import Content
from textual.screen import ModalScreen
from textual.widgets import Button, Static

DISCOVERY_TIPS: tuple[str, ...] = (
    "Try /plan to explore a change before editing files.",
    "Ctrl+G opens the sub-agent inspector without losing your place.",
    "Type @ in the composer to mention a file in your project.",
    "Use /loop with a schedule and prompt to repeat a check.",
    "Use /rewind to review changes before restoring an earlier turn.",
    "In Build mode, ask to create and switch to a Git worktree for isolated changes.",
    "Open /memory to browse and edit private project memory.",
    "Try /handoff to start a fresh session with a summary of this one.",
    "Run /skills to discover reusable workflows available in this project.",
    "Use /permissions to check the current shell and edit approval mode.",
    "Run /copy to copy the last response to your clipboard.",
    "Run /lsp to inspect language-server status and spot missing servers.",
)


class DiscoveryTip(ModalScreen[None]):
    """One dismissible tip modal; cycling stays local and never auto-rotates."""

    BINDINGS = [Binding("escape", "close", "Close tip", show=False, priority=True)]
    AUTO_FOCUS = "#discovery-tip-close"

    DEFAULT_CSS = """
    DiscoveryTip {
        align: center middle;
        padding: 1;
        background: $background 60%;
    }
    DiscoveryTip .discovery-tip-dialog {
        width: 64;
        max-width: 100%;
        height: auto;
        max-height: 100%;
        border: round $accent;
        padding: 1 2;
        background: $surface;
    }
    DiscoveryTip .discovery-tip-header {
        height: 1;
    }
    DiscoveryTip .discovery-tip-badge {
        width: auto;
        height: 1;
        padding: 0 1;
        background: $accent;
        color: auto 100%;
        text-style: bold;
    }
    DiscoveryTip .discovery-tip-position {
        width: 1fr;
        height: 1;
        margin-left: 1;
        text-align: right;
        color: $text-muted;
    }
    DiscoveryTip .discovery-tip-text {
        width: 1fr;
        height: auto;
        margin: 1 0;
        color: $text;
    }
    DiscoveryTip .discovery-tip-actions {
        height: 1;
    }
    DiscoveryTip .discovery-tip-hint {
        width: 1fr;
        height: 1;
        color: $text-muted;
    }
    DiscoveryTip .discovery-tip-dialog Button {
        width: auto;
        min-width: 8;
        height: 1;
        margin: 0 0 0 1;
        padding: 0 1;
        border: none;
        background: transparent;
        background-tint: transparent;
        color: $accent;
        text-style: none;
    }
    DiscoveryTip .discovery-tip-dialog Button:hover,
    DiscoveryTip .discovery-tip-dialog Button:focus {
        background: $accent;
        color: auto 100%;
        text-style: bold underline;
        tint: transparent;
    }
    """

    def __init__(self, *, index: int = 0, id: str | None = None, classes: str | None = None) -> None:
        super().__init__(id=id, classes=classes)
        self._tip_index: int = index % len(DISCOVERY_TIPS)

    @property
    def current_tip(self) -> str:
        """The current literal tip, without the display label."""
        return DISCOVERY_TIPS[self._tip_index]

    def _tip_content(self) -> Content:
        # Native Content preserves selection/copy offsets and literal text.
        # Reverse-video keycaps also stand out in monochrome terminals.
        return Content(self.current_tip).highlight_regex(r"/\w+|Ctrl\+\w+|@", style="bold reverse")

    def _position_label(self) -> str:
        return f"{self._tip_index + 1} / {len(DISCOVERY_TIPS)}"

    def compose(self) -> ComposeResult:
        with VerticalScroll(classes="discovery-tip-dialog"):
            with Horizontal(classes="discovery-tip-header"):
                yield Static("QUICK TIP", markup=False, classes="discovery-tip-badge")
                yield Static(self._position_label(), markup=False, classes="discovery-tip-position")
            yield Static(self._tip_content(), markup=False, classes="discovery-tip-text")
            with Horizontal(classes="discovery-tip-actions"):
                yield Static("Esc", classes="discovery-tip-hint")
                next_button = Button("Next →", classes="discovery-tip-next", compact=True, tooltip="Next discovery tip")
                next_button.active_effect_duration = 0
                yield next_button
                close_button = Button("Got it", id="discovery-tip-close", compact=True, tooltip="Close tip (Esc)")
                close_button.active_effect_duration = 0
                yield close_button

    def action_next_tip(self) -> None:
        """Advance once, wrapping through the fixed list."""
        self._tip_index = (self._tip_index + 1) % len(DISCOVERY_TIPS)
        if self.is_mounted:
            self.query_one(".discovery-tip-text", Static).update(self._tip_content())
            self.query_one(".discovery-tip-position", Static).update(self._position_label())

    @on(Button.Pressed, ".discovery-tip-next")
    def _next_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        self.action_next_tip()

    @on(Button.Pressed, "#discovery-tip-close")
    def _close_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        self.action_close()

    def action_close(self) -> None:
        self.dismiss()
