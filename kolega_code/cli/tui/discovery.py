"""Local, opt-in discovery tips; the startup surface owns visibility."""

from __future__ import annotations

from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.message import Message
from textual.widgets import Button, Static

DISCOVERY_TIPS: tuple[str, ...] = (
    "Try /plan to explore a change before editing files.",
    "Ctrl+G opens the sub-agent inspector without losing your place.",
    "Type @ in the composer to mention a file in your project.",
)


class DiscoveryTip(Horizontal):
    """One selectable tip and a focusable Next button.

    Mount only when ``settings.discovery_tips`` is true on a fresh session.
    The parent must hide/remove this widget once the session is no longer fresh
    or the setting is disabled. Each instance starts at the first tip; cycling
    is local and never persisted, timed, or sent over the network.
    """

    class Changed(Message):
        """Keep the UI-only index when the transcript remounts its startup row."""

        def __init__(self, index: int) -> None:
            super().__init__()
            self.index = index

    DEFAULT_CSS = """
    DiscoveryTip {
        height: auto;
        width: 1fr;
        margin-top: 1;
        background: transparent;
    }
    DiscoveryTip .discovery-tip-text {
        width: 1fr;
        height: auto;
        color: $text-muted;
    }
    DiscoveryTip .discovery-tip-next {
        width: auto;
        min-width: 6;
        height: 1;
        margin: 0 0 0 1;
        padding: 0 1;
        border: none;
        background: transparent;
        background-tint: transparent;
        color: $accent;
        text-style: none;
    }
    DiscoveryTip .discovery-tip-next:hover,
    DiscoveryTip .discovery-tip-next:focus {
        background: $surface-lighten-2;
        color: $text;
        text-style: underline;
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

    def compose(self) -> ComposeResult:
        # Native string Content keeps selection offsets and copy support; no
        # Rich markup, links, terminal escape sequences, or custom renderables.
        yield Static(f"Tip: {self.current_tip}", markup=False, classes="discovery-tip-text")
        button = Button(
            "Next",
            classes="discovery-tip-next",
            compact=True,
            tooltip="Next discovery tip",
        )
        button.active_effect_duration = 0
        yield button

    def action_next_tip(self) -> None:
        """Advance once, wrapping through the fixed list."""
        self._tip_index = (self._tip_index + 1) % len(DISCOVERY_TIPS)
        if self.is_mounted:
            self.query_one(".discovery-tip-text", Static).update(f"Tip: {self.current_tip}")
        self.post_message(self.Changed(self._tip_index))

    @on(Button.Pressed, ".discovery-tip-next")
    def _next_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        self.action_next_tip()
