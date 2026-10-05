"""Form controls with predictable interaction inside the settings editor."""

from typing import TypeVar

from textual import events, on
from textual.widgets import Select

SelectValue = TypeVar("SelectValue")


class SettingsSelect(Select[SelectValue]):
    """Keep an open dropdown's wheel events out of the surrounding form."""

    @on(events.MouseScrollDown)
    @on(events.MouseScrollUp)
    @on(events.MouseScrollLeft)
    @on(events.MouseScrollRight)
    def contain_dropdown_scroll(self, event: events.MouseEvent) -> None:
        if self.expanded:
            # The overlay handles scrolling first. At its boundaries (or when
            # all options fit), Textual bubbles the wheel to this Select.
            # Stop here, before the form scrolls and moves the open menu.
            event.stop()
            event.prevent_default()
