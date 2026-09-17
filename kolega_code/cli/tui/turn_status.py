"""A theme-aware shimmer sampled by the existing working-strip timer."""

from __future__ import annotations

from functools import lru_cache
from typing import ClassVar

from rich.cells import cell_len
from rich.color import Color
from rich.style import Style
from rich.text import Text
from textual.widgets import Static

from .. import theme


_CELLS_PER_SECOND = 8.0
_BAND_RADIUS = 3.5
_SWEEP_GAP = 4.0


@lru_cache(maxsize=32)
def _shimmer_palette(base: Color, accent: Color, peak: Color) -> tuple[Style, ...]:
    colors = theme.gradient_hex(base.get_truecolor().hex, accent.get_truecolor().hex, 9)
    colors += theme.gradient_hex(accent.get_truecolor().hex, peak.get_truecolor().hex, 9)[1:]
    return tuple(Style(color=color) for color in colors)


def shimmer_text(label: Text, elapsed: float, palette: tuple[Style, ...]) -> Text:
    """Sweep a soft highlight in terminal cells, preserving text and inline styles."""
    result = label.copy()
    if not label or not palette:
        return result
    center = (max(0.0, elapsed) * _CELLS_PER_SECOND) % (label.cell_len + 2 * _BAND_RADIUS + _SWEEP_GAP)
    center -= _BAND_RADIUS
    cell = 0
    run_start = 0
    level = 0
    for index, character in enumerate(label.plain):
        width = cell_len(character)
        # Combining marks retain their base character's color.
        next_level = level
        if width:
            distance = abs(cell + (width - 1) / 2 - center)
            strength = max(0.0, 1.0 - distance / _BAND_RADIUS)
            strength = strength * strength * (3.0 - 2.0 * strength)
            next_level = round(strength * (len(palette) - 1))
            cell += width
        if next_level != level:
            if level:
                result.stylize(palette[level], run_start, index)
            run_start, level = index, next_level
    if level:
        result.stylize(palette[level], run_start, len(label))
    return result


class TurnStatus(Static):
    """A single static widget; no additional timer, layout, or animation worker."""

    COMPONENT_CLASSES: ClassVar[set[str]] = {"turn-status--shimmer-accent", "turn-status--shimmer-peak"}
    DEFAULT_CSS = """
    TurnStatus .turn-status--shimmer-accent {
        color: $accent;
    }
    TurnStatus .turn-status--shimmer-peak {
        color: $text;
    }
    """

    def shimmer(self, label: Text, elapsed: float) -> Text:
        console = self.app.console
        if (
            not theme.supports_truecolor(console)
            or console.color_system is None
            or console.no_color
            or self.app.animation_level != "full"
        ):
            return label
        base = self.rich_style.color
        accent = self.get_component_rich_style("turn-status--shimmer-accent").color
        peak = self.get_component_rich_style("turn-status--shimmer-peak").color
        if base is None or accent is None or peak is None:
            return label
        return shimmer_text(label, elapsed, _shimmer_palette(base, accent, peak))
