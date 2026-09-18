"""Fullscreen startup splash widget for the Textual TUI."""

from __future__ import annotations

import time
from pathlib import Path
from typing import ClassVar

from rich.cells import cell_len
from rich.text import Text
from textual.timer import Timer
from textual.widgets import Static

from .. import theme
from .turn_status import _shimmer_palette, shimmer_text


_FRAME_INTERVAL = 0.08
_DEFAULT_STAGE = "Preparing workspace"

_KOLEGA: dict[str, tuple[str, ...]] = {
    "K": ("10001", "10010", "10100", "11000", "10100", "10010", "10001"),
    "O": ("01110", "10001", "10001", "10001", "10001", "10001", "01110"),
    "L": ("10000", "10000", "10000", "10000", "10000", "10000", "11111"),
    "E": ("11111", "10000", "10000", "11110", "10000", "10000", "11111"),
    "G": ("01111", "10000", "10000", "10111", "10001", "10001", "01111"),
    "A": ("01110", "10001", "10001", "11111", "10001", "10001", "10001"),
}

_CODE: dict[str, tuple[str, ...]] = {
    "C": ("111", "100", "100", "100", "111"),
    "O": ("111", "101", "101", "101", "111"),
    "D": ("110", "101", "101", "101", "110"),
    "E": ("111", "100", "110", "100", "111"),
}


def _pack_rows(rows: tuple[str, ...]) -> tuple[str, ...]:
    packed: list[str] = []
    for top_index in range(0, len(rows), 2):
        top = rows[top_index]
        bottom = rows[top_index + 1] if top_index + 1 < len(rows) else "0" * len(top)
        line = []
        for upper, lower in zip(top, bottom):
            match upper == "1", lower == "1":
                case True, True:
                    line.append("█")
                case True, False:
                    line.append("▀")
                case False, True:
                    line.append("▄")
                case _:
                    line.append(" ")
        packed.append("".join(line))
    return tuple(packed)


def _wordmark(word: str, glyphs: dict[str, tuple[str, ...]]) -> tuple[str, ...]:
    packed = [_pack_rows(glyphs[letter]) for letter in word]
    return tuple(" ".join(letter[row] for letter in packed) for row in range(len(packed[0])))


_KOLEGA_LINES = _wordmark("KOLEGA", _KOLEGA)
_CODE_LINES = _wordmark("CODE", _CODE)
_LOGO_WIDTH = max(cell_len(line) for line in _KOLEGA_LINES)
_FULL_SPLASH_MIN_HEIGHT = 11


class BootSplash(Static):
    """A single-widget, theme-aware cinematic boot splash.

    Public interface:
    - ``BootSplash(project_path: Path | None = None, version: str | None = None)``
    - ``set_stage(stage: str) -> None``
    - ``start_animation() -> None``
    - ``stop_animation() -> None``
    """

    COMPONENT_CLASSES: ClassVar[set[str]] = {
        "boot-splash--logo-base",
        "boot-splash--logo-accent",
        "boot-splash--logo-peak",
        "boot-splash--stage",
        "boot-splash--muted",
    }

    DEFAULT_CSS = """
    BootSplash {
        width: 100%;
        height: 100%;
        background: $background;
        color: $text-muted;
    }

    BootSplash .boot-splash--logo-base {
        color: $primary;
    }

    BootSplash .boot-splash--logo-accent {
        color: $secondary;
    }

    BootSplash .boot-splash--logo-peak {
        color: $text;
    }

    BootSplash .boot-splash--stage {
        color: $text;
    }

    BootSplash .boot-splash--muted {
        color: $text-muted;
    }
    """

    def __init__(self, project_path: Path | None = None, version: str | None = None) -> None:
        if version is None:
            from kolega_code import __version__ as version

        super().__init__("", id="boot_splash", markup=False)
        self.project_path = Path.cwd() if project_path is None else Path(project_path)
        self.version = version
        self._stage = _DEFAULT_STAGE
        self._timer: Timer | None = None
        self._started_at: float | None = None

    def set_stage(self, stage: str) -> None:
        """Update the single startup phase line."""
        self._stage = " ".join(str(stage).split()) or _DEFAULT_STAGE
        self.refresh(layout=False)

    def start_animation(self) -> None:
        """Start the repaint-only shimmer timer when the terminal supports it."""
        self._started_at = self._now()
        self._stop_timer(refresh=False)
        if self._can_animate():
            self._timer = self.set_interval(_FRAME_INTERVAL, self._on_animation_frame, name="boot-splash")
        self.refresh(layout=False)

    def stop_animation(self) -> None:
        """Stop all splash animation work and leave a static frame behind."""
        self._started_at = None
        self._stop_timer(refresh=False)
        self.refresh(layout=False)

    def on_unmount(self) -> None:
        self._started_at = None
        self._stop_timer(refresh=False)

    def render(self) -> Text:
        width = max(0, self.size.width)
        height = max(0, self.size.height)
        return self._render_for_size(width, height)

    def _now(self) -> float:
        return time.monotonic()

    def _on_animation_frame(self) -> None:
        if not self._can_animate():
            self._started_at = None
            self._stop_timer(refresh=True)
            return
        self.refresh(layout=False)

    def _stop_timer(self, *, refresh: bool) -> None:
        if self._timer is not None:
            self._timer.stop()
            self._timer = None
        if refresh:
            self.refresh(layout=False)

    def _can_animate(self) -> bool:
        if not self.is_mounted:
            return False
        console = self.app.console
        return (
            theme.supports_truecolor(console)
            and console.color_system is not None
            and not console.no_color
            and self.app.animation_level == "full"
        )

    def _elapsed(self) -> float:
        if self._started_at is None:
            return 0.0
        return max(0.0, self._now() - self._started_at)

    def _render_for_size(self, width: int, height: int) -> Text:
        if width <= 0 or height <= 0:
            return Text()
        if width < _LOGO_WIDTH or height < _FULL_SPLASH_MIN_HEIGHT:
            lines = self._compact_lines(width, height)
        else:
            lines = self._full_lines(width, height)
        return self._compose_lines(lines, width)

    def _full_lines(self, width: int, height: int) -> list[Text]:
        logo_width = min(_LOGO_WIDTH, width)
        code = [_center_padded_plain(line, logo_width) for line in _CODE_LINES]
        logo_plain = [*_KOLEGA_LINES, " " * logo_width, *code]
        logo = self._center_logo(self._style_logo(logo_plain), width, logo_width)

        stage = self._line(self._fit(f"▶ {self._stage}", width), width, "boot-splash--stage")
        meta = self._line(self._fit(self._metadata(), width), width, "boot-splash--muted")
        body: list[Text] = [*logo, Text(), stage, meta]
        top_pad = max(0, (height - len(body)) // 2)
        body = [Text() for _ in range(top_pad)] + body
        return body[:height]

    def _compact_lines(self, width: int, height: int) -> list[Text]:
        body: list[Text] = []
        if height >= 1:
            body.append(self._line(self._fit("KOLEGA CODE", width), width, "boot-splash--logo-base"))
        if height >= 2:
            body.append(self._line(self._fit(self._stage, width), width, "boot-splash--stage"))
        if height >= 3:
            body.append(self._line(self._fit(self._metadata(), width), width, "boot-splash--muted"))
        top_pad = max(0, (height - len(body)) // 2)
        return ([Text() for _ in range(top_pad)] + body)[:height]

    def _style_logo(self, lines: list[str]) -> list[Text]:
        base = self.get_component_rich_style("boot-splash--logo-base")
        logo_width = max((cell_len(line) for line in lines), default=0)
        padded_lines = [_pad_cells(line, logo_width) for line in lines]
        if not self._can_animate():
            return [Text(line, style=base) for line in padded_lines]

        base_color = base.color
        accent = self.get_component_rich_style("boot-splash--logo-accent").color
        peak = self.get_component_rich_style("boot-splash--logo-peak").color
        if base_color is None or accent is None or peak is None:
            return [Text(line, style=base) for line in padded_lines]
        palette = _shimmer_palette(base_color, accent, peak)
        elapsed = self._elapsed()
        # Every line is padded to the same width, so shimmer_text samples one shared
        # sweep coordinate across the whole logo instead of per-row independent cycles.
        return [shimmer_text(Text(line, style=base), elapsed, palette) for line in padded_lines]

    def _center_logo(self, lines: list[Text], width: int, logo_width: int) -> list[Text]:
        prefix = " " * max(0, (width - logo_width) // 2)
        centered: list[Text] = []
        for line in lines:
            output = Text(prefix)
            output.append(line)
            centered.append(output)
        return centered

    def _metadata(self) -> str:
        return f"{self.project_path} · v{self.version}"

    def _line(self, plain: str, width: int, component: str) -> Text:
        return Text(_center_plain(plain, width), style=self.get_component_rich_style(component))

    def _fit(self, plain: str, width: int) -> str:
        return _truncate_cells(plain, width)

    def _compose_lines(self, lines: list[Text], width: int) -> Text:
        output = Text()
        for index, line in enumerate(lines):
            fitted = line.copy()
            if fitted.cell_len > width:
                fitted.truncate(width, overflow="ellipsis")
            output.append(fitted)
            if index < len(lines) - 1:
                output.append("\n")
        return output


def _center_plain(plain: str, width: int) -> str:
    if width <= 0:
        return ""
    fitted = _truncate_cells(plain, width)
    return " " * max(0, (width - cell_len(fitted)) // 2) + fitted


def _center_padded_plain(plain: str, width: int) -> str:
    if width <= 0:
        return ""
    fitted = _truncate_cells(plain, width)
    left = max(0, (width - cell_len(fitted)) // 2)
    return _pad_cells(" " * left + fitted, width)


def _pad_cells(plain: str, width: int) -> str:
    if width <= 0:
        return ""
    return plain + " " * max(0, width - cell_len(plain))


def _truncate_cells(plain: str, width: int) -> str:
    if width <= 0:
        return ""
    if cell_len(plain) <= width:
        return plain
    ellipsis = theme.g(theme.Glyph.ELLIPSIS)
    if cell_len(ellipsis) > width:
        return ""
    budget = width - cell_len(ellipsis)
    out: list[str] = []
    used = 0
    for char in plain:
        char_width = cell_len(char)
        if used + char_width > budget:
            break
        out.append(char)
        used += char_width
    return "".join(out) + ellipsis


__all__ = ["BootSplash"]
