from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from math import isfinite
from typing import Protocol, runtime_checkable

from rich.console import RenderableType
from rich.segment import Segment, Segments
from rich.style import Style
from textual import events
from textual.scrollbar import ScrollBar


@runtime_checkable
class _TranscriptView(Protocol):
    def transcript_scroll_metrics(self) -> tuple[float, float, float]: ...

    def transcript_thumb_rows(self, track_rows: int) -> int: ...

    def seek_transcript(self, fraction: float) -> None: ...


@dataclass(frozen=True, slots=True)
class _Track:
    length: int
    thumb: int
    fraction: float
    page: float
    seek: Callable[[float], None]

    @property
    def travel(self) -> int:
        return self.length - self.thumb

    @property
    def start(self) -> int:
        return round(self.fraction * self.travel)


@dataclass(slots=True)
class _Drag:
    start_y: float
    start_fraction: float
    grip: float
    last_y: float
    last_fraction: float


class TranscriptScrollBar(ScrollBar):
    _down_y: float | None = None
    _drag: _Drag | None = None

    def _track(self) -> _Track | None:
        parent = self.parent
        if not self.vertical or not isinstance(parent, _TranscriptView):
            return None
        position, total, visible = parent.transcript_scroll_metrics()
        if not all(map(isfinite, (position, total, visible))):
            return None

        length = self.size.height
        extent = total - visible
        if length < 2 or visible <= 0 or extent <= 0:
            return _Track(length, 0, 0, 0, parent.seek_transcript)
        thumb = max(0, min(length - 1, parent.transcript_thumb_rows(length)))
        fraction = max(0.0, min(position, extent)) / extent
        return _Track(length, thumb, fraction, min(1.0, visible / extent), parent.seek_transcript)

    def _render_bar(self, scrollbar_style: Style) -> RenderableType:
        track = self._track()
        if track is None:
            return super()._render_bar(scrollbar_style)
        blank = " " * self.size.width
        background = Style(bgcolor=scrollbar_style.bgcolor)
        if not track.thumb:
            return Segments([Segment(blank, background)] * track.length, new_lines=True)
        above = Segment(blank, background + Style(meta={"@mouse.down": "scroll_up"}))
        thumb = Segment(blank, Style(color=scrollbar_style.color, reverse=True, meta={"@mouse.down": "grab"}))
        below = Segment(blank, background + Style(meta={"@mouse.down": "scroll_down"}))
        start = track.start
        return Segments([above] * start + [thumb] * track.thumb + [below] * (track.travel - start), new_lines=True)

    def _page(self, direction: int) -> bool:
        track = self._track()
        if track is None:
            return False
        if track.thumb and self.grabbed is None:
            track.seek(max(0.0, min(1.0, track.fraction + direction * track.page)))
        return True

    def action_scroll_up(self) -> None:
        if not self._page(-1):
            super().action_scroll_up()

    def action_scroll_down(self) -> None:
        if not self._page(1):
            super().action_scroll_down()

    async def _on_mouse_down(self, event: events.MouseDown) -> None:
        self._down_y = event.pointer_screen_y

    def _on_mouse_capture(self, event: events.MouseCapture) -> None:
        track = self._track()
        y = event.mouse_position.y if self._down_y is None else self._down_y
        self._drag = None
        if track is not None and track.thumb:
            offset = y - self.region.y - track.start
            grip = max(0.0, min(track.thumb, offset))
            self._drag = _Drag(y, track.fraction, grip, y, track.fraction)
        self._down_y = None

    async def _on_mouse_move(self, event: events.MouseMove) -> None:
        drag = self._drag
        if self.grabbed is None or drag is None:
            return
        event.stop()
        event.prevent_default()  # Don't also dispatch ScrollBar's local ScrollTo.
        y = event.pointer_screen_y
        if y == drag.last_y:
            # Hiding the Jump bar at the tail can resize the track under a stationary pointer.
            return
        track = self._track()
        if track is None or not track.thumb:
            return
        drag.last_y = y
        row = y - self.region.y
        if row < 1:
            fraction = 0.0
        elif row >= self.size.height - 1:
            fraction = 1.0
        elif y == drag.start_y:
            fraction = drag.start_fraction
        else:
            fraction = max(0.0, min(1.0, (row - drag.grip) / track.travel))
        if fraction != drag.last_fraction:
            drag.last_fraction = fraction
            track.seek(fraction)
