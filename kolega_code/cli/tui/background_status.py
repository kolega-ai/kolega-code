"""Small, report-free presentation state for the fixed background activity strip."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rich.text import Text

from .. import theme
from ..theme import Color, Glyph


@dataclass(frozen=True)
class BackgroundStatus:
    running: int = 0
    queued: int = 0
    ready: int = 0
    failed: int = 0
    interrupted: int = 0
    saving: int = 0
    storage_failed: int = 0
    delivery_failed: bool = False

    @classmethod
    def from_rows(
        cls, rows: list[dict[str, Any]], acknowledged: set[str], *, delivery_failed: bool = False
    ) -> BackgroundStatus:
        counts = dict.fromkeys(("running", "queued", "ready", "failed", "interrupted", "saving", "storage_failed"), 0)
        for row in rows:
            if row["archived"]:
                continue
            status = row["status"]
            if row["persistence_error"]:
                kind = "storage_failed"
            elif status in {"queued", "running"}:
                kind = status
            elif row["settling"]:
                kind = "saving"
            elif status in {"failed", "interrupted"} and row["completion_id"] not in acknowledged:
                kind = status
            elif row["ready"]:
                kind = "ready"
            else:
                continue
            counts[kind] += 1
        return cls(**counts, delivery_failed=delivery_failed)

    @property
    def visible(self) -> bool:
        return bool(self.total or self.delivery_failed)

    @property
    def animating(self) -> bool:
        return bool(self.running or self.queued or self.saving)

    @property
    def total(self) -> int:
        return (
            self.running + self.queued + self.ready + self.failed + self.interrupted + self.saving + self.storage_failed
        )

    def render(self, frame: str, *, width: int | None = None) -> Text:
        """Show counts, state, and the inspector shortcut; task details live in the inspector."""
        if not self.visible or (width is not None and width <= 0):
            return Text()
        warning = self.failed or self.interrupted or self.storage_failed or self.delivery_failed
        glyph = theme.g(Glyph.CROSS) if warning else frame if self.animating else theme.g(Glyph.CHECK)
        color = Color.WARNING if warning else Color.ACCENT if self.animating else Color.SUCCESS
        parts = [
            f"{count} {label}"
            for count, label in (
                (self.running, "running"),
                (self.queued, "queued"),
                (self.ready, "result ready" if self.ready == 1 else "results ready"),
                (self.failed, "failed"),
                (self.interrupted, "interrupted"),
                (self.saving, "saving result" if self.saving == 1 else "saving results"),
                (self.storage_failed, "save failed"),
            )
            if count
        ]
        if self.delivery_failed:
            parts.append("delivery failed")
        noun = "agent" if self.total == 1 else "agents"
        if self.running == self.total and not self.delivery_failed:
            summary = f"{self.total} background {noun}"
        elif self.queued == self.total and not self.delivery_failed:
            summary = f"{self.total} queued {noun}"
        elif self.failed == self.total and not self.delivery_failed:
            summary = f"{self.total} failed {noun}"
        elif self.interrupted == self.total and not self.delivery_failed:
            summary = f"{self.total} interrupted {noun}"
        elif self.ready == self.total and not self.delivery_failed:
            summary = parts[0]
        else:
            summary = f"{self.total} {noun} · " + " · ".join(parts)
        short = f"{self.total} {noun}"
        if warning:
            short = " · ".join(part for part in parts if any(word in part for word in ("failed", "interrupted")))
        label_style = Color.WARNING if warning else ""
        for candidate in (summary, short, f"{self.total} {noun}"):
            text = Text()
            text.append(f"{glyph} ", style=color)
            text.append(candidate, style=label_style)
            text.append(" · Ctrl+G", style="dim")
            if width is None or text.cell_len <= width:
                return text
        text = Text(f"{self.total} {noun}", style=label_style)
        text.append(" · Ctrl+G", style="dim")
        if width is not None:
            if text.cell_len > width:
                text = Text(str(self.total), style=label_style)
                text.append(" · Ctrl+G", style="dim")
            text.truncate(width, overflow="ellipsis")
        return text
