"""Interactive ``/context`` modal: visual grid and categorized token breakdown."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from rich.text import Text
from textual import events
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Static

from kolega_code.agent.context_breakdown import (
    CATEGORY_AGENT_RESPONSES,
    CATEGORY_AUTOCOMPACT_BUFFER,
    CATEGORY_BUILTIN_TOOLS,
    CATEGORY_COMPACTION_SUMMARY,
    CATEGORY_FREE,
    CATEGORY_GUIDANCE_MEMORY,
    CATEGORY_IMAGES,
    CATEGORY_MCP_TOOLS,
    CATEGORY_REASONING,
    CATEGORY_SERVER_OVERHEAD,
    CATEGORY_SKILLS,
    CATEGORY_SYSTEM_PROMPT,
    CATEGORY_SYSTEM_REMINDERS,
    CATEGORY_TOOL_USAGE,
    CATEGORY_USER_MESSAGES,
    ContextBreakdown,
    allocate_grid_cells,
    analyze_agent_context,
    format_context_breakdown_text,
    format_token_count,
)

from .. import theme
from ..theme import Color, Glyph

if TYPE_CHECKING:
    from ..app import KolegaCodeApp

_GRID_ROWS = 10
_STACKED_WIDTH_BREAKPOINT = 92
_WIDE_GRID_BREAKPOINT = 120
_NARROW_GRID_BREAKPOINT = 54


def _ui_text(value: str) -> str:
    return value.replace(Glyph.BULLET_SEP, theme.g(Glyph.BULLET_SEP)).replace(Glyph.ELLIPSIS, theme.g(Glyph.ELLIPSIS))


def _category_glyph_and_style(key: str) -> tuple[str, str]:
    """Return ``(glyph, rich_style)`` using active theme colors and ASCII fallbacks."""
    g = theme.g
    mapping = {
        CATEGORY_SYSTEM_PROMPT: (g(Glyph.CTX_STATIC), "bold"),
        CATEGORY_BUILTIN_TOOLS: (g(Glyph.CTX_STATIC), Color.ACCENT),
        CATEGORY_MCP_TOOLS: (g(Glyph.CTX_STATIC), Color.WARNING),
        CATEGORY_SKILLS: (g(Glyph.CTX_STATIC), Color.SUCCESS),
        CATEGORY_GUIDANCE_MEMORY: (g(Glyph.CTX_STATIC), Color.AGENT),
        CATEGORY_SYSTEM_REMINDERS: (g(Glyph.AGENT), Color.ERROR),
        CATEGORY_COMPACTION_SUMMARY: (g(Glyph.SUB_AGENT), Color.ACCENT),
        CATEGORY_USER_MESSAGES: (g(Glyph.AGENT), Color.USER),
        CATEGORY_AGENT_RESPONSES: (g(Glyph.AGENT), Color.AGENT),
        CATEGORY_REASONING: (g(Glyph.AGENT), Color.MUTED),
        CATEGORY_TOOL_USAGE: (g(Glyph.AGENT), Color.TOOL),
        CATEGORY_IMAGES: (g(Glyph.SUB_AGENT), Color.WARNING),
        CATEGORY_SERVER_OVERHEAD: (g(Glyph.PENDING), Color.MUTED),
        CATEGORY_FREE: (g(Glyph.CTX_EMPTY), f"dim {Color.MUTED}"),
        CATEGORY_AUTOCOMPACT_BUFFER: (g(Glyph.BAR_EMPTY), Color.MUTED),
    }
    return mapping.get(key, (g(Glyph.AGENT), Color.MUTED))


def render_context_grid(breakdown: ContextBreakdown, *, cols: int = 20, rows: int = _GRID_ROWS) -> Text:
    total_cells = rows * cols
    allocations = allocate_grid_cells(breakdown, total_cells=total_cells)
    cells: list[tuple[str, str]] = []
    for key, count in allocations:
        glyph, style = _category_glyph_and_style(key)
        cells.extend((glyph, style) for _ in range(count))
    if len(cells) < total_cells:
        empty_pair = _category_glyph_and_style(CATEGORY_FREE)
        cells.extend(empty_pair for _ in range(total_cells - len(cells)))

    text = Text()
    for row in range(rows):
        if row > 0:
            text.append("\n")
        start = row * cols
        for idx in range(cols):
            glyph, style = cells[start + idx]
            if idx > 0:
                text.append(" ")
            text.append(glyph, style=style)
    return text


def render_context_summary(breakdown: ContextBreakdown) -> Text:
    sep = f" {theme.g(Glyph.BULLET_SEP)} "
    budget = breakdown.model_max_input_tokens
    text = Text()

    model_label = (
        f"{breakdown.provider}/{breakdown.model}" if breakdown.provider and breakdown.model else breakdown.model
    )
    if not model_label:
        model_label = "Active Model"
    if breakdown.thinking_effort:
        model_label += f" ({breakdown.thinking_effort})"
    text.append(model_label, style="bold")
    if budget > 0:
        pct_style = Color.SUCCESS
        if breakdown.usage_percentage >= breakdown.compression_threshold * 100.0:
            pct_style = Color.ERROR
        elif breakdown.usage_percentage >= 60.0:
            pct_style = Color.WARNING
        text.append(sep, style=Color.MUTED)
        text.append(
            f"{format_token_count(breakdown.total_tokens)}/{format_token_count(budget)} tokens ",
            style="bold",
        )
        text.append(f"({breakdown.usage_percentage:.1f}%)", style=f"bold {pct_style}")
    else:
        text.append(sep, style=Color.MUTED)
        text.append(f"{format_token_count(breakdown.total_tokens)} tokens", style="bold")

    if breakdown.strict_context_budget:
        text.append(" [capped]", style=Color.WARNING)
    if breakdown.is_live_turn:
        text.append(sep, style=Color.MUTED)
        text.append("live turn", style=Color.ACCENT)

    if budget > 0:
        text.append("\n")
        compact_pct = breakdown.compression_threshold * 100.0
        if breakdown.total_tokens > breakdown.autocompact_at_tokens:
            text.append(
                f"Auto-compact threshold exceeded ({compact_pct:.0f}% / "
                f"{format_token_count(breakdown.autocompact_at_tokens)})",
                style=Color.WARNING,
            )
        else:
            text.append(
                f"Auto-compact at {compact_pct:.0f}% ({format_token_count(breakdown.autocompact_at_tokens)})",
                style="",
            )
            text.append(sep, style=Color.MUTED)
            text.append(
                f"{format_token_count(breakdown.free_before_compact_tokens)} free before compact",
                style=Color.MUTED,
            )

        if breakdown.model_context_length > budget:
            reserved = breakdown.model_context_length - budget
            text.append("\n")
            text.append(
                f"Input budget: {format_token_count(budget)} "
                f"({format_token_count(breakdown.model_context_length)} window - "
                f"{format_token_count(reserved)} output reserve)",
                style=Color.MUTED,
            )

    status_bits: list[str] = []
    if breakdown.agent_role:
        status_bits.append(f"{breakdown.agent_role} mode")
    if breakdown.compacted_messages > 0:
        status_bits.append(
            f"Compacted: {breakdown.compacted_messages} older messages "
            f"({format_token_count(breakdown.compaction_summary_tokens)} summary)"
        )
    if status_bits:
        text.append("\n")
        text.append(sep.join(status_bits), style=Color.MUTED)

    text.append("\n\nToken usage by category\n", style="bold")

    if not breakdown.categories:
        text.append("  No active tokens counted yet.\n", style=Color.MUTED)

    for category in breakdown.categories:
        glyph, style = _category_glyph_and_style(category.key)
        pct = (category.tokens / budget * 100.0) if budget > 0 else 0.0
        text.append(glyph, style=style)
        text.append(f" {category.label}: ", style="")
        text.append(f"{format_token_count(category.tokens)} tokens", style="bold")
        text.append(f" ({pct:.1f}%)", style=Color.MUTED)
        if category.detail:
            text.append(f"  {_ui_text(category.detail)}", style=Color.MUTED)
        text.append("\n")

    if budget > 0:
        free_glyph, free_style = _category_glyph_and_style(CATEGORY_FREE)
        free_before = breakdown.free_before_compact_tokens
        free_pct = free_before / budget * 100.0
        text.append(free_glyph, style=free_style)
        text.append(" Free space: ", style="")
        text.append(f"{format_token_count(free_before)} tokens", style="bold")
        text.append(f" ({free_pct:.1f}%)", style=Color.MUTED)

        if breakdown.autocompact_buffer_tokens > 0:
            buf_glyph, buf_style = _category_glyph_and_style(CATEGORY_AUTOCOMPACT_BUFFER)
            remaining_buffer = min(breakdown.free_tokens, breakdown.autocompact_buffer_tokens)
            buf_pct = remaining_buffer / budget * 100.0
            text.append("\n")
            text.append(buf_glyph, style=buf_style)
            text.append(" Auto-compact buffer: ", style="")
            text.append(f"{format_token_count(remaining_buffer)} tokens", style="bold")
            text.append(f" ({buf_pct:.1f}%)", style=Color.MUTED)
            if breakdown.total_tokens > breakdown.autocompact_at_tokens:
                text.append("  threshold exceeded", style=Color.WARNING)

    if breakdown.total_source in {"provider", "cached"} and breakdown.local_total_tokens > 0:
        delta = abs(breakdown.total_tokens - breakdown.local_total_tokens)
        if delta > max(200, int(breakdown.total_tokens * 0.05)):
            text.append("\n\n")
            text.append(
                f"Categories estimated locally (cl100k{sep}~{format_token_count(breakdown.local_total_tokens)}); "
                f"total anchored to {breakdown.total_source} count.",
                style=Color.MUTED,
            )

    return text


def render_context_details(breakdown: ContextBreakdown) -> Text:
    sep = f" {theme.g(Glyph.BULLET_SEP)} "
    elbow = theme.g(Glyph.INSET_ELBOW)
    text = Text()
    for group_idx, group in enumerate(breakdown.detail_groups):
        if group_idx > 0:
            text.append("\n\n")
        text.append(group.title, style="bold")
        if group.related_command:
            text.append(sep, style=Color.MUTED)
            text.append(group.related_command, style=Color.ACCENT)
        for item in group.items:
            text.append(f"\n  {elbow} ", style=Color.MUTED)
            text.append(item.label, style="")
            text.append(": ", style=Color.MUTED)
            text.append(f"{format_token_count(item.tokens)} tokens", style="bold")
            if item.note:
                text.append(f" ({_ui_text(item.note)})", style=Color.MUTED)
    return text


class ContextInspectorScreen(ModalScreen[None]):
    """Read-only modal visualizing active context-window usage and breakdown."""

    BINDINGS = [
        Binding("escape", "close", "Close", show=True, priority=True),
        Binding("r", "refresh", "Refresh", show=True),
        Binding("y", "copy_report", "Copy", show=True),
        Binding("c", "compress", "Compress", show=True),
        Binding("up,k", "scroll_up", "Scroll up", show=False),
        Binding("down,j", "scroll_down", "Scroll down", show=False),
        Binding("pageup", "page_up", "Page up", show=False),
        Binding("pagedown", "page_down", "Page down", show=False),
        Binding("home,ctrl+home", "scroll_top", "Top", show=False),
        Binding("end,ctrl+end", "scroll_bottom", "Bottom", show=False),
    ]

    def __init__(self, owner: "KolegaCodeApp") -> None:
        super().__init__()
        self._owner = owner
        self._breakdown: Optional[ContextBreakdown] = None
        self._work_generation = 0

    def compose(self) -> ComposeResult:
        sep = f" {theme.g(Glyph.BULLET_SEP)} "
        elbow = theme.g(Glyph.INSET_ELBOW)
        with Vertical(id="context_shell"):
            with Horizontal(id="context_header"):
                yield Static(f"{elbow} Context Usage", id="context_title", markup=False)
                yield Static(f"r refresh{sep}y copy{sep}esc close", id="context_header_hint", markup=False)
            with VerticalScroll(id="context_scroll"):
                yield Static(f"Analyzing context{theme.g(Glyph.ELLIPSIS)}", id="context_status", markup=False)
                with Horizontal(id="context_hero"):
                    yield Static("", id="context_grid")
                    yield Static("", id="context_summary")
                yield Static("", id="context_details")
            yield Static("", id="context_footer")

    def on_mount(self) -> None:
        self._owner._context_screen = self
        self._apply_responsive_layout()
        self._render_footer()
        self.query_one("#context_scroll", VerticalScroll).focus()
        self._start_refresh(force_provider_count=False)

    def on_unmount(self) -> None:
        self._work_generation += 1
        if self._owner._context_screen is self:
            self._owner._context_screen = None

    def on_resize(self, event: events.Resize) -> None:
        del event
        self._apply_responsive_layout()
        if self._breakdown is not None:
            self._render_breakdown(self._breakdown)

    def _work_current(self, generation: int) -> bool:
        return generation == self._work_generation and self.is_attached and not self._pruning

    def _grid_cols(self) -> int:
        width = self.size.width or 100
        if width >= _WIDE_GRID_BREAKPOINT:
            return 24
        if width < _NARROW_GRID_BREAKPOINT:
            return 16
        return 20

    def _apply_responsive_layout(self) -> None:
        width = self.size.width or 100
        try:
            hero = self.query_one("#context_hero", Horizontal)
        except Exception:
            return
        hero.set_class(width < _STACKED_WIDTH_BREAKPOINT, "stacked")

    def _start_refresh(self, *, force_provider_count: bool) -> None:
        self._work_generation += 1
        generation = self._work_generation
        agent = self._owner.agent
        status_widget = self.query_one("#context_status", Static)
        hero_widget = self.query_one("#context_hero", Horizontal)
        details_widget = self.query_one("#context_details", Static)

        if agent is None:
            self._breakdown = None
            status_widget.update("No active agent - configure a model in /settings first.")
            status_widget.display = True
            hero_widget.display = False
            details_widget.display = False
            self._render_footer()
            return

        if self._breakdown is None:
            status_widget.update(f"Analyzing context{theme.g(Glyph.ELLIPSIS)}")
            status_widget.display = True
            hero_widget.display = False
            details_widget.display = False

        self.run_worker(
            self._load_breakdown(generation, agent, force_provider_count=force_provider_count),
            group="context-load",
            exclusive=True,
        )

    async def _load_breakdown(
        self,
        generation: int,
        agent: object,
        *,
        force_provider_count: bool,
    ) -> None:
        is_live_turn = bool(
            self._owner._turn_active or self._owner.agent_worker is not None or self._owner._agent_execution_draining()
        )
        try:
            breakdown = await analyze_agent_context(
                agent,
                is_live_turn=is_live_turn,
                refresh_provider_count=force_provider_count and not is_live_turn,
            )
        except Exception as exc:
            if not self._work_current(generation):
                return
            status_widget = self.query_one("#context_status", Static)
            status_widget.update(f"Could not analyze context: {exc}")
            status_widget.display = True
            self.query_one("#context_hero", Horizontal).display = False
            self.query_one("#context_details", Static).display = False
            return

        if not self._work_current(generation):
            return
        self._breakdown = breakdown
        self._render_breakdown(breakdown)

    def _render_breakdown(self, breakdown: ContextBreakdown) -> None:
        self._apply_responsive_layout()
        status_widget = self.query_one("#context_status", Static)
        hero_widget = self.query_one("#context_hero", Horizontal)
        grid_widget = self.query_one("#context_grid", Static)
        summary_widget = self.query_one("#context_summary", Static)
        details_widget = self.query_one("#context_details", Static)

        status_widget.display = False
        hero_widget.display = True
        grid_widget.update(render_context_grid(breakdown, cols=self._grid_cols()))
        summary_widget.update(render_context_summary(breakdown))

        if breakdown.detail_groups:
            details_widget.update(render_context_details(breakdown))
            details_widget.display = True
        else:
            details_widget.update("")
            details_widget.display = False

        self._render_footer()

    def _render_footer(self) -> None:
        sep = f" {theme.g(Glyph.BULLET_SEP)} "
        scroll_keys = "↑/↓" if theme.supports_unicode() else "up/down"
        footer = self.query_one("#context_footer", Static)
        text = Text()
        related = sep.join(("/compress", "/mcp", "/memory", "/skills", "/rewind"))
        text.append(f"Related: {related}\n", style=Color.MUTED)
        can_compress = (
            self._owner.agent is not None
            and not self._owner._turn_active
            and self._owner.agent_worker is None
            and not self._owner._agent_execution_draining()
        )
        hints = [
            (scroll_keys, "Scroll"),
            ("pgup/pgdn", "Page"),
            ("home/end", "Top/Bottom"),
            ("r", "Refresh"),
            ("c", "Compress" if can_compress else "Compress (busy)"),
            ("y", "Copy"),
            ("esc", "Close"),
        ]
        for idx, (key_label, desc) in enumerate(hints):
            if idx > 0:
                text.append(sep, style=Color.MUTED)
            key_style = Color.MUTED if key_label == "c" and not can_compress else f"bold {Color.ACCENT}"
            text.append(key_label, style=key_style)
            text.append(f" {desc}", style=Color.MUTED)
        footer.update(text)

    def action_close(self) -> None:
        if self._owner._context_screen is self:
            self._owner._context_screen = None
        self.dismiss()
        self._owner._schedule_primary_focus_restore()

    def action_refresh(self) -> None:
        self._start_refresh(force_provider_count=True)

    def action_copy_report(self) -> None:
        if self._breakdown is None:
            self._owner._notify_user("Context breakdown is not ready yet.", severity="warning")
            return
        self._owner.copy_to_clipboard(format_context_breakdown_text(self._breakdown))
        self._owner._notify_user("Copied context usage breakdown to clipboard.")

    def action_compress(self) -> None:
        if self._owner.agent is None:
            self._owner._notify_user("Configure a model in /settings before compacting.", severity="warning")
            return
        if self._owner._turn_active or self._owner.agent_worker is not None or self._owner._agent_execution_draining():
            self._owner._notify_user("Wait for the active turn to finish before compacting.", severity="warning")
            return
        self.action_close()
        self._owner._start_compress_command()

    def action_scroll_up(self) -> None:
        self.query_one("#context_scroll", VerticalScroll).scroll_up(animate=False)

    def action_scroll_down(self) -> None:
        self.query_one("#context_scroll", VerticalScroll).scroll_down(animate=False)

    def action_page_up(self) -> None:
        self.query_one("#context_scroll", VerticalScroll).scroll_page_up(animate=False)

    def action_page_down(self) -> None:
        self.query_one("#context_scroll", VerticalScroll).scroll_page_down(animate=False)

    def action_scroll_top(self) -> None:
        self.query_one("#context_scroll", VerticalScroll).scroll_home(animate=False)

    def action_scroll_bottom(self) -> None:
        self.query_one("#context_scroll", VerticalScroll).scroll_end(animate=False)
