"""The working label moves in color, never in position or meaning."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path
import time
from unittest.mock import Mock

import pytest
from rich.color import ColorSystem
from rich.console import Console
from rich.style import Style
from rich.text import Text
from textual.pilot import Pilot

from kolega_code.cli.tui.state import TurnState
from kolega_code.cli.tui.turn_status import shimmer_text

from ._app_test_utils import _build_sub_agent_test_app


async def _wait(pilot: Pilot, predicate: Callable[[], bool]) -> None:
    deadline = time.monotonic() + 6
    while time.monotonic() < deadline:
        await pilot.pause(0.02)
        if predicate():
            return
    assert predicate(), "Status-strip layout did not settle."


def _colors(text: Text) -> list[str | None]:
    console = Console()
    return [
        color.name if (color := text.get_style_at_offset(console, offset).color) else None
        for offset in range(len(text))
    ]


def test_shimmer_moves_by_cells_and_time_not_label_length_or_frame_count() -> None:
    palette = tuple(Style(color=f"#{level:02x}{level:02x}{level:02x}") for level in range(17))
    short = shimmer_text(Text("x" * 16), 1.0, palette)
    long = shimmer_text(Text("x" * 40), 1.0, palette)
    later = shimmer_text(Text("x" * 40), 1.5, palette)
    assert _colors(short) == _colors(long)[:16]
    before_peak = max(range(len(long)), key=lambda x: _colors(long)[x] or "")
    after_peak = max(range(len(later)), key=lambda x: _colors(later)[x] or "")
    # Eight cells per second, even if frames were skipped between these samples.
    assert after_peak - before_peak == 4
    assert _colors(shimmer_text(Text("x" * 40), 1.5, palette)) == _colors(later)


def test_shimmer_preserves_unicode_markup_inline_styles_and_source() -> None:
    palette = tuple(Style(color=color) for color in ("#444444", "#888888", "#ffffff"))
    label = Text("[bold]測試 e\u0301[/bold] \\", style="italic")
    original = label.copy()
    frame = shimmer_text(label, 1.5, palette)
    assert label == original
    assert frame.plain == label.plain
    assert frame.cell_len == label.cell_len
    assert Text.from_markup(frame.markup).plain == label.plain
    console = Console()
    assert all(frame.get_style_at_offset(console, offset).italic for offset in range(len(frame)))
    combining_mark = label.plain.index("\u0301")
    assert frame.get_style_at_offset(console, combining_mark) == frame.get_style_at_offset(console, combining_mark - 1)
    assert shimmer_text(Text(), 1, palette).plain == ""
    assert shimmer_text(label, 1, ()) == label
    assert shimmer_text(label, -1, palette) == shimmer_text(label, 0, palette)


@pytest.mark.asyncio
async def test_working_label_shimmers_without_changing_its_text_or_hint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = _build_sub_agent_test_app(tmp_path, monkeypatch)
    now = 100.0
    monkeypatch.setattr(app, "_now", lambda: now)
    label = "Working on status rendering"
    async with app.run_test(size=(100, 30)):
        monkeypatch.setattr(app.console, "_color_system", ColorSystem.TRUECOLOR)
        monkeypatch.setattr(app.console, "no_color", False)
        app.animation_level = "full"
        app._begin_turn_progress(label)
        assert app._turn_timer is not None
        app._turn_timer.pause()
        first = Text.from_markup(app._turn_status_content(width=80))
        now += 0.4
        second = Text.from_markup(app._turn_status_content(width=80))
        assert first.plain == second.plain
        assert second.plain.endswith("0s · Esc to interrupt")
        start = first.plain.index(label)
        first_colors = [first.get_style_at_offset(app.console, x).color for x in range(start, start + len(label))]
        second_colors = [second.get_style_at_offset(app.console, x).color for x in range(start, start + len(label))]
        assert first_colors != second_colors
        assert len(set(second_colors)) > 1
        hint_start = first.plain.index("0s")
        assert all(
            first.get_style_at_offset(app.console, x) == second.get_style_at_offset(app.console, x)
            for x in range(hint_start, len(first))
        )
        app._clear_turn_status_strip()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("color_system", "no_color", "animation_level"),
    [
        (ColorSystem.EIGHT_BIT, False, "full"),
        (ColorSystem.STANDARD, False, "full"),
        (None, False, "full"),
        (ColorSystem.TRUECOLOR, True, "full"),
        (ColorSystem.TRUECOLOR, False, "basic"),
        (ColorSystem.TRUECOLOR, False, "none"),
    ],
)
async def test_shimmer_falls_back_to_the_original_flat_label(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    color_system: ColorSystem | None,
    no_color: bool,
    animation_level: str,
) -> None:
    # An ambient truecolor hint must not override the console's actual mode.
    monkeypatch.setenv("COLORTERM", "truecolor")
    app = _build_sub_agent_test_app(tmp_path, monkeypatch)
    async with app.run_test():
        monkeypatch.setattr(app.console, "_color_system", color_system)
        monkeypatch.setattr(app.console, "no_color", no_color)
        monkeypatch.setattr(app, "animation_level", animation_level)
        label = Text("Working [safely]", style="italic")
        assert app._turn_status.shimmer(label, 0.5) == label
        assert app._turn_status.shimmer(label, 1.5) == label


@pytest.mark.asyncio
@pytest.mark.parametrize("width", [32, 40, 80])
@pytest.mark.parametrize("theme_name", ["kolega-dark", "textual-light"])
async def test_shimmer_keeps_narrow_unicode_status_on_one_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, width: int, theme_name: str
) -> None:
    app = _build_sub_agent_test_app(tmp_path, monkeypatch)
    now = 100.0
    monkeypatch.setattr(app, "_now", lambda: now)
    async with app.run_test(size=(width, 30)) as pilot:
        app._set_sidebar_visible(width >= 80)
        app.theme = theme_name
        app.animation_level = "full"
        monkeypatch.setattr(app.console, "_color_system", ColorSystem.TRUECOLOR)
        monkeypatch.setattr(app.console, "no_color", False)
        app._begin_turn_progress("Indexing [workspace] 測試 e\u0301\n" * 10)
        assert app._turn_timer is not None
        app._turn_timer.pause()
        strip = app._turn_status
        await _wait(pilot, lambda: strip.content_size.width > 0 and strip.content_size.height == 1)
        for elapsed in (0.4, 0.8, 1.2):
            now = 100.0 + elapsed
            content = Text.from_markup(app._turn_status_content(width=strip.content_size.width))
            assert content.cell_len <= strip.content_size.width
            assert "\n" not in content.plain
            assert content.plain.endswith("Esc to interrupt")
            assert not any(span.style and isinstance(span.style, Style) and span.style.link for span in content.spans)
        assert len(strip.children) == 0
        app._clear_turn_status_strip()


@pytest.mark.asyncio
async def test_shimmer_uses_current_theme_and_reduced_motion_setting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = _build_sub_agent_test_app(tmp_path, monkeypatch)
    async with app.run_test() as pilot:
        app.animation_level = "full"
        monkeypatch.setattr(app.console, "_color_system", ColorSystem.TRUECOLOR)
        monkeypatch.setattr(app.console, "no_color", False)
        strip = app._turn_status
        label = Text("Working on the theme")
        dark = strip.shimmer(label, 1)
        assert dark.spans
        app.theme = "textual-light"
        await _wait(pilot, lambda: _colors(strip.shimmer(label, 1)) != _colors(dark))
        light = strip.shimmer(label, 1)
        assert light.plain == dark.plain
        app.animation_level = "none"
        assert strip.shimmer(label, 1) == label
        app.animation_level = "full"
        assert _colors(strip.shimmer(label, 1)) == _colors(light)


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", [TurnState.IDLE, TurnState.STOPPED, TurnState.ERROR])
async def test_shimmer_stops_on_completion_and_restarts_with_the_next_turn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, outcome: TurnState
) -> None:
    app = _build_sub_agent_test_app(tmp_path, monkeypatch)
    now = 100.0
    monkeypatch.setattr(app, "_now", lambda: now)
    async with app.run_test():
        app.animation_level = "full"
        monkeypatch.setattr(app.console, "_color_system", ColorSystem.TRUECOLOR)
        monkeypatch.setattr(app.console, "no_color", False)
        app._begin_turn_progress()
        assert app._turn_timer is not None
        old_timer = app._turn_timer
        old_timer.pause()
        shimmer = Mock(wraps=app._turn_status.shimmer)
        monkeypatch.setattr(app._turn_status, "shimmer", shimmer)
        now += 0.5
        app._refresh_turn_status_strip()
        assert shimmer.call_count == 1
        app._finish_turn_progress("Finished.", outcome)
        assert app._turn_timer is None
        final = app._turn_status_content()
        now += 5
        assert app._turn_status_content() == final
        assert shimmer.call_count == 1
        assert "Esc to interrupt" not in Text.from_markup(final).plain
        app._begin_turn_progress("Next turn")
        assert app._turn_timer is not None and app._turn_timer is not old_timer
        app._turn_timer.pause()
        assert shimmer.call_args.args[1] == 0
        app._clear_turn_status_strip()
        assert app._turn_timer is None
        assert app._turn_status_content() == ""


@pytest.mark.asyncio
async def test_shimmer_frames_do_not_reflow_the_screen_or_speed_up_other_views(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = _build_sub_agent_test_app(tmp_path, monkeypatch)
    now = 100.0
    monkeypatch.setattr(app, "_now", lambda: now)
    async with app.run_test(size=(100, 30)) as pilot:
        app.animation_level = "full"
        monkeypatch.setattr(app.console, "_color_system", ColorSystem.TRUECOLOR)
        monkeypatch.setattr(app.console, "no_color", False)
        schedule = Mock(wraps=app.set_interval)
        monkeypatch.setattr(app, "set_interval", schedule)
        app._begin_turn_progress("Checking repaint cost")
        assert app._turn_timer is not None
        app._turn_timer.pause()
        await _wait(pilot, lambda: app._turn_status.content_size.height == 1)
        # The initial show and queued startup changes legitimately need layout.
        # Wait for their paint before measuring subsequent animation frames.
        painted = asyncio.Event()
        app.call_after_refresh(painted.set)
        await asyncio.wait_for(painted.wait(), timeout=6)
        await _wait(pilot, lambda: not app.screen._layout_required and not app._conversation_anchor_pending)
        schedule.assert_called_once_with(0.08, app._refresh_turn_status_strip, name="turn-status")
        reflow = Mock(wraps=app.screen._compositor.reflow)
        monkeypatch.setattr(app.screen._compositor, "reflow", reflow)
        agents_tick = Mock()
        workflows_tick = Mock()
        monkeypatch.setattr(app, "_tick_running_sub_agents", agents_tick)
        monkeypatch.setattr(app, "_tick_running_workflows", workflows_tick)
        for tick in range(1, 27):
            now = 100.0 + tick * 0.08
            app._refresh_turn_status_strip()
            await pilot.pause(0.02)
        assert agents_tick.call_count == 2
        assert workflows_tick.call_count == 2
        reflow.assert_not_called()
        app._clear_turn_status_strip()
