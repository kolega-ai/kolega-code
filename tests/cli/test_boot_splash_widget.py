from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path
import time
from typing import Literal
from unittest.mock import Mock

import pytest
from rich.cells import cell_len
from rich.color import ColorSystem
from rich.console import Console
from rich.style import Style
from rich.text import Text
from textual.app import App, ComposeResult
from textual.pilot import Pilot

from kolega_code.cli.tui.boot_splash import BootSplash, _LOGO_WIDTH


class BootSplashTestApp(App):
    def __init__(self, splash: BootSplash) -> None:
        super().__init__()
        self.splash = splash

    def compose(self) -> ComposeResult:
        yield self.splash


@pytest.fixture(autouse=True)
def _isolated_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.delenv("COLORTERM", raising=False)
    monkeypatch.chdir(tmp_path)


async def _wait_for_layout(pilot: Pilot, predicate: Callable[[], bool], *, timeout: float = 6.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        await pilot.pause(0.02)
        if predicate():
            return
    raise AssertionError(f"layout did not settle within {timeout}s")


def _lines(widget: BootSplash) -> list[str]:
    return [widget.render_line(y).text.rstrip() for y in range(widget.size.height)]


def _colors(text: Text) -> list[str | None]:
    console = Console()
    return [
        color.name if (color := text.get_style_at_offset(console, offset).color) else None
        for offset in range(len(text.plain))
    ]


def _color_at(text: Text, offset: int) -> str | None:
    if offset < 0 or offset >= len(text.plain):
        return None
    color = text.get_style_at_offset(Console(), offset).color
    return color.name if color else None


@pytest.mark.asyncio
async def test_boot_splash_full_geometry_wordmark_metadata_and_no_children() -> None:
    project = Path("project")
    splash = BootSplash(project_path=project, version="9.8.7")
    app = BootSplashTestApp(splash)

    async with app.run_test(size=(80, 24)) as pilot:
        await _wait_for_layout(
            pilot,
            lambda: (
                splash.size.width == 80 and splash.size.height == 24 and any("█" in line for line in _lines(splash))
            ),
        )

        assert splash.id == "boot_splash"
        assert list(splash.children) == []
        rendered = _lines(splash)
        assert all(cell_len(line) <= splash.size.width for line in rendered)

        block_lines = [line for line in rendered if any(char in line for char in "█▀▄")]
        assert len(block_lines) == 5
        assert max(cell_len(line.strip()) for line in block_lines) == 56
        assert all(len(line) - len(line.lstrip()) == 12 for line in block_lines)
        assert block_lines[0].lstrip().startswith("██   ▄█▀   ▄████▄")
        assert block_lines[-1].endswith("███████  ███████   ▀████▀   ██    ██")
        subtitle = next(line for line in rendered if "C  O  D  E" in line)
        assert subtitle == " " * 27 + "─────   C  O  D  E   ─────"
        assert rendered.index(subtitle) == rendered.index(block_lines[-1]) + 2
        stage_index = next(index for index, line in enumerate(rendered) if "▶ Preparing workspace" in line)
        assert stage_index == rendered.index(subtitle) + 3
        assert any("▶ Preparing workspace" in line for line in rendered)
        assert any(f"{project} · v9.8.7" in line for line in rendered)


@pytest.mark.parametrize("size", [(0, 0), (0, 4), (4, 0)])
def test_boot_splash_zero_sizes_render_empty(size: tuple[int, int], tmp_path: Path) -> None:
    splash = BootSplash(project_path=tmp_path / "非常に長いプロジェクト名🧪", version="1.0.0")

    assert splash._render_for_size(*size).plain == ""


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(1, 1), (2, 3), (8, 2), (16, 3)])
async def test_boot_splash_tiny_sizes_stage_updates_and_unicode_path_do_not_overflow(
    size: tuple[int, int], tmp_path: Path
) -> None:
    splash = BootSplash(project_path=tmp_path / "非常に長いプロジェクト名🧪", version="1.0.0")
    splash.set_stage("   Loading   Ω configuration   ")
    app = BootSplashTestApp(splash)

    async with app.run_test(size=size) as pilot:
        await _wait_for_layout(pilot, lambda: splash.size.width == size[0] and splash.size.height == size[1])
        splash.set_stage("   ")
        await pilot.pause()
        rendered = _lines(splash)

        assert len(rendered) == size[1]
        assert all(cell_len(line) <= size[0] for line in rendered)


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(24, 5), (12, 2)])
async def test_boot_splash_compact_fallback_has_no_overflow_or_block_logo(
    size: tuple[int, int], tmp_path: Path
) -> None:
    splash = BootSplash(project_path=tmp_path / "very-long-project-name", version="1.0.0")
    splash.set_stage("Loading configuration and terminal theme")
    app = BootSplashTestApp(splash)

    async with app.run_test(size=size) as pilot:
        await _wait_for_layout(pilot, lambda: splash.size.width == size[0] and splash.size.height == size[1])
        rendered = _lines(splash)

        assert all(cell_len(line) <= size[0] for line in rendered)
        assert any("KOLEGA" in line for line in rendered)
        assert not any(any(char in line for char in "█▀▄") for line in rendered)


@pytest.mark.asyncio
async def test_boot_splash_resize_across_wordmark_boundaries_keeps_lockup_centered(tmp_path: Path) -> None:
    splash = BootSplash(project_path=tmp_path / "非常に長いプロジェクト名🧪", version="1.0.0")
    app = BootSplashTestApp(splash)

    async with app.run_test(size=(80, 24)) as pilot:
        for width, height, full in [
            (56, 11, True),
            (55, 11, False),
            (80, 10, False),
            (80, 11, True),
            (1, 1, False),
            (80, 24, True),
        ]:
            await pilot.resize_terminal(width, height)
            await _wait_for_layout(
                pilot,
                lambda: (
                    splash.size.width == width
                    and splash.size.height == height
                    and any("█" in line for line in _lines(splash)) == full
                ),
            )
            rendered = _lines(splash)
            assert len(rendered) == height
            assert all(cell_len(line) <= width for line in rendered)
            if full:
                first_logo_row = next(index for index, line in enumerate(rendered) if "█" in line)
                assert first_logo_row == (height - 11) // 2
                assert any("C  O  D  E" in line for line in rendered)
                assert any("▶ Preparing workspace" in line for line in rendered)
                assert any("v1.0.0" in line or "…" in line for line in rendered)
            elif width >= 10:
                assert any("KOLEGA CODE" in line for line in rendered)
                assert not any("─" in line for line in rendered)


@pytest.mark.asyncio
@pytest.mark.parametrize("elapsed", [0.0, 0.75, 3.5, 4.5, 6.75])
async def test_boot_splash_shimmer_uses_shared_columns_for_kolega_and_code_rows(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, elapsed: float
) -> None:
    splash = BootSplash(project_path=tmp_path, version="1.0.0")
    app = BootSplashTestApp(splash)
    started_at = 200.0
    now = started_at
    monkeypatch.setattr(splash, "_now", lambda: now)

    async with app.run_test(size=(80, 24)):
        monkeypatch.setattr(app.console, "_color_system", ColorSystem.TRUECOLOR)
        monkeypatch.setattr(app.console, "no_color", False)
        app.animation_level = "full"
        splash.start_animation()
        assert splash._timer is not None
        splash._timer.pause()

        now = started_at + elapsed
        rendered_lines = list(splash._render_for_size(80, 24).split("\n"))
        block_lines = [line for line in rendered_lines if any(char in line.plain for char in "█▀▄")]
        assert len(block_lines) == 5
        assert {cell_len(line.plain) for line in block_lines} == {cell_len(block_lines[0].plain)}

        logo_start = cell_len(block_lines[0].plain) - _LOGO_WIDTH
        kolega_row = block_lines[0]
        code_row = next(line for line in rendered_lines if "C  O  D  E" in line.plain)
        muted = splash.get_component_rich_style("boot-splash--muted").color
        assert muted is not None
        assert code_row.cell_len == kolega_row.cell_len
        for column in range(_LOGO_WIDTH):
            offset = logo_start + column
            for row in block_lines[1:]:
                assert _color_at(row, offset) == _color_at(kolega_row, offset)
            if code_row.plain[offset] == "─":
                assert _color_at(code_row, offset) == muted.name
            else:
                assert _color_at(code_row, offset) == _color_at(kolega_row, offset)


@pytest.mark.asyncio
async def test_boot_splash_truecolor_full_motion_uses_one_repaint_timer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    splash = BootSplash(project_path=tmp_path, version="1.0.0")
    app = BootSplashTestApp(splash)
    now = 100.0
    monkeypatch.setattr(splash, "_now", lambda: now)

    async with app.run_test(size=(80, 24)):
        monkeypatch.setattr(app.console, "_color_system", ColorSystem.TRUECOLOR)
        monkeypatch.setattr(app.console, "no_color", False)
        app.animation_level = "full"

        schedule = Mock(wraps=splash.set_interval)
        monkeypatch.setattr(splash, "set_interval", schedule)
        splash.start_animation()
        assert splash._timer is not None
        splash._timer.pause()
        schedule.assert_called_once()
        assert schedule.call_args.args[:2] == (pytest.approx(1 / 60), splash._on_animation_frame)
        assert schedule.call_args.kwargs["name"] == "boot-splash"

        first = splash._render_for_size(80, 24)
        now = 100.5
        second = splash._render_for_size(80, 24)
        assert first.plain == second.plain
        assert _colors(first) != _colors(second)
        assert splash._elapsed() == pytest.approx(0.5)

        refresh = Mock(wraps=splash.refresh)
        monkeypatch.setattr(splash, "refresh", refresh)
        splash._on_animation_frame()
        refresh.assert_called_once_with(layout=False)

        timer = splash._timer
        stop = Mock(wraps=timer.stop)
        monkeypatch.setattr(timer, "stop", stop)
        splash.stop_animation()
        stop.assert_called_once()
        assert splash._timer is None


@pytest.mark.asyncio
async def test_boot_splash_shimmer_sweeps_at_sixteen_cells_per_second(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    splash = BootSplash(project_path=tmp_path, version="1.0.0")
    app = BootSplashTestApp(splash)
    now = 100.0
    monkeypatch.setattr(splash, "_now", lambda: now)
    styles = {
        "boot-splash--logo-base": Style(color="#000000"),
        "boot-splash--logo-accent": Style(color="#808080"),
        "boot-splash--logo-peak": Style(color="#ffffff"),
    }

    async with app.run_test(size=(80, 24)):
        monkeypatch.setattr(app.console, "_color_system", ColorSystem.TRUECOLOR)
        monkeypatch.setattr(app.console, "no_color", False)
        app.animation_level = "full"
        splash.start_animation()
        assert splash._timer is not None
        splash._timer.pause()
        monkeypatch.setattr(splash, "get_component_rich_style", styles.__getitem__)
        peaks: list[int] = []
        for elapsed in (1.0, 1.5, 2.0):
            now = 100.0 + elapsed
            row = splash._style_logo(["█" * 56])[0]
            peaks.append(_colors(row).index("#ffffff"))

        # Half a second advances eight cells; the working label still uses its
        # original eight-cells-per-second default.
        assert peaks[1] - peaks[0] == 8
        assert peaks[2] - peaks[1] == 8
        splash.stop_animation()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "palette",
    [("#e8edf5", "#2288bb", "#cc88ee"), ("#202124", "#125b92", "#7f1d80")],
    ids=["dark", "light"],
)
async def test_boot_splash_shimmer_blends_smoothly_between_frames(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, palette: tuple[str, str, str]
) -> None:
    splash = BootSplash(project_path=tmp_path, version="1.0.0")
    app = BootSplashTestApp(splash)
    now = 100.0
    monkeypatch.setattr(splash, "_now", lambda: now)
    styles = {
        "boot-splash--logo-base": Style(color=palette[0]),
        "boot-splash--logo-accent": Style(color=palette[1]),
        "boot-splash--logo-peak": Style(color=palette[2]),
    }

    async with app.run_test(size=(80, 24)):
        monkeypatch.setattr(app.console, "_color_system", ColorSystem.TRUECOLOR)
        monkeypatch.setattr(app.console, "no_color", False)
        app.animation_level = "full"
        splash.start_animation()
        assert splash._timer is not None
        splash._timer.pause()
        monkeypatch.setattr(splash, "get_component_rich_style", styles.__getitem__)
        samples: list[tuple[int, int, int]] = []
        for frame in range(480):
            now = 100.0 + frame / 60
            row = splash._style_logo(["█" * 56])[0]
            color = row.get_style_at_offset(app.console, 20).color
            assert color is not None
            rgb = color.get_truecolor()
            samples.append((rgb.red, rgb.green, rgb.blue))

        # A block letter must fade through intermediate shades, rather than hold
        # one of a few palette entries and suddenly jump to the next.
        assert len(set(samples)) >= 40
        largest_step = max(
            abs(channel - previous_channel)
            for previous, current in zip(samples, samples[1:])
            for previous_channel, channel in zip(previous, current)
        )
        assert largest_step <= 20
        splash.stop_animation()


@pytest.mark.asyncio
async def test_boot_splash_animation_frames_repaint_without_reflow(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    splash = BootSplash(project_path=tmp_path, version="1.0.0")
    app = BootSplashTestApp(splash)
    now = 100.0
    monkeypatch.setattr(splash, "_now", lambda: now)

    async with app.run_test(size=(80, 24)) as pilot:
        monkeypatch.setattr(app.console, "_color_system", ColorSystem.TRUECOLOR)
        monkeypatch.setattr(app.console, "no_color", False)
        app.animation_level = "full"
        splash.start_animation()
        assert splash._timer is not None
        splash._timer.pause()
        await _wait_for_layout(pilot, lambda: splash.size == app.screen.size and not app.screen._layout_required)
        reflow = Mock(wraps=app.screen._compositor.reflow)
        monkeypatch.setattr(app.screen._compositor, "reflow", reflow)
        first = _lines(splash)

        for frame in range(1, 9):
            now = 100.0 + frame / 60
            splash._on_animation_frame()
            painted = asyncio.Event()
            app.call_after_refresh(painted.set)
            await asyncio.wait_for(painted.wait(), timeout=6)

        assert _lines(splash) == first
        reflow.assert_not_called()
        splash.stop_animation()


@pytest.mark.asyncio
async def test_boot_splash_repeated_start_stop_and_preference_change_clean_up_timer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    splash = BootSplash(project_path=tmp_path, version="1.0.0")
    app = BootSplashTestApp(splash)

    async with app.run_test(size=(80, 24)):
        monkeypatch.setattr(app.console, "_color_system", ColorSystem.TRUECOLOR)
        monkeypatch.setattr(app.console, "no_color", False)
        app.animation_level = "full"

        splash.start_animation()
        first_timer = splash._timer
        assert first_timer is not None
        first_timer.pause()
        first_stop = Mock(wraps=first_timer.stop)
        monkeypatch.setattr(first_timer, "stop", first_stop)

        splash.start_animation()
        first_stop.assert_called_once()
        assert splash._timer is not None
        assert splash._timer is not first_timer
        splash._timer.pause()

        app.animation_level = "basic"
        splash._on_animation_frame()
        assert splash._timer is None
        assert splash._started_at is None

        splash.stop_animation()
        assert splash._timer is None


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
async def test_boot_splash_static_fallbacks_do_not_start_timer_or_change_frames(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    color_system: ColorSystem | None,
    no_color: bool,
    animation_level: Literal["full", "basic", "none"],
) -> None:
    monkeypatch.setenv("COLORTERM", "truecolor")
    splash = BootSplash(project_path=tmp_path, version="1.0.0")
    app = BootSplashTestApp(splash)
    now = 100.0
    monkeypatch.setattr(splash, "_now", lambda: now)

    async with app.run_test(size=(80, 24)):
        monkeypatch.setattr(app.console, "_color_system", color_system)
        monkeypatch.setattr(app.console, "no_color", no_color)
        app.animation_level = animation_level

        splash.start_animation()
        assert splash._timer is None
        first = splash._render_for_size(80, 24)
        now = 101.0
        second = splash._render_for_size(80, 24)
        assert first == second


@pytest.mark.asyncio
async def test_boot_splash_palette_tracks_textual_theme(tmp_path: Path) -> None:
    splash = BootSplash(project_path=tmp_path, version="1.0.0")
    app = BootSplashTestApp(splash)

    async with app.run_test(size=(80, 24)) as pilot:
        dark = splash._render_for_size(80, 24)
        app.theme = "textual-light"
        await _wait_for_layout(pilot, lambda: _colors(splash._render_for_size(80, 24)) != _colors(dark))
        light = splash._render_for_size(80, 24)
        assert light.plain == dark.plain


@pytest.mark.asyncio
async def test_boot_splash_unmount_cleans_timer(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    splash = BootSplash(project_path=tmp_path, version="1.0.0")
    app = BootSplashTestApp(splash)

    async with app.run_test(size=(80, 24)):
        monkeypatch.setattr(app.console, "_color_system", ColorSystem.TRUECOLOR)
        monkeypatch.setattr(app.console, "no_color", False)
        app.animation_level = "full"
        splash.start_animation()
        assert splash._timer is not None
        splash._timer.pause()

    assert splash._timer is None
