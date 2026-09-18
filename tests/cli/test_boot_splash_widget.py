from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
import time
from typing import Literal
from unittest.mock import Mock

import pytest
from rich.cells import cell_len
from rich.color import ColorSystem
from rich.console import Console
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

        block_lines = [line.strip() for line in rendered if any(char in line for char in "█▀▄")]
        assert len(block_lines) == 7
        assert all(cell_len(line) == 35 for line in block_lines[:4])
        assert all(cell_len(line) <= 15 for line in block_lines[4:])
        assert all(cell_len(line) < cell_len(block_lines[0]) for line in block_lines[4:])
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
@pytest.mark.parametrize("elapsed", [0.0, 0.75, 4.5])
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
        assert len(block_lines) == 7
        assert {cell_len(line.plain) for line in block_lines} == {cell_len(block_lines[0].plain)}

        logo_start = cell_len(block_lines[0].plain) - _LOGO_WIDTH
        kolega_row = block_lines[0]
        code_row = block_lines[-1]
        for column in range(_LOGO_WIDTH):
            offset = logo_start + column
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
        assert schedule.call_args.args[:2] == (0.08, splash._on_animation_frame)
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
