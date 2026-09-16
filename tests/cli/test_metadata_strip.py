"""Standalone metadata rendering, without constructing the agent application."""

from __future__ import annotations

import subprocess
import time
from collections.abc import Callable
from pathlib import Path

import pytest
from rich.cells import cell_len, split_graphemes
from rich.console import Console
from rich.style import Style
from rich.text import Text
from textual.app import App, ComposeResult
from textual.pilot import Pilot
from textual.widgets import Static

from kolega_code.cli.theme import Color, active_theme, apply_theme, available_themes
from kolega_code.cli.tui.metadata import MetadataStrip

pytestmark = pytest.mark.usefixtures("isolated_cli_env")


def make_strip(
    *,
    project: Path | None = None,
    branch: str = "feat/metadata",
    session: str = "7fa91c02-full-session",
    mode: str = "build",
    permission: str = "auto",
    gigacode: bool = False,
) -> MetadataStrip:
    strip = MetadataStrip(id="session_meta")
    strip.update_context(
        project_path=project or Path.home() / "git" / "kolega-code",
        branch=branch,
        session_id=session,
        interaction_mode=mode,
        permission_mode=permission,
        gigacode_enabled=gigacode,
    )
    return strip


class MetadataApp(App[None]):
    ENABLE_COMMAND_PALETTE = False

    def __init__(self, strip: MetadataStrip) -> None:
        super().__init__()
        self.strip = strip

    def compose(self) -> ComposeResult:
        yield self.strip


async def wait_for_layout(pilot: Pilot, predicate: Callable[[], bool]) -> None:
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        await pilot.pause(0.02)
        if predicate():
            return
    assert predicate()


def test_display_api_and_full_context() -> None:
    strip = make_strip()
    text = strip.content_for_width()
    assert isinstance(text, Text)
    assert str(text) == "~/git/kolega-code · feat/metadata · build · permissions auto · 7fa91c02"
    assert str(Path.home() / "git/kolega-code") in strip.full_context
    assert "Branch: feat/metadata" in strip.full_context
    assert "Session: 7fa91c02-full-session" in strip.full_context
    assert isinstance(strip.tooltip, Text)
    assert strip.tooltip.plain == strip.full_context
    assert "gigacode" not in text.plain


@pytest.mark.parametrize("mode", ["build", "plan"])
@pytest.mark.parametrize("permission", ["auto", "ask"])
@pytest.mark.parametrize("width", [40, 41, 60, 80, 120, 160])
def test_ordinary_widths_protect_mode_and_labeled_permissions(width: int, mode: str, permission: str) -> None:
    strip = make_strip(
        project=Path("/very/long/leading/directories/kolega-code"),
        branch="feat/" + "long-branch-" * 20,
        mode=mode,
        permission=permission,
    )
    text = strip.content_for_width(width)
    assert text.cell_len <= width
    assert mode in text.plain
    assert f"permissions {permission}" in text.plain
    assert "kolega-code" in text.plain
    assert "\n" not in text.plain


def test_session_then_branch_are_omitted_before_project_or_risk() -> None:
    strip = make_strip(project=Path("/a/b/repo"), branch="feature")
    full = strip.content_for_width()
    no_session = strip.content_for_width(full.cell_len - 4).plain
    assert "7fa91c02" not in no_session
    assert "feature" in no_session
    narrow = strip.content_for_width(40).plain
    assert "feature" not in narrow
    assert "repo" in narrow
    assert "build · permissions auto" in narrow
    assert "feature" in strip.full_context


@pytest.mark.parametrize("width", list(range(-2, 41)))
def test_tiny_widths_are_bounded_and_keep_permission_when_possible(width: int) -> None:
    text = make_strip().content_for_width(width)
    assert text.cell_len <= max(0, width)
    assert "\n" not in text.plain
    if width <= 0:
        assert text.plain == ""
    elif width >= 4:
        assert "auto" in text.plain
    if width >= 10:
        assert "build" in text.plain


@pytest.mark.parametrize("basename", ["資料庫", "👩🏽‍💻" * 20, "e\u0301" * 40, "家庭👨‍👩‍👧‍👦" * 12])
def test_unicode_cells_and_graphemes_survive_fitting(basename: str) -> None:
    strip = make_strip(project=Path("/長い/先頭") / basename, branch="枝" * 40, session="👩🏽‍💻" * 12)
    spans, _ = split_graphemes(basename)
    valid_prefixes = {basename[:end] for _, end, _ in spans} | {""}
    for width in range(1, 180):
        text = strip.content_for_width(width)
        assert cell_len(text.plain) <= width
        assert "\n" not in text.plain
        if width >= 40:
            assert "build · permissions auto" in text.plain
            project = text.plain.split(" · ")[0]
            if project.endswith("…"):
                assert project[:-1] in valid_prefixes
    assert basename in strip.full_context


def test_fields_and_tooltip_are_literal_not_markup_or_links() -> None:
    strip = make_strip(
        project=Path("/tmp/[bold]project[/bold]"),
        branch="[link=https://example.invalid]branch[/link]",
        session="[red]id-session",
    )
    text = strip.content_for_width()
    assert "[bold]project[/bold]" in text.plain
    assert "[link=https://example.invalid]branch[/link]" in text.plain
    assert "[red]id-" in text.plain
    assert isinstance(strip.tooltip, Text)
    console = Console()
    for content in (text, strip.tooltip):
        assert all(not segment.style or not segment.style.link for segment in console.render(content))
    assert "[link=https://example.invalid]" in strip.full_context


def test_control_characters_cannot_create_extra_lines() -> None:
    strip = make_strip(project=Path("/tmp/a\nb"), branch="feat\tbad\r\nbranch\u2028next\x1b")
    text = strip.content_for_width()
    assert not any(char in text.plain for char in "\n\r\t\u2028\x1b")
    assert "feat bad  branch next " in text.plain
    assert "/tmp/a\nb" in strip.full_context


def test_gigacode_only_appears_when_enabled_and_fitting() -> None:
    strip = make_strip(gigacode=True)
    text = strip.content_for_width()
    assert text.plain.endswith(" · gigacode")
    assert "gigacode" not in strip.content_for_width(text.cell_len - 1).plain
    assert "gigacode" not in make_strip().content_for_width().plain


def test_missing_optional_fields_and_non_home_paths() -> None:
    project = Path(str(Path.home()) + "-other") / "repo"
    strip = make_strip(project=project, branch="", session="")
    assert strip.content_for_width().plain == f"{project} · build · permissions auto"
    assert "Branch: (none)" in strip.full_context
    assert "Session: (none)" in strip.full_context
    assert " ·  · " not in strip.content_for_width(40).plain
    assert make_strip(project=Path.home()).content_for_width().plain.startswith("~ · ")


def test_context_rendering_does_not_resolve_or_inspect_project(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("metadata must use supplied context, not inspect the project")

    monkeypatch.setattr(Path, "resolve", forbidden)
    monkeypatch.setattr(Path, "exists", forbidden)
    monkeypatch.setattr(Path, "is_dir", forbidden)
    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    strip = make_strip(project=Path("/not-a-real-project"), branch="cached-branch")
    assert "cached-branch" in strip.content_for_width().plain
    assert "cached-branch" in strip.full_context


def test_roles_are_looked_up_from_live_theme() -> None:
    strip = make_strip()
    original_theme = active_theme().name
    try:
        for name in available_themes():
            apply_theme(name)
            text = strip.content_for_width()
            console = Console()
            assert text.get_style_at_offset(console, text.plain.index("build")).color == Style.parse(Color.ACCENT).color
            assert text.get_style_at_offset(console, text.plain.index("auto")).color == Style.parse(Color.WARNING).color
            assert text.get_style_at_offset(console, text.plain.index("feat/")).color == Style.parse(Color.MUTED).color
            ask = make_strip(permission="ask").content_for_width()
            assert ask.get_style_at_offset(console, ask.plain.index("ask")).color == Style.parse(Color.SUCCESS).color
    finally:
        apply_theme(original_theme)


@pytest.mark.asyncio
async def test_mounted_resize_and_context_updates_without_subprocess(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("metadata must not discover repository state")

    strip = make_strip()
    app = MetadataApp(strip)
    async with app.run_test(size=(120, 8)) as pilot:
        assert app.query_one("#session_meta", Static) is strip
        # Install after Textual's own initialization; exercise only widget work.
        monkeypatch.setattr(subprocess, "run", forbidden)
        monkeypatch.setattr(subprocess, "Popen", forbidden)
        for width in (80, 40, 12, 4, 1, 120):
            await pilot.resize_terminal(width, 8)
            await wait_for_layout(pilot, lambda: strip.content_size.width == width and strip.size.height == 1)
            assert strip.render().plain == strip.content_for_width(width).plain
            assert strip.render().cell_len <= width
            assert strip.render_line(0).cell_length <= width
            assert strip.render_line(0).text.rstrip() == strip.render().plain
            assert not any(segment.style and segment.style.link for segment in strip.render_line(0))
        strip.update_context(
            project_path=Path("/new/project"),
            branch="new-branch",
            session_id="12345678-new-session",
            interaction_mode="plan",
            permission_mode="ask",
            gigacode_enabled=True,
        )
        await wait_for_layout(pilot, lambda: "new-branch" in strip.render_line(0).text)
        assert "plan · permissions ask" in strip.render().plain
        assert "12345678" in strip.render().plain
        assert "gigacode" in strip.render().plain
        assert "auto" not in strip.render().plain
        assert "Session: 12345678-new-session" in strip.full_context
        assert isinstance(strip.tooltip, Text)
        assert strip.tooltip.plain == strip.full_context


@pytest.mark.asyncio
async def test_unicode_resize_uses_content_width_inside_padding() -> None:
    strip = make_strip(
        project=Path("/長い/先頭") / ("資料👩🏽‍💻e\u0301" * 20),
        branch="枝👨‍👩‍👧‍👦" * 30,
        session="👩🏽‍💻" * 12,
    )
    strip.styles.padding = (0, 2)
    app = MetadataApp(strip)
    async with app.run_test(size=(160, 8)) as pilot:
        for width in (160, 80, 44, 10, 4, 3, 80):
            await pilot.resize_terminal(width, 8)
            content_width = max(0, width - 4)
            # Textual cannot shrink the outer box below its four padding cells.
            await wait_for_layout(
                pilot, lambda: strip.region.width == max(width, 4) and strip.content_size.width == content_width
            )
            assert strip.size.height == 1
            assert strip.render().cell_len <= content_width
            assert strip.render_line(0).cell_length <= content_width
            if content_width >= 40:
                assert "build · permissions auto" in strip.render_line(0).text
            if content_width == 0:
                assert strip.render().plain == ""
