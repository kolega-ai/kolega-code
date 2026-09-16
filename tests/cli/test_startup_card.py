"""Startup disclosures preserve diagnostics and survive transcript lifecycle changes."""

from __future__ import annotations

import time
from pathlib import Path

import pytest
from rich.console import Console
from rich.text import Text
from textual.selection import Selection
from textual.widgets import Collapsible
from textual.widgets._collapsible import CollapsibleTitle

from kolega_code.cli.tui.startup import StartupEntryWidget, StartupText, display_project_path
from kolega_code.cli.tui.state import ConversationEntry
from kolega_code.cli.tui.widgets import ChatComposer
from kolega_code.llm.models import Message, TextBlock

from ._app_test_utils import _build_mention_test_app


async def _wait_for_layout(pilot, predicate, *, timeout: float = 6.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        await pilot.pause(0.02)
        if predicate():
            return
    raise AssertionError("startup layout did not settle")


def test_home_abbreviation_is_unambiguous(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: Path("/Users/person")))
    assert display_project_path(Path("/Users/person")) == "~"
    assert display_project_path(Path("/Users/person/project")) == "~/project"
    assert display_project_path(Path("/Users/person-other/project")) == "/Users/person-other/project"


@pytest.mark.parametrize(
    ("credential", "compact"),
    [
        ("present via ANTHROPIC_API_KEY", "key present"),
        ("present in local settings", "key present"),
        ("present via environment override", "key present"),
        ("signed in as person@example.com (subscription)", "signed in"),
        ("not signed in", "not signed in"),
        ("missing", "key missing"),
        ("not required for the local provider", "no key needed"),
        ("key not set (optional)", "key not set (optional)"),
        ("endpoint not defined", "endpoint not defined"),
    ],
)
def test_startup_compacts_credentials_without_losing_configuration_details(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, credential: str, compact: str
) -> None:
    from kolega_code.cli import app as app_module

    app = _build_mention_test_app(tmp_path, monkeypatch)
    monkeypatch.setattr(app_module, "key_status", lambda *args: credential)
    entry = ConversationEntry(kind="startup", content=app._startup_content())
    output = Console(width=160, record=True)
    output.print(app._startup_summary(entry, 160))
    lines = output.export_text().splitlines()
    assert len(lines) == 2
    assert lines[1] == f"build · ask permissions · {compact}"
    assert f"API key: {credential}" in entry.content


@pytest.mark.parametrize("setup_needed", [False, True])
def test_startup_summarizes_lsp_and_keeps_server_details_in_configuration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, setup_needed: bool
) -> None:
    app = _build_mention_test_app(tmp_path, monkeypatch)
    server_lines = ["  Python → pyright", "  TypeScript → typescript-language-server"]
    if setup_needed:
        server_lines.append("  Rust → rust-analyzer (install: rustup component add rust-analyzer)")
    monkeypatch.setattr(app, "_startup_lsp_lines", lambda: ["", "LSP:", *server_lines, ""])
    entry = ConversationEntry(kind="startup", content=app._startup_content())
    output = Console(width=160, record=True)
    output.print(app._startup_summary(entry, 160))
    text = output.export_text()
    assert len(text.splitlines()) == 2
    assert f"LSP: {'setup needed' if setup_needed else 'ready'}" in text
    assert "pyright" not in text
    assert all(line in entry.content for line in server_lines)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("width", "sidebar"),
    [(40, False), (60, False), (80, False), (120, False), (120, True), (160, True)],
)
async def test_startup_configuration_follows_summary_without_blank_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, width: int, sidebar: bool
) -> None:
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path.resolve()))
    app = _build_mention_test_app(tmp_path, monkeypatch)
    async with app.run_test(size=(width, 50)) as pilot:
        app._set_sidebar_visible(sidebar)
        await _wait_for_layout(pilot, lambda: bool(app.query(StartupEntryWidget)))
        card = app.query_one(StartupEntryWidget)
        summary = card.query_one(".startup-summary", StartupText)
        configuration = card.query_one(".startup-configuration", Collapsible)
        title = configuration.query_one(CollapsibleTitle)

        def disclosure_follows_content() -> bool:
            lines = [summary.render_line(y).text for y in range(summary.content_size.height)]
            occupied_rows = [y for y, line in enumerate(lines) if line.strip()]
            return bool(occupied_rows) and title.region.y == summary.content_region.y + occupied_rows[-1] + 1

        await _wait_for_layout(pilot, disclosure_follows_content)
        configuration.collapsed = False
        await _wait_for_layout(pilot, lambda: card.entry.startup_details_expanded and disclosure_follows_content())
        await pilot.resize_terminal(width + 8, 50)
        await _wait_for_layout(pilot, disclosure_follows_content)
        configuration.collapsed = True
        await _wait_for_layout(pilot, lambda: not card.entry.startup_details_expanded and disclosure_follows_content())


@pytest.mark.asyncio
@pytest.mark.parametrize("width", [40, 60, 80, 120, 160])
async def test_startup_is_compact_selectable_and_width_aware(tmp_path, monkeypatch, width) -> None:
    # Exercise a normal home-relative project, not the runner's arbitrarily
    # deep --basetemp path (which intentionally wraps without losing content).
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path.resolve()))
    app = _build_mention_test_app(tmp_path, monkeypatch)
    async with app.run_test(size=(width, 50)) as pilot:
        app._set_sidebar_visible(False)
        await _wait_for_layout(pilot, lambda: bool(app.query(StartupEntryWidget)))
        card = app.query_one(StartupEntryWidget)
        outer = card.query_one(Collapsible)
        assert not outer.collapsed
        details = card.query_one(".startup-configuration", Collapsible)
        assert details.collapsed
        await _wait_for_layout(pilot, lambda: 0 < card.region.width <= width and card.region.height > 1)
        await _wait_for_layout(pilot, lambda: card.region.y == app._conversation.region.y)
        # Header, two summary rows, disclosure, and two border rows. Only
        # narrow terminals may add wrapped rows; no decorative blank rows.
        await _wait_for_layout(pilot, lambda: card.region.height <= (8 if width == 40 else 6))
        title = outer.query_one(CollapsibleTitle)
        assert title.size.height == 1
        assert "~/project" in Text.from_markup(app._startup_title(card.entry)).plain
        summary = card.query_one(".startup-summary", StartupText)
        selected = summary.get_selection(Selection(None, None))
        assert selected is not None
        assert app.config is not None
        assert app.config.long_context_config.model in selected[0]
        assert "effort" in selected[0]
        assert "key" in selected[0]
        assert "MODEL" not in selected[0]
        assert "WORKSPACE" not in selected[0]
        assert "ANTHROPIC_API_KEY" not in selected[0]
        assert "build" in selected[0]
        assert "ask permissions" in selected[0]
        assert any(segment.style and "offset" in segment.style.meta for segment in summary.render_line(0))
        assert not any(segment.style and segment.style.link for segment in summary.render_line(0))
        details.collapsed = False
        await _wait_for_layout(pilot, lambda: card.entry.startup_details_expanded)
        config = card.query_one(".startup-details", StartupText)
        await _wait_for_layout(pilot, lambda: config.region.height > 1)
        copied = config.get_selection(Selection(None, None))
        assert copied is not None and str(app.project_path) in copied[0].replace("\n", "")
        assert "Project:" in copied[0]
        assert "Session:" in copied[0] and "API key:" in copied[0]


@pytest.mark.asyncio
async def test_first_submission_folds_once_and_manual_reopening_survives_rebuild(tmp_path, monkeypatch) -> None:
    app = _build_mention_test_app(tmp_path, monkeypatch)
    async with app.run_test(size=(120, 50)) as pilot:
        app._set_sidebar_visible(False)
        card = app.query_one(StartupEntryWidget)
        entry = card.entry
        composer = app.query_one(ChatComposer)
        composer.load_text("inspect the project")
        composer.focus()
        await pilot.press("enter")
        await _wait_for_layout(pilot, lambda: entry.startup_auto_folded and app.agent_worker is None)
        await _wait_for_layout(pilot, lambda: card.query_one(Collapsible).collapsed)
        assert entry.startup_collapsed
        title = card.query_one(CollapsibleTitle)
        title.focus()
        await pilot.press("enter")
        await _wait_for_layout(pilot, lambda: not entry.startup_collapsed)
        app._add_conversation_entry(ConversationEntry(kind="user", content="second message"))
        app._ensure_startup_entry()
        app._render_conversation()
        await _wait_for_layout(pilot, lambda: app.query_one(StartupEntryWidget) is not card)
        assert not app.query_one(StartupEntryWidget).query_one(Collapsible).collapsed
        assert app.query_one(StartupEntryWidget).entry is entry


@pytest.mark.asyncio
async def test_restore_folds_and_reset_opens_a_fresh_card(tmp_path, monkeypatch) -> None:
    app = _build_mention_test_app(tmp_path, monkeypatch)
    history = [Message(role="user", content=[TextBlock(text="restored prompt")]).to_dict()]
    async with app.run_test(size=(100, 40)) as pilot:
        app._restore_conversation_history(history)
        await _wait_for_layout(pilot, lambda: app.query_one(StartupEntryWidget).entry.startup_collapsed)
        entry = app.query_one(StartupEntryWidget).entry
        app.query_one(StartupEntryWidget).query_one(Collapsible).collapsed = False
        await _wait_for_layout(pilot, lambda: not entry.startup_collapsed)
        app._restore_conversation_history(history)
        await _wait_for_layout(pilot, lambda: app.query_one(StartupEntryWidget).entry is entry)
        assert not entry.startup_collapsed
        await app._reset_current_thread()
        await _wait_for_layout(pilot, lambda: app.query_one(StartupEntryWidget).entry is not entry)
        fresh = app.query_one(StartupEntryWidget).entry
        assert not fresh.startup_auto_folded and not fresh.startup_collapsed
        assert [e.kind for e in app.conversation_entries].count("startup") == 1


@pytest.mark.asyncio
async def test_configuration_is_literal_and_same_length_edits_refresh(tmp_path, monkeypatch) -> None:
    app = _build_mention_test_app(tmp_path, monkeypatch)
    async with app.run_test(size=(120, 50)) as pilot:
        card = app.query_one(StartupEntryWidget)
        card.query_one(".startup-configuration", Collapsible).collapsed = False
        card.entry.content = "Kolega Code v0\n\nProject: [bold]not markup[/bold]\nModel: first"
        card.refresh_content()
        details = card.query_one(".startup-details", StartupText)
        await _wait_for_layout(pilot, lambda: details.region.height > 0)
        assert "[bold]not markup[/bold]" in str(details.render())
        card.entry.content = card.entry.content.replace("first", "other")
        card.refresh_content()
        assert "other" in str(details.render()) and "first" not in str(details.render())
        for width in [0, 1, 7, 24]:
            text = Text.from_markup(app._startup_title(card.entry, width))
            assert text.cell_len <= width
        output = Console(width=80, record=True)
        output.print(app._startup_summary(card.entry, 80))
        assert "build · ask permissions" in output.export_text()
