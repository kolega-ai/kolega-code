# ruff: noqa: F401,F811,E402
"""Portable shortcuts for tmux / constrained terminals.

Covers:
- /attach with no path reading the system clipboard
- ChatComposer bindings for Ctrl+Shift+V and Alt+V
- Ctrl+J newline fallback
- startup content tmux hint when TMUX/TERM indicates a multiplexer
"""

from __future__ import annotations

from pathlib import Path

import pytest

from kolega_code.cli import messages
from kolega_code.cli.config import config_summary
from kolega_code.cli.session_store import SessionStore

from ._app_test_utils import build_test_config, install_fake_agents


def _make_app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    pytest.importorskip("textual")

    from kolega_code.cli.app import KolegaCodeApp

    install_fake_agents(monkeypatch)

    project = tmp_path / "project"
    project.mkdir()
    config = build_test_config(project)
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", config_summary(config))
    return KolegaCodeApp(project_path=project, config=config, mode="code", store=store, session=session)


def _binding_keys(bindings) -> set[str]:
    keys: set[str] = set()
    for binding in bindings:
        key = getattr(binding, "key", None)
        if not key:
            continue
        for part in str(key).split(","):
            keys.add(part.strip())
    return keys


@pytest.mark.asyncio
async def test_attach_no_args_reads_clipboard_image(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    app = _make_app(tmp_path, monkeypatch)

    async def _fake_read():
        return (b"fake-png-bytes", "image/png")

    monkeypatch.setattr("kolega_code.cli.clipboard_image.read_clipboard_image", _fake_read)

    async with app.run_test():
        await app._command_attach("")
        assert len(app._pending_image_attachments) == 1
        assert app._pending_image_attachments[0]["path"] == "clipboard"
        assert app._pending_image_attachments[0]["media_type"] == "image/png"


@pytest.mark.asyncio
async def test_attach_no_args_empty_clipboard_shows_warning(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    app = _make_app(tmp_path, monkeypatch)

    async def _fake_read():
        return None

    monkeypatch.setattr("kolega_code.cli.clipboard_image.read_clipboard_image", _fake_read)
    hints: list[tuple[str, str]] = []

    async with app.run_test():
        app._show_composer_hint = lambda text, tone="warning": hints.append((text, tone))
        await app._command_attach("")
        assert app._pending_image_attachments == []
        assert hints
        assert hints[-1][1] == "warning"
        assert hints[-1][0] == messages.ATTACH_CLIPBOARD_EMPTY


def test_chat_composer_image_paste_bindings_include_portable_alt_v() -> None:
    from textual.binding import Binding

    from kolega_code.cli.tui.widgets import ChatComposer

    keys = _binding_keys(ChatComposer.BINDINGS)
    assert "ctrl+shift+v" in keys
    assert "alt+v" in keys

    paste_binding = next(
        b for b in ChatComposer.BINDINGS if isinstance(b, Binding) and b.action == "paste_clipboard_image"
    )
    display = paste_binding.key_display or ""
    assert "Ctrl+Shift+V" in display
    assert "Alt+V" in display

    newline_binding = next(b for b in ChatComposer.BINDINGS if isinstance(b, Binding) and b.action == "insert_newline")
    assert "Ctrl+J" in (newline_binding.key_display or "")


@pytest.mark.asyncio
async def test_textual_app_composer_ctrl_j_inserts_line_break(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from kolega_code.cli.tui.widgets import ChatComposer

    app = _make_app(tmp_path, monkeypatch)

    async with app.run_test() as pilot:
        composer = app.query_one("#composer", ChatComposer)
        composer.focus()

        await pilot.press("h", "i")
        await pilot.press("ctrl+j")
        await pilot.press("t", "h", "e", "r", "e")

        assert composer.text == "hi\nthere"


@pytest.mark.asyncio
async def test_paste_clipboard_image_action_attaches_via_alt_v_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """action_paste_clipboard_image (shared by Ctrl+Shift+V and Alt+V) attaches a clipboard image."""
    from kolega_code.cli.tui.widgets import ChatComposer

    app = _make_app(tmp_path, monkeypatch)

    async def _fake_read():
        return (b"fake-png-bytes", "image/png")

    monkeypatch.setattr("kolega_code.cli.clipboard_image.read_clipboard_image", _fake_read)

    async with app.run_test() as pilot:
        composer = app.query_one("#composer", ChatComposer)
        composer.focus()
        composer.action_paste_clipboard_image()
        # Worker runs the clipboard read asynchronously.
        for _ in range(20):
            await pilot.pause()
            if app._pending_image_attachments:
                break
        assert len(app._pending_image_attachments) == 1
        assert app._pending_image_attachments[0]["path"] == "clipboard"


def test_running_under_tmux_or_screen_detection(monkeypatch: pytest.MonkeyPatch) -> None:
    from kolega_code.cli.app import KolegaCodeApp

    monkeypatch.delenv("TMUX", raising=False)
    monkeypatch.setenv("TERM", "xterm-256color")
    assert KolegaCodeApp._running_under_tmux_or_screen() is False

    monkeypatch.setenv("TMUX", "/tmp/tmux-1000/default,1234,0")
    assert KolegaCodeApp._running_under_tmux_or_screen() is True

    monkeypatch.delenv("TMUX", raising=False)
    monkeypatch.setenv("TERM", "screen-256color")
    assert KolegaCodeApp._running_under_tmux_or_screen() is True

    monkeypatch.setenv("TERM", "tmux-256color")
    assert KolegaCodeApp._running_under_tmux_or_screen() is True


@pytest.mark.asyncio
async def test_startup_content_includes_tmux_hint_when_nested(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TMUX", "/tmp/tmux-1000/default,1,0")
    app = _make_app(tmp_path, monkeypatch)

    async with app.run_test():
        content = app._startup_content()
        assert messages.TMUX_SHORTCUT_HINT in content
        assert "Ctrl+J" in content
        assert "/attach" in content
        assert "Alt+V" in content


@pytest.mark.asyncio
async def test_startup_content_omits_tmux_hint_outside_multiplexer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("TMUX", raising=False)
    monkeypatch.setenv("TERM", "xterm-ghostty")
    app = _make_app(tmp_path, monkeypatch)

    async with app.run_test():
        content = app._startup_content()
        assert messages.TMUX_SHORTCUT_HINT not in content
        # Portable fallbacks still appear in the general help lines.
        assert "Ctrl+J" in content
        assert "/attach" in content
