# ruff: noqa: F401,F811,E402
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from kolega_code.memory import MISSING_REVISION

from ._app_test_utils import (
    FakeCoderAgent,
    _build_mention_test_app,
    build_test_config,
    install_fake_agents,
)


async def _wait_for_entry(screen, pilot, reference: str) -> None:
    for _ in range(40):
        await pilot.pause(0.025)
        if screen._loaded_entry is not None and screen._loaded_entry.reference == reference:
            return
    raise AssertionError(f"Timed out waiting for memory entry {reference}")


@pytest.mark.asyncio
async def test_app_injects_one_memory_manager_and_closes_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("textual")

    app = _build_mention_test_app(tmp_path, monkeypatch)
    manager = app.memory_manager
    backend = manager.backend
    assert backend is not None
    close_calls = 0
    original_close = backend.close

    def close() -> None:
        nonlocal close_calls
        close_calls += 1
        original_close()

    monkeypatch.setattr(backend, "close", close)

    async with app.run_test():
        assert isinstance(app.agent, FakeCoderAgent)
        assert app.agent.kwargs["memory_manager"] is manager
        first_agent = app.agent
        assert app.config is not None
        await app._build_agent(app.config, rebuild=True)
        assert app.agent is not first_agent
        assert app.agent.kwargs["memory_manager"] is manager
        assert close_calls == 0

    assert close_calls == 1


def test_app_memory_manager_close_is_idempotent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = _build_mention_test_app(tmp_path, monkeypatch)
    backend = app.memory_manager.backend
    assert backend is not None
    close_calls = 0
    original_close = backend.close

    def close() -> None:
        nonlocal close_calls
        close_calls += 1
        original_close()

    monkeypatch.setattr(backend, "close", close)

    app._close_memory_manager()
    app._close_memory_manager()

    assert close_calls == 1


@pytest.mark.asyncio
async def test_app_quit_closes_memory_when_session_save_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = _build_mention_test_app(tmp_path, monkeypatch)
    backend = app.memory_manager.backend
    assert backend is not None
    close = MagicMock(wraps=backend.close)
    monkeypatch.setattr(backend, "close", close)
    app.agent = MagicMock()
    app.agent.fire_hook = AsyncMock()
    app._save_session_history_async = AsyncMock(side_effect=RuntimeError("session save failed"))
    app.exit = MagicMock()

    with pytest.raises(RuntimeError, match="session save failed"):
        await app.action_quit()

    close.assert_called_once_with()
    app.exit.assert_called_once_with()


@pytest.mark.asyncio
async def test_app_memory_prompt_refresh_failure_is_best_effort(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("textual")

    app = _build_mention_test_app(tmp_path, monkeypatch)
    warnings: list[tuple[str, str]] = []

    def fail_refresh() -> None:
        raise RuntimeError("simulated prompt refresh failure")

    monkeypatch.setattr(app.memory_manager, "refresh", fail_refresh)
    monkeypatch.setattr(
        app,
        "_notify_user",
        lambda message, *, severity="information": warnings.append((message, severity)),
    )

    async with app.run_test():
        await app._refresh_agent_memory()

    assert warnings
    assert warnings[-1][1] == "warning"
    assert "Memory was updated" in warnings[-1][0]


@pytest.mark.asyncio
async def test_memory_slash_status_files_show_and_disable_preserves_bank(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli.tui.widgets import ChatComposer

    app = _build_mention_test_app(tmp_path, monkeypatch)
    created = app.memory_manager.replace_entry(
        "MEMORY.md",
        "# Durable facts\n\nUse `uv run pytest`.\n",
        MISSING_REVISION,
    )
    assert created.ok

    async with app.run_test() as pilot:
        composer = app.query_one("#composer", ChatComposer)

        for command in ("/memory status", "/memory files", "/memory show"):
            composer.load_text(command)
            await app.on_chat_composer_submitted(ChatComposer.Submitted(composer, composer.text))

        contents = [entry.content for entry in app.conversation_entries[-3:]]
        assert "Project memory is on" in contents[0]
        assert "Private storage:" in contents[0]
        assert "MEMORY.md" in contents[1]
        assert "Use `uv run pytest`." in contents[2]

        composer.load_text("/memory off")
        await app.on_chat_composer_submitted(ChatComposer.Submitted(composer, composer.text))
        assert app.memory_manager.enabled is False
        assert app.memory_manager.read_entry(
            "MEMORY.md",
            allow_disabled=True,
        ).present

        composer.load_text("/memory on")
        await app.on_chat_composer_submitted(ChatComposer.Submitted(composer, composer.text))
        assert app.memory_manager.enabled is True
        await pilot.pause()


@pytest.mark.asyncio
async def test_memory_screen_preserves_editor_on_stale_cas(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("textual")

    from textual.widgets import Button, Static, TextArea

    from kolega_code.cli.tui.memory_screen import MemoryScreen

    app = _build_mention_test_app(tmp_path, monkeypatch)
    created = app.memory_manager.replace_entry(
        "MEMORY.md",
        "# Original\n",
        MISSING_REVISION,
    )
    assert created.ok

    async with app.run_test() as pilot:
        app.action_open_memory()
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, MemoryScreen)
        await _wait_for_entry(screen, pilot, "MEMORY.md")

        original = screen._loaded_entry
        assert original is not None
        screen.action_edit()
        editor = screen.query_one("#memory_editor", TextArea)
        editor.text = "# My unsaved edit\n"

        concurrent = app.memory_manager.replace_entry(
            "MEMORY.md",
            "# Concurrent edit\n",
            original.revision,
        )
        assert concurrent.ok

        screen.action_save()
        for _ in range(40):
            await pilot.pause(0.025)
            notice = str(screen.query_one("#memory_notice", Static).render())
            if "current revision" in notice:
                break

        assert screen._editing is True
        assert editor.text == "# My unsaved edit\n"
        assert "current revision" in str(screen.query_one("#memory_notice", Static).render())
        assert screen.query_one("#memory_reload", Button).display

        await screen._reload_latest("MEMORY.md")
        assert screen._editing is True
        assert screen._stale_conflict is False
        assert editor.text == "# Concurrent edit\n"
        assert screen._loaded_entry is not None
        assert screen._loaded_entry.revision == concurrent.revision


@pytest.mark.asyncio
async def test_memory_screen_requires_explicit_inspection_when_disabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli.tui.memory_screen import MemoryScreen

    app = _build_mention_test_app(tmp_path, monkeypatch)
    assert app.memory_manager.replace_entry(
        "MEMORY.md",
        "# Preserved\n",
        MISSING_REVISION,
    ).ok
    app.memory_manager.set_enabled(False)

    async with app.run_test() as pilot:
        app.action_open_memory()
        await pilot.pause(0.1)
        screen = app.screen
        assert isinstance(screen, MemoryScreen)
        assert screen._entries == []
        assert screen._loaded_entry is None

        screen.action_inspect_disabled()
        await _wait_for_entry(screen, pilot, "MEMORY.md")
        assert screen._loaded_entry is not None
        assert app.memory_manager.enabled is False
