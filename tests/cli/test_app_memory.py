# ruff: noqa: F401,F811,E402
from dataclasses import replace
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from kolega_code.memory import MISSING_REVISION

from ._app_test_utils import (
    FakeCoderAgent,
    _build_mention_test_app,
    build_test_config,
    install_fake_agents,
    open_settings_screen,
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
    secret_like_content = "API_KEY=supersecretvalue123"
    created = app.memory_manager.replace_entry(
        "MEMORY.md",
        f"# Durable facts\n\nUse `uv run pytest`.\n\n{secret_like_content}\n",
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
        assert "Agent startup context" in contents[0]
        assert secret_like_content in contents[0]
        assert "Redacted MEMORY.md preview" not in contents[0]
        assert "MEMORY.md" in contents[1]
        assert "Use `uv run pytest`." in contents[2]
        assert secret_like_content in contents[2]

        composer.load_text("/memory bogus")
        await app.on_chat_composer_submitted(ChatComposer.Submitted(composer, composer.text))
        assert "browse" in app.conversation_entries[-1].content

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

    from textual.widgets import Button, Markdown, Static, TextArea

    from kolega_code.cli.tui.memory_screen import MemoryScreen

    app = _build_mention_test_app(tmp_path, monkeypatch)
    secret_like_content = "API_KEY=supersecretvalue123"
    original_content = f"# Original\n\n{secret_like_content}\n"
    created = app.memory_manager.replace_entry(
        "MEMORY.md",
        original_content,
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
        assert original.content == original_content
        assert secret_like_content in screen.query_one("#memory_preview", Markdown).source
        assert "withheld" not in str(screen.query_one("#memory_status", Static).render()).casefold()
        assert screen.query_one("#memory_edit", Button).disabled is False
        warned_entry = replace(original, warnings=("generic backend notice",))
        screen._loaded_entry = warned_entry
        screen._render_entry(warned_entry)
        assert "generic backend notice" in str(screen.query_one("#memory_notice", Static).render())
        screen.action_edit()
        assert screen._editing is True
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
async def test_memory_screen_edit_mode_blocks_roster_and_filter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("textual")

    from textual.widgets import Input, OptionList, TextArea

    from kolega_code.cli.tui.memory_screen import MemoryScreen

    app = _build_mention_test_app(tmp_path, monkeypatch)
    assert app.memory_manager.replace_entry("MEMORY.md", "# Index\n", MISSING_REVISION).ok
    assert app.memory_manager.replace_entry("topics/build.md", "# Build\n", MISSING_REVISION).ok

    async with app.run_test() as pilot:
        app.action_open_memory()
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, MemoryScreen)
        await _wait_for_entry(screen, pilot, "MEMORY.md")

        screen.action_edit()
        assert screen._editing is True
        editor = screen.query_one("#memory_editor", TextArea)
        editor.text = "# Unsaved edit\n"
        assert screen.query_one("#memory_filter", Input).disabled is True
        assert screen.query_one("#memory_entries", OptionList).disabled is True

        screen._start_refresh()
        screen._start_load("topics/build.md")
        await pilot.pause(0.2)

        assert screen._editing is True
        assert screen._loaded_entry is not None
        assert screen._loaded_entry.reference == "MEMORY.md"
        assert editor.text == "# Unsaved edit\n"

        screen.action_cancel_edit()
        assert screen.query_one("#memory_filter", Input).disabled is False
        assert screen.query_one("#memory_entries", OptionList).disabled is False


@pytest.mark.asyncio
async def test_memory_screen_filter_is_client_side_and_preserves_selection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("textual")

    from textual.widgets import Input

    from kolega_code.cli.tui.memory_screen import MemoryScreen

    app = _build_mention_test_app(tmp_path, monkeypatch)
    for reference, content in (
        ("MEMORY.md", "# Index\n"),
        ("topics/build.md", "# Build\n"),
        ("topics/design.md", "# Design\n"),
    ):
        assert app.memory_manager.replace_entry(reference, content, MISSING_REVISION).ok

    async with app.run_test() as pilot:
        app.action_open_memory()
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, MemoryScreen)
        await _wait_for_entry(screen, pilot, "MEMORY.md")

        screen._start_load("topics/build.md")
        await _wait_for_entry(screen, pilot, "topics/build.md")
        loaded = screen._loaded_entry
        assert loaded is not None

        list_calls = 0
        original_list_entries = app.memory_manager.list_entries

        def counting_list_entries(*args, **kwargs):
            nonlocal list_calls
            list_calls += 1
            return original_list_entries(*args, **kwargs)

        monkeypatch.setattr(app.memory_manager, "list_entries", counting_list_entries)

        screen.query_one("#memory_filter", Input).value = "build"
        await pilot.pause(0.1)
        assert [entry.reference for entry in screen._entries] == ["topics/build.md"]
        assert screen._loaded_entry is loaded
        assert list_calls == 0

        screen.query_one("#memory_filter", Input).value = "design"
        await _wait_for_entry(screen, pilot, "topics/design.md")
        assert [entry.reference for entry in screen._entries] == ["topics/design.md"]
        assert list_calls == 0


@pytest.mark.asyncio
async def test_memory_screen_agent_view_matches_prompt_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("textual")

    from textual.widgets import Markdown, Static

    from kolega_code.cli.tui.memory_screen import MemoryScreen

    app = _build_mention_test_app(tmp_path, monkeypatch)
    assert app.memory_manager.replace_entry(
        "MEMORY.md",
        "# Index\n\nA distinctive durable fact.\n",
        MISSING_REVISION,
    ).ok

    async with app.run_test() as pilot:
        app.action_open_memory()
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, MemoryScreen)
        await _wait_for_entry(screen, pilot, "MEMORY.md")

        screen.action_agent_view()
        for _ in range(40):
            await pilot.pause(0.025)
            if screen._agent_view:
                break
        assert screen._agent_view is True

        expected = app.memory_manager.prompt_context()
        preview = screen.query_one("#memory_preview", Markdown).source
        assert "Private project memory" in preview
        assert "A distinctive durable fact." in preview
        assert preview == expected.text
        metadata = str(screen.query_one("#memory_metadata", Static).render())
        assert "Agent startup context" in metadata

        screen._start_load("MEMORY.md")
        await pilot.pause(0.2)
        assert screen._agent_view is False

        app.memory_manager.set_enabled(False)
        screen.action_agent_view()
        for _ in range(40):
            await pilot.pause(0.025)
            if screen._agent_view:
                break
        assert "receives no memory context" in screen.query_one("#memory_preview", Markdown).source


@pytest.mark.asyncio
async def test_memory_screen_destructive_confirmations_use_danger(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli.tui.memory_screen import MemoryScreen
    from kolega_code.cli.tui.settings_screen import ConfirmSettingsActionScreen

    app = _build_mention_test_app(tmp_path, monkeypatch)
    assert app.memory_manager.replace_entry("MEMORY.md", "# Index\n", MISSING_REVISION).ok

    async with app.run_test() as pilot:
        app.action_open_memory()
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, MemoryScreen)
        await _wait_for_entry(screen, pilot, "MEMORY.md")

        captured: list[ConfirmSettingsActionScreen] = []

        def capture_screen(screen_obj, callback=None, **kwargs):
            assert isinstance(screen_obj, ConfirmSettingsActionScreen)
            captured.append(screen_obj)

        monkeypatch.setattr(app, "push_screen", capture_screen)

        screen.action_delete()
        screen.action_clear()
        screen.action_edit()
        screen.action_close()

        assert [confirm.danger for confirm in captured] == [True, True, True]


@pytest.mark.asyncio
async def test_confirm_memory_clear_helper_clears_and_reports(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli.tui.settings_screen import ConfirmSettingsActionScreen

    app = _build_mention_test_app(tmp_path, monkeypatch)
    assert app.memory_manager.replace_entry("MEMORY.md", "# Index\n", MISSING_REVISION).ok

    async with app.run_test() as pilot:

        def confirm_immediately(screen_obj, callback=None, **kwargs):
            assert isinstance(screen_obj, ConfirmSettingsActionScreen)
            assert screen_obj.danger is True
            assert callback is not None
            callback(True)

        notifications: list[str] = []
        monkeypatch.setattr(app, "push_screen", confirm_immediately)
        monkeypatch.setattr(app, "notify", lambda message, **kwargs: notifications.append(message))

        cleared: list[int] = []
        app.confirm_memory_clear(on_done=cleared.append)
        for _ in range(40):
            await pilot.pause(0.025)
            if cleared:
                break

        assert cleared == [1]
        assert app.memory_manager.list_entries(None, allow_disabled=True) == []
        assert any("Cleared 1 private memory entries" in message for message in notifications)


@pytest.mark.asyncio
async def test_settings_memory_page_stages_and_applies(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("textual")

    from textual.widgets import Select, Static

    app = _build_mention_test_app(tmp_path, monkeypatch)
    assert app.memory_manager.replace_entry("MEMORY.md", "# Index\n", MISSING_REVISION).ok

    async with app.run_test(size=(100, 40)) as pilot:
        screen = await open_settings_screen(app, pilot, "memory")
        for _ in range(40):
            await pilot.pause(0.025)
            status_text = str(screen.query_one("#memory_settings_status", Static).render())
            if "markdown" in status_text:
                break
        await pilot.pause()
        assert screen.dirty is False

        screen.query_one("#memory_enabled_select", Select).value = "false"
        await pilot.pause()
        assert app.memory_manager.enabled is True
        assert screen.dirty is True

        assert await screen.apply_memory_draft() is True
        assert app.memory_manager.enabled is False
        for _ in range(40):
            await pilot.pause(0.025)
            if not screen.dirty:
                break
        assert screen.dirty is False


@pytest.mark.asyncio
async def test_settings_memory_apply_failure_preserves_draft(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("textual")

    from textual.widgets import Button, Select, Static

    app = _build_mention_test_app(tmp_path, monkeypatch)

    async with app.run_test(size=(100, 40)) as pilot:
        screen = await open_settings_screen(app, pilot, "memory")
        for _ in range(40):
            await pilot.pause(0.025)
            if "markdown" in str(screen.query_one("#memory_settings_status", Static).render()):
                break
        await pilot.pause()

        screen.query_one("#memory_enabled_select", Select).value = "false"
        await pilot.pause()

        def fail_set_enabled(_enabled: bool) -> None:
            raise OSError("manifest unavailable")

        monkeypatch.setattr(app.memory_manager, "set_enabled", fail_set_enabled)

        assert await screen.apply_memory_draft() is False
        await pilot.pause()
        assert screen.query_one("#memory_enabled_select", Select).value == "false"
        assert screen.dirty is True
        assert screen.query_one("#save_settings", Button).disabled is False
        assert app.memory_manager.enabled is True


@pytest.mark.asyncio
async def test_settings_clear_refresh_preserves_memory_draft(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("textual")

    from textual.widgets import Select, Static

    app = _build_mention_test_app(tmp_path, monkeypatch)
    assert app.memory_manager.replace_entry("MEMORY.md", "# Index\n", MISSING_REVISION).ok

    async with app.run_test(size=(100, 40)) as pilot:
        screen = await open_settings_screen(app, pilot, "memory")
        for _ in range(40):
            await pilot.pause(0.025)
            if "markdown" in str(screen.query_one("#memory_settings_status", Static).render()):
                break
        await pilot.pause()

        screen.query_one("#memory_enabled_select", Select).value = "false"
        await pilot.pause()
        assert screen.dirty is True

        screen._after_memory_clear(1)
        await pilot.pause(0.1)

        assert screen.query_one("#memory_enabled_select", Select).value == "false"
        assert screen.dirty is True


@pytest.mark.asyncio
async def test_settings_save_does_not_report_success_when_memory_apply_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("textual")

    from textual.widgets import Button, Select, Static

    app = _build_mention_test_app(tmp_path, monkeypatch)

    async with app.run_test(size=(100, 40)) as pilot:
        screen = await open_settings_screen(app, pilot, "memory")
        for _ in range(40):
            await pilot.pause(0.025)
            if "markdown" in str(screen.query_one("#memory_settings_status", Static).render()):
                break
        await pilot.pause()

        screen.query_one("#memory_enabled_select", Select).value = "false"
        await pilot.pause()

        async def fail_memory_apply() -> bool:
            return False

        notices: list[str] = []
        monkeypatch.setattr(screen, "apply_memory_draft", fail_memory_apply)
        monkeypatch.setattr(app, "_notify_user", lambda message, **_kwargs: notices.append(message))

        await app._save_settings_from_ui()
        await pilot.pause()

        assert screen.dirty is True
        assert screen.query_one("#save_settings", Button).disabled is False
        assert "project memory changes failed" in str(screen.query_one("#settings_status", Static).render())
        assert all("Settings saved" not in message for message in notices)


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
