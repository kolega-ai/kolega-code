from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from kolega_code.cli.settings import CliSettings, SettingsStore

if TYPE_CHECKING:
    from textual.pilot import Pilot

    from kolega_code.cli.app import KolegaCodeApp


def test_discovery_tips_default_off(tmp_path: Path) -> None:
    assert CliSettings().discovery_tips is False
    assert CliSettings().to_dict()["discovery_tips"] is False
    assert SettingsStore(tmp_path).load().discovery_tips is False


@pytest.mark.parametrize("schema_version", [1, 2, 3])
def test_legacy_settings_leave_discovery_tips_off(tmp_path: Path, schema_version: int) -> None:
    store = SettingsStore(tmp_path)
    store.path.write_text(json.dumps({"schema_version": schema_version}), encoding="utf-8")
    assert store.load().discovery_tips is False


@pytest.mark.parametrize("value", [None, "false", "true", "", 0, 1, [], {}, ["true"]])
def test_invalid_discovery_tips_never_opt_in(value: object) -> None:
    settings = CliSettings.from_dict({"schema_version": 3, "discovery_tips": value})
    assert settings.discovery_tips is False


@pytest.mark.parametrize("enabled", [False, True])
def test_discovery_tips_save_load(tmp_path: Path, enabled: bool) -> None:
    store = SettingsStore(tmp_path)
    settings = store.load()
    settings.discovery_tips = enabled
    store.save(settings)
    assert json.loads(store.path.read_text(encoding="utf-8"))["discovery_tips"] is enabled
    assert store.load().discovery_tips is enabled
    assert SettingsStore(tmp_path).load().discovery_tips is enabled

    settings.discovery_tips = not enabled
    store.save(settings)
    assert SettingsStore(tmp_path).load().discovery_tips is not enabled


def test_unrelated_stale_settings_save_preserves_discovery_opt_in(tmp_path: Path) -> None:
    store = SettingsStore(tmp_path)
    stale = store.load()
    current = store.load()
    current.discovery_tips = True
    store.save(current)
    stale.active_theme = "light"
    store.save(stale)
    assert store.load().discovery_tips is True


async def _wait_for_layout(pilot: Pilot, predicate: Callable[[], bool]) -> None:
    for _ in range(50):
        await pilot.pause()
        if predicate():
            return
    assert predicate()


@pytest.mark.asyncio
@pytest.mark.parametrize("width", [32, 80])
@pytest.mark.parametrize("theme_name", ["textual-dark", "textual-light"])
async def test_discovery_tip_cycles_by_click_and_keyboard_and_is_selectable(width: int, theme_name: str) -> None:
    pytest.importorskip("textual")
    from textual.app import App, ComposeResult
    from textual.selection import Selection
    from textual.widgets import Button, Static

    from kolega_code.cli.tui.discovery import DISCOVERY_TIPS, DiscoveryTip

    class TipApp(App[None]):
        def compose(self) -> ComposeResult:
            yield DiscoveryTip(id="tip")

    app = TipApp()
    app.theme = theme_name
    async with app.run_test(size=(width, 12)) as pilot:
        tip = app.query_one(DiscoveryTip)
        copy = tip.query_one(".discovery-tip-text", Static)
        button = tip.query_one(Button)
        await _wait_for_layout(
            pilot,
            lambda: (
                copy.region.width > 0
                and button.region.width > 0
                and button.region.right <= width
                and tip.region.height >= copy.region.height
            ),
        )
        assert tip.current_tip == DISCOVERY_TIPS[0]
        assert copy.get_selection(Selection(None, None)) == (f"Tip: {tip.current_tip}", "\n")
        # Native Content must expose mouse-selection offsets, not just get_selection.
        await _wait_for_layout(
            pilot,
            lambda: app.screen.get_widget_and_offset_at(copy.region.x + 1, copy.region.y)[1] is not None,
        )
        assert button.can_focus
        assert button.tooltip == "Next discovery tip"
        assert button.active_effect_duration == 0

        assert await pilot.click(button)
        await pilot.pause()
        assert tip.current_tip == DISCOVERY_TIPS[1]
        button.focus()
        await pilot.press("enter")
        await pilot.pause()
        assert tip.current_tip == DISCOVERY_TIPS[2]
        await pilot.press("enter")
        await pilot.pause()
        assert tip.current_tip == DISCOVERY_TIPS[0]
        assert copy.get_selection(Selection(None, None)) == (f"Tip: {tip.current_tip}", "\n")
        # A fresh widget never inherits another widget's position.
        tip.action_next_tip()
        assert DiscoveryTip().current_tip == DISCOVERY_TIPS[0]


def _configured_app(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    enabled: bool,
    *,
    resuming: bool = False,
    history: list[dict] | None = None,
) -> KolegaCodeApp:
    from kolega_code.cli.app import KolegaCodeApp
    from kolega_code.cli.config import build_agent_config, config_summary
    from kolega_code.cli.provider_registry import UI_DEFAULT_MODEL, UI_DEFAULT_PROVIDER
    from kolega_code.cli.session_store import SessionStore
    from kolega_code.llm.models import Message

    from ._app_test_utils import install_fake_agents

    install_fake_agents(monkeypatch)
    project = tmp_path / "project"
    project.mkdir()
    settings_store = SettingsStore(tmp_path / "state")
    settings = CliSettings(
        active_provider=UI_DEFAULT_PROVIDER,
        active_model=UI_DEFAULT_MODEL,
        discovery_tips=enabled,
    )
    settings.set_api_key(UI_DEFAULT_PROVIDER, "fake-test-key")
    settings_store.save(settings)
    config = build_agent_config(project, env={}, settings=settings, settings_store=settings_store)
    store = SessionStore(settings_store.root)
    session = store.create(project, "code", config_summary(config))
    recorder = store.recorder(session.session_id)
    for payload in history or []:
        message = Message.from_dict(payload)
        recorder.record_context_message(message, actor=message.role)
    return KolegaCodeApp(
        project_path=project,
        config=config,
        mode="code",
        store=store,
        settings_store=settings_store,
        session=session,
        resuming_session=resuming,
    )


@pytest.mark.asyncio
async def test_startup_tip_opt_in_preserves_index_across_refresh_and_remount(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, isolated_cli_env: None
) -> None:
    from kolega_code.cli.tui.discovery import DISCOVERY_TIPS, DiscoveryTip
    from kolega_code.cli.tui.startup import StartupEntryWidget

    app = _configured_app(tmp_path, monkeypatch, False)
    async with app.run_test(size=(120, 40)) as pilot:
        assert not app.query(DiscoveryTip)
        app.settings.discovery_tips = True
        app._ensure_startup_entry()
        await _wait_for_layout(pilot, lambda: bool(app.query(DiscoveryTip)))
        tip = app.query_one(DiscoveryTip)
        await _wait_for_layout(pilot, lambda: tip.region.height > 0)
        card = app.query_one(StartupEntryWidget)
        tip.action_next_tip()
        await _wait_for_layout(pilot, lambda: tip.current_tip == DISCOVERY_TIPS[1])
        for _ in range(3):
            app._ensure_startup_entry()
            await pilot.pause()
        assert app.query_one(DiscoveryTip) is tip
        assert tip.current_tip == DISCOVERY_TIPS[1]
        app.settings.discovery_tips = False
        app._ensure_startup_entry()
        await _wait_for_layout(pilot, lambda: not tip.display and tip.region.height == 0)
        app.settings.discovery_tips = True
        app._ensure_startup_entry()
        await _wait_for_layout(pilot, lambda: tip.display and tip.region.height > 0)
        assert app.query_one(DiscoveryTip) is tip
        app._render_conversation()
        await _wait_for_layout(pilot, lambda: app.query_one(StartupEntryWidget) is not card)
        assert app.query_one(DiscoveryTip).current_tip == DISCOVERY_TIPS[1]
        # Discovery stays presentation-only: no model message or persisted tip index.
        assert not app.session.history
        assert all(tip_text not in entry.content for tip_text in DISCOVERY_TIPS for entry in app.conversation_entries)
        assert "tip_index" not in app.settings_store.path.read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_first_submission_hides_tip_even_when_card_reopens_and_reset_starts_fresh(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, isolated_cli_env: None
) -> None:
    from textual.widgets import Collapsible

    from kolega_code.cli.tui.discovery import DISCOVERY_TIPS, DiscoveryTip
    from kolega_code.cli.tui.startup import StartupEntryWidget
    from kolega_code.cli.tui.widgets import ChatComposer

    app = _configured_app(tmp_path, monkeypatch, True)
    async with app.run_test(size=(120, 40)) as pilot:
        await _wait_for_layout(pilot, lambda: bool(app.query(DiscoveryTip)))
        tip = app.query_one(DiscoveryTip)
        tip.action_next_tip()
        composer = app.query_one(ChatComposer)
        composer.load_text("inspect the workspace")
        composer.focus()
        await pilot.press("enter")
        await _wait_for_layout(pilot, lambda: app.agent_worker is None and not tip.display)
        card = app.query_one(StartupEntryWidget)
        card.query_one(Collapsible).collapsed = False
        app._ensure_startup_entry()
        await pilot.pause()
        assert not tip.display
        assert app.config is not None
        await app._build_agent(app.config, rebuild=True)
        await _wait_for_layout(pilot, lambda: app.query_one(StartupEntryWidget) is not card)
        assert not any(widget.display for widget in app.query(DiscoveryTip))
        await app._reset_current_thread()
        await _wait_for_layout(pilot, lambda: any(widget.display for widget in app.query(DiscoveryTip)))
        assert app.query_one(DiscoveryTip).current_tip == DISCOVERY_TIPS[0]


@pytest.mark.asyncio
@pytest.mark.parametrize("resuming,with_history", [(True, False), (True, True), (False, True)])
async def test_discovery_never_appears_when_resuming_or_restoring_history(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    isolated_cli_env: None,
    resuming: bool,
    with_history: bool,
) -> None:
    from kolega_code.cli.tui.discovery import DiscoveryTip
    from kolega_code.llm.models import Message, TextBlock

    history = [Message(role="user", content=[TextBlock(text="earlier prompt")]).to_dict()] if with_history else []
    app = _configured_app(tmp_path, monkeypatch, True, resuming=resuming, history=history)
    async with app.run_test(size=(120, 40)) as pilot:
        assert bool(app.session.history) is with_history
        assert not app.query(DiscoveryTip)
        app._ensure_startup_entry()
        await pilot.pause()
        assert not app.query(DiscoveryTip)


@pytest.mark.asyncio
@pytest.mark.parametrize("enabled", [False, True])
async def test_discovery_settings_switch_roundtrip_and_discard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, isolated_cli_env: None, enabled: bool
) -> None:
    pytest.importorskip("textual")
    from textual.widgets import Button, Switch

    from kolega_code.cli.tui.settings_screen import ConfirmDiscardSettingsScreen, SettingsScreen

    app = _configured_app(tmp_path, monkeypatch, enabled)
    async with app.run_test(size=(100, 40)) as pilot:
        app.action_open_settings(category="appearance")
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, SettingsScreen)
        await _wait_for_layout(pilot, lambda: not screen._initializing)
        switch = screen.query_one("#discovery_tips_switch", Switch)
        apply_button = screen.query_one("#save_settings", Button)
        assert switch.value is enabled
        assert not screen.dirty
        assert apply_button.disabled

        switch.focus()
        await pilot.press("space")
        await _wait_for_layout(pilot, lambda: screen.dirty and not apply_button.disabled)
        assert switch.value is not enabled
        assert app.settings.discovery_tips is enabled
        assert app.settings_store.load().discovery_tips is enabled
        candidate, *_ = app._settings_candidate_from_ui()
        assert candidate.discovery_tips is not enabled
        assert app.settings.discovery_tips is enabled

        switch.value = enabled
        await _wait_for_layout(pilot, lambda: not screen.dirty and apply_button.disabled)
        switch.value = not enabled
        await _wait_for_layout(pilot, lambda: screen.dirty and not apply_button.disabled)
        await app._save_settings_from_ui()
        await _wait_for_layout(pilot, lambda: not screen.dirty and apply_button.disabled)
        assert app.settings.discovery_tips is not enabled
        assert app.settings_store.load().discovery_tips is not enabled

        screen.action_close()
        await pilot.pause()
        app.action_open_settings(category="appearance")
        await pilot.pause()
        reopened = app.screen
        assert isinstance(reopened, SettingsScreen)
        await _wait_for_layout(pilot, lambda: not reopened._initializing)
        assert reopened.query_one("#discovery_tips_switch", Switch).value is not enabled
        reopened.query_one("#discovery_tips_switch", Switch).value = enabled
        await _wait_for_layout(pilot, lambda: reopened.dirty)
        reopened.action_close()
        await pilot.pause()
        assert isinstance(app.screen, ConfirmDiscardSettingsScreen)
        app.screen.query_one("#settings_confirm_discard", Button).press()
        await pilot.pause()
        assert app.settings.discovery_tips is not enabled
        assert app.settings_store.load().discovery_tips is not enabled
