from __future__ import annotations

import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from kolega_code.cli.settings import CliSettings, SettingsStore

if TYPE_CHECKING:
    from textual.pilot import Pilot

    from kolega_code.cli.app import KolegaCodeApp


TIP_SHORTCUTS: tuple[str, ...] = (
    "/plan",
    "Ctrl+G",
    "@",
    "/loop",
    "/rewind",
    "",
    "/memory",
    "/handoff",
    "/skills",
    "/permissions",
    "/copy",
    "/lsp",
)


def test_discovery_tip_catalog_preserves_originals_and_uses_registered_commands() -> None:
    from kolega_code.cli.slash_commands import TUI_COMMAND_NAMES
    from kolega_code.cli.tui.discovery import DISCOVERY_TIPS

    assert DISCOVERY_TIPS[:3] == (
        "Try /plan to explore a change before editing files.",
        "Ctrl+G opens the sub-agent inspector without losing your place.",
        "Type @ in the composer to mention a file in your project.",
    )
    assert len(DISCOVERY_TIPS) == len(TIP_SHORTCUTS)
    assert len(set(DISCOVERY_TIPS)) == len(DISCOVERY_TIPS)
    for tip, shortcut in zip(DISCOVERY_TIPS, TIP_SHORTCUTS, strict=True):
        assert "".join(re.findall(r"/[\w-]+|Ctrl\+\w+|@", tip)) == shortcut
        assert set(re.findall(r"/[\w-]+", tip)) <= TUI_COMMAND_NAMES
        assert "\n" not in tip


def test_discovery_tip_initial_index_wraps_and_fresh_widgets_reset() -> None:
    from kolega_code.cli.tui.discovery import DISCOVERY_TIPS, DiscoveryTip

    count = len(DISCOVERY_TIPS)
    for index in (-count - 1, -1, 0, count - 1, count, count + 1):
        tip = DiscoveryTip(index=index)
        assert tip.current_tip == DISCOVERY_TIPS[index % count]
        tip.action_next_tip()
        assert tip.current_tip == DISCOVERY_TIPS[(index + 1) % count]
        assert DiscoveryTip().current_tip == DISCOVERY_TIPS[0]


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
@pytest.mark.parametrize("width", [32, 40, 80])
@pytest.mark.parametrize("theme_name", ["textual-dark", "textual-light"])
async def test_discovery_tip_cycles_by_click_and_keyboard_and_is_selectable(width: int, theme_name: str) -> None:
    pytest.importorskip("textual")
    from textual.app import App
    from textual.content import Content
    from textual.selection import Selection
    from textual.widgets import Button, Static

    from kolega_code.cli.tui.discovery import DISCOVERY_TIPS, DiscoveryTip

    class TipApp(App[None]):
        async def on_mount(self) -> None:
            await self.push_screen(DiscoveryTip(id="tip"))

    app = TipApp()
    app.theme = theme_name
    async with app.run_test(size=(width, 24)) as pilot:
        tip = app.screen
        assert isinstance(tip, DiscoveryTip)
        dialog = tip.query_one(".discovery-tip-dialog")
        copy = tip.query_one(".discovery-tip-text", Static)
        button = tip.query_one(".discovery-tip-next", Button)
        close = tip.query_one("#discovery-tip-close", Button)
        badge = tip.query_one(".discovery-tip-badge", Static)
        position = tip.query_one(".discovery-tip-position", Static)

        async def assert_tip_presentation(index: int) -> None:
            expected_tip = DISCOVERY_TIPS[index]
            await _wait_for_layout(
                pilot,
                lambda: (
                    tip.current_tip == expected_tip
                    and str(position.render()) == f"{index + 1} / {len(DISCOVERY_TIPS)}"
                    and dialog.region.height <= 12
                    and dialog.region.width <= width - 2
                    and all(
                        widget.region.width > 0 and dialog.region.contains_region(widget.region)
                        for widget in (badge, position, copy, button, close)
                    )
                    and badge.region.y < copy.region.y < button.region.y
                    and copy.region.bottom <= button.region.y
                    and "".join("".join(copy.render_line(y).text.split()) for y in range(copy.content_size.height))
                    == "".join(expected_tip.split())
                ),
            )
            content = copy.render()
            assert isinstance(content, Content)
            assert content.plain == expected_tip
            assert copy.get_selection(Selection(None, None)) == (expected_tip, "\n")
            highlighted = "".join(
                segment.text
                for y in range(copy.content_size.height)
                for segment in copy.render_line(y)
                if segment.style and segment.style.bold and segment.style.reverse
            )
            assert highlighted == TIP_SHORTCUTS[index]
            assert not any(
                segment.style and (segment.style.link or segment.style.blink)
                for y in range(copy.content_size.height)
                for segment in copy.render_line(y)
            )
            # Check native mouse-selection offsets after each wrapped layout.
            await _wait_for_layout(
                pilot,
                lambda: app.screen.get_widget_and_offset_at(copy.region.x + 1, copy.region.y)[1] is not None,
            )

        await assert_tip_presentation(0)
        assert dialog.styles.border_left[0] == "round"
        assert app.screen.focused is close
        assert not close.styles.text_style.reverse
        assert badge.styles.background.a > 0
        assert badge.styles.color != badge.styles.background
        assert str(badge.render()) == "QUICK TIP"
        assert button.can_focus
        assert button.tooltip == "Next discovery tip"
        assert button.active_effect_duration == 0

        assert await pilot.click(button)
        await assert_tip_presentation(1)
        button.focus()
        for step in range(2, len(DISCOVERY_TIPS) + 1):
            await pilot.press("enter")
            await assert_tip_presentation(step % len(DISCOVERY_TIPS))
        # A fresh widget never inherits another widget's position.
        tip.action_next_tip()
        assert DiscoveryTip().current_tip == DISCOVERY_TIPS[0]
        await pilot.press("escape")
        await _wait_for_layout(pilot, lambda: len(app.screen_stack) == 1)


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
async def test_startup_tip_modal_is_opt_in_and_stays_dismissed_across_refresh_and_remount(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, isolated_cli_env: None
) -> None:
    from kolega_code.cli.tui.discovery import DISCOVERY_TIPS, DiscoveryTip
    from kolega_code.cli.tui.startup import StartupEntryWidget

    app = _configured_app(tmp_path, monkeypatch, False)
    async with app.run_test(size=(120, 40)) as pilot:
        assert not isinstance(app.screen, DiscoveryTip)
        app.settings.discovery_tips = True
        app._ensure_startup_entry()
        await _wait_for_layout(pilot, lambda: isinstance(app.screen, DiscoveryTip))
        tip = app.screen
        assert isinstance(tip, DiscoveryTip)
        await _wait_for_layout(pilot, lambda: tip.region.height > 0)
        card = app.query_one(StartupEntryWidget)
        tip.action_next_tip()
        await _wait_for_layout(pilot, lambda: tip.current_tip == DISCOVERY_TIPS[1])
        for _ in range(3):
            app._ensure_startup_entry()
            await pilot.pause()
        assert app.screen is tip
        assert tip.current_tip == DISCOVERY_TIPS[1]
        assert not card.query(DiscoveryTip)
        await pilot.press("escape")
        await _wait_for_layout(pilot, lambda: len(app.screen_stack) == 1)
        app._ensure_startup_entry()
        app._render_conversation()
        await _wait_for_layout(pilot, lambda: app.query_one(StartupEntryWidget) is not card)
        assert not isinstance(app.screen, DiscoveryTip)
        # Discovery stays presentation-only: no model message or persisted tip index.
        assert not app.session.history
        assert all(tip_text not in entry.content for tip_text in DISCOVERY_TIPS for entry in app.conversation_entries)
        assert "tip_index" not in app.settings_store.path.read_text(encoding="utf-8")


@pytest.mark.asyncio
@pytest.mark.parametrize("dismiss", ["escape", "enter", "click"])
async def test_tip_dismissal_restores_focus_and_reset_starts_fresh(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, isolated_cli_env: None, dismiss: str
) -> None:
    from textual.widgets import Button, Collapsible

    from kolega_code.cli.tui.discovery import DISCOVERY_TIPS, DiscoveryTip
    from kolega_code.cli.tui.startup import StartupEntryWidget
    from kolega_code.cli.tui.widgets import ChatComposer

    app = _configured_app(tmp_path, monkeypatch, True)
    async with app.run_test(size=(120, 40)) as pilot:
        await _wait_for_layout(pilot, lambda: isinstance(app.screen, DiscoveryTip))
        tip = app.screen
        assert isinstance(tip, DiscoveryTip)
        tip.action_next_tip()
        composer = app.query_one(ChatComposer)
        close = tip.query_one("#discovery-tip-close", Button)
        await _wait_for_layout(pilot, lambda: tip.focused is close)
        if dismiss == "click":
            assert await pilot.click(close)
        else:
            await pilot.press(dismiss)
        await _wait_for_layout(pilot, lambda: len(app.screen_stack) == 1 and app.screen.focused is composer)
        assert not app.session.history
        composer.load_text("inspect the workspace")
        composer.focus()
        await pilot.press("enter")
        await _wait_for_layout(pilot, lambda: app.agent_worker is None and bool(app.session.history))
        card = app.query_one(StartupEntryWidget)
        card.query_one(Collapsible).collapsed = False
        app._ensure_startup_entry()
        await pilot.pause()
        assert not isinstance(app.screen, DiscoveryTip)
        assert app.config is not None
        await app._build_agent(app.config, rebuild=True)
        await _wait_for_layout(pilot, lambda: app.query_one(StartupEntryWidget) is not card)
        assert not isinstance(app.screen, DiscoveryTip)
        await app._reset_current_thread()
        await _wait_for_layout(pilot, lambda: isinstance(app.screen, DiscoveryTip))
        assert isinstance(app.screen, DiscoveryTip)
        assert app.screen.current_tip == DISCOVERY_TIPS[0]


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
        assert not isinstance(app.screen, DiscoveryTip)
        app._ensure_startup_entry()
        await pilot.pause()
        assert not isinstance(app.screen, DiscoveryTip)
        await app._reset_current_thread()
        await _wait_for_layout(pilot, lambda: isinstance(app.screen, DiscoveryTip))


@pytest.mark.asyncio
@pytest.mark.parametrize("cover", ["settings", "onboarding"])
@pytest.mark.parametrize("submit_before_close", [False, True])
async def test_tip_waits_for_other_modals_and_rechecks_thread_freshness(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cover: str, submit_before_close: bool
) -> None:
    from kolega_code.cli.tui.discovery import DiscoveryTip
    from kolega_code.cli.tui.state import ConversationEntry

    app = _configured_app(tmp_path, monkeypatch, False)
    async with app.run_test(size=(100, 40)) as pilot:
        if cover == "settings":
            app.action_open_settings(category="appearance")
        else:
            config = app.config
            app.config = None
            await app.action_open_onboarding()
            app.config = config
        covered = app.screen
        app.settings.discovery_tips = True
        app._ensure_startup_entry()
        await pilot.pause()
        assert app.screen is covered
        assert len(app.screen_stack) == 2
        if submit_before_close:
            app._add_conversation_entry(ConversationEntry(kind="user", content="a new request"))
        await pilot.press("escape")
        if submit_before_close:
            await _wait_for_layout(pilot, lambda: len(app.screen_stack) == 1)
            assert not isinstance(app.screen, DiscoveryTip)
        else:
            await _wait_for_layout(pilot, lambda: isinstance(app.screen, DiscoveryTip))
            assert len(app.screen_stack) == 2


@pytest.mark.asyncio
async def test_disabling_tips_closes_the_open_modal_without_reopening_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from kolega_code.cli.tui.discovery import DiscoveryTip

    app = _configured_app(tmp_path, monkeypatch, True)
    async with app.run_test(size=(80, 30)) as pilot:
        assert isinstance(app.screen, DiscoveryTip)
        app.settings.discovery_tips = False
        app._ensure_startup_entry()
        await _wait_for_layout(pilot, lambda: len(app.screen_stack) == 1)
        app.settings.discovery_tips = True
        app._ensure_startup_entry()
        await pilot.pause()
        assert len(app.screen_stack) == 1


@pytest.mark.asyncio
async def test_tip_does_not_cover_active_prompts_or_let_them_steal_modal_focus(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from textual.widgets import Button

    from kolega_code.cli.tui.discovery import DiscoveryTip
    from kolega_code.cli.tui.state import PendingQuestion
    from kolega_code.cli.tui.widgets import ActionList

    app = _configured_app(tmp_path, monkeypatch, False)
    async with app.run_test(size=(100, 40)) as pilot:
        question = PendingQuestion(question="Choose?", options=["A", "B"], request_id="tip-focus")
        app._pending_question = question
        app._show_question_options(question.question, question.options)
        app.settings.discovery_tips = True
        await app._maybe_show_discovery_tip()
        assert len(app.screen_stack) == 1
        app._pending_question = None
        app._set_question_actions_visible(False)
        await app._maybe_show_discovery_tip()
        tip = app.screen
        assert isinstance(tip, DiscoveryTip)
        close = tip.query_one("#discovery-tip-close", Button)
        app._pending_question = question
        app._show_question_options(question.question, question.options)
        app._heal_prompt_focus()
        await _wait_for_layout(pilot, lambda: tip.focused is close)
        await pilot.press("escape")
        actions = app.query_one("#question_actions", ActionList)
        await _wait_for_layout(pilot, lambda: len(app.screen_stack) == 1 and app.screen.focused is actions)
        assert app._pending_question is question
        app._pending_question = None
        app._set_question_actions_visible(False)


@pytest.mark.asyncio
@pytest.mark.parametrize("enabled", [False, True])
async def test_discovery_settings_switch_roundtrip_and_discard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, isolated_cli_env: None, enabled: bool
) -> None:
    pytest.importorskip("textual")
    from textual.widgets import Button, Switch

    from kolega_code.cli.tui.discovery import DiscoveryTip
    from kolega_code.cli.tui.settings_screen import ConfirmDiscardSettingsScreen, SettingsScreen

    app = _configured_app(tmp_path, monkeypatch, enabled)
    async with app.run_test(size=(100, 40)) as pilot:
        if enabled:
            await pilot.press("escape")
            await _wait_for_layout(pilot, lambda: len(app.screen_stack) == 1)
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
        if not enabled:
            await _wait_for_layout(pilot, lambda: isinstance(app.screen, DiscoveryTip))
            await pilot.press("escape")
            await _wait_for_layout(pilot, lambda: len(app.screen_stack) == 1)
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
