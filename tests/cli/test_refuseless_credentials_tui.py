from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock

import pytest

from kolega_code.cli.model_connection import ModelConnectionResult
from kolega_code.cli.provider_registry import UI_DEFAULT_MODEL, UI_DEFAULT_PROVIDER
from kolega_code.config import AGENT_ROLE_BY_NAME, ModelProvider

from ._app_test_utils import install_fake_agents, wait_for_layout, wait_for_onboarding_screen
from .test_tui_settings_screens import _configured_app, _open_settings

if TYPE_CHECKING:
    from textual.pilot import Pilot
    from textual.screen import Screen

    from kolega_code.cli.tui.settings_screen import SettingsScreen


REFUSELESS_MODELS = (
    "glm-5.3-abliterated",
    "glm-5.3-flash-abliterated",
    "deepseek-v4.1-flash-abliterated",
    "deepseek-v4.1-flash-fast",
    "minimax-m3.1-flash",
)


def _select_options(screen: Screen, widget_id: str) -> list[tuple[str, str]]:
    from textual.widgets import Select

    select = screen.query_one(f"#{widget_id}", Select)
    return [(str(label), str(value)) for label, value in select._options if value is not Select.NULL]


def _select_values(screen: Screen, widget_id: str) -> tuple[str, ...]:
    return tuple(value for _label, value in _select_options(screen, widget_id))


def _efforts(model: str) -> tuple[str, ...]:
    return ("off", "low", "high", "max") if model.startswith("deepseek-") else ("low", "high", "max")


async def _wait_for_selects(pilot: Pilot, screen: Screen, values: dict[str, str]) -> None:
    from textual.widgets import Select

    refreshed = False

    def after_refresh() -> None:
        nonlocal refreshed
        refreshed = True

    screen.call_after_refresh(after_refresh)
    await wait_for_layout(
        pilot,
        lambda: (
            refreshed
            and all(
                bool(screen.query(f"#{widget_id}")) and screen.query_one(f"#{widget_id}", Select).value == value
                for widget_id, value in values.items()
            )
        ),
    )


async def _highlight_provider(screen: SettingsScreen, pilot: Pilot, provider: str) -> None:
    from textual.widgets import Input, OptionList

    await wait_for_layout(
        pilot,
        lambda: bool(screen.query("#provider_list")) and bool(screen.query("#provider_api_key_input")),
    )
    provider_list = screen.query_one("#provider_list", OptionList)
    await wait_for_layout(pilot, lambda: provider_list.option_count > 0)
    provider_list.highlighted = provider_list.get_option_index(f"provider_row_{provider}")
    key_input = screen.query_one("#provider_api_key_input", Input)
    await wait_for_layout(
        pilot,
        lambda: screen.credential_provider == provider and key_input.display and key_input.region.width > 0,
    )


async def _open_credential_editor(screen: SettingsScreen, pilot: Pilot, provider: str) -> None:
    from textual.widgets import Input

    await wait_for_layout(pilot, lambda: bool(screen.query("#provider_api_key_input")))
    screen.owner._switch_provider_credential(provider)
    key_input = screen.query_one("#provider_api_key_input", Input)
    await wait_for_layout(
        pilot,
        lambda: screen.credential_provider == provider and key_input.display and key_input.region.width > 0,
    )


@pytest.mark.asyncio
async def test_settings_providers_row_includes_refuseless(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, isolated_cli_env: None
) -> None:
    pytest.importorskip("textual")
    from textual.widgets import OptionList

    app, _ = _configured_app(tmp_path, monkeypatch)
    async with app.run_test(size=(120, 42)) as pilot:
        screen = await _open_settings(pilot, app, "providers")
        await wait_for_layout(pilot, lambda: bool(screen.query("#provider_list")))
        provider_list = screen.query_one("#provider_list", OptionList)
        await wait_for_layout(pilot, lambda: provider_list.option_count > 0)
        row_ids = [provider_list.get_option_at_index(index).id for index in range(provider_list.option_count)]
        assert "provider_row_refuseless" in row_ids
        row = provider_list.get_option("provider_row_refuseless")
        assert "Refuseless" in str(row.prompt)
        assert "missing" in str(row.prompt)


@pytest.mark.asyncio
async def test_refuseless_credential_editor_stages_a_password_without_persisting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, isolated_cli_env: None
) -> None:
    pytest.importorskip("textual")
    from textual.widgets import Input, Static

    app, settings_store = _configured_app(tmp_path, monkeypatch)
    key = "abl_FAKE_staged_refuseless_key"

    async with app.run_test(size=(120, 42)) as pilot:
        screen = await _open_settings(pilot, app, "providers")
        await _open_credential_editor(screen, pilot, "refuseless")
        key_input = screen.query_one("#provider_api_key_input", Input)
        assert key_input.password is True
        assert key_input.value == ""

        key_input.value = key
        await wait_for_layout(pilot, lambda: screen.pending_api_keys.get("refuseless") == key and screen.dirty)
        status_text = str(screen.query_one("#provider_credential_status", Static).render())
        assert "present in local settings" in status_text
        assert key not in status_text
        assert settings_store.load().get_api_key("refuseless") is None
        assert app.settings.get_api_key("refuseless") is None

        await _open_credential_editor(screen, pilot, UI_DEFAULT_PROVIDER)
        assert key_input.value == ""
        await _open_credential_editor(screen, pilot, "refuseless")
        assert key_input.value == key
        assert screen.pending_api_keys == {"refuseless": key}


@pytest.mark.asyncio
async def test_apply_refuseless_key_keeps_the_existing_active_provider_and_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, isolated_cli_env: None
) -> None:
    pytest.importorskip("textual")
    from textual.widgets import Button, Input, Select

    app, settings_store = _configured_app(tmp_path, monkeypatch)
    key = "abl_FAKE_apply_refuseless_key"

    async with app.run_test(size=(120, 42)) as pilot:
        screen = await _open_settings(pilot, app, "providers")
        await _open_credential_editor(screen, pilot, "refuseless")
        key_input = screen.query_one("#provider_api_key_input", Input)
        key_input.value = key
        apply_button = screen.query_one("#save_settings", Button)
        await wait_for_layout(
            pilot,
            lambda: (
                screen.pending_api_keys.get("refuseless") == key
                and not apply_button.disabled
                and apply_button.region.width > 0
            ),
        )
        assert await pilot.click("#save_settings")
        await wait_for_layout(
            pilot,
            lambda: not screen.dirty and key_input.value == "" and apply_button.disabled,
        )

        saved = settings_store.load()
        assert saved.get_api_key("refuseless") == key
        assert saved.get_api_key(UI_DEFAULT_PROVIDER) == "stored-key"
        assert saved.active_provider == UI_DEFAULT_PROVIDER
        assert saved.active_model == UI_DEFAULT_MODEL
        assert screen.pending_api_keys == {}
        assert str(screen.query_one("#provider_select", Select).value) == UI_DEFAULT_PROVIDER
        assert str(screen.query_one("#model_select", Select).value) == UI_DEFAULT_MODEL
        assert app.config is not None
        assert app.config.long_context_config.provider == ModelProvider(UI_DEFAULT_PROVIDER)
        assert app.config.long_context_config.model == UI_DEFAULT_MODEL


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("credential_source", "expected_status"),
    [
        ("missing", "missing"),
        ("saved", "present in local settings"),
        ("environment", "present via REFUSELESS_API_KEY"),
    ],
)
async def test_refuseless_credential_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    isolated_cli_env: None,
    credential_source: str,
    expected_status: str,
) -> None:
    pytest.importorskip("textual")
    from textual.widgets import Button, Input, Static

    key = "abl_FAKE_status_refuseless_key"
    if credential_source == "environment":
        monkeypatch.setenv("REFUSELESS_API_KEY", key)
    app, settings_store = _configured_app(tmp_path, monkeypatch)
    if credential_source == "saved":
        app.settings.set_api_key("refuseless", key)
        settings_store.save(app.settings)

    async with app.run_test(size=(120, 42)) as pilot:
        screen = await _open_settings(pilot, app, "providers")
        await _open_credential_editor(screen, pilot, "refuseless")
        status_text = str(screen.query_one("#provider_credential_status", Static).render())
        assert expected_status in status_text
        assert key not in status_text
        key_input = screen.query_one("#provider_api_key_input", Input)
        assert key_input.password is True
        assert key_input.value == ""
        assert screen.query_one("#provider_remove_api_key", Button).disabled is (credential_source != "saved")
        assert screen.pending_api_keys == {}
        assert screen.dirty is False


@pytest.mark.asyncio
async def test_refuseless_connection_probe_uses_highlighted_provider_default_and_staged_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, isolated_cli_env: None
) -> None:
    pytest.importorskip("textual")
    from textual.widgets import Button, Input, Static

    app, settings_store = _configured_app(tmp_path, monkeypatch)
    key = "abl_FAKE_probe_refuseless_key"
    message = "Connected to refuseless/glm-5.3-abliterated."
    probe = AsyncMock(return_value=ModelConnectionResult(True, message))
    monkeypatch.setattr("kolega_code.cli.tui.settings_panel.test_model_connection", probe)

    async with app.run_test(size=(120, 42)) as pilot:
        screen = await _open_settings(pilot, app, "providers")
        await _highlight_provider(screen, pilot, "refuseless")
        screen.query_one("#provider_api_key_input", Input).value = key
        test_button = screen.query_one("#provider_test_connection", Button)
        await wait_for_layout(
            pilot,
            lambda: screen.pending_api_keys.get("refuseless") == key and test_button.region.width > 0,
        )
        test_button.scroll_visible(animate=False)
        await wait_for_layout(
            pilot,
            lambda: (
                not test_button.disabled
                and screen.get_widget_at(test_button.region.x + 1, test_button.region.y + 1)[0] is test_button
            ),
        )
        assert await pilot.click("#provider_test_connection", offset=(1, 1))
        await wait_for_layout(
            pilot,
            lambda: (
                probe.await_count == 1
                and not test_button.disabled
                and message in str(screen.query_one("#provider_credential_status", Static).render())
            ),
        )

        probe.assert_awaited_once()
        assert probe.await_args is not None
        assert probe.await_args.args == (ModelProvider.REFUSELESS, "glm-5.3-abliterated")
        assert probe.await_args.kwargs["api_key"] == key
        assert settings_store.load().get_api_key("refuseless") is None
        assert app.settings.active_provider == UI_DEFAULT_PROVIDER
        assert app.settings.active_model == UI_DEFAULT_MODEL


@pytest.mark.asyncio
async def test_onboarding_refuseless_key_and_real_model_choices_persist_only_on_finish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, isolated_cli_env: None
) -> None:
    pytest.importorskip("textual")
    from textual.widgets import Input, Select, Static

    from kolega_code.cli.app import KolegaCodeApp
    from kolega_code.cli.session_store import SessionStore
    from kolega_code.cli.settings import SettingsStore

    install_fake_agents(monkeypatch)
    project = tmp_path / "project"
    project.mkdir()
    state_dir = tmp_path / "state"
    settings_store = SettingsStore(state_dir)
    store = SessionStore(state_dir)
    session = store.create(project, "code", {})
    app = KolegaCodeApp(
        project_path=project,
        mode="code",
        store=store,
        settings_store=settings_store,
        session=session,
    )
    key = "abl_FAKE_onboarding_refuseless_key"

    async with app.run_test(size=(120, 42)) as pilot:
        screen = await wait_for_onboarding_screen(app, pilot)
        assert ("Refuseless", "refuseless") in screen._api_provider_options()
        await screen._continue()
        await wait_for_layout(
            pilot,
            lambda: screen.step_index == 1 and screen.query_one("#onboarding_api_panel").region.width > 0,
        )
        assert str(screen.query_one("#onboarding_auth_method", Select).value) == "api"
        screen.query_one("#onboarding_provider", Select).value = "refuseless"
        key_input = screen.query_one("#onboarding_api_key", Input)
        await wait_for_layout(
            pilot,
            lambda: (
                "missing" in str(screen.query_one("#onboarding_key_status", Static).render())
                and key_input.region.width > 0
            ),
        )
        assert key_input.password is True
        assert key_input.value == ""
        assert screen._collect_connection() is False
        assert "Enter an API key" in str(screen.query_one("#onboarding_status", Static).render())

        key_input.value = key
        assert screen._collect_connection() is True
        assert screen.draft.get_api_key("refuseless") == key
        assert screen.draft.active_provider == "refuseless"
        assert screen.step_index == 1
        assert screen.query_one("#onboarding_step_model").display is False
        assert app.settings.get_api_key("refuseless") is None
        assert settings_store.load().get_api_key("refuseless") is None

        await screen._continue()
        await wait_for_layout(
            pilot,
            lambda: screen.step_index == 2 and screen.query_one("#onboarding_model", Select).region.width > 0,
        )
        await _wait_for_selects(
            pilot,
            screen,
            {"onboarding_model": REFUSELESS_MODELS[0], "onboarding_effort": "low"},
        )
        assert _select_values(screen, "onboarding_model") == REFUSELESS_MODELS
        for model in REFUSELESS_MODELS:
            screen.query_one("#onboarding_model", Select).value = model
            await _wait_for_selects(pilot, screen, {"onboarding_model": model})
            await wait_for_layout(pilot, lambda: _select_values(screen, "onboarding_effort") == _efforts(model))
            assert _select_values(screen, "onboarding_effort") == _efforts(model)
            if model.startswith("deepseek-"):
                assert ("Off", "off") in _select_options(screen, "onboarding_effort")
            for effort in _efforts(model):
                screen.query_one("#onboarding_effort", Select).value = effort
                await _wait_for_selects(pilot, screen, {"onboarding_effort": effort})
                assert screen._collect_model() is True
                assert screen.draft.active_model == model
                assert screen.draft.active_thinking_effort == effort
            assert settings_store.load().active_provider is None
            assert settings_store.load().get_api_key("refuseless") is None

        selected_model = "deepseek-v4.1-flash-fast"
        screen.query_one("#onboarding_model", Select).value = selected_model
        await _wait_for_selects(pilot, screen, {"onboarding_model": selected_model})
        await wait_for_layout(pilot, lambda: "off" in _select_values(screen, "onboarding_effort"))
        screen.query_one("#onboarding_effort", Select).value = "off"
        await _wait_for_selects(pilot, screen, {"onboarding_effort": "off"})
        await screen._continue()
        await wait_for_layout(
            pilot,
            lambda: screen.step_index == 3 and screen.query_one("#onboarding_summary", Static).region.width > 0,
        )
        summary = str(screen.query_one("#onboarding_summary", Static).render())
        assert selected_model in summary
        assert "Thinking effort: off" in summary
        assert key not in summary
        assert settings_store.load().get_api_key("refuseless") is None
        await screen._continue()
        await wait_for_layout(pilot, lambda: app._onboarding_screen is None and app.screen is not screen)

        saved = settings_store.load()
        assert saved.get_api_key("refuseless") == key
        assert saved.active_provider == "refuseless"
        assert saved.active_model == selected_model
        assert saved.active_thinking_effort == "off"
        assert app.config is not None
        assert app.config.long_context_config.provider is ModelProvider.REFUSELESS
        assert app.config.long_context_config.model == selected_model
        assert app.config.long_context_config.thinking_effort == "off"


@pytest.mark.asyncio
@pytest.mark.parametrize("model", REFUSELESS_MODELS)
async def test_refuseless_settings_round_trip_main_fast_and_every_role_with_vision_filtering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, isolated_cli_env: None, model: str
) -> None:
    pytest.importorskip("textual")
    from textual.widgets import Select

    app, settings_store = _configured_app(tmp_path, monkeypatch, extra_key_providers=("refuseless",))
    effort = "off" if model.startswith("deepseek-") else "max"
    model_index = REFUSELESS_MODELS.index(model)
    fast_model = REFUSELESS_MODELS[(model_index + 1) % len(REFUSELESS_MODELS)]
    roles = {role.value: agent_name for agent_name, role in AGENT_ROLE_BY_NAME.items()}
    role_models = {
        role: REFUSELESS_MODELS[(model_index + index + 2) % len(REFUSELESS_MODELS)] for index, role in enumerate(roles)
    }
    if role_models["browser"] == REFUSELESS_MODELS[0]:
        role_models["browser"] = REFUSELESS_MODELS[1]
    role_efforts = {
        role: "off" if role_model.startswith("deepseek-") else "max" for role, role_model in role_models.items()
    }

    async with app.run_test(size=(140, 44)) as pilot:
        screen = await _open_settings(pilot, app)
        await _wait_for_selects(pilot, screen, {"provider_select": UI_DEFAULT_PROVIDER})
        screen.query_one("#provider_select", Select).value = "refuseless"
        await _wait_for_selects(
            pilot,
            screen,
            {"provider_select": "refuseless", "model_select": REFUSELESS_MODELS[0]},
        )
        assert _select_values(screen, "model_select") == REFUSELESS_MODELS
        screen.query_one("#model_select", Select).value = model
        await _wait_for_selects(pilot, screen, {"model_select": model})
        await wait_for_layout(pilot, lambda: _select_values(screen, "thinking_effort_select") == _efforts(model))
        assert _select_values(screen, "thinking_effort_select") == _efforts(model)
        if effort == "off":
            assert ("Off", "off") in _select_options(screen, "thinking_effort_select")
        screen.query_one("#thinking_effort_select", Select).value = effort
        await _wait_for_selects(pilot, screen, {"thinking_effort_select": effort})

        screen.query_one("#slot_provider_fast", Select).value = "refuseless"
        await _wait_for_selects(pilot, screen, {"slot_model_fast": REFUSELESS_MODELS[0]})
        assert _select_values(screen, "slot_model_fast") == REFUSELESS_MODELS
        screen.query_one("#slot_model_fast", Select).value = fast_model
        await _wait_for_selects(pilot, screen, {"slot_model_fast": fast_model})

        screen._show_category("agents")
        for role, role_model in role_models.items():
            screen.query_one(f"#am_provider_{role}", Select).value = "refuseless"
            default_model = REFUSELESS_MODELS[1] if role == "browser" else REFUSELESS_MODELS[0]
            await _wait_for_selects(pilot, screen, {f"am_model_{role}": default_model, f"am_effort_{role}": "low"})
            expected_models = REFUSELESS_MODELS[1:] if role == "browser" else REFUSELESS_MODELS
            assert _select_values(screen, f"am_model_{role}") == expected_models
            screen.query_one(f"#am_model_{role}", Select).value = role_model
            await _wait_for_selects(pilot, screen, {f"am_model_{role}": role_model})
            await wait_for_layout(pilot, lambda: _select_values(screen, f"am_effort_{role}") == _efforts(role_model))
            assert _select_values(screen, f"am_effort_{role}") == _efforts(role_model)
            if role_efforts[role] == "off":
                assert ("Off", "off") in _select_options(screen, f"am_effort_{role}")
            screen.query_one(f"#am_effort_{role}", Select).value = role_efforts[role]
            await _wait_for_selects(pilot, screen, {f"am_effort_{role}": role_efforts[role]})

        await app._save_settings_from_ui()
        await wait_for_layout(pilot, lambda: not screen.dirty)
        saved = settings_store.load()
        assert saved.active_provider == "refuseless"
        assert saved.active_model == model
        assert saved.active_thinking_effort == effort
        assert saved.get_api_key("refuseless") == "stored-key"
        assert saved.get_api_key(UI_DEFAULT_PROVIDER) == "stored-key"
        assert saved.get_model_slot("fast") == {"provider": "refuseless", "model": fast_model}
        assert app.config is not None
        assert app.config.long_context_config.provider is ModelProvider.REFUSELESS
        assert app.config.long_context_config.model == model
        assert app.config.long_context_config.thinking_effort == effort
        assert app.config.fast_config.provider is ModelProvider.REFUSELESS
        assert app.config.fast_config.model == fast_model
        for role, role_model in role_models.items():
            assert saved.get_agent_model(role) == {
                "provider": "refuseless",
                "model": role_model,
                "thinking_effort": role_efforts[role],
            }
            selected = app.config.model_config_for_agent(roles[role])
            assert selected.provider is ModelProvider.REFUSELESS
            assert selected.model == role_model
            assert selected.thinking_effort == role_efforts[role]

        screen.dismiss()
        await wait_for_layout(pilot, lambda: app.screen is not screen and app._settings_screen is None)
        restored = await _open_settings(pilot, app)
        await _wait_for_selects(
            pilot,
            restored,
            {
                "provider_select": "refuseless",
                "model_select": model,
                "thinking_effort_select": effort,
                "slot_provider_fast": "refuseless",
                "slot_model_fast": fast_model,
                **{f"am_provider_{role}": "refuseless" for role in roles},
                **{f"am_model_{role}": role_model for role, role_model in role_models.items()},
                **{f"am_effort_{role}": role_efforts[role] for role in roles},
            },
        )
        assert restored.dirty is False
