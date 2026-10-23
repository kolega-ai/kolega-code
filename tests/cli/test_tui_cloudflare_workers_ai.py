"""Cloudflare Workers AI account ID on the Settings → Providers page and in onboarding."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from kolega_code.cli import messages
from kolega_code.cli.session_store import SessionStore
from kolega_code.cli.settings import SettingsStore

from ._app_test_utils import install_fake_agents, stage_provider_api_key, wait_for_layout, wait_for_onboarding_screen
from .test_tui_settings_screens import _configured_app, _open_settings

PROVIDER = "cloudflare_workers_ai"
ACCOUNT_ID = "0123456789abcdef0123456789abcdef"
BASE_URL = f"https://api.cloudflare.com/client/v4/accounts/{ACCOUNT_ID}/ai/v1"


async def _highlight(screen: Any, pilot: Any, provider: str) -> None:
    from textual.widgets import OptionList

    provider_list = screen.query_one("#provider_list", OptionList)
    provider_list.highlighted = provider_list.get_option_index(f"provider_row_{provider}")
    await wait_for_layout(pilot, lambda: screen.credential_provider == provider)


def _row(screen: Any, provider: str) -> str:
    from textual.widgets import OptionList

    provider_list = screen.query_one("#provider_list", OptionList)
    return str(provider_list.get_option_at_index(provider_list.get_option_index(f"provider_row_{provider}")).prompt)


@pytest.mark.asyncio
async def test_account_field_is_shown_only_for_cloudflare(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, isolated_cli_env: None
) -> None:
    pytest.importorskip("textual")
    from textual.widgets import Input

    app, _ = _configured_app(tmp_path, monkeypatch)

    async with app.run_test(size=(100, 40)) as pilot:
        screen = await _open_settings(pilot, app, "providers")
        assert screen.query_one("#provider_account_id_input").display is False

        await _highlight(screen, pilot, PROVIDER)

        assert screen.query_one("#provider_account_id_input").display is True
        assert screen.query_one("#provider_account_id_hint").display is True
        assert (
            screen.query_one("#provider_api_key_input", Input).placeholder == messages.CLOUDFLARE_API_TOKEN_PLACEHOLDER
        )

        await _highlight(screen, pilot, "deepseek")
        assert screen.query_one("#provider_account_id_input").display is False


@pytest.mark.asyncio
async def test_typed_account_id_stages_reports_and_applies_normalized(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, isolated_cli_env: None
) -> None:
    pytest.importorskip("textual")
    from textual.widgets import Input

    app, settings_store = _configured_app(tmp_path, monkeypatch)

    async with app.run_test(size=(100, 40)) as pilot:
        screen = await _open_settings(pilot, app, "providers")
        await stage_provider_api_key(screen, pilot, PROVIDER, "cf-test-token")
        assert "account ID missing" in _row(screen, PROVIDER)

        screen.query_one("#provider_account_id_input", Input).value = ACCOUNT_ID.upper()
        await wait_for_layout(pilot, lambda: "account ID" not in _row(screen, PROVIDER))
        assert screen.dirty is True

        await app._save_settings_from_ui()

        saved = settings_store.load()
        assert saved.cloudflare_account_id == ACCOUNT_ID
        assert saved.get_api_key(PROVIDER) == "cf-test-token"
        assert screen.query_one("#provider_account_id_input", Input).value == ACCOUNT_ID
        await wait_for_layout(pilot, lambda: screen.dirty is False)


@pytest.mark.asyncio
async def test_invalid_account_id_blocks_apply(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, isolated_cli_env: None
) -> None:
    pytest.importorskip("textual")
    from textual.widgets import Input

    app, settings_store = _configured_app(tmp_path, monkeypatch)

    async with app.run_test(size=(100, 40)) as pilot:
        screen = await _open_settings(pilot, app, "providers")
        await _highlight(screen, pilot, PROVIDER)
        screen.query_one("#provider_account_id_input", Input).value = "not-an-account"
        await wait_for_layout(pilot, lambda: screen.dirty is True)

        await app._save_settings_from_ui()

        assert "32-character hex" in str(screen.query_one("#settings_status").render())
        assert settings_store.load().cloudflare_account_id is None
        assert screen.dirty is True


@pytest.mark.asyncio
async def test_test_connection_targets_the_typed_account(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, isolated_cli_env: None
) -> None:
    pytest.importorskip("textual")
    from textual.widgets import Input, Static

    from kolega_code.config import ModelProvider

    app, _ = _configured_app(tmp_path, monkeypatch)
    probes: list[dict[str, Any]] = []

    async def fake_probe(provider, model, **kwargs):
        from kolega_code.cli.model_connection import ModelConnectionResult

        probes.append({"provider": provider, "model": model, **kwargs})
        return ModelConnectionResult(True, f"Connected to {provider.value}/{model}.")

    monkeypatch.setattr("kolega_code.cli.tui.settings_panel.test_model_connection", fake_probe)
    monkeypatch.setenv("CLOUDFLARE_AI_GATEWAY_ID", "gw")

    async with app.run_test(size=(100, 40)) as pilot:
        screen = await _open_settings(pilot, app, "providers")
        await stage_provider_api_key(screen, pilot, PROVIDER, "cf-test-token")

        await app._test_settings_connection()
        assert probes == []
        assert "CLOUDFLARE_ACCOUNT_ID" in str(screen.query_one("#provider_credential_status", Static).render())

        screen.query_one("#provider_account_id_input", Input).value = ACCOUNT_ID
        await pilot.pause()
        await app._test_settings_connection()

        assert probes[-1]["provider"] == ModelProvider.CLOUDFLARE_WORKERS_AI
        assert probes[-1]["model"] == "@cf/zai-org/glm-5.3"
        assert probes[-1]["api_key"] == "cf-test-token"
        assert probes[-1]["base_url"] == BASE_URL
        assert probes[-1]["default_headers"] == {"cf-aig-gateway-id": "gw"}


@pytest.mark.asyncio
async def test_onboarding_requires_and_saves_the_account_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, isolated_cli_env: None
) -> None:
    pytest.importorskip("textual")
    from textual.widgets import Input, Select

    from kolega_code.cli.app import KolegaCodeApp

    install_fake_agents(monkeypatch)
    project = tmp_path / "project"
    project.mkdir()
    state_dir = tmp_path / "state"
    store = SessionStore(state_dir)
    settings_store = SettingsStore(state_dir)
    app = KolegaCodeApp(
        project_path=project,
        mode="code",
        store=store,
        settings_store=settings_store,
        session=store.create(project, "code", {}),
        startup_config_error="No provider/model configured.",
    )

    async with app.run_test(size=(100, 40)) as pilot:
        screen = await wait_for_onboarding_screen(app, pilot)
        await screen._continue()
        assert screen.step_index == 1
        assert screen.query_one("#onboarding_account_id_panel").display is False

        screen.query_one("#onboarding_provider", Select).value = PROVIDER
        await wait_for_layout(pilot, lambda: screen.query_one("#onboarding_account_id_panel").display is True)
        screen.query_one("#onboarding_api_key", Input).value = "cf-test-token"
        await pilot.pause()

        await screen._continue()
        assert screen.step_index == 1
        assert "CLOUDFLARE_ACCOUNT_ID" in str(screen.query_one("#onboarding_status").render())

        screen.query_one("#onboarding_account_id", Input).value = "bad"
        await screen._continue()
        assert screen.step_index == 1
        assert "32-character hex" in str(screen.query_one("#onboarding_status").render())

        screen.query_one("#onboarding_account_id", Input).value = ACCOUNT_ID.upper()
        await screen._continue()
        assert screen.step_index == 2
        await screen._continue()
        await screen._finish()
        await pilot.pause()

        saved = settings_store.load()
        assert saved.active_provider == PROVIDER
        assert saved.active_model == "@cf/zai-org/glm-5.3"
        assert saved.cloudflare_account_id == ACCOUNT_ID
        assert saved.get_api_key(PROVIDER) == "cf-test-token"
        assert app.config is not None and app.config.cloudflare_account_id == ACCOUNT_ID
