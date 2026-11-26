"""Config, CLI, env, settings, and TUI coverage for context-usage reminders."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from kolega_code.agent.coder import CoderAgent
from kolega_code.cli.config import (
    CONTEXT_USAGE_REMINDERS_ENV,
    CliConfigError,
    CliConfigOverrides,
    _context_usage_reminders_enabled,
    build_agent_config,
)
from kolega_code.cli.settings import CliSettings, SettingsStore

from ._app_test_utils import build_test_config, install_fake_agents, open_settings_screen
from .test_app_lsp_settings_status import _static_text


class RecordingCoderAgent(CoderAgent):
    instances: list["RecordingCoderAgent"] = []

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        RecordingCoderAgent.instances.append(self)

    async def process_message_stream(self, message, attachments=None):
        yield {"type": "response", "content": "done", "complete": True, "uuid": "response-1"}


def _setup_ask(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from kolega_code.cli import main as main_module

    RecordingCoderAgent.instances = []
    monkeypatch.setattr(main_module, "CoderAgent", RecordingCoderAgent)
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("KOLEGA_CODE_PROVIDER", "anthropic")
    monkeypatch.setenv("KOLEGA_CODE_MODEL", "claude-opus-5")
    return main_module, project


def _make_app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, overrides=None, settings=None):
    pytest.importorskip("textual")

    from kolega_code.cli.app import KolegaCodeApp
    from kolega_code.cli.config import config_summary
    from kolega_code.cli.session_store import SessionStore

    install_fake_agents(monkeypatch)

    project = tmp_path / "project"
    project.mkdir(parents=True)
    config = build_test_config(project)
    store = SessionStore(tmp_path / "state")
    if settings is not None:
        SettingsStore(tmp_path / "state").save(settings)
    session = store.create(project, "code", config_summary(config))
    return KolegaCodeApp(
        project_path=project, config=config, mode="code", store=store, session=session, overrides=overrides
    )


async def _wait_for_init(screen, pilot, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while screen._initializing and time.monotonic() < deadline:
        await pilot.pause(0.05)
    assert not screen._initializing


def test_context_usage_reminders_resolution_precedence() -> None:
    settings_on = CliSettings(context_usage_reminders_enabled=True)
    settings_off = CliSettings(context_usage_reminders_enabled=False)
    env_off = {CONTEXT_USAGE_REMINDERS_ENV: "off"}

    # flag > env > settings > default (True)
    assert _context_usage_reminders_enabled(env_off, settings_off, "on") is True
    assert _context_usage_reminders_enabled(env_off, settings_on, None) is False
    assert _context_usage_reminders_enabled({}, settings_on, None) is True
    assert _context_usage_reminders_enabled({}, settings_off, None) is False
    assert _context_usage_reminders_enabled({}, None, None) is True
    assert _context_usage_reminders_enabled({CONTEXT_USAGE_REMINDERS_ENV: "  "}, settings_off, None) is False


@pytest.mark.parametrize("raw", ["sideways", "yes", ""])
def test_context_usage_reminders_rejects_invalid_flag(raw: str) -> None:
    with pytest.raises(CliConfigError, match="context usage reminders"):
        _context_usage_reminders_enabled({}, None, raw)


def test_context_usage_reminders_rejects_invalid_env() -> None:
    with pytest.raises(CliConfigError, match="context usage reminders"):
        _context_usage_reminders_enabled({CONTEXT_USAGE_REMINDERS_ENV: "maybe"}, None, None)


def test_build_agent_config_carries_context_usage_reminders_enabled(tmp_path: Path, isolated_cli_env: None) -> None:
    project = tmp_path / "project"
    project.mkdir()
    settings = CliSettings(
        active_provider="anthropic",
        active_model="claude-opus-5",
        api_keys={"anthropic": "test-key"},
        context_usage_reminders_enabled=False,
    )
    cfg_from_settings = build_agent_config(project, CliConfigOverrides(), settings=settings)
    assert cfg_from_settings.context_usage_reminders_enabled is False

    cfg_forced_on = build_agent_config(
        project, CliConfigOverrides(context_usage_reminders_mode="on"), settings=settings
    )
    assert cfg_forced_on.context_usage_reminders_enabled is True


def test_ask_context_usage_reminders_default_flag_env_and_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, isolated_cli_env: None
) -> None:
    main_module, project = _setup_ask(tmp_path, monkeypatch)

    assert main_module.main(["ask", "hi", "--project", str(project)]) == 0
    assert RecordingCoderAgent.instances[-1].context_usage_reminders_enabled is True

    assert main_module.main(["ask", "hi", "--project", str(project), "--context-usage-reminders", "off"]) == 0
    assert RecordingCoderAgent.instances[-1].context_usage_reminders_enabled is False

    store = SettingsStore(tmp_path / "state")
    saved = store.load()
    saved.context_usage_reminders_enabled = False
    store.save(saved)

    assert main_module.main(["ask", "hi", "--project", str(project)]) == 0
    assert RecordingCoderAgent.instances[-1].context_usage_reminders_enabled is False

    assert main_module.main(["ask", "hi", "--project", str(project), "--context-usage-reminders", "on"]) == 0
    assert RecordingCoderAgent.instances[-1].context_usage_reminders_enabled is True

    monkeypatch.setenv(CONTEXT_USAGE_REMINDERS_ENV, "off")
    assert main_module.main(["ask", "hi", "--project", str(project)]) == 0
    assert RecordingCoderAgent.instances[-1].context_usage_reminders_enabled is False


@pytest.mark.asyncio
async def test_tui_settings_context_usage_reminders_restore_collect_and_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, isolated_cli_env: None
) -> None:
    from textual.widgets import Select, Static

    app = _make_app(tmp_path, monkeypatch, settings=CliSettings(context_usage_reminders_enabled=False))
    async with app.run_test() as pilot:
        screen = await open_settings_screen(app, pilot, "tools")
        await _wait_for_init(screen, pilot)
        select = screen.query_one("#context_usage_reminders_select", Select)
        assert str(select.value) == "false"
        assert "forced" not in _static_text(screen.query_one("#context_usage_reminders_status", Static))

        select.value = "true"
        app._collect_context_usage_reminders_from_ui()
        assert app.settings.context_usage_reminders_enabled is True

        select.value = "false"
        app._collect_context_usage_reminders_from_ui()
        assert app.settings.context_usage_reminders_enabled is False


@pytest.mark.asyncio
async def test_tui_settings_context_usage_reminders_reports_forced_flag_and_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, isolated_cli_env: None
) -> None:
    from textual.widgets import Static

    app_flag = _make_app(
        tmp_path / "flag",
        monkeypatch,
        overrides=CliConfigOverrides(context_usage_reminders_mode="off"),
    )
    async with app_flag.run_test() as pilot:
        screen = await open_settings_screen(app_flag, pilot, "tools")
        await _wait_for_init(screen, pilot)
        text = _static_text(screen.query_one("#context_usage_reminders_status", Static))
        assert "forced off" in text
        assert "--context-usage-reminders" in text

    monkeypatch.setenv(CONTEXT_USAGE_REMINDERS_ENV, "off")
    app_env = _make_app(tmp_path / "env", monkeypatch)
    async with app_env.run_test() as pilot:
        screen = await open_settings_screen(app_env, pilot, "tools")
        await _wait_for_init(screen, pilot)
        text = _static_text(screen.query_one("#context_usage_reminders_status", Static))
        assert "forced off" in text
        assert CONTEXT_USAGE_REMINDERS_ENV in text
