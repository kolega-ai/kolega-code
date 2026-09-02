"""GatewayConfig resolution: settings.json storage, defaults, and precedence."""

from pathlib import Path

import pytest

from kolega_code.cli.settings import CliSettings, SettingsStore
from kolega_code.gateway.config import (
    DEFAULT_EDIT_THROTTLE_SECONDS,
    DEFAULT_PAIRING_TTL_SECONDS,
    DEFAULT_REQUEST_TIMEOUT_SECONDS,
    default_gateway_project,
    load_gateway_config,
)

TEST_BOT_TOKEN = "123:fake-bot-token-for-tests-only"


def save_settings(
    tmp_path: Path,
    *,
    gateway: dict | None = None,
    token: str | None = None,
    stt: dict | None = None,
) -> Path:
    state_dir = tmp_path / "state"
    settings = CliSettings(telegram_bot_token=token, gateway=gateway or {})
    if stt:
        for key, value in stt.items():
            setattr(settings, key, value)
    SettingsStore(root=state_dir).save(settings)
    return state_dir


def test_defaults(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOME", "/home/tester")
    config = load_gateway_config(state_dir=tmp_path / "state")
    assert config.adapter == "echo"
    assert config.permission_mode == "ask"
    assert config.allowed_users == ()
    assert config.group_ids == ()
    assert config.pairing_enabled is False
    assert config.pairing_code_ttl_seconds == DEFAULT_PAIRING_TTL_SECONDS
    assert config.request_timeout_seconds == DEFAULT_REQUEST_TIMEOUT_SECONDS
    assert config.edit_throttle_seconds == DEFAULT_EDIT_THROTTLE_SECONDS
    assert config.max_sessions == 50
    assert config.session_idle_ttl_seconds == 3600.0
    assert config.telegram_token is None
    assert config.telegram_proxy is None
    assert config.stt_enabled is False
    assert config.stt_provider == "groq"
    assert config.stt_model is None
    assert config.stt_api_key is None
    assert config.project_path == Path("/home/tester/kolega-code-workspace")


def test_default_project_is_home_workspace(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOME", "/home/tester")
    assert default_gateway_project() == Path("/home/tester/kolega-code-workspace")


def test_stored_gateway_section_maps_onto_config(tmp_path: Path) -> None:
    state_dir = save_settings(
        tmp_path,
        gateway={
            "adapter": "telegram",
            "project": str(tmp_path / "checkout"),
            "allowed_users": ["111", "222"],
            "group_ids": ["-1001"],
            "pairing_enabled": True,
            "pairing_code_ttl_seconds": 1800,
            "permission_mode": "auto",
            "request_timeout_seconds": 120.0,
            "max_sessions": 3,
            "session_idle_ttl_seconds": 90.0,
            "edit_throttle_seconds": 0.5,
            "telegram_proxy": "http://127.0.0.1:3128",
        },
        token=TEST_BOT_TOKEN,
    )
    config = load_gateway_config(state_dir=state_dir)
    assert config.adapter == "telegram"
    assert config.project_path == (tmp_path / "checkout").resolve()
    assert config.allowed_users == ("111", "222")
    assert config.group_ids == ("-1001",)
    assert config.pairing_enabled is True
    assert config.pairing_code_ttl_seconds == 1800.0
    assert config.permission_mode == "auto"
    assert config.request_timeout_seconds == 120.0
    assert config.max_sessions == 3
    assert config.session_idle_ttl_seconds == 90.0
    assert config.edit_throttle_seconds == 0.5
    assert config.stt_enabled is False
    assert config.stt_provider == "groq"
    assert config.telegram_token == TEST_BOT_TOKEN
    assert config.telegram_proxy == "http://127.0.0.1:3128"


def test_top_level_stt_settings_apply(tmp_path: Path) -> None:
    state_dir = save_settings(
        tmp_path,
        stt={"stt_enabled": True, "stt_provider": "groq", "stt_model": "whisper-large-v3"},
    )
    config = load_gateway_config(state_dir=state_dir)
    assert config.stt_enabled is True
    assert config.stt_provider == "groq"
    assert config.stt_model == "whisper-large-v3"


def test_groq_stt_uses_the_stored_groq_api_key(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    settings = CliSettings(stt_enabled=True, stt_provider="groq", api_keys={"groq": "gsk_stored"})
    SettingsStore(root=state_dir).save(settings)
    config = load_gateway_config(state_dir=state_dir)
    assert config.stt_enabled is True
    assert config.stt_provider == "groq"
    assert config.stt_model is None  # provider default applies downstream
    assert config.stt_api_key == "gsk_stored"


def test_groq_stt_api_key_env_beats_the_stored_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "gsk_env")
    state_dir = tmp_path / "state"
    settings = CliSettings(stt_provider="groq", api_keys={"groq": "gsk_stored"})
    SettingsStore(root=state_dir).save(settings)
    config = load_gateway_config(state_dir=state_dir)
    assert config.stt_api_key == "gsk_env"


def test_unknown_stt_provider_falls_back_to_groq(tmp_path: Path) -> None:
    state_dir = save_settings(tmp_path, stt={"stt_provider": "not-a-provider"})
    config = load_gateway_config(state_dir=state_dir)
    assert config.stt_provider == "groq"


def test_cli_flags_override_stored_values(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOME", "/home/tester")
    state_dir = save_settings(
        tmp_path,
        gateway={"adapter": "echo", "project": str(tmp_path / "stored")},
    )
    config = load_gateway_config(
        state_dir=state_dir,
        adapter="telegram",
        project=Path("/flag/project"),
    )
    assert config.adapter == "telegram"
    assert config.project_path == Path("/flag/project").resolve()
    # Stored values still apply when no flag is given.
    config = load_gateway_config(state_dir=state_dir)
    assert config.adapter == "echo"
    assert config.project_path == (tmp_path / "stored").resolve()


def test_state_dir_precedence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KOLEGA_CODE_STATE_DIR", str(tmp_path / "shared"))
    assert load_gateway_config().state_dir == (tmp_path / "shared").resolve()
    assert load_gateway_config(state_dir=Path("/param/state")).state_dir == Path("/param/state").resolve()


def test_corrupt_settings_fall_back_to_defaults(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    state_dir.mkdir(parents=True)
    (state_dir / "settings.json").write_text("{not valid json", encoding="utf-8")
    config = load_gateway_config(state_dir=state_dir)
    assert config.adapter == "echo"
    assert config.telegram_token is None


def test_invalid_stored_values_are_dropped(tmp_path: Path) -> None:
    state_dir = save_settings(
        tmp_path,
        gateway={
            "adapter": "telegram",
            "permission_mode": "bogus",
            "max_sessions": -5,
            "pairing_enabled": "not-a-bool",
            "allowed_users": ["111", 222],  # non-string entry invalidates the list
            "unknown_key": "dropped",
        },
    )
    config = load_gateway_config(state_dir=state_dir)
    assert config.adapter == "telegram"  # valid keys survive
    assert config.permission_mode == "ask"
    assert config.max_sessions == 50
    assert config.pairing_enabled is False
    assert config.allowed_users == ()


def test_idle_ttl_can_be_disabled_with_null(tmp_path: Path) -> None:
    state_dir = save_settings(tmp_path, gateway={"session_idle_ttl_seconds": None})
    config = load_gateway_config(state_dir=state_dir)
    assert config.session_idle_ttl_seconds is None
