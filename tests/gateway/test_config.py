"""GatewayConfig resolution: env parsing, precedence, and validation."""

from pathlib import Path

import pytest

from kolega_code.gateway.config import (
    DEFAULT_EDIT_THROTTLE_SECONDS,
    DEFAULT_PAIRING_TTL_SECONDS,
    DEFAULT_REQUEST_TIMEOUT_SECONDS,
    ENV_ALLOWED_USERS,
    ENV_GROUP_IDS,
    ENV_PAIRING,
    ENV_PAIRING_TTL_SECONDS,
    ENV_PERMISSION_MODE,
    ENV_PROJECT,
    ENV_REQUEST_TIMEOUT_SECONDS,
    ENV_STATE_DIR,
    ENV_STT,
    ENV_TELEGRAM_PROXY,
    ENV_TELEGRAM_TOKEN,
    GatewayConfigError,
    default_gateway_project,
    load_gateway_config,
)


def test_defaults() -> None:
    config = load_gateway_config(env={}, state_dir=Path("/tmp/gw-state"))
    assert config.adapter == "echo"
    assert config.permission_mode == "ask"
    assert config.allowed_users == ()
    assert config.request_timeout_seconds == DEFAULT_REQUEST_TIMEOUT_SECONDS
    assert config.edit_throttle_seconds == DEFAULT_EDIT_THROTTLE_SECONDS
    assert config.max_sessions == 50
    assert config.session_idle_ttl_seconds == 3600.0
    assert config.telegram_token is None
    assert config.telegram_proxy is None
    assert config.stt_enabled is False


def test_default_project_is_home_workspace(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOME", "/home/tester")
    assert default_gateway_project() == Path("/home/tester/kolega-code-workspace")
    config = load_gateway_config(env={}, state_dir=Path("/tmp/gw-state"))
    assert config.project_path == Path("/home/tester/kolega-code-workspace")


def test_project_precedence_flag_over_env_over_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOME", "/home/tester")
    # Env beats the default.
    config = load_gateway_config(env={ENV_PROJECT: "/env/project"}, state_dir=Path("/tmp/gw-state"))
    assert config.project_path == Path("/env/project").resolve()
    # Explicit parameter beats env.
    config = load_gateway_config(
        env={ENV_PROJECT: "/env/project"},
        state_dir=Path("/tmp/gw-state"),
        project=Path("/flag/project"),
    )
    assert config.project_path == Path("/flag/project").resolve()


def test_state_dir_precedence() -> None:
    # Explicit parameter wins.
    config = load_gateway_config(
        env={ENV_STATE_DIR: "/env/state"},
        state_dir=Path("/param/state"),
    )
    assert config.state_dir == Path("/param/state").resolve()
    # Gateway env beats the shared KOLEGA_CODE_STATE_DIR.
    config = load_gateway_config(env={ENV_STATE_DIR: "/env/state", "KOLEGA_CODE_STATE_DIR": "/code/state"})
    assert config.state_dir == Path("/env/state").resolve()
    # Shared env is the final fallback before the platform default.
    config = load_gateway_config(env={"KOLEGA_CODE_STATE_DIR": "/code/state"})
    assert config.state_dir == Path("/code/state").resolve()


def test_allowed_users_parsing() -> None:
    config = load_gateway_config(env={ENV_ALLOWED_USERS: " 123, 456 ,,789 "}, state_dir=Path("/tmp/gw-state"))
    assert config.allowed_users == ("123", "456", "789")
    # Empty / whitespace-only means "anyone allowed".
    config = load_gateway_config(env={ENV_ALLOWED_USERS: " , "}, state_dir=Path("/tmp/gw-state"))
    assert config.allowed_users == ()


def test_permission_mode_validation() -> None:
    config = load_gateway_config(env={ENV_PERMISSION_MODE: "auto"}, state_dir=Path("/tmp/gw-state"))
    assert config.permission_mode == "auto"
    with pytest.raises(ValueError):
        load_gateway_config(env={ENV_PERMISSION_MODE: "bogus"}, state_dir=Path("/tmp/gw-state"))


def test_numeric_validation() -> None:
    config = load_gateway_config(env={ENV_REQUEST_TIMEOUT_SECONDS: "120.5"}, state_dir=Path("/tmp/gw-state"))
    assert config.request_timeout_seconds == 120.5
    for bad in ("abc", "0", "-5"):
        with pytest.raises(GatewayConfigError):
            load_gateway_config(env={ENV_REQUEST_TIMEOUT_SECONDS: bad}, state_dir=Path("/tmp/gw-state"))


def test_boolean_parsing() -> None:
    for raw in ("1", "true", "yes", "on", "TRUE"):
        assert load_gateway_config(env={ENV_STT: raw}, state_dir=Path("/tmp/gw-state")).stt_enabled is True
    for raw in ("0", "false", "off", "no"):
        assert load_gateway_config(env={ENV_STT: raw}, state_dir=Path("/tmp/gw-state")).stt_enabled is False


def test_telegram_credentials_passthrough() -> None:
    config = load_gateway_config(
        env={ENV_TELEGRAM_TOKEN: "123:abc", ENV_TELEGRAM_PROXY: "http://127.0.0.1:3128"},
        state_dir=Path("/tmp/gw-state"),
    )
    assert config.telegram_token == "123:abc"
    assert config.telegram_proxy == "http://127.0.0.1:3128"


def test_pairing_and_group_config() -> None:
    config = load_gateway_config(env={}, state_dir=Path("/tmp/gw-state"))
    assert config.pairing_enabled is False
    assert config.group_ids == ()
    assert config.pairing_code_ttl_seconds == DEFAULT_PAIRING_TTL_SECONDS

    config = load_gateway_config(
        env={ENV_PAIRING: "1", ENV_GROUP_IDS: "-1001, -1002 ,", ENV_PAIRING_TTL_SECONDS: "1800"},
        state_dir=Path("/tmp/gw-state"),
    )
    assert config.pairing_enabled is True
    assert config.group_ids == ("-1001", "-1002")
    assert config.pairing_code_ttl_seconds == 1800.0
