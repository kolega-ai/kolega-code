"""Service installation: systemd user units and launchd agents, file-level."""

import plistlib
from pathlib import Path

from kolega_code.gateway.config import GatewayConfig
from kolega_code.gateway.service import (
    ENV_FILE_NAME,
    LAUNCHD_LABEL,
    LAUNCHD_PLIST_NAME,
    SYSTEMD_UNIT_NAME,
    capture_gateway_env,
    install_service,
    service_command,
    uninstall_service,
)


def make_config(tmp_path: Path) -> GatewayConfig:
    return GatewayConfig(
        adapter="telegram",
        project_path=tmp_path / "workspace",
        state_dir=tmp_path / "state",
        telegram_token="123:AAHdqTcvCH1vGWJxfSeofSAs0K5PALDsaw",
    )


TEST_ENV = {
    "KOLEGA_GATEWAY_TELEGRAM_TOKEN": "123:AAHdqTcvCH1vGWJxfSeofSAs0K5PALDsaw",
    "KOLEGA_GATEWAY_ALLOWED_USERS": "111,222",
    "ANTHROPIC_API_KEY": "sk-ant-test",
    "UNRELATED_VAR": "noise",
}


def test_capture_gateway_env_keeps_only_service_keys() -> None:
    captured = capture_gateway_env(TEST_ENV)
    assert captured["KOLEGA_GATEWAY_TELEGRAM_TOKEN"] == TEST_ENV["KOLEGA_GATEWAY_TELEGRAM_TOKEN"]
    assert captured["KOLEGA_GATEWAY_ALLOWED_USERS"] == "111,222"
    assert captured["ANTHROPIC_API_KEY"] == "sk-ant-test"
    assert "UNRELATED_VAR" not in captured


def test_service_command_argv() -> None:
    config = make_config(Path("/tmp/x"))
    argv = service_command(config, "/usr/local/bin/kolega-code")
    assert argv == [
        "/usr/local/bin/kolega-code",
        "gateway",
        "run",
        "--adapter",
        "telegram",
        "--state-dir",
        str(config.state_dir),
        "--project",
        str(config.project_path),
    ]


def test_install_writes_systemd_unit_and_env_file(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    systemd_dir = tmp_path / "systemd-user"
    result = install_service(
        config,
        executable="/usr/local/bin/kolega-code",
        systemd_dir=systemd_dir,
        launchd_dir=tmp_path / "launchd",
        env=TEST_ENV,
        platform="linux",
    )
    unit_path = systemd_dir / SYSTEMD_UNIT_NAME
    assert result.unit_path == unit_path
    assert unit_path.exists()
    unit_text = unit_path.read_text(encoding="utf-8")
    assert "ExecStart=/usr/local/bin/kolega-code gateway run" in unit_text
    assert f"--state-dir {config.state_dir}" in unit_text
    assert f"EnvironmentFile={config.state_dir / ENV_FILE_NAME}" in unit_text
    env_text = (config.state_dir / ENV_FILE_NAME).read_text(encoding="utf-8")
    assert "KOLEGA_GATEWAY_TELEGRAM_TOKEN=123:AAHdqTcvCH1vGWJxfSeofSAs0K5PALDsaw" in env_text
    assert "KOLEGA_GATEWAY_ALLOWED_USERS=111,222" in env_text
    # The env file is private (600).
    assert (config.state_dir / ENV_FILE_NAME).stat().st_mode & 0o777 == 0o600


def test_install_writes_launchd_plist(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    launchd_dir = tmp_path / "LaunchAgents"
    result = install_service(
        config,
        executable="/usr/local/bin/kolega-code",
        systemd_dir=tmp_path / "systemd-user",
        launchd_dir=launchd_dir,
        env=TEST_ENV,
        platform="darwin",
    )
    plist_path = launchd_dir / LAUNCHD_PLIST_NAME
    assert result.unit_path == plist_path
    plist = plistlib.loads(plist_path.read_bytes())
    assert plist["Label"] == LAUNCHD_LABEL
    assert plist["ProgramArguments"][:3] == ["/usr/local/bin/kolega-code", "gateway", "run"]
    assert plist["EnvironmentVariables"]["KOLEGA_GATEWAY_TELEGRAM_TOKEN"] == TEST_ENV["KOLEGA_GATEWAY_TELEGRAM_TOKEN"]


def test_uninstall_removes_files(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    systemd_dir = tmp_path / "systemd-user"
    install_service(
        config,
        executable="/usr/local/bin/kolega-code",
        systemd_dir=systemd_dir,
        launchd_dir=tmp_path / "launchd",
        env=TEST_ENV,
        platform="linux",
    )
    notes = uninstall_service(config, systemd_dir=systemd_dir, launchd_dir=tmp_path / "launchd", platform="linux")
    assert not (systemd_dir / SYSTEMD_UNIT_NAME).exists()
    assert not (config.state_dir / ENV_FILE_NAME).exists()
    assert notes  # reports what it removed
