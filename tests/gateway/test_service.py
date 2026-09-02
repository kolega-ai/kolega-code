"""Service installation: systemd user units and launchd agents, file-level."""

import plistlib
from pathlib import Path

from kolega_code.gateway.config import GatewayConfig
from kolega_code.gateway.service import (
    LAUNCHD_LABEL,
    LAUNCHD_PLIST_NAME,
    SYSTEMD_UNIT_NAME,
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


def test_install_writes_systemd_unit_without_env_file(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    systemd_dir = tmp_path / "systemd-user"
    result = install_service(
        config,
        executable="/usr/local/bin/kolega-code",
        systemd_dir=systemd_dir,
        launchd_dir=tmp_path / "launchd",
        platform="linux",
    )
    unit_path = systemd_dir / SYSTEMD_UNIT_NAME
    assert result.unit_path == unit_path
    assert unit_path.exists()
    unit_text = unit_path.read_text(encoding="utf-8")
    assert "ExecStart=/usr/local/bin/kolega-code gateway run" in unit_text
    assert f"--state-dir {config.state_dir}" in unit_text
    # Configuration comes from settings.json, so no EnvironmentFile.
    assert "EnvironmentFile" not in unit_text
    assert any("settings.json" in note for note in result.notes)


def test_install_writes_launchd_plist_without_env_variables(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    launchd_dir = tmp_path / "LaunchAgents"
    result = install_service(
        config,
        executable="/usr/local/bin/kolega-code",
        systemd_dir=tmp_path / "systemd-user",
        launchd_dir=launchd_dir,
        platform="darwin",
    )
    plist_path = launchd_dir / LAUNCHD_PLIST_NAME
    assert result.unit_path == plist_path
    plist = plistlib.loads(plist_path.read_bytes())
    assert plist["Label"] == LAUNCHD_LABEL
    assert plist["ProgramArguments"][:3] == ["/usr/local/bin/kolega-code", "gateway", "run"]
    assert "EnvironmentVariables" not in plist


def test_uninstall_removes_the_unit(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    systemd_dir = tmp_path / "systemd-user"
    install_service(
        config,
        executable="/usr/local/bin/kolega-code",
        systemd_dir=systemd_dir,
        launchd_dir=tmp_path / "launchd",
        platform="linux",
    )
    notes = uninstall_service(config, systemd_dir=systemd_dir, launchd_dir=tmp_path / "launchd", platform="linux")
    assert not (systemd_dir / SYSTEMD_UNIT_NAME).exists()
    assert notes  # reports what it removed
