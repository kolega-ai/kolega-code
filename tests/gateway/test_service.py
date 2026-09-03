"""Service installation: systemd user units and launchd agents, file-level."""

import plistlib
import subprocess
from pathlib import Path

from kolega_code.gateway.config import GatewayConfig
from kolega_code.gateway.service import (
    LAUNCHD_LABEL,
    LAUNCHD_PLIST_NAME,
    SERVICE_NAME,
    SYSTEMD_UNIT_NAME,
    install_service,
    is_service_installed,
    login_shell,
    restart_service,
    service_command,
    service_launch_command,
    service_unit_path,
    uninstall_service,
)


def make_config(tmp_path: Path) -> GatewayConfig:
    return GatewayConfig(
        adapter="telegram",
        project_path=tmp_path / "workspace",
        state_dir=tmp_path / "state",
        telegram_token="123:fake-bot-token-for-tests-only",
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


def test_service_launch_command_wraps_the_daemon_in_a_login_shell() -> None:
    config = make_config(Path("/tmp/x"))
    argv = service_launch_command(config, "/usr/local/bin/kolega-code", shell="/bin/zsh")
    assert argv[:3] == ["/bin/zsh", "-l", "-c"]
    # `exec` replaces the shell so signals reach the daemon directly.
    assert argv[3].startswith("exec /usr/local/bin/kolega-code gateway run")
    assert f"--state-dir {config.state_dir}" in argv[3]


def test_service_launch_command_quotes_paths_with_spaces() -> None:
    config = GatewayConfig(
        adapter="telegram",
        project_path=Path("/tmp/my project"),
        state_dir=Path("/tmp/ Application Support/state"),
        telegram_token="123:fake-bot-token-for-tests-only",
    )
    argv = service_launch_command(config, "/usr/local/bin/kolega-code", shell="/bin/bash")
    assert "'/tmp/ Application Support/state'" in argv[3]
    assert "'/tmp/my project'" in argv[3]


def test_install_writes_systemd_unit_without_env_file(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    systemd_dir = tmp_path / "systemd-user"
    result = install_service(
        config,
        executable="/usr/local/bin/kolega-code",
        shell="/bin/bash",
        systemd_dir=systemd_dir,
        launchd_dir=tmp_path / "launchd",
        platform="linux",
    )
    unit_path = systemd_dir / SYSTEMD_UNIT_NAME
    assert result.unit_path == unit_path
    assert unit_path.exists()
    unit_text = unit_path.read_text(encoding="utf-8")
    # The daemon launches through the login shell so it inherits the user's
    # normal environment (launchd/systemd alone provide a minimal PATH).
    assert "ExecStart=/bin/bash -l -c 'exec /usr/local/bin/kolega-code gateway run" in unit_text
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
        shell="/bin/zsh",
        systemd_dir=tmp_path / "systemd-user",
        launchd_dir=launchd_dir,
        platform="darwin",
    )
    plist_path = launchd_dir / LAUNCHD_PLIST_NAME
    assert result.unit_path == plist_path
    plist = plistlib.loads(plist_path.read_bytes())
    assert plist["Label"] == LAUNCHD_LABEL
    argv = plist["ProgramArguments"]
    assert argv[:3] == ["/bin/zsh", "-l", "-c"]
    assert argv[3].startswith("exec /usr/local/bin/kolega-code gateway run")
    assert "EnvironmentVariables" not in plist
    assert any("login shell" in note for note in result.notes)


def test_install_uses_shell_env_var_when_set(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("SHELL", "/bin/fish")
    launchd_dir = tmp_path / "LaunchAgents"
    install_service(
        make_config(tmp_path),
        executable="/usr/local/bin/kolega-code",
        systemd_dir=tmp_path / "systemd-user",
        launchd_dir=launchd_dir,
        platform="darwin",
    )
    plist = plistlib.loads((launchd_dir / LAUNCHD_PLIST_NAME).read_bytes())
    assert plist["ProgramArguments"][0] == "/bin/fish"


def test_install_falls_back_to_platform_default_shell(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("SHELL", raising=False)
    assert login_shell(platform="darwin") == "/bin/zsh"
    assert login_shell(platform="linux") == "/bin/bash"

    launchd_dir = tmp_path / "LaunchAgents"
    install_service(
        make_config(tmp_path),
        executable="/usr/local/bin/kolega-code",
        systemd_dir=tmp_path / "systemd-user",
        launchd_dir=launchd_dir,
        platform="darwin",
    )
    plist = plistlib.loads((launchd_dir / LAUNCHD_PLIST_NAME).read_bytes())
    assert plist["ProgramArguments"][0] == "/bin/zsh"


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


def test_service_unit_path_resolves_platform_defaults(tmp_path: Path) -> None:
    systemd_dir = tmp_path / "systemd"
    launchd_dir = tmp_path / "launchd"

    linux_path = service_unit_path(systemd_dir=systemd_dir, launchd_dir=launchd_dir, platform="linux")
    assert linux_path == systemd_dir / SYSTEMD_UNIT_NAME

    darwin_path = service_unit_path(systemd_dir=systemd_dir, launchd_dir=launchd_dir, platform="darwin")
    assert darwin_path == launchd_dir / LAUNCHD_PLIST_NAME


def test_is_service_installed(tmp_path: Path) -> None:
    systemd_dir = tmp_path / "systemd"
    systemd_dir.mkdir(parents=True)
    assert not is_service_installed(systemd_dir=systemd_dir, platform="linux")

    unit_path = systemd_dir / SYSTEMD_UNIT_NAME
    unit_path.write_text("[Unit]", encoding="utf-8")
    assert is_service_installed(systemd_dir=systemd_dir, platform="linux")


def test_restart_service_linux_runs_daemon_reload_and_restart(tmp_path: Path, monkeypatch) -> None:
    systemd_dir = tmp_path / "systemd"
    systemd_dir.mkdir(parents=True)
    unit_path = systemd_dir / SYSTEMD_UNIT_NAME
    unit_path.write_text("[Unit]", encoding="utf-8")

    commands_run = []

    def fake_run(argv, **kwargs):
        commands_run.append(argv)
        return subprocess.CompletedProcess(args=argv, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    success, notes = restart_service(systemd_dir=systemd_dir, platform="linux")
    assert success is True
    assert "systemd unit restarted" in notes
    assert commands_run == [
        ["systemctl", "--user", "daemon-reload"],
        ["systemctl", "--user", "restart", SERVICE_NAME],
    ]


def test_restart_service_darwin_kickstart(tmp_path: Path, monkeypatch) -> None:
    launchd_dir = tmp_path / "launchd"
    launchd_dir.mkdir(parents=True)
    plist_path = launchd_dir / LAUNCHD_PLIST_NAME
    plist_path.write_text("<plist></plist>", encoding="utf-8")

    commands_run = []

    def fake_run(argv, **kwargs):
        commands_run.append(argv)
        return subprocess.CompletedProcess(args=argv, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    success, notes = restart_service(launchd_dir=launchd_dir, platform="darwin")
    assert success is True
    assert "launchd agent restarted" in notes
    assert len(commands_run) == 1
    assert commands_run[0][:3] == ["launchctl", "kickstart", "-k"]
    assert LAUNCHD_LABEL in commands_run[0][3]


def test_restart_service_darwin_fallback(tmp_path: Path, monkeypatch) -> None:
    launchd_dir = tmp_path / "launchd"
    launchd_dir.mkdir(parents=True)
    plist_path = launchd_dir / LAUNCHD_PLIST_NAME
    plist_path.write_text("<plist></plist>", encoding="utf-8")

    commands_run = []

    def fake_run(argv, **kwargs):
        commands_run.append(argv)
        # First kickstart fails (e.g. exit code 113)
        if "kickstart" in argv:
            return subprocess.CompletedProcess(args=argv, returncode=113, stdout="", stderr="not found")
        return subprocess.CompletedProcess(args=argv, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    success, notes = restart_service(launchd_dir=launchd_dir, platform="darwin")
    assert success is True
    assert "launchd agent restarted" in notes
    assert len(commands_run) == 3
    assert commands_run[0][:3] == ["launchctl", "kickstart", "-k"]
    assert commands_run[1][:2] == ["launchctl", "bootout"]
    assert commands_run[2][:2] == ["launchctl", "bootstrap"]
    assert str(plist_path) in commands_run[2]


def test_restart_service_not_installed(tmp_path: Path) -> None:
    systemd_dir = tmp_path / "systemd"
    success, notes = restart_service(systemd_dir=systemd_dir, platform="linux")
    assert success is False
    assert any("not installed" in note for note in notes)
