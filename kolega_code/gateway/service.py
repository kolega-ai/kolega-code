"""Install the gateway as a user-level background service.

systemd user units (Linux) and launchd LaunchAgents (macOS) — no root
needed. ``gateway install`` resolves the current ``kolega-code`` executable
and writes a unit that runs the gateway against this state dir and project.
All configuration — the bot token, allowlists, pairing, STT — lives in
``settings.json`` under the same state dir, so the service needs no captured
environment. Uninstall unloads and removes everything.
"""

from __future__ import annotations

import os
import plistlib
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from kolega_code.gateway.config import GatewayConfig

SERVICE_NAME = "kolega-code-gateway"
SYSTEMD_UNIT_NAME = f"{SERVICE_NAME}.service"
LAUNCHD_LABEL = "com.kolega.gateway"
LAUNCHD_PLIST_NAME = f"{LAUNCHD_LABEL}.plist"

_SYSTEMD_UNIT_TEMPLATE = """[Unit]
Description=Kolega Code messaging gateway
After=network-online.target

[Service]
Type=simple
ExecStart={command}
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
"""


@dataclass
class ServiceInstallResult:
    """Where a service definition landed, for activation and status output."""

    notes: list[str] = field(default_factory=list)
    unit_path: Optional[Path] = None


def login_shell(platform: Optional[str] = None) -> str:
    """The user's login shell: ``$SHELL`` when set, else the platform default.

    launchd and systemd start service programs directly, so the daemon would
    inherit the service manager's minimal environment (launchd gives agents
    ``PATH=/usr/bin:/bin:/usr/sbin:/sbin``), which starves gateway-driven
    agent sessions of Homebrew and other user-installed tools.  The service
    therefore launches through the user's login shell: ``-l`` runs the shell
    profile files, so the daemon's environment matches what an interactive
    terminal sees, and nothing is captured at install time.
    """
    platform = platform if platform is not None else sys.platform
    from_env = os.environ.get("SHELL")
    if from_env:
        return from_env
    return "/bin/zsh" if platform == "darwin" else "/bin/bash"


def service_launch_command(config: GatewayConfig, executable: str, *, shell: str) -> list[str]:
    """The service-level argv: the login shell exec-ing the daemon.

    ``exec`` replaces the shell process, so termination signals reach the
    daemon directly and launchd/KeepAlive track the real process.
    """
    inner = " ".join(shlex.quote(part) for part in service_command(config, executable))
    return [shell, "-l", "-c", f"exec {inner}"]


def service_command(config: GatewayConfig, executable: str) -> list[str]:
    """The exact argv the service runs."""
    return [
        executable,
        "gateway",
        "run",
        "--adapter",
        config.adapter,
        "--state-dir",
        str(config.state_dir),
        "--project",
        str(config.project_path),
    ]


def install_service(
    config: GatewayConfig,
    *,
    executable: Optional[str] = None,
    shell: Optional[str] = None,
    systemd_dir: Optional[Path] = None,
    launchd_dir: Optional[Path] = None,
    platform: Optional[str] = None,
) -> ServiceInstallResult:
    """Write the service definition.

    ``systemd_dir``/``launchd_dir`` are the unit destination directories;
    callers resolve the platform defaults (tests pass temp dirs).  ``shell``
    overrides the login shell the service launches through (tests pass one
    for determinism); by default the user's ``$SHELL`` or the platform
    default is used.
    """
    platform = platform if platform is not None else sys.platform
    executable = executable or shutil.which("kolega-code") or ""
    if not executable:
        raise RuntimeError("kolega-code executable not found; is it on PATH?")
    launch_shell = shell or login_shell(platform)
    config.state_dir.mkdir(parents=True, exist_ok=True)
    result = ServiceInstallResult()
    if platform == "darwin":
        launchd_dir = launchd_dir or (Path.home() / "Library" / "LaunchAgents")
        launchd_dir.mkdir(parents=True, exist_ok=True)
        plist_path = launchd_dir / LAUNCHD_PLIST_NAME
        plist: dict[str, object] = {
            "Label": LAUNCHD_LABEL,
            "ProgramArguments": service_launch_command(config, executable, shell=launch_shell),
            "RunAtLoad": True,
            "KeepAlive": True,
            "StandardOutPath": str(config.state_dir / "gateway.log"),
            "StandardErrorPath": str(config.state_dir / "gateway.log"),
        }
        plist_path.write_bytes(plistlib.dumps(plist))
        result.unit_path = plist_path
        result.notes.append(f"launchd agent written to {plist_path}")
    else:
        systemd_dir = systemd_dir or (Path.home() / ".config" / "systemd" / "user")
        systemd_dir.mkdir(parents=True, exist_ok=True)
        unit_path = systemd_dir / SYSTEMD_UNIT_NAME
        command = " ".join(shlex.quote(part) for part in service_launch_command(config, executable, shell=launch_shell))
        unit_path.write_text(_SYSTEMD_UNIT_TEMPLATE.format(command=command), encoding="utf-8")
        result.unit_path = unit_path
        result.notes.append(f"systemd user unit written to {unit_path}")
    result.notes.append(
        f"the service launches via your login shell ({launch_shell}), so it inherits your "
        "normal terminal environment; restart it after changing your shell profile"
    )
    result.notes.append(
        "the service reads settings.json from the same state dir — make sure your provider "
        "API keys are saved in Settings (they apply to the gateway too)"
    )
    return result


def uninstall_service(
    config: GatewayConfig,
    *,
    systemd_dir: Optional[Path] = None,
    launchd_dir: Optional[Path] = None,
    platform: Optional[str] = None,
) -> list[str]:
    """Remove the service definition."""
    platform = platform if platform is not None else sys.platform
    notes: list[str] = []
    if platform == "darwin":
        launchd_dir = launchd_dir or (Path.home() / "Library" / "LaunchAgents")
        plist_path = launchd_dir / LAUNCHD_PLIST_NAME
        if plist_path.exists():
            plist_path.unlink()
            notes.append(f"removed {plist_path}")
    else:
        systemd_dir = systemd_dir or (Path.home() / ".config" / "systemd" / "user")
        unit_path = systemd_dir / SYSTEMD_UNIT_NAME
        if unit_path.exists():
            unit_path.unlink()
            notes.append(f"removed {unit_path}")
    return notes


def service_unit_path(
    *,
    systemd_dir: Optional[Path] = None,
    launchd_dir: Optional[Path] = None,
    platform: Optional[str] = None,
) -> Path:
    """Resolve the destination path for the gateway's background service unit."""
    platform = platform if platform is not None else sys.platform
    if platform == "darwin":
        launchd_dir = launchd_dir or (Path.home() / "Library" / "LaunchAgents")
        return launchd_dir / LAUNCHD_PLIST_NAME
    systemd_dir = systemd_dir or (Path.home() / ".config" / "systemd" / "user")
    return systemd_dir / SYSTEMD_UNIT_NAME


def is_service_installed(
    *,
    systemd_dir: Optional[Path] = None,
    launchd_dir: Optional[Path] = None,
    platform: Optional[str] = None,
) -> bool:
    """Return whether the gateway background service definition exists on disk."""
    return service_unit_path(
        systemd_dir=systemd_dir,
        launchd_dir=launchd_dir,
        platform=platform,
    ).exists()


def run_service_command(argv: list[str]) -> subprocess.CompletedProcess[str] | None:
    """Run a service command with safe timeouts and error handling."""
    try:
        return subprocess.run(argv, capture_output=True, text=True, check=False, timeout=30)
    except (OSError, subprocess.SubprocessError) as exc:
        print(f"gateway: warning: {argv[0]} failed: {exc}", file=sys.stderr)
        return None


def restart_service(
    *,
    unit_path: Optional[Path] = None,
    systemd_dir: Optional[Path] = None,
    launchd_dir: Optional[Path] = None,
    platform: Optional[str] = None,
) -> tuple[bool, list[str]]:
    """Restart the background gateway service if installed.

    Returns ``(success, notes)``.
    """
    platform = platform if platform is not None else sys.platform
    target_unit = unit_path or service_unit_path(
        systemd_dir=systemd_dir,
        launchd_dir=launchd_dir,
        platform=platform,
    )
    if not target_unit.exists():
        return False, ["gateway background service is not installed"]

    notes: list[str] = []
    if platform == "darwin":
        uid = os.getuid()
        proc = run_service_command(["launchctl", "kickstart", "-k", f"gui/{uid}/{LAUNCHD_LABEL}"])
        if proc is not None and proc.returncode == 0:
            notes.append("launchd agent restarted")
            return True, notes

        # Fallback: if kickstart failed (e.g. not loaded yet), bootout then bootstrap
        run_service_command(["launchctl", "bootout", f"gui/{uid}", LAUNCHD_LABEL])
        boot_proc = run_service_command(["launchctl", "bootstrap", f"gui/{uid}", str(target_unit)])
        if boot_proc is not None and boot_proc.returncode == 0:
            notes.append("launchd agent restarted")
            return True, notes
        notes.append("failed to restart launchd agent")
        return False, notes

    run_service_command(["systemctl", "--user", "daemon-reload"])
    restart_proc = run_service_command(["systemctl", "--user", "restart", SERVICE_NAME])
    if restart_proc is not None and restart_proc.returncode == 0:
        notes.append("systemd unit restarted")
        return True, notes
    notes.append("failed to restart systemd unit")
    return False, notes
