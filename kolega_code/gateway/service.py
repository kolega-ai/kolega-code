"""Install the gateway as a user-level background service.

systemd user units (Linux) and launchd LaunchAgents (macOS) — no root
needed. ``gateway install`` captures the current gateway/provider environment
into a private env file under the state dir, resolves the current
``kolega-code`` executable, and writes a unit that runs the gateway against
that state dir and project. Uninstall unloads and removes everything.
"""

from __future__ import annotations

import os
import plistlib
import shlex
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Optional

from kolega_code.gateway.config import GatewayConfig
from kolega_code.local_state import write_private_text

SERVICE_NAME = "kolega-code-gateway"
ENV_FILE_NAME = "gateway.env"
SYSTEMD_UNIT_NAME = f"{SERVICE_NAME}.service"
LAUNCHD_LABEL = "com.kolega.gateway"
LAUNCHD_PLIST_NAME = f"{LAUNCHD_LABEL}.plist"

_SYSTEMD_UNIT_TEMPLATE = """[Unit]
Description=Kolega Code messaging gateway
After=network-online.target

[Service]
Type=simple
EnvironmentFile={env_path}
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
    env_path: Optional[Path] = None


def capture_gateway_env(env: Optional[Mapping[str, str]] = None) -> dict[str, str]:
    """The environment the service needs, from the current process env.

    Everything KOLEGA_GATEWAY_*/KOLEGA_CODE_*, plus every provider API key
    the CLI knows about, so the daemon resolves models the same way the
    operator's shell does.
    """
    from kolega_code.cli.config import API_KEY_ENV

    source = env if env is not None else os.environ
    keys = {key for key in source if key.startswith(("KOLEGA_GATEWAY_", "KOLEGA_CODE_")) or key in API_KEY_ENV.values()}
    return {key: str(source[key]) for key in sorted(keys) if str(source[key])}


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
    systemd_dir: Optional[Path] = None,
    launchd_dir: Optional[Path] = None,
    env: Optional[Mapping[str, str]] = None,
    platform: Optional[str] = None,
) -> ServiceInstallResult:
    """Write the service definition and the captured env file.

    ``systemd_dir``/``launchd_dir`` are the unit destination directories;
    callers resolve the platform defaults (tests pass temp dirs).
    """
    platform = platform if platform is not None else sys.platform
    executable = executable or shutil.which("kolega-code") or ""
    if not executable:
        raise RuntimeError("kolega-code executable not found; is it on PATH?")
    state_dir = config.state_dir
    state_dir.mkdir(parents=True, exist_ok=True)
    env_path = state_dir / ENV_FILE_NAME
    env_text = "\n".join(f"{key}={shlex.quote(value)}" for key, value in sorted(capture_gateway_env(env).items()))
    write_private_text(env_path, env_text + "\n")
    result = ServiceInstallResult(notes=[], env_path=env_path)
    if platform == "darwin":
        launchd_dir = launchd_dir or (Path.home() / "Library" / "LaunchAgents")
        launchd_dir.mkdir(parents=True, exist_ok=True)
        plist_path = launchd_dir / LAUNCHD_PLIST_NAME
        plist: dict[str, object] = {
            "Label": LAUNCHD_LABEL,
            "ProgramArguments": service_command(config, executable),
            "EnvironmentVariables": capture_gateway_env(env),
            "RunAtLoad": True,
            "KeepAlive": True,
            "StandardOutPath": str(state_dir / "gateway.log"),
            "StandardErrorPath": str(state_dir / "gateway.log"),
        }
        plist_path.write_bytes(plistlib.dumps(plist))
        result.unit_path = plist_path
        result.notes.append(f"launchd agent written to {plist_path}")
    else:
        systemd_dir = systemd_dir or (Path.home() / ".config" / "systemd" / "user")
        systemd_dir.mkdir(parents=True, exist_ok=True)
        unit_path = systemd_dir / SYSTEMD_UNIT_NAME
        command = " ".join(shlex.quote(part) for part in service_command(config, executable))
        unit_path.write_text(
            _SYSTEMD_UNIT_TEMPLATE.format(env_path=env_path, command=command),
            encoding="utf-8",
        )
        result.unit_path = unit_path
        result.notes.append(f"systemd user unit written to {unit_path}")
    result.notes.append(f"environment captured to {env_path}")
    return result


def uninstall_service(
    config: GatewayConfig,
    *,
    systemd_dir: Optional[Path] = None,
    launchd_dir: Optional[Path] = None,
    platform: Optional[str] = None,
) -> list[str]:
    """Remove the service definition and the captured env file."""
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
    env_path = config.state_dir / ENV_FILE_NAME
    if env_path.exists():
        env_path.unlink()
        notes.append(f"removed {env_path}")
    return notes
