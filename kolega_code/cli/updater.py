"""Version checking and self-update helpers for the CLI."""

from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Callable, Literal, Optional
from urllib import error, request

from packaging.specifiers import SpecifierSet
from packaging.version import InvalidVersion, Version

import kolega_code


PACKAGE_NAME = "kolega-code"
PYPI_JSON_URL = f"https://pypi.org/pypi/{PACKAGE_NAME}/json"
# Without --python, uv uses its default interpreter, often a release our pinned dependencies
# ship no wheels for yet. scripts/install-kolega-code.sh repeats both ranges.
# The 3.11.5 floor is soupsieve 4.x's, which fixes the 2026-11 selector DoS advisories.
TOOL_PYTHON = ">=3.11.5,<3.15"
# onnxruntime has no Intel-mac wheels for 3.14, and its 3.13 ones need macOS 13+.
INTEL_MAC_TOOL_PYTHON = ">=3.11.5,<3.13"
_PYVENV_VERSION_RE = re.compile(r"^(\d+\.\d+(?:\.\d+)?)")
_DIST_INFO_PREFIX = "kolega_code-"
_DIST_INFO_SUFFIX = ".dist-info"


def tool_python(sys_platform: str, machine: str) -> str:
    if sys_platform == "darwin" and machine == "x86_64":
        return INTEL_MAC_TOOL_PYTHON
    return TOOL_PYTHON


UPGRADE_COMMAND = ("uv", "tool", "upgrade", PACKAGE_NAME)
UPDATE_COMMAND = (
    "uv",
    "tool",
    "install",
    "--force",
    "--upgrade",
    "--python",
    tool_python(sys.platform, platform.machine()),
    PACKAGE_NAME,
)


@dataclass(frozen=True)
class UpdateCheckResult:
    current_version: str
    latest_version: Optional[str] = None
    update_available: bool = False
    error: Optional[str] = None


@dataclass(frozen=True)
class UpdateRunResult:
    returncode: int
    command: tuple[str, ...] = UPDATE_COMMAND
    stdout: str = ""
    stderr: str = ""
    error: Optional[str] = None
    gateway_restart_notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class InstallFingerprint:
    package_file: Path
    site_packages: Path
    dist_info_names: frozenset[str]
    version: Optional[str]


@dataclass(frozen=True)
class InstallDrift:
    reason: Literal["missing", "updated"]
    previous_version: Optional[str] = None
    current_version: Optional[str] = None


def _uv_tool_dir() -> Optional[Path]:
    configured = os.environ.get("UV_TOOL_DIR")
    if configured:
        return Path(configured).expanduser()
    try:
        completed = subprocess.run(
            ["uv", "tool", "dir"],
            text=True,
            capture_output=True,
            check=False,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    raw = completed.stdout.strip()
    return Path(raw).expanduser() if raw else None


def _read_venv_python_version(pyvenv_cfg: Path) -> Optional[Version]:
    try:
        lines = pyvenv_cfg.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return None
    for line in lines:
        key, sep, value = line.partition("=")
        if not sep:
            continue
        if key.strip() not in {"version_info", "version"}:
            continue
        match = _PYVENV_VERSION_RE.match(value.strip())
        if match is None:
            continue
        try:
            return Version(match.group(1))
        except InvalidVersion:
            return None
    return None


def existing_tool_python_compatible(
    *,
    tool_dir: Optional[Path] = None,
    sys_platform: Optional[str] = None,
    machine: Optional[str] = None,
) -> bool:
    """True when an existing uv tool venv for kolega-code already uses a supported Python."""
    resolved_dir = tool_dir if tool_dir is not None else _uv_tool_dir()
    if resolved_dir is None:
        return False
    env_dir = resolved_dir / PACKAGE_NAME
    if not (env_dir / "uv-receipt.toml").is_file():
        return False
    pyvenv_cfg = env_dir / "pyvenv.cfg"
    if not pyvenv_cfg.is_file():
        return False
    plat = sys.platform if sys_platform is None else sys_platform
    mach = platform.machine() if machine is None else machine
    venv_python = env_dir / ("Scripts/python.exe" if plat == "win32" else "bin/python")
    if not venv_python.exists():
        return False
    version = _read_venv_python_version(pyvenv_cfg)
    if version is None:
        return False
    return version in SpecifierSet(tool_python(plat, mach))


def build_update_command(
    *,
    tool_dir: Optional[Path] = None,
    sys_platform: Optional[str] = None,
    machine: Optional[str] = None,
) -> tuple[str, ...]:
    """Prefer in-place ``uv tool upgrade`` when the existing venv Python is compatible.

    ``uv tool install --force --python <range>`` deletes and recreates the tool
    venv and can switch Python minor versions, removing ``lib/python3.x/`` out
    from under other still-running sessions.
    """
    plat = sys.platform if sys_platform is None else sys_platform
    mach = platform.machine() if machine is None else machine
    if existing_tool_python_compatible(tool_dir=tool_dir, sys_platform=plat, machine=mach):
        return UPGRADE_COMMAND
    return (
        "uv",
        "tool",
        "install",
        "--force",
        "--upgrade",
        "--python",
        tool_python(plat, mach),
        PACKAGE_NAME,
    )


def _list_dist_info_names(site_packages: Path) -> Optional[frozenset[str]]:
    try:
        return frozenset(
            entry.name
            for entry in os.scandir(site_packages)
            if entry.name.startswith(_DIST_INFO_PREFIX) and entry.name.endswith(_DIST_INFO_SUFFIX)
        )
    except OSError:
        return None


def _version_from_dist_info_names(names: frozenset[str]) -> Optional[str]:
    if len(names) != 1:
        return None
    name = next(iter(names))
    if name.startswith(_DIST_INFO_PREFIX) and name.endswith(_DIST_INFO_SUFFIX):
        return name[len(_DIST_INFO_PREFIX) : -len(_DIST_INFO_SUFFIX)] or None
    return None


def capture_install_fingerprint(package_file: Optional[Path] = None) -> Optional[InstallFingerprint]:
    """Snapshot the installed wheel location so a running TUI can detect out-of-band updates.

    Returns ``None`` for editable/source checkouts (which have no sibling
    ``.dist-info`` next to ``kolega_code/``).
    """
    raw_file = package_file if package_file is not None else getattr(kolega_code, "__file__", None)
    if not raw_file:
        return None
    pkg_file = Path(raw_file)
    if not pkg_file.is_file():
        return None
    site_packages = pkg_file.parent.parent
    dist_info_names = _list_dist_info_names(site_packages)
    if not dist_info_names:
        return None
    return InstallFingerprint(
        package_file=pkg_file,
        site_packages=site_packages,
        dist_info_names=dist_info_names,
        version=_version_from_dist_info_names(dist_info_names),
    )


def detect_install_drift(fingerprint: Optional[InstallFingerprint]) -> Optional[InstallDrift]:
    if fingerprint is None:
        return None
    if not fingerprint.package_file.exists():
        return InstallDrift(reason="missing", previous_version=fingerprint.version)
    current_names = _list_dist_info_names(fingerprint.site_packages)
    if current_names is None:
        return InstallDrift(reason="missing", previous_version=fingerprint.version)
    if current_names != fingerprint.dist_info_names:
        return InstallDrift(
            reason="updated",
            previous_version=fingerprint.version,
            current_version=_version_from_dist_info_names(current_names),
        )
    return None


def install_drift_warning(drift: InstallDrift) -> str:
    if (
        drift.reason == "updated"
        and drift.previous_version
        and drift.current_version
        and drift.previous_version != drift.current_version
    ):
        return (
            f"Kolega Code was updated on disk ({drift.previous_version} -> {drift.current_version}). "
            "Run /restart to switch to the updated version and avoid errors in this session."
        )
    return "Kolega Code's installation changed on disk. Run /restart to reload this session cleanly."


def count_other_live_sessions(
    state_root: Optional[Path] = None,
    *,
    exclude_pid: Optional[int] = None,
) -> int:
    """Count live Kolega Code sessions owned by other processes under ``state_root``."""
    from kolega_code.cli.session_store import default_state_dir
    from kolega_code.session.inbox import MESSAGING_DIR_NAME, is_pid_alive, parse_socket_name

    root = state_root if state_root is not None else default_state_dir()
    sock_dir = root / MESSAGING_DIR_NAME
    if not sock_dir.is_dir():
        return 0
    own_pid = os.getpid() if exclude_pid is None else exclude_pid
    live_sessions: set[str] = set()
    try:
        entries = list(sock_dir.iterdir())
    except OSError:
        return 0
    for entry in entries:
        parsed = parse_socket_name(entry.name)
        if parsed is None:
            continue
        session_id, pid = parsed
        if pid == own_pid:
            continue
        if is_pid_alive(pid):
            live_sessions.add(session_id)
    return len(live_sessions)


def other_live_sessions_note(count: int) -> Optional[str]:
    if count <= 0:
        return None
    if count == 1:
        return "1 other running Kolega Code session detected. Run /restart in it to use the updated version."
    return f"{count} other running Kolega Code sessions detected. Run /restart in each to use the updated version."


def current_version() -> str:
    try:
        return metadata.version(PACKAGE_NAME)
    except metadata.PackageNotFoundError:
        return kolega_code.__version__


def check_for_update(timeout: float = 2.0) -> UpdateCheckResult:
    current = current_version()
    try:
        http_request = request.Request(
            PYPI_JSON_URL,
            headers={"User-Agent": f"{PACKAGE_NAME}/{current}"},
        )
        with request.urlopen(http_request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
        latest = str(payload["info"]["version"])
        update_available = Version(latest) > Version(current)
        return UpdateCheckResult(current_version=current, latest_version=latest, update_available=update_available)
    except (KeyError, TypeError, ValueError, InvalidVersion, OSError, error.URLError) as exc:
        return UpdateCheckResult(current_version=current, error=str(exc))


def update_status_message(
    result: UpdateCheckResult,
    *,
    include_up_to_date: bool = False,
    include_errors: bool = False,
) -> Optional[str]:
    if result.update_available and result.latest_version:
        return f"Update available: {result.current_version} -> {result.latest_version}. Run `kolega-code update`."
    if result.error and include_errors:
        return f"Update check failed: {result.error}"
    if include_up_to_date:
        return f"Kolega Code is up to date ({result.current_version})."
    return None


def run_self_update(*, capture_output: bool = False, restart_gateway: bool = True) -> UpdateRunResult:
    if shutil.which("uv") is None:
        return UpdateRunResult(
            returncode=2,
            error="uv is required to update Kolega Code. Install uv, then run `kolega-code update` again.",
        )

    command = build_update_command()
    # Import gateway helpers before uv modifies or recreates the tool venv so a
    # Python-version migration cannot break the post-update import.
    gateway_helpers: Optional[tuple[Callable[[], bool], Callable[[], tuple[bool, list[str]]]]] = None
    gateway_import_error: Optional[str] = None
    if restart_gateway:
        try:
            from kolega_code.gateway.service import is_service_installed, restart_service

            gateway_helpers = (is_service_installed, restart_service)
        except Exception as exc:  # noqa: BLE001
            gateway_import_error = str(exc)

    try:
        completed = subprocess.run(
            list(command),
            text=True,
            capture_output=capture_output,
            check=False,
        )
    except OSError as exc:
        return UpdateRunResult(returncode=2, command=command, error=str(exc))

    gateway_notes: list[str] = []
    if completed.returncode == 0 and restart_gateway:
        if gateway_import_error is not None:
            gateway_notes.append(f"could not restart gateway service: {gateway_import_error}")
        elif gateway_helpers is not None:
            is_service_installed, restart_service = gateway_helpers
            try:
                if is_service_installed():
                    _, notes = restart_service()
                    gateway_notes.extend(notes)
            except Exception as exc:  # noqa: BLE001
                gateway_notes.append(f"could not restart gateway service: {exc}")

    return UpdateRunResult(
        returncode=completed.returncode,
        command=command,
        stdout=completed.stdout or "",
        stderr=completed.stderr or "",
        gateway_restart_notes=tuple(gateway_notes),
    )
