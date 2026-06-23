"""Version checking and self-update helpers for the CLI."""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from importlib import metadata
from typing import Optional
from urllib import error, request

from packaging.version import InvalidVersion, Version

import kolega_code


PACKAGE_NAME = "kolega-code"
PYPI_JSON_URL = f"https://pypi.org/pypi/{PACKAGE_NAME}/json"
UPDATE_COMMAND = ("uv", "tool", "install", "--force", "--upgrade", PACKAGE_NAME)


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


def run_self_update(*, capture_output: bool = False) -> UpdateRunResult:
    if shutil.which(UPDATE_COMMAND[0]) is None:
        return UpdateRunResult(
            returncode=2,
            error="uv is required to update Kolega Code. Install uv, then run `kolega-code update` again.",
        )

    try:
        completed = subprocess.run(
            list(UPDATE_COMMAND),
            text=True,
            capture_output=capture_output,
            check=False,
        )
    except OSError as exc:
        return UpdateRunResult(returncode=2, error=str(exc))

    return UpdateRunResult(
        returncode=completed.returncode,
        stdout=completed.stdout or "",
        stderr=completed.stderr or "",
    )
