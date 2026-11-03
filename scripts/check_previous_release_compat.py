#!/usr/bin/env python3
"""Verify current state survives an actual previous PyPI release's writes.

The baseline is the latest published stable PyPI release, even when the
checkout still has that version. Release preflight instead selects the newest
published version strictly older than the candidate. ``--previous-python``
allows an already installed baseline for offline development.

This checks one old-writer settings save and a normal journal append with a
complete tail. It does not certify repeated saves by an old long-lived writer,
concurrent access, journal repair, or arbitrary future schemas/payloads.
The baseline wheel is hash-verified; its resolved dependencies are not locked
or hash-pinned by this check.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import tomllib
from typing import Any
from urllib.parse import urlsplit
from urllib.request import Request, urlopen
import venv

from packaging.version import InvalidVersion, Version

from previous_release_compat_fixture import create_fixture, verify_after_previous

ROOT = Path(__file__).resolve().parent.parent
PYPI_JSON = "https://pypi.org/pypi/kolega-code/json"
WORKER = ROOT / "scripts" / "previous_release_compat_worker.py"


def clean_environment(home: Path) -> dict[str, str]:
    """Keep platform/network essentials, not project configuration or credentials."""
    allowed = {
        "PATH",
        "SYSTEMROOT",
        "WINDIR",
        "TMP",
        "TEMP",
        "TMPDIR",
        "LANG",
        "LC_ALL",
        "SSL_CERT_FILE",
        "SSL_CERT_DIR",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "NO_PROXY",
        "http_proxy",
        "https_proxy",
        "no_proxy",
    }
    env = {key: value for key, value in os.environ.items() if key in allowed}
    env.update(HOME=str(home), XDG_CONFIG_HOME=str(home / "config"), XDG_CACHE_HOME=str(home / "cache"))
    return env


def discover_release(before: Version | None = None) -> tuple[str, dict[str, Any]]:
    """Select the newest non-yanked stable release and require a universal wheel."""
    request = Request(PYPI_JSON, headers={"Accept": "application/json", "User-Agent": "kolega-code-compat-check"})
    with urlopen(request, timeout=30) as response:
        metadata = json.load(response)
    releases = []
    for name, artifacts in metadata["releases"].items():
        try:
            parsed = Version(name)
        except InvalidVersion:
            continue
        if (
            not parsed.is_prerelease
            and not parsed.is_devrelease
            and (before is None or parsed < before)
            and any(not item.get("yanked") for item in artifacts)
        ):
            releases.append((parsed, name))
    if not releases:
        raise ValueError(f"PyPI did not provide a non-yanked stable release before {before}")
    _, version = max(releases)
    wheels = [
        item
        for item in metadata["releases"][version]
        if item.get("packagetype") == "bdist_wheel"
        and not item.get("yanked", False)
        and str(item.get("filename", "")).endswith("-py3-none-any.whl")
    ]
    if len(wheels) != 1:
        raise ValueError(f"Expected one non-yanked universal wheel for {version}; found {len(wheels)}")
    wheel = wheels[0]
    filename = wheel["filename"]
    if Path(filename).name != filename or not filename.startswith("kolega_code-"):
        raise ValueError(f"Unexpected wheel filename: {filename!r}")
    if urlsplit(wheel["url"]).scheme != "https" or urlsplit(wheel["url"]).hostname != "files.pythonhosted.org":
        raise ValueError("PyPI returned an unexpected wheel download host")
    digest = wheel["digests"]["sha256"]
    if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("PyPI did not provide a valid SHA-256 digest")
    return version, wheel


def download_wheel(wheel: dict[str, Any], directory: Path) -> Path:
    path = directory / wheel["filename"]
    digest = hashlib.sha256()
    with urlopen(Request(wheel["url"], headers={"User-Agent": "kolega-code-compat-check"}), timeout=60) as response:
        with path.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                digest.update(chunk)
                output.write(chunk)
    if digest.hexdigest() != wheel["digests"]["sha256"]:
        raise ValueError("Downloaded previous release wheel failed its SHA-256 check")
    print(f"Downloaded {path.name} (sha256 {digest.hexdigest()})", flush=True)
    return path


def install_previous(directory: Path, env: dict[str, str], before: Version | None = None) -> tuple[Path, str]:
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError("uv is required to install the previous release")
    version, wheel = discover_release(before)
    wheel_path = download_wheel(wheel, directory)
    environment = directory / "previous"
    # uv's standalone macOS Python locates libpython relative to the original
    # executable. Copying it into a nested venv breaks that lookup.
    venv.EnvBuilder(with_pip=False, symlinks=True).create(environment)
    python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    command = [
        uv,
        "pip",
        "install",
        "--no-config",
        "--python",
        str(python),
        "--index-url",
        "https://pypi.org/simple",
        str(wheel_path),
    ]
    subprocess.run(command, cwd=directory, env=env, check=True, timeout=600)
    return python, version


def installed_version(python: Path, directory: Path, env: dict[str, str]) -> str:
    result = subprocess.run(
        [str(python), "-I", "-c", "from importlib.metadata import version; print(version('kolega-code'))"],
        cwd=directory,
        env=env,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return result.stdout.strip()


def main() -> None:
    if not __debug__:
        raise RuntimeError("The compatibility check requires Python assertions to be enabled")
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--install-previous", action="store_true", help="Install the latest published release from PyPI")
    group.add_argument("--previous-python", type=Path, help="Python interpreter with an installed previous release")
    parser.add_argument(
        "--release-preflight",
        action="store_true",
        help="Require the baseline to be strictly older than the checkout's release version",
    )
    args = parser.parse_args()
    before = None
    if args.release_preflight:
        with (ROOT / "pyproject.toml").open("rb") as file:
            before = Version(tomllib.load(file)["project"]["version"])
    scratch = os.environ.get("KOLEGA_SCRATCHPAD")
    with tempfile.TemporaryDirectory(prefix="previous-release-compat-", dir=scratch or ROOT) as temporary:
        directory = Path(temporary).resolve()
        env = clean_environment(directory)
        if args.install_previous:
            python, version = install_previous(directory, env, before)
        else:
            python = args.previous_python.absolute()
            version = installed_version(python, directory, env)
            if before is not None and Version(version) >= before:
                raise ValueError(f"Previous release {version} must be older than candidate {before}")
        root = directory / "state"
        root.mkdir()
        env["KOLEGA_CODE_STATE_DIR"] = str(root)
        manifest = create_fixture(root)
        manifest_path = directory / "manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        subprocess.run(
            [str(python), "-I", str(WORKER), str(root), str(manifest_path), version, str(ROOT)],
            cwd=directory,
            env=env,
            check=True,
            timeout=120,
        )
        verify_after_previous(root, manifest)
        print("Previous-release settings and journal compatibility verified.", flush=True)


if __name__ == "__main__":
    main()
