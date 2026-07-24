"""Dedicated, kolega-code-managed Python environment for eval kernels.

The eval Python kernel deliberately runs neither on the project's venv nor on
the kolega-code host interpreter. It gets its own managed environment under the
user state dir so a curated library bundle (numpy/pandas/matplotlib/pillow) is
always available and agent-driven ``pip_install`` calls can never break the
CLI's own exactly-pinned dependency closure.

Provisioning is lazy (first ``py`` cell), serialized across processes with a
file lock, and self-healing: a manifest records the requested Python version
and bundle hash, and any mismatch triggers a clean reprovision. When
provisioning fails entirely (offline host, no compilers) the manager degrades
to the host interpreter with a model-facing note instead of failing the cell.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from importlib.resources import files
from pathlib import Path
from typing import List, Optional

import filelock

BUNDLE_RESOURCE = "bundle-requirements.txt"
ENV_DIR_NAME = "eval-env"
LOCK_FILE_NAME = "eval-env.lock"
MANIFEST_NAME = "kolega-eval-manifest.json"

_INSTALL_TIMEOUT_S = 900
_VENV_TIMEOUT_S = 300


@dataclass
class KernelEnvInfo:
    """Resolved kernel interpreter and provisioning outcome."""

    python: str
    env_path: Optional[Path]
    bundled: bool
    degraded: bool
    provisioned_now: bool
    note: Optional[str] = None
    # Command prefix for prelude pip_install(); empty when installs are not
    # possible (degraded host interpreter or uv-managed env without uv).
    pip_install_cmd: List[str] = field(default_factory=list)
    bundle: List[str] = field(default_factory=list)


def default_state_dir() -> Path:
    """User state dir for kolega-code (mirrors cli.session_store.default_state_dir).

    Mirrored here rather than imported so the agent layer does not depend on the
    CLI layer. Keep the semantics in sync: KOLEGA_CODE_STATE_DIR override, then
    the per-platform application-state location.
    """
    env = os.environ.get("KOLEGA_CODE_STATE_DIR")
    if env:
        return Path(env).expanduser()
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "kolega-code"
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
        return base / "kolega-code"
    return Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local" / "state")) / "kolega-code"


def _venv_python(env_path: Path) -> Path:
    if sys.platform == "win32":
        return env_path / "Scripts" / "python.exe"
    return env_path / "bin" / "python"


def _run(cmd: List[str], timeout: int) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)


class EvalEnvironmentManager:
    """Provisions and resolves the dedicated eval Python environment."""

    def __init__(
        self,
        *,
        state_dir: Optional[Path] = None,
        env_path: Optional[Path] = None,
        python_version: str = "3.12",
        extra_packages: Optional[List[str]] = None,
        override_python: Optional[str] = None,
    ) -> None:
        self._state_dir = Path(state_dir) if state_dir else default_state_dir()
        self._env_path = Path(env_path) if env_path else self._state_dir / ENV_DIR_NAME
        self._python_version = python_version
        self._extra_packages = list(extra_packages or [])
        self._override_python = override_python
        self._cached: Optional[KernelEnvInfo] = None

    @property
    def env_path(self) -> Path:
        return self._env_path

    def _bundle_source(self) -> bytes:
        return files("kolega_code.agent.eval").joinpath(BUNDLE_RESOURCE).read_bytes()

    def _bundle_hash(self) -> str:
        digest = hashlib.sha256()
        digest.update(self._bundle_source())
        for package in self._extra_packages:
            digest.update(b"\0")
            digest.update(package.encode("utf-8"))
        return digest.hexdigest()

    def _bundle_names(self) -> List[str]:
        names: List[str] = []
        for line in self._bundle_source().decode("utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                names.append(line)
        return names + self._extra_packages

    def _materialize_bundle_file(self) -> Path:
        """Write the bundle requirements to a real file installers can read."""
        digest = self._bundle_hash()[:16]
        target = self._state_dir / f"eval-bundle-{digest}.txt"
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(self._bundle_source())
            if self._extra_packages:
                with target.open("a", encoding="utf-8") as handle:
                    for package in self._extra_packages:
                        handle.write(package + "\n")
        return target

    def _read_manifest(self) -> Optional[dict]:
        manifest_path = self._env_path / MANIFEST_NAME
        try:
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return data if isinstance(data, dict) else None

    def _write_manifest(self, installer: str) -> None:
        manifest = {
            "python_version": self._python_version,
            "bundle_hash": self._bundle_hash(),
            "installer": installer,
        }
        (self._env_path / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    def _manifest_matches(self) -> bool:
        manifest = self._read_manifest()
        if manifest is None:
            return False
        if manifest.get("bundle_hash") != self._bundle_hash():
            return False
        if manifest.get("python_version") != self._python_version:
            return False
        return _venv_python(self._env_path).exists()

    def _pip_install_cmd(self, installer: str, python: str) -> List[str]:
        uv = shutil.which("uv")
        if installer == "uv":
            # uv-created venvs have no pip module; only uv can install into them.
            return [uv, "pip", "install", "--python", python] if uv else []
        return [python, "-m", "pip", "install"]

    async def ensure(self) -> KernelEnvInfo:
        """Resolve the kernel interpreter, provisioning the env on first use."""
        if self._cached is not None:
            return self._cached
        import asyncio

        info = await asyncio.to_thread(self._ensure_sync)
        self._cached = info
        return info

    def _ensure_sync(self) -> KernelEnvInfo:
        if self._override_python:
            probe = _run([self._override_python, "--version"], timeout=30)
            if probe.returncode == 0:
                return KernelEnvInfo(
                    python=self._override_python,
                    env_path=None,
                    bundled=False,
                    degraded=False,
                    provisioned_now=False,
                    note=(
                        "eval kernel is using the configured custom interpreter "
                        f"({self._override_python}); the bundled libraries are not installed there. "
                        "Use pip_install(package) to add packages."
                    ),
                    pip_install_cmd=[self._override_python, "-m", "pip", "install"],
                )
            # Fall through to normal provisioning if the override is broken.

        if self._manifest_matches() and self._read_manifest() is not None:
            python = str(_venv_python(self._env_path))
            installer = str((self._read_manifest() or {}).get("installer") or "pip")
            return KernelEnvInfo(
                python=python,
                env_path=self._env_path,
                bundled=True,
                degraded=False,
                provisioned_now=False,
                pip_install_cmd=self._pip_install_cmd(installer, python),
                bundle=self._bundle_names(),
            )

        lock_path = self._state_dir / LOCK_FILE_NAME
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with filelock.FileLock(str(lock_path)):
            # Another process may have provisioned while we waited on the lock.
            if self._manifest_matches():
                python = str(_venv_python(self._env_path))
                installer = str((self._read_manifest() or {}).get("installer") or "pip")
                return KernelEnvInfo(
                    python=python,
                    env_path=self._env_path,
                    bundled=True,
                    degraded=False,
                    provisioned_now=False,
                    pip_install_cmd=self._pip_install_cmd(installer, python),
                    bundle=self._bundle_names(),
                )
            return self._provision_locked()

    def _provision_locked(self) -> KernelEnvInfo:
        """Create the env and install the bundle. Caller holds the file lock."""
        try:
            if self._env_path.exists():
                shutil.rmtree(self._env_path, ignore_errors=True)
            self._env_path.parent.mkdir(parents=True, exist_ok=True)
            bundle_file = self._materialize_bundle_file()
            uv = shutil.which("uv")
            if uv:
                step = _run(
                    [uv, "venv", str(self._env_path), "--python", self._python_version], timeout=_VENV_TIMEOUT_S
                )
                if step.returncode != 0:
                    raise RuntimeError(f"uv venv failed: {step.stderr[-500:]}")
                python = str(_venv_python(self._env_path))
                step = _run(
                    [uv, "pip", "install", "--python", python, "-r", str(bundle_file)],
                    timeout=_INSTALL_TIMEOUT_S,
                )
                if step.returncode != 0:
                    raise RuntimeError(f"uv pip install failed: {step.stderr[-500:]}")
                installer = "uv"
            else:
                step = _run([sys.executable, "-m", "venv", str(self._env_path)], timeout=_VENV_TIMEOUT_S)
                if step.returncode != 0:
                    raise RuntimeError(f"python -m venv failed: {step.stderr[-500:]}")
                python = str(_venv_python(self._env_path))
                step = _run(
                    [python, "-m", "pip", "install", "-r", str(bundle_file)],
                    timeout=_INSTALL_TIMEOUT_S,
                )
                if step.returncode != 0:
                    raise RuntimeError(f"pip install failed: {step.stderr[-500:]}")
                installer = "pip"
            self._write_manifest(installer)
            return KernelEnvInfo(
                python=python,
                env_path=self._env_path,
                bundled=True,
                degraded=False,
                provisioned_now=True,
                note=(
                    "Provisioned the dedicated eval Python environment on first use "
                    f"({self._python_version} with the bundled data libraries); this is a one-time setup."
                ),
                pip_install_cmd=self._pip_install_cmd(installer, python),
                bundle=self._bundle_names(),
            )
        except Exception as exc:  # degrade rather than fail the cell
            shutil.rmtree(self._env_path, ignore_errors=True)
            return KernelEnvInfo(
                python=sys.executable,
                env_path=None,
                bundled=False,
                degraded=True,
                provisioned_now=False,
                note=(
                    "Could not provision the dedicated eval Python environment "
                    f"({exc}); running on the host interpreter {sys.executable} without the bundled "
                    "libraries (numpy/pandas/matplotlib/pillow). pip_install is disabled."
                ),
                pip_install_cmd=[],
            )


def resolve_state_dir_from_config(config: object) -> Path:
    """State dir override from AgentConfig (Mock-safe), else the default."""
    raw = getattr(config, "eval_env_path", None)
    if isinstance(raw, str) and raw.strip():
        return Path(raw).expanduser()
    return default_state_dir()
