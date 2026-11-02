"""Dedicated, kolega-code-managed CPython 3.14 environment for eval kernels.

Each Python/bundle combination gets its own directory. New versions are built
and verified without modifying previous environments, so running kernels keep
working and failed provisioning cannot destroy a working installation.
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
from typing import List, Optional, Tuple

import filelock

EVAL_PYTHON_VERSION = "3.14"
BUNDLE_RESOURCE = "bundle-requirements.txt"
ENV_DIR_PREFIX = f"eval-env-py{EVAL_PYTHON_VERSION}"
LOCK_FILE_NAME = "eval-env.lock"
MANIFEST_NAME = "kolega-eval-manifest.json"

_INSTALL_TIMEOUT_S = 900
_VENV_TIMEOUT_S = 300
_PROBE_TIMEOUT_S = 30


class EvalEnvironmentError(RuntimeError):
    """The fixed-version managed eval environment could not be provisioned."""


@dataclass
class KernelEnvInfo:
    """Resolved kernel interpreter and provisioning outcome."""

    python: str
    env_path: Optional[Path]
    bundled: bool
    provisioned_now: bool
    note: Optional[str] = None
    pip_install_cmd: List[str] = field(default_factory=list)
    bundle: List[str] = field(default_factory=list)


def default_state_dir() -> Path:
    """User state dir for kolega-code (mirrors cli.session_store.default_state_dir)."""
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


def _version_is_target(version: str) -> bool:
    return version.split(".")[:2] == EVAL_PYTHON_VERSION.split(".")


class EvalEnvironmentManager:
    """Provisions and resolves the fixed-version eval Python environment."""

    def __init__(
        self,
        *,
        state_dir: Optional[Path] = None,
        env_path: Optional[Path] = None,
        extra_packages: Optional[List[str]] = None,
        override_python: Optional[str] = None,
        legacy_python_version: Optional[str] = None,
    ) -> None:
        self._state_dir = Path(state_dir) if state_dir else default_state_dir()
        self._env_root = Path(env_path) if env_path else self._state_dir
        self._extra_packages = list(extra_packages or [])
        self._override_python = override_python
        self._legacy_python_version = legacy_python_version
        self._cached: Optional[KernelEnvInfo] = None
        self._env_path = self._env_root / f"{ENV_DIR_PREFIX}-{self._bundle_hash()[:16]}"

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
        try:
            data = json.loads((self._env_path / MANIFEST_NAME).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return data if isinstance(data, dict) else None

    def _write_manifest(self, installer: str, actual_python_version: str) -> None:
        manifest = {
            "python_version": EVAL_PYTHON_VERSION,
            "actual_python_version": actual_python_version,
            "bundle_hash": self._bundle_hash(),
            "installer": installer,
        }
        (self._env_path / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    def _manifest_matches(self) -> bool:
        manifest = self._read_manifest()
        if manifest is None:
            return False
        actual = manifest.get("actual_python_version")
        return (
            manifest.get("bundle_hash") == self._bundle_hash()
            and manifest.get("python_version") == EVAL_PYTHON_VERSION
            and isinstance(actual, str)
            and _version_is_target(actual)
            and _venv_python(self._env_path).exists()
        )

    def _pip_install_cmd(self, installer: str, python: str) -> List[str]:
        uv = shutil.which("uv")
        if installer == "uv":
            return [uv, "pip", "install", "--python", python] if uv else []
        return [python, "-m", "pip", "install"]

    def _with_deprecation_note(self, note: Optional[str]) -> Optional[str]:
        if self._legacy_python_version is None:
            return note
        deprecation = (
            "eval_python_version is deprecated and ignored; the managed eval environment uses "
            f"CPython {EVAL_PYTHON_VERSION}. Use eval_python_path for an explicit unbundled interpreter."
        )
        return f"{note}\n{deprecation}" if note else deprecation

    def _probe_python(self, python: str) -> str:
        code = "import json, platform; print(json.dumps([platform.python_implementation(), platform.python_version()]))"
        try:
            step = _run([python, "-I", "-c", code], timeout=_PROBE_TIMEOUT_S)
        except (OSError, subprocess.SubprocessError) as exc:
            raise RuntimeError(f"interpreter probe could not run: {exc}") from exc
        if step.returncode != 0:
            raise RuntimeError(f"interpreter probe failed: {step.stderr[-500:]}")
        try:
            implementation, version = json.loads(step.stdout)
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            raise RuntimeError(f"interpreter probe returned invalid output: {step.stdout[-500:]}") from exc
        if implementation != "CPython" or not isinstance(version, str) or not _version_is_target(version):
            raise RuntimeError(f"expected CPython {EVAL_PYTHON_VERSION}, found {implementation} {version}")
        return version

    def _verify_bundle(self, python: str) -> None:
        if not self._bundle_source().strip():
            return
        try:
            step = _run([python, "-I", "-c", "import numpy, pandas, matplotlib, PIL"], timeout=_PROBE_TIMEOUT_S)
        except (OSError, subprocess.SubprocessError) as exc:
            raise RuntimeError(f"bundled-library smoke test could not run: {exc}") from exc
        if step.returncode != 0:
            raise RuntimeError(f"bundled-library smoke test failed: {step.stderr[-500:]}")

    def _existing_info(self, manifest: dict) -> KernelEnvInfo:
        python = str(_venv_python(self._env_path))
        self._probe_python(python)
        self._verify_bundle(python)
        installer = str(manifest.get("installer") or "pip")
        return KernelEnvInfo(
            python=python,
            env_path=self._env_path,
            bundled=True,
            provisioned_now=False,
            note=self._with_deprecation_note(None),
            pip_install_cmd=self._pip_install_cmd(installer, python),
            bundle=self._bundle_names(),
        )

    async def ensure(self) -> KernelEnvInfo:
        """Resolve the kernel interpreter, provisioning the env on first use."""
        if self._cached is not None:
            return self._cached
        import asyncio

        info = await asyncio.to_thread(self._ensure_sync)
        self._cached = info
        return info

    def _reuse_existing(self) -> Optional[KernelEnvInfo]:
        manifest = self._read_manifest()
        if manifest is None or not self._manifest_matches():
            return None
        try:
            return self._existing_info(manifest)
        except RuntimeError as exc:
            raise EvalEnvironmentError(
                f"The managed eval environment at {self._env_path} failed verification: {exc}. "
                "Close sessions using it and delete that directory to rebuild it."
            ) from exc

    def _ensure_sync(self) -> KernelEnvInfo:
        if self._override_python:
            try:
                probe = _run([self._override_python, "--version"], timeout=_PROBE_TIMEOUT_S)
            except (OSError, subprocess.SubprocessError):
                probe = None
            if probe is not None and probe.returncode == 0:
                return KernelEnvInfo(
                    python=self._override_python,
                    env_path=None,
                    bundled=False,
                    provisioned_now=False,
                    note=self._with_deprecation_note(
                        "eval kernel is using the configured custom interpreter "
                        f"({self._override_python}); the bundled libraries are not installed there. "
                        "Use pip_install(package) to add packages."
                    ),
                    pip_install_cmd=[self._override_python, "-m", "pip", "install"],
                )

        existing = self._reuse_existing()
        if existing is not None:
            return existing

        lock_path = self._state_dir / LOCK_FILE_NAME
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with filelock.FileLock(str(lock_path)):
            existing = self._reuse_existing()
            if existing is not None:
                return existing
            if self._env_path.exists() and self._read_manifest() is not None:
                raise EvalEnvironmentError(
                    f"The managed eval environment at {self._env_path} has an invalid manifest. "
                    "Close sessions using it and delete that directory to rebuild it."
                )
            return self._provision_locked()

    def _create_env(self) -> Tuple[str, str]:
        self._env_path.parent.mkdir(parents=True, exist_ok=True)
        bundle_file = self._materialize_bundle_file()
        uv = shutil.which("uv")
        if uv:
            step = _run([uv, "venv", str(self._env_path), "--python", EVAL_PYTHON_VERSION], timeout=_VENV_TIMEOUT_S)
            if step.returncode != 0:
                raise RuntimeError(f"uv venv failed: {step.stderr[-500:]}")
            installer = "uv"
        else:
            system_python = shutil.which(f"python{EVAL_PYTHON_VERSION}")
            if system_python is None and sys.version_info[:2] == tuple(map(int, EVAL_PYTHON_VERSION.split("."))):
                system_python = sys.executable
            if system_python is not None:
                venv_cmd = [system_python, "-m", "venv", str(self._env_path)]
            elif sys.platform == "win32" and shutil.which("py"):
                venv_cmd = ["py", "-3.14", "-m", "venv", str(self._env_path)]
            else:
                raise RuntimeError("uv was not found and no CPython 3.14 executable is available on PATH")
            step = _run(venv_cmd, timeout=_VENV_TIMEOUT_S)
            if step.returncode != 0:
                raise RuntimeError(f"python -m venv failed: {step.stderr[-500:]}")
            installer = "pip"

        python = str(_venv_python(self._env_path))
        actual_version = self._probe_python(python)
        if installer == "uv":
            assert uv is not None
            step = _run(
                [uv, "pip", "install", "--python", python, "-r", str(bundle_file)],
                timeout=_INSTALL_TIMEOUT_S,
            )
        else:
            step = _run([python, "-m", "pip", "install", "-r", str(bundle_file)], timeout=_INSTALL_TIMEOUT_S)
        if step.returncode != 0:
            raise RuntimeError(f"bundle installation failed: {step.stderr[-500:]}")

        self._verify_bundle(python)
        self._write_manifest(installer, actual_version)
        return installer, actual_version

    def _provision_locked(self) -> KernelEnvInfo:
        try:
            if self._env_path.exists():
                shutil.rmtree(self._env_path)
            installer, actual_version = self._create_env()
        except Exception as exc:
            shutil.rmtree(self._env_path, ignore_errors=True)
            raise EvalEnvironmentError(
                f"Could not provision the managed CPython {EVAL_PYTHON_VERSION} eval environment: {exc}. "
                "Install uv or CPython 3.14 and retry, or set eval_python_path to an explicit unbundled "
                "interpreter. Previous environments were left unchanged."
            ) from exc

        python = str(_venv_python(self._env_path))
        return KernelEnvInfo(
            python=python,
            env_path=self._env_path,
            bundled=True,
            provisioned_now=True,
            note=self._with_deprecation_note(
                f"Provisioned the managed CPython {actual_version} eval environment with the bundled "
                "data libraries; this is a one-time setup."
            ),
            pip_install_cmd=self._pip_install_cmd(installer, python),
            bundle=self._bundle_names(),
        )
