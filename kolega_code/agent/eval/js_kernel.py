"""JavaScript eval kernel: a Bun/Node subprocess with a persistent vm context."""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Dict, List, Optional

from .kernel import BaseKernel, _cfg_str

_RUNNER_RESOURCE = "runner.js"
_PRELUDE_RESOURCE = "prelude.js"
_MIN_NODE_MAJOR = 18


@dataclass(frozen=True)
class JsRuntime:
    """A probed JavaScript runtime executable."""

    name: str  # "bun" | "node"
    path: str

    def npm_install_cmd(self, home: Path) -> List[str]:
        if self.name == "bun":
            return [self.path, "add", "--cwd", str(home)]
        npm = shutil.which("npm")
        if not npm:
            return []
        return [npm, "install", "--prefix", str(home), "--no-audit", "--no-fund"]


def _probe_node_version(path: str) -> bool:
    try:
        proc = subprocess.run([path, "--version"], capture_output=True, text=True, timeout=15, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return False
    version = (proc.stdout or "").strip().lstrip("v")
    try:
        major = int(version.split(".", 1)[0])
    except (ValueError, IndexError):
        return False
    return major >= _MIN_NODE_MAJOR


def probe_js_runtime(config: object) -> Optional[JsRuntime]:
    """Find a JS runtime: explicit config, else bun preferred, then node >= 18."""
    preferred = _cfg_str(config, "eval_js_runtime")
    candidates = [preferred] if preferred else ["bun", "node"]
    for candidate in candidates:
        assert candidate is not None
        if os.sep in candidate or (os.altsep and os.altsep in candidate):
            path: Optional[str] = candidate if os.path.isfile(candidate) else None
        else:
            path = shutil.which(candidate)
        if not path:
            continue
        name = "bun" if "bun" in os.path.basename(path).lower() else "node"
        if name == "node" and not _probe_node_version(path):
            continue
        return JsRuntime(name=name, path=path)
    return None


def _resource_text(name: str) -> str:
    return files("kolega_code.agent.eval").joinpath(name).read_text(encoding="utf-8")


def materialize_runner(state_dir: Path) -> Path:
    """Write the JS runner script next to the Python one (content-hash cached)."""
    source = _resource_text(_RUNNER_RESOURCE)
    digest = hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]
    target_dir = state_dir / "eval-kernels"
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"runner-{digest}.js"
    if not target.exists():
        target.write_text(source, encoding="utf-8")
    return target


class JsKernel(BaseKernel):
    """Persistent JavaScript kernel (one subprocess, NDJSON cell protocol)."""

    def __init__(self, *, cwd: str, env: Dict[str, str], runtime: JsRuntime, state_dir: Path) -> None:
        super().__init__(language="js", cwd=cwd, env=env)
        self._runtime = runtime
        self._runner_path = materialize_runner(state_dir)

    def argv(self) -> List[str]:
        return [self._runtime.path, str(self._runner_path)]

    def init_cells(self) -> List[str]:
        return [_resource_text(_PRELUDE_RESOURCE)]
