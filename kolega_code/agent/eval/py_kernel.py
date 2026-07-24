"""Python eval kernel: a subprocess on the dedicated managed environment."""

from __future__ import annotations

import hashlib
import json
from importlib.resources import files
from pathlib import Path
from typing import Dict, List

from .kernel import BaseKernel

_RUNNER_RESOURCE = "runner.py"
_PRELUDE_RESOURCE = "prelude.py"


def _resource_text(name: str) -> str:
    return files("kolega_code.agent.eval").joinpath(name).read_text(encoding="utf-8")


def materialize_runner(state_dir: Path) -> Path:
    """Write the runner script to a real file the subprocess can execute.

    Cached by content hash under the user state dir so wheels/zip installs work
    and repeated starts don't rewrite the file.
    """
    source = _resource_text(_RUNNER_RESOURCE)
    digest = hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]
    target_dir = state_dir / "eval-kernels"
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"runner-{digest}.py"
    if not target.exists():
        target.write_text(source, encoding="utf-8")
    return target


class PythonKernel(BaseKernel):
    """Persistent Python kernel (one subprocess, NDJSON cell protocol)."""

    def __init__(self, *, cwd: str, env: Dict[str, str], python_path: str, state_dir: Path) -> None:
        super().__init__(language="py", cwd=cwd, env=env)
        self._python_path = python_path
        self._runner_path = materialize_runner(state_dir)

    def argv(self) -> List[str]:
        return [self._python_path, "-u", str(self._runner_path)]

    def init_cells(self) -> List[str]:
        init = (
            "import os as _os, sys as _sys\n"
            f"_cwd = {json.dumps(self.cwd)}\n"
            "_os.chdir(_cwd)\n"
            "if _cwd not in _sys.path:\n"
            "    _sys.path.insert(0, _cwd)\n"
        )
        return [init, _resource_text(_PRELUDE_RESOURCE)]
