"""Kolega Code eval-kernel prelude (Python).

Executed once in the kernel's persistent namespace at startup (sent by the host
as a silent init cell). Defines the model-facing in-kernel API: display(),
read()/write()/env(), the tool.<name>(args) bridge proxy, list_tools(),
parallel(), pip_install(), python_info(), log()/phase().

Stdlib only. Runner provides __kolega_display__, __kolega_status__, and the
__kolega_run_id__ global (updated per cell).
"""

from __future__ import annotations

import json as _json
import os as _os
import subprocess as _subprocess
import urllib.error as _urlerror
import urllib.request as _urlrequest
from concurrent.futures import ThreadPoolExecutor as _ThreadPoolExecutor
from pathlib import Path as _Path


def display(value):
    """Render a value as a rich output (images, HTML, JSON) in the tool result."""
    # __kolega_display__ is injected into the kernel namespace by the runner.
    globals()["__kolega_display__"](value)


def _emit_status(op, **data):
    # __kolega_status__ is injected into the kernel namespace by the runner.
    globals()["__kolega_status__"](op, **data)


def log(message):
    """Add a note to this cell's status output."""
    _emit_status("log", message=str(message))


def phase(title):
    """Mark a named phase of this cell in its status output."""
    _emit_status("phase", title=str(title))


def env(key=None, value=None):
    """Get or set kernel environment variables."""
    if key is None:
        return dict(sorted(_os.environ.items()))
    if value is not None:
        _os.environ[str(key)] = str(value)
        return value
    return _os.environ.get(str(key))


def read(path, offset=1, limit=None):
    """Read a file's contents. offset/limit are 1-indexed lines."""
    p = _Path(path)
    data = p.read_text(encoding="utf-8")
    if offset > 1 or limit is not None:
        lines = data.splitlines(keepends=True)
        start = max(0, int(offset) - 1)
        end = start + int(limit) if limit else len(lines)
        data = "".join(lines[start:end])
    _emit_status("read", path=str(p), chars=len(data))
    return data


def write(path, content):
    """Write text to a file, creating parent directories. Returns the path."""
    p = _Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(str(content), encoding="utf-8")
    _emit_status("write", path=str(p), chars=len(str(content)))
    return p


def _bridge_config():
    base = _os.environ.get("KOLEGA_TOOL_BRIDGE_URL")
    token = _os.environ.get("KOLEGA_TOOL_BRIDGE_TOKEN")
    session = _os.environ.get("KOLEGA_TOOL_BRIDGE_SESSION")
    if not base or not token or not session:
        raise RuntimeError("tool bridge is unavailable in this kernel")
    return base.rstrip("/"), token, session


def _bridge_call(name, args):
    """POST one request to the host tool bridge and return its value."""
    base, token, session = _bridge_config()
    run_id = globals().get("__kolega_run_id__") or ""
    payload = _json.dumps({"session": session, "run": run_id, "name": name, "args": args}).encode("utf-8")
    request = _urlrequest.Request(
        f"{base}/v1/tool",
        data=payload,
        method="POST",
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"},
    )
    try:
        with _urlrequest.urlopen(request) as response:
            body = response.read()
    except _urlerror.HTTPError as exc:
        body = exc.read()
    except _urlerror.URLError as exc:
        raise RuntimeError(f"bridge call {name!r} failed: {exc}") from None
    try:
        data = _json.loads(body)
    except _json.JSONDecodeError:
        raise RuntimeError(f"bridge call {name!r}: non-JSON response: {body[:200]!r}") from None
    if not isinstance(data, dict) or not data.get("ok"):
        message = data.get("error") if isinstance(data, dict) else None
        raise RuntimeError(message or f"bridge call {name!r} failed")
    return data.get("value")


def _unwrap_tool_value(value):
    """Tool results are plain text unless images are attached."""
    if isinstance(value, dict) and "text" in value and set(value) <= {"text", "images"}:
        if value.get("images"):
            return value
        return value["text"]
    return value


class _ToolCallable:
    """Invokes one host-side tool via the loopback bridge."""

    __slots__ = ("_name",)

    def __init__(self, name):
        self._name = name

    def __repr__(self):
        return f"<tool.{self._name}>"

    def __call__(self, args=None, /, **kwargs):
        if args is None:
            merged = {}
        elif isinstance(args, dict):
            merged = dict(args)
        else:
            raise TypeError(f"tool.{self._name}(...) expects a dict of arguments (got {type(args).__name__})")
        merged.update(kwargs)
        return _unwrap_tool_value(_bridge_call(self._name, merged))


class _ToolProxy:
    """``tool.<name>(args)`` proxy over the host's tool registry."""

    __slots__ = ()

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        return _ToolCallable(name)

    def __getitem__(self, name):
        return _ToolCallable(name)

    def __dir__(self):
        try:
            return sorted(entry.get("name", "") for entry in list_tools() or [])
        except Exception:
            return []

    def __repr__(self):
        return "<tool proxy: call list_tools() to discover available tools>"


tool = _ToolProxy()


def list_tools():
    """List the tools available through `tool.<name>` as [{name, summary}]."""
    return _bridge_call("__list_tools__", {})


def parallel(thunks, max_workers=None):
    """Run callables concurrently (threads) and return their results in order."""
    thunks = list(thunks)
    with _ThreadPoolExecutor(max_workers=max_workers or max(1, len(thunks))) as pool:
        futures = [pool.submit(thunk) for thunk in thunks]
        return [future.result() for future in futures]


def pip_install(*packages):
    """Install packages into the kernel's dedicated Python environment."""
    if not packages:
        raise ValueError("pip_install(...) needs at least one package name")
    cmd = _json.loads(_os.environ.get("KOLEGA_EVAL_PIP_INSTALL_CMD") or "[]")
    if not cmd:
        raise RuntimeError(
            "pip_install is unavailable: the kernel is not running in a managed environment "
            "(custom interpreter or provisioning was skipped). Install packages into that "
            "interpreter yourself, e.g. via tool.exec_command."
        )
    proc = _subprocess.run(list(cmd) + [str(p) for p in packages], capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"pip_install failed (exit {proc.returncode}):\n{proc.stderr[-2000:]}")
    _emit_status("pip_install", packages=list(packages))
    return (proc.stdout or "")[-2000:]


def python_info():
    """Return interpreter version, env path, and bundled packages for this kernel."""
    import sys as _sys

    return {
        "version": _sys.version.split()[0],
        "executable": _sys.executable,
        "env_path": _os.environ.get("KOLEGA_EVAL_ENV_PATH") or None,
        "bundle": [p for p in (_os.environ.get("KOLEGA_EVAL_BUNDLE") or "").split(",") if p],
    }
