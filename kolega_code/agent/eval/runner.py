"""Kolega Code eval-kernel runner (Python).

Spawned as ``python -u runner.py`` by the host. Speaks NDJSON over stdin/stdout:
one JSON frame per line. See protocol.py in the same package for the frame
shapes. Stdlib only; requires Python 3.11+.

Key behaviors:
- Persistent global namespace across cells.
- Top-level ``await`` via ast.PyCF_ALLOW_TOP_LEVEL_AWAIT; coroutine cells are
  driven on one persistent event loop so loop-bound objects survive cells.
- REPL echo: the last top-level expression is captured and emitted as a result
  frame (when not None).
- print()/logging are captured into stdout/stderr frames; protocol frames go to
  the real fd 1 saved before streams are replaced.
- SIGINT raises KeyboardInterrupt inside the running cell.
"""

from __future__ import annotations

import ast
import asyncio
import base64
import inspect
import json
import linecache
import os
import sys
import traceback
from typing import Any, Optional

# The runner's own directory is added to sys.path[0] by the interpreter; remove
# it so cell imports only see the project and normal site-packages.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if sys.path and os.path.abspath(sys.path[0]) == _SCRIPT_DIR:
    sys.path.pop(0)

_REAL_STDOUT = sys.stdout

_CURRENT = {"id": 0, "run": ""}
_LOOP: Optional[asyncio.AbstractEventLoop] = None
_NS: dict = {}

# Cap individual traceback payloads so a huge recursive failure cannot flood
# the host with frames.
_MAX_TRACEBACK_LINES = 60
_MAX_REPR_CHARS = 4000


def _emit(frame: dict) -> None:
    data = json.dumps(frame, ensure_ascii=False, default=str).encode("utf-8", "replace") + b"\n"
    _REAL_STDOUT.buffer.write(data)
    _REAL_STDOUT.buffer.flush()


def _current_id() -> int:
    return _CURRENT["id"]


class _FrameStream:
    """sys.stdout/sys.stderr replacement emitting stdout/stderr frames."""

    def __init__(self, kind: str) -> None:
        self._kind = kind
        self._buf = ""

    def write(self, text: Any) -> int:
        if not isinstance(text, str):
            text = str(text)
        self._buf += text
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            _emit({"type": self._kind, "id": _current_id(), "data": line + "\n"})
        return len(text)

    def writelines(self, lines: Any) -> None:
        for line in lines:
            self.write(line)

    def flush(self) -> None:
        if self._buf:
            _emit({"type": self._kind, "id": _current_id(), "data": self._buf})
            self._buf = ""

    def isatty(self) -> bool:
        return False

    def writable(self) -> bool:
        return True

    def fileno(self) -> int:
        raise OSError("eval kernel streams have no file descriptor")


def _safe_repr(value: Any) -> str:
    try:
        text = repr(value)
    except Exception:
        text = f"<{type(value).__name__} object>"
    if len(text) > _MAX_REPR_CHARS:
        text = text[:_MAX_REPR_CHARS] + f"\n[…{len(text) - _MAX_REPR_CHARS}ch elided…]"
    return text


def _try_b64(data: Any) -> Optional[str]:
    if isinstance(data, str):
        return data
    if isinstance(data, (bytes, bytearray)):
        return base64.b64encode(bytes(data)).decode("ascii")
    return None


def _mime_bundle(value: Any) -> dict:
    """Best-effort IPython-style MIME bundle for a value."""
    bundle: dict = {}

    if isinstance(value, (dict, list, int, float, bool)):
        try:
            json.dumps(value)
            bundle["application/json"] = value
        except (TypeError, ValueError):
            pass

    repr_json = getattr(value, "_repr_json_", None)
    if callable(repr_json):
        try:
            parsed = repr_json()
            if isinstance(parsed, str):
                parsed = json.loads(parsed)
            json.dumps(parsed)
            bundle.setdefault("application/json", parsed)
        except Exception:
            pass

    for attr, mime in (
        ("_repr_png_", "image/png"),
        ("_repr_jpeg_", "image/jpeg"),
        ("_repr_svg_", "image/svg+xml"),
        ("_repr_html_", "text/html"),
        ("_repr_markdown_", "text/markdown"),
    ):
        renderer = getattr(value, attr, None)
        if not callable(renderer):
            continue
        try:
            rendered = renderer()
        except Exception:
            continue
        if mime in ("image/png", "image/jpeg"):
            rendered = _try_b64(rendered)
        if isinstance(rendered, str) and rendered:
            bundle.setdefault(mime, rendered)

    # matplotlib Figure (and lookalikes) without rich reprs: render via savefig.
    savefig = getattr(value, "savefig", None)
    if "image/png" not in bundle and callable(savefig):
        try:
            import io

            buffer = io.BytesIO()
            savefig(buffer, format="png", bbox_inches="tight")
            bundle["image/png"] = base64.b64encode(buffer.getvalue()).decode("ascii")
        except Exception:
            pass

    bundle.setdefault("text/plain", _safe_repr(value))
    return bundle


def _display(value: Any, raw: bool = False) -> None:
    """Injected as __kolega_display__; the prelude's display() wraps it."""
    if raw and isinstance(value, dict):
        bundle = value
    else:
        bundle = _mime_bundle(value)
    _emit({"type": "display", "id": _current_id(), "bundle": bundle})


def _status(op: str, **data: Any) -> None:
    """Injected as __kolega_status__ for prelude log()/phase()."""
    _emit({"type": "status", "id": _current_id(), "event": {"op": op, **data}})


def _emit_error(msg_id: int, exc: BaseException) -> None:
    tb_lines = traceback.format_exception(type(exc), exc, exc.__traceback__)
    if len(tb_lines) > _MAX_TRACEBACK_LINES:
        tb_lines = tb_lines[:_MAX_TRACEBACK_LINES] + [f"[…{len(tb_lines) - _MAX_TRACEBACK_LINES} lines elided…]\n"]
    _emit(
        {
            "type": "error",
            "id": msg_id,
            "ename": type(exc).__name__,
            "evalue": str(exc),
            "traceback": tb_lines,
        }
    )


_RESULT_NAME = "__kolega_result__"


def _compile_cell(code: str, msg_id: int, *, silent: bool):
    filename = f"<cell-{msg_id}>"
    # Register the source so tracebacks show cell code lines (like IPython).
    linecache.cache[filename] = (len(code), None, code.splitlines(keepends=True), filename)
    tree = ast.parse(code, filename=filename, mode="exec")
    want_result = False
    if not silent and tree.body and isinstance(tree.body[-1], ast.Expr):
        last = tree.body.pop()
        assert isinstance(last, ast.Expr)
        assign = ast.Assign(
            targets=[ast.Name(id=_RESULT_NAME, ctx=ast.Store())],
            value=last.value,
        )
        ast.copy_location(assign, last)
        tree.body.append(assign)
        want_result = True
    ast.fix_missing_locations(tree)
    compiled = compile(tree, filename, "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
    return compiled, want_result


def _run_cell(code: str, msg_id: int, *, silent: bool) -> None:
    if _LOOP is None:
        raise RuntimeError("eval kernel event loop is not running")
    _NS.pop(_RESULT_NAME, None)
    compiled, want_result = _compile_cell(code, msg_id, silent=silent)
    outcome = eval(compiled, _NS)
    if inspect.iscoroutine(outcome):
        _LOOP.run_until_complete(outcome)
    if want_result:
        value = _NS.get(_RESULT_NAME)
        if value is not None:
            _emit({"type": "result", "id": msg_id, "bundle": _mime_bundle(value)})


def _execute(msg: dict) -> None:
    msg_id = int(msg.get("id") or 0)
    _CURRENT["id"] = msg_id
    _CURRENT["run"] = str(msg.get("run") or "")
    _NS["__kolega_run_id__"] = _CURRENT["run"]
    cwd = msg.get("cwd")
    if isinstance(cwd, str) and cwd:
        try:
            os.chdir(cwd)
        except OSError as exc:
            _emit({"type": "stderr", "id": msg_id, "data": f"[kernel] chdir failed: {exc}\n"})
    silent = bool(msg.get("silent"))

    _emit({"type": "started", "id": msg_id})
    status = "ok"
    try:
        _run_cell(str(msg.get("code") or ""), msg_id, silent=silent)
    except KeyboardInterrupt as exc:
        status = "error"
        _emit_error(msg_id, exc)
    except BaseException as exc:  # every cell failure is a result, not a runner crash
        status = "error"
        _emit_error(msg_id, exc)
    finally:
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:
            pass
        _emit({"type": "done", "id": msg_id, "status": status})
        _CURRENT["run"] = ""
        _NS["__kolega_run_id__"] = ""


def main() -> None:
    global _LOOP, _NS
    _LOOP = asyncio.new_event_loop()
    _NS = {
        "__name__": "__eval__",
        "__kolega_display__": _display,
        "__kolega_status__": _status,
        "__kolega_run_id__": "",
    }
    sys.stdout = _FrameStream("stdout")
    sys.stderr = _FrameStream("stderr")

    stdin = sys.stdin.buffer
    while True:
        line = stdin.readline()
        if not line:
            break
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(msg, dict):
            continue
        frame_type = msg.get("type")
        if frame_type == "exit":
            break
        if frame_type == "exec":
            _execute(msg)

    try:
        if _LOOP is not None:
            _LOOP.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()
