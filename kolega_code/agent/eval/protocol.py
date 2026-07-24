"""NDJSON wire protocol between the host and eval kernel subprocesses.

One JSON object per line in each direction over the kernel's stdin/stdout.

Host -> kernel frames:
    {"type": "exec", "id": int, "run": str, "code": str, "cwd": str, "silent": bool}
    {"type": "exit"}

Kernel -> host frames:
    {"type": "started", "id": int}
    {"type": "stdout",  "id": int, "data": str}
    {"type": "stderr",  "id": int, "data": str}
    {"type": "display", "id": int, "bundle": {<mime>: <value>}}
    {"type": "result",  "id": int, "bundle": {<mime>: <value>}}
    {"type": "status",  "id": int, "event": {<op>: ...}}
    {"type": "error",   "id": int, "ename": str, "evalue": str, "traceback": [str]}
    {"type": "done",    "id": int, "status": "ok" | "error"}
"""

from __future__ import annotations

import json
from typing import Any, Dict

# Frame type constants (host side; the kernels mirror these literals).
FRAME_EXEC = "exec"
FRAME_EXIT = "exit"

FRAME_STARTED = "started"
FRAME_STDOUT = "stdout"
FRAME_STDERR = "stderr"
FRAME_DISPLAY = "display"
FRAME_RESULT = "result"
FRAME_STATUS = "status"
FRAME_ERROR = "error"
FRAME_DONE = "done"

STATUS_OK = "ok"
STATUS_ERROR = "error"

# StreamReader limit for kernel stdout. Display bundles carry base64 images, so
# frames are routinely larger than the asyncio default of 64 KiB.
FRAME_STREAM_LIMIT = 16 * 1024 * 1024

# Largest single frame the host will parse. Generous because of images, but
# bounded so a runaway kernel cannot exhaust host memory with one line.
MAX_FRAME_BYTES = 64 * 1024 * 1024


class ProtocolError(Exception):
    """A kernel emitted a line that is not a well-formed protocol frame."""


def encode_frame(frame: Dict[str, Any]) -> bytes:
    """Serialize one frame as a single NDJSON line (UTF-8, newline terminated)."""
    return (json.dumps(frame, ensure_ascii=False) + "\n").encode("utf-8", "replace")


def parse_frame(line: bytes) -> Dict[str, Any]:
    """Parse one NDJSON line from a kernel into a frame dict.

    Raises:
        ProtocolError: If the line is not valid JSON or not an object.
    """
    if len(line) > MAX_FRAME_BYTES:
        raise ProtocolError(f"kernel frame exceeds {MAX_FRAME_BYTES} bytes")
    try:
        frame = json.loads(line)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"malformed kernel frame: {line[:200]!r}") from exc
    if not isinstance(frame, dict) or not isinstance(frame.get("type"), str):
        raise ProtocolError(f"kernel frame is not an object with a type: {line[:200]!r}")
    return frame
