"""Best-effort async helper to read an image from the system clipboard.

This module is deliberately free of Textual imports (pure stdlib + asyncio)
so it can be imported and unit-tested in isolation. Callers are responsible
for surfacing failures to the user.

Returns ``(image_bytes, media_type)`` or ``None`` when no image is available
or no suitable clipboard tool exists on the platform.
"""

from __future__ import annotations

import asyncio
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Optional, Tuple

# A tiny 1x1 transparent PNG used to recognize "empty"/non-image clipboard
# results from tools that still exit 0. Anything smaller than this is treated
# as "no image".
_MIN_IMAGE_BYTES = 8


async def _run_exec(*args: str) -> Tuple[bytes, bytes, int]:
    """Run a subprocess, returning (stdout, stderr, returncode)."""
    proc = await asyncio.create_subprocess_exec(
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    return stdout or b"", stderr or b"", int(proc.returncode or 0)


async def _run_shell(cmd: str) -> Tuple[bytes, bytes, int]:
    """Run a shell command, returning (stdout, stderr, returncode)."""
    proc = await asyncio.create_subprocess_shell(
        cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    return stdout or b"", stderr or b"", int(proc.returncode or 0)


async def _read_macos_clipboard() -> Optional[Tuple[bytes, str]]:
    """macOS: use osascript to write clipboard PNG to a temp file."""
    tmp = Path(tempfile.gettempdir()) / f"kc_clip_{asyncio.get_event_loop().time()}.png"
    try:
        script = (
            f'set the_file to (POSIX file "{tmp}")\n'
            "try\n"
            "\tset png_data to the clipboard as \u00abclass PNGf\u00bb\n"
            "\ton error\n"
            "\t\treturn\n"
            "\tend try\n"
            "\tset f to open for access the_file with write permission\n"
            "\twrite png_data to f\n"
            "\tclose access f"
        )
        _stdout, _stderr, rc = await _run_exec("osascript", "-e", script)
        if rc == 0 and tmp.exists():
            data = tmp.read_bytes()
            if data and len(data) >= _MIN_IMAGE_BYTES:
                return data, "image/png"
        # Fallback to pngpaste if available
        if shutil.which("pngpaste") is not None:
            _stdout, _stderr, rc = await _run_exec("pngpaste", str(tmp))
            if rc == 0 and tmp.exists():
                data = tmp.read_bytes()
                if data and len(data) >= _MIN_IMAGE_BYTES:
                    return data, "image/png"
        return None
    except Exception:
        return None
    finally:
        tmp.unlink(missing_ok=True)


async def _read_linux_clipboard() -> Optional[Tuple[bytes, str]]:
    """Linux/WSL: try xclip, wl-paste, then powershell.exe (WSL)."""
    try:
        if shutil.which("xclip") is not None:
            stdout, _stderr, rc = await _run_exec("xclip", "-selection", "clipboard", "-t", "image/png", "-o")
            if rc == 0 and stdout and len(stdout) >= _MIN_IMAGE_BYTES:
                return stdout, "image/png"
        if shutil.which("wl-paste") is not None:
            stdout, _stderr, rc = await _run_exec("wl-paste", "--type", "image/png")
            if rc == 0 and stdout and len(stdout) >= _MIN_IMAGE_BYTES:
                return stdout, "image/png"
        if shutil.which("powershell.exe") is not None:
            tmp = Path(tempfile.gettempdir()) / f"kc_clip_{asyncio.get_event_loop().time()}.png"
            try:
                ps = (
                    "Add-Type -AssemblyName System.Windows.Forms;"
                    " $img = [System.Windows.Forms.Clipboard]::GetImage();"
                    f" if ($img) {{ $img.Save('{tmp}') }}"
                )
                _stdout, _stderr, rc = await _run_exec("powershell.exe", "-NoProfile", "-Command", ps)
                if rc == 0 and tmp.exists():
                    data = tmp.read_bytes()
                    if data and len(data) >= _MIN_IMAGE_BYTES:
                        return data, "image/png"
            finally:
                tmp.unlink(missing_ok=True)
        return None
    except Exception:
        return None


async def _read_windows_clipboard() -> Optional[Tuple[bytes, str]]:
    """Windows: use PowerShell to save the clipboard image to a temp PNG."""
    tmp = Path(tempfile.gettempdir()) / f"kc_clip_{asyncio.get_event_loop().time()}.png"
    try:
        ps = (
            "Add-Type -AssemblyName System.Windows.Forms;"
            " $img = [System.Windows.Forms.Clipboard]::GetImage();"
            f" if ($img) {{ $img.Save('{tmp}') }}"
        )
        _stdout, _stderr, rc = await _run_exec("powershell", "-NoProfile", "-Command", ps)
        if rc == 0 and tmp.exists():
            data = tmp.read_bytes()
            if data and len(data) >= _MIN_IMAGE_BYTES:
                return data, "image/png"
        return None
    except Exception:
        return None
    finally:
        tmp.unlink(missing_ok=True)


async def read_clipboard_image() -> Optional[Tuple[bytes, str]]:
    """Best-effort read of an image from the system clipboard.

    Returns ``(image_bytes, media_type)`` or ``None`` when no image is
    available or no suitable clipboard tool exists on the current platform.
    Never raises: all failures map to ``None``.
    """
    try:
        if sys.platform == "darwin":
            return await _read_macos_clipboard()
        if sys.platform.startswith("linux"):
            return await _read_linux_clipboard()
        if sys.platform == "win32":
            return await _read_windows_clipboard()
        return None
    except Exception:
        return None
