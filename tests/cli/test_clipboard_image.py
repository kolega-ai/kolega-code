"""Unit tests for ``kolega_code.cli.clipboard_image``.

These tests mock at the subprocess boundary (``asyncio.create_subprocess_exec``
and ``asyncio.create_subprocess_shell``) so they never touch the real system
clipboard. They mirror the mocking pattern used in
``tests/agent/tool_backend/test_terminal_tool.py``.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

import kolega_code.cli.clipboard_image as clipboard_image
from kolega_code.cli.clipboard_image import read_clipboard_image

# A minimal valid PNG (1x1 red pixel) so the size check passes.
_PNG_BYTES = (
    b"\x89PNG\r\n\x1a\n"
    b"\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde"
    b"\x00\x00\x00\x0cIDATx\x9cc\xf8\xcf\xc0\x00\x00\x00\x03\x00\x01\x8a\xaf\x1e\x8e"
    b"\x00\x00\x00\x00IEND\xaeB`\x82"
)


def _make_proc(communicate_return=(b"", b""), returncode=0):
    """Build an AsyncMock resembling an asyncio subprocess."""
    proc = AsyncMock()
    proc.communicate = AsyncMock(return_value=communicate_return)
    proc.returncode = returncode
    return proc


@pytest.mark.asyncio
async def test_macos_returns_png_when_osascript_succeeds(tmp_path, monkeypatch):
    """On darwin, a successful osascript run writes a PNG file we read back."""
    monkeypatch.setattr(clipboard_image.sys, "platform", "darwin")

    # Make tempfile.gettempdir point at tmp_path so we control the temp file.
    monkeypatch.setattr(clipboard_image.tempfile, "gettempdir", lambda: str(tmp_path))

    # The osascript subprocess "succeeds"; the temp file must exist with PNG
    # bytes for the helper to read it. We pre-create the file the helper will
    # generate. Because the filename is dynamic, we intercept by writing the
    # PNG into whatever path the helper chose *after* the exec call via a
    # side effect.
    written_paths: list[str] = []

    async def _fake_exec(*args, **kwargs):
        # The osascript invocation embeds the temp .png path inside the
        # AppleScript string (as a POSIX file). Extract it and "write" the
        # clipboard PNG there so the helper can read it back.
        import re

        from pathlib import Path

        for a in args:
            if not isinstance(a, str):
                continue
            m = re.search(r"(/\S*kc_clip_\S*\.png)", a)
            if m:
                p = m.group(1)
                Path(p).write_bytes(_PNG_BYTES)
                written_paths.append(p)
                break
        return _make_proc(communicate_return=(b"", b""), returncode=0)

    monkeypatch.setattr(clipboard_image.asyncio, "create_subprocess_exec", _fake_exec)
    # No pngpaste fallback needed.
    monkeypatch.setattr(clipboard_image.shutil, "which", lambda _: None)

    result = await read_clipboard_image()
    assert result is not None
    data, media_type = result
    assert media_type == "image/png"
    assert data == _PNG_BYTES
    assert written_paths, "expected the helper to invoke osascript with a temp .png path"


@pytest.mark.asyncio
async def test_linux_no_tools_returns_none(monkeypatch):
    """On linux with no clipboard tools installed, returns None."""
    monkeypatch.setattr(clipboard_image.sys, "platform", "linux")
    monkeypatch.setattr(clipboard_image.shutil, "which", lambda _: None)

    result = await read_clipboard_image()
    assert result is None


@pytest.mark.asyncio
async def test_macos_osascript_failure_returns_none(tmp_path, monkeypatch):
    """When osascript returns a non-zero exit and pngpaste is absent, returns None."""
    monkeypatch.setattr(clipboard_image.sys, "platform", "darwin")
    monkeypatch.setattr(clipboard_image.tempfile, "gettempdir", lambda: str(tmp_path))
    monkeypatch.setattr(clipboard_image.asyncio, "create_subprocess_exec", lambda *a, **k: _make_proc(returncode=1))
    monkeypatch.setattr(clipboard_image.shutil, "which", lambda _: None)

    result = await read_clipboard_image()
    assert result is None


@pytest.mark.asyncio
async def test_subprocess_raises_returns_none(tmp_path, monkeypatch):
    """If subprocess creation raises, the helper swallows it and returns None."""
    monkeypatch.setattr(clipboard_image.sys, "platform", "darwin")
    monkeypatch.setattr(clipboard_image.tempfile, "gettempdir", lambda: str(tmp_path))

    async def _boom(*args, **kwargs):
        raise RuntimeError("osascript exploded")

    monkeypatch.setattr(clipboard_image.asyncio, "create_subprocess_exec", _boom)
    monkeypatch.setattr(clipboard_image.shutil, "which", lambda _: None)

    result = await read_clipboard_image()
    assert result is None


@pytest.mark.asyncio
async def test_unknown_platform_returns_none(monkeypatch):
    """An unrecognized platform yields None without raising."""
    monkeypatch.setattr(clipboard_image.sys, "platform", "freebsd9")

    result = await read_clipboard_image()
    assert result is None


@pytest.mark.asyncio
async def test_linux_xclip_returns_png(monkeypatch):
    """On linux with xclip, stdout PNG bytes are returned directly."""
    monkeypatch.setattr(clipboard_image.sys, "platform", "linux")

    def _which(name):
        return "/usr/bin/xclip" if name == "xclip" else None

    monkeypatch.setattr(clipboard_image.shutil, "which", _which)

    async def _fake_exec(*args, **kwargs):
        return _make_proc(communicate_return=(_PNG_BYTES, b""), returncode=0)

    monkeypatch.setattr(clipboard_image.asyncio, "create_subprocess_exec", _fake_exec)

    result = await read_clipboard_image()
    assert result is not None
    data, media_type = result
    assert media_type == "image/png"
    assert data == _PNG_BYTES


@pytest.mark.asyncio
async def test_macos_pngpaste_fallback(tmp_path, monkeypatch):
    """If osascript fails but pngpaste is present and writes a file, it is used."""
    monkeypatch.setattr(clipboard_image.sys, "platform", "darwin")
    monkeypatch.setattr(clipboard_image.tempfile, "gettempdir", lambda: str(tmp_path))

    call_count = {"n": 0}

    async def _fake_exec(*args, **kwargs):
        call_count["n"] += 1
        # First call: osascript -> fails (rc=1), no file.
        # Second call: pngpaste <path> -> succeeds, write the PNG to that path.
        if call_count["n"] == 2:
            for a in args:
                if isinstance(a, str) and a.endswith(".png") and "kc_clip_" in a:
                    from pathlib import Path

                    Path(a).write_bytes(_PNG_BYTES)
                    break
            return _make_proc(communicate_return=(b"", b""), returncode=0)
        return _make_proc(communicate_return=(b"", b"no image"), returncode=1)

    monkeypatch.setattr(clipboard_image.asyncio, "create_subprocess_exec", _fake_exec)

    def _which(name):
        return "/usr/local/bin/pngpaste" if name == "pngpaste" else None

    monkeypatch.setattr(clipboard_image.shutil, "which", _which)

    result = await read_clipboard_image()
    assert result is not None
    data, media_type = result
    assert media_type == "image/png"
    assert data == _PNG_BYTES
