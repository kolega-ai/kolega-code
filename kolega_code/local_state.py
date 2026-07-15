"""Helpers for safely writing local Kolega Code state files."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

PRIVATE_DIR_MODE = 0o700
PRIVATE_FILE_MODE = 0o600


def ensure_private_dir(path: Path) -> None:
    """Create a local state directory and make it owner-only when supported."""
    missing: list[Path] = []
    cursor = path
    while not cursor.exists():
        missing.append(cursor)
        if cursor.parent == cursor:
            break
        cursor = cursor.parent
    path.mkdir(parents=True, exist_ok=True)
    for created in reversed(missing):
        _chmod(created, PRIVATE_DIR_MODE)
    _chmod(path, PRIVATE_DIR_MODE)


def write_private_text(path: Path, content: str, *, encoding: str = "utf-8") -> None:
    """Atomically write text to a file and make the final file owner-only."""
    write_private_bytes(path, content.encode(encoding))


def write_private_bytes(path: Path, content: bytes) -> None:
    """Atomically write binary local state with owner-only permissions."""
    ensure_private_dir(path.parent)
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temp = Path(temp_name)
    try:
        _chmod(temp, PRIVATE_FILE_MODE)
        view = memoryview(content)
        while view:
            written = os.write(fd, view)
            if written <= 0:
                raise OSError("short write while persisting local state")
            view = view[written:]
        os.fsync(fd)
        os.close(fd)
        fd = -1
        temp.replace(path)
        _chmod(path, PRIVATE_FILE_MODE)
        _fsync_directory(path.parent)
    finally:
        if fd >= 0:
            try:
                os.close(fd)
            except OSError:
                pass
        try:
            temp.unlink(missing_ok=True)
        except OSError:
            pass


def write_private_secret_text(path: Path, content: str, *, encoding: str = "utf-8") -> None:
    """Atomically write deliberate local credential state to an owner-only file.

    This mirrors the existing private-file provider API key model: data is kept
    local with POSIX owner-only permissions where supported, but it is not
    encrypted beyond filesystem/OS protections.
    """
    # Intentional credential persistence for local OAuth state. Keep this
    # suppression at the credential-specific sink so generic private writes
    # continue to be analyzed normally.
    # codeql[py/clear-text-storage-sensitive-data]
    write_private_bytes(path, content.encode(encoding))


def ensure_private_file(path: Path) -> None:
    """Best-effort chmod for an existing local state file."""
    _chmod(path, PRIVATE_FILE_MODE)


def _chmod(path: Path, mode: int) -> None:
    try:
        os.chmod(path, mode)
    except OSError:
        # Windows and some mounted filesystems may not support POSIX mode changes.
        pass


def _fsync_directory(path: Path) -> None:
    """Flush a replaced directory entry where the platform supports it."""
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        fd = os.open(path, flags)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        # Some filesystems do not support directory fsync.
        pass
    finally:
        os.close(fd)
