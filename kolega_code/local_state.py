"""Helpers for safely writing local Kolega Code state files."""

from __future__ import annotations

import os
from pathlib import Path

PRIVATE_DIR_MODE = 0o700
PRIVATE_FILE_MODE = 0o600


def ensure_private_dir(path: Path) -> None:
    """Create a local state directory and make it owner-only when supported."""
    path.mkdir(parents=True, exist_ok=True)
    _chmod(path, PRIVATE_DIR_MODE)


def write_private_text(path: Path, content: str, *, encoding: str = "utf-8") -> None:
    """Atomically write text to a file and make the final file owner-only."""
    ensure_private_dir(path.parent)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(content, encoding=encoding)
    _chmod(temp, PRIVATE_FILE_MODE)
    temp.replace(path)
    _chmod(path, PRIVATE_FILE_MODE)


def ensure_private_file(path: Path) -> None:
    """Best-effort chmod for an existing local state file."""
    _chmod(path, PRIVATE_FILE_MODE)


def _chmod(path: Path, mode: int) -> None:
    try:
        os.chmod(path, mode)
    except OSError:
        # Windows and some mounted filesystems may not support POSIX mode changes.
        pass
