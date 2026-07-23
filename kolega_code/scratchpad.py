"""Per-session scratchpad working directories under the OS temp dir.

The scratchpad is the one sanctioned location outside the project working
directory where agents may create files without asking: throwaway scripts,
ad-hoc virtual environments, downloads, extracted archives, and intermediate
outputs that must not pollute the user's working tree.

Scratchpads are deliberately *not* durable state. They live under the OS temp
dir keyed by user, project identity, and session id, and the OS reclaims them
on its own schedule. Kolega Code never sweeps or migrates them, and agents are
told (via the prompt extension) not to place deliverables there.
"""

from __future__ import annotations

import getpass
import os
import re
import tempfile
from pathlib import Path

from kolega_code.local_state import ensure_private_dir
from kolega_code.memory.identity import resolve_project_identity

# Reserved prompt-extension id. Hosts and the BaseAgent fallback check for this
# id so the section is never injected twice.
SCRATCHPAD_PROMPT_EXTENSION_ID = "kolega-session-scratchpad"

_SCRATCHPAD_DIRNAME = "scratchpad"


def _user_suffix() -> str:
    """Return a per-user path suffix for shared temp dirs."""
    getuid = getattr(os, "getuid", None)
    if callable(getuid):
        return str(getuid())
    # Windows has no getuid; its temp dir is already per-user, but keep the
    # path shape uniform across platforms.
    try:
        name = getpass.getuser()
    except Exception:  # pragma: no cover - getuser rarely fails, but never break a session over it
        name = ""
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip(".-")
    return slug or "user"


def scratchpad_root() -> Path:
    """Return the per-user root under the OS temp dir for all scratchpads."""
    return Path(tempfile.gettempdir()) / f"kolega-code-{_user_suffix()}"


def _validate_session_id(session_id: str) -> str:
    """Return ``session_id`` stripped, or raise when unsafe as a path component."""
    normalized = (session_id or "").strip()
    if not normalized or normalized in {".", ".."} or "/" in normalized or "\\" in normalized or "\0" in normalized:
        raise ValueError(f"session_id is not a safe path component: {session_id!r}")
    return normalized


def scratchpad_dir_for(project_path: Path | str, session_id: str) -> Path:
    """Return the deterministic per-session scratchpad directory for a project.

    The project component reuses the project-memory identity key, so linked
    Git worktrees share one namespace while separate clones do not; the
    session id provides the per-session scope. The same project + session id
    always maps to the same directory, including across session resumes.
    """
    identity = resolve_project_identity(project_path)
    return scratchpad_root() / identity.directory_key / _validate_session_id(session_id) / _SCRATCHPAD_DIRNAME


def ensure_scratchpad_dir(project_path: Path | str, session_id: str) -> Path:
    """Create the session scratchpad directory (owner-only) and return its path.

    Idempotent and cheap; intended to run whenever the scratchpad prompt
    extension is (re)built. Callers treat this as best-effort and catch
    ``OSError`` to run without a scratchpad.
    """
    path = scratchpad_dir_for(project_path, session_id)
    ensure_private_dir(path)
    return path
