"""Private CLI plan artifacts for preserving approved plans across compaction."""

from __future__ import annotations

from pathlib import Path

from kolega_code.local_state import ensure_private_dir, ensure_private_file, write_private_text

PLAN_ARTIFACTS_DIR = "plans"
CURRENT_PLAN_FILENAME = "current-plan.md"


def current_plan_artifact_path(state_root: Path, session_id: str) -> Path:
    """Return the deterministic artifact path for a session's current approved plan."""
    return Path(state_root).expanduser() / PLAN_ARTIFACTS_DIR / session_id / CURRENT_PLAN_FILENAME


def normalize_plan_artifact_content(plan_markdown: str) -> str:
    """Normalize plan Markdown before persisting it as an artifact."""
    stripped = plan_markdown.strip()
    return f"{stripped}\n" if stripped else ""


def write_current_plan_artifact(state_root: Path, session_id: str, plan_markdown: str) -> Path:
    """Persist ``plan_markdown`` to the session's private current-plan artifact.

    The session store remains the canonical source of the latest plan. This file is
    a readable, stable artifact that agents can re-open after conversation
    compaction has summarized away the original implementation prompt.
    """
    root = Path(state_root).expanduser()
    path = current_plan_artifact_path(root, session_id)
    content = normalize_plan_artifact_content(plan_markdown)
    ensure_private_dir(root)
    ensure_private_dir(root / PLAN_ARTIFACTS_DIR)
    ensure_private_dir(path.parent)

    if path.exists():
        try:
            if path.read_text(encoding="utf-8") == content:
                ensure_private_file(path)
                return path
        except OSError:
            # Fall through and rewrite atomically; write_private_text will surface
            # any real write failure to the caller.
            pass

    write_private_text(path, content)
    return path
