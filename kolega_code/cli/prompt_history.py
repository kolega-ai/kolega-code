"""Per-project prompt history persisted in the local state directory.

The composer's recall history (GitHub #622) is project-scoped and shell-like:
prompts submitted in any session under the project's state root are recallable
in later runs. The file is small (capped list of strings), so persistence is a
whole-file dump on every submit; concurrent instances last-write-win, which is
benign for a recall list.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

from kolega_code.local_state import write_private_text

PROMPT_HISTORY_FILENAME = "prompt_history.json"
PROMPT_HISTORY_MAX = 100


def prompt_history_path(state_root: Path) -> Path:
    """Location of the persisted prompt history inside a state root."""
    return state_root / PROMPT_HISTORY_FILENAME


def load_prompt_history(state_root: Path) -> List[str]:
    """Load persisted prompts, oldest first (newest last).

    A missing, corrupt, or unexpected-shape file yields an empty list — recall
    history is a convenience and must never block startup.
    """
    try:
        raw = prompt_history_path(state_root).read_text(encoding="utf-8")
    except OSError:
        return []
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if not isinstance(data, list):
        return []
    return [entry for entry in data if isinstance(entry, str)][:PROMPT_HISTORY_MAX]


def save_prompt_history(state_root: Path, entries: List[str]) -> None:
    """Persist the capped prompt list (newest last) as private local state.

    Writing an empty list is a no-op so runs without submitted prompts never
    create the file.
    """
    capped = entries[-PROMPT_HISTORY_MAX:]
    if not capped:
        return
    write_private_text(prompt_history_path(state_root), json.dumps(capped, ensure_ascii=False, indent=2))
