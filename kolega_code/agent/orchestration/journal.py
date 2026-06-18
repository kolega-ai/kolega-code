"""Per-run artifacts and the resume journal for a workflow.

Everything a run needs lives under::

    <state-dir>/workflows/<run_id>/
        script.py        the authored source (its path is returned for edit + re-run)
        run.json         meta, args, status, timing, token totals
        journal.jsonl    one line per completed agent() call (drives resume)
        agents/          per sub-agent transcripts (wired via sub_agent_recorder)

Resume contract: scripts are deterministic (the runtime blocks ``random``/``time``
and ``import``), so the same source + args produce ``agent()`` calls in the same
order. On resume we replay cached results for the longest unchanged prefix —
matched on ``(call_index, cache_key)`` — and run live from the first changed or
new call onward.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


def workflows_root(state_dir: Path) -> Path:
    """Directory holding every run's artifacts, under the CLI state dir."""
    return Path(state_dir) / "workflows"


def saved_workflows_dir(state_dir: Path) -> Path:
    """Directory holding reusable named workflows."""
    return workflows_root(state_dir) / "_saved"


class RunJournal:
    """Owns the on-disk artifacts for a single workflow run."""

    def __init__(self, run_dir: Path, run_id: str) -> None:
        self.run_dir = Path(run_dir)
        self.run_id = run_id
        self.agents_dir = self.run_dir / "agents"
        self._journal_path = self.run_dir / "journal.jsonl"
        self._script_path = self.run_dir / "script.py"
        self._meta_path = self.run_dir / "run.json"

    @classmethod
    def for_run(cls, state_dir: Path, run_id: str) -> "RunJournal":
        return cls(workflows_root(state_dir) / run_id, run_id)

    def ensure_dirs(self) -> None:
        self.agents_dir.mkdir(parents=True, exist_ok=True)

    # --- script -----------------------------------------------------------
    @property
    def script_path(self) -> Path:
        return self._script_path

    def write_script(self, source: str) -> Path:
        self.ensure_dirs()
        self._script_path.write_text(source, encoding="utf-8")
        return self._script_path

    def read_script(self) -> str:
        return self._script_path.read_text(encoding="utf-8")

    # --- meta -------------------------------------------------------------
    def write_meta(self, meta: Dict[str, Any]) -> None:
        self.ensure_dirs()
        self._meta_path.write_text(json.dumps(meta, indent=2, default=str) + "\n", encoding="utf-8")

    def update_meta(self, **fields: Any) -> None:
        meta: Dict[str, Any] = {}
        if self._meta_path.exists():
            try:
                meta = json.loads(self._meta_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                meta = {}
        meta.update(fields)
        self.write_meta(meta)

    # --- journal ----------------------------------------------------------
    def record(
        self,
        call_index: int,
        key: str,
        label: Optional[str],
        value: Any,
        status: str = "completed",
    ) -> None:
        """Append one completed ``agent()`` result for resume replay."""
        self.ensure_dirs()
        line = json.dumps(
            {"index": call_index, "key": key, "label": label, "status": status, "value": value},
            default=str,
        )
        with self._journal_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    def load_cache(self) -> Dict[int, Tuple[str, Any]]:
        """Read a prior run's journal into ``{call_index: (key, value)}``.

        Only ``completed`` entries are cached — a failed/skipped call should be
        retried live on resume.
        """
        cache: Dict[int, Tuple[str, Any]] = {}
        if not self._journal_path.exists():
            return cache
        for raw in self._journal_path.read_text(encoding="utf-8").splitlines():
            raw = raw.strip()
            if not raw:
                continue
            try:
                entry = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if entry.get("status", "completed") != "completed":
                continue
            cache[int(entry["index"])] = (entry["key"], entry.get("value"))
        return cache

    def agent_transcript_path(self, agent_id: str) -> Path:
        self.ensure_dirs()
        return self.agents_dir / f"agent-{agent_id}.jsonl"
