"""Per-run artifacts and the resume journal for a workflow.

Everything a run needs lives under::

    <state-dir>/workflows/<run_id>/
        script.py        the authored source (its path is returned for edit + re-run)
        run.json         meta, args, status, timing, token totals, artifact paths
        journal.jsonl    one line per completed agent() call (drives resume)
        result.json      full JSON-rendered workflow return value
        result.md        readable workflow return value
        transcript.jsonl raw workflow events/call outcomes
        transcript.md    readable workflow transcript/index
        agents/          per sub-agent raw/readable transcripts

Resume contract: scripts are deterministic (the runtime blocks ``random``/``time``
and ``import``), so the same source + args produce ``agent()`` calls in the same
order. On resume we replay cached results for the longest unchanged prefix —
matched on ``(call_index, cache_key)`` — and run live from the first changed or
new call onward.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


def workflows_root(state_dir: Path) -> Path:
    """Directory holding every run's artifacts, under the CLI state dir."""
    return Path(state_dir) / "workflows"


def saved_workflows_dir(state_dir: Path) -> Path:
    """Directory holding reusable named workflows."""
    return workflows_root(state_dir) / "_saved"


def _slugify(value: str, *, fallback: str = "agent", max_length: int = 64) -> str:
    """Return a filesystem-friendly slug for workflow artifact filenames."""
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value or "").strip().lower())
    slug = re.sub(r"-+", "-", slug).strip("-._")
    if not slug:
        slug = fallback
    return slug[:max_length].rstrip("-._") or fallback


class RunJournal:
    """Owns the on-disk artifacts for a single workflow run."""

    def __init__(self, run_dir: Path, run_id: str) -> None:
        self.run_dir = Path(run_dir)
        self.run_id = run_id
        self.agents_dir = self.run_dir / "agents"
        self._journal_path = self.run_dir / "journal.jsonl"
        self._transcript_jsonl_path = self.run_dir / "transcript.jsonl"
        self._transcript_md_path = self.run_dir / "transcript.md"
        self._result_json_path = self.run_dir / "result.json"
        self._result_md_path = self.run_dir / "result.md"
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

    @property
    def journal_path(self) -> Path:
        return self._journal_path

    @property
    def transcript_jsonl_path(self) -> Path:
        return self._transcript_jsonl_path

    @property
    def transcript_md_path(self) -> Path:
        return self._transcript_md_path

    @property
    def result_json_path(self) -> Path:
        return self._result_json_path

    @property
    def result_md_path(self) -> Path:
        return self._result_md_path

    def agent_artifact_paths(self, call_index: int, label: Optional[str] = None) -> Dict[str, Path]:
        """Return stable per-agent raw/readable artifact paths for a workflow call."""
        self.ensure_dirs()
        suffix = _slugify(label or f"call-{call_index}")
        stem = f"agent-{call_index:03d}-{suffix}"
        return {"markdown": self.agents_dir / f"{stem}.md", "jsonl": self.agents_dir / f"{stem}.jsonl"}

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
        **metadata: Any,
    ) -> None:
        """Append one completed ``agent()`` result for resume replay.

        ``journal.jsonl`` is the resume contract. Extra fields are allowed and ignored
        by ``load_cache()`` so newer artifact metadata remains backward-compatible.
        """
        self.ensure_dirs()
        payload = {"index": call_index, "key": key, "label": label, "status": status, "value": value}
        payload.update({k: v for k, v in metadata.items() if v is not None})
        line = json.dumps(payload, default=str)
        with self._journal_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    def append_transcript_event(self, event: Dict[str, Any]) -> None:
        """Append one raw workflow transcript event as JSONL."""
        self.ensure_dirs()
        with self._transcript_jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, default=str) + "\n")

    def write_result_artifacts(self, result: Any, markdown: str) -> None:
        """Persist the full workflow return value in JSON and readable Markdown."""
        self.ensure_dirs()
        try:
            rendered_json = json.dumps(result, indent=2, default=str)
        except (TypeError, ValueError):
            rendered_json = json.dumps(str(result), indent=2)
        self._result_json_path.write_text(rendered_json + "\n", encoding="utf-8")
        self._result_md_path.write_text(markdown, encoding="utf-8")

    def write_transcript_markdown(self, markdown: str) -> None:
        """Persist the readable workflow transcript."""
        self.ensure_dirs()
        self._transcript_md_path.write_text(markdown, encoding="utf-8")

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
