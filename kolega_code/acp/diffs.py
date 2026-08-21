"""Full-file diff content for ACP tool results, sourced from snapshots.

Edit tools record pre/post file states in the session's ``SnapshotService``
keyed by tool-call id. The bridge turns those into ``FileEditToolCallContent``
(ACP v1's oldText/newText diff) so clients render the change instead of bare
result text. Missing snapshots, binary content, and oversized files fall back
to the ordinary text rendering.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from acp.schema import FileEditToolCallContent

from kolega_code.services.snapshots import FileState, SnapshotRecord, SnapshotService

logger = logging.getLogger(__name__)

MAX_DIFF_CHARS = 200_000
_SNAPSHOT_SCAN_LIMIT = 50


def _absolute(record: SnapshotRecord, path: str) -> str:
    """Snapshot paths are project-relative; ACP requires absolute paths."""
    if Path(path).is_absolute():
        return path
    return str(Path(record.project_path) / path)


class AcpDiffProvider:
    """Builds diff content for tool results from the session's snapshot store."""

    def __init__(self, snapshot_service: SnapshotService | None) -> None:
        self._snapshots = snapshot_service

    @classmethod
    def for_session(cls, session: Any) -> "AcpDiffProvider":
        collection = getattr(session.agent, "tool_collection", None)
        return cls(getattr(collection, "snapshot_service", None) if collection is not None else None)

    def build_for_tool_result(self, tool_call_id: str) -> list[FileEditToolCallContent]:
        if not tool_call_id or self._snapshots is None:
            return []
        record = self._find_record(tool_call_id)
        if record is None:
            return []
        contents: list[FileEditToolCallContent] = []
        for path in record.touched_paths:
            content = self._build_one(_absolute(record, path), record.before.get(path), record.after.get(path))
            if content is not None:
                contents.append(content)
        return contents

    def _find_record(self, tool_call_id: str) -> SnapshotRecord | None:
        assert self._snapshots is not None
        try:
            records = self._snapshots.list_snapshots(limit=_SNAPSHOT_SCAN_LIMIT)
        except Exception:  # noqa: BLE001 — a broken snapshot store must never break rendering
            logger.exception("acp diff: snapshot listing failed")
            return None
        return next((record for record in records if record.tool_call_id == tool_call_id), None)

    def _build_one(
        self, path: str, before: FileState | None, after: FileState | None
    ) -> FileEditToolCallContent | None:
        if after is None or after.kind != "file":
            return None
        if before is not None and before.kind != "file":
            if before.kind != "missing":
                return None
            before = None
        try:
            old_text = None if before is None else self._read_text(before)
            new_text = self._read_text(after)
        except Exception:  # noqa: BLE001 — unreadable blobs fall back to text rendering
            logger.debug("acp diff: blob read failed for %s", path)
            return None
        if old_text == new_text:
            return None
        if (old_text is not None and "\x00" in old_text) or "\x00" in new_text:
            return None
        if len(old_text or "") + len(new_text or "") > MAX_DIFF_CHARS:
            return None
        return FileEditToolCallContent(type="diff", path=path, old_text=old_text, new_text=new_text)

    def _read_text(self, state: FileState) -> str:
        assert self._snapshots is not None
        if not state.blob_id:
            raise ValueError("snapshot state has no blob")
        return self._snapshots.read_blob(state).decode("utf-8")
