"""Workspace file index powering @ mention autocomplete in the CLI."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import List, Optional

import pathspec

from kolega_code.agent.tool_backend.glob_tool import GlobTool
from kolega_code.utils.images import image_media_type


@dataclass(frozen=True)
class IndexEntry:
    path: str  # relative to project root, posix-style separators
    is_dir: bool


class WorkspaceFileIndex:
    """Cached, gitignore-aware listing of workspace files for completion."""

    MAX_FILES = 5000
    TTL_SECONDS = 5.0

    def __init__(self, project_path: Path) -> None:
        self.project_path = Path(project_path)
        self._entries: List[IndexEntry] = []
        self._refreshed_at: Optional[float] = None

    def entries(self) -> List[IndexEntry]:
        if self.is_stale():
            self.refresh()
        return self._entries

    def is_stale(self) -> bool:
        """Whether the cache is empty or older than the TTL (a refresh is due)."""
        if self._refreshed_at is None:
            return True
        return time.monotonic() - self._refreshed_at > self.TTL_SECONDS

    def refresh(self) -> None:
        gitignore = self._load_gitignore_spec()
        entries: List[IndexEntry] = []
        root = self.project_path
        for dirpath, dirnames, filenames in os.walk(root, topdown=True):
            rel_dir = Path(dirpath).relative_to(root)
            kept_dirs = []
            for name in sorted(dirnames):
                if name in GlobTool.EXCLUDE_DIRS:
                    continue
                rel = self._posix(rel_dir / name)
                if gitignore is not None and gitignore.match_file(rel + "/"):
                    continue
                kept_dirs.append(name)
                entries.append(IndexEntry(path=rel + "/", is_dir=True))
            dirnames[:] = kept_dirs

            for name in sorted(filenames):
                # Exclude binaries, but keep attachable images: @-mentioning an image
                # attaches it (build_file_attachments handles it via the same
                # image_media_type), so it belongs in completions for vision models.
                if Path(name).suffix.lower() in GlobTool.BINARY_EXTENSIONS and image_media_type(name) is None:
                    continue
                rel = self._posix(rel_dir / name)
                if gitignore is not None and gitignore.match_file(rel):
                    continue
                entries.append(IndexEntry(path=rel, is_dir=False))

            if len(entries) >= self.MAX_FILES:
                entries = entries[: self.MAX_FILES]
                break

        self._entries = entries
        self._refreshed_at = time.monotonic()

    def search(self, query: str, limit: int = 8) -> List[IndexEntry]:
        """Search, refreshing the cache first if stale (may block on os.walk)."""
        return self._rank(self.entries(), query, limit)

    def cached_search(self, query: str, limit: int = 8) -> List[IndexEntry]:
        """Search the current cached entries WITHOUT refreshing.

        The UI uses this so a keystroke never blocks on ``os.walk``; a background
        refresh (see ``app._refresh_file_index``) updates the cache out-of-band.
        """
        return self._rank(self._entries, query, limit)

    @staticmethod
    def _rank(entries: List[IndexEntry], query: str, limit: int) -> List[IndexEntry]:
        scored = []
        for entry in entries:
            score = fuzzy_score(query, entry.path)
            if score is not None:
                scored.append((score, entry))
        scored.sort(key=lambda pair: (-pair[0], pair[1].path))
        return [entry for _, entry in scored[:limit]]

    def _load_gitignore_spec(self) -> Optional[pathspec.PathSpec]:
        gitignore_path = self.project_path / ".gitignore"
        try:
            if not gitignore_path.is_file():
                return None
            content = gitignore_path.read_text(encoding="utf-8")
            return pathspec.PathSpec.from_lines(
                pathspec.patterns.GitWildMatchPattern,  # pyright: ignore[reportPrivateImportUsage]
                content.splitlines(),
            )
        except Exception:
            return None

    @staticmethod
    def _posix(path: Path) -> str:
        rel = str(PurePosixPath(path.as_posix()))
        return "" if rel == "." else rel


def fuzzy_score(query: str, path: str) -> Optional[float]:
    """Score how well ``query`` matches ``path``; higher is better, None means no match.

    Case-insensitive subsequence match with bonuses for consecutive runs,
    matches at path-segment boundaries, and matches inside the basename.
    """
    if not query:
        # Empty query matches everything; prefer shallow paths.
        return -float(path.count("/"))

    q = query.lower()
    p = path.lower()
    basename_start = p.rstrip("/").rfind("/") + 1
    penalty = path.count("/") * 0.5 + len(path) * 0.01

    # Substring matches outrank scattered subsequences.
    idx = p.find(q)
    if idx != -1:
        score = 100.0
        if idx >= basename_start:
            score += 50.0
        if idx == 0 or p[idx - 1] in "/._-":
            score += 25.0
        return score - penalty

    # Subsequence match: try a basename-anchored scan too, so dense matches in
    # the filename beat scattered matches that start in the directory part.
    full = _subsequence_score(q, p, 0, basename_start)
    anchored = _subsequence_score(q, p, basename_start, basename_start)
    candidates = [score for score in (full, anchored) if score is not None]
    if not candidates:
        return None
    return max(candidates) - penalty


def _subsequence_score(q: str, p: str, start: int, basename_start: int) -> Optional[float]:
    score = 0.0
    pos = start - 1
    prev_matched = -2
    for ch in q:
        pos = p.find(ch, pos + 1)
        if pos == -1:
            return None
        if pos == prev_matched + 1:
            score += 3.0  # consecutive run
        if pos == 0 or p[pos - 1] in "/._-":
            score += 5.0  # segment/word boundary
        if pos >= basename_start:
            score += 2.0
        prev_matched = pos
        score += 1.0
    return score
