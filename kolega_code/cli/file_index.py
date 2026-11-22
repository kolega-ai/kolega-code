"""Workspace file index powering @ mention autocomplete in the CLI."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Iterable, List, Optional

if TYPE_CHECKING:
    import pathspec

from kolega_code.utils.images import image_media_type

# Directories that never belong in @-mention completions (mirrors the
# exclusions the agent's file-discovery tools used before they were removed).
_EXCLUDE_DIRS = frozenset(
    {
        ".git",
        ".svn",
        ".hg",
        ".idea",
        ".vscode",
        "__pycache__",
        "node_modules",
        "venv",
        "env",
        ".env",
        "dist",
        "build",
        "target",
        "bin",
        "obj",
        ".next",
        ".nuxt",
        "coverage",
    }
)

# Binary extensions skipped from completions unless the file is an attachable
# image (see refresh()).
_BINARY_EXTENSIONS = frozenset(
    {
        ".pyc",
        ".so",
        ".dll",
        ".exe",
        ".bin",
        ".jar",
        ".war",
        ".jpg",
        ".jpeg",
        ".png",
        ".gif",
        ".bmp",
        ".ico",
        ".svg",
        ".pdf",
        ".zip",
        ".tar",
        ".gz",
        ".tgz",
        ".rar",
        ".7z",
        ".mp3",
        ".mp4",
        ".avi",
        ".mov",
        ".mkv",
        ".wav",
        ".o",
        ".obj",
        ".class",
        ".binary",
        ".wasm",
        ".node",
    }
)


@dataclass(frozen=True)
class IndexEntry:
    path: str  # relative to project root, posix-style separators
    is_dir: bool


@dataclass(frozen=True)
class _CachedRanking:
    snapshot: tuple[IndexEntry, ...]
    query: str
    matches: list[tuple[float, IndexEntry]]


class WorkspaceFileIndex:
    """Cached, gitignore-aware listing of workspace files for completion."""

    MAX_FILES = 5000
    TTL_SECONDS = 5.0

    def __init__(self, project_path: Path) -> None:
        self.project_path = Path(project_path)
        self._entries: tuple[IndexEntry, ...] = ()
        self._refreshed_at: Optional[float] = None
        self._cached_ranking: _CachedRanking | None = None

    def entries(self) -> List[IndexEntry]:
        if self.is_stale():
            self.refresh()
        return list(self._entries)

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
                if name in _EXCLUDE_DIRS:
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
                if Path(name).suffix.lower() in _BINARY_EXTENSIONS and image_media_type(name) is None:
                    continue
                rel = self._posix(rel_dir / name)
                if gitignore is not None and gitignore.match_file(rel):
                    continue
                entries.append(IndexEntry(path=rel, is_dir=False))

            if len(entries) >= self.MAX_FILES:
                entries = entries[: self.MAX_FILES]
                break

        self._entries = tuple(entries)
        self._refreshed_at = time.monotonic()

    def search(self, query: str, limit: int = 8) -> List[IndexEntry]:
        """Search, refreshing the cache first if stale (may block on os.walk)."""
        return [entry for _, entry in self._rank(self.entries(), query)[:limit]]

    def cached_search(self, query: str, limit: int = 8) -> List[IndexEntry]:
        """Search the current cached entries WITHOUT refreshing.

        The UI uses this so a keystroke never blocks on ``os.walk``; a background
        refresh (see ``app._refresh_file_index``) updates the cache out-of-band.
        """
        snapshot = self._entries
        previous = self._cached_ranking
        candidates: Iterable[IndexEntry] = snapshot
        if previous is not None and previous.snapshot is snapshot:
            if query == previous.query:
                return [entry for _, entry in previous.matches[:limit]]
            # A subsequence can only lose matches as it grows. Check the
            # lowercase prefix too, since appending can change a final sigma.
            if query.startswith(previous.query) and query.lower().startswith(previous.query.lower()):
                candidates = (entry for _, entry in previous.matches)
        matches = self._rank(candidates, query)
        self._cached_ranking = _CachedRanking(snapshot, query, matches)
        return [entry for _, entry in matches[:limit]]

    @staticmethod
    def _rank(entries: Iterable[IndexEntry], query: str) -> list[tuple[float, IndexEntry]]:
        scored = []
        for entry in entries:
            score = fuzzy_score(query, entry.path)
            if score is not None:
                scored.append((score, entry))
        scored.sort(key=lambda pair: (-pair[0], pair[1].path))
        return scored

    def _load_gitignore_spec(self) -> Optional[pathspec.PathSpec]:
        import pathspec

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
    p, basename_start, penalty = _folded_path(path)

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
    # A query with no subsequence anywhere has none in the basename either, so
    # the anchored rescan is only needed once the full scan found a match.
    full = _subsequence_score(q, p, 0, basename_start)
    if full is None:
        return None
    anchored = _subsequence_score(q, p, basename_start, basename_start)
    if anchored is None:
        return full - penalty
    return max(full, anchored) - penalty


@lru_cache(maxsize=8192)
def _folded_path(path: str) -> tuple[str, int, float]:
    """Case-folded path, basename offset, and depth penalty, memoized per path.

    ``fuzzy_score`` runs for every indexed path on every keystroke, and the index
    is bounded (``MAX_FILES``), so folding each path once keeps a keystroke from
    re-lowercasing the same snapshot. The bound must exceed ``MAX_FILES`` for one
    full scan to stay warm.
    """
    lowered = path.lower()
    return lowered, lowered.rstrip("/").rfind("/") + 1, path.count("/") * 0.5 + len(path) * 0.01


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
