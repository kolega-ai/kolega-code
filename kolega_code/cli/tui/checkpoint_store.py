"""Disk spool for the file copies that turn checkpoints keep for rewind.

Copies are keyed by sha256, so identical content is stored once and reference
counted. They are appended to pack segments rather than written one file each:
creating a file costs ~0.1 ms on APFS, which dominated capture time in repos
with tens of thousands of untracked files. A segment is deleted once nothing in
it is referenced, and compacted once most of it is dead.

The spool is scratch data owned by one tracker: ``close()`` deletes it, and the
first write of a later process sweeps spools whose owner has exited.
"""

from __future__ import annotations

import os
import shutil
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Optional

from kolega_code.local_state import ensure_private_dir

# Copying stops once a write would leave less than this free on the volume.
MIN_FREE_BYTES = 5 * 1024**3
# Free space is re-queried after this many bytes are written.
FREE_SPACE_RECHECK_BYTES = 64 * 1024 * 1024
SEGMENT_BYTES = 64 * 1024 * 1024
# A sealed segment whose live share drops below this is rewritten.
COMPACT_LIVE_FRACTION = 0.5
# A spool with no owner file yet may be mid-creation by another process.
_UNOWNED_GRACE_SECONDS = 3600
_OWNER_FILE = "owner.pid"

STORAGE_LIMIT = "checkpoint storage limit reached"
LOW_DISK_SPACE = "not enough free disk space for a checkpoint copy"
WRITE_FAILED = "could not write a checkpoint copy"

_swept_roots: set[Path] = set()
_sweep_lock = threading.Lock()


@dataclass(slots=True)
class _Blob:
    segment: int
    offset: int
    size: int
    refs: int


@dataclass(slots=True)
class _Segment:
    size: int = 0
    live: int = 0


class CheckpointBlobStore:
    """Reference-counted, sha256-keyed blobs in a private spool directory."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.path = root / uuid.uuid4().hex
        self.disk_bytes = 0  # segment bytes on disk, live or dead; what the limit caps
        self.live_bytes = 0
        self._blobs: dict[str, _Blob] = {}
        self._segments: dict[int, _Segment] = {}
        self._active: Optional[int] = None
        self._active_handle: Optional[BinaryIO] = None
        self._next_segment = 0
        self._lock = threading.Lock()
        self._created = False
        self._closed = False
        self._free_bytes: Optional[int] = None
        self._written_since_free_check = 0

    def __contains__(self, sha: str) -> bool:
        return sha in self._blobs

    def __len__(self) -> int:
        return len(self._blobs)

    def segment_paths(self) -> list[Path]:
        return [self._segment_path(index) for index in sorted(self._segments)]

    def acquire(self, sha: str) -> bool:
        """Add a reference to a stored blob; False when it is not stored."""
        with self._lock:
            blob = self._blobs.get(sha)
            if blob is None:
                return False
            blob.refs += 1
            return True

    def release(self, sha: str) -> None:
        with self._lock:
            blob = self._blobs.get(sha)
            if blob is None:
                return
            if blob.refs > 1:
                blob.refs -= 1
                return
            del self._blobs[sha]
            self.live_bytes -= blob.size
            segment = self._segments[blob.segment]
            segment.live -= blob.size
            if blob.segment != self._active:
                if segment.live <= 0:
                    self._delete_segment(blob.segment)
                elif segment.live < segment.size * COMPACT_LIVE_FRACTION:
                    self._compact(blob.segment)

    def put(self, sha: str, data: bytes, *, limit: int) -> str:
        """Store ``data`` with one reference. Returns "" on success, else why it was refused."""
        with self._lock:
            if self._closed:
                return WRITE_FAILED
            blob = self._blobs.get(sha)
            if blob is not None:
                blob.refs += 1
                return ""
            size = len(data)
            if self.disk_bytes + size > limit:
                return STORAGE_LIMIT
            if not self._free_space_allows(size):
                return LOW_DISK_SPACE
            try:
                segment, offset = self._append(data)
            except OSError:
                return WRITE_FAILED
            self._blobs[sha] = _Blob(segment=segment, offset=offset, size=size, refs=1)
            self._segments[segment].live += size
            self.live_bytes += size
            return ""

    def read(self, sha: str) -> Optional[bytes]:
        with self._lock:
            blob = self._blobs.get(sha)
            if blob is None:
                return None
            return self._read_blob(blob)

    def clear(self) -> None:
        """Drop every blob but keep the store usable."""
        with self._lock:
            self._remove_dir()

    def close(self) -> None:
        """Delete the spool. Later writes are refused and reads return None."""
        with self._lock:
            self._closed = True
            self._remove_dir()

    # ---- segments ------------------------------------------------------------

    def _segment_path(self, index: int) -> Path:
        return self.path / f"pack-{index:06d}"

    def _append(self, data: bytes) -> tuple[int, int]:
        self._ensure_dir()
        active = self._active
        if active is None or (self._segments[active].size and self._segments[active].size + len(data) > SEGMENT_BYTES):
            active = self._open_segment()
        handle = self._active_handle
        assert handle is not None
        segment = self._segments[active]
        offset = segment.size
        try:
            view = memoryview(data)
            while view:
                written = handle.write(view)
                if not written:
                    raise OSError("short write to checkpoint spool")
                view = view[written:]
        except OSError:
            # The file now ends past the recorded size; never append after it.
            self._seal_active()
            raise
        segment.size += len(data)
        self.disk_bytes += len(data)
        return active, offset

    def _open_segment(self) -> int:
        self._seal_active()
        index = self._next_segment
        self._next_segment += 1
        self._active_handle = open(self._segment_path(index), "ab", buffering=0)
        self._segments[index] = _Segment()
        self._active = index
        return index

    def _seal_active(self) -> None:
        handle, active = self._active_handle, self._active
        self._active_handle = None
        self._active = None
        if handle is not None:
            handle.close()
        if active is not None and self._segments[active].live <= 0:
            self._delete_segment(active)

    def _delete_segment(self, index: int) -> None:
        segment = self._segments.pop(index)
        self.disk_bytes -= segment.size
        try:
            self._segment_path(index).unlink(missing_ok=True)
        except OSError:
            pass

    def _compact(self, index: int) -> None:
        """Move a segment's live blobs into the active segment, then delete it."""
        moving = [(sha, blob) for sha, blob in self._blobs.items() if blob.segment == index]
        for sha, blob in moving:
            data = self._read_blob(blob)
            if data is None:
                continue
            try:
                segment, offset = self._append(data)
            except OSError:
                return  # leave the segment in place; its blobs are still readable
            self._segments[index].live -= blob.size
            self._segments[segment].live += blob.size
            self._blobs[sha] = _Blob(segment=segment, offset=offset, size=blob.size, refs=blob.refs)
        if self._segments[index].live <= 0:
            self._delete_segment(index)

    def _read_blob(self, blob: _Blob) -> Optional[bytes]:
        try:
            with self._segment_path(blob.segment).open("rb") as handle:
                handle.seek(blob.offset)
                data = handle.read(blob.size)
        except OSError:
            return None
        return data if len(data) == blob.size else None

    def _remove_dir(self) -> None:
        if self._active_handle is not None:
            self._active_handle.close()
        self._active_handle = None
        self._active = None
        self._blobs = {}
        self._segments = {}
        self.disk_bytes = 0
        self.live_bytes = 0
        if self._created:
            shutil.rmtree(self.path, ignore_errors=True)
            self._created = False

    def _ensure_dir(self) -> None:
        if self._created:
            return
        _sweep_stale_spools(self.root)
        ensure_private_dir(self.path)
        (self.path / _OWNER_FILE).write_text(str(os.getpid()), encoding="ascii")
        self._created = True

    def _free_space_allows(self, size: int) -> bool:
        if self._free_bytes is None or self._written_since_free_check + size > FREE_SPACE_RECHECK_BYTES:
            self._free_bytes = _free_bytes(self.root)
            self._written_since_free_check = 0
        if self._free_bytes is None:
            return True
        if self._free_bytes - size < MIN_FREE_BYTES:
            return False
        self._free_bytes -= size
        self._written_since_free_check += size
        return True


def _free_bytes(path: Path) -> Optional[int]:
    probe = path
    while not probe.exists():
        if probe.parent == probe:
            return None
        probe = probe.parent
    try:
        return shutil.disk_usage(probe).free
    except OSError:
        return None


def _sweep_stale_spools(root: Path) -> None:
    """Delete spools left behind by processes that exited without closing them."""
    with _sweep_lock:
        if root in _swept_roots:
            return
        _swept_roots.add(root)
    try:
        entries = list(root.iterdir())
    except OSError:
        return
    for entry in entries:
        try:
            if not entry.is_dir():
                continue
            pid = _owner_pid(entry)
            if pid is None:
                if time.time() - entry.stat().st_mtime < _UNOWNED_GRACE_SECONDS:
                    continue
            elif pid == os.getpid() or _pid_alive(pid):
                continue
        except OSError:
            continue
        shutil.rmtree(entry, ignore_errors=True)


def _owner_pid(spool: Path) -> Optional[int]:
    try:
        return int((spool / _OWNER_FILE).read_text(encoding="ascii").strip())
    except (OSError, ValueError):
        return None


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True
