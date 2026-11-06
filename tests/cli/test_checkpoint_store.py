from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from collections import namedtuple
from pathlib import Path

import pytest

from kolega_code.cli.tui import checkpoint_store
from kolega_code.cli.tui.checkpoint_store import (
    LOW_DISK_SPACE,
    STORAGE_LIMIT,
    WRITE_FAILED,
    CheckpointBlobStore,
)

_Usage = namedtuple("_Usage", "total used free")


@pytest.fixture(autouse=True)
def _fresh_sweep(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(checkpoint_store, "_swept_roots", set())


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def test_identical_content_is_stored_once_and_freed_with_its_last_reference(tmp_path: Path) -> None:
    store = CheckpointBlobStore(tmp_path / "spool")
    data = b"same bytes\n"
    sha = _sha(data)

    assert store.put(sha, data, limit=1024) == ""
    assert store.put(sha, data, limit=1024) == ""
    assert store.acquire(sha)
    assert len(store) == 1
    assert store.live_bytes == store.disk_bytes == len(data)
    assert store.read(sha) == data

    store.release(sha)
    store.release(sha)
    assert store.read(sha) == data
    store.release(sha)
    assert sha not in store
    assert store.read(sha) is None
    assert store.live_bytes == 0


def test_put_refuses_past_the_limit(tmp_path: Path) -> None:
    store = CheckpointBlobStore(tmp_path / "spool")
    first, second = b"a" * 60, b"b" * 60

    assert store.put(_sha(first), first, limit=100) == ""
    assert store.put(_sha(second), second, limit=100) == STORAGE_LIMIT
    assert store.read(_sha(second)) is None
    assert store.disk_bytes == 60


def test_put_refuses_when_the_volume_is_nearly_full(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    free = checkpoint_store.MIN_FREE_BYTES + 100
    monkeypatch.setattr(checkpoint_store.shutil, "disk_usage", lambda _path: _Usage(free * 2, free, free))
    store = CheckpointBlobStore(tmp_path / "spool")
    small, large = b"s" * 50, b"l" * 200

    assert store.put(_sha(small), small, limit=10**9) == ""
    assert store.put(_sha(large), large, limit=10**9) == LOW_DISK_SPACE
    assert _sha(large) not in store


def test_small_copies_share_pack_segments_that_rotate_and_are_freed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(checkpoint_store, "SEGMENT_BYTES", 100)
    store = CheckpointBlobStore(tmp_path / "spool")
    first, second, third = b"a" * 60, b"b" * 60, b"c" * 30

    for data in (first, second, third):
        assert store.put(_sha(data), data, limit=10**6) == ""
    assert len(store.segment_paths()) == 2  # "c" still fits beside "b"
    assert [store.read(_sha(data)) for data in (first, second, third)] == [first, second, third]

    store.release(_sha(first))
    assert len(store.segment_paths()) == 1
    assert store.disk_bytes == 90
    assert store.read(_sha(second)) == second


def test_a_mostly_dead_segment_is_compacted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(checkpoint_store, "SEGMENT_BYTES", 100)
    store = CheckpointBlobStore(tmp_path / "spool")
    dead, survivor, later = b"d" * 50, b"s" * 30, b"l" * 40
    for data in (dead, survivor, later):
        assert store.put(_sha(data), data, limit=10**6) == ""
    sealed = store.segment_paths()[0]

    store.release(_sha(dead))

    assert not sealed.exists()
    assert store.read(_sha(survivor)) == survivor
    assert store.read(_sha(later)) == later
    assert store.disk_bytes == store.live_bytes == 70


def test_close_deletes_the_spool_and_refuses_later_writes(tmp_path: Path) -> None:
    store = CheckpointBlobStore(tmp_path / "spool")
    data = b"kept until close\n"
    assert store.put(_sha(data), data, limit=1024) == ""
    assert store.path.is_dir()

    store.close()

    assert not store.path.exists()
    assert store.read(_sha(data)) is None
    assert store.put(_sha(data), data, limit=1024) == WRITE_FAILED
    assert not store.path.exists()


def test_clear_drops_blobs_but_keeps_the_store_usable(tmp_path: Path) -> None:
    store = CheckpointBlobStore(tmp_path / "spool")
    old, new = b"old\n", b"new\n"
    assert store.put(_sha(old), old, limit=1024) == ""

    store.clear()

    assert len(store) == 0
    assert store.read(_sha(old)) is None
    assert store.put(_sha(new), new, limit=1024) == ""
    assert store.read(_sha(new)) == new
    assert store.disk_bytes == len(new)


def test_a_store_that_never_writes_creates_nothing(tmp_path: Path) -> None:
    store = CheckpointBlobStore(tmp_path / "spool")
    store.close()
    assert not (tmp_path / "spool").exists()


def _exited_pid() -> int:
    process = subprocess.Popen([sys.executable, "-c", "pass"])
    process.wait()
    return process.pid


def test_first_write_sweeps_spools_whose_owner_exited(tmp_path: Path) -> None:
    root = tmp_path / "spool"
    dead = root / "dead"
    dead.mkdir(parents=True)
    (dead / "owner.pid").write_text(str(_exited_pid()), encoding="ascii")
    (dead / "blob").write_bytes(b"left behind")
    alive = root / "alive"
    alive.mkdir()
    (alive / "owner.pid").write_text(str(os.getppid()), encoding="ascii")
    unowned = root / "being-created"
    unowned.mkdir()

    store = CheckpointBlobStore(root)
    data = b"fresh\n"
    assert store.put(_sha(data), data, limit=1024) == ""

    assert not dead.exists()
    assert alive.is_dir()
    assert unowned.is_dir()
    assert store.path.is_dir()
