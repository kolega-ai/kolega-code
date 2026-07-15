from __future__ import annotations

import multiprocessing
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from kolega_code.memory import MISSING_REVISION, MarkdownMemoryBackend, MemorySafetyError


def backend(tmp_path: Path) -> MarkdownMemoryBackend:
    return MarkdownMemoryBackend(tmp_path / "projects" / "p" / "memory" / "backends" / "markdown")


def _append_in_process(storage_dir: str, value: str) -> None:
    result = MarkdownMemoryBackend(Path(storage_dir)).append_entry("MEMORY.md", value)
    if not result.ok:
        raise RuntimeError(result.error or "append failed")


def test_lazy_exact_append_cas_and_clear(tmp_path: Path) -> None:
    store = backend(tmp_path)
    assert not store.root.exists()
    assert not store.read_entry("MEMORY.md").present
    first = store.append_entry("MEMORY.md", "one")
    second = store.append_entry("MEMORY.md", "\ntwo")
    assert first.ok and second.ok
    assert store.read_entry("MEMORY.md").content == "one\ntwo"
    assert not store.replace_entry("MEMORY.md", "lost", first.revision or "").ok
    replaced = store.replace_entry("MEMORY.md", "new", second.revision or "")
    assert replaced.ok
    assert not store.delete_entry("MEMORY.md", second.revision or "").ok
    assert store.delete_entry("MEMORY.md", replaced.revision or "").ok
    created = store.replace_entry("topic.md", "topic", MISSING_REVISION)
    assert created.ok
    (store.root / "keep.bin").write_bytes(b"keep")
    assert store.clear() == 1
    assert (store.root / "keep.bin").read_bytes() == b"keep"


@pytest.mark.parametrize(
    "reference",
    ["", "/tmp/a.md", "../a.md", "a/../b.md", "./a.md", "a.txt", "a\\b.md", "manifest.json", "a\0.md"],
)
def test_rejects_unsafe_paths(tmp_path: Path, reference: str) -> None:
    with pytest.raises(MemorySafetyError):
        backend(tmp_path).append_entry(reference, "safe")


def test_rejects_symlinks_without_partial_write(tmp_path: Path) -> None:
    store = backend(tmp_path)
    store.append_entry("MEMORY.md", "safe")
    before = store.read_entry("MEMORY.md")

    outside = tmp_path / "outside.md"
    outside.write_text("outside")
    (store.root / "link.md").symlink_to(outside)
    with pytest.raises(MemorySafetyError):
        store.read_entry("link.md")
    assert store.read_entry("MEMORY.md").revision == before.revision


def test_rejects_nested_directory_symlink_for_reads_writes_and_listing(tmp_path: Path) -> None:
    store = backend(tmp_path)
    assert store.append_entry("MEMORY.md", "safe").ok
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "topic.md").write_text("outside")
    (store.root / "nested").symlink_to(outside, target_is_directory=True)

    with pytest.raises(MemorySafetyError, match="symlink"):
        store.list_entries()
    with pytest.raises(MemorySafetyError):
        store.read_entry("nested/topic.md")
    with pytest.raises(MemorySafetyError):
        store.append_entry("nested/topic.md", "changed")
    with pytest.raises(MemorySafetyError, match="symlink"):
        store.clear()
    assert (outside / "topic.md").read_text() == "outside"


def test_credential_like_content_round_trips_through_backend_and_prompt(tmp_path: Path) -> None:
    store = backend(tmp_path)
    content = "API_KEY=supersecretvalue123"
    appended = store.append_entry("MEMORY.md", content)
    assert appended.ok
    assert store.read_entry("MEMORY.md").content == content

    replacement = f"{content}\npassword=anothersecretvalue456"
    replaced = store.replace_entry("MEMORY.md", replacement, appended.revision or "")
    assert replaced.ok
    entry = store.read_entry("MEMORY.md")
    assert entry.content == replacement
    assert entry.warnings == ()

    prompt = store.prepare_prompt_context()
    assert replacement in prompt.text
    assert prompt.warnings == ()
    assert store.status().warnings == ()


def test_directly_stored_credential_like_content_is_returned_verbatim(tmp_path: Path) -> None:
    store = backend(tmp_path)
    content = "password=supersecretvalue123"
    store.root.mkdir(parents=True)
    (store.root / "MEMORY.md").write_text(content)

    assert store.read_entry("MEMORY.md").content == content
    assert content in store.prepare_prompt_context().text


def test_limits_and_prompt_bounds(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = backend(tmp_path)
    monkeypatch.setattr("kolega_code.memory.markdown.MAX_FILE_BYTES", 12)
    assert not store.append_entry("large.md", "x" * 13).ok
    monkeypatch.setattr("kolega_code.memory.markdown.MAX_FILE_BYTES", 128 * 1024)
    content = "".join(f"line {i}\n" for i in range(250))
    store.append_entry("MEMORY.md", content)
    prompt = store.prepare_prompt_context()
    assert prompt.truncated and prompt.line_count == 200
    assert "line 200" not in prompt.text
    assert "not already authoritative in code or documentation" in prompt.authoring_guidance
    assert "non-obvious build or tooling quirks" in prompt.authoring_guidance
    assert "architectural constraints" in prompt.authoring_guidance
    assert "recurring failure causes" in prompt.authoring_guidance
    assert "user-confirmed conventions" in prompt.authoring_guidance
    assert "Before finishing a substantive task" in prompt.authoring_guidance
    assert "use its current revision" in prompt.authoring_guidance
    assert "compare-and-swap" not in prompt.authoring_guidance


def test_count_and_total_limits_leave_existing_content(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = backend(tmp_path)
    assert store.append_entry("one.md", "1234").ok
    monkeypatch.setattr("kolega_code.memory.markdown.MAX_FILES", 1)
    assert not store.append_entry("two.md", "x").ok
    monkeypatch.setattr("kolega_code.memory.markdown.MAX_FILES", 100)
    monkeypatch.setattr("kolega_code.memory.markdown.MAX_TOTAL_BYTES", 5)
    assert not store.append_entry("one.md", "56").ok
    assert store.read_entry("one.md").content == "1234"


def test_scan_file_count_limit_rejects_corrupt_bank(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = backend(tmp_path)
    store.root.mkdir(parents=True)
    (store.root / "one.md").write_text("one")
    (store.root / "two.md").write_text("two")
    monkeypatch.setattr("kolega_code.memory.markdown.MAX_FILES", 1)
    with pytest.raises(MemorySafetyError, match="file count"):
        store.list_entries()


def test_scan_rejects_excessive_directory_depth(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = backend(tmp_path)
    nested = store.root / "one" / "two" / "three"
    nested.mkdir(parents=True)
    (nested / "topic.md").write_text("topic")
    monkeypatch.setattr("kolega_code.memory.markdown.MAX_DIRECTORY_DEPTH", 2)

    with pytest.raises(MemorySafetyError, match="directory depth"):
        store.list_entries()


def test_concurrent_appends_and_private_modes(tmp_path: Path) -> None:
    store = backend(tmp_path)

    def append(index: int) -> None:
        assert store.append_entry("MEMORY.md", f"{index},").ok

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(append, range(40)))
    values = (store.read_entry("MEMORY.md").content or "").strip(",").split(",")
    assert sorted(map(int, values)) == list(range(40))
    if os.name == "posix":
        assert store.root.stat().st_mode & 0o777 == 0o700
        assert (store.root / "MEMORY.md").stat().st_mode & 0o777 == 0o600
        assert (store.root / ".memory.lock").stat().st_mode & 0o777 == 0o600


@pytest.mark.skipif(os.name != "posix", reason="cross-process file locking requires POSIX")
def test_cross_process_appends_preserve_every_fragment(tmp_path: Path) -> None:
    store = backend(tmp_path)
    context = multiprocessing.get_context("spawn")
    processes = [context.Process(target=_append_in_process, args=(str(store.root), f"{index},")) for index in range(8)]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=15)
        assert process.exitcode == 0

    values = (store.read_entry("MEMORY.md").content or "").strip(",").split(",")
    assert sorted(map(int, values)) == list(range(8))
