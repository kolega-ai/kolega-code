from __future__ import annotations

import os
from dataclasses import fields
from pathlib import Path
from typing import Any

import pytest

from kolega_code.memory import markdown as markdown_module
from kolega_code.memory import (
    MarkdownMemoryBackend,
    MemoryAccessScope,
    MemoryCapability,
    MemorySafetyError,
)


def backend(tmp_path: Path) -> MarkdownMemoryBackend:
    return MarkdownMemoryBackend(tmp_path / "projects" / "p" / "memory" / "backends" / "markdown")


def _schema_keys(value: Any) -> set[str]:
    if isinstance(value, dict):
        return set(value) | {key for nested in value.values() for key in _schema_keys(nested)}
    if isinstance(value, (list, tuple)):
        return {key for nested in value for key in _schema_keys(nested)}
    return set()


def test_lazy_complete_writes_overwrite_and_path_only_delete(tmp_path: Path) -> None:
    store = backend(tmp_path)
    assert not store.root.exists()
    assert not store.read_entry("MEMORY.md").present

    first = store.write_entry("MEMORY.md", "one")
    second = store.write_entry("MEMORY.md", "complete replacement")

    assert first.ok and second.ok
    assert store.read_entry("MEMORY.md").content == "complete replacement"
    assert store.delete_entry("MEMORY.md").ok
    assert not store.read_entry("MEMORY.md").present
    assert not store.delete_entry("MEMORY.md").ok


def test_capabilities_bindings_schemas_and_results_have_no_legacy_mutation_fields(
    tmp_path: Path,
) -> None:
    store = backend(tmp_path)
    assert MemoryCapability.LIST in store.metadata.capabilities
    assert MemoryCapability.WRITE in store.metadata.capabilities
    assert {capability.value for capability in store.metadata.capabilities}.isdisjoint({"append", "replace", "clear"})

    bindings = store.tool_bindings(MemoryAccessScope.TOP_LEVEL)
    assert [binding.name for binding in bindings] == [
        "read_memory",
        "list_memory",
        "write_memory",
        "edit_memory",
        "delete_memory",
    ]
    forbidden = {"revision", "sha", "sha256", "expected_sha256", "mode"}
    for binding in bindings:
        assert _schema_keys(binding.definition).isdisjoint(forbidden)

    write_result = next(binding for binding in bindings if binding.name == "write_memory").handler(content="content")
    read_result = next(binding for binding in bindings if binding.name == "read_memory").handler()
    list_result = next(binding for binding in bindings if binding.name == "list_memory").handler()
    delete_result = next(binding for binding in bindings if binding.name == "delete_memory").handler(path="MEMORY.md")
    for result in (write_result, read_result, *list_result, delete_result):
        assert {field.name for field in fields(result)}.isdisjoint(forbidden)


def test_edit_memory_binding_replaces_one_exact_unique_match(tmp_path: Path) -> None:
    store = backend(tmp_path)
    assert store.write_entry("MEMORY.md", "before\nunique text\nafter").ok
    edit = next(
        binding for binding in store.tool_bindings(MemoryAccessScope.TOP_LEVEL) if binding.name == "edit_memory"
    )

    result = edit.handler(old_string="unique text", new_string="replacement")

    assert result.ok
    assert store.read_entry("MEMORY.md").content == "before\nreplacement\nafter"


@pytest.mark.parametrize(
    ("content", "old_string", "error"),
    [
        ("unchanged", "", "empty"),
        ("unchanged", "missing", "not found"),
        ("repeat repeat", "repeat", "2 times"),
        ("aaa", "aa", "2 times"),
    ],
)
def test_edit_memory_binding_rejects_non_unique_match_without_writing(
    tmp_path: Path,
    content: str,
    old_string: str,
    error: str,
) -> None:
    store = backend(tmp_path)
    assert store.write_entry("MEMORY.md", content).ok
    edit = next(
        binding for binding in store.tool_bindings(MemoryAccessScope.TOP_LEVEL) if binding.name == "edit_memory"
    )

    result = edit.handler(old_string=old_string, new_string="changed")

    assert not result.ok
    assert error in (result.error or "")
    assert store.read_entry("MEMORY.md").content == content


def test_case_insensitive_alias_is_accounted_as_an_overwrite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = backend(tmp_path)
    assert store.write_entry("Topics/Build.md", "old").ok
    if not (store.root / "topics" / "build.md").exists():
        pytest.skip("filesystem is case-sensitive")
    monkeypatch.setattr(markdown_module, "MAX_FILES", 1)

    result = store.write_entry("topics/build.md", "replacement")

    assert result.ok
    assert store.read_entry("Topics/Build.md").content == "replacement"
    assert len(store.list_entries()) == 1


@pytest.mark.parametrize(
    "reference",
    ["", "/tmp/a.md", "../a.md", "a/../b.md", "./a.md", "a.txt", "a\\b.md", "manifest.json", "a\0.md"],
)
def test_rejects_unsafe_paths(tmp_path: Path, reference: str) -> None:
    with pytest.raises(MemorySafetyError):
        backend(tmp_path).write_entry(reference, "safe")


def test_rejects_symlinks_without_partial_write(tmp_path: Path) -> None:
    store = backend(tmp_path)
    assert store.write_entry("MEMORY.md", "safe").ok

    outside = tmp_path / "outside.md"
    outside.write_text("outside")
    (store.root / "link.md").symlink_to(outside)
    with pytest.raises(MemorySafetyError):
        store.read_entry("link.md")
    with pytest.raises(MemorySafetyError):
        store.write_entry("link.md", "changed")
    with pytest.raises(MemorySafetyError):
        store.delete_entry("link.md")
    assert store.read_entry("MEMORY.md").content == "safe"
    assert outside.read_text() == "outside"


def test_rejects_nested_directory_symlink_for_reads_writes_deletes_and_listing(tmp_path: Path) -> None:
    store = backend(tmp_path)
    assert store.write_entry("MEMORY.md", "safe").ok
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "topic.md").write_text("outside")
    (store.root / "nested").symlink_to(outside, target_is_directory=True)

    with pytest.raises(MemorySafetyError, match="symlink"):
        store.list_entries()
    with pytest.raises(MemorySafetyError):
        store.read_entry("nested/topic.md")
    with pytest.raises(MemorySafetyError):
        store.write_entry("nested/topic.md", "changed")
    with pytest.raises(MemorySafetyError):
        store.delete_entry("nested/topic.md")
    assert (outside / "topic.md").read_text() == "outside"


def test_credential_like_content_round_trips_through_backend_and_prompt(tmp_path: Path) -> None:
    store = backend(tmp_path)
    content = "API_KEY=supersecretvalue123\npassword=anothersecretvalue456"
    result = store.write_entry("MEMORY.md", content)
    assert result.ok
    entry = store.read_entry("MEMORY.md")
    assert entry.content == content
    assert entry.warnings == ()

    prompt = store.prepare_prompt_context()
    assert content in prompt.text
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
    assert not store.write_entry("large.md", "x" * 13).ok
    monkeypatch.setattr("kolega_code.memory.markdown.MAX_FILE_BYTES", 128 * 1024)
    content = "".join(f"line {i}\n" for i in range(250))
    assert store.write_entry("MEMORY.md", content).ok
    prompt = store.prepare_prompt_context()
    assert prompt.truncated and prompt.line_count == 200
    assert "line 200" not in prompt.text
    assert "not already authoritative in code or documentation" in prompt.authoring_guidance
    assert "non-obvious build or tooling quirks" in prompt.authoring_guidance
    assert "architectural constraints" in prompt.authoring_guidance
    assert "recurring failure causes" in prompt.authoring_guidance
    assert "user-confirmed conventions" in prompt.authoring_guidance
    assert "Before finishing a substantive task" in prompt.authoring_guidance
    assert "200-line prompt budget" in prompt.authoring_guidance
    assert "targeted list_memory query" in prompt.authoring_guidance
    assert "first 200 of 250 lines" in prompt.text
    assert prompt.warnings and "first 200 of 250 lines" in prompt.warnings[0]
    assert "list_memory" in prompt.recall_guidance


def test_prompt_byte_bound_does_not_count_an_empty_partial_line(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = backend(tmp_path)
    monkeypatch.setattr("kolega_code.memory.markdown.PROMPT_MAX_BYTES", 5)
    assert store.write_entry("MEMORY.md", "1234\nnext\n").ok

    prompt = store.prepare_prompt_context()

    assert prompt.truncated is True
    assert prompt.line_count == 1
    assert "first 1 of 2 lines" in prompt.warnings[0]


def test_prompt_byte_bound_counts_a_nonempty_partial_line(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = backend(tmp_path)
    monkeypatch.setattr("kolega_code.memory.markdown.PROMPT_MAX_BYTES", 5)
    assert store.write_entry("MEMORY.md", "ééé\nnext\n").ok

    prompt = store.prepare_prompt_context()

    assert prompt.truncated is True
    assert prompt.line_count == 1
    assert "éé" in prompt.text
    assert "first 1 of 2 lines" in prompt.warnings[0]


def test_nested_entries_arbitrary_markdown_frontmatter_queries_and_titles(tmp_path: Path) -> None:
    store = backend(tmp_path)
    index = "---\ntitle: untouched\ncustom: [one, two]\n---\n# Project Memory\nBuild overview"
    topic = "---\nopaque: !!custom value\n---\nIntroduction\n## Build Notes\nDeployment sentinel"
    plain = "one\ntwo\nthree\nfour\nfive\n# Too Late\nDeployment details"
    assert store.write_entry("MEMORY.md", index).ok
    assert store.write_entry("topics/nested/build.md", topic).ok
    assert store.write_entry("topics/plain.md", plain).ok

    assert store.read_entry("MEMORY.md").content == index
    assert store.read_entry("topics/nested/build.md").content == topic
    entries = store.list_entries()
    assert [entry.reference for entry in entries] == [
        "MEMORY.md",
        "topics/nested/build.md",
        "topics/plain.md",
    ]
    assert [entry.display_name for entry in entries] == [
        "Project Memory",
        "Build Notes",
        "topics/plain.md",
    ]
    assert [entry.reference for entry in store.list_entries("bUiLd")] == [
        "MEMORY.md",
        "topics/nested/build.md",
    ]
    assert [entry.reference for entry in store.list_entries("dEpLoYmEnT sEnTiNeL")] == ["topics/nested/build.md"]
    assert [entry.reference for entry in store.list_entries("PLAIN.MD")] == ["topics/plain.md"]


def test_index_write_over_prompt_budget_warns(tmp_path: Path) -> None:
    store = backend(tmp_path)
    content = "".join(f"line {index}\n" for index in range(201))

    index_result = store.write_entry("MEMORY.md", content)
    topic_result = store.write_entry("topics/large.md", content)

    assert index_result.warnings
    assert "201 lines" in index_result.warnings[0]
    assert f"{len(content.encode()):,} bytes" in index_result.warnings[0]
    assert topic_result.warnings == ()


def test_recall_guidance_only_when_index_present(tmp_path: Path) -> None:
    store = backend(tmp_path)
    assert store.prepare_prompt_context().recall_guidance == ""
    assert store.write_entry("topics/build.md", "# Build").ok
    assert store.prepare_prompt_context().recall_guidance == ""

    assert store.write_entry("MEMORY.md", "- [Build](topics/build.md)").ok

    assert store.prepare_prompt_context().recall_guidance == (
        "The MEMORY.md index below is a table of contents, not the full memory. Read any linked "
        "topic relevant to the current task with read_memory before acting on it; use list_memory "
        "to search memory the index does not surface.\n"
    )


def test_count_and_total_limits_leave_existing_content(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = backend(tmp_path)
    assert store.write_entry("one.md", "1234").ok
    monkeypatch.setattr("kolega_code.memory.markdown.MAX_FILES", 1)
    assert not store.write_entry("two.md", "x").ok
    monkeypatch.setattr("kolega_code.memory.markdown.MAX_FILES", 100)
    monkeypatch.setattr("kolega_code.memory.markdown.MAX_TOTAL_BYTES", 5)
    assert not store.write_entry("one.md", "123456").ok
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


def test_private_modes_and_no_content_lock(tmp_path: Path) -> None:
    store = backend(tmp_path)
    assert store.write_entry("topics/build.md", "content").ok

    if os.name == "posix":
        assert store.root.stat().st_mode & 0o777 == 0o700
        assert (store.root / "topics").stat().st_mode & 0o777 == 0o700
        assert (store.root / "topics/build.md").stat().st_mode & 0o777 == 0o600
    assert not (store.root / ".memory.lock").exists()


def test_atomic_write_failure_leaves_existing_content_unchanged(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = backend(tmp_path)
    assert store.write_entry("MEMORY.md", "original").ok

    def fail_write(*args: object, **kwargs: object) -> None:
        raise OSError("injected failure")

    monkeypatch.setattr("kolega_code.memory.markdown.write_private_bytes", fail_write)
    with pytest.raises(MemorySafetyError, match="persist"):
        store.write_entry("MEMORY.md", "replacement")

    assert store.read_entry("MEMORY.md").content == "original"
