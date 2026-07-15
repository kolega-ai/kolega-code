from __future__ import annotations

import json
import subprocess
import threading
import time
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from kolega_code.memory import (
    MemoryAccessError,
    MemoryAccessScope,
    MemoryBackendMetadata,
    MemoryBackendRegistry,
    MemoryCapability,
    MemoryToolBinding,
    MemoryUnavailableError,
    MemoryWriteResult,
    ProjectMemoryManager,
    resolve_project_identity,
)
from kolega_code.memory.models import MemoryBackendStatus, MemoryPromptContext


def test_lazy_private_persistence_and_project_isolation(tmp_path: Path) -> None:
    project = tmp_path / "repo"
    project.mkdir()
    state = tmp_path / "state"
    manager = ProjectMemoryManager(project, state)
    assert manager.enabled and not state.exists()
    assert not manager.status().manifest_exists
    result = manager.append_entry("MEMORY.md", "stable fact")
    assert result.ok
    assert manager.manifest_path.exists()
    assert not (project / "MEMORY.md").exists()
    reopened = ProjectMemoryManager(project, state)
    assert reopened.read_entry("MEMORY.md").content == "stable fact"
    other = tmp_path / "other"
    other.mkdir()
    assert not ProjectMemoryManager(other, state).read_entry("MEMORY.md").present


def test_disable_scope_and_unknown_backend_are_safe(tmp_path: Path) -> None:
    project = tmp_path / "p"
    project.mkdir()
    manager = ProjectMemoryManager(project, tmp_path / "state")
    manager.append_entry("MEMORY.md", "kept")
    child = manager.with_scope(MemoryAccessScope.SUBAGENT)
    manager.set_enabled(False)
    assert not child.enabled
    assert not manager.tool_bindings()
    with pytest.raises(MemoryUnavailableError):
        manager.read_entry("MEMORY.md")
    assert manager.read_entry("MEMORY.md", allow_disabled=True).content == "kept"
    manager.set_enabled(True)
    assert child.enabled
    assert child.with_scope(MemoryAccessScope.SUBAGENT) is child
    assert [binding.name for binding in child.tool_bindings()] == ["read_memory"]
    with pytest.raises(MemoryAccessError):
        child.append_entry("MEMORY.md", "no")
    with pytest.raises(MemoryAccessError, match="cannot derive writable access"):
        child.with_scope(MemoryAccessScope.TOP_LEVEL)
    manager.select_backend("not-installed")
    status = manager.status()
    assert not status.available and status.diagnostic
    assert not manager.tool_bindings()
    with pytest.raises(MemoryUnavailableError):
        manager.prompt_context()


@pytest.mark.parametrize(
    "payload",
    [
        "[]",
        "null",
        '{"schema_version": 1, "identity": [], "enabled": true, "backend_id": "markdown"}',
    ],
)
def test_structurally_invalid_manifest_fails_closed(
    tmp_path: Path,
    payload: str,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    state = tmp_path / "state"
    initial = ProjectMemoryManager(project, state)
    initial.memory_dir.mkdir(parents=True)
    initial.manifest_path.write_text(payload)

    reopened = ProjectMemoryManager(project, state)

    status = reopened.status()
    assert not reopened.enabled
    assert not status.available
    assert status.diagnostic and "invalid memory manifest" in status.diagnostic


def test_malformed_selected_backend_settings_fail_closed(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    state = tmp_path / "state"
    initial = ProjectMemoryManager(project, state)
    initial.set_enabled(True)
    payload = json.loads(initial.manifest_path.read_text())
    payload["backend_settings"] = {"markdown": []}
    initial.manifest_path.write_text(json.dumps(payload))

    reopened = ProjectMemoryManager(project, state)

    status = reopened.status()
    assert not reopened.enabled
    assert status.diagnostic and "backend settings" in status.diagnostic


def test_disable_is_serialized_against_in_flight_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    state = tmp_path / "state"
    writer = ProjectMemoryManager(project, state)
    disabler = ProjectMemoryManager(project, state)
    backend = writer.backend
    assert backend is not None
    append_started = threading.Event()
    release_append = threading.Event()
    disable_started = threading.Event()
    disable_finished = threading.Event()
    original_append = backend.append_entry

    def blocking_append(reference: str, content: str):
        append_started.set()
        assert release_append.wait(timeout=5)
        return original_append(reference, content)

    def disable() -> None:
        disable_started.set()
        disabler.set_enabled(False)
        disable_finished.set()

    monkeypatch.setattr(backend, "append_entry", blocking_append)
    with ThreadPoolExecutor(max_workers=2) as executor:
        write_future = executor.submit(writer.append_entry, "MEMORY.md", "committed")
        assert append_started.wait(timeout=5)
        disable_future = executor.submit(disable)
        assert disable_started.wait(timeout=5)
        assert not disable_finished.wait(timeout=0.05)
        release_append.set()
        assert write_future.result(timeout=5).ok
        disable_future.result(timeout=5)

    reopened = ProjectMemoryManager(project, state)
    assert not reopened.enabled
    assert reopened.read_entry("MEMORY.md", allow_disabled=True).content == "committed"


def test_backend_refresh_is_explicit_and_best_effort(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    manager = ProjectMemoryManager(project, tmp_path / "state")
    backend = manager.backend
    assert backend is not None

    def fail_refresh() -> None:
        raise RuntimeError("simulated backend refresh failure")

    monkeypatch.setattr(backend, "refresh", fail_refresh)

    result = manager.append_entry("MEMORY.md", "stable fact")

    assert result.ok
    assert manager.read_entry("MEMORY.md").content == "stable fact"
    assert manager.status().diagnostic is None

    manager.refresh()

    assert manager.status().diagnostic == "memory backend refresh failed"


def test_symlink_below_private_state_root_is_rejected(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    state = tmp_path / "state"
    manager = ProjectMemoryManager(project, state)
    outside = tmp_path / "outside"
    outside.mkdir()
    manager.project_dir.parent.mkdir(parents=True)
    manager.project_dir.symlink_to(outside, target_is_directory=True)

    status = manager.status()
    assert not status.available
    assert status.diagnostic and "symlink" in status.diagnostic
    with pytest.raises(MemoryUnavailableError, match="symlink"):
        manager.append_entry("MEMORY.md", "must not escape")
    assert not list(outside.iterdir())


def test_state_root_inside_project_is_rejected_without_repository_writes(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    state = project / ".private-state"
    manager = ProjectMemoryManager(project, state)

    assert not manager.status().available
    with pytest.raises(MemoryUnavailableError, match="inside the project"):
        manager.append_entry("MEMORY.md", "must stay outside")
    assert not state.exists()


def test_state_root_inside_containing_repository_is_rejected_from_subdirectory(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    project = repository / "packages" / "app"
    project.mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(repository)], check=True)
    state = repository / ".private-state"
    manager = ProjectMemoryManager(project, state)

    assert not manager.status().available
    with pytest.raises(MemoryUnavailableError, match="Git repository"):
        manager.append_entry("MEMORY.md", "must stay outside")
    assert not state.exists()


@pytest.mark.parametrize("artifact_name", ["manifest.json", ".manifest.lock"])
def test_manifest_artifact_symlinks_are_rejected_without_touching_target(
    tmp_path: Path,
    artifact_name: str,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    manager = ProjectMemoryManager(project, tmp_path / "state")
    manager.memory_dir.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.write_text("unchanged")
    (manager.memory_dir / artifact_name).symlink_to(outside)

    assert not manager.status().available
    with pytest.raises(MemoryUnavailableError, match="symlink"):
        manager.append_entry("MEMORY.md", "must not commit")
    assert outside.read_text() == "unchanged"


def test_concurrent_manifest_change_preserves_memory_opt_out(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    state = tmp_path / "state"
    first = ProjectMemoryManager(project, state)
    stale = ProjectMemoryManager(project, state)

    first.set_enabled(False)
    stale.select_backend("not-installed")

    reopened = ProjectMemoryManager(project, state)
    assert reopened.enabled is False
    assert reopened.backend_id == "not-installed"


@pytest.mark.parametrize("backend_id", ["../escape", "/absolute", "nested/name", "white space"])
def test_registry_rejects_backend_ids_that_are_not_safe_path_components(
    backend_id: str,
) -> None:
    registry = MemoryBackendRegistry()
    with pytest.raises(ValueError, match="backend ID"):
        registry.register(backend_id, lambda _path, _settings: FakeBackend())


def test_first_mutation_persists_manifest_before_backend_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    manager = ProjectMemoryManager(project, tmp_path / "state")

    def fail_manifest(*_args, **_kwargs) -> None:
        raise OSError("simulated manifest failure")

    monkeypatch.setattr("kolega_code.memory.manager.save_manifest", fail_manifest)
    with pytest.raises(OSError, match="simulated manifest failure"):
        manager.append_entry("MEMORY.md", "must not commit")
    assert manager.backend is not None
    assert not manager.backend.read_entry("MEMORY.md").present


class FakeBackend:
    metadata = MemoryBackendMetadata(
        "fake",
        "Fake",
        1,
        7,
        frozenset({MemoryCapability.PROMPT_CONTEXT}),
    )

    def status(self) -> MemoryBackendStatus:
        return MemoryBackendStatus(True, False)

    def initialize(self) -> None:
        pass

    def prepare_prompt_context(self) -> MemoryPromptContext:
        return MemoryPromptContext("opaque fake context")

    def list_entries(self, query=None):
        raise AssertionError("unsupported capability called")

    def read_entry(self, reference, *, redact=False):
        raise AssertionError("unsupported capability called")

    def append_entry(self, reference, content):
        raise AssertionError("unsupported capability called")

    def replace_entry(self, reference, content, expected_revision):
        raise AssertionError("unsupported capability called")

    def delete_entry(self, reference, expected_revision):
        raise AssertionError("unsupported capability called")

    def clear(self):
        raise AssertionError("unsupported capability called")

    def tool_bindings(self, scope: MemoryAccessScope) -> tuple[MemoryToolBinding, ...]:
        del scope
        return ()

    def refresh(self) -> None:
        pass

    def close(self) -> None:
        pass


def test_same_backend_settings_change_replaces_stale_backend(tmp_path: Path) -> None:
    created_settings: list[dict[str, object]] = []

    class ConfigurableBackend(FakeBackend):
        metadata = MemoryBackendMetadata(
            "fake",
            "Fake",
            1,
            1,
            frozenset({MemoryCapability.APPEND}),
        )

        def __init__(self, settings: dict[str, object]) -> None:
            self.settings = settings

        def append_entry(self, reference: str, content: str) -> MemoryWriteResult:
            del content
            return MemoryWriteResult(
                True,
                reference,
                revision="committed",
                byte_count=1,
            )

    def factory(
        _path: Path,
        settings: Mapping[str, object],
    ) -> ConfigurableBackend:
        captured = dict(settings)
        created_settings.append(captured)
        return ConfigurableBackend(captured)

    registry = MemoryBackendRegistry()
    registry.register("fake", factory)
    project = tmp_path / "project"
    project.mkdir()
    state = tmp_path / "state"
    first = ProjectMemoryManager(project, state, registry=registry)
    second = ProjectMemoryManager(project, state, registry=registry)
    first.select_backend("fake", {"generation": 1})
    assert first.append_entry("MEMORY.md", "first").ok
    second.select_backend("fake", {"generation": 2})

    assert first.append_entry("MEMORY.md", "second").ok

    assert created_settings == [{"generation": 1}, {"generation": 2}]


def test_concurrent_backend_initialization_creates_one_instance(tmp_path: Path) -> None:
    create_count = 0
    create_lock = threading.Lock()

    def factory(_path: Path, _settings: Mapping[str, object]) -> FakeBackend:
        nonlocal create_count
        time.sleep(0.02)
        with create_lock:
            create_count += 1
        return FakeBackend()

    registry = MemoryBackendRegistry()
    registry.register("fake", factory)
    project = tmp_path / "project"
    project.mkdir()
    manager = ProjectMemoryManager(project, tmp_path / "state", registry=registry)
    manager.select_backend("fake")

    with ThreadPoolExecutor(max_workers=8) as executor:
        statuses = list(executor.map(lambda _index: manager.status(), range(16)))

    assert all(status.available for status in statuses)
    assert create_count == 1


def test_configuration_change_attempts_backend_close_once(tmp_path: Path) -> None:
    class RetryCloseBackend(FakeBackend):
        def __init__(self, *, fail_first_close: bool) -> None:
            self.fail_first_close = fail_first_close
            self.close_calls = 0

        def close(self) -> None:
            self.close_calls += 1
            if self.fail_first_close and self.close_calls == 1:
                raise RuntimeError("simulated close failure")

    created: list[RetryCloseBackend] = []

    def factory(_path: Path, settings: Mapping[str, object]) -> RetryCloseBackend:
        backend = RetryCloseBackend(fail_first_close=settings.get("generation") == 1)
        created.append(backend)
        return backend

    registry = MemoryBackendRegistry()
    registry.register("fake", factory)
    project = tmp_path / "project"
    project.mkdir()
    manager = ProjectMemoryManager(project, tmp_path / "state", registry=registry)
    manager.select_backend("fake", {"generation": 1})
    assert manager.backend is created[0]

    manager.select_backend("fake", {"generation": 2})
    assert created[0].close_calls == 1
    assert manager.backend is created[1]

    manager.close()

    assert created[0].close_calls == 1
    assert created[1].close_calls == 1


def test_close_is_terminal_idempotent_and_cleanup_is_best_effort(tmp_path: Path) -> None:
    class CloseOnceBackend(FakeBackend):
        def __init__(self) -> None:
            self.close_calls = 0

        def close(self) -> None:
            self.close_calls += 1
            if self.close_calls == 1:
                raise RuntimeError("simulated close failure")

    created: list[CloseOnceBackend] = []

    def factory(_path: Path, _settings: Mapping[str, object]) -> CloseOnceBackend:
        backend = CloseOnceBackend()
        created.append(backend)
        return backend

    registry = MemoryBackendRegistry()
    registry.register("fake", factory)
    project = tmp_path / "project"
    project.mkdir()
    manager = ProjectMemoryManager(project, tmp_path / "state", registry=registry)
    manager.select_backend("fake")
    assert manager.backend is created[0]

    manager.close()

    assert manager.backend is None
    assert manager.tool_bindings() == ()
    status = manager.status()
    assert not status.available
    assert status.diagnostic == "project memory manager is closed"
    with pytest.raises(MemoryUnavailableError, match="manager is closed"):
        manager.prompt_context()
    assert len(created) == 1

    manager.close()

    assert created[0].close_calls == 1
    assert len(created) == 1


def test_registry_fake_backend_has_no_markdown_assumptions(tmp_path: Path) -> None:
    registry = MemoryBackendRegistry()
    registry.register("fake", lambda _path, _settings: FakeBackend())
    project = tmp_path / "p"
    project.mkdir()
    manager = ProjectMemoryManager(project, tmp_path / "state", registry=registry)
    manager.select_backend("fake", {"opaque": True})
    top_context = manager.prompt_context().text
    assert "opaque fake context" in top_context
    assert "MEMORY.md" not in top_context
    assert "read_memory" not in top_context
    child_context = manager.with_scope(MemoryAccessScope.SUBAGENT).prompt_context().text
    assert "read-only access" in child_context
    assert "Record only stable" not in child_context
    with pytest.raises(MemoryUnavailableError):
        manager.list_entries()


@pytest.mark.parametrize(
    ("metadata", "diagnostic"),
    [
        (
            MemoryBackendMetadata(
                "different-id",
                "Fake",
                1,
                1,
                frozenset({MemoryCapability.PROMPT_CONTEXT}),
            ),
            "declared a different backend ID",
        ),
        (
            MemoryBackendMetadata(
                "fake",
                "Fake",
                999,
                1,
                frozenset({MemoryCapability.PROMPT_CONTEXT}),
            ),
            "unsupported contract version",
        ),
    ],
)
def test_incompatible_backend_metadata_fails_closed(
    tmp_path: Path,
    metadata: MemoryBackendMetadata,
    diagnostic: str,
) -> None:
    class IncompatibleBackend(FakeBackend):
        def __init__(self) -> None:
            self.close_calls = 0

        def close(self) -> None:
            self.close_calls += 1

    IncompatibleBackend.metadata = metadata
    created: list[IncompatibleBackend] = []

    def factory(_path: Path, _settings: Mapping[str, object]) -> IncompatibleBackend:
        backend = IncompatibleBackend()
        created.append(backend)
        return backend

    registry = MemoryBackendRegistry()
    registry.register("fake", factory)
    project = tmp_path / "project"
    project.mkdir()
    manager = ProjectMemoryManager(project, tmp_path / "state", registry=registry)
    manager.select_backend("fake")

    status = manager.status()

    assert not status.available
    assert status.backend is None
    assert status.diagnostic and diagnostic in status.diagnostic
    assert manager.tool_bindings() == ()
    assert created[0].close_calls == 1
    with pytest.raises(MemoryUnavailableError):
        manager.prompt_context()


def test_identity_paths_symlinks_and_git_worktrees(tmp_path: Path) -> None:
    plain = tmp_path / "plain"
    plain.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(plain, target_is_directory=True)
    assert resolve_project_identity(plain).identity == resolve_project_identity(alias).identity

    repo = tmp_path / "repo"
    worktree = tmp_path / "worktree"
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.email", "test@example.invalid"], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "Test"], check=True)
    (repo / "a").write_text("a")
    subprocess.run(["git", "-C", str(repo), "add", "a"], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-qm", "init"], check=True)
    subprocess.run(["git", "-C", str(repo), "worktree", "add", "-q", str(worktree)], check=True)
    repo_identity = resolve_project_identity(repo)
    worktree_identity = resolve_project_identity(worktree)
    assert repo_identity.identity == worktree_identity.identity
    assert repo_identity.directory_key == worktree_identity.directory_key

    state = tmp_path / "state"
    primary_manager = ProjectMemoryManager(repo, state)
    linked_manager = ProjectMemoryManager(worktree, state)
    assert primary_manager.append_entry("MEMORY.md", "shared fact").ok
    assert linked_manager.read_entry("MEMORY.md").content == "shared fact"
