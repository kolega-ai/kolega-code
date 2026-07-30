from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import pytest

from kolega_code.cli.main import _active_project_for_resume
from kolega_code.cli.session_store import SessionStore, SessionStoreError
from kolega_code.worktrees import WorktreeInfo

from ._app_test_utils import FakeCoderAgent as _FakeCoderAgent


pytestmark = pytest.mark.usefixtures("hermetic_git_config")


def _git(project: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=project, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def _repository_with_worktree(tmp_path: Path) -> tuple[Path, Path]:
    main = tmp_path / "main"
    linked = tmp_path / "linked"
    main.mkdir()
    _git(main, "init", "-b", "main")
    _git(main, "config", "user.email", "test@example.com")
    _git(main, "config", "user.name", "Test User")
    (main / "tracked.txt").write_text("initial\n", encoding="utf-8")
    _git(main, "add", ".")
    _git(main, "commit", "-m", "initial")
    _git(main, "worktree", "add", "-b", "feature/resume", str(linked))
    return main.resolve(), linked.resolve()


def test_resume_restores_valid_active_worktree(tmp_path: Path) -> None:
    main, linked = _repository_with_worktree(tmp_path)
    store = SessionStore(tmp_path / "state")
    session = store.create(main, "code", {})
    session.active_project_path = str(linked)
    store.save(session)

    active = _active_project_for_resume(store.load(session.session_id), store)

    assert active == linked
    assert store.load(session.session_id).active_project_path == str(linked)


def test_resume_falls_back_to_launch_checkout_and_clears_deleted_active_path(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    main, linked = _repository_with_worktree(tmp_path)
    store = SessionStore(tmp_path / "state")
    session = store.create(main, "code", {})
    session.active_project_path = str(linked)
    store.save(session)
    _git(main, "worktree", "remove", str(linked))

    active = _active_project_for_resume(store.load(session.session_id), store)

    assert active == main
    assert store.load(session.session_id).active_project_path is None
    assert "Saved active worktree is unavailable" in capsys.readouterr().err


def test_resume_uses_canonical_fallback_when_metadata_projection_write_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    main, linked = _repository_with_worktree(tmp_path)
    store = SessionStore(tmp_path / "state")
    session = store.create(main, "code", {})
    session.active_project_path = str(linked)
    store.save(session)
    _git(main, "worktree", "remove", str(linked))
    original_write_metadata = store._write_metadata
    failed = False

    def fail_once(metadata: dict[str, Any]) -> None:
        nonlocal failed
        if metadata.get("active_project_path") is None and not failed:
            failed = True
            raise OSError("projection unavailable")
        original_write_metadata(metadata)

    monkeypatch.setattr(store, "_write_metadata", fail_once)

    active = _active_project_for_resume(store.load(session.session_id), store)

    assert active == main
    assert store.load(session.session_id).active_project_path is None
    assert "Saved active worktree is unavailable" in capsys.readouterr().err


def test_resume_fails_when_active_and_launch_checkouts_are_unavailable(tmp_path: Path) -> None:
    main, linked = _repository_with_worktree(tmp_path)
    store = SessionStore(tmp_path / "state")
    session = store.create(main, "code", {})
    session.active_project_path = str(linked)
    store.save(session)
    session = store.load(session.session_id)
    session.project_path = str(tmp_path / "missing-launch")
    session.active_project_path = str(tmp_path / "missing-active")

    with pytest.raises(SessionStoreError, match="cannot resume"):
        _active_project_for_resume(session, store)

    # A hard resume failure must not clear the persisted selection.
    assert store.load(session.session_id).active_project_path == str(linked)


def test_ask_resume_uses_active_tools_but_preserves_launch_identity_services(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    isolated_cli_env: None,
) -> None:
    del isolated_cli_env
    import kolega_code.cli.main as main_module
    from kolega_code.cli.settings import SettingsStore

    main, linked = _repository_with_worktree(tmp_path)
    state = tmp_path / "state"
    store = SessionStore(state)
    session = store.create(main, "code", {})
    session.active_project_path = str(linked)
    store.save(session)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    seen: dict[str, list[Path]] = {"skills": [], "agents": [], "config": [], "hooks": [], "mcp": []}
    original_discover_skills = main_module.discover_skills
    original_discover_agents = main_module.discover_custom_agents
    original_build_config = main_module.build_agent_config
    original_load_hooks = main_module.load_hook_config

    def discover_skills(path: Path):
        seen["skills"].append(path)
        return original_discover_skills(path)

    def discover_agents(path: Path, state_dir: Path):
        seen["agents"].append(path)
        return original_discover_agents(path, state_dir)

    def build_config(path: Path, *args: Any, **kwargs: Any):
        seen["config"].append(path)
        return original_build_config(path, *args, **kwargs)

    def load_hooks(path: Path, *args: Any, **kwargs: Any):
        seen["hooks"].append(path)
        return original_load_hooks(path, *args, **kwargs)

    def build_mcp(path: Path, *_args: Any, **kwargs: Any):
        seen["mcp"].append(path)
        assert kwargs["project_trusted"] is True
        assert kwargs["loaded_config"] is None
        return None

    class FakeCoderAgent(_FakeCoderAgent):
        agent_name = "coder"
        instances: list[FakeCoderAgent] = []

        def __init__(self, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            self.__class__.instances.append(self)

    monkeypatch.setattr(main_module, "discover_skills", discover_skills)
    monkeypatch.setattr(main_module, "discover_custom_agents", discover_agents)
    monkeypatch.setattr(main_module, "build_agent_config", build_config)
    monkeypatch.setattr(main_module, "load_hook_config", load_hooks)
    monkeypatch.setattr(main_module, "build_mcp_tool_extension", build_mcp)
    monkeypatch.setattr(main_module, "CoderAgent", FakeCoderAgent)

    result = main_module.main(
        [
            "ask",
            "inspect",
            "--project",
            str(main),
            "--session",
            session.session_id,
            "--state-dir",
            str(state),
            "--provider",
            "anthropic",
            "--model",
            "claude-sonnet-4-6",
            "--trust-hooks",
            "--trust-mcp",
            "--trust-lsp",
        ]
    )

    assert result == 0
    assert seen["skills"] == [linked]
    assert seen["agents"] == [linked]
    assert seen["mcp"] == [linked]
    assert seen["config"] == [main]
    assert seen["hooks"] == [main]
    agent_kwargs = FakeCoderAgent.instances[0].kwargs
    assert agent_kwargs["project_path"] == linked
    assert agent_kwargs["memory_project_path"] == main
    settings = SettingsStore(state).load()
    assert settings.is_hook_project_trusted(main)
    assert settings.is_mcp_project_trusted(main)
    assert settings.is_lsp_project_trusted(main)
    assert not settings.is_hook_project_trusted(linked)
    assert not settings.is_mcp_project_trusted(linked)
    assert not settings.is_lsp_project_trusted(linked)


@pytest.mark.parametrize("select_worktree", [False, True])
def test_ask_session_rejects_another_launch_checkout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    select_worktree: bool,
) -> None:
    import kolega_code.cli.main as main_module

    requested = tmp_path / "requested"
    selected = tmp_path / "selected"
    other = tmp_path / "other"
    state = tmp_path / "state"
    requested.mkdir()
    selected.mkdir()
    other.mkdir()
    store = SessionStore(state)
    session = store.create(other, "code", {})

    argv = [
        "ask",
        "inspect",
        "--project",
        str(requested),
        "--session",
        session.session_id,
        "--state-dir",
        str(state),
    ]
    expected_project = requested.resolve()
    if select_worktree:
        expected_project = selected.resolve()
        argv.extend(["--worktree", "feature"])
        monkeypatch.setattr(
            main_module,
            "resolve_worktree",
            lambda source, target: (
                WorktreeInfo(
                    path=selected.resolve(),
                    branch="feature",
                    head="abc",
                )
                if target == "feature" and source == requested.resolve()
                else pytest.fail(f"unexpected worktree resolution: {source}, {target}")
            ),
        )

    result = main_module.main(argv)

    assert result == 2
    assert f"belongs to project {other.resolve()}, not {expected_project}" in capsys.readouterr().err
