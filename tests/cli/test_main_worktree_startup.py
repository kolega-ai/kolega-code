from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from kolega_code.cli.session_store import SessionRecord
from kolega_code.worktrees import WorktreeInfo

from ._app_test_utils import FakeCoderAgent as _FakeCoderAgent


pytestmark = pytest.mark.usefixtures("hermetic_git_config")


@pytest.mark.parametrize(
    ("argv", "project_attribute"),
    [
        (["/repo", "--worktree", "feature"], "project_path"),
        (["tui", "/repo", "--worktree", "feature"], "project_path"),
        (["ask", "inspect", "--project", "/repo", "--worktree", "feature"], "project"),
    ],
)
def test_parse_worktree_selection_forms(argv: list[str], project_attribute: str) -> None:
    from kolega_code.cli.main import parse_args

    args = parse_args(argv)

    assert getattr(args, project_attribute) == Path("/repo")
    assert args.worktree == "feature"
    assert args.create_worktree is None


@pytest.mark.parametrize(
    "argv",
    [
        [
            "/repo",
            "--create-worktree",
            "fix/startup",
            "--from",
            "origin/main",
            "--worktree-path",
            "/tmp/startup",
        ],
        [
            "tui",
            "/repo",
            "--create-worktree",
            "fix/startup",
            "--from",
            "origin/main",
            "--worktree-path",
            "/tmp/startup",
        ],
        [
            "ask",
            "inspect",
            "--project",
            "/repo",
            "--create-worktree",
            "fix/startup",
            "--from",
            "origin/main",
            "--worktree-path",
            "/tmp/startup",
        ],
    ],
)
def test_parse_worktree_creation_forms(argv: list[str]) -> None:
    from kolega_code.cli.main import parse_args

    args = parse_args(argv)

    assert args.create_worktree == "fix/startup"
    assert args.worktree_from == "origin/main"
    assert args.worktree_path == Path("/tmp/startup")
    assert args.worktree is None


@pytest.mark.parametrize(
    "argv",
    [
        ["/repo", "--worktree", "feature", "--create-worktree", "other"],
        ["/repo", "--from", "main"],
        ["ask", "inspect", "--worktree-path", "/tmp/checkout"],
        ["/repo", "--create-worktree", "feature", "--resume"],
        ["tui", "/repo", "--create-worktree", "feature", "--session", "legacy-id"],
        ["ask", "inspect", "--create-worktree", "feature", "--session", "legacy-id"],
    ],
)
def test_parse_rejects_incompatible_worktree_options(argv: list[str]) -> None:
    from kolega_code.cli.main import parse_args

    with pytest.raises(SystemExit) as exc_info:
        parse_args(argv)

    assert exc_info.value.code == 2


def test_tui_selected_worktree_is_effective_before_project_services(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import kolega_code.cli.app as app_module
    import kolega_code.cli.main as main_module

    source = tmp_path / "source"
    selected = tmp_path / "selected"
    source.mkdir()
    selected.mkdir()
    calls: list[tuple[str, Path | None]] = []

    def fake_resolve(source_path: Path, target: str) -> WorktreeInfo:
        assert source_path == source.resolve()
        assert target == "feature"
        calls.append(("resolve", source_path))
        return WorktreeInfo(path=selected.resolve(), branch="feature", head="abc")

    class FakeSettings:
        permission_mode = "ask"

        def trust_hook_project(self, path: Path) -> None:
            calls.append(("trust_hooks", path))

        def trust_mcp_project(self, path: Path) -> None:
            calls.append(("trust_mcp", path))

        def trust_lsp_project(self, path: Path) -> None:
            calls.append(("trust_lsp", path))

    settings = FakeSettings()

    class FakeSettingsStore:
        def load(self) -> FakeSettings:
            calls.append(("settings", None))
            return settings

        def save(self, saved: FakeSettings) -> None:
            assert saved is settings
            calls.append(("save_settings", None))

    class FakeStore:
        def save(self, session: SessionRecord) -> None:
            calls.append(("save_session", Path(session.project_path)))

    session = SessionRecord.create(selected.resolve(), main_module.CLI_AGENT_MODE, {})

    def fake_build_config(path: Path, *args: Any, **kwargs: Any) -> object:
        del args, kwargs
        calls.append(("config", path))
        return object()

    def fake_resolve_session(store: object, path: Path, *args: Any) -> SessionRecord:
        del store, args
        calls.append(("session", path))
        return session

    class FakeApp:
        def __init__(self, **kwargs: Any) -> None:
            calls.append(("app", kwargs["project_path"]))
            assert kwargs["session"] is session

        def run(self) -> None:
            calls.append(("run", None))

    monkeypatch.setattr(main_module, "resolve_worktree", fake_resolve)
    monkeypatch.setattr(main_module, "_store_from_args", lambda _args: FakeStore())
    monkeypatch.setattr(main_module, "_settings_store_from_args", lambda _args: FakeSettingsStore())
    monkeypatch.setattr(main_module, "build_agent_config", fake_build_config)
    monkeypatch.setattr(main_module, "config_summary", lambda _config: {})
    monkeypatch.setattr(main_module, "_resolve_tui_session", fake_resolve_session)
    monkeypatch.setattr(app_module, "KolegaCodeApp", FakeApp)

    result = main_module.main(
        [
            str(source),
            "--worktree",
            "feature",
            "--trust-hooks",
            "--trust-mcp",
            "--trust-lsp",
        ]
    )

    assert result == 0
    assert calls[0] == ("resolve", source.resolve())
    for name in ("trust_hooks", "trust_mcp", "trust_lsp", "config", "session", "app"):
        assert (name, selected.resolve()) in calls


def test_ask_selected_worktree_flows_through_discovery_config_hooks_and_agent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    isolated_cli_env: None,
) -> None:
    del isolated_cli_env
    import kolega_code.cli.main as main_module

    source = tmp_path / "source"
    selected = tmp_path / "selected"
    state = tmp_path / "state"
    source.mkdir()
    selected.mkdir()
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    selected_info = WorktreeInfo(path=selected.resolve(), branch="feature", head="abc")
    monkeypatch.setattr(main_module, "resolve_worktree", lambda _source_path, _target: selected_info)

    seen: dict[str, list[Path]] = {"skills": [], "agents": [], "config": [], "hooks": []}
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

    class FakeCoderAgent(_FakeCoderAgent):
        agent_name = "coder"
        instances: list[FakeCoderAgent] = []

        def __init__(self, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            self.__class__.instances.append(self)

        async def process_message_stream(self, message: Any):
            self.messages.append(message)
            yield {"type": "response", "content": "ok", "complete": True, "uuid": "response-1"}

        async def cleanup(self) -> None:
            return None

    monkeypatch.setattr(main_module, "discover_skills", discover_skills)
    monkeypatch.setattr(main_module, "discover_custom_agents", discover_agents)
    monkeypatch.setattr(main_module, "build_agent_config", build_config)
    monkeypatch.setattr(main_module, "load_hook_config", load_hooks)
    monkeypatch.setattr(main_module, "CoderAgent", FakeCoderAgent)
    result = main_module.main(
        [
            "ask",
            "inspect",
            "--project",
            str(source),
            "--worktree",
            "feature",
            "--state-dir",
            str(state),
            "--provider",
            "anthropic",
            "--model",
            "claude-sonnet-4-6",
            "--trust-hooks",
        ]
    )

    assert result == 0
    assert all(paths == [selected.resolve()] for paths in seen.values())
    assert FakeCoderAgent.instances[0].kwargs["project_path"] == selected.resolve()
    from kolega_code.cli.settings import SettingsStore

    assert SettingsStore(state).load().is_hook_project_trusted(selected.resolve())


def test_created_worktree_path_is_reported_and_retained_when_startup_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import kolega_code.cli.main as main_module

    source = tmp_path / "source"
    created = tmp_path / "created"
    source.mkdir()

    def fake_create(*args: Any, **kwargs: Any) -> WorktreeInfo:
        del args, kwargs
        created.mkdir()
        return WorktreeInfo(path=created.resolve(), branch="feature", head="abc")

    monkeypatch.setattr(main_module, "create_worktree", fake_create)
    monkeypatch.setattr(
        main_module,
        "discover_skills",
        lambda _path: (_ for _ in ()).throw(ValueError("startup failed")),
    )

    result = main_module.main(["ask", "inspect", "--project", str(source), "--create-worktree", "feature"])

    captured = capsys.readouterr()
    assert result == 2
    assert created.is_dir()
    assert f"Created worktree for 'feature' at {created.resolve()}; it will be retained." in captured.err
    assert "startup failed" in captured.err or "startup failed" in captured.out


def test_direct_positional_worktree_path_does_not_invoke_explicit_selection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import kolega_code.cli.main as main_module

    direct = tmp_path / "direct"
    direct.mkdir()
    args = main_module.parse_args([str(direct)])

    monkeypatch.setattr(
        main_module,
        "resolve_worktree",
        lambda *_args, **_kwargs: pytest.fail("direct path should not be explicitly resolved"),
    )
    monkeypatch.setattr(
        main_module,
        "create_worktree",
        lambda *_args, **_kwargs: pytest.fail("direct path should not create a worktree"),
    )

    assert main_module._select_startup_project(args.project_path, args) == direct.resolve()
