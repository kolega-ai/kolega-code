"""Tests for TUI /update and /restart exit actions and launcher re-exec."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Callable

import pytest

from kolega_code.cli import main as main_module
from kolega_code.cli.session_store import SessionRecord, SessionStore
from kolega_code.cli.tui.state import TuiExitAction
from kolega_code.cli.updater import UpdateRunResult


def test_every_tui_cli_option_is_classified_for_restart() -> None:
    parser = main_module._build_tui_parser()
    all_dests = {action.dest for action in parser._actions}
    flag_dests = {dest for dest, _ in main_module._RESTART_PRESERVED_FLAG_ARGS}
    value_dests = {dest for dest, _ in main_module._RESTART_PRESERVED_VALUE_ARGS}
    stripped_dests = set(main_module._RESTART_STRIPPED_ARG_DESTS)

    assert not (flag_dests & value_dests)
    assert not (flag_dests & stripped_dests)
    assert not (value_dests & stripped_dests)
    assert all_dests <= (flag_dests | value_dests | stripped_dests)


def test_build_restart_args_preserves_runtime_flags_and_strips_one_shot_flags(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    state_dir = tmp_path / "state"
    ext_cfg = tmp_path / "ext.json"
    args = main_module.parse_args(
        [
            str(project),
            "--new",
            "--name",
            "initial-name",
            "--permission-mode",
            "ask",
            "--create-worktree",
            "feat/demo",
            "--from",
            "main",
            "--worktree-path",
            str(tmp_path / "wt"),
            "--trust-hooks",
            "--trust-mcp",
            "--trust-lsp",
            "--show-logs",
            "--browser-visible",
            "--state-dir",
            str(state_dir),
            "--provider",
            "deepseek",
            "--model",
            "deepseek-flash",
            "--thinking-effort",
            "max",
            "--compression-threshold",
            "90",
            "--context-usage-reminders",
            "off",
            "--lsp",
            "off",
            "--subagents",
            "off",
            "--skills",
            "off",
            "--extension",
            "pkg.mod:factory",
            "--extension-config",
            str(ext_cfg),
        ]
    )
    action = TuiExitAction(kind="update_and_restart", session_id="sess-123", project_path=project.resolve())

    restart_argv = main_module._build_restart_args(args, action)
    reparsed = main_module.parse_args(restart_argv)

    assert reparsed.command == "tui"
    assert reparsed.project_path == project.resolve()
    assert reparsed.resume == "sess-123"
    assert reparsed.session is None
    assert reparsed.new is False
    assert reparsed.name is None
    assert reparsed.permission_mode is None
    assert reparsed.create_worktree is None
    assert reparsed.worktree is None
    assert reparsed.worktree_from is None
    assert reparsed.worktree_path is None
    assert reparsed.trust_hooks is False
    assert reparsed.trust_mcp is False
    assert reparsed.trust_lsp is False
    assert reparsed.show_logs is True
    assert reparsed.browser_visible is True
    assert reparsed.state_dir == state_dir
    assert reparsed.provider == "deepseek"
    assert reparsed.model == "deepseek-flash"
    assert reparsed.thinking_effort == "max"
    assert reparsed.compression_threshold == "90"
    assert reparsed.context_usage_reminders == "off"
    assert reparsed.lsp == "off"
    assert reparsed.subagents == "off"
    assert reparsed.skills == "off"
    assert reparsed.extension == "pkg.mod:factory"
    assert reparsed.extension_config == ext_cfg


def test_capture_launch_executable_keeps_symlink_path_and_filters_non_cli(tmp_path: Path) -> None:
    target = tmp_path / "tools" / "kolega-code" / "bin" / "kolega-code"
    target.parent.mkdir(parents=True)
    target.write_text("#!/bin/sh\n", encoding="utf-8")
    target.chmod(0o755)

    symlink = tmp_path / "bin" / "kolega-code"
    symlink.parent.mkdir(parents=True)
    symlink.symlink_to(target)

    captured = main_module._capture_launch_executable(str(symlink))
    # Must preserve the symlink path rather than resolving to the tool venv target.
    assert captured == symlink

    pytest_bin = tmp_path / "bin" / "pytest"
    pytest_bin.write_text("#!/bin/sh\n", encoding="utf-8")
    pytest_bin.chmod(0o755)
    assert main_module._capture_launch_executable(str(pytest_bin)) is None

    py_script = tmp_path / "kolega_code.py"
    py_script.write_text("#!/usr/bin/env python3\n", encoding="utf-8")
    py_script.chmod(0o755)
    assert main_module._capture_launch_executable(str(py_script)) is None


def test_resolve_restart_command_preferences(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cli_bin = tmp_path / "kolega-code"
    cli_bin.write_text("#!/bin/sh\n", encoding="utf-8")
    cli_bin.chmod(0o755)
    restart_args = ["tui", "/proj", "--resume", "s1"]

    # Launch executable takes precedence when present.
    assert main_module._resolve_restart_command(cli_bin, restart_args, prefer_installed_cli=True) == (
        str(cli_bin),
        [str(cli_bin), *restart_args],
    )

    # Without a launch executable (e.g. python -m), update_and_restart prefers
    # the installed CLI on PATH while plain restart keeps sys.executable -m.
    monkeypatch.setattr(main_module.shutil, "which", lambda name: "/usr/local/bin/kolega-code")
    assert main_module._resolve_restart_command(None, restart_args, prefer_installed_cli=True) == (
        "/usr/local/bin/kolega-code",
        ["/usr/local/bin/kolega-code", *restart_args],
    )
    resolved_restart = main_module._resolve_restart_command(None, restart_args, prefer_installed_cli=False)
    assert resolved_restart is not None
    assert resolved_restart[1][1:3] == ["-m", "kolega_code.cli.main"]


def _install_run_tui_harness(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    exit_action_factory: Callable[[SessionRecord], TuiExitAction | None],
    clean_quit: bool = True,
) -> tuple[Any, list[threading.Event], SessionStore]:
    import kolega_code.cli.app as app_module

    watchdogs: list[threading.Event] = []
    store = SessionStore(tmp_path / "state")

    def arm_watchdog() -> threading.Event:
        ev = threading.Event()
        watchdogs.append(ev)
        return ev

    class _FakeApp:
        def __init__(self, **kwargs: Any) -> None:
            self.session: SessionRecord = kwargs["session"]
            self.resume_session: SessionRecord | None = None
            self.exit_action: TuiExitAction | None = exit_action_factory(self.session)
            self._quit_cleanly = clean_quit
            self._agent_cleanup_failed = False
            self._on_shutdown_requested: Callable[[], None] | None = None
            store.claim_for_resume(self.session.session_id, Path(self.session.project_path))

        async def run_async(self) -> None:
            if self._on_shutdown_requested is not None:
                self._on_shutdown_requested()

        async def _cleanup_agent_generation(self) -> None:
            pass

    args = main_module.parse_args(["tui", str(tmp_path), "--state-dir", str(tmp_path / "state"), "--show-logs"])
    monkeypatch.setattr(app_module, "KolegaCodeApp", _FakeApp)
    monkeypatch.setattr(main_module, "_arm_exit_watchdog", arm_watchdog)
    monkeypatch.setattr(main_module, "_store_from_args", lambda _args: store)
    monkeypatch.setattr(main_module, "_select_startup_project", lambda path, _args: path.resolve())
    monkeypatch.setattr(main_module, "_resolve_extension_selection_from_args", lambda _args: None)
    monkeypatch.setattr(main_module, "build_agent_config", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(main_module, "config_summary", lambda _config: {})
    return args, watchdogs, store


def test_run_tui_update_disarms_watchdogs_releases_locks_updates_and_execs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    cli_bin = tmp_path / "bin" / "kolega-code"
    cli_bin.parent.mkdir()
    cli_bin.write_text("#!/bin/sh\n", encoding="utf-8")
    cli_bin.chmod(0o755)

    args, watchdogs, store = _install_run_tui_harness(
        monkeypatch,
        tmp_path,
        exit_action_factory=lambda s: TuiExitAction(
            kind="update_and_restart", session_id=s.session_id, project_path=Path(s.project_path)
        ),
    )
    events: list[str] = []
    exec_calls: list[tuple[str, list[str]]] = []
    hints: list[str] = []

    def fake_update() -> UpdateRunResult:
        # Both exit watchdogs and session locks must be cleared BEFORE uv runs.
        assert watchdogs and all(w.is_set() for w in watchdogs)
        assert not store._owned_session_locks
        events.append("update")
        return UpdateRunResult(returncode=0, gateway_restart_notes=("launchd restarted",))

    def fake_exec(path: str, argv: list[str]) -> int:
        events.append("exec")
        exec_calls.append((path, argv))
        return 0

    monkeypatch.setattr(main_module, "_capture_launch_executable", lambda: cli_bin)
    monkeypatch.setattr(main_module, "run_self_update", fake_update)
    monkeypatch.setattr(main_module, "count_other_live_sessions", lambda _root: 2)
    monkeypatch.setattr(main_module, "_exec_replace_process", fake_exec)
    monkeypatch.setattr(main_module, "_print_quit_resume_hint", hints.append)

    rc = main_module._run_tui(args)

    assert rc == 0
    assert events == ["update", "exec"]
    assert hints == []
    assert len(exec_calls) == 1
    exec_path, exec_argv = exec_calls[0]
    assert exec_path == str(cli_bin)
    assert exec_argv[0] == str(cli_bin)
    reparsed = main_module.parse_args(exec_argv[1:])
    assert reparsed.resume is not None
    assert reparsed.show_logs is True
    assert reparsed.state_dir == tmp_path / "state"
    out = capsys.readouterr().out
    assert "launchd restarted" in out
    assert "2 other running Kolega Code sessions" in out


def test_run_tui_update_failure_skips_exec_and_prints_resume_hint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    args, watchdogs, store = _install_run_tui_harness(
        monkeypatch,
        tmp_path,
        exit_action_factory=lambda s: TuiExitAction(
            kind="update_and_restart", session_id=s.session_id, project_path=Path(s.project_path)
        ),
    )
    exec_calls: list[Any] = []
    hints: list[str] = []

    monkeypatch.setattr(main_module, "run_self_update", lambda: UpdateRunResult(returncode=1, error="network down"))
    monkeypatch.setattr(main_module, "_exec_replace_process", lambda p, a: exec_calls.append((p, a)) or 0)
    monkeypatch.setattr(main_module, "_print_quit_resume_hint", hints.append)

    rc = main_module._run_tui(args)

    assert rc == 1
    assert exec_calls == []
    assert len(hints) == 1
    assert all(w.is_set() for w in watchdogs)
    assert not store._owned_session_locks
    assert "network down" in capsys.readouterr().err


def test_run_tui_restart_execs_without_updating(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cli_bin = tmp_path / "bin" / "kolega-code"
    cli_bin.parent.mkdir()
    cli_bin.write_text("#!/bin/sh\n", encoding="utf-8")
    cli_bin.chmod(0o755)

    args, watchdogs, store = _install_run_tui_harness(
        monkeypatch,
        tmp_path,
        exit_action_factory=lambda s: TuiExitAction(
            kind="restart", session_id=s.session_id, project_path=Path(s.project_path)
        ),
    )
    update_called = False
    exec_calls: list[tuple[str, list[str]]] = []

    def fail_if_update() -> UpdateRunResult:
        nonlocal update_called
        update_called = True
        return UpdateRunResult(returncode=0)

    monkeypatch.setattr(main_module, "_capture_launch_executable", lambda: cli_bin)
    monkeypatch.setattr(main_module, "run_self_update", fail_if_update)
    monkeypatch.setattr(main_module, "_exec_replace_process", lambda p, a: exec_calls.append((p, a)) or 0)

    rc = main_module._run_tui(args)

    assert rc == 0
    assert update_called is False
    assert len(exec_calls) == 1
    assert all(w.is_set() for w in watchdogs)
    assert not store._owned_session_locks


def test_run_tui_dirty_quit_ignores_exit_action(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    args, watchdogs, _store = _install_run_tui_harness(
        monkeypatch,
        tmp_path,
        exit_action_factory=lambda s: TuiExitAction(
            kind="update_and_restart", session_id=s.session_id, project_path=Path(s.project_path)
        ),
        clean_quit=False,
    )
    called: list[str] = []
    monkeypatch.setattr(main_module, "run_self_update", lambda: called.append("update") or UpdateRunResult(0))
    monkeypatch.setattr(main_module, "_exec_replace_process", lambda *_args: called.append("exec") or 0)

    rc = main_module._run_tui(args)

    assert rc == 0
    assert called == []
    assert all(w.is_set() for w in watchdogs)


def test_run_tui_exec_oserror_prints_error_and_resume_hint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    cli_bin = tmp_path / "bin" / "kolega-code"
    cli_bin.parent.mkdir()
    cli_bin.write_text("#!/bin/sh\n", encoding="utf-8")
    cli_bin.chmod(0o755)

    args, _watchdogs, _store = _install_run_tui_harness(
        monkeypatch,
        tmp_path,
        exit_action_factory=lambda s: TuiExitAction(
            kind="restart", session_id=s.session_id, project_path=Path(s.project_path)
        ),
    )
    hints: list[str] = []

    def boom(_path: str, _argv: list[str]) -> int:
        raise OSError("exec failed")

    monkeypatch.setattr(main_module, "_capture_launch_executable", lambda: cli_bin)
    monkeypatch.setattr(main_module, "_exec_replace_process", boom)
    monkeypatch.setattr(main_module, "_print_quit_resume_hint", hints.append)

    rc = main_module._run_tui(args)

    assert rc == 1
    assert len(hints) == 1
    assert "exec failed" in capsys.readouterr().err


def test_cli_update_subcommand_warns_about_other_live_sessions(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(main_module, "run_self_update", lambda: UpdateRunResult(returncode=0))
    monkeypatch.setattr(main_module, "count_other_live_sessions", lambda: 2)

    rc = main_module.main(["update"])

    assert rc == 0
    out = capsys.readouterr().out
    assert "2 other running Kolega Code sessions" in out
    assert "/restart" in out
