import asyncio
import json
from pathlib import Path

import pytest

from kolega_code import __version__
from kolega_code.config import ModelProvider
from kolega_code.cli.main import (
    CLI_AGENT_MODE,
    RESUME_LATEST,
    _resolve_tui_permission_mode,
    _resolve_tui_session,
    main,
    parse_args,
)
from kolega_code.cli.provider_registry import DEEPSEEK_DEFAULT_MODEL, UI_DEFAULT_MODEL, UI_DEFAULT_PROVIDER
from kolega_code.cli.session_store import SessionStore, SessionStoreError
from kolega_code.cli.settings import CliSettings, SettingsStore
from kolega_code.cli.updater import UpdateCheckResult, UpdateRunResult
from kolega_code.agent.prompt_overrides import PROMPT_OVERRIDE_DIR
from kolega_code.llm.exceptions import LLMBillingError
from ._app_test_utils import FakeCoderAgent as _FakeCoderAgent


def write_skill(root: Path, name: str = "demo-skill") -> None:
    skill_dir = root / ".agents" / "skills" / name
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: Use this demo skill.\n---\n\nFollow demo instructions.\n",
        encoding="utf-8",
    )


def no_update_result() -> UpdateCheckResult:
    return UpdateCheckResult(current_version=__version__, latest_version=__version__, update_available=False)


def test_parse_default_command_as_tui() -> None:
    args = parse_args(["/tmp/project", "--new"])

    assert args.command == "tui"
    assert args.project_path == Path("/tmp/project")
    assert args.new is True
    assert args.resume is None
    assert args.mode == CLI_AGENT_MODE
    assert args.permission_mode is None
    assert args.show_logs is False


def test_parse_explicit_tui_subcommand() -> None:
    args = parse_args(["tui", "/tmp/project"])

    assert args.command == "tui"
    assert args.project_path == Path("/tmp/project")
    assert args.mode == CLI_AGENT_MODE


@pytest.mark.parametrize("flag", ["--help", "-h"])
def test_root_help_uses_generated_command_help(flag: str, capsys) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main([flag])

    assert exc_info.value.code == 0
    output = capsys.readouterr().out
    assert "usage: kolega-code" in output
    assert "commands:" in output
    for command in ["ask", "sessions", "doctor", "prompts", "update", "tui"]:
        assert command in output


def test_explicit_tui_help_shows_tui_arguments(capsys) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["tui", "--help"])

    assert exc_info.value.code == 0
    output = capsys.readouterr().out
    assert "usage: kolega-code tui" in output
    assert "project_path" in output
    assert "--resume" in output
    assert "--permission-mode" in output


def test_parse_tui_show_logs_flag() -> None:
    args = parse_args(["/tmp/project", "--show-logs"])

    assert args.command == "tui"
    assert args.show_logs is True


def test_version_flag_prints_package_version(capsys, monkeypatch: pytest.MonkeyPatch) -> None:
    from kolega_code.cli import main as main_module

    monkeypatch.setattr(main_module, "check_for_update", no_update_result)

    exit_code = main_module.main(["--version"])

    assert exit_code == 0
    assert f"kolega-code {__version__}" in capsys.readouterr().out


def test_version_flag_prints_available_update(capsys, monkeypatch: pytest.MonkeyPatch) -> None:
    from kolega_code.cli import main as main_module

    monkeypatch.setattr(
        main_module,
        "check_for_update",
        lambda: UpdateCheckResult(current_version="0.2.0", latest_version="0.3.0", update_available=True),
    )

    exit_code = main_module.main(["--version"])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "kolega-code" in output
    assert "Update available: 0.2.0 -> 0.3.0" in output


def test_parse_tui_resume_latest() -> None:
    args = parse_args(["/tmp/project", "--resume"])

    assert args.command == "tui"
    assert args.resume == RESUME_LATEST


def test_parse_tui_resume_specific_thread() -> None:
    args = parse_args(["/tmp/project", "--resume", "thread-123"])

    assert args.command == "tui"
    assert args.resume == "thread-123"


def test_parse_tui_legacy_session_alias() -> None:
    args = parse_args(["/tmp/project", "--session", "session-123"])

    assert args.command == "tui"
    assert args.session == "session-123"


def test_parse_ask_subcommand() -> None:
    args = parse_args(["ask", "hello", "--project", "/tmp/project", "--save", "--json", "--thinking-effort", "high"])

    assert args.command == "ask"
    assert args.prompt == "hello"
    assert args.project == Path("/tmp/project")
    assert args.save is True
    assert args.json is True
    assert args.mode == CLI_AGENT_MODE
    assert args.thinking_effort == "high"
    assert args.permission_mode == "auto"


def test_parse_permission_mode_flags() -> None:
    tui_args = parse_args(["/tmp/project", "--permission-mode", "ask"])
    ask_args = parse_args(["ask", "hello", "--project", "/tmp/project", "--permission-mode", "ask"])

    assert tui_args.permission_mode == "ask"
    assert ask_args.permission_mode == "ask"


def test_parse_sessions_list_subcommand() -> None:
    args = parse_args(["sessions", "list", "--project", "/tmp/project"])

    assert args.command == "sessions"
    assert args.sessions_command == "list"
    assert args.project == Path("/tmp/project")


def test_parse_update_subcommand() -> None:
    args = parse_args(["update"])

    assert args.command == "update"


def test_parse_prompts_dump_subcommand() -> None:
    args = parse_args(["prompts", "dump", "coder", "planning", "--project", "/tmp/project", "--force"])

    assert args.command == "prompts"
    assert args.prompts_command == "dump"
    assert args.prompt_selectors == ["coder", "planning"]
    assert args.project == Path("/tmp/project")
    assert args.force is True


def test_parse_prompts_validate_subcommand() -> None:
    args = parse_args(["prompts", "validate", "--project", "/tmp/project"])

    assert args.command == "prompts"
    assert args.prompts_command == "validate"
    assert args.project == Path("/tmp/project")


def test_prompts_dump_subcommand_writes_prompt_files(tmp_path: Path, capsys) -> None:
    project = tmp_path / "project"
    project.mkdir()

    exit_code = main(["prompts", "dump", "--project", str(project)])

    assert exit_code == 0
    assert (project / ".kolega" / "prompts" / "CODER.md").is_file()
    assert "Written:" in capsys.readouterr().out


def test_prompts_dump_subcommand_writes_only_selected_prompt_files(tmp_path: Path, capsys) -> None:
    project = tmp_path / "project"
    project.mkdir()

    exit_code = main(["prompts", "dump", "coder", "compaction", "--project", str(project)])

    assert exit_code == 0
    assert (project / PROMPT_OVERRIDE_DIR / "CODER.md").is_file()
    assert (project / PROMPT_OVERRIDE_DIR / "COMPACTION.md").is_file()
    assert not (project / PROMPT_OVERRIDE_DIR / "PLANNING.md").exists()
    assert "Written:" in capsys.readouterr().out


def test_prompts_validate_subcommand_reports_valid_or_missing_prompts(tmp_path: Path, capsys) -> None:
    project = tmp_path / "project"
    project.mkdir()

    exit_code = main(["prompts", "validate", "--project", str(project)])

    assert exit_code == 0
    assert "nothing to validate" in capsys.readouterr().out


def test_prompts_validate_subcommand_returns_one_for_malformed_prompt(tmp_path: Path, capsys) -> None:
    project = tmp_path / "project"
    project.mkdir()
    prompt_dir = project / PROMPT_OVERRIDE_DIR
    prompt_dir.mkdir(parents=True)
    (prompt_dir / "GENERAL.md").write_text("{{ missing_variable }}", encoding="utf-8")

    exit_code = main(["prompts", "validate", "--project", str(project)])

    assert exit_code == 1
    output = capsys.readouterr().out
    assert "Could not render prompt override .kolega/prompts/GENERAL.md" in output
    assert "Falling back to the default prompt" in output


def test_prompts_list_subcommand_reports_prompt_files(tmp_path: Path, capsys) -> None:
    project = tmp_path / "project"
    project.mkdir()

    exit_code = main(["prompts", "list", "--project", str(project)])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "CODER.md" in output
    assert "missing" in output


def test_update_subcommand_runs_self_update(capsys, monkeypatch: pytest.MonkeyPatch) -> None:
    from kolega_code.cli import main as main_module

    monkeypatch.setattr(main_module, "run_self_update", lambda: UpdateRunResult(returncode=0))

    exit_code = main_module.main(["update"])

    assert exit_code == 0
    assert "update completed" in capsys.readouterr().out


def test_update_subcommand_reports_missing_uv(capsys, monkeypatch: pytest.MonkeyPatch) -> None:
    from kolega_code.cli import main as main_module

    monkeypatch.setattr(
        main_module,
        "run_self_update",
        lambda: UpdateRunResult(returncode=2, error="uv is required to update Kolega Code."),
    )

    exit_code = main_module.main(["update"])

    assert exit_code == 2
    captured = capsys.readouterr()
    assert "uv is required" in captured.err or "uv is required" in captured.out


def test_install_script_upgrades_existing_install() -> None:
    root = Path(__file__).resolve().parents[2]
    installer = (root / "scripts" / "install-kolega-code.sh").read_text(encoding="utf-8")

    assert 'uv tool install --force --upgrade "$PACKAGE_SPEC"' in installer


def test_ask_skills_lists_discovered_skills_without_api_key(tmp_path: Path, capsys, isolated_cli_env: None) -> None:
    project = tmp_path / "project"
    project.mkdir()
    write_skill(project)

    exit_code = main(["ask", "/skills", "--project", str(project)])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "`/demo-skill`" in output
    assert "Use this demo skill." in output


def test_ask_skill_only_prints_activation_without_model_call(tmp_path: Path, capsys, isolated_cli_env: None) -> None:
    project = tmp_path / "project"
    project.mkdir()
    write_skill(project)

    exit_code = main(["ask", "/demo-skill", "--project", str(project)])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert '<skill_content name="demo-skill">' in output
    assert "Follow demo instructions." in output


def test_ask_requires_model_selection_even_with_api_key(
    tmp_path: Path, capsys, monkeypatch: pytest.MonkeyPatch, isolated_cli_env: None
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    exit_code = main(["ask", "hello", "--project", str(project)])

    assert exit_code == 2
    captured = capsys.readouterr()
    assert "No provider/model configured" in captured.out or "No provider/model configured" in captured.err


def test_ask_skill_with_prompt_activates_before_dispatch(
    tmp_path: Path, capsys, monkeypatch: pytest.MonkeyPatch, isolated_cli_env: None
) -> None:
    from kolega_code.cli import main as main_module

    class FakeCoderAgent(_FakeCoderAgent):
        agent_name = "coder"
        instances = []

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.cleaned = False
            self.__class__.instances.append(self)

        async def process_message_stream(self, message):
            self.messages.append(message)
            yield {"type": "response", "content": "ok", "complete": True, "uuid": "response-1"}

        async def cleanup(self):
            self.cleaned = True

    project = tmp_path / "project"
    project.mkdir()
    write_skill(project)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("KOLEGA_CODE_PROVIDER", "anthropic")
    monkeypatch.setattr(main_module, "CoderAgent", FakeCoderAgent)

    exit_code = main_module.main(["ask", "/demo-skill do the task", "--project", str(project)])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "ok" in output
    agent = FakeCoderAgent.instances[0]
    assert agent.messages == ["do the task"]
    assert '<skill_content name="demo-skill">' in agent.history[0].get_text_content()
    assert any(extension.name == "cli-agent-skills" for extension in agent.kwargs["tool_extensions"])


def test_ask_plain_handles_billing_error_without_traceback(
    tmp_path: Path, capsys, monkeypatch: pytest.MonkeyPatch, isolated_cli_env: None
) -> None:
    from kolega_code.cli import main as main_module

    class FakeCoderAgent(_FakeCoderAgent):
        agent_name = "coder"
        instances = []

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.cleaned = False
            self.__class__.instances.append(self)

        async def process_message_stream(self, message):
            raise LLMBillingError("DeepSeek APIError: Insufficient Balance", provider=ModelProvider.DEEPSEEK.value)
            yield {"type": "response", "content": "unreachable"}

        async def cleanup(self):
            self.cleaned = True

    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-key")
    monkeypatch.setattr(main_module, "CoderAgent", FakeCoderAgent)

    exit_code = main_module.main(
        [
            "ask",
            "test",
            "--project",
            str(project),
            "--provider",
            ModelProvider.DEEPSEEK.value,
            "--model",
            DEEPSEEK_DEFAULT_MODEL,
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.out == ""
    assert "DeepSeek/deepseek-v4-pro could not run this request" in captured.err
    assert "Add credits to your DeepSeek account" in captured.err
    assert FakeCoderAgent.instances[0].cleaned is True


def test_ask_json_handles_billing_error_without_traceback(
    tmp_path: Path, capsys, monkeypatch: pytest.MonkeyPatch, isolated_cli_env: None
) -> None:
    from kolega_code.cli import main as main_module

    class FakeCoderAgent(_FakeCoderAgent):
        agent_name = "coder"

        async def process_message_stream(self, message):
            raise LLMBillingError(
                "DeepSeek APIError: Insufficient Balance raw-secret-token",
                provider="raw-exception-provider",
            )
            yield {"type": "response", "content": "unreachable"}

    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-key")
    monkeypatch.setattr(main_module, "CoderAgent", FakeCoderAgent)

    exit_code = main_module.main(
        [
            "ask",
            "test",
            "--project",
            str(project),
            "--provider",
            ModelProvider.DEEPSEEK.value,
            "--model",
            DEEPSEEK_DEFAULT_MODEL,
            "--json",
        ]
    )

    captured = capsys.readouterr()
    lines = [json.loads(line) for line in captured.out.splitlines() if line.strip()]
    assert exit_code == 1
    assert lines[-1]["kind"] == "error"
    assert lines[-1]["data"]["type"] == "billing_error"
    assert lines[-1]["data"]["provider"] == "configured"
    assert "The selected provider could not run this request" in lines[-1]["data"]["message"]
    assert "DeepSeek/deepseek-v4-pro" not in captured.out
    assert "raw-secret-token" not in captured.out
    assert "raw-exception-provider" not in captured.out
    assert "Traceback" not in captured.err


def test_doctor_uses_stored_kimi_settings(
    tmp_path: Path, capsys, monkeypatch: pytest.MonkeyPatch, isolated_cli_env: None
) -> None:
    from kolega_code.cli import main as main_module

    project = tmp_path / "project"
    project.mkdir()
    state_dir = tmp_path / "state"
    settings = CliSettings(active_provider=UI_DEFAULT_PROVIDER, active_model=UI_DEFAULT_MODEL)
    settings.set_api_key(UI_DEFAULT_PROVIDER, "moonshot-key")
    SettingsStore(state_dir).save(settings)
    monkeypatch.setattr(main_module, "check_for_update", no_update_result)

    exit_code = main_module.main(["doctor", "--project", str(project), "--state-dir", str(state_dir)])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert f"Update: Kolega Code is up to date ({__version__})." in output
    assert f"Stored active model: {UI_DEFAULT_PROVIDER}/{UI_DEFAULT_MODEL}" in output
    assert "Thinking effort: auto" in output
    assert "Stored API key" not in output
    assert "moonshot-key" not in output


def test_doctor_requires_model_selection_even_with_api_key(
    tmp_path: Path, capsys, monkeypatch: pytest.MonkeyPatch, isolated_cli_env: None
) -> None:
    from kolega_code.cli import main as main_module

    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(main_module, "check_for_update", no_update_result)

    exit_code = main_module.main(["doctor", "--project", str(project)])

    assert exit_code == 2
    output = capsys.readouterr().out
    assert "Stored active model: not configured" in output
    assert "No provider/model configured" in output


def test_deprecated_thinking_tokens_flag_fails(
    tmp_path: Path, capsys, monkeypatch: pytest.MonkeyPatch, isolated_cli_env: None
) -> None:
    from kolega_code.cli import main as main_module

    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.setattr(main_module, "check_for_update", no_update_result)

    exit_code = main_module.main(
        [
            "doctor",
            "--project",
            str(project),
            "--thinking-tokens",
            "1024",
        ]
    )

    assert exit_code == 2
    captured = capsys.readouterr()
    assert "--thinking-effort" in captured.out or "--thinking-effort" in captured.err


def test_tui_default_creates_new_session_even_when_latest_exists(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    store = SessionStore(tmp_path / "state")
    existing = store.create(project, "code", {})

    session = _resolve_tui_session(store, project, {}, resume=None, legacy_session_id=None)

    assert session.session_id != existing.session_id
    assert session.thread_id != existing.thread_id
    assert session.mode == CLI_AGENT_MODE


def test_tui_resume_without_id_loads_latest_project_session(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    store = SessionStore(tmp_path / "state")
    store.create(project, "code", {}, title="older")
    newer = store.create(project, "code", {}, title="newer")

    session = _resolve_tui_session(store, project, {}, resume=RESUME_LATEST, legacy_session_id=None)

    assert session.session_id == newer.session_id
    assert session.mode == CLI_AGENT_MODE
    assert store.load(newer.session_id).mode == CLI_AGENT_MODE


def test_tui_resume_specific_session_id(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    store = SessionStore(tmp_path / "state")
    existing = store.create(project, "code", {})

    session = _resolve_tui_session(store, project, {}, resume=existing.session_id, legacy_session_id=None)

    assert session.session_id == existing.session_id
    assert session.mode == CLI_AGENT_MODE
    assert store.load(existing.session_id).mode == CLI_AGENT_MODE


def test_tui_resume_specific_thread_id(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    store = SessionStore(tmp_path / "state")
    existing = store.create(project, "code", {})

    session = _resolve_tui_session(store, project, {}, resume=existing.thread_id, legacy_session_id=None)

    assert session.session_id == existing.session_id
    assert session.mode == CLI_AGENT_MODE


def test_tui_resume_missing_id_raises(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    store = SessionStore(tmp_path / "state")

    with pytest.raises(SessionStoreError):
        _resolve_tui_session(store, project, {}, resume="missing-thread", legacy_session_id=None)


def test_tui_resume_project_mismatch_raises(tmp_path: Path) -> None:
    project = tmp_path / "project"
    other_project = tmp_path / "other"
    project.mkdir()
    other_project.mkdir()
    store = SessionStore(tmp_path / "state")
    existing = store.create(other_project, "code", {})

    with pytest.raises(SessionStoreError, match="belongs to project"):
        _resolve_tui_session(store, project, {}, resume=existing.thread_id, legacy_session_id=None)


def test_tui_legacy_session_alias_loads_specific_session(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    store = SessionStore(tmp_path / "state")
    existing = store.create(project, "code", {})

    session = _resolve_tui_session(
        store,
        project,
        {},
        resume=None,
        legacy_session_id=existing.session_id,
    )

    assert session.session_id == existing.session_id
    assert session.mode == CLI_AGENT_MODE


def test_tui_permission_mode_new_session_uses_settings(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    store = SessionStore(tmp_path / "state")
    session = _resolve_tui_session(store, project, {}, resume=None, legacy_session_id=None)
    settings = CliSettings(permission_mode="auto")

    mode = _resolve_tui_permission_mode(session, settings, None, resumed=False)

    assert mode == "auto"


def test_tui_permission_mode_cli_overrides_settings(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    store = SessionStore(tmp_path / "state")
    session = _resolve_tui_session(store, project, {}, resume=None, legacy_session_id=None)
    settings = CliSettings(permission_mode="auto")

    mode = _resolve_tui_permission_mode(session, settings, "ask", resumed=False)

    assert mode == "ask"


def test_tui_permission_mode_resume_uses_session_value(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", {})
    session.permission_mode = "ask"
    settings = CliSettings(permission_mode="auto")

    mode = _resolve_tui_permission_mode(session, settings, None, resumed=True)

    assert mode == "ask"


def test_tui_permission_mode_cli_overrides_resumed_session(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", {})
    session.permission_mode = "auto"
    settings = CliSettings(permission_mode="auto")

    mode = _resolve_tui_permission_mode(session, settings, "ask", resumed=True)

    assert mode == "ask"


def _sub_agent_test_event():
    from kolega_code.events import AgentEvent

    return AgentEvent(
        event_type="chat_message",
        sender="general-agent",
        content={"status": "GENERATING", "message": "Starting general-agent task"},
        sub_agent_info={
            "agent_id": "agent-1",
            "agent_name": "general-agent",
            "task": "do sub-task",
            "parent_tool_call_id": "exec-1",
            "conversation_id": None,
            "depth": 1,
        },
    )


class _SubAgentEventCoderAgent(_FakeCoderAgent):
    """Fake CoderAgent that broadcasts a sub-agent event mid-stream."""

    agent_name = "coder"
    instances = []

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.__class__.instances.append(self)

    async def process_message_stream(self, message):
        yield {"type": "response", "content": "first ", "complete": False, "uuid": "response-1"}
        manager = self.kwargs["connection_manager"]
        await manager.broadcast_event(_sub_agent_test_event(), "ws", "thread")
        # Give the event pump a chance to run before the final chunk
        for _ in range(5):
            await asyncio.sleep(0)
        yield {"type": "response", "content": "second", "complete": True, "uuid": "response-1"}


def test_ask_json_interleaves_sub_agent_events(
    tmp_path: Path, capsys, monkeypatch: pytest.MonkeyPatch, isolated_cli_env: None
) -> None:
    from kolega_code.cli import main as main_module

    _SubAgentEventCoderAgent.instances = []
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("KOLEGA_CODE_PROVIDER", "anthropic")
    monkeypatch.setattr(main_module, "CoderAgent", _SubAgentEventCoderAgent)

    exit_code = main_module.main(["ask", "do the task", "--project", str(project), "--json"])

    assert exit_code == 0
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.strip()]
    kinds = [line["kind"] for line in lines]
    event_index = kinds.index("event")
    final_chunk_index = max(i for i, line in enumerate(lines) if line["kind"] == "chunk")
    assert event_index < final_chunk_index, "sub-agent event should interleave before the final chunk"
    event_line = lines[event_index]
    assert event_line["data"]["sub_agent_info"]["agent_name"] == "general-agent"


def test_ask_plain_writes_sub_agent_lifecycle_to_stderr(
    tmp_path: Path, capsys, monkeypatch: pytest.MonkeyPatch, isolated_cli_env: None
) -> None:
    from kolega_code.cli import main as main_module

    _SubAgentEventCoderAgent.instances = []
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("KOLEGA_CODE_PROVIDER", "anthropic")
    monkeypatch.setattr(main_module, "CoderAgent", _SubAgentEventCoderAgent)

    exit_code = main_module.main(["ask", "do the task", "--project", str(project)])

    assert exit_code == 0
    captured = capsys.readouterr()
    assert captured.out.strip() == "first second"
    from kolega_code.cli import theme

    sep = theme.g(theme.Glyph.BULLET_SEP)
    glyph = theme.g(theme.Glyph.SUB_AGENT)
    assert f"{glyph} general-agent {sep} generating {sep} Starting general-agent task" in captured.err


def test_ask_prompt_with_file_mention_attaches_content(
    tmp_path: Path, capsys, monkeypatch: pytest.MonkeyPatch, isolated_cli_env: None
) -> None:
    from kolega_code.cli import main as main_module

    class FakeCoderAgent(_FakeCoderAgent):
        agent_name = "coder"
        instances = []

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.__class__.instances.append(self)

        async def process_message_stream(self, message, attachments=None):
            self.messages.append(message)
            self.attachments.append(attachments)
            yield {"type": "response", "content": "ok", "complete": True, "uuid": "response-1"}

    project = tmp_path / "project"
    project.mkdir()
    (project / "notes.md").write_text("remember the milk\n", encoding="utf-8")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("KOLEGA_CODE_PROVIDER", "anthropic")
    monkeypatch.setattr(main_module, "CoderAgent", FakeCoderAgent)

    exit_code = main_module.main(["ask", "summarize @notes.md", "--project", str(project)])

    assert exit_code == 0
    assert "ok" in capsys.readouterr().out
    agent = FakeCoderAgent.instances[0]
    assert agent.messages == ["summarize @notes.md"]
    attachments = agent.attachments[0]
    assert attachments is not None and len(attachments) == 1
    assert attachments[0]["type"] == "file"
    assert attachments[0]["path"] == "notes.md"
    assert attachments[0]["content"] == "remember the milk\n"


def test_ask_prompt_with_unresolved_mention_warns_on_stderr(
    tmp_path: Path, capsys, monkeypatch: pytest.MonkeyPatch, isolated_cli_env: None
) -> None:
    from kolega_code.cli import main as main_module

    class FakeCoderAgent(_FakeCoderAgent):
        agent_name = "coder"
        instances = []

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.__class__.instances.append(self)

    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("KOLEGA_CODE_PROVIDER", "anthropic")
    monkeypatch.setattr(main_module, "CoderAgent", FakeCoderAgent)

    exit_code = main_module.main(["ask", "summarize @missing.md", "--project", str(project)])

    assert exit_code == 0
    captured = capsys.readouterr()
    assert "@missing.md not found" in captured.err
    assert FakeCoderAgent.instances[0].attachments == [None]
