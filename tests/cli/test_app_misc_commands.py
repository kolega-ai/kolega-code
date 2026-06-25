# ruff: noqa: F401,F811,E402
from pathlib import Path
import asyncio
import json
import time

import pytest

from kolega_code.cli.tui import agent_runtime as agent_runtime_module

from kolega_code.config import ModelProvider
from kolega_code.llm.exceptions import (
    LLMBillingError,
    LLMAuthenticationError,
    LLMContextWindowExceededError,
    LLMError,
    LLMInternalServerError,
)
from kolega_code.llm.models import Message, TextBlock, ToolCall, ToolResult
from kolega_code.events import AgentEvent
from kolega_code.agent.prompt_overrides import PROMPT_OVERRIDE_DIR
from kolega_code.agent.prompt_provider import AgentMode, PromptContext, PromptProvider
from kolega_code.cli.config import build_agent_config, config_summary
from kolega_code.cli.provider_registry import (
    DEEPSEEK_DEFAULT_MODEL,
    MOONSHOT_K26_MODEL,
    UI_DEFAULT_MODEL,
    UI_DEFAULT_PROVIDER,
)
from kolega_code.cli.session_store import SessionStore
from kolega_code.cli.settings import CliSettings, SettingsStore

from ._app_test_utils import (
    _build_mention_test_app,
    _build_sub_agent_test_app,
    _sub_agent_context_event,
    _sub_agent_entries,
    _sub_agent_event,
    _workflow_event,
    build_test_config,
    extension_by_name,
    first_text_styles,
    question_payload,
    renderable_text,
)


@pytest.mark.asyncio
async def test_textual_app_copy_and_version_slash_commands(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("textual")

    import kolega_code
    from kolega_code.cli.tui import command_handlers as command_handlers_module
    from kolega_code.cli.tui.state import ConversationEntry
    from kolega_code.cli.tui.widgets import ChatComposer
    from kolega_code.cli.updater import UpdateCheckResult

    app = _build_mention_test_app(tmp_path, monkeypatch)
    copied: list[str] = []
    monkeypatch.setattr(
        command_handlers_module,
        "check_for_update",
        lambda: UpdateCheckResult(current_version=kolega_code.__version__, latest_version=kolega_code.__version__),
    )

    async with app.run_test():
        monkeypatch.setattr(app, "copy_to_clipboard", copied.append)
        composer = app.query_one("#composer", ChatComposer)

        composer.load_text("/copy")
        await app.on_chat_composer_submitted(ChatComposer.Submitted(composer, composer.text))
        assert copied == []

        app._add_conversation_entry(ConversationEntry(kind="assistant", content="the answer"))
        composer.load_text("/copy")
        await app.on_chat_composer_submitted(ChatComposer.Submitted(composer, composer.text))
        assert copied == ["the answer"]

        composer.load_text("/version")
        await app.on_chat_composer_submitted(ChatComposer.Submitted(composer, composer.text))
        entry = app.conversation_entries[-1]
        assert entry.kind == "system"
        assert kolega_code.__version__ in entry.content
        assert "up to date" in entry.content


@pytest.mark.asyncio
async def test_textual_app_update_slash_command_runs_self_update(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli.tui import command_handlers as command_handlers_module
    from kolega_code.cli.tui.widgets import ChatComposer
    from kolega_code.cli.updater import UpdateRunResult

    app = _build_mention_test_app(tmp_path, monkeypatch)
    monkeypatch.setattr(
        command_handlers_module,
        "run_self_update",
        lambda *, capture_output=False: UpdateRunResult(returncode=0, stdout="installed\n"),
    )

    async with app.run_test():
        composer = app.query_one("#composer", ChatComposer)
        composer.load_text("/update")
        await app.on_chat_composer_submitted(ChatComposer.Submitted(composer, composer.text))

        entry = app.conversation_entries[-1]
        assert entry.kind == "system"
        assert "Kolega Code update completed" in entry.content
        assert "installed" in entry.content


@pytest.mark.asyncio
async def test_textual_app_startup_update_check_notifies_when_newer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli import app as app_module
    from kolega_code.cli.updater import UpdateCheckResult

    app = _build_mention_test_app(tmp_path, monkeypatch)
    app.check_for_updates = True
    monkeypatch.setattr(
        app_module,
        "check_for_update",
        lambda: UpdateCheckResult(current_version="0.2.0", latest_version="0.3.0", update_available=True),
    )

    async with app.run_test():
        for _ in range(20):
            if any("Update available: 0.2.0 -> 0.3.0" in entry.content for entry in app.conversation_entries):
                break
            await asyncio.sleep(0.05)

        assert any("Update available: 0.2.0 -> 0.3.0" in entry.content for entry in app.conversation_entries)


@pytest.mark.asyncio
@pytest.mark.parametrize("command", ["/quit", "/exit"])
async def test_textual_app_quit_slash_command_exits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, command: str
) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli.tui.widgets import ChatComposer

    app = _build_mention_test_app(tmp_path, monkeypatch)

    async with app.run_test():
        composer = app.query_one("#composer", ChatComposer)
        composer.load_text(command)
        await app.on_chat_composer_submitted(ChatComposer.Submitted(composer, composer.text))

    assert app.return_value is None
    assert not app.is_running


@pytest.mark.asyncio
async def test_textual_app_unknown_slash_command_falls_through_to_agent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli.tui.widgets import ChatComposer

    app = _build_mention_test_app(tmp_path, monkeypatch)

    async with app.run_test() as pilot:
        composer = app.query_one("#composer", ChatComposer)
        composer.load_text("/help")
        await app.on_chat_composer_submitted(ChatComposer.Submitted(composer, composer.text))
        await pilot.pause()

        assert app.agent.messages == ["/help"]


@pytest.mark.asyncio
async def test_textual_app_prompt_list_recovers_focus_after_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A shown prompt list must regain keyboard focus if focus drifts away.

    Regression for: after a prompt appears, a background click or a resize could
    leave the option list without focus and the user had no keyboard way back
    (arrow keys / Enter dead). The composer is disabled during an approval, so the
    list is always the only valid focus target here.
    """
    pytest.importorskip("textual")

    from kolega_code.cli.app import KolegaCodeApp
    from kolega_code.cli.tui.state import PendingApproval
    from kolega_code.cli.tui.widgets import ActionList
    from kolega_code.permissions import PermissionDecision, allow_rule_options, permission_request_for_tool

    class FakeCoderAgent:
        def __init__(self, **kwargs):
            pass

        def restore_message_history(self, history):
            return None

        def dump_compaction_state(self):
            return {}

        def restore_compaction_state(self, data):
            pass

        def dump_message_history(self):
            return []

        async def cleanup(self):
            return None

    monkeypatch.setattr(agent_runtime_module, "CoderAgent", FakeCoderAgent)

    project = tmp_path / "project"
    project.mkdir()
    config = build_test_config(project)
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", config_summary(config))
    app = KolegaCodeApp(project_path=project, config=config, mode="code", store=store, session=session)

    async with app.run_test() as pilot:
        request = permission_request_for_tool("exec_command", {"command": "npm run test"})
        assert request is not None
        future: asyncio.Future[PermissionDecision] = asyncio.get_running_loop().create_future()
        app._pending_approval = PendingApproval(
            request=request, future=future, rule_options=allow_rule_options(request)
        )
        app._set_approval_actions_visible(True)
        app._set_chat_enabled(False)

        approval_actions = app.query_one("#approval_actions", ActionList)
        assert app.focused is approval_actions

        # Focus drifts to the conversation transcript (the AUTO_FOCUS magnet that
        # would otherwise win on resize/resume). The focus hook pulls it back.
        app.screen.set_focus(app.query_one("#conversation"))
        await pilot.pause()
        assert app.focused is approval_actions

        # A background (NoWidget) click does set_focus(None); the blur hook restores
        # via call_after_refresh. On slower CI runs the deferred focus callback can
        # land one or two refreshes later, so wait for the observable state instead
        # of assuming a single pause is enough.
        app.screen.set_focus(None)
        for _ in range(5):
            await pilot.pause()
            if app.focused is approval_actions:
                break
        assert app.focused is approval_actions

        app._pending_approval = None
        app._set_approval_actions_visible(False)


@pytest.mark.asyncio
async def test_textual_app_question_recovers_focus_but_allows_free_form_answer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """During a question the option list self-heals, but a deliberate move to the
    enabled composer (to type a free-form answer) must NOT be fought."""
    pytest.importorskip("textual")

    from kolega_code.cli.app import KolegaCodeApp
    from kolega_code.cli.tui.state import PendingQuestion
    from kolega_code.cli.tui.widgets import ActionList, ChatComposer

    class FakeCoderAgent:
        def __init__(self, **kwargs):
            pass

        def restore_message_history(self, history):
            return None

        def dump_compaction_state(self):
            return {}

        def restore_compaction_state(self, data):
            pass

        def dump_message_history(self):
            return []

        async def cleanup(self):
            return None

    monkeypatch.setattr(agent_runtime_module, "CoderAgent", FakeCoderAgent)

    project = tmp_path / "project"
    project.mkdir()
    config = build_test_config(project)
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", config_summary(config))
    app = KolegaCodeApp(project_path=project, config=config, mode="code", store=store, session=session)

    async with app.run_test() as pilot:
        future: asyncio.Future[str] = asyncio.get_running_loop().create_future()
        app._pending_question = PendingQuestion(question="Choose?", options=["A", "B"], future=future)
        app._show_question_options("Choose?", ["A", "B"])
        app._set_chat_enabled(True)

        question_actions = app.query_one("#question_actions", ActionList)
        assert app.focused is question_actions

        # Drift to the transcript is pulled back to the option list.
        app.screen.set_focus(app.query_one("#conversation"))
        await pilot.pause()
        assert app.focused is question_actions

        # A deliberate move to the ENABLED composer is preserved (free-form answer).
        composer = app.query_one("#composer", ChatComposer)
        assert composer.disabled is False
        app.screen.set_focus(composer)
        await pilot.pause()
        assert app.focused is composer

        app._pending_question = None
        app._set_question_actions_visible(False)


class PromptCommandFakeAgent:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.prompt_provider = PromptProvider()
        self.agent_mode = AgentMode.CLI
        self.project_template_slug = None
        self.refresh_count = 0

    def restore_message_history(self, history):
        return None

    def dump_compaction_state(self):
        return {}

    def restore_compaction_state(self, data):
        pass

    def dump_message_history(self):
        return []

    def build_prompt_context(self):
        return PromptContext(
            project_path=str(self.kwargs["project_path"]),
            platform="TestOS",
            date_today="2026-06-25",
            model_name="test-model",
        )

    def refresh_system_prompt(self):
        self.refresh_count += 1

    async def cleanup(self):
        return None


@pytest.mark.asyncio
async def test_textual_app_prompts_validate_posts_validation_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli.app import KolegaCodeApp
    from kolega_code.cli.tui.widgets import ChatComposer

    monkeypatch.setattr(agent_runtime_module, "CoderAgent", PromptCommandFakeAgent)
    project = tmp_path / "project"
    project.mkdir()
    prompt_dir = project / PROMPT_OVERRIDE_DIR
    prompt_dir.mkdir(parents=True)
    (prompt_dir / "GENERAL.md").write_text("{{ missing_variable }}", encoding="utf-8")
    config = build_test_config(project)
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", config_summary(config))
    app = KolegaCodeApp(project_path=project, config=config, mode="code", store=store, session=session)

    async with app.run_test():
        composer = app.query_one("#composer", ChatComposer)
        composer.load_text("/prompts validate")
        await app.on_chat_composer_submitted(ChatComposer.Submitted(composer, composer.text))

        entry = app.conversation_entries[-1]
        assert entry.kind == "system"
        assert "Prompt override validation" in entry.content
        assert "Could not render prompt override .kolega/prompts/GENERAL.md" in entry.content


@pytest.mark.asyncio
async def test_textual_app_prompts_dump_accepts_selected_prompts_and_force(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli.app import KolegaCodeApp
    from kolega_code.cli.tui.widgets import ChatComposer

    monkeypatch.setattr(agent_runtime_module, "CoderAgent", PromptCommandFakeAgent)
    project = tmp_path / "project"
    project.mkdir()
    prompt_dir = project / PROMPT_OVERRIDE_DIR
    prompt_dir.mkdir(parents=True)
    coder = prompt_dir / "CODER.md"
    coder.write_text("custom coder", encoding="utf-8")
    config = build_test_config(project)
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", config_summary(config))
    app = KolegaCodeApp(project_path=project, config=config, mode="code", store=store, session=session)

    async with app.run_test():
        composer = app.query_one("#composer", ChatComposer)
        composer.load_text("/prompts dump coder --force")
        await app.on_chat_composer_submitted(ChatComposer.Submitted(composer, composer.text))

        entry = app.conversation_entries[-1]
        assert entry.kind == "system"
        assert "Written:" in entry.content
        assert "powerful AI coding assistant" in coder.read_text(encoding="utf-8")
        assert not (prompt_dir / "PLANNING.md").exists()
        assert app.agent.refresh_count == 1
