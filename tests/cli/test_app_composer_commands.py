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
from kolega_code.agent.prompt_provider import AgentMode
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
async def test_textual_app_composer_shift_enter_inserts_line_break_and_enter_submits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli.app import KolegaCodeApp
    from kolega_code.cli.tui.widgets import ChatComposer

    class FakeCoderAgent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.messages: list[str] = []

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

        async def process_message_stream(self, message):
            self.messages.append(message)
            yield {"type": "response", "content": "ok", "complete": True, "uuid": "response-1"}

    monkeypatch.setattr(agent_runtime_module, "CoderAgent", FakeCoderAgent)

    project = tmp_path / "project"
    project.mkdir()
    config = build_test_config(project)
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", config_summary(config))
    app = KolegaCodeApp(project_path=project, config=config, mode="code", store=store, session=session)

    async with app.run_test() as pilot:
        composer = app.query_one("#composer", ChatComposer)
        composer.focus()

        await pilot.press("h", "i")
        await pilot.press("shift+enter")
        await pilot.press("t", "h", "e", "r", "e")
        assert composer.text == "hi\nthere"

        await pilot.press("enter")
        await pilot.pause()

        assert app.agent is not None
        assert app.agent.messages == ["hi\nthere"]
        assert composer.text == ""
        user_entries = [entry for entry in app.conversation_entries if entry.kind == "user"]
        assert user_entries[-1].content == "hi\nthere"


@pytest.mark.asyncio
async def test_textual_app_composer_ctrl_enter_still_inserts_line_break(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli.app import KolegaCodeApp
    from kolega_code.cli.tui.widgets import ChatComposer

    class FakeCoderAgent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

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
        composer = app.query_one("#composer", ChatComposer)
        composer.focus()

        await pilot.press("h", "i")
        await pilot.press("ctrl+enter")
        await pilot.press("t", "h", "e", "r", "e")

        assert composer.text == "hi\nthere"


@pytest.mark.asyncio
async def test_textual_app_composer_preserves_multiline_paste(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("textual")

    from textual import events

    from kolega_code.cli.app import KolegaCodeApp
    from kolega_code.cli.tui.widgets import ChatComposer

    class FakeCoderAgent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.messages: list[str] = []

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

        async def process_message_stream(self, message):
            self.messages.append(message)
            yield {"type": "response", "content": "ok", "complete": True, "uuid": "response-1"}

    monkeypatch.setattr(agent_runtime_module, "CoderAgent", FakeCoderAgent)

    pasted = "line one\n    line two\nline three"
    project = tmp_path / "project"
    project.mkdir()
    config = build_test_config(project)
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", config_summary(config))
    app = KolegaCodeApp(project_path=project, config=config, mode="code", store=store, session=session)

    async with app.run_test():
        composer = app.query_one("#composer", ChatComposer)
        composer.focus()

        await composer._on_paste(events.Paste(pasted))
        assert composer.text == pasted

        await app.on_chat_composer_submitted(ChatComposer.Submitted(composer, composer.text))
        worker = app.agent_worker
        assert worker is not None
        await worker.wait()

        assert app.agent is not None
        assert app.agent.messages == [pasted]
        assert composer.text == ""


@pytest.mark.asyncio
async def test_textual_app_skill_slash_commands_list_and_activate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli.app import KolegaCodeApp
    from kolega_code.cli.tui.widgets import ChatComposer

    class FakeCoderAgent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.history = []

        def append_user_message(self, content):
            self.history.append(Message(role="user", content=content))

        def restore_message_history(self, history):
            self.history = [Message.from_dict(item) for item in history]

        def dump_compaction_state(self):
            return {}

        def restore_compaction_state(self, data):
            pass

        def dump_message_history(self):
            return [message.to_dict() for message in self.history]

        async def cleanup(self):
            return None

    monkeypatch.setattr(agent_runtime_module, "CoderAgent", FakeCoderAgent)

    project = tmp_path / "project"
    skill_dir = project / ".agents" / "skills" / "demo-skill"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: demo-skill\ndescription: Use this demo skill.\n---\n\nFollow demo instructions.\n",
        encoding="utf-8",
    )
    config = build_test_config(project)
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", config_summary(config))
    app = KolegaCodeApp(project_path=project, config=config, mode="code", store=store, session=session)

    async with app.run_test():
        composer = app.query_one("#composer", ChatComposer)
        composer.load_text("/skills")
        await app.on_chat_composer_submitted(ChatComposer.Submitted(composer, composer.text))

        assert app.conversation_entries[-1].kind == "system"
        assert "`/demo-skill`" in app.conversation_entries[-1].content

        composer.load_text("/demo-skill")
        await app.on_chat_composer_submitted(ChatComposer.Submitted(composer, composer.text))

        assert app.conversation_entries[-1].kind == "skill"
        assert '<skill_content name="demo-skill">' in app.agent.history[-1].get_text_content()
        assert '<skill_content name="demo-skill">' in store.load(session.session_id).history[-1]["content"][0]["text"]


@pytest.mark.asyncio
async def test_textual_app_skill_slash_command_with_prompt_starts_turn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli.app import KolegaCodeApp
    from kolega_code.cli.tui.widgets import ChatComposer

    class FakeCoderAgent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.history = []
            self.messages = []

        def append_user_message(self, content):
            self.history.append(Message(role="user", content=content))

        def restore_message_history(self, history):
            self.history = [Message.from_dict(item) for item in history]

        def dump_compaction_state(self):
            return {}

        def restore_compaction_state(self, data):
            pass

        def dump_message_history(self):
            return [message.to_dict() for message in self.history]

        async def cleanup(self):
            return None

        async def process_message_stream(self, message):
            self.messages.append(message)
            yield {"type": "response", "content": "done", "complete": True, "uuid": "response-1"}

    monkeypatch.setattr(agent_runtime_module, "CoderAgent", FakeCoderAgent)

    project = tmp_path / "project"
    skill_dir = project / ".agents" / "skills" / "demo-skill"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: demo-skill\ndescription: Use this demo skill.\n---\n\nFollow demo instructions.\n",
        encoding="utf-8",
    )
    config = build_test_config(project)
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", config_summary(config))
    app = KolegaCodeApp(project_path=project, config=config, mode="code", store=store, session=session)

    async with app.run_test() as pilot:
        composer = app.query_one("#composer", ChatComposer)
        composer.load_text("/demo-skill Build the feature")
        await app.on_chat_composer_submitted(ChatComposer.Submitted(composer, composer.text))
        await pilot.pause()

        assert app.agent.messages == ["Build the feature"]
        assert any(entry.kind == "skill" for entry in app.conversation_entries)
        assert any(entry.kind == "user" and entry.content == "Build the feature" for entry in app.conversation_entries)


@pytest.mark.asyncio
async def test_textual_app_mention_dropdown_opens_and_escape_dismisses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli.tui.widgets import ChatComposer, CompletionDropdown

    app = _build_mention_test_app(tmp_path, monkeypatch)

    async with app.run_test() as pilot:
        composer = app.query_one("#composer", ChatComposer)
        dropdown = app.query_one("#completion_dropdown", CompletionDropdown)
        composer.focus()
        await pilot.pause()

        composer.insert("@alp")
        await pilot.pause()
        assert dropdown.is_open
        assert dropdown.option_count > 0

        await pilot.press("escape")
        assert not dropdown.is_open
        assert composer.text == "@alp"


@pytest.mark.asyncio
async def test_textual_app_mention_dropdown_not_opened_by_email_address(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli.tui.widgets import ChatComposer, CompletionDropdown

    app = _build_mention_test_app(tmp_path, monkeypatch)

    async with app.run_test() as pilot:
        composer = app.query_one("#composer", ChatComposer)
        dropdown = app.query_one("#completion_dropdown", CompletionDropdown)
        composer.focus()
        composer.insert("mail user@example")
        await pilot.pause()
        assert not dropdown.is_open


@pytest.mark.asyncio
async def test_textual_app_mention_dropdown_down_and_tab_completes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli.tui.widgets import ChatComposer, CompletionDropdown

    app = _build_mention_test_app(tmp_path, monkeypatch)

    async with app.run_test() as pilot:
        composer = app.query_one("#composer", ChatComposer)
        dropdown = app.query_one("#completion_dropdown", CompletionDropdown)
        composer.focus()
        composer.insert("@alp")
        await pilot.pause()
        assert dropdown.is_open

        expected = dropdown.entry_at(1).path
        await pilot.press("down")
        assert dropdown.highlighted == 1
        await pilot.press("tab")
        assert composer.text == f"@{expected} "
        assert not dropdown.is_open


@pytest.mark.asyncio
async def test_textual_app_mention_enter_completes_instead_of_submitting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli.tui.widgets import ChatComposer, CompletionDropdown

    app = _build_mention_test_app(tmp_path, monkeypatch)

    async with app.run_test() as pilot:
        composer = app.query_one("#composer", ChatComposer)
        dropdown = app.query_one("#completion_dropdown", CompletionDropdown)
        composer.focus()
        composer.insert("@README")
        await pilot.pause()
        assert dropdown.is_open

        await pilot.press("enter")
        await pilot.pause()
        assert composer.text == "@README.md "
        assert not dropdown.is_open
        # No message was submitted, only the completion was applied.
        assert app.agent.messages == []


@pytest.mark.asyncio
async def test_textual_app_submitting_mention_attaches_file_and_keeps_short_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli.tui.widgets import ChatComposer

    app = _build_mention_test_app(tmp_path, monkeypatch)

    async with app.run_test() as pilot:
        composer = app.query_one("#composer", ChatComposer)
        composer.load_text("summarize @src/alpha.py please")
        await app.on_chat_composer_submitted(ChatComposer.Submitted(composer, composer.text))
        await pilot.pause()

        assert app.agent.messages == ["summarize @src/alpha.py please"]
        attachments = app.agent.attachments[0]
        assert attachments is not None and len(attachments) == 1
        assert attachments[0]["type"] == "file"
        assert attachments[0]["path"] == "src/alpha.py"
        assert attachments[0]["content"] == "print('alpha')\n"
        assert any(
            entry.kind == "user" and entry.content == "summarize @src/alpha.py please"
            for entry in app.conversation_entries
        )


@pytest.mark.asyncio
async def test_textual_app_unresolved_mention_clears_hint_and_sends_plain_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from textual.widgets import Static

    from kolega_code.cli.tui.widgets import ChatComposer

    app = _build_mention_test_app(tmp_path, monkeypatch)

    async with app.run_test() as pilot:
        composer = app.query_one("#composer", ChatComposer)
        composer.load_text("look at @does/not/exist.py")
        await app.on_chat_composer_submitted(ChatComposer.Submitted(composer, composer.text))

        # Unresolved mentions are sent as plain text; the compose-time hint must
        # not linger after the message has been submitted.
        hint = app.query_one("#composer_hint", Static)
        assert str(hint.render()) == ""

        await pilot.pause()
        assert app.agent.messages == ["look at @does/not/exist.py"]
        assert app.agent.attachments == [None]


@pytest.mark.asyncio
async def test_textual_app_slash_dropdown_opens_filters_and_tab_completes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli.tui.widgets import ChatComposer, CompletionDropdown
    from kolega_code.cli.slash_commands import SlashCommandEntry

    app = _build_mention_test_app(tmp_path, monkeypatch)

    async with app.run_test() as pilot:
        composer = app.query_one("#composer", ChatComposer)
        dropdown = app.query_one("#completion_dropdown", CompletionDropdown)
        composer.focus()
        await pilot.pause()

        composer.insert("/")
        await pilot.pause()
        assert dropdown.is_open
        assert dropdown.option_count > 1
        assert isinstance(dropdown.highlighted_entry(), SlashCommandEntry)

        composer.insert("pl")
        await pilot.pause()
        assert dropdown.is_open
        assert dropdown.highlighted_entry().name == "plan"

        await pilot.press("tab")
        assert composer.text == "/plan "
        assert not dropdown.is_open


@pytest.mark.asyncio
async def test_textual_app_slash_dropdown_lists_skills_with_descriptions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli.tui.widgets import ChatComposer, CompletionDropdown

    app = _build_mention_test_app(tmp_path, monkeypatch)
    skill_dir = app.project_path / ".agents" / "skills" / "demo-skill"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: demo-skill\ndescription: Use this demo skill.\n---\n\nFollow demo instructions.\n",
        encoding="utf-8",
    )

    async with app.run_test() as pilot:
        composer = app.query_one("#composer", ChatComposer)
        dropdown = app.query_one("#completion_dropdown", CompletionDropdown)
        composer.focus()
        composer.insert("/demo")
        await pilot.pause()

        assert dropdown.is_open
        entry = dropdown.highlighted_entry()
        assert entry.name == "demo-skill"
        assert entry.description == "Use this demo skill."

        await pilot.press("tab")
        assert composer.text == "/demo-skill "


@pytest.mark.asyncio
async def test_textual_app_slash_dropdown_enter_completes_instead_of_submitting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli.tui.widgets import ChatComposer, CompletionDropdown

    app = _build_mention_test_app(tmp_path, monkeypatch)

    async with app.run_test() as pilot:
        composer = app.query_one("#composer", ChatComposer)
        dropdown = app.query_one("#completion_dropdown", CompletionDropdown)
        composer.focus()
        composer.insert("/versio")
        await pilot.pause()
        assert dropdown.is_open

        await pilot.press("enter")
        await pilot.pause()
        assert composer.text == "/version "
        assert not dropdown.is_open
        assert app.agent.messages == []


@pytest.mark.asyncio
async def test_textual_app_slash_dropdown_does_not_open_mid_text_or_after_args(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli.tui.widgets import ChatComposer, CompletionDropdown

    app = _build_mention_test_app(tmp_path, monkeypatch)

    async with app.run_test() as pilot:
        composer = app.query_one("#composer", ChatComposer)
        dropdown = app.query_one("#completion_dropdown", CompletionDropdown)
        composer.focus()

        composer.insert("see src/")
        await pilot.pause()
        assert not dropdown.is_open

        composer.load_text("")
        composer.insert("first line")
        composer.action_insert_newline()
        composer.insert("/")
        await pilot.pause()
        assert not dropdown.is_open

        composer.load_text("")
        composer.insert("/skills extra")
        await pilot.pause()
        assert not dropdown.is_open


@pytest.mark.asyncio
async def test_chat_composer_active_slash_query(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli.tui.widgets import ChatComposer

    app = _build_mention_test_app(tmp_path, monkeypatch)

    async with app.run_test():
        composer = app.query_one("#composer", ChatComposer)
        composer.focus()

        composer.insert("/")
        assert composer.active_slash_query() == ("", 0, 1)

        composer.insert("mod")
        assert composer.active_slash_query() == ("mod", 0, 4)

        composer.load_text("")
        composer.insert("  /he")
        assert composer.active_slash_query() == ("he", 2, 5)

        composer.load_text("")
        composer.insert("hello /he")
        assert composer.active_slash_query() is None

        composer.load_text("")
        composer.insert("/model kimi")
        assert composer.active_slash_query() is None


@pytest.mark.asyncio
async def test_textual_app_plan_and_build_slash_commands_switch_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli.app import KolegaCodeApp
    from kolega_code.cli.tui.widgets import ChatComposer

    class FakeAgent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

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

    class FakeCoderAgent(FakeAgent):
        pass

    class FakePlanningAgent(FakeAgent):
        pass

    monkeypatch.setattr(agent_runtime_module, "CoderAgent", FakeCoderAgent)
    monkeypatch.setattr(agent_runtime_module, "PlanningAgent", FakePlanningAgent)

    project = tmp_path / "project"
    project.mkdir()
    config = build_test_config(project)
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", config_summary(config))
    app = KolegaCodeApp(project_path=project, config=config, mode="code", store=store, session=session)

    async with app.run_test():
        composer = app.query_one("#composer", ChatComposer)
        assert app.interaction_mode == "build"

        composer.load_text("/plan")
        await app.on_chat_composer_submitted(ChatComposer.Submitted(composer, composer.text))
        assert app.interaction_mode == "plan"
        assert isinstance(app.agent, FakePlanningAgent)
        assert composer.text == ""

        composer.load_text("/build")
        await app.on_chat_composer_submitted(ChatComposer.Submitted(composer, composer.text))
        assert app.interaction_mode == "build"
        assert isinstance(app.agent, FakeCoderAgent)


@pytest.mark.asyncio
async def test_textual_app_sidebar_slash_command_toggles_sidebar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli.tui.widgets import ChatComposer

    app = _build_mention_test_app(tmp_path, monkeypatch)

    async with app.run_test():
        composer = app.query_one("#composer", ChatComposer)
        side_panel = app.query_one("#side_panel")

        composer.load_text("/sidebar")
        await app.on_chat_composer_submitted(ChatComposer.Submitted(composer, composer.text))

        assert composer.text == ""
        assert app.sidebar_visible is False
        assert side_panel.display is False

        composer.load_text("/sidebar")
        await app.on_chat_composer_submitted(ChatComposer.Submitted(composer, composer.text))

        assert composer.text == ""
        assert app.sidebar_visible is True
        assert side_panel.display is True


@pytest.mark.asyncio
async def test_textual_app_init_slash_command_starts_agents_md_turn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli.tui.widgets import ChatComposer

    app = _build_mention_test_app(tmp_path, monkeypatch)

    async with app.run_test() as pilot:
        composer = app.query_one("#composer", ChatComposer)
        composer.load_text("/init focus on test commands")
        await app.on_chat_composer_submitted(ChatComposer.Submitted(composer, composer.text))
        await pilot.pause()

        assert composer.text == ""
        assert len(app.agent.messages) == 1
        prompt = app.agent.messages[0]
        assert "Create or update `AGENTS.md` for this repository." in prompt
        assert "`focus on test commands`" in prompt
        assert "$ARGUMENTS" not in prompt
        assert any(
            entry.kind == "user" and entry.content == "/init focus on test commands"
            for entry in app.conversation_entries
        )


@pytest.mark.asyncio
async def test_textual_app_init_slash_command_switches_from_plan_to_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli.app import KolegaCodeApp
    from kolega_code.cli.tui.widgets import ChatComposer

    class FakeBaseAgent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.messages = []
            self.history = []

        def restore_message_history(self, history):
            self.history = list(history)

        def dump_compaction_state(self):
            return {}

        def restore_compaction_state(self, data):
            pass

        def dump_message_history(self):
            return self.history

        async def cleanup(self):
            return None

        async def process_message_stream(self, message):
            self.messages.append(message)
            yield {"type": "response", "content": "done", "complete": True, "uuid": "response-1"}

    class FakeCoderAgent(FakeBaseAgent):
        pass

    class FakePlanningAgent(FakeBaseAgent):
        pass

    monkeypatch.setattr(agent_runtime_module, "CoderAgent", FakeCoderAgent)
    monkeypatch.setattr(agent_runtime_module, "PlanningAgent", FakePlanningAgent)

    project = tmp_path / "project"
    project.mkdir()
    config = build_test_config(project)
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", config_summary(config))
    app = KolegaCodeApp(project_path=project, config=config, mode="code", store=store, session=session)

    async with app.run_test() as pilot:
        composer = app.query_one("#composer", ChatComposer)
        composer.load_text("/plan")
        await app.on_chat_composer_submitted(ChatComposer.Submitted(composer, composer.text))
        assert app.interaction_mode == "plan"
        assert isinstance(app.agent, FakePlanningAgent)

        composer.load_text("/init focus on docs")
        await app.on_chat_composer_submitted(ChatComposer.Submitted(composer, composer.text))
        await pilot.pause()

        assert app.interaction_mode == "build"
        assert isinstance(app.agent, FakeCoderAgent)
        assert app.agent.messages
        assert "`focus on docs`" in app.agent.messages[0]


@pytest.mark.asyncio
async def test_textual_app_init_slash_command_blocks_during_active_turn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from textual.widgets import Static

    from kolega_code.cli.tui.widgets import ChatComposer

    app = _build_mention_test_app(tmp_path, monkeypatch)

    async with app.run_test():
        composer = app.query_one("#composer", ChatComposer)
        app._turn_active = True
        composer.load_text("/init")
        await app.on_chat_composer_submitted(ChatComposer.Submitted(composer, composer.text))

        assert app.agent.messages == []
        assert "Stop the current turn before running /init." in str(app.query_one("#composer_hint", Static).render())
        app._turn_active = False


@pytest.mark.asyncio
async def test_textual_app_model_slash_command_shows_and_switches_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, isolated_cli_env: None
) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli.app import KolegaCodeApp
    from kolega_code.cli.tui import command_handlers as command_handlers_module
    from kolega_code.cli.tui import settings_panel as settings_panel_module
    from kolega_code.cli.tui.widgets import ChatComposer

    class FakeCoderAgent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

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

    def fake_model_options(provider):
        return [
            ("Kimi K2.7 Code", UI_DEFAULT_MODEL),
            ("Kimi K2.6", "kimi-k2.6"),
            ("Kimi K3", "kimi-k3"),
        ]

    def fake_effort_options(provider, model):
        return [("High", "high")] if model == "kimi-k3" else [("Auto", "auto")]

    def fake_default_effort(provider, model):
        return "high" if model == "kimi-k3" else "auto"

    for module in (settings_panel_module, command_handlers_module):
        monkeypatch.setattr(module, "ui_model_options", fake_model_options)
        monkeypatch.setattr(module, "ui_thinking_effort_options", fake_effort_options)
        monkeypatch.setattr(module, "default_ui_thinking_effort", fake_default_effort)

    project = tmp_path / "project"
    project.mkdir()
    state_dir = tmp_path / "state"
    store = SessionStore(state_dir)
    settings_store = SettingsStore(state_dir)
    session = store.create(project, "code", {})
    app = KolegaCodeApp(
        project_path=project,
        mode="code",
        store=store,
        settings_store=settings_store,
        session=session,
    )

    async with app.run_test():
        from textual.widgets import Input

        from kolega_code.cli.tui.widgets import ActionList

        app.query_one("#api_key_input", Input).value = "moonshot-key"
        await app._save_settings_from_ui()
        assert isinstance(app.agent, FakeCoderAgent)

        composer = app.query_one("#composer", ChatComposer)
        composer.load_text("/effort")
        await app.on_chat_composer_submitted(ChatComposer.Submitted(composer, composer.text))
        effort_entry = app.conversation_entries[-1]
        assert effort_entry.kind == "system"
        assert "Available thinking efforts:" in effort_entry.content
        assert "`auto`" in effort_entry.content
        assert "`none`" not in effort_entry.content
        effort_actions = app.query_one("#effort_actions", ActionList)
        assert effort_actions.display is True
        assert app.focused is effort_actions
        assert effort_actions.get_option("effort_option_0").prompt.startswith("1. Auto (auto)")

        composer.load_text("/effort auto")
        await app.on_chat_composer_submitted(ChatComposer.Submitted(composer, composer.text))
        assert settings_store.load().active_thinking_effort == "auto"
        assert effort_actions.display is False
        first_agent = app.agent
        assert isinstance(first_agent, FakeCoderAgent)

        composer.load_text("/model")
        await app.on_chat_composer_submitted(ChatComposer.Submitted(composer, composer.text))
        entry = app.conversation_entries[-1]
        assert entry.kind == "system"
        assert UI_DEFAULT_MODEL in entry.content and "kimi-k2.6" in entry.content and "kimi-k3" in entry.content
        assert "Thinking effort: auto" in entry.content
        model_actions = app.query_one("#model_actions", ActionList)
        assert model_actions.display is True
        assert app.focused is model_actions
        assert model_actions.option_count == 3
        assert model_actions.get_option("model_option_0").prompt.startswith(f"1. Kimi K2.7 Code ({UI_DEFAULT_MODEL})")

        # kimi-k3 is a fake model the real config builder rejects, so stub it for the rebuild step.
        saved_config = app.config
        monkeypatch.setattr(agent_runtime_module, "build_agent_config", lambda *args, **kwargs: saved_config)

        composer.load_text("/model kimi-k3")
        await app.on_chat_composer_submitted(ChatComposer.Submitted(composer, composer.text))
        switched_settings = settings_store.load()
        assert switched_settings.active_model == "kimi-k3"
        assert switched_settings.active_thinking_effort == "high"
        assert model_actions.display is False
        assert isinstance(app.agent, FakeCoderAgent)
        assert app.agent is not first_agent

        composer.load_text("/model does-not-exist")
        await app.on_chat_composer_submitted(ChatComposer.Submitted(composer, composer.text))
        assert settings_store.load().active_model == "kimi-k3"


@pytest.mark.asyncio
async def test_textual_app_model_slash_command_selects_from_action_list(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, isolated_cli_env: None
) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli.app import KolegaCodeApp
    from kolega_code.cli.tui.widgets import ActionList, ChatComposer

    class FakeCoderAgent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

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
    state_dir = tmp_path / "state"
    store = SessionStore(state_dir)
    settings_store = SettingsStore(state_dir)
    settings = CliSettings(
        active_provider=ModelProvider.MOONSHOT.value,
        active_model=UI_DEFAULT_MODEL,
        active_thinking_effort="auto",
    )
    settings.set_api_key(ModelProvider.MOONSHOT.value, "moonshot-key")
    settings_store.save(settings)
    session = store.create(project, "code", {})
    app = KolegaCodeApp(
        project_path=project,
        mode="code",
        store=store,
        settings_store=settings_store,
        session=session,
    )

    async with app.run_test() as pilot:
        composer = app.query_one("#composer", ChatComposer)
        composer.load_text("/model")
        await app.on_chat_composer_submitted(ChatComposer.Submitted(composer, composer.text))

        model_actions = app.query_one("#model_actions", ActionList)
        assert model_actions.display is True
        assert app.focused is model_actions
        assert model_actions.option_count == 3

        await pilot.press("down", "enter")
        await pilot.pause()
        assert settings_store.load().active_model == MOONSHOT_K26_MODEL
        assert model_actions.display is False

        composer.load_text("/model")
        await app.on_chat_composer_submitted(ChatComposer.Submitted(composer, composer.text))
        assert app.focused is model_actions

        await pilot.press("1")
        await pilot.pause()
        assert settings_store.load().active_model == UI_DEFAULT_MODEL
        assert model_actions.display is False


@pytest.mark.asyncio
async def test_textual_app_model_slash_command_accepts_typed_selection_and_rejects_invalid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, isolated_cli_env: None
) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli.app import KolegaCodeApp
    from kolega_code.cli.tui.widgets import ActionList, ChatComposer

    class FakeCoderAgent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.messages = []

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

        async def process_message_stream(self, message):
            self.messages.append(message)
            yield {"type": "response", "content": "unexpected"}

    monkeypatch.setattr(agent_runtime_module, "CoderAgent", FakeCoderAgent)

    project = tmp_path / "project"
    project.mkdir()
    state_dir = tmp_path / "state"
    store = SessionStore(state_dir)
    settings_store = SettingsStore(state_dir)
    settings = CliSettings(
        active_provider=ModelProvider.MOONSHOT.value,
        active_model=UI_DEFAULT_MODEL,
        active_thinking_effort="auto",
    )
    settings.set_api_key(ModelProvider.MOONSHOT.value, "moonshot-key")
    settings_store.save(settings)
    session = store.create(project, "code", {})
    app = KolegaCodeApp(
        project_path=project,
        mode="code",
        store=store,
        settings_store=settings_store,
        session=session,
    )

    async with app.run_test():
        composer = app.query_one("#composer", ChatComposer)
        composer.load_text("/model")
        await app.on_chat_composer_submitted(ChatComposer.Submitted(composer, composer.text))

        model_actions = app.query_one("#model_actions", ActionList)
        assert model_actions.display is True

        composer.load_text("bogus-model")
        await app.on_chat_composer_submitted(ChatComposer.Submitted(composer, composer.text))
        assert settings_store.load().active_model == UI_DEFAULT_MODEL
        assert model_actions.display is True
        assert not any(entry.kind == "user" and entry.content == "bogus-model" for entry in app.conversation_entries)
        assert app.agent.messages == []

        composer.load_text(MOONSHOT_K26_MODEL.upper())
        await app.on_chat_composer_submitted(ChatComposer.Submitted(composer, composer.text))
        switched_settings = settings_store.load()
        assert switched_settings.active_model == MOONSHOT_K26_MODEL
        assert switched_settings.active_thinking_effort == "auto"
        assert model_actions.display is False


@pytest.mark.asyncio
async def test_textual_app_model_slash_command_blocks_selector_during_active_turn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, isolated_cli_env: None
) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli.app import KolegaCodeApp
    from kolega_code.cli.tui.widgets import ActionList, ChatComposer

    class FakeCoderAgent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

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
    state_dir = tmp_path / "state"
    store = SessionStore(state_dir)
    settings_store = SettingsStore(state_dir)
    settings = CliSettings(
        active_provider=ModelProvider.MOONSHOT.value,
        active_model=UI_DEFAULT_MODEL,
        active_thinking_effort="auto",
    )
    settings.set_api_key(ModelProvider.MOONSHOT.value, "moonshot-key")
    settings_store.save(settings)
    session = store.create(project, "code", {})
    app = KolegaCodeApp(
        project_path=project,
        mode="code",
        store=store,
        settings_store=settings_store,
        session=session,
    )

    async with app.run_test():
        composer = app.query_one("#composer", ChatComposer)
        app._turn_active = True
        composer.load_text("/model")
        await app.on_chat_composer_submitted(ChatComposer.Submitted(composer, composer.text))

        assert app._pending_model_selection is None
        assert app.query_one("#model_actions", ActionList).display is False
        assert settings_store.load().active_model == UI_DEFAULT_MODEL
        app._turn_active = False


@pytest.mark.asyncio
async def test_textual_app_effort_slash_command_selects_from_action_list(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, isolated_cli_env: None
) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli.app import KolegaCodeApp
    from kolega_code.cli.tui.widgets import ActionList, ChatComposer

    class FakeCoderAgent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

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
    state_dir = tmp_path / "state"
    store = SessionStore(state_dir)
    settings_store = SettingsStore(state_dir)
    settings = CliSettings(
        active_provider=ModelProvider.DEEPSEEK.value,
        active_model=DEEPSEEK_DEFAULT_MODEL,
        active_thinking_effort="high",
    )
    settings.set_api_key(ModelProvider.DEEPSEEK.value, "deepseek-key")
    settings_store.save(settings)
    session = store.create(project, "code", {})
    app = KolegaCodeApp(
        project_path=project,
        mode="code",
        store=store,
        settings_store=settings_store,
        session=session,
    )

    async with app.run_test() as pilot:
        composer = app.query_one("#composer", ChatComposer)
        composer.load_text("/effort")
        await app.on_chat_composer_submitted(ChatComposer.Submitted(composer, composer.text))

        effort_actions = app.query_one("#effort_actions", ActionList)
        assert effort_actions.display is True
        assert app.focused is effort_actions
        assert effort_actions.option_count == 3

        await pilot.press("down", "down", "enter")
        await pilot.pause()
        assert settings_store.load().active_thinking_effort == "max"
        assert effort_actions.display is False

        composer.load_text("/effort")
        await app.on_chat_composer_submitted(ChatComposer.Submitted(composer, composer.text))
        assert app.focused is effort_actions

        await pilot.press("1")
        await pilot.pause()
        assert settings_store.load().active_thinking_effort == "none"
        assert effort_actions.display is False


@pytest.mark.asyncio
async def test_textual_app_effort_slash_command_accepts_typed_selection_and_rejects_invalid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, isolated_cli_env: None
) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli.app import KolegaCodeApp
    from kolega_code.cli.tui.widgets import ActionList, ChatComposer

    class FakeCoderAgent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.messages = []

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

        async def process_message_stream(self, message):
            self.messages.append(message)
            yield {"type": "response", "content": "unexpected"}

    monkeypatch.setattr(agent_runtime_module, "CoderAgent", FakeCoderAgent)

    project = tmp_path / "project"
    project.mkdir()
    state_dir = tmp_path / "state"
    store = SessionStore(state_dir)
    settings_store = SettingsStore(state_dir)
    settings = CliSettings(
        active_provider=ModelProvider.DEEPSEEK.value,
        active_model=DEEPSEEK_DEFAULT_MODEL,
        active_thinking_effort="high",
    )
    settings.set_api_key(ModelProvider.DEEPSEEK.value, "deepseek-key")
    settings_store.save(settings)
    session = store.create(project, "code", {})
    app = KolegaCodeApp(
        project_path=project,
        mode="code",
        store=store,
        settings_store=settings_store,
        session=session,
    )

    async with app.run_test():
        composer = app.query_one("#composer", ChatComposer)
        composer.load_text("/effort")
        await app.on_chat_composer_submitted(ChatComposer.Submitted(composer, composer.text))

        effort_actions = app.query_one("#effort_actions", ActionList)
        assert effort_actions.display is True

        composer.load_text("bogus")
        await app.on_chat_composer_submitted(ChatComposer.Submitted(composer, composer.text))
        assert settings_store.load().active_thinking_effort == "high"
        assert effort_actions.display is True
        assert not any(entry.kind == "user" and entry.content == "bogus" for entry in app.conversation_entries)
        assert app.agent.messages == []

        composer.load_text("MAX")
        await app.on_chat_composer_submitted(ChatComposer.Submitted(composer, composer.text))
        assert settings_store.load().active_thinking_effort == "max"
        assert effort_actions.display is False


@pytest.mark.asyncio
async def test_textual_app_effort_slash_command_blocks_selector_during_active_turn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, isolated_cli_env: None
) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli.app import KolegaCodeApp
    from kolega_code.cli.tui.widgets import ActionList, ChatComposer

    class FakeCoderAgent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

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
    state_dir = tmp_path / "state"
    store = SessionStore(state_dir)
    settings_store = SettingsStore(state_dir)
    settings = CliSettings(
        active_provider=ModelProvider.DEEPSEEK.value,
        active_model=DEEPSEEK_DEFAULT_MODEL,
        active_thinking_effort="high",
    )
    settings.set_api_key(ModelProvider.DEEPSEEK.value, "deepseek-key")
    settings_store.save(settings)
    session = store.create(project, "code", {})
    app = KolegaCodeApp(
        project_path=project,
        mode="code",
        store=store,
        settings_store=settings_store,
        session=session,
    )

    async with app.run_test():
        composer = app.query_one("#composer", ChatComposer)
        app._turn_active = True
        composer.load_text("/effort")
        await app.on_chat_composer_submitted(ChatComposer.Submitted(composer, composer.text))

        assert app._pending_effort_selection is None
        assert app.query_one("#effort_actions", ActionList).display is False
        assert settings_store.load().active_thinking_effort == "high"
        app._turn_active = False


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

        # A background (NoWidget) click does set_focus(None); the blur hook restores.
        app.screen.set_focus(None)
        await pilot.pause()
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
