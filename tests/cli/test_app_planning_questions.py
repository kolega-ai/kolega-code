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
async def test_textual_app_planning_question_tool_accepts_option_list_answer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from textual.widgets import OptionList

    from kolega_code.cli.app import KolegaCodeApp
    from kolega_code.cli.tui.widgets import ActionList, ChatComposer
    from kolega_code.cli.messages import COMPOSER_PLACEHOLDER, QUEUE_PLACEHOLDER

    class FakeBaseAgent:
        instances: list["FakeBaseAgent"] = []

        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.history = []
            self.__class__.instances.append(self)

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
        await app.action_toggle_interaction_mode()
        ask_user_choice = extension_by_name(
            FakeBaseAgent.instances[-1].kwargs["tool_extensions"], "cli-planning-questions"
        ).tools["ask_user_choice"]

        app._turn_active = True
        answer_task = asyncio.create_task(
            ask_user_choice(
                questions=question_payload(
                    "Which approach should we use?",
                    [("Keep state local", "Store in memory"), ("Persist it", "Write to disk")],
                    header="Approach",
                )
            )
        )
        await pilot.pause()

        assert app._pending_question is not None
        assert app.query_one("#composer", ChatComposer).disabled is False
        question_actions = app.query_one("#question_actions", ActionList)
        assert question_actions.display is True
        assert app.focused is question_actions
        assert question_actions.highlighted == 0
        assert question_actions.get_option("question_option_0").prompt == "1. Keep state local — Store in memory"
        # While pending, the prompt lives only in the combined panel — no chat bubble.
        assert all(entry.kind != "question" for entry in app.conversation_entries)

        selected = question_actions.get_option("question_option_1")
        await app.on_option_list_option_selected(OptionList.OptionSelected(question_actions, selected, 1))

        assert json.loads(await answer_task) == {"Approach": "Persist it"}
        assert app._pending_question is None
        assert app.query_one("#question_actions").display is False
        assert app.query_one("#composer", ChatComposer).disabled is False
        assert app.query_one("#composer", ChatComposer).placeholder == QUEUE_PLACEHOLDER
        # After answering, the question is recorded followed by the chosen answer.
        assert app.conversation_entries[-2].kind == "question"
        assert app.conversation_entries[-2].content == "Which approach should we use?"
        assert app.conversation_entries[-1].kind == "user"
        assert app.conversation_entries[-1].content == "Persist it"
        app._turn_active = False


@pytest.mark.asyncio
async def test_textual_app_planning_question_supports_arrow_and_digit_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli.app import KolegaCodeApp
    from kolega_code.cli.tui.widgets import ActionList

    class FakeBaseAgent:
        instances: list["FakeBaseAgent"] = []

        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.history = []
            self.__class__.instances.append(self)

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
        await app.action_toggle_interaction_mode()
        ask_user_choice = extension_by_name(
            FakeBaseAgent.instances[-1].kwargs["tool_extensions"], "cli-planning-questions"
        ).tools["ask_user_choice"]

        options = ["Alpha", "Beta", "Gamma", "Delta"]
        answer_task = asyncio.create_task(
            ask_user_choice(questions=question_payload("Pick one of four?", options, header="Pick"))
        )
        await pilot.pause()

        question_actions = app.query_one("#question_actions", ActionList)
        assert question_actions.option_count == 4
        assert app.focused is question_actions

        await pilot.press("down", "down", "enter")
        assert json.loads(await answer_task) == {"Pick": "Gamma"}
        assert question_actions.display is False

        answer_task = asyncio.create_task(
            ask_user_choice(questions=question_payload("Pick again?", options, header="Pick"))
        )
        await pilot.pause()

        assert app.focused is app.query_one("#question_actions", ActionList)
        await pilot.press("4")
        assert json.loads(await answer_task) == {"Pick": "Delta"}


@pytest.mark.asyncio
async def test_textual_app_planning_question_tool_accepts_custom_text_answer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli.app import KolegaCodeApp
    from kolega_code.cli.tui.widgets import ActionList, ChatComposer

    class FakeBaseAgent:
        instances: list["FakeBaseAgent"] = []

        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.history = []
            self.__class__.instances.append(self)

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
        await app.action_toggle_interaction_mode()
        ask_user_choice = extension_by_name(
            FakeBaseAgent.instances[-1].kwargs["tool_extensions"], "cli-planning-questions"
        ).tools["ask_user_choice"]

        answer_task = asyncio.create_task(
            ask_user_choice(questions=question_payload("Which scope?", ["Small fix", "Full workflow"], header="Scope"))
        )
        await pilot.pause()

        question_actions = app.query_one("#question_actions", ActionList)
        assert question_actions.get_option("question_option_0").prompt == "1. Small fix — details"

        composer = app.query_one("#composer", ChatComposer)
        composer.load_text("Start with the small fix, but keep the API extensible.")
        await app.on_chat_composer_submitted(ChatComposer.Submitted(composer, composer.text))

        assert json.loads(await answer_task) == {"Scope": "Start with the small fix, but keep the API extensible."}
        assert composer.text == ""
        assert app._pending_question is None
        assert question_actions.display is False
        assert question_actions.option_count == 0
        assert app.conversation_entries[-1].content == "Start with the small fix, but keep the API extensible."


@pytest.mark.asyncio
async def test_textual_app_planning_question_tool_asks_multiple_questions_sequentially(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli.app import KolegaCodeApp
    from kolega_code.cli.tui.widgets import ActionList
    from textual.widgets import OptionList

    class FakeBaseAgent:
        instances: list["FakeBaseAgent"] = []

        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.history = []
            self.__class__.instances.append(self)

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
        await app.action_toggle_interaction_mode()
        ask_user_choice = extension_by_name(
            FakeBaseAgent.instances[-1].kwargs["tool_extensions"], "cli-planning-questions"
        ).tools["ask_user_choice"]

        questions = question_payload("First?", ["A1", "B1"], header="First") + question_payload(
            "Second?", ["A2", "B2"], header="Second"
        )
        answer_task = asyncio.create_task(ask_user_choice(questions=questions))
        await pilot.pause()

        # First question is presented; answer it, then the second appears.
        question_actions = app.query_one("#question_actions", ActionList)
        assert question_actions.option_count == 2
        selected = question_actions.get_option("question_option_0")
        await app.on_option_list_option_selected(OptionList.OptionSelected(question_actions, selected, 0))
        await pilot.pause()

        assert app._pending_question is not None
        question_actions = app.query_one("#question_actions", ActionList)
        selected = question_actions.get_option("question_option_1")
        await app.on_option_list_option_selected(OptionList.OptionSelected(question_actions, selected, 1))

        assert json.loads(await answer_task) == {"First": "A1", "Second": "B2"}
        assert app._pending_question is None


@pytest.mark.asyncio
async def test_textual_app_planning_question_tool_rejects_malformed_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli.app import KolegaCodeApp
    from kolega_code.tools import ToolError

    class FakeBaseAgent:
        instances: list["FakeBaseAgent"] = []

        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.history = []
            self.__class__.instances.append(self)

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

    async with app.run_test():
        await app.action_toggle_interaction_mode()
        ask_user_choice = extension_by_name(
            FakeBaseAgent.instances[-1].kwargs["tool_extensions"], "cli-planning-questions"
        ).tools["ask_user_choice"]

        # Empty / non-list questions.
        with pytest.raises(ToolError):
            await ask_user_choice(questions=[])
        with pytest.raises(ToolError):
            await ask_user_choice(questions="Which approach?")

        # Fewer than two valid options.
        with pytest.raises(ToolError):
            await ask_user_choice(questions=question_payload("Q?", ["only one"]))

        # Options that are bare strings rather than {label, description} objects.
        with pytest.raises(ToolError):
            await ask_user_choice(
                questions=[{"question": "Q?", "header": "H", "multiSelect": False, "options": ["A", "B"]}]
            )

        # Missing question text.
        with pytest.raises(ToolError):
            await ask_user_choice(
                questions=[
                    {
                        "question": "  ",
                        "header": "H",
                        "multiSelect": False,
                        "options": [
                            {"label": "A", "description": "d"},
                            {"label": "B", "description": "d"},
                        ],
                    }
                ]
            )

        assert app._pending_question is None
