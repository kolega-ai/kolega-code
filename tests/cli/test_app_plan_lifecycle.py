# ruff: noqa: F401,F811,E402
from pathlib import Path
import asyncio
import json
import time

import pytest

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
from kolega_code.cli.plan_artifacts import current_plan_artifact_path
from kolega_code.cli.provider_registry import (
    DEEPSEEK_DEFAULT_MODEL,
    MOONSHOT_K26_MODEL,
    UI_DEFAULT_MODEL,
    UI_DEFAULT_PROVIDER,
)
from kolega_code.cli.session_store import SessionStore
from kolega_code.cli.settings import CliSettings, SettingsStore

from ._app_test_utils import (
    FakeCoderAgent,
    _build_mention_test_app,
    _build_sub_agent_test_app,
    _sub_agent_context_event,
    _sub_agent_entries,
    _sub_agent_event,
    _workflow_event,
    build_test_config,
    extension_by_name,
    first_text_styles,
    install_fake_agents,
    question_payload,
    renderable_text,
)


@pytest.mark.asyncio
async def test_textual_app_shows_plan_decision_when_planning_agent_writes_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli.tui.widgets import PlanningMarkdown

    from kolega_code.cli.app import KolegaCodeApp
    from kolega_code.cli.tui.widgets import ActionList, ChatComposer

    class FakePlanningAgent(FakeCoderAgent):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.completed_plan = "# Plan\n\n" + "\n".join(
                f"- Step {index}: keep the planning sidebar readable." for index in range(1, 26)
            )

        async def process_message_stream(self, message):
            yield {"type": "response", "content": "I have a plan.", "complete": True, "uuid": "response-1"}

        def consume_completed_plan(self):
            plan = self.completed_plan
            self.completed_plan = None
            return plan

    install_fake_agents(monkeypatch, planning_cls=FakePlanningAgent)

    project = tmp_path / "project"
    project.mkdir()
    config = build_test_config(project)
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", config_summary(config))
    app = KolegaCodeApp(project_path=project, config=config, mode="code", store=store, session=session)

    async with app.run_test():
        await app.action_toggle_interaction_mode()
        await app._process_message("plan this")

        assert isinstance(app.agent, FakePlanningAgent)
        initial_plan = app.agent.completed_plan or app._latest_plan
        assert initial_plan is not None
        assert app._plan_decision_active is True
        assert app._plan_pending is True
        assert app._latest_plan == initial_plan
        assert app.query_one("#composer", ChatComposer).disabled is True
        assert (
            app.query_one("#composer", ChatComposer).placeholder
            == "Plan ready. Choose Implement plan or Discuss further."
        )
        plan_actions = app.query_one("#plan_actions", ActionList)
        assert plan_actions.display is True
        assert [option.id for option in plan_actions.options] == [
            "implement_plan",
            "implement_plan_clear",
            "discuss_plan",
        ]
        assert app.focused is plan_actions
        assert app.query_one("#planning_plan_markdown", PlanningMarkdown).source == initial_plan
        assert "Step 25" in app.query_one("#planning_plan_markdown", PlanningMarkdown).source
        assert app.conversation_entries[-1].kind == "plan"
        loaded = store.load(session.session_id)
        assert loaded.latest_plan_markdown == initial_plan
        assert loaded.plan_reofferable is True
        assert loaded.interaction_mode == "plan"
        plan_artifact = current_plan_artifact_path(store.root, session.session_id)
        assert plan_artifact.read_text(encoding="utf-8") == initial_plan + "\n"

        await app._discuss_pending_plan()

        assert app._plan_decision_active is False
        assert app._plan_pending is False
        assert app._plan_reofferable is True
        assert app._latest_plan == initial_plan
        assert app.query_one("#composer", ChatComposer).disabled is False
        assert app.query_one("#planning_plan_markdown", PlanningMarkdown).source == initial_plan
        assert plan_actions.display is False
        assert plan_actions.option_count == 0
        loaded = store.load(session.session_id)
        assert loaded.latest_plan_markdown == initial_plan
        assert loaded.plan_pending is False
        assert loaded.plan_reofferable is True

        await app._process_message("keep discussing")

        assert app._plan_decision_active is True
        assert app._plan_pending is True
        assert app._latest_plan == initial_plan
        assert app.query_one("#composer", ChatComposer).disabled is True
        assert plan_actions.display is True
        assert [option.id for option in plan_actions.options] == [
            "implement_plan",
            "implement_plan_clear",
            "discuss_plan",
        ]
        assert app.conversation_entries[-1].kind == "plan"
        assert app.conversation_entries[-1].content == initial_plan
        loaded = store.load(session.session_id)
        assert loaded.latest_plan_markdown == initial_plan
        assert loaded.plan_pending is True
        assert loaded.plan_reofferable is True

        await app._discuss_pending_plan()

        assert isinstance(app.agent, FakePlanningAgent)
        app.agent.completed_plan = "# Revised plan\n\nBuild planning mode carefully."
        await app._capture_completed_plan()

        assert app._plan_decision_active is True
        assert app._plan_pending is True
        assert app._latest_plan == "# Revised plan\n\nBuild planning mode carefully."
        assert app.query_one("#composer", ChatComposer).disabled is True
        assert plan_actions.display is True
        assert [option.id for option in plan_actions.options] == [
            "implement_plan",
            "implement_plan_clear",
            "discuss_plan",
        ]
        assert (
            app.query_one("#planning_plan_markdown", PlanningMarkdown).source
            == "# Revised plan\n\nBuild planning mode carefully."
        )
        loaded = store.load(session.session_id)
        assert loaded.latest_plan_markdown == "# Revised plan\n\nBuild planning mode carefully."
        assert loaded.plan_reofferable is True
        assert plan_artifact.read_text(encoding="utf-8") == "# Revised plan\n\nBuild planning mode carefully.\n"
        assert initial_plan not in plan_artifact.read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_textual_app_implement_plan_switches_to_build_and_sends_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli.tui.widgets import PlanningMarkdown

    from kolega_code.cli.app import KolegaCodeApp

    install_fake_agents(monkeypatch, planning_cls=FakeCoderAgent)

    project = tmp_path / "project"
    project.mkdir()
    config = build_test_config(project)
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", config_summary(config))
    app = KolegaCodeApp(project_path=project, config=config, mode="code", store=store, session=session)

    async with app.run_test():
        await app.action_toggle_interaction_mode()
        app._latest_plan = "# Plan\n\nBuild it."
        app._plan_pending = True
        app._plan_reofferable = True
        app._plan_decision_active = True

        await app._implement_pending_plan()
        assert app.agent_worker is not None
        await app.agent_worker.wait()

        assert app.interaction_mode == "build"
        assert isinstance(app.agent, FakeCoderAgent)
        assert app.agent.messages
        plan_artifact = current_plan_artifact_path(store.root, session.session_id)
        assert plan_artifact.read_text(encoding="utf-8") == "# Plan\n\nBuild it.\n"
        assert "# Plan\n\nBuild it." in app.agent.messages[-1]
        assert str(plan_artifact) in app.agent.messages[-1]
        artifact_extension = extension_by_name(app.agent.kwargs["prompt_extensions"], "cli-current-plan-artifact")
        assert str(plan_artifact) in artifact_extension.markdown
        assert artifact_extension.propagate_to_sub_agents is True
        assert app._plan_decision_active is False
        # The plan is kept as a read-only sidebar reference, but it is no longer
        # pending a decision so the action must not be re-offered.
        assert app._plan_pending is False
        assert app._plan_reofferable is False
        assert app._latest_plan == "# Plan\n\nBuild it."
        assert app.query_one("#planning_plan_markdown", PlanningMarkdown).source == "# Plan\n\nBuild it."
        assert app.query_one("#plan_actions").display is False
        loaded = store.load(session.session_id)
        assert loaded.latest_plan_markdown == "# Plan\n\nBuild it."
        assert loaded.plan_pending is False
        assert loaded.plan_reofferable is False
        assert loaded.interaction_mode == "build"


@pytest.mark.asyncio
async def test_textual_app_implemented_plan_not_reoffered_on_reentry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli.tui.widgets import PlanningMarkdown

    from kolega_code.cli.app import KolegaCodeApp
    from kolega_code.cli.tui.constants import PLAN_INTERACTION_MODE
    from kolega_code.cli.tui.widgets import ActionList

    install_fake_agents(monkeypatch, planning_cls=FakeCoderAgent)

    project = tmp_path / "project"
    project.mkdir()
    config = build_test_config(project)
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", config_summary(config))
    app = KolegaCodeApp(project_path=project, config=config, mode="code", store=store, session=session)

    async with app.run_test():
        # Enter plan mode with a freshly captured plan awaiting a decision.
        await app.action_toggle_interaction_mode()
        app._latest_plan = "# Plan\n\nBuild it."
        app._plan_pending = True
        app._plan_reofferable = True
        app._plan_decision_active = True

        # Implement it: switches to build and runs the plan.
        await app._implement_pending_plan()
        assert app.agent_worker is not None
        await app.agent_worker.wait()
        assert app.interaction_mode == "build"

        # Re-enter plan mode. The already-implemented plan must NOT be re-offered,
        # but it stays visible in the sidebar as a read-only reference.
        await app._set_interaction_mode(PLAN_INTERACTION_MODE)

        assert app._plan_pending is False
        assert app._plan_reofferable is False
        assert app._latest_plan == "# Plan\n\nBuild it."
        plan_actions = app.query_one("#plan_actions", ActionList)
        assert plan_actions.display is False
        assert plan_actions.option_count == 0
        assert app.query_one("#planning_plan_markdown", PlanningMarkdown).source == "# Plan\n\nBuild it."

        # A restart (reloading from the persisted session) must also not re-offer it.
        loaded = store.load(session.session_id)
        assert loaded.plan_pending is False
        assert loaded.plan_reofferable is False


@pytest.mark.asyncio
async def test_textual_app_clear_context_and_implement_plan_starts_build_agent_fresh(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli.tui.widgets import PlanningMarkdown

    from kolega_code.cli.app import KolegaCodeApp

    install_fake_agents(monkeypatch, planning_cls=FakeCoderAgent)

    project = tmp_path / "project"
    project.mkdir()
    config = build_test_config(project)
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", config_summary(config))
    app = KolegaCodeApp(project_path=project, config=config, mode="code", store=store, session=session)

    async with app.run_test():
        await app.action_toggle_interaction_mode()
        # Seed the planning agent with prior conversation that the normal implement flow
        # would carry forward into the build agent.
        assert app.agent is not None
        app.agent.history = ["planning message 1", "planning message 2"]
        prior_entry_count = len(app.conversation_entries)
        app._latest_plan = "# Plan\n\nBuild it."
        app._plan_pending = True
        app._plan_reofferable = True
        app._plan_decision_active = True

        await app._implement_pending_plan(clear_context=True)
        assert app.agent_worker is not None
        await app.agent_worker.wait()

        assert app.interaction_mode == "build"
        assert isinstance(app.agent, FakeCoderAgent)
        # The build agent started a new epoch: the planning conversation never reached
        # it, while the new implement-plan turn was durably recorded.
        history_text = [
            message if isinstance(message, str) else message.get_text_content() for message in app.agent.history
        ]
        assert "planning message 1" not in history_text
        assert "planning message 2" not in history_text
        assert len(app.session.history) == 2
        # The plan is still delivered to the build agent via the implement prompt.
        assert app.agent.messages
        assert "# Plan\n\nBuild it." in app.agent.messages[-1]
        # The plan itself is preserved (sidebar keeps showing it).
        assert app._plan_decision_active is False
        assert app._plan_pending is False
        assert app._plan_reofferable is False
        assert app._latest_plan == "# Plan\n\nBuild it."
        assert app.query_one("#planning_plan_markdown", PlanningMarkdown).source == "# Plan\n\nBuild it."
        assert app.query_one("#plan_actions").display is False
        # LLM-context-only clear: the visible transcript is preserved, plus the new
        # "Implement the approved plan." entry.
        assert len(app.conversation_entries) > prior_entry_count
        assert any(
            entry.kind == "user" and entry.content == "Implement the approved plan."
            for entry in app.conversation_entries
        )


@pytest.mark.asyncio
async def test_textual_app_discuss_plan_preserves_old_plan_until_new_plan_is_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli.tui.widgets import PlanningMarkdown

    from kolega_code.cli.app import KolegaCodeApp

    install_fake_agents(monkeypatch, planning_cls=FakeCoderAgent)

    project = tmp_path / "project"
    project.mkdir()
    config = build_test_config(project)
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", config_summary(config))
    app = KolegaCodeApp(project_path=project, config=config, mode="code", store=store, session=session)

    async with app.run_test():
        await app.action_toggle_interaction_mode()
        app._latest_plan = "# Plan\n\nBuild it after discussing."
        app._plan_pending = True
        app._plan_reofferable = True
        app._plan_decision_active = True

        await app._discuss_pending_plan()

        assert app._latest_plan == "# Plan\n\nBuild it after discussing."
        assert app._plan_pending is False
        assert app._plan_reofferable is True
        assert app._plan_decision_active is False
        assert (
            app.query_one("#planning_plan_markdown", PlanningMarkdown).source == "# Plan\n\nBuild it after discussing."
        )
        assert app.query_one("#plan_actions").display is False
        loaded = store.load(session.session_id)
        assert loaded.latest_plan_markdown == "# Plan\n\nBuild it after discussing."
        assert loaded.plan_pending is False
        assert loaded.plan_reofferable is True

        await app._implement_pending_plan()
        first_worker = app.agent_worker
        assert first_worker is None
        assert app.interaction_mode == "plan"

        app._latest_plan = "# New plan\n\nBuild this instead."
        app._plan_pending = True
        app._plan_reofferable = True
        app._plan_decision_active = True

        await app._implement_pending_plan()
        assert app.agent_worker is not None
        await app.agent_worker.wait()

        assert app.interaction_mode == "build"
        assert isinstance(app.agent, FakeCoderAgent)
        plan_artifact = current_plan_artifact_path(store.root, session.session_id)
        assert plan_artifact.read_text(encoding="utf-8") == "# New plan\n\nBuild this instead.\n"
        assert "# New plan\n\nBuild this instead." in app.agent.messages[-1]
        assert "# Plan\n\nBuild it after discussing." not in app.agent.messages[-1]
        assert "# Plan\n\nBuild it after discussing." not in plan_artifact.read_text(encoding="utf-8")
        assert app._latest_plan == "# New plan\n\nBuild this instead."
        assert app._plan_reofferable is False
        assert app.query_one("#planning_plan_markdown", PlanningMarkdown).source == "# New plan\n\nBuild this instead."
        assert app.query_one("#plan_actions").display is False
        loaded = store.load(session.session_id)
        assert loaded.latest_plan_markdown == "# New plan\n\nBuild this instead."
        assert loaded.plan_reofferable is False
