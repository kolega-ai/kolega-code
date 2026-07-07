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

_FakeCoderAgentBase = FakeCoderAgent


@pytest.mark.asyncio
async def test_textual_app_context_usage_updates_status_without_raw_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from textual.widgets import Static

    from kolega_code.cli.app import KolegaCodeApp
    from kolega_code.cli.tui.widgets import ChatComposer
    from kolega_code.cli.messages import COMPOSER_PLACEHOLDER

    install_fake_agents(monkeypatch)

    project = tmp_path / "project"
    project.mkdir()
    config = build_test_config(project)
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", config_summary(config))
    app = KolegaCodeApp(project_path=project, config=config, mode="code", store=store, session=session)

    async with app.run_test():
        composer = app.query_one("#composer", ChatComposer)
        app._render_event(
            AgentEvent(
                event_type="llm_context_update",
                sender="coder",
                content={
                    "input_tokens": 123456,
                    "max_tokens": 200000,
                    "usage_percentage": 61.7,
                    "alert_level": "info",
                    "message": "Context is getting large.",
                    "compression_threshold": 80.0,
                },
            )
        )
        dashboard = str(app.query_one("#status_dashboard", Static).render())

        assert "61.7%" in dashboard
        assert "123,456 / 200,000" in dashboard
        assert "Compresses at 80%" in dashboard
        assert "Context is getting large." in dashboard
        assert "input_tokens" not in dashboard
        assert composer.placeholder == COMPOSER_PLACEHOLDER

        app._render_event(AgentEvent(event_type="status_update", sender="coder", content={"input_tokens": 5}))
        assert composer.placeholder == COMPOSER_PLACEHOLDER


@pytest.mark.asyncio
async def test_textual_app_status_dashboard_tracks_interaction_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from textual.widgets import Static

    from kolega_code.cli.app import KolegaCodeApp

    install_fake_agents(monkeypatch, planning_cls=FakeCoderAgent)

    project = tmp_path / "project"
    project.mkdir()
    config = build_test_config(project)
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", config_summary(config))
    app = KolegaCodeApp(project_path=project, config=config, mode="code", store=store, session=session)

    async with app.run_test():
        await app._set_interaction_mode("plan")
        dashboard = str(app.query_one("#status_dashboard", Static).render())
        assert "Plan" in dashboard

        await app._set_interaction_mode("build")
        dashboard = str(app.query_one("#status_dashboard", Static).render())
        assert "Build" in dashboard


@pytest.mark.asyncio
async def test_textual_app_resumes_gigacode_enabled_across_mode_rebuilds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from textual.widgets import Static

    from kolega_code.cli.app import KolegaCodeApp

    class FakeCoderAgent(_FakeCoderAgentBase):
        instances = []

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.prompt_extensions = kwargs.get("prompt_extensions") or []
            self.gigacode_enabled = False
            self.cleaned = False
            self.__class__.instances.append(self)

        async def cleanup(self):
            self.cleaned = True

    class FakePlanningAgent(FakeCoderAgent):
        instances = []

    monkeypatch.setattr(agent_runtime_module, "CoderAgent", FakeCoderAgent)
    monkeypatch.setattr(agent_runtime_module, "PlanningAgent", FakePlanningAgent)

    project = tmp_path / "project"
    project.mkdir()
    config = build_test_config(project)
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", config_summary(config))
    session.gigacode_enabled = True
    store.save(session)
    app = KolegaCodeApp(project_path=project, config=config, mode="code", store=store, session=session)

    async with app.run_test():
        assert app._gigacode_enabled is True
        assert FakeCoderAgent.instances
        coder = FakeCoderAgent.instances[-1]
        assert coder.gigacode_enabled is True
        assert extension_by_name(coder.prompt_extensions, "gigacode").title == "gigacode — workflow orchestration"
        assert "Gigacode: on" in app.conversation_entries[0].content
        assert "gigacode on" in str(app.query_one("#session_meta", Static).render())
        dashboard = str(app.query_one("#status_dashboard", Static).render())
        assert "Gigacode" in dashboard
        assert "On" in dashboard

        await app._set_interaction_mode("plan")

        assert FakePlanningAgent.instances
        planning = FakePlanningAgent.instances[-1]
        assert app._gigacode_enabled is True
        assert planning.gigacode_enabled is True
        assert extension_by_name(planning.prompt_extensions, "gigacode").title == "gigacode — workflow orchestration"
        assert store.load(session.session_id).gigacode_enabled is True

        await app._set_interaction_mode("build")

        rebuilt = FakeCoderAgent.instances[-1]
        assert rebuilt.gigacode_enabled is True
        assert extension_by_name(rebuilt.prompt_extensions, "gigacode").title == "gigacode — workflow orchestration"


@pytest.mark.asyncio
async def test_textual_app_gigacode_command_persists_and_updates_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from textual.widgets import Static

    from kolega_code.cli.app import KolegaCodeApp

    class FakeCoderAgent(_FakeCoderAgentBase):
        instances = []

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.gigacode_enabled = False
            self.prompt_extensions = kwargs.get("prompt_extensions") or []
            self.apply_calls = []
            self.__class__.instances.append(self)

        def apply_gigacode(self, enabled, prompt_extension=None):
            self.apply_calls.append((enabled, prompt_extension))
            self.gigacode_enabled = enabled
            self.prompt_extensions = [
                extension for extension in self.prompt_extensions if getattr(extension, "id", None) != "gigacode"
            ]
            if enabled and prompt_extension is not None:
                self.prompt_extensions.append(prompt_extension)

    monkeypatch.setattr(agent_runtime_module, "CoderAgent", FakeCoderAgent)

    project = tmp_path / "project"
    project.mkdir()
    config = build_test_config(project)
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", config_summary(config))
    app = KolegaCodeApp(project_path=project, config=config, mode="code", store=store, session=session)

    async with app.run_test():
        await app._command_gigacode("on")

        assert app._gigacode_enabled is True
        coder = FakeCoderAgent.instances[-1]
        assert coder.apply_calls[-1][0] is True
        assert coder.apply_calls[-1][1].id == "gigacode"
        assert store.load(session.session_id).gigacode_enabled is True
        assert "Gigacode: on" in app.conversation_entries[0].content
        dashboard = str(app.query_one("#status_dashboard", Static).render())
        assert "Gigacode" in dashboard
        assert "On" in dashboard

        await app._command_gigacode("off")

        assert app._gigacode_enabled is False
        assert FakeCoderAgent.instances[-1].apply_calls[-1] == (False, None)
        assert store.load(session.session_id).gigacode_enabled is False
        assert "Gigacode: off" in app.conversation_entries[0].content
        assert "gigacode off" in str(app.query_one("#session_meta", Static).render())


@pytest.mark.asyncio
async def test_textual_app_mode_switch_rebuild_skips_transcript_restore(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from textual.widgets import Static

    from kolega_code.cli.app import KolegaCodeApp

    history = [{"role": "user", "content": [{"type": "text", "text": "keep me"}]}]
    compaction = {"summary": "summary", "compacted_through": 1, "compacted_history_length": 1}

    class FakeCoderAgent(_FakeCoderAgentBase):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.restored_history = None
            self.restored_compaction = None

        def restore_message_history(self, restored):
            self.restored_history = restored

        def dump_compaction_state(self):
            return compaction

        def restore_compaction_state(self, data):
            self.restored_compaction = data

        def dump_message_history(self):
            return history

    class FakePlanningAgent(FakeCoderAgent):
        instances = []

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.instances.append(self)

    monkeypatch.setattr(agent_runtime_module, "CoderAgent", FakeCoderAgent)
    monkeypatch.setattr(agent_runtime_module, "PlanningAgent", FakePlanningAgent)

    project = tmp_path / "project"
    project.mkdir()
    config = build_test_config(project)
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", config_summary(config))
    app = KolegaCodeApp(project_path=project, config=config, mode="code", store=store, session=session)

    async with app.run_test():
        restore_calls = []
        render_calls = []

        def spy_restore(restored):
            restore_calls.append(restored)

        def spy_render():
            render_calls.append(True)

        monkeypatch.setattr(app, "_restore_conversation_history", spy_restore)
        monkeypatch.setattr(app, "_render_conversation", spy_render)

        await app._set_interaction_mode("plan")

        assert restore_calls == []
        assert render_calls == []
        assert app.interaction_mode == "plan"
        assert "plan" in str(app.query_one("#session_meta", Static).render())
        assert "Plan" in str(app.query_one("#status_dashboard", Static).render())

        assert FakePlanningAgent.instances
        planning_agent = FakePlanningAgent.instances[-1]
        assert planning_agent.restored_history == history
        assert planning_agent.restored_compaction == compaction


@pytest.mark.asyncio
async def test_textual_app_mode_switch_preserves_transcript_entries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli.app import KolegaCodeApp
    from kolega_code.cli.tui.state import ConversationEntry

    history = [{"role": "user", "content": [{"type": "text", "text": "persisted"}]}]

    class FakeAgent(FakeCoderAgent):
        def dump_message_history(self):
            return history

    monkeypatch.setattr(agent_runtime_module, "CoderAgent", FakeAgent)
    monkeypatch.setattr(agent_runtime_module, "PlanningAgent", FakeAgent)

    project = tmp_path / "project"
    project.mkdir()
    config = build_test_config(project)
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", config_summary(config))
    app = KolegaCodeApp(project_path=project, config=config, mode="code", store=store, session=session)

    async with app.run_test():
        startup = app.conversation_entries[0]
        user = ConversationEntry(kind="user", content="hello")
        assistant = ConversationEntry(kind="assistant", content="hi", complete=True)
        tool = ConversationEntry(
            kind="tool_result",
            content="done",
            complete=True,
            tool_name="read_file",
            tool_call_id="tool-1",
            full_content="done",
        )
        app.conversation_entries = [startup, user, assistant, tool]
        non_startup_entries = app.conversation_entries[1:]

        await app._set_interaction_mode("plan")

        assert app.conversation_entries[0] is startup
        assert app.conversation_entries[1:] == non_startup_entries
        assert app.conversation_entries[1] is user
        assert app.conversation_entries[2] is assistant
        assert app.conversation_entries[3] is tool


@pytest.mark.asyncio
async def test_textual_app_turn_status_formats_error_duration(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("textual")

    from textual.widgets import Static

    from kolega_code.cli.app import KolegaCodeApp
    from kolega_code.cli.tui.state import TurnState

    install_fake_agents(monkeypatch)

    project = tmp_path / "project"
    project.mkdir()
    config = build_test_config(project)
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", config_summary(config))
    app = KolegaCodeApp(project_path=project, config=config, mode="code", store=store, session=session)
    now = 0.0
    monkeypatch.setattr(app, "_now", lambda: now)

    async with app.run_test():
        app._begin_turn_progress()
        now = 83.0
        app._finish_turn_progress("Stopped due to an error: boom", TurnState.ERROR)

        assert "Errored after 1m 23s" in str(app.query_one("#turn_status", Static).render())


@pytest.mark.asyncio
async def test_textual_app_shift_tab_toggles_between_build_and_plan_agents(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from textual.binding import Binding

    from kolega_code.cli.tui.widgets import PlanningMarkdown

    from kolega_code.cli.app import KolegaCodeApp
    from kolega_code.cli.tui.constants import BUILD_INTERACTION_MODE, PLAN_INTERACTION_MODE
    from kolega_code.cli.tui.state import PendingQuestion

    class FakePlanningAgent(FakeCoderAgent):
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
        toggle_binding = next(
            b for b in app.BINDINGS if isinstance(b, Binding) and b.action == "toggle_interaction_mode"
        )
        assert toggle_binding.key == "shift+tab"
        assert toggle_binding.key_display == "Shift+Tab"
        assert toggle_binding.priority is True

        assert isinstance(app.agent, FakeCoderAgent)
        assert app.interaction_mode == BUILD_INTERACTION_MODE

        await pilot.press("shift+tab")

        assert app.interaction_mode == PLAN_INTERACTION_MODE
        assert isinstance(app.agent, FakePlanningAgent)
        startup = app.conversation_entries[0].content
        assert "Interaction: plan" in startup

        app._latest_plan = "# Plan\n\nDo it."
        app._plan_decision_active = False
        app._set_plan_actions_visible(True)
        question_future = asyncio.get_running_loop().create_future()
        app._pending_question = PendingQuestion(
            question="Choose?",
            options=["A", "B"],
            future=question_future,
        )
        app._set_question_actions_visible(True)

        await pilot.press("shift+tab")

        assert app.interaction_mode == BUILD_INTERACTION_MODE
        assert isinstance(app.agent, FakeCoderAgent)
        assert app._latest_plan == "# Plan\n\nDo it."
        assert app._plan_decision_active is False
        assert app._pending_question is None
        assert question_future.cancelled()
        assert app.query_one("#planning_plan_markdown", PlanningMarkdown).source == "# Plan\n\nDo it."
        assert app.query_one("#plan_actions").display is False
        assert app.query_one("#question_actions").display is False
        loaded = store.load(session.session_id)
        assert loaded.latest_plan_markdown == "# Plan\n\nDo it."
        assert loaded.interaction_mode == BUILD_INTERACTION_MODE


@pytest.mark.asyncio
async def test_textual_app_ctrl_p_toggles_permission_mode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("textual")

    from textual.widgets import Static

    from textual.binding import Binding

    from kolega_code.cli.app import KolegaCodeApp
    from kolega_code.permissions import PermissionMode

    class FakeCoderAgent(_FakeCoderAgentBase):
        instances = []

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.permission_mode = kwargs["permission_mode"]
            self.__class__.instances.append(self)

        def set_permission_mode(self, permission_mode):
            self.permission_mode = permission_mode

        def set_permission_callback(self, permission_callback):
            self.permission_callback = permission_callback

    monkeypatch.setattr(agent_runtime_module, "CoderAgent", FakeCoderAgent)

    project = tmp_path / "project"
    project.mkdir()
    config = build_test_config(project)
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", config_summary(config))
    app = KolegaCodeApp(project_path=project, config=config, mode="code", store=store, session=session)

    async with app.run_test() as pilot:
        toggle_binding = next(
            b for b in app.BINDINGS if isinstance(b, Binding) and b.action == "toggle_permission_mode"
        )
        assert toggle_binding.key == "ctrl+p"
        assert app.permission_mode == PermissionMode.ASK
        coder = FakeCoderAgent.instances[-1]
        assert coder.kwargs["permission_mode"] == PermissionMode.ASK

        await pilot.press("ctrl+p")

        assert app.permission_mode == PermissionMode.AUTO
        assert app.agent is not None
        assert app.agent.permission_mode == PermissionMode.AUTO
        assert store.load(session.session_id).permission_mode == "auto"
        assert SettingsStore(store.root).load().permission_mode == "auto"
        assert "Permissions: auto" in app.conversation_entries[0].content
        assert "Auto" in str(app.query_one("#status_dashboard", Static).render())

        await app._command_permissions("ask")

        assert app.permission_mode == PermissionMode.ASK
        assert app.agent is not None
        assert app.agent.permission_mode == PermissionMode.ASK
        assert store.load(session.session_id).permission_mode == "ask"
        assert SettingsStore(store.root).load().permission_mode == "ask"


@pytest.mark.asyncio
async def test_footer_renders_ctrl_p_permissions_exactly_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression: "Ctrl+P Permissions" must appear once, not twice, in the footer.

    Textual's command palette defaults to ctrl+p, which collided with the
    toggle_permission_mode binding and rendered "Ctrl+P Permissions" twice.
    Disabling the command palette on KolegaCodeApp resolves the collision.
    """
    pytest.importorskip("textual")

    from textual.widgets import Footer
    from textual.widgets._footer import FooterKey

    from kolega_code.cli.app import KolegaCodeApp

    # The command palette (default ctrl+p) is disabled to avoid the collision.
    assert KolegaCodeApp.ENABLE_COMMAND_PALETTE is False

    class FakeCoderAgent(_FakeCoderAgentBase):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.permission_mode = kwargs["permission_mode"]

        def set_permission_mode(self, permission_mode):
            self.permission_mode = permission_mode

        def set_permission_callback(self, permission_callback):
            self.permission_callback = permission_callback

    monkeypatch.setattr(agent_runtime_module, "CoderAgent", FakeCoderAgent)

    project = tmp_path / "project"
    project.mkdir()
    config = build_test_config(project)
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", config_summary(config))
    app = KolegaCodeApp(project_path=project, config=config, mode="code", store=store, session=session)

    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.pause()

        footer = app.query_one(Footer)
        ctrl_p_keys = [key for key in footer.query(FooterKey) if key.key == "ctrl+p"]

        assert len(ctrl_p_keys) == 1
        assert ctrl_p_keys[0].key_display == "Ctrl+P"
        assert ctrl_p_keys[0].description == "Permissions"


def test_app_ctrl_bindings_use_explicit_key_display() -> None:
    """Every Ctrl binding must set a ``Ctrl+X`` key_display.

    Without an explicit key_display, Textual falls back to caret notation
    (e.g. ctrl+q renders as "^q"), which is inconsistent with the other
    Ctrl bindings. This guards against that regression at the source.
    """
    from textual.binding import Binding

    from kolega_code.cli.app import KolegaCodeApp

    ctrl_bindings = [b for b in KolegaCodeApp.BINDINGS if isinstance(b, Binding) and b.key.startswith("ctrl+")]
    assert ctrl_bindings, "expected at least one ctrl binding"

    for binding in ctrl_bindings:
        expected = "Ctrl+" + binding.key.split("+", 1)[1].upper()
        assert binding.key_display == expected, (
            f"binding {binding.key!r} ({binding.action}) should display as {expected!r}, got {binding.key_display!r}"
        )


@pytest.mark.asyncio
async def test_textual_app_ctrl_o_toggles_sidebar_and_keeps_active_tab(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from textual.widgets import TabbedContent

    from textual.binding import Binding

    from kolega_code.cli.app import KolegaCodeApp

    install_fake_agents(monkeypatch)

    project = tmp_path / "project"
    project.mkdir()
    config = build_test_config(project)
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", config_summary(config))
    app = KolegaCodeApp(project_path=project, config=config, mode="code", store=store, session=session)

    async with app.run_test() as pilot:
        toggle_binding = next(b for b in app.BINDINGS if isinstance(b, Binding) and b.action == "toggle_sidebar")
        assert toggle_binding.key == "ctrl+o"
        assert toggle_binding.key_display == "Ctrl+O"
        assert toggle_binding.priority is True

        side_panel = app.query_one("#side_panel")
        tabs = app.query_one("#events", TabbedContent)
        tabs.active = "terminal_pane"
        await pilot.pause()

        assert app.sidebar_visible is True
        assert side_panel.display is True

        await pilot.press("ctrl+o")

        assert app.sidebar_visible is False
        assert side_panel.display is False
        assert tabs.active == "terminal_pane"

        await pilot.press("ctrl+o")

        assert app.sidebar_visible is True
        assert side_panel.display is True
        assert tabs.active == "terminal_pane"


@pytest.mark.asyncio
async def test_textual_app_blocks_mode_toggle_during_active_turn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from textual.widgets import Static

    from kolega_code.cli.app import KolegaCodeApp
    from kolega_code.cli.tui.widgets import ChatComposer
    from kolega_code.cli.messages import COMPOSER_PLACEHOLDER

    install_fake_agents(monkeypatch)

    project = tmp_path / "project"
    project.mkdir()
    config = build_test_config(project)
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", config_summary(config))
    app = KolegaCodeApp(project_path=project, config=config, mode="code", store=store, session=session)

    async with app.run_test():
        app._turn_active = True

        await app.action_toggle_interaction_mode()

        assert app.interaction_mode == "build"
        assert app.query_one("#composer", ChatComposer).placeholder == COMPOSER_PLACEHOLDER
        hint = app.query_one("#composer_hint", Static)
        assert hint.display is True
        assert "Stop the current turn before switching modes." in str(hint.render())
