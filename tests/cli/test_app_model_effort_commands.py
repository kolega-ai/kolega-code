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
