from pathlib import Path
import asyncio

import pytest

from kolega_code.agent.config import ModelProvider
from kolega_code.agent.llm.models import Message, TextBlock, ToolCall, ToolResult
from kolega_code.agent.models.public import AgentEvent
from kolega_code.agent.prompt_provider import AgentMode
from kolega_code.cli.config import build_agent_config, config_summary
from kolega_code.cli.provider_registry import DEEPSEEK_DEFAULT_MODEL, UI_DEFAULT_MODEL, UI_DEFAULT_PROVIDER
from kolega_code.cli.session_store import SessionStore
from kolega_code.cli.settings import CliSettings, SettingsStore


@pytest.mark.asyncio
async def test_textual_app_mounts_with_fake_agent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli import app as app_module
    from kolega_code.cli.app import KolegaCodeApp

    class FakeCoderAgent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.history_restored = False

        def restore_message_history(self, history):
            self.history_restored = bool(history)

        def dump_message_history(self):
            return []

        async def cleanup(self):
            return None

    monkeypatch.setattr(app_module, "CoderAgent", FakeCoderAgent)

    project = tmp_path / "project"
    project.mkdir()
    config = build_agent_config(project, env={"ANTHROPIC_API_KEY": "test-key"})
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", config_summary(config))

    app = KolegaCodeApp(
        project_path=project,
        config=config,
        mode="code",
        store=store,
        session=session,
    )

    async with app.run_test():
        assert isinstance(app.agent, FakeCoderAgent)
        assert app.mode == AgentMode.CLI.value
        assert app.session.mode == AgentMode.CLI.value
        assert app.agent.kwargs["agent_mode"] == AgentMode.CLI
        assert app.query_one("#conversation") is not None
        assert app.query_one("#composer") is not None
        assert app.conversation_entries[0].kind == "startup"
        startup = app.conversation_entries[0].content
        assert "____          _" in startup
        assert f"Project: {project}" in startup
        assert f"Session: {session.session_id[:8]}" in startup
        assert "Mode: cli" in startup
        expected_model = f"{config.long_context_config.provider.value}/{config.long_context_config.model}"
        assert f"Model: {expected_model}" in startup


@pytest.mark.asyncio
async def test_textual_app_does_not_save_startup_entry_to_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli import app as app_module
    from kolega_code.cli.app import KolegaCodeApp

    saved_history = [Message(role="assistant", content=[TextBlock("saved response")]).to_dict()]

    class FakeCoderAgent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def restore_message_history(self, history):
            return None

        def dump_message_history(self):
            return saved_history

        async def cleanup(self):
            return None

    monkeypatch.setattr(app_module, "CoderAgent", FakeCoderAgent)

    project = tmp_path / "project"
    project.mkdir()
    config = build_agent_config(project, env={"ANTHROPIC_API_KEY": "test-key"})
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", config_summary(config))
    app = KolegaCodeApp(project_path=project, config=config, mode="code", store=store, session=session)

    async with app.run_test():
        assert app.conversation_entries[0].kind == "startup"
        app._save_session_history()

        assert session.history == saved_history
        assert all("Kolega Code" not in str(item) for item in session.history)


@pytest.mark.asyncio
async def test_textual_app_mounts_settings_without_api_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("textual")

    from textual.widgets import Input

    from kolega_code.cli import app as app_module
    from kolega_code.cli.app import KolegaCodeApp

    class FakeCoderAgent:
        def __init__(self, **kwargs):
            raise AssertionError("agent should not be built without a valid API key")

    monkeypatch.setattr(app_module, "CoderAgent", FakeCoderAgent)

    project = tmp_path / "project"
    project.mkdir()
    store = SessionStore(tmp_path / "state")
    settings_store = SettingsStore(tmp_path / "state")
    session = store.create(project, "code", {})

    app = KolegaCodeApp(
        project_path=project,
        mode="code",
        store=store,
        settings_store=settings_store,
        session=session,
    )

    async with app.run_test():
        assert app.agent is None
        assert app.query_one("#composer", Input).disabled is True
        startup = app.conversation_entries[0].content
        assert f"Model: {UI_DEFAULT_PROVIDER}/{UI_DEFAULT_MODEL}" in startup
        assert "API key: missing" in startup
        status = str(app.query_one("#settings_status").render())
        assert "Configuration incomplete" in status
        assert "MOONSHOT_API_KEY" in status


@pytest.mark.asyncio
async def test_textual_app_mounts_with_stored_kimi_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from textual.widgets import Input

    from kolega_code.cli import app as app_module
    from kolega_code.cli.app import KolegaCodeApp

    class FakeCoderAgent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def restore_message_history(self, history):
            return None

        def dump_message_history(self):
            return []

        async def cleanup(self):
            return None

    monkeypatch.setattr(app_module, "CoderAgent", FakeCoderAgent)

    project = tmp_path / "project"
    project.mkdir()
    state_dir = tmp_path / "state"
    store = SessionStore(state_dir)
    settings_store = SettingsStore(state_dir)
    settings = CliSettings(active_provider=UI_DEFAULT_PROVIDER, active_model=UI_DEFAULT_MODEL)
    settings.set_api_key(UI_DEFAULT_PROVIDER, "moonshot-key")
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
        assert isinstance(app.agent, FakeCoderAgent)
        assert app.agent.kwargs["config"].long_context_config.provider == ModelProvider.MOONSHOT
        assert app.agent.kwargs["config"].long_context_config.model == UI_DEFAULT_MODEL
        assert app.query_one("#composer", Input).disabled is False


@pytest.mark.asyncio
async def test_textual_app_mounts_with_stored_deepseek_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from textual.widgets import Input

    from kolega_code.cli import app as app_module
    from kolega_code.cli.app import KolegaCodeApp

    class FakeCoderAgent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def restore_message_history(self, history):
            return None

        def dump_message_history(self):
            return []

        async def cleanup(self):
            return None

    monkeypatch.setattr(app_module, "CoderAgent", FakeCoderAgent)

    project = tmp_path / "project"
    project.mkdir()
    state_dir = tmp_path / "state"
    store = SessionStore(state_dir)
    settings_store = SettingsStore(state_dir)
    settings = CliSettings(active_provider=ModelProvider.DEEPSEEK.value, active_model=DEEPSEEK_DEFAULT_MODEL)
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
        assert isinstance(app.agent, FakeCoderAgent)
        assert app.agent.kwargs["config"].long_context_config.provider == ModelProvider.DEEPSEEK
        assert app.agent.kwargs["config"].long_context_config.model == DEEPSEEK_DEFAULT_MODEL
        assert app.query_one("#composer", Input).disabled is False


@pytest.mark.asyncio
async def test_textual_app_saves_settings_and_builds_agent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("textual")

    from textual.widgets import Input

    from kolega_code.cli import app as app_module
    from kolega_code.cli.app import KolegaCodeApp

    class FakeCoderAgent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def restore_message_history(self, history):
            return None

        def dump_message_history(self):
            return []

        async def cleanup(self):
            return None

    monkeypatch.setattr(app_module, "CoderAgent", FakeCoderAgent)

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
        assert app.agent is None
        app.query_one("#api_key_input", Input).value = "moonshot-key"
        await app._save_settings_from_ui()

        assert isinstance(app.agent, FakeCoderAgent)
        assert app.agent.kwargs["config"].long_context_config.provider == ModelProvider.MOONSHOT
        assert settings_store.load().get_api_key(UI_DEFAULT_PROVIDER) == "moonshot-key"
        assert app.query_one("#composer", Input).disabled is False
        assert [entry.kind for entry in app.conversation_entries].count("startup") == 1
        startup = app.conversation_entries[0].content
        assert f"Model: {UI_DEFAULT_PROVIDER}/{UI_DEFAULT_MODEL}" in startup
        assert "API key: present in local settings" in startup


@pytest.mark.asyncio
async def test_textual_app_saves_deepseek_settings_and_builds_agent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from textual.widgets import Input, Select

    from kolega_code.cli import app as app_module
    from kolega_code.cli.app import KolegaCodeApp

    class FakeCoderAgent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def restore_message_history(self, history):
            return None

        def dump_message_history(self):
            return []

        async def cleanup(self):
            return None

    monkeypatch.setattr(app_module, "CoderAgent", FakeCoderAgent)

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
        assert app.agent is None
        app.query_one("#provider_select", Select).value = ModelProvider.DEEPSEEK.value
        model_select = app.query_one("#model_select", Select)
        model_select.set_options([("DeepSeek V4 Pro", DEEPSEEK_DEFAULT_MODEL)])
        model_select.value = DEEPSEEK_DEFAULT_MODEL
        app.query_one("#api_key_input", Input).value = "deepseek-key"
        await app._save_settings_from_ui()

        assert isinstance(app.agent, FakeCoderAgent)
        assert app.agent.kwargs["config"].long_context_config.provider == ModelProvider.DEEPSEEK
        assert app.agent.kwargs["config"].long_context_config.model == DEEPSEEK_DEFAULT_MODEL
        assert settings_store.load().get_api_key(ModelProvider.DEEPSEEK.value) == "deepseek-key"
        assert app.query_one("#composer", Input).disabled is False


@pytest.mark.asyncio
async def test_textual_app_merges_streamed_response_chunks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from textual.widgets import Input

    from kolega_code.cli import app as app_module
    from kolega_code.cli.app import COMPOSER_PLACEHOLDER, KolegaCodeApp

    chunks = [
        {"type": "response", "content": "hello ", "complete": False, "uuid": "response-1"},
        {"type": "response", "content": "world", "complete": False, "uuid": "response-1"},
        {"type": "response", "content": "", "complete": True, "uuid": "response-1"},
    ]

    class FakeCoderAgent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def restore_message_history(self, history):
            return None

        def dump_message_history(self):
            return []

        async def cleanup(self):
            return None

        async def process_message_stream(self, message):
            for chunk in chunks:
                yield chunk

    monkeypatch.setattr(app_module, "CoderAgent", FakeCoderAgent)

    project = tmp_path / "project"
    project.mkdir()
    config = build_agent_config(project, env={"ANTHROPIC_API_KEY": "test-key"})
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", config_summary(config))
    app = KolegaCodeApp(project_path=project, config=config, mode="code", store=store, session=session)

    async with app.run_test():
        await app._process_message("hi")

        assistant_entries = [entry for entry in app.conversation_entries if entry.kind == "assistant"]
        assert len(assistant_entries) == 1
        assert assistant_entries[0].content == "hello world"
        assert assistant_entries[0].complete is True
        assert [entry for entry in app.conversation_entries if entry.kind == "progress"] == []
        assert app.query_one("#composer", Input).placeholder == COMPOSER_PLACEHOLDER


@pytest.mark.asyncio
async def test_textual_app_merges_streamed_thinking_chunks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli import app as app_module
    from kolega_code.cli.app import KolegaCodeApp

    chunks = [
        {"type": "thinking", "content": "checking ", "complete": False, "uuid": "thinking-1"},
        {"type": "thinking", "content": "context", "complete": False, "uuid": "thinking-1"},
        {"type": "thinking", "content": "", "complete": True, "uuid": "thinking-1"},
        {"type": "response", "content": "done", "complete": True, "uuid": "response-1"},
    ]

    class FakeCoderAgent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def restore_message_history(self, history):
            return None

        def dump_message_history(self):
            return []

        async def cleanup(self):
            return None

        async def process_message_stream(self, message):
            for chunk in chunks:
                yield chunk

    monkeypatch.setattr(app_module, "CoderAgent", FakeCoderAgent)

    project = tmp_path / "project"
    project.mkdir()
    config = build_agent_config(project, env={"ANTHROPIC_API_KEY": "test-key"})
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", config_summary(config))
    app = KolegaCodeApp(project_path=project, config=config, mode="code", store=store, session=session)

    async with app.run_test():
        await app._process_message("hi")

        thinking_entries = [entry for entry in app.conversation_entries if entry.kind == "thinking"]
        assert len(thinking_entries) == 1
        assert thinking_entries[0].content == "checking context"
        assert thinking_entries[0].complete is True


@pytest.mark.asyncio
async def test_textual_app_formats_thinking_as_italic_chat_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli import app as app_module
    from kolega_code.cli.app import ConversationEntry, KolegaCodeApp

    class FakeCoderAgent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def restore_message_history(self, history):
            return None

        def dump_message_history(self):
            return []

        async def cleanup(self):
            return None

    monkeypatch.setattr(app_module, "CoderAgent", FakeCoderAgent)

    project = tmp_path / "project"
    project.mkdir()
    config = build_agent_config(project, env={"ANTHROPIC_API_KEY": "test-key"})
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", config_summary(config))
    app = KolegaCodeApp(project_path=project, config=config, mode="code", store=store, session=session)

    async with app.run_test():
        formatted = app._format_conversation_entry(
            ConversationEntry(kind="thinking", content="inspect [red]markup[/red]", complete=False)
        )

        assert formatted.startswith("[dim italic]Thinking[/dim italic]\n[italic]")
        assert "\\[red]" in formatted
        assert "[/italic]" in formatted
        assert formatted.endswith("\n[dim]...[/dim]")


def test_textual_app_separates_chat_entries_with_blank_lines(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli.app import ConversationEntry, KolegaCodeApp

    class FakeConversation:
        def __init__(self) -> None:
            self.cleared = False
            self.writes: list[object] = []

        def clear(self) -> None:
            self.cleared = True

        def write(self, renderable: object) -> None:
            self.writes.append(renderable)

    project = tmp_path / "project"
    project.mkdir()
    config = build_agent_config(project, env={"ANTHROPIC_API_KEY": "test-key"})
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", config_summary(config))
    app = KolegaCodeApp(project_path=project, config=config, mode="code", store=store, session=session)
    fake_conversation = FakeConversation()
    monkeypatch.setattr(KolegaCodeApp, "_conversation", property(lambda self: fake_conversation))
    app.conversation_entries = [
        ConversationEntry(kind="user", content="first"),
        ConversationEntry(kind="assistant", content="second"),
        ConversationEntry(kind="user", content="third"),
    ]

    app._render_conversation()

    assert fake_conversation.cleared is True
    assert fake_conversation.writes == [
        "[bold cyan]You[/bold cyan]\nfirst",
        "",
        "[bold magenta]Agent[/bold magenta]\nsecond",
        "",
        "[bold cyan]You[/bold cyan]\nthird",
    ]


@pytest.mark.asyncio
async def test_textual_app_formats_agent_and_tool_chat_entries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli import app as app_module
    from kolega_code.cli.app import ConversationEntry, KolegaCodeApp

    class FakeCoderAgent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def restore_message_history(self, history):
            return None

        def dump_message_history(self):
            return []

        async def cleanup(self):
            return None

    monkeypatch.setattr(app_module, "CoderAgent", FakeCoderAgent)

    project = tmp_path / "project"
    project.mkdir()
    config = build_agent_config(project, env={"ANTHROPIC_API_KEY": "test-key"})
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", config_summary(config))
    app = KolegaCodeApp(project_path=project, config=config, mode="code", store=store, session=session)

    async with app.run_test():
        assistant = app._format_conversation_entry(
            ConversationEntry(kind="assistant", content="hello", complete=False)
        )
        tool_call = app._format_conversation_entry(
            ConversationEntry(
                kind="tool_call",
                content="inspect [red]markup[/red]\nthen continue",
                tool_name="read_file",
                complete=False,
            )
        )
        tool_result = app._format_conversation_entry(
            ConversationEntry(kind="tool_result", content="completed\nok", tool_name="read_file")
        )
        tool_error = app._format_conversation_entry(
            ConversationEntry(kind="tool_error", content="Permission denied", tool_name="write_file")
        )

        assert assistant.startswith("[bold magenta]Agent[/bold magenta]")
        assert "Kolega" not in assistant
        assert "[black on yellow] TOOL [/black on yellow]" in tool_call
        assert "[dim]  │[/dim] inspect \\[red]markup\\[/red]" in tool_call
        assert "[dim]  │[/dim] then continue" in tool_call
        assert "[black on green] TOOL [/black on green]" in tool_result
        assert "[white on red] TOOL ERROR [/white on red]" in tool_error


@pytest.mark.asyncio
async def test_textual_app_ignores_empty_final_response_without_existing_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from textual.widgets import Input

    from kolega_code.cli import app as app_module
    from kolega_code.cli.app import COMPOSER_PLACEHOLDER, KolegaCodeApp

    class FakeCoderAgent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def restore_message_history(self, history):
            return None

        def dump_message_history(self):
            return []

        async def cleanup(self):
            return None

        async def process_message_stream(self, message):
            yield {"type": "response", "content": "", "complete": True, "uuid": "response-empty"}

    monkeypatch.setattr(app_module, "CoderAgent", FakeCoderAgent)

    project = tmp_path / "project"
    project.mkdir()
    config = build_agent_config(project, env={"ANTHROPIC_API_KEY": "test-key"})
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", config_summary(config))
    app = KolegaCodeApp(project_path=project, config=config, mode="code", store=store, session=session)

    async with app.run_test():
        await app._process_message("hi")

        assert [entry for entry in app.conversation_entries if entry.kind == "assistant"] == []
        assert [entry for entry in app.conversation_entries if entry.kind == "progress"] == []
        assert app.query_one("#composer", Input).placeholder == COMPOSER_PLACEHOLDER


@pytest.mark.asyncio
async def test_textual_app_shows_working_progress_during_active_turn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from textual.widgets import Input

    from kolega_code.cli import app as app_module
    from kolega_code.cli.app import COMPOSER_PLACEHOLDER, KolegaCodeApp

    started = asyncio.Event()
    release = asyncio.Event()

    class FakeCoderAgent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def restore_message_history(self, history):
            return None

        def dump_message_history(self):
            return []

        async def cleanup(self):
            return None

        async def process_message_stream(self, message):
            started.set()
            await release.wait()
            yield {"type": "response", "content": "done", "complete": True, "uuid": "response-1"}

    monkeypatch.setattr(app_module, "CoderAgent", FakeCoderAgent)

    project = tmp_path / "project"
    project.mkdir()
    config = build_agent_config(project, env={"ANTHROPIC_API_KEY": "test-key"})
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", config_summary(config))
    app = KolegaCodeApp(project_path=project, config=config, mode="code", store=store, session=session)

    async with app.run_test():
        composer = app.query_one("#composer", Input)
        task = asyncio.create_task(app._process_message("hi"))
        await started.wait()

        progress_entries = [entry for entry in app.conversation_entries if entry.kind == "progress"]
        assert progress_entries == []
        assert composer.placeholder == "Agent is working..."
        assert composer.disabled is True

        app._render_event(
            AgentEvent(event_type="status_update", sender="coder", content={"text": "Indexing workspace"})
        )
        assert composer.placeholder == "Indexing workspace"

        release.set()
        await task

        assert [entry for entry in app.conversation_entries if entry.kind == "progress"] == []
        assert composer.placeholder == COMPOSER_PLACEHOLDER
        assert composer.disabled is False


@pytest.mark.asyncio
async def test_textual_app_renders_tool_events_in_chat(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli import app as app_module
    from kolega_code.cli.app import KolegaCodeApp, TOOL_RESULT_PREVIEW_CHARS

    class FakeCoderAgent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def restore_message_history(self, history):
            return None

        def dump_message_history(self):
            return []

        async def cleanup(self):
            return None

    monkeypatch.setattr(app_module, "CoderAgent", FakeCoderAgent)

    project = tmp_path / "project"
    project.mkdir()
    config = build_agent_config(project, env={"ANTHROPIC_API_KEY": "test-key"})
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", config_summary(config))
    app = KolegaCodeApp(project_path=project, config=config, mode="code", store=store, session=session)

    async with app.run_test():
        app._render_event(
            AgentEvent(
                event_type="chat_message",
                sender="coder",
                content={
                    "message_type": "tool_call",
                    "text": "Calling read_file",
                    "tool_description": "read_file",
                    "tool_call_id": "tool-1",
                },
            )
        )
        app._render_event(
            AgentEvent(
                event_type="chat_message",
                sender="coder",
                content={
                    "message_type": "tool_result",
                    "text": "short result",
                    "tool_description": "read_file",
                    "tool_call_id": "tool-1",
                },
            )
        )
        app._render_event(
            AgentEvent(
                event_type="chat_message",
                sender="coder",
                content={
                    "message_type": "tool_error",
                    "text": "x" * (TOOL_RESULT_PREVIEW_CHARS + 10),
                    "tool_description": "read_file",
                    "tool_call_id": "tool-2",
                },
            )
        )

        tool_entries = [entry for entry in app.conversation_entries if entry.kind.startswith("tool")]
        assert [entry.kind for entry in tool_entries] == ["tool_result", "tool_error"]
        assert tool_entries[0].content == "completed\nshort result"
        assert tool_entries[0].tool_call_id == "tool-1"
        assert tool_entries[1].content.endswith("...")
        assert tool_entries[1].tool_call_id == "tool-2"
        assert len(tool_entries[1].content) == TOOL_RESULT_PREVIEW_CHARS + 3


@pytest.mark.asyncio
async def test_textual_app_updates_tool_streaming_events_in_chat(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli import app as app_module
    from kolega_code.cli.app import KolegaCodeApp

    class FakeCoderAgent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def restore_message_history(self, history):
            return None

        def dump_message_history(self):
            return []

        async def cleanup(self):
            return None

    monkeypatch.setattr(app_module, "CoderAgent", FakeCoderAgent)

    project = tmp_path / "project"
    project.mkdir()
    config = build_agent_config(project, env={"ANTHROPIC_API_KEY": "test-key"})
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", config_summary(config))
    app = KolegaCodeApp(project_path=project, config=config, mode="code", store=store, session=session)

    async with app.run_test():
        app._render_event(
            AgentEvent(
                event_type="chat_message",
                sender="coder",
                content={
                    "message_type": "tool_call",
                    "text": "Calling think_hard",
                    "tool_description": "think_hard",
                    "tool_call_id": "tool-1",
                },
            )
        )
        app._render_event(
            AgentEvent(
                event_type="tool_streaming_update",
                sender="coder",
                content={
                    "text": "partial analysis",
                    "tool_name": "think_hard",
                    "tool_call_id": "tool-1",
                    "is_complete": False,
                },
            )
        )
        app._render_event(
            AgentEvent(
                event_type="tool_streaming_update",
                sender="coder",
                content={
                    "text": "final analysis",
                    "tool_name": "think_hard",
                    "tool_call_id": "tool-1",
                    "is_complete": True,
                },
            )
        )

        tool_entries = [entry for entry in app.conversation_entries if entry.kind.startswith("tool")]
        assert len(tool_entries) == 1
        assert tool_entries[0].kind == "tool_result"
        assert tool_entries[0].content == "completed\nfinal analysis"
        assert tool_entries[0].complete is True


@pytest.mark.asyncio
async def test_textual_app_renders_queued_tool_events_during_active_turn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from textual.widgets import Input

    from kolega_code.cli import app as app_module
    from kolega_code.cli.app import COMPOSER_PLACEHOLDER, KolegaCodeApp

    started = asyncio.Event()
    release = asyncio.Event()

    class FakeCoderAgent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.connection_manager = kwargs["connection_manager"]
            self.workspace_id = kwargs["workspace_id"]
            self.thread_id = kwargs["thread_id"]

        def restore_message_history(self, history):
            return None

        def dump_message_history(self):
            return []

        async def cleanup(self):
            return None

        async def process_message_stream(self, message):
            await self.connection_manager.broadcast_event(
                AgentEvent(
                    event_type="chat_message",
                    sender="coder",
                    content={
                        "message_type": "tool_call",
                        "text": "Calling read_file",
                        "tool_description": "read_file",
                        "tool_call_id": "tool-1",
                    },
                ),
                self.workspace_id,
                self.thread_id,
            )
            started.set()
            await release.wait()
            await self.connection_manager.broadcast_event(
                AgentEvent(
                    event_type="chat_message",
                    sender="coder",
                    content={
                        "message_type": "tool_result",
                        "text": "README contents",
                        "tool_description": "read_file",
                        "tool_call_id": "tool-1",
                    },
                ),
                self.workspace_id,
                self.thread_id,
            )
            yield {"type": "response", "content": "done", "complete": True, "uuid": "response-1"}

    async def wait_for_tool_entries(app: KolegaCodeApp, count: int) -> list:
        while True:
            entries = [entry for entry in app.conversation_entries if entry.kind.startswith("tool")]
            if len(entries) >= count:
                return entries
            await asyncio.sleep(0.01)

    monkeypatch.setattr(app_module, "CoderAgent", FakeCoderAgent)

    project = tmp_path / "project"
    project.mkdir()
    config = build_agent_config(project, env={"ANTHROPIC_API_KEY": "test-key"})
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", config_summary(config))
    app = KolegaCodeApp(project_path=project, config=config, mode="code", store=store, session=session)

    async with app.run_test():
        input_widget = app.query_one("#composer", Input)
        await app.on_input_submitted(Input.Submitted(input_widget, "hi"))
        worker = app.agent_worker
        assert worker is not None
        assert worker.group == "turns"

        await started.wait()
        event_worker = next(worker for worker in app.workers if worker.name == "kolega-events")
        assert event_worker.group == "events"
        assert not event_worker.is_cancelled

        tool_entries = await asyncio.wait_for(wait_for_tool_entries(app, 1), timeout=1)
        assert tool_entries[0].kind == "tool_call"
        assert tool_entries[0].content == "Calling read_file"
        assert input_widget.placeholder == "Running read_file..."

        release.set()
        await worker.wait()

        tool_entries = [entry for entry in app.conversation_entries if entry.kind.startswith("tool")]
        assert len(tool_entries) == 1
        assert tool_entries[0].kind == "tool_result"
        assert tool_entries[0].content == "completed\nREADME contents"
        assert input_widget.placeholder == COMPOSER_PLACEHOLDER


@pytest.mark.asyncio
async def test_textual_app_late_tool_result_updates_existing_tool_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli import app as app_module
    from kolega_code.cli.app import KolegaCodeApp

    class FakeCoderAgent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def restore_message_history(self, history):
            return None

        def dump_message_history(self):
            return []

        async def cleanup(self):
            return None

    monkeypatch.setattr(app_module, "CoderAgent", FakeCoderAgent)

    project = tmp_path / "project"
    project.mkdir()
    config = build_agent_config(project, env={"ANTHROPIC_API_KEY": "test-key"})
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", config_summary(config))
    app = KolegaCodeApp(project_path=project, config=config, mode="code", store=store, session=session)

    async with app.run_test():
        app._render_event(
            AgentEvent(
                event_type="chat_message",
                sender="coder",
                content={
                    "message_type": "tool_call",
                    "text": "Calling read_file",
                    "tool_description": "read_file",
                    "tool_call_id": "tool-1",
                },
            )
        )
        app._active_progress_entry = None
        app._turn_active = False

        app._render_event(
            AgentEvent(
                event_type="chat_message",
                sender="coder",
                content={
                    "message_type": "tool_result",
                    "text": "late result",
                    "tool_description": "read_file",
                    "tool_call_id": "tool-1",
                },
            )
        )

        tool_entries = [entry for entry in app.conversation_entries if entry.kind.startswith("tool")]
        assert len(tool_entries) == 1
        assert tool_entries[0].kind == "tool_result"
        assert tool_entries[0].content == "completed\nlate result"


@pytest.mark.asyncio
async def test_textual_app_cancellation_is_visible_in_chat(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from textual.widgets import Input

    from kolega_code.cli import app as app_module
    from kolega_code.cli.app import COMPOSER_PLACEHOLDER, KolegaCodeApp

    started = asyncio.Event()

    class FakeCoderAgent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def restore_message_history(self, history):
            return None

        def dump_message_history(self):
            return []

        async def cleanup(self):
            return None

        async def process_message_stream(self, message):
            started.set()
            while True:
                await asyncio.sleep(1)
                yield {"type": "thinking", "content": "still working", "complete": False, "uuid": "thinking-1"}

    monkeypatch.setattr(app_module, "CoderAgent", FakeCoderAgent)

    project = tmp_path / "project"
    project.mkdir()
    config = build_agent_config(project, env={"ANTHROPIC_API_KEY": "test-key"})
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", config_summary(config))
    app = KolegaCodeApp(project_path=project, config=config, mode="code", store=store, session=session)

    async with app.run_test():
        composer = app.query_one("#composer", Input)
        task = asyncio.create_task(app._process_message("hi"))
        app.agent_worker = task
        await started.wait()

        app.action_cancel_generation()
        progress_entries = [entry for entry in app.conversation_entries if entry.kind == "progress"]
        assert progress_entries == []
        assert composer.placeholder == "Stop requested..."

        await task

        progress_entries = [entry for entry in app.conversation_entries if entry.kind == "progress"]
        assert len(progress_entries) == 1
        assert progress_entries[0].content == "Stopped by user"
        assert progress_entries[0].complete is True
        assert composer.placeholder == COMPOSER_PLACEHOLDER


@pytest.mark.asyncio
async def test_textual_app_renders_resumed_history_in_chat(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli import app as app_module
    from kolega_code.cli.app import KolegaCodeApp

    class FakeCoderAgent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.restored_history = None

        def restore_message_history(self, history):
            self.restored_history = history

        def dump_message_history(self):
            return self.restored_history or []

        async def cleanup(self):
            return None

    monkeypatch.setattr(app_module, "CoderAgent", FakeCoderAgent)

    project = tmp_path / "project"
    project.mkdir()
    config = build_agent_config(project, env={"ANTHROPIC_API_KEY": "test-key"})
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", config_summary(config))
    session.history = [
        Message(role="user", content=[TextBlock("Please read the README")]).to_dict(),
        Message(
            role="assistant",
            content=[
                TextBlock("I'll inspect it."),
                ToolCall(id="tool-1", name="read_file", input={"relative_path": "README.md"}),
            ],
        ).to_dict(),
        Message(
            role="user",
            content=[ToolResult(tool_use_id="tool-1", content="README contents", name="read_file", is_error=False)],
        ).to_dict(),
        Message(
            role="user",
            content=[ToolResult(tool_use_id="tool-2", content="Permission denied", name="write_file", is_error=True)],
        ).to_dict(),
        Message(role="assistant", content=[TextBlock("Done.")]).to_dict(),
    ]

    app = KolegaCodeApp(project_path=project, config=config, mode="code", store=store, session=session)

    async with app.run_test():
        assert app.agent.restored_history == session.history
        assert app.conversation_entries[0].kind == "startup"
        startup = app.conversation_entries[0].content
        expected_model = f"{config.long_context_config.provider.value}/{config.long_context_config.model}"
        assert f"Project: {project}" in startup
        assert f"Model: {expected_model}" in startup
        assert [(entry.kind, entry.content, entry.tool_name) for entry in app.conversation_entries[1:]] == [
            ("user", "Please read the README", None),
            ("assistant", "I'll inspect it.", None),
            ("tool_call", "Calling read_file", "read_file"),
            ("tool_result", "completed\nREADME contents", "read_file"),
            ("tool_error", "Permission denied", "write_file"),
            ("assistant", "Done.", None),
        ]
