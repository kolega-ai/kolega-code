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
async def test_textual_app_startup_entry_updates_incrementally(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli.app import KolegaCodeApp

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

    monkeypatch.setattr(agent_runtime_module, "CoderAgent", FakeAgent)

    project = tmp_path / "project"
    project.mkdir()
    config = build_test_config(project)
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", config_summary(config))
    app = KolegaCodeApp(project_path=project, config=config, mode="code", store=store, session=session)

    async with app.run_test():
        startup = app.conversation_entries[0]
        original_content = startup.content
        invalidated = []
        rendered = []

        def spy_invalidate(entry):
            invalidated.append(entry)

        def spy_render():
            rendered.append(True)

        monkeypatch.setattr(app, "_invalidate_conversation", spy_invalidate)
        monkeypatch.setattr(app, "_render_conversation", spy_render)

        app.interaction_mode = "plan"
        app._ensure_startup_entry()

        assert app.conversation_entries[0] is startup
        assert startup.content != original_content
        assert "Interaction: plan" in startup.content
        assert invalidated == [startup]
        assert rendered == []

        app.conversation_entries = []
        app._ensure_startup_entry()

        assert app.conversation_entries[0].kind == "startup"
        assert rendered == [True]

@pytest.mark.asyncio
async def test_textual_app_does_not_save_startup_entry_to_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli.app import KolegaCodeApp

    saved_history = [Message(role="assistant", content=[TextBlock("saved response")]).to_dict()]

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
            return saved_history

        async def cleanup(self):
            return None

    monkeypatch.setattr(agent_runtime_module, "CoderAgent", FakeCoderAgent)

    project = tmp_path / "project"
    project.mkdir()
    config = build_test_config(project)
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", config_summary(config))
    app = KolegaCodeApp(project_path=project, config=config, mode="code", store=store, session=session)

    async with app.run_test():
        assert app.conversation_entries[0].kind == "startup"
        await app._save_session_history_async()

        assert session.history == saved_history
        assert all("Kolega Code" not in str(item) for item in session.history)

@pytest.mark.asyncio
async def test_textual_app_history_save_runs_off_event_loop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli.app import KolegaCodeApp

    saved_history = [Message(role="assistant", content=[TextBlock("saved response")]).to_dict()]

    class FakeAgent:
        def dump_message_history(self):
            return saved_history

        def dump_compaction_state(self):
            return {}

    project = tmp_path / "project"
    project.mkdir()
    config = build_test_config(project)
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", config_summary(config))
    original_save = store.save

    def slow_save(record):
        time.sleep(0.2)
        original_save(record)

    monkeypatch.setattr(store, "save", slow_save)
    app = KolegaCodeApp(project_path=project, config=config, mode="code", store=store, session=session)
    app.agent = FakeAgent()

    save_task = asyncio.create_task(app._save_session_history_async())
    marker_task = asyncio.create_task(asyncio.sleep(0.05))
    done, _ = await asyncio.wait({save_task, marker_task}, timeout=0.1, return_when=asyncio.FIRST_COMPLETED)

    assert marker_task in done
    assert not save_task.done()
    await save_task
    assert store.load(session.session_id).history == saved_history

@pytest.mark.asyncio
async def test_textual_app_history_save_persists_session_and_compaction(tmp_path: Path) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli.app import KolegaCodeApp

    saved_history = [Message(role="assistant", content=[TextBlock("persist me")]).to_dict()]
    saved_compaction = {"summary": "older turns", "compacted_through": 3, "compacted_history_length": 5}

    class FakeAgent:
        def dump_message_history(self):
            return saved_history

        def dump_compaction_state(self):
            return saved_compaction

    project = tmp_path / "project"
    project.mkdir()
    config = build_test_config(project)
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", config_summary(config))
    app = KolegaCodeApp(project_path=project, config=config, mode="code", store=store, session=session)
    app.agent = FakeAgent()

    await app._save_session_history_async()

    assert app.session.history == saved_history
    assert app.session.compaction == saved_compaction
    loaded = store.load(session.session_id)
    assert loaded.history == saved_history
    assert loaded.compaction == saved_compaction

@pytest.mark.asyncio
async def test_textual_app_overlapping_saves_preserve_later_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli.app import KolegaCodeApp

    saved_history = [Message(role="assistant", content=[TextBlock("history from first save")]).to_dict()]

    class FakeAgent:
        def dump_message_history(self):
            return saved_history

        def dump_compaction_state(self):
            return {}

    project = tmp_path / "project"
    project.mkdir()
    config = build_test_config(project)
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", config_summary(config))
    app = KolegaCodeApp(project_path=project, config=config, mode="code", store=store, session=session)
    app.agent = FakeAgent()

    original_save = store.save
    saved_snapshots: list[tuple[str, list[dict]]] = []

    def slow_first_save(record):
        saved_snapshots.append((record.task_list_markdown, list(record.history)))
        if len(saved_snapshots) == 1:
            time.sleep(0.2)
        original_save(record)

    monkeypatch.setattr(store, "save", slow_first_save)

    first_save = asyncio.create_task(app._save_session_history_async())
    await asyncio.sleep(0.05)
    app.session.task_list_markdown = "- [x] later state"
    second_save = asyncio.create_task(app._save_session_async())

    await asyncio.gather(first_save, second_save)

    assert len(saved_snapshots) == 2
    assert saved_snapshots[0] == ("", saved_history)
    assert saved_snapshots[1] == ("- [x] later state", saved_history)
    loaded = store.load(session.session_id)
    assert loaded.task_list_markdown == "- [x] later state"
    assert loaded.history == saved_history

@pytest.mark.asyncio
@pytest.mark.parametrize("command", ["/clear", "/reset"])
async def test_textual_app_reset_command_clears_current_thread(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, command: str
) -> None:
    pytest.importorskip("textual")

    from textual.widgets import Markdown

    from kolega_code.cli.app import KolegaCodeApp
    from kolega_code.cli.tui.state import PendingQuestion
    from kolega_code.cli.tui.widgets import ChatComposer
    from kolega_code.cli.messages import THREAD_RESET_MESSAGE

    class FakeCoderAgent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
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
            raise AssertionError("reset commands should not be sent to the agent")

    monkeypatch.setattr(agent_runtime_module, "CoderAgent", FakeCoderAgent)

    saved_history = [
        Message(role="user", content=[TextBlock("old request")]).to_dict(),
        Message(role="assistant", content=[TextBlock("old response")]).to_dict(),
    ]
    project = tmp_path / "project"
    project.mkdir()
    config = build_test_config(project)
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", config_summary(config))
    session.history = saved_history
    session.task_list_markdown = "- [ ] old task"
    session.latest_plan_markdown = "# Plan\n\nOld plan."
    store.save(session)

    app = KolegaCodeApp(project_path=project, config=config, mode="code", store=store, session=session)

    async with app.run_test():
        assert app.agent is not None
        assert len(app.agent.history) == 2
        assert any(entry.content == "old request" for entry in app.conversation_entries)
        app._latest_plan = "# Plan\n\nOld plan."
        app._plan_reofferable = True
        app._plan_decision_active = False
        app._set_plan_actions_visible(True)
        question_future = asyncio.get_running_loop().create_future()
        app._pending_question = PendingQuestion(
            question="Old question?",
            options=["A", "B"],
            future=question_future,
        )
        app._set_question_actions_visible(True)

        composer = app.query_one("#composer", ChatComposer)
        composer.load_text(command)
        await app.on_chat_composer_submitted(ChatComposer.Submitted(composer, composer.text))

        assert app.agent_worker is None
        assert len(app.agent.history) == 0
        assert app.session.history == []
        assert app.session.task_list_markdown == ""
        assert app._latest_plan is None
        assert app._plan_reofferable is False
        assert app._plan_decision_active is False
        assert app._pending_question is None
        assert question_future.cancelled()
        assert app.query_one("#plan_actions").display is False
        assert app.query_one("#question_actions").display is False
        assert app.query_one("#planning_plan_markdown", Markdown).source == "No plan captured yet."
        assert app.query_one("#planning_task_list_markdown", Markdown).source == "No task list has been set."
        assert store.load(session.session_id).history == []
        assert store.load(session.session_id).task_list_markdown == ""
        assert store.load(session.session_id).latest_plan_markdown == ""
        assert store.load(session.session_id).plan_reofferable is False
        assert composer.text == ""
        assert [entry.kind for entry in app.conversation_entries] == ["startup", "progress"]
        assert app.conversation_entries[-1].content == THREAD_RESET_MESSAGE
        assert all(entry.content != command for entry in app.conversation_entries)

@pytest.mark.asyncio
async def test_textual_app_reset_command_waits_for_active_turn(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("textual")

    from textual.widgets import Static

    from kolega_code.cli.app import KolegaCodeApp
    from kolega_code.cli.tui.widgets import ChatComposer
    from kolega_code.cli.messages import COMPOSER_PLACEHOLDER

    class FakeCoderAgent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.history = ["old history"]

        def restore_message_history(self, history):
            return None

        def dump_compaction_state(self):
            return {}

        def restore_compaction_state(self, data):
            pass

        def dump_message_history(self):
            return self.history

        async def cleanup(self):
            return None

    monkeypatch.setattr(agent_runtime_module, "CoderAgent", FakeCoderAgent)

    saved_history = [Message(role="user", content=[TextBlock("old request")]).to_dict()]
    project = tmp_path / "project"
    project.mkdir()
    config = build_test_config(project)
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", config_summary(config))
    session.history = saved_history
    store.save(session)

    app = KolegaCodeApp(project_path=project, config=config, mode="code", store=store, session=session)

    async with app.run_test():
        composer = app.query_one("#composer", ChatComposer)
        composer.load_text("/clear")
        app._turn_active = True

        await app.on_chat_composer_submitted(ChatComposer.Submitted(composer, composer.text))

        assert app.session.history == saved_history
        assert store.load(session.session_id).history == saved_history
        assert composer.text == "/clear"
        assert composer.placeholder == COMPOSER_PLACEHOLDER
        hint = app.query_one("#composer_hint", Static)
        assert hint.display is True
        assert "Stop the current turn before resetting the thread." in str(hint.render())

@pytest.mark.asyncio
async def test_textual_app_renders_resumed_history_in_chat(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli.app import KolegaCodeApp

    class FakeCoderAgent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.restored_history = None

        def restore_message_history(self, history):
            self.restored_history = history

        def dump_compaction_state(self):
            return {}

        def restore_compaction_state(self, data):
            pass

        def dump_message_history(self):
            return self.restored_history or []

        async def cleanup(self):
            return None

    monkeypatch.setattr(agent_runtime_module, "CoderAgent", FakeCoderAgent)

    project = tmp_path / "project"
    project.mkdir()
    config = build_test_config(project)
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", config_summary(config))
    session.history = [
        Message(role="user", content=[TextBlock("Please read the README")]).to_dict(),
        Message(
            role="assistant",
            content=[
                TextBlock("I'll inspect it."),
                ToolCall(id="tool-1", name="read_file", input={"path": "README.md"}),
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
            ("tool_result", "README contents", "read_file"),
            ("tool_error", "Permission denied", "write_file"),
            ("assistant", "Done.", None),
        ]

@pytest.mark.asyncio
async def test_textual_app_restore_tool_history_matches_legacy_and_execution_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli.app import KolegaCodeApp

    class FakeCoderAgent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.restored_history = None

        def restore_message_history(self, history):
            self.restored_history = history

        def dump_compaction_state(self):
            return {}

        def restore_compaction_state(self, data):
            pass

        def dump_message_history(self):
            return self.restored_history or []

        async def cleanup(self):
            return None

    monkeypatch.setattr(agent_runtime_module, "CoderAgent", FakeCoderAgent)

    project = tmp_path / "project"
    project.mkdir()
    config = build_test_config(project)
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", config_summary(config))
    legacy_call = {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "Legacy first."},
            {
                "type": "tool_call",
                "id": "provider-legacy",
                "name": "read_file",
                "input": {"path": "legacy.md"},
            },
        ],
        "stop_reason": "tool_use",
        "usage_metadata": {},
    }
    legacy_result = {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": "provider-legacy",
                "content": "legacy contents",
                "name": "read_file",
                "is_error": False,
                "cache_checkpoint": False,
            }
        ],
        "stop_reason": None,
        "usage_metadata": {},
    }
    execution_call = Message(
        role="assistant",
        content=[
            TextBlock("Modern first."),
            ToolCall(
                id="provider-modern",
                name="search_codebase",
                input={"pattern": "needle"},
                execution_id="tool_exec_modern",
            ),
        ],
    ).to_dict()
    execution_result = Message(
        role="user",
        content=[
            ToolResult(
                tool_use_id="provider-modern",
                content="modern contents",
                name="search_codebase",
                is_error=False,
                execution_id="tool_exec_modern",
            ),
            TextBlock("Thanks for the tool output."),
        ],
    ).to_dict()
    pending_call = Message(
        role="assistant",
        content=[ToolCall(id="provider-pending", name="list_directory", input={"path": "."})],
    ).to_dict()
    orphan_result = Message(
        role="user",
        content=[ToolResult(tool_use_id="provider-orphan", content="orphan failed", name="write_file", is_error=True)],
    ).to_dict()
    session.history = [legacy_call, legacy_result, execution_call, execution_result, pending_call, orphan_result]

    app = KolegaCodeApp(project_path=project, config=config, mode="code", store=store, session=session)

    async with app.run_test():
        restored = app.conversation_entries[1:]
        assert [(entry.kind, entry.content, entry.tool_name) for entry in restored] == [
            ("assistant", "Legacy first.", None),
            ("tool_result", "legacy contents", "read_file"),
            ("assistant", "Modern first.", None),
            ("tool_result", "modern contents", "search_codebase"),
            ("user", "Thanks for the tool output.", None),
            ("tool_call", "Calling list_directory", "list_directory"),
            ("tool_error", "orphan failed", "write_file"),
        ]
        modern_entry = next(entry for entry in restored if entry.tool_name == "search_codebase")
        assert modern_entry.tool_call_id == "tool_exec_modern"
        pending_entry = next(entry for entry in restored if entry.tool_name == "list_directory")
        assert pending_entry.complete is False

@pytest.mark.asyncio
async def test_textual_app_model_rebuild_rerenders_completed_tool_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli.app import KolegaCodeApp

    class FakeCoderAgent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.history = []
            self.compaction = {}

        def restore_message_history(self, history):
            self.history = history

        def dump_compaction_state(self):
            return self.compaction

        def restore_compaction_state(self, data):
            self.compaction = data or {}

        def dump_message_history(self):
            return self.history

        async def cleanup(self):
            return None

    monkeypatch.setattr(agent_runtime_module, "CoderAgent", FakeCoderAgent)

    project = tmp_path / "project"
    project.mkdir()
    config = build_test_config(project)
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", config_summary(config))
    app = KolegaCodeApp(project_path=project, config=config, mode="code", store=store, session=session)
    history = [
        Message(
            role="assistant",
            content=[ToolCall(id="provider-1", name="list_directory", input={"path": "."})],
        ).to_dict(),
        Message(
            role="user",
            content=[ToolResult(tool_use_id="provider-1", content="files", name="list_directory", is_error=False)],
        ).to_dict(),
    ]

    async with app.run_test():
        app.agent.history = history
        await app._build_agent(config, rebuild=True)

        tool_entries = [entry for entry in app.conversation_entries if entry.tool_name == "list_directory"]
        assert [(entry.kind, entry.content) for entry in tool_entries] == [("tool_result", "files")]
