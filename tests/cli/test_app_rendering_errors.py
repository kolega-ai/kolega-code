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
async def test_textual_app_cancellation_is_visible_in_chat(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("textual")

    from textual.widgets import Static

    from kolega_code.cli.app import KolegaCodeApp
    from kolega_code.cli.tui.widgets import ChatComposer
    from kolega_code.cli.messages import COMPOSER_PLACEHOLDER

    started = asyncio.Event()

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

        async def process_message_stream(self, message):
            started.set()
            while True:
                await asyncio.sleep(1)
                yield {"type": "thinking", "content": "still working", "complete": False, "uuid": "thinking-1"}

    monkeypatch.setattr(agent_runtime_module, "CoderAgent", FakeCoderAgent)

    project = tmp_path / "project"
    project.mkdir()
    config = build_test_config(project)
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", config_summary(config))
    app = KolegaCodeApp(project_path=project, config=config, mode="code", store=store, session=session)
    now = 10.0
    monkeypatch.setattr(app, "_now", lambda: now)

    async with app.run_test():
        composer = app.query_one("#composer", ChatComposer)
        turn_status = app.query_one("#turn_status", Static)
        task = asyncio.create_task(app._process_message("hi"))
        app.agent_worker = task
        await started.wait()

        now = 52.0
        app.action_cancel_generation()
        progress_entries = [entry for entry in app.conversation_entries if entry.kind == "progress"]
        assert progress_entries == []
        assert composer.placeholder == COMPOSER_PLACEHOLDER
        assert "Stopping…" in str(turn_status.render())
        assert "42s" in str(turn_status.render())

        await task

        progress_entries = [entry for entry in app.conversation_entries if entry.kind == "progress"]
        assert len(progress_entries) == 1
        assert progress_entries[0].content == "Stopped by user."
        assert progress_entries[0].complete is True
        assert composer.placeholder == COMPOSER_PLACEHOLDER
        assert "Stopped after 42s" in str(turn_status.render())

@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error,provider,model,expected_message",
    [
        pytest.param(
            LLMBillingError(
                "DeepSeek APIError: Insufficient Balance",
                provider=ModelProvider.DEEPSEEK.value,
            ),
            ModelProvider.DEEPSEEK,
            DEEPSEEK_DEFAULT_MODEL,
            "DeepSeek/deepseek-v4-pro could not run this request",
            id="billing",
        ),
        pytest.param(
            LLMContextWindowExceededError("context too large", provider=ModelProvider.ANTHROPIC.value),
            ModelProvider.ANTHROPIC,
            "claude-haiku-4-5-20251001",
            "The conversation context became too large for the model",
            id="context-window",
        ),
        pytest.param(
            LLMInternalServerError(
                "provider overloaded",
                provider=ModelProvider.ANTHROPIC.value,
            ),
            ModelProvider.ANTHROPIC,
            "claude-haiku-4-5-20251001",
            "There is high traffic on our LLM provider",
            id="internal-server",
        ),
        pytest.param(
            LLMAuthenticationError(
                "invalid key",
                provider=ModelProvider.ANTHROPIC.value,
            ),
            ModelProvider.ANTHROPIC,
            "claude-haiku-4-5-20251001",
            "Anthropic/claude-haiku-4-5-20251001 could not authenticate",
            id="authentication",
        ),
        pytest.param(
            LLMError(
                "unexpected provider error",
                provider=ModelProvider.ANTHROPIC.value,
            ),
            ModelProvider.ANTHROPIC,
            "claude-haiku-4-5-20251001",
            "Anthropic/claude-haiku-4-5-20251001 returned an error",
            id="generic-llm",
        ),
    ],
)
async def test_textual_app_handles_llm_error_without_worker_traceback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    error,
    provider,
    model,
    expected_message,
) -> None:
    pytest.importorskip("textual")

    from textual.widgets import Static

    from kolega_code.cli.app import KolegaCodeApp
    from kolega_code.cli.tui.state import TurnState
    from kolega_code.cli.tui.widgets import ChatComposer
    from kolega_code.cli.messages import COMPOSER_PLACEHOLDER

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

        async def process_message_stream(self, message):
            raise error
            yield {"type": "response", "content": "unreachable"}

    monkeypatch.setattr(agent_runtime_module, "CoderAgent", FakeCoderAgent)

    project = tmp_path / "project"
    project.mkdir()
    config = build_test_config(project)
    config.long_context_config.provider = provider
    config.long_context_config.model = model
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", config_summary(config))
    app = KolegaCodeApp(project_path=project, config=config, mode="code", store=store, session=session)
    monkeypatch.setattr(app, "_now", lambda: 10.0)

    async with app.run_test():
        composer = app.query_one("#composer", ChatComposer)
        turn_status = app.query_one("#turn_status", Static)

        await app._process_message("hi")

        progress_entries = [entry for entry in app.conversation_entries if entry.kind == "progress"]
        assert len(progress_entries) == 1
        assert expected_message in progress_entries[0].content
        assert progress_entries[0].tone == "error"
        assert composer.placeholder == COMPOSER_PLACEHOLDER
        assert composer.disabled is False
        assert app.agent_worker is None
        assert app._status_state.turn_state is TurnState.ERROR
        assert "Errored after" in str(turn_status.render())

@pytest.mark.asyncio
async def test_textual_app_reraises_non_llm_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli.app import KolegaCodeApp
    from kolega_code.cli.tui.state import TurnState
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

        async def process_message_stream(self, message):
            raise RuntimeError("tool host exploded")
            yield {"type": "response", "content": "unreachable"}

    monkeypatch.setattr(agent_runtime_module, "CoderAgent", FakeCoderAgent)

    project = tmp_path / "project"
    project.mkdir()
    config = build_test_config(project)
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", config_summary(config))
    app = KolegaCodeApp(project_path=project, config=config, mode="code", store=store, session=session)

    async with app.run_test():
        composer = app.query_one("#composer", ChatComposer)

        with pytest.raises(RuntimeError, match="tool host exploded"):
            await app._process_message("hi")

        progress_entries = [entry for entry in app.conversation_entries if entry.kind == "progress"]
        assert len(progress_entries) == 1
        assert progress_entries[0].content == "Stopped due to an error: tool host exploded"
        assert progress_entries[0].tone == "error"
        assert composer.disabled is False
        assert app._status_state.turn_state is TurnState.ERROR

