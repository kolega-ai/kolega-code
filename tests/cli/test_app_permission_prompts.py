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
async def test_textual_app_permission_approval_actions_show_rule_labels_without_descriptions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli.app import KolegaCodeApp
    from kolega_code.cli.tui.state import PendingApproval
    from kolega_code.cli.tui.widgets import ActionList
    from kolega_code.permissions import PermissionDecision, allow_rule_options, permission_request_for_tool

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

    async with app.run_test():
        request = permission_request_for_tool("exec_command", {"command": "npm run test"})
        assert request is not None
        future: asyncio.Future[PermissionDecision] = asyncio.get_running_loop().create_future()
        app._pending_approval = PendingApproval(
            request=request,
            future=future,
            rule_options=allow_rule_options(request),
        )

        app._set_approval_actions_visible(True)

        approval_actions = app.query_one("#approval_actions", ActionList)
        # The options must be focused synchronously (no pilot.pause() above) so arrow
        # keys + Enter work without a click. A deferred Widget.focus() would not have run
        # yet here, and in a real terminal it races the refresh loop and loses focus.
        assert app.focused is approval_actions
        prompts = [
            approval_actions.get_option(f"approval_option_{index}").prompt
            for index in range(approval_actions.option_count)
        ]

        assert prompts == [
            "1. Allow once",
            "2. Deny",
            "3. Always allow this exact command",
            "4. Always allow commands starting with `npm run`",
            "5. Always allow `npm` commands",
        ]
        assert all(" — " not in str(prompt) for prompt in prompts)
        assert all("Allow commands whose" not in str(prompt) for prompt in prompts)

        app._pending_approval = None
        app._set_approval_actions_visible(False)


@pytest.mark.asyncio
async def test_textual_app_long_permission_command_keeps_approval_actions_visible_and_selectable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli.app import KolegaCodeApp
    from kolega_code.cli.tui.state import PendingApproval
    from kolega_code.cli.tui.widgets import ActionList, PromptPanel
    from kolega_code.permissions import PermissionDecision, allow_rule_options, permission_request_for_tool

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

    async with app.run_test(size=(100, 40)) as pilot:
        long_command = 'python -c "' + "print('approval layout') ; " * 80 + '"'
        request = permission_request_for_tool("exec_command", {"command": long_command})
        assert request is not None
        future: asyncio.Future[PermissionDecision] = asyncio.get_running_loop().create_future()
        app._pending_approval = PendingApproval(
            request=request,
            future=future,
            rule_options=allow_rule_options(request),
        )

        app._set_approval_actions_visible(True)
        await pilot.pause()

        approval_prompt = app.query_one("#approval_prompt", PromptPanel)
        approval_actions = app.query_one("#approval_actions", ActionList)
        assert approval_prompt.display is True
        assert approval_actions.display is True
        assert approval_actions.option_count == 5
        assert app.focused is approval_actions
        assert approval_actions.region.y + approval_actions.region.height <= (
            approval_prompt.region.y + approval_prompt.region.height
        )
        assert approval_actions.region.y + approval_actions.region.height <= app.size.height

        await pilot.press("1")
        decision = await asyncio.wait_for(future, timeout=1)

        assert decision.allowed is True
        assert app._pending_approval is None
        assert approval_actions.display is False


@pytest.mark.asyncio
async def test_textual_app_long_question_keeps_actions_visible_and_selectable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli.app import KolegaCodeApp
    from kolega_code.cli.tui.state import PendingQuestion
    from kolega_code.cli.tui.widgets import ActionList, PromptPanel

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

    async with app.run_test(size=(100, 40)) as pilot:
        long_question = "Which migration path should we use? " + "Consider all edge cases and rollout steps. " * 80
        options = ["Keep current path", "Use bounded prompt header", "Defer the decision"]
        future: asyncio.Future[str] = asyncio.get_running_loop().create_future()
        app._pending_question = PendingQuestion(
            question=long_question,
            options=options,
            future=future,
            descriptions=["Least change", "Fixes the layout", "Needs follow-up"],
        )

        app._show_question_options(long_question, options, app._pending_question.descriptions)
        await pilot.pause()

        question_prompt = app.query_one("#question_prompt", PromptPanel)
        question_actions = app.query_one("#question_actions", ActionList)
        assert question_prompt.display is True
        assert question_actions.display is True
        assert question_actions.option_count == 3
        assert app.focused is question_actions
        assert question_actions.region.y + question_actions.region.height <= (
            question_prompt.region.y + question_prompt.region.height
        )
        assert question_actions.region.y + question_actions.region.height <= app.size.height

        await pilot.press("2")
        answer = await asyncio.wait_for(future, timeout=1)

        assert answer == "Use bounded prompt header"
        assert app._pending_question is None
        assert question_actions.display is False
