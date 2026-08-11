# ruff: noqa: F401,F811,E402
import asyncio
import time
from pathlib import Path
from typing import Any

import pytest

from kolega_code.cli.tui import agent_runtime as agent_runtime_module
from kolega_code.cli.config import build_agent_config, config_summary
from kolega_code.cli.session_store import SessionStore
from kolega_code.events import AgentEvent, KnownEventType
from kolega_code.llm.models import Message, TextBlock


class _FakeToolCollection:
    """Minimal stand-in satisfying app startup's ``await tool_collection.initialize()``.

    The LSP branch wires ``agent.tool_collection.initialize()`` into ``_build_agent``;
    the fake agents used by the rendering tests need this attribute so the app mounts.
    """

    lsp_manager = None

    async def initialize(self):
        return []


class FakeCoderAgent:
    """Shared stand-in for ``CoderAgent`` in TUI app tests.

    Consolidates the app-mount contract (``tool_collection.initialize``,
    message-history round-trip, compaction stubs, ``cleanup``, ``apply_goal``)
    so individual tests don't re-declare the same boilerplate. Tests needing
    custom streaming/error/goal behavior subclass this and override only the
    relevant method (typically ``process_message_stream``).
    """

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.tool_collection = _FakeToolCollection()
        self.history: list[Message] = []
        self.messages: list = []
        self.attachments: list = []
        self.active_goal_condition = None
        self.loop_active = False
        self.loop_prompt_extension = None
        self.gigacode_enabled = False
        self.gigacode_prompt_extension = None
        self.clear_history_calls = 0
        self.cleanup_calls = 0
        self.session_recorder = kwargs.get("session_recorder")
        self.queued_input_provider = None
        # Volatile-context providers registered by _build_agent (plan handle, task list).
        self.volatile_section_providers: list = []
        # Set by _build_agent when the scratchpad prompt extension is active.
        self.scratchpad_dir = None

    def apply_goal(self, condition, prompt_extension=None):
        self.active_goal_condition = condition

    def apply_loop(self, active, prompt_extension=None):
        self.loop_active = active
        self.loop_prompt_extension = prompt_extension if active else None

    def apply_gigacode(self, enabled, prompt_extension=None):
        self.gigacode_enabled = enabled
        self.gigacode_prompt_extension = prompt_extension if enabled else None

    def add_volatile_section(self, provider):
        self.volatile_section_providers.append(provider)

    def reset_volatile_context(self):
        # The fake carries no volatile-context sent-state; nothing to clear.
        pass

    def clear_history(self):
        self.clear_history_calls += 1
        self.history = []

    def set_queued_input_provider(self, provider):
        self.queued_input_provider = provider

    def append_user_message(self, content):
        self.history.append(Message(role="user", content=content))

    def restore_message_history(self, history):
        self.history = [Message.from_dict(item) for item in history]

    def dump_message_history(self):
        return [message.to_dict() for message in self.history]

    def dump_compaction_state(self):
        return {}

    def restore_compaction_state(self, data):
        pass

    async def cleanup(self):
        self.cleanup_calls += 1
        return None

    async def process_message_stream(self, message, attachments=None):
        self.messages.append(message)
        self.attachments.append(attachments)
        user_message = Message(role="user", content=[TextBlock(message)])
        assistant_message = Message(role="assistant", content=[TextBlock("done")], stop_reason="end_turn")
        if self.session_recorder is not None:
            self.session_recorder.start_turn(user_message)
            self.session_recorder.record_assistant(assistant_message)
            self.session_recorder.finish_turn("completed")
        self.history.extend([user_message, assistant_message])
        yield {"type": "response", "content": "done", "complete": True, "uuid": "response-1"}


# Backwards-compatible alias; new code should use ``FakeCoderAgent`` directly.
MinimalFakeCoderAgent = FakeCoderAgent


class _RegistryStub:
    def __init__(self, names):
        self._names = list(names)

    def names(self):
        return list(self._names)


class ToolCollectionWithRegistry(_FakeToolCollection):
    """Fake tool collection that also answers ``registry()``, so the app's
    mode-switch notice sees a real tool-name diff."""

    def __init__(self, names):
        self._names = names

    def registry(self):
        return _RegistryStub(self._names)


class FakeBuildAgentWithRegistry(FakeCoderAgent):
    tool_names = ["edit", "read", "update_task_list", "write"]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.tool_collection = ToolCollectionWithRegistry(self.tool_names)


class FakePlanAgentWithRegistry(FakeCoderAgent):
    tool_names = ["read", "write_plan"]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.tool_collection = ToolCollectionWithRegistry(self.tool_names)


def install_fake_agents(monkeypatch: pytest.MonkeyPatch, *, coder_cls=FakeCoderAgent, planning_cls=None):
    monkeypatch.setattr(agent_runtime_module, "CoderAgent", coder_cls)
    if planning_cls is not None:
        monkeypatch.setattr(agent_runtime_module, "PlanningAgent", planning_cls)


def extension_by_name(extensions, name: str):
    return next(
        extension
        for extension in extensions
        if getattr(extension, "name", None) == name or getattr(extension, "id", None) == name
    )


def question_payload(question, options, *, header="Choice", multi_select=False):
    """Build a structured `questions` list for a single question.

    options: a list of labels, or (label, description) tuples.
    """
    built = []
    for option in options:
        label, description = option if isinstance(option, tuple) else (option, "details")
        built.append({"label": label, "description": description})
    return [{"question": question, "header": header, "multiSelect": multi_select, "options": built}]


def renderable_text(renderable) -> str:
    from rich.console import Console

    console = Console(width=240, color_system=None, force_terminal=False)
    with console.capture() as capture:
        console.print(renderable, soft_wrap=True, end="")
    return capture.get()


async def wait_for_turn_idle(app, pilot, timeout: float = 6.0) -> None:
    """Wait until the submitted turn's worker has fully finished.

    A test that submits a turn must not exit ``run_test`` while the worker is
    still finalizing: app shutdown prunes the screen before workers are
    cancelled, so the worker's UI cleanup (``_set_chat_enabled`` →
    ``query_one("#composer")``) races teardown and can surface as
    ``WorkerFailed: NoMatches`` under CI load. ``agent_worker`` is cleared at
    the very end of ``_process_message``, strictly after that cleanup ran, so
    waiting for it closes the race.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if app.agent_worker is None and not app._turn_active:
            return
        await pilot.pause(0.01)
    raise AssertionError("Timed out waiting for the agent turn worker to finish")


async def wait_for_question_prompt(app, pilot, timeout: float = 6.0):
    """Wait until a pending ask_user_choice question is presented in the UI.

    Presenting a question crosses several event-loop hops (the tool task
    records the pending question, then the app populates and shows
    ``#question_actions``), so a single ``pilot.pause()`` after creating the
    tool task intermittently loses the race under CI load and asserts against
    an empty option list. Waiting for the populated, visible list closes the
    race. Returns the ``ActionList`` for convenience.
    """
    from kolega_code.cli.tui.widgets import ActionList

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if app._pending_question is not None:
            question_actions = app.query_one("#question_actions", ActionList)
            if question_actions.display and question_actions.option_count > 0:
                return question_actions
        await pilot.pause(0.01)
    raise AssertionError("Timed out waiting for the planning question to be presented")


async def wait_for_session_diff_baseline(app, timeout: float = 5.0) -> None:
    """The session-diff baseline is captured in a worker; wait for checkpoint 0.

    Needed before asserting on tracker state from the event loop. Work that
    reaches the tracker through ``_session_diff_lock`` (turn checkpoints,
    refresh, scope) serializes behind the baseline on its own.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        tracker = app._session_diff_tracker
        if tracker is not None and tracker.checkpoints():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("Timed out waiting for the session-diff baseline")


async def settle_changes_inspector(app, pilot, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not app._session_diff_refresh_running and app._session_diff_timer is None:
            await pilot.pause(0.1)
            return
        await pilot.pause(0.01)
    raise AssertionError("Timed out waiting for changes inspector refresh to settle")


def first_text_styles(renderable) -> list[str]:
    renderables = list(getattr(renderable, "renderables", [renderable]))
    text = renderables[0]
    return [str(span.style) for span in getattr(text, "spans", [])]


def build_test_config(project: Path):
    return build_agent_config(
        project,
        env={
            "ANTHROPIC_API_KEY": "test-key",
            "KOLEGA_CODE_PROVIDER": "anthropic",
            "KOLEGA_CODE_MODEL": "claude-opus-5",
        },
    )


async def wait_for_onboarding_screen(app: Any, pilot: Any, *, timeout: float = 6.0) -> Any:
    """Wait until the auto-opened onboarding wizard is initialized and laid out."""
    from textual.css.query import NoMatches
    from textual.widgets import Button, Static

    from kolega_code.cli.tui.onboarding_screen import OnboardingScreen

    deadline = time.monotonic() + timeout
    last_state: dict[str, object] = {}
    while time.monotonic() < deadline:
        screen = app._onboarding_screen
        last_state = {
            "screen_type": type(screen).__name__ if screen is not None else None,
            "active": screen is not None and app.screen is screen,
        }
        if isinstance(screen, OnboardingScreen) and app.screen is screen:
            try:
                status = screen.query_one("#onboarding_status", Static)
                key_status = screen.query_one("#onboarding_key_status", Static)
                next_button = screen.query_one("#onboarding_next", Button)
            except NoMatches as exc:
                last_state["query_error"] = repr(exc)
            else:
                status_text = str(status.render())
                key_status_text = str(key_status.render())
                startup_error = getattr(app, "startup_config_error", None)
                startup_status_ready = not startup_error or str(startup_error) in status_text
                credentials_ready = key_status_text.startswith("Credential status:")
                continue_laid_out = next_button.region.width > 0 and next_button.region.height > 0
                last_state.update(
                    {
                        "status": status_text,
                        "key_status": key_status_text,
                        "continue_region": repr(next_button.region),
                    }
                )
                if credentials_ready and startup_status_ready and continue_laid_out:
                    return screen
        await pilot.pause()
    raise AssertionError(f"onboarding screen did not become ready within {timeout:.1f}s: {last_state!r}")


async def open_settings_screen(app, pilot, category: str = "model"):
    """Open the full-screen settings editor, skipping auto-onboarding if it is up."""
    from kolega_code.cli.tui.settings_screen import SettingsScreen

    # The wizard auto-opens whenever config is None; wait for it deterministically
    # (not just one pause) before skipping, or it can land after the settings screen.
    if app.config is None and not app._onboarding_skipped:
        onboarding = await wait_for_onboarding_screen(app, pilot)
        onboarding.action_skip()
        await pilot.pause()
    else:
        await pilot.pause()
    app.action_open_settings(category)
    await pilot.pause()
    screen = app.screen
    assert isinstance(screen, SettingsScreen)
    return screen


async def stage_provider_api_key(screen, pilot, provider: str, key: str) -> None:
    """Type an API key for ``provider`` on the Providers page.

    Mirrors the real interaction: highlight the provider's row, let the editor follow,
    then type. Keys stage per provider and are only written on Apply.
    """
    from textual.widgets import Input, OptionList

    provider_list = screen.query_one("#provider_list", OptionList)
    provider_list.highlighted = provider_list.get_option_index(f"provider_row_{provider}")
    # The highlight drives credential_provider through a posted message; wait for the
    # editor to actually follow before typing, or the key lands on the previous row.
    for _ in range(50):
        await pilot.pause()
        if screen.credential_provider == provider:
            break
    screen.query_one("#provider_api_key_input", Input).value = key
    await pilot.pause()


def _build_sub_agent_test_app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, **app_kwargs):
    pytest.importorskip("textual")

    from kolega_code.cli.app import KolegaCodeApp

    install_fake_agents(monkeypatch)

    project = tmp_path / "project"
    project.mkdir()
    config = build_test_config(project)
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", config_summary(config))
    return KolegaCodeApp(project_path=project, config=config, mode="code", store=store, session=session, **app_kwargs)


def _sub_agent_event(
    agent_id="agent-1",
    agent_name="general-agent",
    task="inspect sessions",
    parent_tool_call_id="tc-1",
    uuid=None,
    is_streaming=False,
    effective_routing=None,
    requested_routing=None,
    **content,
):
    kwargs = {"uuid": uuid} if uuid is not None else {}
    sub_agent_info = {
        "agent_id": agent_id,
        "agent_name": agent_name,
        "task": task,
        "parent_tool_call_id": parent_tool_call_id,
        "conversation_id": None,
        "depth": 1,
    }
    # Opt-in so the default fixture doubles as missing-routing coverage.
    if effective_routing is not None:
        sub_agent_info["effective_routing"] = effective_routing
        sub_agent_info["requested_routing"] = requested_routing
    return AgentEvent(
        event_type="chat_message",
        sender=agent_name,
        content=content,
        is_streaming=is_streaming,
        sub_agent_info=sub_agent_info,
        **kwargs,
    )


def _sub_agent_delta(
    text,
    *,
    thinking=False,
    complete=True,
    uuid="d-1",
    agent_id="agent-1",
    agent_name="general-agent",
    task="inspect sessions",
):
    """One streamed reasoning or prose delta from a dispatched agent.

    This is the shape AgentTool leaves on the stream: the sub-agent's own
    generator is mirrored as assistant_delta/thinking_delta carrying its
    sub_agent_info, rather than re-broadcast as chat_message.
    """
    return AgentEvent(
        event_type=KnownEventType.THINKING_DELTA if thinking else KnownEventType.ASSISTANT_DELTA,
        sender=agent_name,
        content={"text": text, "complete": complete},
        is_streaming=not complete,
        uuid=uuid,
        sub_agent_info={
            "agent_id": agent_id,
            "agent_name": agent_name,
            "task": task,
            "parent_tool_call_id": "tc-1",
            "conversation_id": None,
            "depth": 1,
        },
    )


def _sub_agent_entries(app):
    return [entry for entry in app.conversation_entries if entry.kind == "sub_agent"]


def _sub_agent_context_event(usage_percentage, *, input_tokens=5000, agent_id="agent-1", agent_name="general-agent"):
    return AgentEvent(
        event_type="llm_context_update",
        sender=agent_name,
        content={
            "input_tokens": input_tokens,
            "max_tokens": 200000,
            "usage_percentage": usage_percentage,
            "alert_level": "normal",
            "message": None,
            "compression_threshold": 80.0,
        },
        sub_agent_info={
            "agent_id": agent_id,
            "agent_name": agent_name,
            "task": "inspect sessions",
            "parent_tool_call_id": "tc-1",
            "conversation_id": None,
            "depth": 1,
        },
    )


def _workflow_event(message_type, run_id="wf-1", **content):
    return AgentEvent(
        event_type="chat_message",
        sender="gigacode",
        content={
            "message_type": message_type,
            "workflow_run_id": run_id,
            "text": content.pop("text", ""),
            **content,
        },
    )


def _build_mention_test_app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from kolega_code.cli.app import KolegaCodeApp

    install_fake_agents(monkeypatch)

    project = tmp_path / "project"
    project.mkdir()
    (project / "src").mkdir()
    (project / "src" / "alpha.py").write_text("print('alpha')\n", encoding="utf-8")
    (project / "src" / "alpine.txt").write_text("mountains\n", encoding="utf-8")
    (project / "README.md").write_text("# Readme\n", encoding="utf-8")
    config = build_test_config(project)
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", config_summary(config))
    app = KolegaCodeApp(project_path=project, config=config, mode="code", store=store, session=session)
    # Pre-warm the @-mention index so cached_search is populated deterministically in tests.
    # In production the app warms it off-thread on mount; the completion dropdown reads the
    # cached snapshot only (never walks on a keystroke).
    app.file_index.refresh()
    return app
