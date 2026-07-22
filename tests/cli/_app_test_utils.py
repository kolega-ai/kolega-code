# ruff: noqa: F401,F811,E402
import time
from pathlib import Path

import pytest

from kolega_code.cli.tui import agent_runtime as agent_runtime_module
from kolega_code.cli.config import build_agent_config, config_summary
from kolega_code.cli.session_store import SessionStore
from kolega_code.events import AgentEvent
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
        self.session_recorder = kwargs.get("session_recorder")
        self.queued_input_provider = None

    def apply_goal(self, condition, prompt_extension=None):
        self.active_goal_condition = condition

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
        },
    )


async def wait_for_onboarding_screen(app, pilot):
    """Wait until the auto-opened onboarding wizard is up AND fully composed.

    The app pushes OnboardingScreen via call_after_refresh and the screen's
    children mount a further message-pump cycle later, so a single
    pilot.pause() can lose the race on slow CI runners (NoMatches on
    #onboarding_next). Pause until the widgets exist and the screen's mount
    hook has initialized the owner's screen reference and startup status.
    """
    from kolega_code.cli.tui.onboarding_screen import OnboardingScreen

    for _ in range(20):
        await pilot.pause()
        screen = app.screen
        if (
            isinstance(screen, OnboardingScreen)
            and app._onboarding_screen is screen
            and screen.query("#onboarding_next")
        ):
            return screen
    raise AssertionError("onboarding screen did not finish mounting")


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
    **content,
):
    kwargs = {"uuid": uuid} if uuid is not None else {}
    return AgentEvent(
        event_type="chat_message",
        sender=agent_name,
        content=content,
        is_streaming=is_streaming,
        sub_agent_info={
            "agent_id": agent_id,
            "agent_name": agent_name,
            "task": task,
            "parent_tool_call_id": parent_tool_call_id,
            "conversation_id": None,
            "depth": 1,
        },
        **kwargs,
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
