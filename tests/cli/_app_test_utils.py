# ruff: noqa: F401,F811,E402
from pathlib import Path

import pytest

from kolega_code.cli.tui import agent_runtime as agent_runtime_module
from kolega_code.cli.config import build_agent_config, config_summary
from kolega_code.cli.session_store import SessionStore
from kolega_code.events import AgentEvent
from kolega_code.llm.models import Message


class MinimalFakeCoderAgent:
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


def install_fake_agents(monkeypatch: pytest.MonkeyPatch, *, coder_cls=MinimalFakeCoderAgent, planning_cls=None):
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


def _build_sub_agent_test_app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, **app_kwargs):
    pytest.importorskip("textual")

    from kolega_code.cli.app import KolegaCodeApp

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
    return KolegaCodeApp(project_path=project, config=config, mode="code", store=store, session=session, **app_kwargs)


def _sub_agent_event(
    agent_id="agent-1",
    agent_name="general-agent",
    task="inspect sessions",
    parent_tool_call_id="tc-1",
    uuid=None,
    **content,
):
    kwargs = {"uuid": uuid} if uuid is not None else {}
    return AgentEvent(
        event_type="chat_message",
        sender=agent_name,
        content=content,
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

    class FakeCoderAgent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.history = []
            self.messages = []
            self.attachments = []

        def append_user_message(self, content):
            self.history.append(Message(role="user", content=content))

        def restore_message_history(self, history):
            self.history = [Message.from_dict(item) for item in history]

        def dump_compaction_state(self):
            return {}

        def restore_compaction_state(self, data):
            pass

        def dump_message_history(self):
            return [message.to_dict() for message in self.history]

        async def cleanup(self):
            return None

        async def process_message_stream(self, message, attachments=None):
            self.messages.append(message)
            self.attachments.append(attachments)
            yield {"type": "response", "content": "done", "complete": True, "uuid": "response-1"}

    monkeypatch.setattr(agent_runtime_module, "CoderAgent", FakeCoderAgent)

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
