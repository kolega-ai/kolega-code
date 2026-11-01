from __future__ import annotations

import asyncio
import io
import os
import pty
from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import pytest_asyncio

from kolega_code.cli import main as main_module
from kolega_code.cli.main import _AskPermissionController
from kolega_code.events import AgentEvent, KnownEventType
from kolega_code.hooks import HookEvent
from kolega_code.permissions import (
    PermissionDecision,
    PermissionKind,
    PermissionMode,
    PermissionRequest,
    PermissionRule,
    PermissionRuleOption,
    PermissionStoreError,
    ProjectPermissionStore,
)
from kolega_code.session import runtime as runtime_module
from kolega_code.session.runtime import SessionRuntime
from tests.cli._app_test_utils import FakeCoderAgent

pytestmark = pytest.mark.asyncio

REQUEST = PermissionRequest(
    kind=PermissionKind.COMMAND,
    tool_name="exec_command",
    inputs={"command": "echo hello"},
    command="echo hello",
)


class _Terminal(io.StringIO):
    def __init__(self) -> None:
        super().__init__("1\n")
        self.interactive: bool = True
        self.reads: int = 0
        self.read_error: Exception | None = None
        self.output: io.StringIO = io.StringIO()

    def isatty(self) -> bool:
        return self.interactive

    def readline(self, size: int = -1) -> str:
        self.reads += 1
        if self.read_error is not None:
            raise self.read_error
        return super().readline(size)

    def feed(self, text: str) -> None:
        self.seek(0)
        self.truncate(0)
        self.write(text)
        self.seek(0)


class _Agent:
    def __init__(self, terminal: _Terminal) -> None:
        self.terminal: _Terminal = terminal
        self.permission_mode: PermissionMode = PermissionMode.AUTO
        self.permission_callback: Callable[[PermissionRequest], Awaitable[PermissionDecision]] | None = None
        self.notifications: list[tuple[dict[str, Any], str | None, str]] = []

    def set_permission_mode(self, mode: PermissionMode) -> None:
        self.permission_mode = mode

    def set_permission_callback(self, callback: Callable[[PermissionRequest], Awaitable[PermissionDecision]]) -> None:
        self.permission_callback = callback

    async def fire_hook(self, event: HookEvent, payload: dict[str, Any], *, target: str | None = None) -> Any:
        if event == HookEvent.NOTIFICATION:
            self.notifications.append((payload, target, self.terminal.output.getvalue()))
        return SimpleNamespace(additional_context=None)


@pytest.fixture
def terminal(monkeypatch: pytest.MonkeyPatch) -> _Terminal:
    terminal = _Terminal()
    monkeypatch.setattr(main_module, "sys", SimpleNamespace(stdin=terminal, stderr=terminal.output))
    return terminal


@pytest.fixture
def agent(terminal: _Terminal) -> _Agent:
    return _Agent(terminal)


@pytest_asyncio.fixture
async def controller(tmp_path: Path, agent: _Agent) -> AsyncIterator[_AskPermissionController]:
    controller = _AskPermissionController(
        session_id="ask-test", project_path=tmp_path, permission_mode=PermissionMode.ASK
    )
    controller.runtime.adopt(agent)
    try:
        yield controller
    finally:
        await controller.close()


async def _decide(controller: _AskPermissionController, request: PermissionRequest = REQUEST) -> PermissionDecision:
    return await asyncio.wait_for(controller.runtime.permission_callback(request), timeout=10)


async def test_noninteractive_denial_has_exact_reason_and_no_prompt(
    controller: _AskPermissionController, terminal: _Terminal, agent: _Agent
) -> None:
    terminal.interactive = False
    decision = await _decide(controller)
    assert not decision.allowed
    assert decision.reason == "Permission required, but stdin is not interactive."
    assert terminal.reads == 0
    assert terminal.output.getvalue() == ""
    assert agent.notifications == []


async def test_saved_rule_is_owned_by_runtime_and_never_prompts(
    controller: _AskPermissionController, tmp_path: Path, terminal: _Terminal, agent: _Agent
) -> None:
    rule = PermissionRule.create(kind=REQUEST.kind, tool=REQUEST.tool_name, match_type="exact", pattern=REQUEST.command)
    ProjectPermissionStore(tmp_path).add_rule(rule)
    terminal.interactive = False
    decision = await _decide(controller)
    assert decision.allowed
    assert rule.id in decision.reason
    assert terminal.reads == 0
    assert terminal.output.getvalue() == ""
    assert agent.notifications == []


async def test_adopt_and_mode_changes_bind_runtime_policy(
    controller: _AskPermissionController, terminal: _Terminal, agent: _Agent
) -> None:
    assert agent.permission_mode == PermissionMode.ASK
    assert agent.permission_callback == controller.runtime.permission_callback
    controller.runtime.set_permission_mode(PermissionMode.AUTO)
    assert agent.permission_mode == PermissionMode.AUTO
    assert (await _decide(controller)).allowed
    assert terminal.reads == 0
    assert agent.notifications == []
    controller.runtime.set_permission_mode(PermissionMode.ASK)
    terminal.feed("2\n")
    assert not (await _decide(controller)).allowed
    assert agent.permission_mode == PermissionMode.ASK
    assert len(agent.notifications) == 1


async def test_only_canonical_runtime_candidates_are_rendered_and_saved(
    controller: _AskPermissionController,
    terminal: _Terminal,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    agent: _Agent,
) -> None:
    rule = PermissionRule.create(kind=REQUEST.kind, tool=REQUEST.tool_name, match_type="exact", pattern=REQUEST.command)
    option = PermissionRuleOption(label="Canonical runtime candidate", description="runtime-owned", rule=rule)
    monkeypatch.setattr(runtime_module, "allow_rule_options", lambda request: [option])
    terminal.feed("3\n")
    decision = await _decide(controller)
    assert decision.allowed
    assert decision.rule == rule
    assert ProjectPermissionStore(tmp_path).load() == [rule]
    assert "3. Canonical runtime candidate" in terminal.output.getvalue()
    assert "4." not in terminal.output.getvalue()
    assert (await _decide(controller)).allowed
    assert terminal.reads == 1
    assert len(agent.notifications) == 1


async def test_save_failure_allows_once_and_runtime_reports_warning(
    controller: _AskPermissionController,
    terminal: _Terminal,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_save(store: ProjectPermissionStore, rule: PermissionRule) -> None:
        raise PermissionStoreError("Cannot save this rule.")

    monkeypatch.setattr(ProjectPermissionStore, "add_rule", fail_save)
    terminal.feed("3\n")
    decision = await _decide(controller)
    assert decision.allowed
    assert decision.rule is None
    assert decision.reason == "Allowed once because the rule could not be saved."
    assert "Warning: Cannot save this rule." in terminal.output.getvalue()
    assert ProjectPermissionStore(tmp_path).load() == []


async def test_unreadable_rules_warn_then_present_prompt(
    controller: _AskPermissionController, terminal: _Terminal, tmp_path: Path, agent: _Agent
) -> None:
    path = ProjectPermissionStore(tmp_path).path
    path.parent.mkdir(parents=True)
    path.write_text("not json", encoding="utf-8")
    assert (await _decide(controller)).allowed
    assert "Warning: Permissions file is not valid JSON:" in terminal.output.getvalue()
    assert len(agent.notifications) == 1


@pytest.mark.parametrize(
    ("permission_request", "heading", "detail"),
    [
        (REQUEST, "run this command?", "echo hello"),
        (
            PermissionRequest(kind=PermissionKind.EDIT, tool_name="write", inputs={}, path="/project/file"),
            "run write on /project/file?",
            "/project/file",
        ),
        (
            PermissionRequest(
                kind=PermissionKind.MCP, tool_name="mcp__server__tool", inputs={}, mcp_server="server", mcp_tool="tool"
            ),
            "call this MCP tool?",
            "server: server",
        ),
        (
            PermissionRequest(kind=PermissionKind.MESSAGE, tool_name="send_message", inputs={}, command="peer: hello"),
            "send this peer message?",
            "peer: hello",
        ),
    ],
)
async def test_notification_fires_once_after_complete_prompt_is_rendered(
    controller: _AskPermissionController,
    terminal: _Terminal,
    agent: _Agent,
    permission_request: PermissionRequest,
    heading: str,
    detail: str,
) -> None:
    terminal.feed("\ninvalid\n0\n999\n1\n")
    assert (await _decide(controller, permission_request)).allowed
    assert terminal.reads == 5
    assert len(agent.notifications) == 1
    payload, target, rendered = agent.notifications[0]
    assert heading in rendered
    assert detail in rendered
    assert "1. Allow once" in rendered
    assert "2. Deny" in rendered
    assert rendered.endswith("Choose an option: ")
    assert target == "permission_prompt"
    assert payload["tool_name"] == permission_request.tool_name


@pytest.mark.parametrize("failure", ["heading", "input_prompt", "flush"])
async def test_render_failure_denies_without_notification_or_read(
    controller: _AskPermissionController,
    terminal: _Terminal,
    agent: _Agent,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    class BrokenOutput(io.StringIO):
        def write(self, text: str) -> int:
            if (failure == "heading" and "Allow the agent" in text) or (
                failure == "input_prompt" and "Choose an option" in text
            ):
                raise OSError("Cannot render.")
            return super().write(text)

        def flush(self) -> None:
            if failure == "flush":
                raise OSError("Cannot flush.")

    monkeypatch.setattr(main_module.sys, "stderr", BrokenOutput())
    decision = await _decide(controller)
    assert not decision.allowed
    assert terminal.reads == 0
    assert agent.notifications == []
    assert not controller.runtime.control.pending()


@pytest.mark.parametrize("error", [OSError("Cannot read."), EOFError()])
async def test_read_failure_denies_after_one_rendered_notification(
    controller: _AskPermissionController, terminal: _Terminal, agent: _Agent, error: Exception
) -> None:
    terminal.read_error = error
    decision = await _decide(controller)
    assert not decision.allowed
    assert terminal.reads == 1
    assert len(agent.notifications) == 1
    assert not controller.runtime.control.pending()


async def test_eof_denies_instead_of_spinning(
    controller: _AskPermissionController, terminal: _Terminal, agent: _Agent
) -> None:
    terminal.feed("")
    decision = await _decide(controller)
    assert not decision.allowed
    assert decision.reason == "Permission denied because stdin reached EOF."
    assert terminal.reads == 1
    assert terminal.output.getvalue().count("Choose an option: ") == 1
    assert len(agent.notifications) == 1


async def test_duplicate_delivery_is_not_rendered_or_notified_twice(
    controller: _AskPermissionController, terminal: _Terminal, agent: _Agent, monkeypatch: pytest.MonkeyPatch
) -> None:
    started = asyncio.Event()
    release = asyncio.Event()
    original = agent.fire_hook

    async def gated_hook(event: HookEvent, payload: dict[str, Any], *, target: str | None = None) -> Any:
        result = await original(event, payload, target=target)
        started.set()
        await release.wait()
        return result

    monkeypatch.setattr(agent, "fire_hook", gated_hook)
    invocation = asyncio.create_task(_decide(controller))
    try:
        await asyncio.wait_for(started.wait(), timeout=10)
        pending = controller.runtime.control.pending()[0]
        event = AgentEvent(
            sender="system",
            event_type=KnownEventType.CONTROL_REQUESTED,
            content={**pending.describe(), "has_controller": True},
        )
        await controller.emit(event)
        assert not await controller.runtime.prompt_delivered(pending.request_id)
        release.set()
        assert (await invocation).allowed
        assert terminal.output.getvalue().count("Allow the agent") == 1
        assert len(agent.notifications) == 1
    finally:
        release.set()
        await controller.close()
        await invocation


@pytest.mark.parametrize("action", ["close", "cancel_turn", "answer"])
async def test_cleanup_releases_control_and_joins_prompt_tasks(
    controller: _AskPermissionController, monkeypatch: pytest.MonkeyPatch, action: str
) -> None:
    reading = asyncio.Event()
    cancelled = asyncio.Event()

    async def blocked_read() -> str:
        reading.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()
        return ""

    monkeypatch.setattr(controller, "_readline", blocked_read)
    invocation = asyncio.create_task(_decide(controller))
    try:
        await asyncio.wait_for(reading.wait(), timeout=10)
        tasks = list(controller._tasks.values())
        assert len(tasks) == 1
        pending = controller.runtime.control.pending()[0]
        if action == "cancel_turn":
            invocation.cancel()
            with pytest.raises(asyncio.CancelledError):
                await invocation
        elif action == "answer":
            assert controller.runtime.answer_permission(
                pending.request_id, PermissionDecision(allowed=False), client_id=controller.client_id
            )
            await asyncio.wait_for(cancelled.wait(), timeout=10)
        await controller.close()
        if action != "cancel_turn":
            assert not (await invocation).allowed
        assert cancelled.is_set()
        assert all(task.done() for task in tasks)
        assert controller._tasks == {}
        assert controller.runtime.control.controller is None
        assert controller.runtime.control.pending() == []
        await controller.emit(
            AgentEvent(
                sender="system",
                event_type=KnownEventType.CONTROL_REQUESTED,
                content={**pending.describe(), "has_controller": True},
            )
        )
        assert controller._tasks == {}
        await controller.close()
    finally:
        await controller.close()
        if not invocation.done():
            invocation.cancel()
            with pytest.raises(asyncio.CancelledError):
                await invocation


@pytest.mark.parametrize("settle", ["timeout", "cancel"])
async def test_cancelled_pipe_read_cannot_consume_the_next_prompts_answer(
    controller: _AskPermissionController, monkeypatch: pytest.MonkeyPatch, settle: str
) -> None:
    read_fd, write_fd = os.pipe()
    started = asyncio.Queue[None]()
    stopped = asyncio.Queue[None]()
    original_read = controller._readline

    class PipeTerminal:
        encoding: str = "utf-8"
        errors: str = "strict"

        def isatty(self) -> bool:
            return True

        def fileno(self) -> int:
            return read_fd

    async def observed_read() -> str:
        started.put_nowait(None)
        try:
            return await original_read()
        finally:
            stopped.put_nowait(None)

    monkeypatch.setattr(main_module.sys, "stdin", PipeTerminal())
    monkeypatch.setattr(controller, "_readline", observed_read)
    controller.runtime.control.timeout = 0.1 if settle == "timeout" else 30
    first = asyncio.create_task(_decide(controller))
    second: asyncio.Task[PermissionDecision] | None = None
    try:
        await asyncio.wait_for(started.get(), 10)
        if settle == "cancel":
            first.cancel()
            with pytest.raises(asyncio.CancelledError):
                await first
        else:
            assert (await first).allowed is False
        await asyncio.wait_for(stopped.get(), 10)
        controller.runtime.control.timeout = 30
        second = asyncio.create_task(_decide(controller))
        await asyncio.wait_for(started.get(), 10)
        os.write(write_fd, b"1")
        await asyncio.sleep(0)
        assert not second.done(), "a partial line must not block the event loop or finish the answer"
        os.write(write_fd, b"\n")
        assert (await asyncio.wait_for(second, 10)).allowed is True
    finally:
        await controller.close()
        await asyncio.gather(first, *([second] if second is not None else []), return_exceptions=True)
        os.close(read_fd)
        os.close(write_fd)


@pytest.mark.parametrize("descriptor", ["pipe", "pty"])
async def test_back_to_back_permissions_wait_for_previous_stdin_reader_cleanup(
    controller: _AskPermissionController, monkeypatch: pytest.MonkeyPatch, descriptor: str
) -> None:
    if descriptor == "pty":
        write_fd, read_fd = pty.openpty()
        assert os.isatty(read_fd)
    else:
        read_fd, write_fd = os.pipe()
    started = asyncio.Queue[None]()
    stopped = asyncio.Queue[None]()
    original_read = controller._readline
    first_decisions: list[PermissionDecision] = []

    class DescriptorTerminal:
        encoding: str = "utf-8"
        errors: str = "strict"

        def isatty(self) -> bool:
            return True

        def fileno(self) -> int:
            return read_fd

    async def observed_read() -> str:
        started.put_nowait(None)
        try:
            return await original_read()
        finally:
            stopped.put_nowait(None)

    async def ask_twice() -> PermissionDecision:
        first_decisions.append(await controller.runtime.permission_callback(REQUEST))
        controller.runtime.control.timeout = 30
        return await controller.runtime.permission_callback(REQUEST)

    monkeypatch.setattr(main_module.sys, "stdin", DescriptorTerminal())
    monkeypatch.setattr(controller, "_readline", observed_read)
    controller.runtime.control.timeout = 0.1
    invocation = asyncio.create_task(ask_twice())
    try:
        await asyncio.wait_for(started.get(), 10)
        await asyncio.wait_for(started.get(), 10)
        await asyncio.wait_for(stopped.get(), 10)
        assert len(first_decisions) == 1
        assert first_decisions[0].allowed is False
        os.write(write_fd, b"1\n")
        assert (await asyncio.wait_for(invocation, 10)).allowed is True
    finally:
        await controller.close()
        invocation.cancel()
        await asyncio.gather(invocation, return_exceptions=True)
        os.close(read_fd)
        os.close(write_fd)


@pytest.mark.parametrize(("mode", "interactive"), [("ask", False), ("ask", True), ("auto", False), ("auto", True)])
async def test_full_ask_binds_runtime_and_cleans_up(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    isolated_cli_env: Any,
    terminal: _Terminal,
    mode: str,
    interactive: bool,
) -> None:
    instances: list[Any] = []
    terminal.interactive = interactive
    terminal.feed("3\n")

    class AskAgent(FakeCoderAgent):
        agent_name: str = "coder"

        def __init__(self, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            self.permission_mode: PermissionMode = kwargs["permission_mode"]
            self.permission_callback: Callable[[PermissionRequest], Awaitable[PermissionDecision]] = kwargs[
                "permission_callback"
            ]
            self.decisions: list[PermissionDecision] = []
            self.notifications: list[Any] = []
            instances.append(self)

        def set_permission_mode(self, mode: PermissionMode) -> None:
            self.permission_mode = mode

        def set_permission_callback(
            self, callback: Callable[[PermissionRequest], Awaitable[PermissionDecision]]
        ) -> None:
            self.permission_callback = callback

        async def fire_hook(self, event: HookEvent, payload: dict[str, Any], *, target: str | None = None) -> Any:
            if event == HookEvent.NOTIFICATION:
                self.notifications.append(payload)
            return SimpleNamespace(additional_context=None)

        async def process_message_stream(self, message: str, attachments: Any = None) -> Any:
            self.decisions.append(await self.permission_callback(REQUEST))
            self.decisions.append(await self.permission_callback(REQUEST))
            async for chunk in super().process_message_stream(message, attachments):
                yield chunk

        async def cleanup(self) -> None:
            runtime = getattr(self.permission_callback, "__self__")
            assert runtime.control.controller is None
            await super().cleanup()

    monkeypatch.setattr(main_module, "CoderAgent", AskAgent)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("KOLEGA_CODE_PROVIDER", "anthropic")
    monkeypatch.setenv("KOLEGA_CODE_MODEL", "claude-opus-5")
    args = main_module.parse_args(["ask", "work", "--project", str(tmp_path), "--permission-mode", mode])
    assert await main_module._run_ask(args) == 0
    agent = instances[0]
    runtime = agent.permission_callback.__self__
    assert isinstance(runtime, SessionRuntime)
    assert runtime.agent is None
    assert agent.permission_mode == PermissionMode(mode)
    assert len(agent.decisions) == 2
    assert all(decision.allowed == (mode == "auto" or interactive) for decision in agent.decisions)
    if mode == "ask" and not interactive:
        assert agent.decisions[0].reason == "Permission required, but stdin is not interactive."
    remembered = mode == "ask" and interactive
    assert len(agent.notifications) == int(remembered)
    if remembered:
        rule = agent.decisions[0].rule
        assert rule is not None
        assert ProjectPermissionStore(tmp_path).load() == [rule]
        assert rule.id in agent.decisions[1].reason
    assert agent.cleanup_calls == 1
    assert runtime.control.pending() == []
    assert runtime.control.controller is None
    assert terminal.reads == int(remembered)


async def test_constructor_failure_closes_permission_controller(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, isolated_cli_env: Any, terminal: _Terminal
) -> None:
    controllers: list[_AskPermissionController] = []

    class RecordingController(_AskPermissionController):
        def __init__(self, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            controllers.append(self)

    class BrokenAgent:
        agent_name: str = "coder"

        def __init__(self, **kwargs: Any) -> None:
            raise ValueError("Agent construction failed.")

    monkeypatch.setattr(main_module, "_AskPermissionController", RecordingController)
    monkeypatch.setattr(main_module, "CoderAgent", BrokenAgent)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("KOLEGA_CODE_PROVIDER", "anthropic")
    monkeypatch.setenv("KOLEGA_CODE_MODEL", "claude-opus-5")
    args = main_module.parse_args(["ask", "work", "--project", str(tmp_path), "--permission-mode", "ask"])
    with pytest.raises(ValueError, match="Agent construction failed"):
        await main_module._run_ask(args)
    assert len(controllers) == 1
    controller = controllers[0]
    assert controller.runtime.agent is None
    assert controller.runtime.control.controller is None
    assert controller.runtime.control.pending() == []
    assert controller._closed
    assert controller._tasks == {}
    assert terminal.reads == 0
