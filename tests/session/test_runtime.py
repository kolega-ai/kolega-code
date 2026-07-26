"""SessionRuntime: driving a session with no user interface attached.

The point of these tests is what they do *not* import. An agent used to be
inseparable from a Textual App, so "can a session run without a UI" was
unanswerable; here it is asserted directly, including a full turn that stops for a
permission prompt and is answered over the control channel.
"""

from __future__ import annotations

import asyncio
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from kolega_code.events import AgentEvent, KnownEventType
from kolega_code.permissions import (
    PermissionDecision,
    PermissionKind,
    PermissionMode,
    PermissionRequest,
    ProjectPermissionStore,
    allow_rule_options,
)
from kolega_code.session.control import ControlChannel
from kolega_code.session.runtime import (
    SessionRuntime,
    SessionRuntimeError,
    deserialize_permission_request,
    serialize_permission_request,
)

CLIENT = "test-client"


class _FakeAgent:
    """Stands in for a real agent: records what the runtime asks of it."""

    def __init__(self) -> None:
        self.cleaned_up = False
        self.permission_mode: PermissionMode | None = None
        self.sent: list[tuple[str, Any]] = []

    async def cleanup(self) -> None:
        self.cleaned_up = True

    def set_permission_mode(self, mode: PermissionMode) -> None:
        self.permission_mode = mode

    async def process_message_stream(self, text: str, attachments: Any = None):
        self.sent.append((text, attachments))
        yield {"type": "response", "content": f"handled {text}", "complete": True, "uuid": "u1"}


def _runtime(
    tmp_path: Path,
    *,
    mode: PermissionMode = PermissionMode.ASK,
    timeout: float = 5.0,
) -> tuple[SessionRuntime, list[AgentEvent], list[str]]:
    emitted: list[AgentEvent] = []
    notices: list[str] = []

    async def emit(event: AgentEvent) -> None:
        emitted.append(event)

    channel = ControlChannel(session_id="s1", emit=emit, timeout=timeout)

    async def factory() -> _FakeAgent:
        return _FakeAgent()

    runtime = SessionRuntime(
        session_id="s1",
        project_path=tmp_path,
        control=channel,
        agent_factory=factory,
        permission_mode=mode,
        on_notice=notices.append,
    )
    return runtime, emitted, notices


def _request(command: str = "rm -rf build") -> PermissionRequest:
    return PermissionRequest(
        kind=PermissionKind.COMMAND,
        tool_name="exec_command",
        inputs={"command": command},
        command=command,
    )


def test_runtime_imports_and_runs_with_ui_toolkits_blocked(tmp_path: Path) -> None:
    """Proof of headlessness: build and run a session with textual/rich unimportable.

    A subprocess with a meta-path blocker is the only honest way to assert this.
    Checking the source for the word "textual" would pass on prose and fail on a
    comment, and checking sys.modules would be meaningless because another test
    may already have imported it.
    """
    program = """
import asyncio, sys

class Blocker:
    def find_module(self, name, path=None):
        return self if name.split(".")[0] in {"textual", "rich"} else None
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in {"textual", "rich"}:
            raise ImportError(f"{name} is blocked for this test")
        return None
    def load_module(self, name):
        raise ImportError(f"{name} is blocked for this test")

sys.meta_path.insert(0, Blocker())
for name in list(sys.modules):
    if name.split(".")[0] in {"textual", "rich"}:
        del sys.modules[name]

from pathlib import Path
from kolega_code.events import AgentEvent
from kolega_code.permissions import PermissionKind, PermissionRequest
from kolega_code.session.control import ControlChannel
from kolega_code.session.runtime import SessionRuntime

class Agent:
    async def cleanup(self): pass
    async def process_message_stream(self, text, attachments=None):
        yield {"type": "response", "content": "ok", "complete": True, "uuid": "u"}

async def emit(event): pass

async def main():
    runtime = SessionRuntime(
        session_id="s1",
        project_path=Path(sys.argv[1]),
        control=ControlChannel(session_id="s1", emit=emit, timeout=0.05),
        agent_factory=lambda: _agent(),
    )
    await runtime.start()
    chunks = [c async for c in runtime.send_message("go")]
    assert chunks[0]["content"] == "ok", chunks
    decision = await runtime.permission_callback(
        PermissionRequest(kind=PermissionKind.COMMAND, tool_name="t", inputs={}, command="ls")
    )
    assert decision.allowed is False, "no controller must deny"
    await runtime.shutdown()
    assert "textual" not in sys.modules and "rich" not in sys.modules
    print("HEADLESS OK")

async def _agent():
    return Agent()

asyncio.run(main())
"""
    result = subprocess.run(
        [sys.executable, "-c", program, str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, f"headless run failed:\n{result.stdout}\n{result.stderr}"
    assert "HEADLESS OK" in result.stdout


@pytest.mark.asyncio
async def test_lifecycle_builds_replaces_and_disposes_the_agent(tmp_path: Path) -> None:
    runtime, _, _ = _runtime(tmp_path)
    assert runtime.running is False
    with pytest.raises(SessionRuntimeError):
        runtime.send_message("too early")

    first = await runtime.start()
    assert runtime.running is True

    second = await runtime.rebuild()
    assert first.cleaned_up is True, "a rebuild must dispose of the previous agent"
    assert second is not first

    await runtime.shutdown()
    assert second.cleaned_up is True
    assert runtime.running is False


@pytest.mark.asyncio
async def test_runs_a_turn_with_no_ui_attached(tmp_path: Path) -> None:
    runtime, _, _ = _runtime(tmp_path)
    agent = await runtime.start()

    chunks = [chunk async for chunk in runtime.send_message("build it")]

    assert chunks == [{"type": "response", "content": "handled build it", "complete": True, "uuid": "u1"}]
    assert agent.sent == [("build it", None)]


@pytest.mark.asyncio
async def test_auto_mode_never_asks(tmp_path: Path) -> None:
    """Policy is decided in the runtime, so no client is disturbed needlessly."""
    runtime, emitted, _ = _runtime(tmp_path, mode=PermissionMode.AUTO)
    await runtime.start()

    decision = await runtime.permission_callback(_request())

    assert decision.allowed is True
    assert emitted == [], "auto mode must not emit a control request"


@pytest.mark.asyncio
async def test_saved_rule_is_honoured_without_asking(tmp_path: Path) -> None:
    runtime, emitted, _ = _runtime(tmp_path)
    await runtime.start()
    request = _request("ls -la")
    option = allow_rule_options(request)[0]
    ProjectPermissionStore(tmp_path).add_rule(option.rule)

    decision = await runtime.permission_callback(request)

    assert decision.allowed is True
    assert "saved rule" in decision.reason
    assert emitted == [], "a matching saved rule must not reach a client"


@pytest.mark.asyncio
async def test_full_turn_stops_for_a_permission_answered_over_the_channel(tmp_path: Path) -> None:
    """The whole point: a headless client answers a prompt and the turn proceeds."""
    runtime, emitted, _ = _runtime(tmp_path)
    await runtime.start()
    runtime.control.acquire(CLIENT)

    async def act_as_client() -> None:
        # Wait for the request to be announced, exactly as a real client would.
        for _ in range(200):
            pending = runtime.control.pending()
            if pending:
                break
            await asyncio.sleep(0.005)
        (request,) = runtime.control.pending()
        payload = request.payload["request"]
        # A client renders from the wire form, never from a Python object.
        rebuilt = deserialize_permission_request(payload)
        assert rebuilt.command == "rm -rf build"
        assert payload["summary"] == "rm -rf build"
        runtime.answer_permission(
            request.request_id,
            PermissionDecision(allowed=True, reason="Allowed once by the user."),
            client_id=CLIENT,
        )

    client = asyncio.create_task(act_as_client())
    decision = await runtime.permission_callback(_request())
    await client

    assert decision.allowed is True
    assert decision.reason == "Allowed once by the user."
    kinds = [event.event_type for event in emitted]
    assert KnownEventType.CONTROL_REQUESTED in kinds
    assert KnownEventType.CONTROL_RESOLVED in kinds


@pytest.mark.asyncio
async def test_answer_can_carry_a_rule_to_persist(tmp_path: Path) -> None:
    runtime, _, _ = _runtime(tmp_path)
    await runtime.start()
    runtime.control.acquire(CLIENT)
    request = _request("pytest -q")
    option = allow_rule_options(request)[0]

    async def act_as_client() -> None:
        for _ in range(200):
            if runtime.control.pending():
                break
            await asyncio.sleep(0.005)
        pending = runtime.control.pending()[0]
        # The offered rules travel on the wire too, so a remote client does not
        # have to reimplement how they are derived.
        assert pending.payload["rule_options"][0]["label"] == option.label
        runtime.answer_permission(
            pending.request_id,
            PermissionDecision(allowed=True, reason="Allowed by a saved rule.", rule=option.rule),
            client_id=CLIENT,
        )

    client = asyncio.create_task(act_as_client())
    decision = await runtime.permission_callback(request)
    await client

    assert decision.rule is not None
    assert decision.rule.to_dict() == option.rule.to_dict(), "the rule must survive the round trip"


@pytest.mark.asyncio
async def test_unanswerable_prompt_denies(tmp_path: Path) -> None:
    """With no client holding control, the turn is denied rather than stranded."""
    runtime, _, _ = _runtime(tmp_path, timeout=30.0)
    await runtime.start()

    decision = await asyncio.wait_for(runtime.permission_callback(_request()), timeout=2.0)

    assert decision.allowed is False
    assert "able to answer" in decision.reason


@pytest.mark.asyncio
async def test_concurrent_requests_are_serialised(tmp_path: Path) -> None:
    """Two tool calls must not both seize the client's single approval surface."""
    runtime, _, _ = _runtime(tmp_path, timeout=30.0)
    await runtime.start()
    runtime.control.acquire(CLIENT)
    observed: list[int] = []

    async def act_as_client() -> None:
        for _ in range(4):
            for _ in range(400):
                if runtime.control.pending():
                    break
                await asyncio.sleep(0.005)
            pending = runtime.control.pending()
            observed.append(len(pending))
            if not pending:
                return
            runtime.answer_permission(
                pending[0].request_id,
                PermissionDecision(allowed=True),
                client_id=CLIENT,
            )
            await asyncio.sleep(0)

    client = asyncio.create_task(act_as_client())
    decisions = await asyncio.gather(
        runtime.permission_callback(_request("first")),
        runtime.permission_callback(_request("second")),
    )
    client.cancel()

    assert all(decision.allowed for decision in decisions)
    assert max(observed) == 1, f"more than one prompt was open at once: {observed}"


@pytest.mark.asyncio
async def test_unreadable_rule_file_asks_rather_than_guessing(tmp_path: Path) -> None:
    """A corrupt rule file must not silently allow or deny."""
    rules = tmp_path / ".kolega" / "permissions.json"
    rules.parent.mkdir(parents=True, exist_ok=True)
    rules.write_text("{not json", encoding="utf-8")
    runtime, _, notices = _runtime(tmp_path, timeout=0.05)
    await runtime.start()
    runtime.control.acquire(CLIENT)

    decision = await runtime.permission_callback(_request())

    assert decision.allowed is False, "an unreadable policy must fall back to asking, then denying"
    assert notices, "the user must be told the rule file could not be read"


@pytest.mark.asyncio
async def test_permission_mode_change_reaches_the_agent(tmp_path: Path) -> None:
    runtime, _, _ = _runtime(tmp_path)
    agent = await runtime.start()

    runtime.set_permission_mode(PermissionMode.AUTO)

    assert runtime.permission_mode == PermissionMode.AUTO
    assert agent.permission_mode == PermissionMode.AUTO


def test_permission_request_survives_serialisation() -> None:
    for request in (
        _request(),
        PermissionRequest(kind=PermissionKind.EDIT, tool_name="edit", inputs={"path": "a.py"}, path="a.py"),
        PermissionRequest(
            kind=PermissionKind.MCP,
            tool_name="mcp__srv__tool",
            inputs={},
            mcp_server="srv",
            mcp_tool="tool",
        ),
    ):
        assert deserialize_permission_request(serialize_permission_request(request)) == request
