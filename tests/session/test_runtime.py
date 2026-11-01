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
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from kolega_code.events import AgentEvent, KnownEventType
from kolega_code.hooks.events import HookEvent
from kolega_code.permissions import (
    PermissionDecision,
    PermissionKind,
    PermissionMode,
    PermissionRequest,
    PermissionRule,
    PermissionStoreError,
    ProjectPermissionStore,
    allow_rule_options,
    permission_request_for_tool,
)
from kolega_code.session.control import ControlChannel, ControlRequest
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
        self.permission_callback: Any = None
        self.sent: list[tuple[str, Any]] = []

    async def cleanup(self) -> None:
        self.cleaned_up = True

    def set_permission_mode(self, mode: PermissionMode) -> None:
        self.permission_mode = mode

    def set_permission_callback(self, callback: Any) -> None:
        self.permission_callback = callback

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
async def test_saved_message_rule_is_honoured_without_asking(tmp_path: Path) -> None:
    """A saved send_message grant dispatches without reaching a client."""
    runtime, emitted, _ = _runtime(tmp_path)
    await runtime.start()
    request = permission_request_for_tool("send_message", {"recipient": "deploy-bot", "text": "hi"})
    assert request is not None and request.kind == PermissionKind.MESSAGE
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
    offered: list[PermissionRule] = []

    async def act_as_client() -> None:
        for _ in range(200):
            if runtime.control.pending():
                break
            await asyncio.sleep(0.005)
        pending = runtime.control.pending()[0]
        # The offered rules travel on the wire too, so a remote client does not
        # have to reimplement how they are derived.
        assert pending.payload["rule_options"][0]["label"] == option.label
        offered.append(PermissionRule.from_dict(pending.payload["rule_options"][0]["rule"]))
        runtime.answer_permission(
            pending.request_id,
            PermissionDecision(allowed=True, reason="Allowed by a saved rule.", rule=offered[0]),
            client_id=CLIENT,
        )

    client = asyncio.create_task(act_as_client())
    decision = await runtime.permission_callback(request)
    await client

    assert decision.rule is not None
    assert decision.rule == offered[0], "the runtime-owned rule must survive the round trip"
    assert ProjectPermissionStore(tmp_path).first_match(request) == decision.rule


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


@pytest.mark.asyncio
async def test_planning_question_round_trips_over_the_channel(tmp_path: Path) -> None:
    """Questions use the same transport as permissions, so any client can answer."""
    runtime, emitted, _ = _runtime(tmp_path)
    await runtime.start()
    runtime.control.acquire(CLIENT)

    async def act_as_client() -> None:
        for _ in range(200):
            if runtime.control.pending():
                break
            await asyncio.sleep(0.005)
        pending = runtime.control.pending()[0]
        assert pending.kind == "question"
        assert pending.payload["options"] == ["Rewrite", "Patch"]
        runtime.answer_question(pending.request_id, {"answer": "Patch"}, client_id=CLIENT)

    client = asyncio.create_task(act_as_client())
    response = await runtime.control.request(
        "question",
        {"question": "How should we fix it?", "options": ["Rewrite", "Patch"], "descriptions": []},
        default={"answer": None},
    )
    await client

    assert response["answer"] == "Patch"
    assert KnownEventType.CONTROL_REQUESTED in [event.event_type for event in emitted]


@pytest.mark.asyncio
async def test_unanswerable_question_returns_no_answer(tmp_path: Path) -> None:
    """With nobody to ask, the caller is told rather than left waiting."""
    runtime, _, _ = _runtime(tmp_path, timeout=30.0)
    await runtime.start()

    response = await asyncio.wait_for(
        runtime.control.request("question", {"question": "?", "options": ["a", "b"]}, default={"answer": None}),
        timeout=2.0,
    )

    assert response["answer"] is None


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


class _HookAgent(_FakeAgent):
    def __init__(self, *, fail: bool = False) -> None:
        super().__init__()
        self.hooks: list[tuple[HookEvent, dict[str, Any], str]] = []
        self.fail = fail

    async def fire_hook(self, event: HookEvent, payload: dict[str, Any], *, target: str) -> None:
        self.hooks.append((event, payload, target))
        if self.fail:
            raise RuntimeError("notification hook failed")


async def _pending_request(runtime: SessionRuntime) -> ControlRequest:
    async with asyncio.timeout(5):
        while not runtime.control.pending():
            await asyncio.sleep(0)
    return runtime.control.pending()[0]


@pytest.mark.asyncio
async def test_start_adopt_and_rebuild_bind_runtime_mode_and_callback(tmp_path: Path) -> None:
    runtime, _, _ = _runtime(tmp_path, mode=PermissionMode.AUTO)
    first = await runtime.start()
    assert first.permission_mode == PermissionMode.AUTO
    assert first.permission_callback == runtime.permission_callback
    replacement = _FakeAgent()
    runtime.adopt(replacement)
    assert replacement.permission_mode == PermissionMode.AUTO
    assert replacement.permission_callback == runtime.permission_callback
    runtime.set_permission_mode(PermissionMode.ASK)
    rebuilt = await runtime.rebuild()
    assert rebuilt.permission_mode == PermissionMode.ASK
    assert rebuilt.permission_callback == runtime.permission_callback


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["permission", "question"])
async def test_notification_requires_delivery_and_acknowledges_exactly_once(tmp_path: Path, kind: str) -> None:
    runtime, _, _ = _runtime(tmp_path)
    agent = _HookAgent()
    runtime.adopt(agent)
    runtime.control.acquire(CLIENT)
    task = asyncio.create_task(
        runtime.permission_callback(_request())
        if kind == "permission"
        else runtime.control.request("question", {"question": "Which approach?"}, default={"answer": None})
    )
    request = await _pending_request(runtime)
    assert agent.hooks == [], "an event and lease do not prove human delivery"
    assert await runtime.prompt_delivered(request.request_id) is True
    assert await runtime.prompt_delivered(request.request_id) is False
    target = "permission_prompt" if kind == "permission" else "question"
    assert len(agent.hooks) == 1
    assert agent.hooks[0][0] == HookEvent.NOTIFICATION
    assert agent.hooks[0][1]["notification_type"] == target
    assert agent.hooks[0][2] == target
    runtime.control.respond(request.request_id, request.default, client_id=CLIENT)
    await task
    assert await runtime.prompt_delivered(request.request_id) is False
    assert len(agent.hooks) == 1


@pytest.mark.asyncio
async def test_delivery_resolves_current_agent_after_adopt(tmp_path: Path) -> None:
    runtime, _, _ = _runtime(tmp_path)
    old = _HookAgent()
    runtime.adopt(old)
    runtime.control.acquire(CLIENT)
    task = asyncio.create_task(runtime.permission_callback(_request()))
    pending = await _pending_request(runtime)
    current = _HookAgent()
    runtime.adopt(current)
    await runtime.prompt_delivered(pending.request_id)
    runtime.answer_permission(pending.request_id, PermissionDecision(allowed=False), client_id=CLIENT)
    await task
    assert old.hooks == []
    assert len(current.hooks) == 1


@pytest.mark.asyncio
async def test_notification_failure_is_advisory_and_not_retried(tmp_path: Path) -> None:
    runtime, _, _ = _runtime(tmp_path)
    agent = _HookAgent(fail=True)
    runtime.adopt(agent)
    runtime.control.acquire(CLIENT)
    task = asyncio.create_task(runtime.permission_callback(_request()))
    pending = await _pending_request(runtime)
    assert await runtime.prompt_delivered(pending.request_id) is True
    assert await runtime.prompt_delivered(pending.request_id) is False
    runtime.answer_permission(pending.request_id, PermissionDecision(allowed=True), client_id=CLIENT)
    assert (await task).allowed is True
    assert len(agent.hooks) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("policy", ["auto", "saved", "no_controller"])
async def test_nonhuman_policy_paths_never_notify(tmp_path: Path, policy: str) -> None:
    runtime, _, _ = _runtime(tmp_path, mode=PermissionMode.AUTO if policy == "auto" else PermissionMode.ASK)
    agent = _HookAgent()
    runtime.adopt(agent)
    if policy == "saved":
        ProjectPermissionStore(tmp_path).add_rule(allow_rule_options(_request())[0].rule)
    if policy != "no_controller":
        runtime.control.acquire(CLIENT)
    decision = await runtime.permission_callback(_request())
    assert decision.allowed is (policy != "no_controller")
    assert agent.hooks == []


@pytest.mark.asyncio
async def test_save_failure_degrades_to_allow_once_and_notifies_notice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime, _, notices = _runtime(tmp_path)
    runtime.adopt(_HookAgent())
    runtime.control.acquire(CLIENT)

    def fail_save(self: ProjectPermissionStore, rule: PermissionRule) -> None:
        raise PermissionStoreError("permission storage unavailable")

    monkeypatch.setattr(ProjectPermissionStore, "add_rule", fail_save)
    task = asyncio.create_task(runtime.permission_callback(_request()))
    pending = await _pending_request(runtime)
    rule = PermissionRule.from_dict(pending.payload["rule_options"][0]["rule"])
    runtime.answer_permission(pending.request_id, PermissionDecision(allowed=True, rule=rule), client_id=CLIENT)
    decision = await task
    assert decision.allowed is True
    assert decision.rule is None
    assert decision.reason == "Allowed once because the rule could not be saved."
    assert notices == ["permission storage unavailable"]
    assert not (tmp_path / ".kolega" / "permissions.json").exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("settle", ["timeout", "cancel", "release"])
async def test_stale_response_never_persists_or_notifies(tmp_path: Path, settle: str) -> None:
    runtime, _, _ = _runtime(tmp_path, timeout=0.02 if settle == "timeout" else 30)
    agent = _HookAgent()
    runtime.adopt(agent)
    runtime.control.acquire(CLIENT)
    task = asyncio.create_task(runtime.permission_callback(_request()))
    pending = await _pending_request(runtime)
    rule = PermissionRule.from_dict(pending.payload["rule_options"][0]["rule"])
    if settle == "cancel":
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    else:
        if settle == "release":
            runtime.control.release(CLIENT)
        assert (await task).allowed is False
    assert runtime.answer_permission(pending.request_id, PermissionDecision(allowed=True, rule=rule)) is False
    assert await runtime.prompt_delivered(pending.request_id) is False
    assert ProjectPermissionStore(tmp_path).load() == []
    assert agent.hooks == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response",
    [{"allowed": False}, {"allowed": "true"}, {"allowed": 1}, {"allowed": None}, {"allowed": True, "rule": {}}],
)
async def test_denied_and_malformed_answers_do_not_save_rules(tmp_path: Path, response: dict[str, Any]) -> None:
    runtime, _, _ = _runtime(tmp_path)
    runtime.adopt(_HookAgent())
    runtime.control.acquire(CLIENT)
    task = asyncio.create_task(runtime.permission_callback(_request()))
    pending = await _pending_request(runtime)
    if "rule" not in response:
        response = {**response, "rule": pending.payload["rule_options"][0]["rule"]}
    runtime.control.respond(pending.request_id, response, client_id=CLIENT)
    decision = await task
    assert decision.rule is None
    assert decision.allowed is (response["allowed"] is True)
    assert ProjectPermissionStore(tmp_path).load() == []


@pytest.mark.parametrize("field", ["id", "kind", "match_type", "pattern", "created_at"])
def test_response_rule_requires_parseable_metadata(tmp_path: Path, field: str) -> None:
    runtime, _, _ = _runtime(tmp_path)
    options = allow_rule_options(_request())
    rule_data = options[0].rule.to_dict()
    del rule_data[field]

    decision = runtime._decision_from_response({"allowed": True, "reason": "once", "rule": rule_data}, options)

    assert decision == PermissionDecision(allowed=True, reason="once")
    assert ProjectPermissionStore(tmp_path).load() == []


@pytest.mark.parametrize(
    ("field", "value"),
    [("kind", "edit"), ("tool", "other_tool"), ("match_type", "unoffered"), ("pattern", "*")],
)
def test_response_rule_must_match_every_candidate_field(tmp_path: Path, field: str, value: str) -> None:
    runtime, _, _ = _runtime(tmp_path)
    options = allow_rule_options(_request())
    rule_data = {**options[0].rule.to_dict(), field: value}

    decision = runtime._decision_from_response({"allowed": True, "reason": "once", "rule": rule_data}, options)

    assert decision == PermissionDecision(allowed=True, reason="once")
    assert ProjectPermissionStore(tmp_path).load() == []


def test_response_rule_selects_first_runtime_owned_candidate(tmp_path: Path) -> None:
    runtime, _, _ = _runtime(tmp_path)
    option = allow_rule_options(_request())[0]
    duplicate = replace(option, rule=replace(option.rule, id="duplicate"))
    rule_data = {**option.rule.to_dict(), "id": "supplied", "created_at": "supplied"}

    decision = runtime._decision_from_response({"allowed": True, "reason": 42, "rule": rule_data}, [option, duplicate])

    assert decision.allowed is True
    assert decision.reason == "42"
    assert decision.rule is option.rule
    assert ProjectPermissionStore(tmp_path).load() == [option.rule]


@pytest.mark.asyncio
async def test_unoffered_rule_does_not_become_a_persistent_grant(tmp_path: Path) -> None:
    runtime, _, _ = _runtime(tmp_path)
    await runtime.start()
    runtime.control.acquire(CLIENT)
    task = asyncio.create_task(runtime.permission_callback(_request("pytest")))
    pending = await _pending_request(runtime)
    unrelated = allow_rule_options(_request("git push"))[0].rule
    runtime.answer_permission(pending.request_id, PermissionDecision(allowed=True, rule=unrelated), client_id=CLIENT)
    decision = await task
    assert decision.allowed is True
    assert decision.rule is None
    assert ProjectPermissionStore(tmp_path).load() == []


@pytest.mark.asyncio
async def test_structurally_matching_response_persists_only_runtime_owned_rule_metadata(tmp_path: Path) -> None:
    runtime, _, _ = _runtime(tmp_path)
    await runtime.start()
    runtime.control.acquire(CLIENT)
    task = asyncio.create_task(runtime.permission_callback(_request()))
    pending = await _pending_request(runtime)
    offered = pending.payload["rule_options"][0]["rule"]
    response = {**offered, "id": "not-offered", "created_at": "not-a-timestamp"}
    runtime.control.respond(pending.request_id, {"allowed": True, "rule": response}, client_id=CLIENT)
    decision = await task
    assert decision.rule == PermissionRule.from_dict(offered)
    assert ProjectPermissionStore(tmp_path).load() == [PermissionRule.from_dict(offered)]


@pytest.mark.asyncio
async def test_queued_permission_checks_current_mode_after_waiting(tmp_path: Path) -> None:
    runtime, _, _ = _runtime(tmp_path)
    await runtime.start()
    runtime.control.acquire(CLIENT)
    first = asyncio.create_task(runtime.permission_callback(_request("first")))
    pending = await _pending_request(runtime)
    second = asyncio.create_task(runtime.permission_callback(_request("second")))
    await asyncio.sleep(0)
    runtime.set_permission_mode(PermissionMode.AUTO)
    runtime.answer_permission(pending.request_id, PermissionDecision(allowed=True), client_id=CLIENT)
    assert all(decision.allowed for decision in await asyncio.gather(first, second))
    assert runtime.control.pending() == []
