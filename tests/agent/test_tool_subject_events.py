"""Tool subjects are optional, safe metadata on real agent execution events."""

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock

import pytest

from kolega_code.agent.baseagent import BaseAgent
from kolega_code.auth.tokens import ChatGPTTokenManager, OAuthTokens
from kolega_code.config import AgentConfig, CustomEndpointConfig, ModelConfig, ModelProvider
from kolega_code.events import AgentConnectionManager, AgentEvent
from kolega_code.hooks import HookOutcome
from kolega_code.llm.models import ToolCall, ToolDefinition
from kolega_code.permissions import PermissionDecision, PermissionMode
from kolega_code.security.secrets import SECRET_PLACEHOLDER
from kolega_code.tool_subjects import MAX_SUBJECT_LENGTH
from kolega_code.tools import Tool, ToolError, ToolRegistry

pytestmark = [pytest.mark.asyncio, pytest.mark.usefixtures("isolated_cli_env")]


@pytest.fixture
def agent(tmp_path: Path) -> BaseAgent:
    model = ModelConfig(provider=ModelProvider.ANTHROPIC, model="claude-haiku-4-5-20251001")
    return BaseAgent(
        project_path=tmp_path,
        workspace_id="subject-workspace",
        thread_id="subject-thread",
        connection_manager=AsyncMock(spec=AgentConnectionManager),
        config=AgentConfig(
            anthropic_api_key="fake-fixture-provider",
            long_context_config=model,
            fast_config=model,
        ),
    )


def _install(
    agent: BaseAgent,
    monkeypatch: pytest.MonkeyPatch,
    *names: str,
    handler: AsyncMock | None = None,
    exclusive: frozenset[str] = frozenset(),
) -> AsyncMock:
    handler = handler if handler is not None else AsyncMock(return_value="tool output")
    registry = ToolRegistry(
        [
            Tool(
                name=name,
                definition=ToolDefinition(name=name, description="", parameters=[]),
                handler=handler,
                parallel_safe=True,
            )
            for name in names
        ]
    )
    monkeypatch.setattr(agent, "tool_collection", SimpleNamespace(registry=lambda: registry, exclusive_tools=exclusive))
    return handler


def _call(name: str = "read", inputs: Any = None, index: int = 0, *, freeform: bool = False) -> ToolCall:
    return ToolCall(
        id=f"provider-{index}",
        execution_id=f"execution-{index}",
        name=name,
        input=inputs,
        input_kind="freeform" if freeform else "json",
    )


def _events(agent: BaseAgent) -> list[AgentEvent]:
    manager = cast(AsyncMock, agent.emitter.connection_manager)
    return [
        call.args[0] for call in manager.broadcast_event.await_args_list if call.args[0].event_type == "chat_message"
    ]


def _assert_pair(agent: BaseAgent, subject: str, terminal: str = "tool_result") -> None:
    events = _events(agent)
    assert [event.content["message_type"] for event in events] == ["tool_call", terminal]
    assert [event.content["tool_subject"] for event in events] == [subject, subject]
    assert [event.content["tool_call_id"] for event in events] == ["execution-0", "execution-0"]
    assert all("input" not in event.content and "arguments" not in event.content for event in events)


async def test_success_emits_paired_relative_path_without_changing_result(
    agent: BaseAgent, monkeypatch: pytest.MonkeyPatch
) -> None:
    handler = _install(agent, monkeypatch, "read")
    inputs = {"file_path": str(agent.project_path / "src/main.py"), "content": "not a subject"}
    result = await agent.execute_single_tool(_call(inputs=inputs))
    _assert_pair(agent, "src/main.py")
    handler.assert_awaited_once_with(**inputs)
    assert result.content == "tool output" and result.tool_use_id == "provider-0"
    assert result.execution_id == "execution-0" and not result.is_error
    assert agent.current_tool_call_id is None
    assert agent.current_tool_execution_id is None
    assert agent.current_provider_tool_call_id is None


@pytest.mark.parametrize("failure", [ToolError("expected failure"), RuntimeError("unexpected failure")])
async def test_execution_errors_keep_call_subject(
    agent: BaseAgent, monkeypatch: pytest.MonkeyPatch, failure: Exception
) -> None:
    _install(agent, monkeypatch, "read", handler=AsyncMock(side_effect=failure))
    result = await agent.execute_single_tool(_call(inputs={"file_path": "src/main.py"}))
    _assert_pair(agent, "src/main.py", "tool_error")
    assert result.is_error and result.content == str(failure)


async def test_unknown_tool_has_subject_without_extra_call(agent: BaseAgent, monkeypatch: pytest.MonkeyPatch) -> None:
    _install(agent, monkeypatch, "read")
    result = await agent.execute_single_tool(_call("not_installed", {"path": "src/missing.py"}))
    events = _events(agent)
    assert result.is_error and len(events) == 1
    assert events[0].content["message_type"] == "tool_error"
    assert events[0].content["tool_subject"] == "src/missing.py"


@pytest.mark.parametrize("decision", ["denied", "exception", "invalid"])
async def test_permission_rejection_has_subject_without_execution(
    agent: BaseAgent, monkeypatch: pytest.MonkeyPatch, decision: str
) -> None:
    handler = _install(agent, monkeypatch, "exec_command")
    agent.permission_mode = PermissionMode.ASK
    callback = AsyncMock(
        return_value=PermissionDecision(allowed=False, reason="test denial") if decision == "denied" else None,
        side_effect=RuntimeError("callback failed") if decision == "exception" else None,
    )
    monkeypatch.setattr(agent, "permission_callback", callback)
    result = await agent.execute_single_tool(_call("exec_command", {"command": "ls src"}))
    handler.assert_not_awaited()
    callback.assert_awaited_once()
    events = _events(agent)
    assert result.is_error and len(events) == 1
    assert events[0].content["message_type"] == "tool_error"
    assert events[0].content["tool_subject"] == "ls src"


@pytest.mark.parametrize("blocked", [False, True])
async def test_hook_rewrite_or_rejection_uses_actual_input(
    agent: BaseAgent, monkeypatch: pytest.MonkeyPatch, blocked: bool
) -> None:
    handler = _install(agent, monkeypatch, "read")
    pre = HookOutcome.deny("test hook denial") if blocked else HookOutcome(updated_input={"file_path": "after.py"})
    monkeypatch.setattr(agent, "fire_hook", AsyncMock(side_effect=[pre, HookOutcome.empty()]))
    original = _call(inputs={"file_path": "before.py"})
    result = await agent.execute_single_tool(original)
    assert original.input == {"file_path": "before.py"}
    if blocked:
        handler.assert_not_awaited()
        assert result.is_error
        events = _events(agent)
        assert len(events) == 1 and events[0].content["tool_subject"] == "before.py"
    else:
        handler.assert_awaited_once_with(file_path="after.py")
        _assert_pair(agent, "after.py")


async def test_exclusive_batch_rejection_builds_each_subject(agent: BaseAgent, monkeypatch: pytest.MonkeyPatch) -> None:
    handler = _install(agent, monkeypatch, "read", "control", exclusive=frozenset({"control"}))
    hook = AsyncMock()
    monkeypatch.setattr(agent, "fire_hook", hook)
    results = await agent.process_tool_calls(
        [_call("control", {"path": "first.py"}), _call(inputs={"file_path": "second.py"}, index=1)]
    )
    assert all(result.is_error for result in results)
    handler.assert_not_awaited()
    hook.assert_not_awaited()
    events = _events(agent)
    assert [event.content["message_type"] for event in events] == ["tool_error", "tool_error"]
    assert [event.content["tool_subject"] for event in events] == ["first.py", "second.py"]
    assert [event.content["tool_call_id"] for event in events] == ["execution-0", "execution-1"]


async def test_parallel_same_name_calls_do_not_share_subject(agent: BaseAgent, monkeypatch: pytest.MonkeyPatch) -> None:
    entered = asyncio.Event()
    release = asyncio.Event()

    async def read(file_path: str) -> str:
        if file_path == "first.py":
            entered.set()
            await release.wait()
        else:
            await entered.wait()
            release.set()
        return "read complete"

    _install(agent, monkeypatch, "read", handler=AsyncMock(side_effect=read))
    results = await asyncio.wait_for(
        agent.process_tool_calls(
            [_call(inputs={"file_path": "first.py"}), _call(inputs={"file_path": "second.py"}, index=1)]
        ),
        timeout=5,
    )
    assert not any(result.is_error for result in results)
    for index, path in enumerate(("first.py", "second.py")):
        events = [event for event in _events(agent) if event.content["tool_call_id"] == f"execution-{index}"]
        assert [event.content["message_type"] for event in events] == ["tool_call", "tool_result"]
        assert [event.content["tool_subject"] for event in events] == [path, path]


@pytest.mark.parametrize(
    ("name", "inputs", "freeform", "subject"),
    [
        ("read", None, False, ""),
        ("read", ["do not stringify"], False, ""),
        ("read", {"file_path": {"nested": "not a subject"}}, False, ""),
        ("read", '{"file_path": "do not parse.py"}', False, ""),
        ("eval", {"code": "print('private payload')"}, False, ""),
        ("apply_patch", "*** Begin Patch\n*** Add File: added.py\n+private payload\n*** End Patch", True, "added.py"),
        ("apply_patch", "not a patch; private payload", True, ""),
        ("exec_command", {"command": "unterminated '"}, False, ""),
    ],
)
async def test_freeform_and_invalid_inputs_do_not_leak_payloads(
    agent: BaseAgent,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    inputs: Any,
    freeform: bool,
    subject: str,
) -> None:
    _install(agent, monkeypatch, name)
    result = await agent.execute_single_tool(_call(name, inputs, freeform=freeform))
    assert not result.is_error
    assert len(_events(agent)) == 2
    for event in _events(agent):
        if subject:
            assert event.content["tool_subject"] == subject
        else:
            assert "tool_subject" not in event.content


async def test_scoped_credentials_redacted_without_global_mutation(
    agent: BaseAgent, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Deliberately fake, non-pattern keys, absent from the process environment.
    fields = [name for name in AgentConfig.model_fields if name.endswith("_api_key")]
    values = [f"fake-scoped-value-{index:02d}" for index in range(len(fields) + 9)]
    for field, value in zip(fields, values):
        setattr(agent.config, field, value)
    extras = values[len(fields) :]
    agent.config.openai_chatgpt_tokens = OAuthTokens(
        access_token=extras[0], refresh_token=extras[1], id_token=extras[2]
    )
    agent.config.attach_chatgpt_token_manager(
        ChatGPTTokenManager(OAuthTokens(access_token=extras[3], refresh_token=extras[4], id_token=extras[5]))
    )
    agent.config.custom_endpoints["fixture"] = CustomEndpointConfig(
        base_url="https://example.invalid/v1", api_style="openai_chat", api_key=extras[6]
    )
    agent.config.langfuse_public_key = extras[7]
    agent.config.langfuse_secret_key = extras[8]
    _install(agent, monkeypatch, "web_search")
    # One value per call: truncation must not make the later credential checks vacuous.
    for index, value in enumerate(values):
        await agent.execute_single_tool(_call("web_search", {"query": f"find {value}"}, index=index))
        subjects = [event.content["tool_subject"] for event in _events(agent)[-2:]]
        assert subjects == [f"find {SECRET_PLACEHOLDER}", f"find {SECRET_PLACEHOLDER}"]
    # A fresh config must not inherit another agent's redaction values.
    model = agent.config.long_context_config
    monkeypatch.setattr(
        agent,
        "config",
        AgentConfig(anthropic_api_key="fake-unrelated", long_context_config=model, fast_config=model),
    )
    assert agent._build_tool_subject("web_search", {"query": values[0]}) == values[0]


@pytest.mark.parametrize(
    ("action", "item_type", "subject"),
    [
        (
            {"type": "search", "queries": ["first query", "second query"]},
            "web_search_call",
            "first query, second query",
        ),
        (
            {"type": "open_page", "url": "https://user:pass@example.com/page?token=private#fragment"},
            "web_search_call",
            "https://example.com/page",
        ),
        (
            {"urls": ["https://example.com/a?private=yes", "https://example.org/b#private"]},
            "fetch_url_results",
            "https://example.com/a https://example.org/b",
        ),
        ({"queries": ["first query"]}, "search_results", "first query"),
        ({"queries": [{"private": "nested payload"}]}, "web_search_call", ""),
        ({}, "web_search_call", ""),
    ],
)
async def test_hosted_pairs_use_only_action_inputs(
    agent: BaseAgent, action: dict[str, Any], item_type: str, subject: str
) -> None:
    await agent._emit_hosted_tool_call(
        {"id": "hosted-0", "status": "completed", "action": action, "item_type": item_type}
    )
    events = _events(agent)
    assert [event.content["message_type"] for event in events] == ["tool_call", "tool_result"]
    assert all(event.content["tool_call_id"] == "hosted-0" for event in events)
    for event in events:
        if subject:
            assert event.content["tool_subject"] == subject
        else:
            assert "tool_subject" not in event.content


@pytest.mark.parametrize("with_input", [False, True])
async def test_streamed_write_announces_once_and_enriches_completion(
    agent: BaseAgent, monkeypatch: pytest.MonkeyPatch, with_input: bool
) -> None:
    _install(agent, monkeypatch, "write")
    delta: dict[str, Any] = {"name": "write", "id": "provider-0", "execution_id": "execution-0"}
    if with_input:
        delta["input"] = {"path": "streamed.py"}
    await agent.on_tool_use_start(delta)
    await agent.execute_single_tool(_call("write", {"path": "streamed.py", "content": "file body"}))
    events = _events(agent)
    assert [event.content["message_type"] for event in events] == ["tool_call", "tool_result"]
    assert events[1].content["tool_subject"] == "streamed.py"
    if with_input:
        assert events[0].content["tool_subject"] == "streamed.py"
    else:
        assert "tool_subject" not in events[0].content


async def test_emitter_optional_shape_and_sanitization(agent: BaseAgent) -> None:
    await agent.emitter.chat("response", "unchanged")
    assert _events(agent)[0].content == {
        "message_type": "response",
        "text": "unchanged",
        "tool_description": None,
        "tool_call_id": None,
    }
    for subject in ("", " \n\t", None):
        await agent.emitter.chat("tool_call", "unchanged", tool_subject=subject)
        assert "tool_subject" not in _events(agent)[-1].content
    await agent.emitter.chat("tool_call", "unchanged", tool_subject="\x1b[31mhello\n" + "x" * 300)
    subject = _events(agent)[-1].content["tool_subject"]
    assert len(subject) <= MAX_SUBJECT_LENGTH and "\x1b" not in subject and "\n" not in subject


async def test_wrapper_forwards_subject_and_keeps_old_call_shape(
    agent: BaseAgent, monkeypatch: pytest.MonkeyPatch
) -> None:
    chat = AsyncMock()
    monkeypatch.setattr(agent.emitter, "chat", chat)
    await agent.send_chat_message("response", "unchanged")
    assert "tool_subject" not in chat.call_args.kwargs
    await agent.send_chat_message("tool_call", "unchanged", tool_subject="src/main.py")
    assert chat.call_args.kwargs["tool_subject"] == "src/main.py"


@pytest.mark.parametrize("boundary", ["builder", "emitter"])
async def test_subject_failure_does_not_fail_tool_execution(
    agent: BaseAgent, monkeypatch: pytest.MonkeyPatch, boundary: str
) -> None:
    def fail(*args: Any, **kwargs: Any) -> str:
        raise RuntimeError("fake formatting failure")

    target = (
        "kolega_code.agent.baseagent.build_tool_subject"
        if boundary == "builder"
        else "kolega_code.events.sanitize_tool_subject"
    )
    monkeypatch.setattr(target, fail)
    handler = _install(agent, monkeypatch, "read")
    result = await agent.execute_single_tool(_call(inputs={"file_path": "src/main.py"}))
    handler.assert_awaited_once()
    assert not result.is_error and result.content == "tool output"
    assert len(_events(agent)) == 2
    assert all("tool_subject" not in event.content for event in _events(agent))


async def test_cancellation_propagates_without_extra_result(agent: BaseAgent, monkeypatch: pytest.MonkeyPatch) -> None:
    _install(agent, monkeypatch, "read", handler=AsyncMock(side_effect=asyncio.CancelledError))
    with pytest.raises(asyncio.CancelledError):
        await agent.execute_single_tool(_call(inputs={"file_path": "src/main.py"}))
    events = _events(agent)
    assert len(events) == 1 and events[0].content["tool_subject"] == "src/main.py"
    assert agent.current_tool_call_id is None


async def test_nested_eval_callback_does_not_replace_parent_subject(
    agent: BaseAgent, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def handler(**inputs: Any) -> str:
        if "code" in inputs:
            result = await agent.execute_single_tool(_call(inputs={"file_path": "nested.py"}, index=1))
            return str(result.content)
        return "nested output"

    _install(agent, monkeypatch, "eval", "read", handler=AsyncMock(side_effect=handler))
    result = await agent.execute_single_tool(_call("eval", {"code": "tool.read(file_path='nested.py')"}))
    assert not result.is_error
    events = _events(agent)
    assert [event.content["tool_call_id"] for event in events] == [
        "execution-0",
        "execution-1",
        "execution-1",
        "execution-0",
    ]
    assert "tool_subject" not in events[0].content and "tool_subject" not in events[3].content
    assert events[1].content["tool_subject"] == events[2].content["tool_subject"] == "nested.py"


async def test_subagent_metadata_survives_subject_emission(agent: BaseAgent, monkeypatch: pytest.MonkeyPatch) -> None:
    _install(agent, monkeypatch, "read")
    agent.sub_agent = True
    monkeypatch.setattr(
        agent, "sub_agent_context", {"agent_id": "child-1", "parent_tool_call_id": "parent-1", "depth": 1}
    )
    await agent.execute_single_tool(_call(inputs={"file_path": "child.py"}))
    _assert_pair(agent, "child.py")
    assert all(event.sub_agent_info == agent.sub_agent_context for event in _events(agent))
