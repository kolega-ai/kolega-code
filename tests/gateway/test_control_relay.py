"""ControlRelay: control-channel prompts rendered as buttons and answered by taps."""

import asyncio
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable
from unittest.mock import AsyncMock

import pytest

from kolega_code.events import AgentEvent, KnownEventType
from kolega_code.gateway.adapters.base import AdapterCapabilities, ButtonOption, ChatRef, GatewayAdapter, InboundMessage
from kolega_code.gateway.control_relay import ControlRelay
from kolega_code.gateway.event_router import EventRouter
from kolega_code.hooks.events import HookEvent
from kolega_code.permissions import (
    PERMISSIONS_RELATIVE_PATH,
    PermissionDecision,
    PermissionKind,
    PermissionMode,
    PermissionRequest,
    PermissionStoreError,
    ProjectPermissionStore,
    allow_rule_options,
)
from kolega_code.session.control import ControlChannel
from kolega_code.session.plans import PLAN_APPROVAL_OPTIONS
from kolega_code.session.runtime import SessionRuntime, serialize_permission_request


class ButtonAdapter(GatewayAdapter):
    name = "buttony"

    def __init__(self) -> None:
        super().__init__()
        self.capabilities = AdapterCapabilities(supports_edits=True, supports_inline_buttons=True)
        self.buttons: list[tuple[str, str, list[ButtonOption]]] = []
        self.tokens: list[str] = []
        self.sent: list[str] = []

    async def send_text(self, chat_id: str, text: str, *, reply_to_message_id: str | None = None) -> str:
        self.sent.append(text)
        return f"text-{len(self.sent)}"

    async def send_buttons(self, chat_id: str, prompt: str, options: list[ButtonOption]) -> str:
        await asyncio.sleep(0)
        token = f"tok-{len(self.tokens) + 1}"
        self.tokens.append(token)
        self.buttons.append((chat_id, prompt, list(options)))
        return token


def make_request() -> PermissionRequest:
    return PermissionRequest(
        kind=PermissionKind.COMMAND,
        tool_name="bash",
        inputs={},
        command="ls -la",
        path="",
        mcp_server="",
        mcp_tool="",
    )


def rule_options_payload(request: PermissionRequest) -> list[dict[str, Any]]:
    return [
        {"label": option.label, "description": option.description, "rule": option.rule.to_dict()}
        for option in allow_rule_options(request)
    ]


async def wait_for(predicate, timeout: float = 3.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        await asyncio.sleep(0.01)
    return False


async def make_relay(
    tmp_path: Path,
    *,
    on_notice: Callable[[str], None] | None = None,
) -> tuple[ControlRelay, EventRouter, ControlChannel, ButtonAdapter, SessionRuntime]:
    adapter = ButtonAdapter()
    source: asyncio.Queue[AgentEvent] = asyncio.Queue()

    async def emit(event: AgentEvent) -> None:
        await source.put(event)

    channel = ControlChannel(session_id="s-1", emit=emit)
    router = EventRouter(source)
    control_events = router.subscribe(KnownEventType.CONTROL_REQUESTED, KnownEventType.CONTROL_RESOLVED)
    runtime = SessionRuntime(
        session_id="s-1",
        project_path=tmp_path,
        control=channel,
        permission_mode=PermissionMode.ASK,
        on_notice=on_notice,
    )
    relay = ControlRelay(
        chat_ref=ChatRef("buttony", "42"),
        runtime=runtime,
        adapter=adapter,
        event_queue=control_events,
        client_id="buttony:42",
    )
    router.start()
    relay.start()
    return relay, router, channel, adapter, runtime


def tap(token: str, option: str) -> InboundMessage:
    return InboundMessage(
        channel="buttony",
        chat_id="42",
        sender_id="7",
        message_id="tap-1",
        callback_token=token,
        callback_option=option,
    )


@pytest.mark.asyncio
async def test_permission_allow_once(tmp_path: Path) -> None:
    relay, router, channel, adapter, _ = await make_relay(tmp_path)
    try:
        request = make_request()
        task = asyncio.create_task(
            channel.request(
                "permission",
                {"request": serialize_permission_request(request), "rule_options": rule_options_payload(request)},
                default={"allowed": False, "reason": "unanswered"},
            )
        )
        assert await wait_for(lambda: bool(adapter.buttons))
        prompt, options = adapter.buttons[0][1], adapter.buttons[0][2]
        assert "ls -la" in prompt
        assert [option.option_id for option in options][0] == "allow_once"
        assert [option.option_id for option in options][-1] == "deny"
        await relay.handle_tap(tap(adapter.tokens[0], "allow_once"))
        response = await task
        assert response["allowed"] is True
    finally:
        await relay.stop()
        await router.stop()


@pytest.mark.asyncio
async def test_permission_deny(tmp_path: Path) -> None:
    relay, router, channel, adapter, _ = await make_relay(tmp_path)
    try:
        request = make_request()
        task = asyncio.create_task(
            channel.request(
                "permission",
                {"request": serialize_permission_request(request), "rule_options": rule_options_payload(request)},
                default={"allowed": True, "reason": "unanswered"},
            )
        )
        assert await wait_for(lambda: bool(adapter.buttons))
        await relay.handle_tap(tap(adapter.tokens[0], "deny"))
        response = await task
        assert response["allowed"] is False
        assert "Denied" in str(response["reason"])
    finally:
        await relay.stop()
        await router.stop()


@pytest.mark.asyncio
async def test_permission_allow_always_persists_the_rule(tmp_path: Path) -> None:
    relay, router, channel, adapter, runtime = await make_relay(tmp_path)
    try:
        request = make_request()
        task = asyncio.create_task(runtime.permission_callback(request))
        assert await wait_for(lambda: bool(adapter.buttons))
        assert not (tmp_path / PERMISSIONS_RELATIVE_PATH).exists()
        expected_rule = channel.pending()[0].payload["rule_options"][0]["rule"]
        await relay.handle_tap(tap(adapter.tokens[0], "allow_always:0"))
        response = await task
        assert response.allowed is True
        assert response.rule is not None
        assert response.rule.to_dict() == expected_rule
        assert ProjectPermissionStore(tmp_path).load() == [response.rule]
    finally:
        await relay.stop()
        await router.stop()


@pytest.mark.asyncio
async def test_question_prompt_and_answer(tmp_path: Path) -> None:
    relay, router, channel, adapter, _ = await make_relay(tmp_path)
    try:
        task = asyncio.create_task(
            channel.request(
                "question",
                {"question": "Which approach?", "options": ["red", "green"], "descriptions": ["warm", "cool"]},
                default={"answer": None},
            )
        )
        assert await wait_for(lambda: bool(adapter.buttons))
        prompt, options = adapter.buttons[0][1], adapter.buttons[0][2]
        assert prompt == "Which approach?"
        assert [option.label for option in options] == ["red", "green"]
        await relay.handle_tap(tap(adapter.tokens[0], "1"))
        response = await task
        assert response == {"answer": "green"}
    finally:
        await relay.stop()
        await router.stop()


@pytest.mark.asyncio
async def test_stale_or_unknown_taps_are_ignored(tmp_path: Path) -> None:
    relay, router, channel, adapter, _ = await make_relay(tmp_path)
    try:
        request = make_request()
        task = asyncio.create_task(
            channel.request(
                "permission",
                {"request": serialize_permission_request(request), "rule_options": rule_options_payload(request)},
                default={"allowed": False, "reason": "unanswered"},
            )
        )
        assert await wait_for(lambda: bool(adapter.buttons))
        await relay.handle_tap(tap("no-such-token", "allow_once"))
        await relay.handle_tap(tap(adapter.tokens[0], "allow_once"))
        await relay.handle_tap(tap(adapter.tokens[0], "deny"))  # second tap: token already consumed
        response = await task
        assert response["allowed"] is True
    finally:
        await relay.stop()
        await router.stop()


@pytest.mark.asyncio
async def test_releasing_the_lease_resolves_prompts_to_defaults(tmp_path: Path) -> None:
    relay, router, channel, adapter, _ = await make_relay(tmp_path)
    request = make_request()
    task = asyncio.create_task(
        channel.request(
            "permission",
            {"request": serialize_permission_request(request), "rule_options": rule_options_payload(request)},
            default={"allowed": False, "reason": "unanswered"},
        )
    )
    assert await wait_for(lambda: bool(adapter.buttons))
    await relay.stop()
    await router.stop()
    response = await task
    assert response["allowed"] is False
    assert response["reason"] == "unanswered"


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["permission", "question"])
async def test_notification_fires_once_only_after_buttons_are_sent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    relay, router, channel, adapter, runtime = await make_relay(tmp_path)
    hook = AsyncMock()
    runtime.adopt(SimpleNamespace(fire_hook=hook))
    entered, deliver = asyncio.Event(), asyncio.Event()
    send_buttons = adapter.send_buttons

    async def blocked_send(chat_id: str, prompt: str, options: list[ButtonOption]) -> str:
        entered.set()
        await deliver.wait()
        return await send_buttons(chat_id, prompt, options)

    monkeypatch.setattr(adapter, "send_buttons", blocked_send)
    task = asyncio.create_task(
        runtime.permission_callback(make_request())
        if kind == "permission"
        else channel.request(
            "question", {"question": "Which approach?", "options": ["red", "green"]}, default={"answer": None}
        )
    )
    try:
        await asyncio.wait_for(entered.wait(), timeout=5)
        hook.assert_not_awaited()
        request = channel.pending()[0]
        event = AgentEvent(
            sender="system", event_type=KnownEventType.CONTROL_REQUESTED, content=request.describe(), session_id="s-1"
        )
        deliver.set()
        assert await wait_for(lambda: hook.await_count == 1)
        assert adapter.buttons
        target = "permission_prompt" if kind == "permission" else "question"
        assert hook.await_args is not None
        assert hook.await_args.args[0] == HookEvent.NOTIFICATION
        assert hook.await_args.kwargs == {"target": target}
        assert hook.await_args.args[1]["notification_type"] == target
        await relay._render(event)
        hook.assert_awaited_once()
        await relay.handle_tap(tap(adapter.tokens[-1], "allow_once" if kind == "permission" else "1"))
        response = await asyncio.wait_for(task, timeout=5)
        if kind == "permission":
            assert isinstance(response, PermissionDecision) and response.allowed
        else:
            assert response == {"answer": "green"}
        assert not (tmp_path / PERMISSIONS_RELATIVE_PATH).exists()
    finally:
        deliver.set()
        await relay.stop()
        await router.stop()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_save_failure_is_reported_by_runtime_and_allows_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    notices: list[str] = []
    relay, router, _channel, adapter, runtime = await make_relay(tmp_path, on_notice=notices.append)
    hook = AsyncMock()
    runtime.adopt(SimpleNamespace(fire_hook=hook))
    saves: list[Path] = []

    def fail_save(store: ProjectPermissionStore, _rules: Any) -> None:
        saves.append(store.path)
        raise PermissionStoreError("cannot save test rules")

    monkeypatch.setattr(ProjectPermissionStore, "save", fail_save)
    task = asyncio.create_task(runtime.permission_callback(make_request()))
    try:
        assert await wait_for(lambda: hook.await_count == 1)
        await relay.handle_tap(tap(adapter.tokens[0], "allow_always:0"))
        decision = await asyncio.wait_for(task, timeout=5)
        assert decision.allowed and decision.rule is None
        assert decision.reason == "Allowed once because the rule could not be saved."
        assert notices == ["cannot save test rules"]
        assert saves == [tmp_path / PERMISSIONS_RELATIVE_PATH]
        hook.assert_awaited_once()
        assert not (tmp_path / PERMISSIONS_RELATIVE_PATH).exists()
    finally:
        await relay.stop()
        await router.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize("policy", ["saved", "auto", "no_controller"])
async def test_noninteractive_decisions_do_not_render_or_notify(tmp_path: Path, policy: str) -> None:
    relay, router, channel, adapter, runtime = await make_relay(tmp_path)
    hook = AsyncMock()
    runtime.adopt(SimpleNamespace(fire_hook=hook))
    observed = router.subscribe(KnownEventType.CONTROL_REQUESTED)
    request = make_request()
    if policy == "saved":
        ProjectPermissionStore(tmp_path).add_rule(allow_rule_options(request)[0].rule)
    elif policy == "auto":
        runtime.permission_mode = PermissionMode.AUTO
    else:
        channel.release("buttony:42")
    try:
        decision = await runtime.permission_callback(request)
        assert decision.allowed is (policy != "no_controller")
        if policy == "no_controller":
            event = await asyncio.wait_for(observed.get(), timeout=5)
            await relay._render(event)
        assert channel.pending() == []
        assert adapter.buttons == []
        hook.assert_not_awaited()
        if policy != "saved":
            assert not (tmp_path / PERMISSIONS_RELATIVE_PATH).exists()
    finally:
        await relay.stop()
        await router.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["permission", "question"])
async def test_failed_button_delivery_does_not_notify(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    relay, router, channel, adapter, runtime = await make_relay(tmp_path)
    hook = AsyncMock()
    runtime.adopt(SimpleNamespace(fire_hook=hook))
    send = AsyncMock(side_effect=RuntimeError("chat unavailable"))
    monkeypatch.setattr(adapter, "send_buttons", send)
    task = asyncio.create_task(
        runtime.permission_callback(make_request())
        if kind == "permission"
        else channel.request("question", {"question": "Pick", "options": ["a", "b"]}, default={"answer": None})
    )
    try:
        assert await wait_for(lambda: send.await_count == 1)
        assert not task.done()
        assert relay._pending == {}
        hook.assert_not_awaited()
        assert relay._pump_task is not None and not relay._pump_task.done()
        channel.release("buttony:42")
        await asyncio.wait_for(task, timeout=5)
        assert not (tmp_path / PERMISSIONS_RELATIVE_PATH).exists()
    finally:
        await relay.stop()
        await router.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["permission", "question"])
async def test_request_settled_during_send_does_not_notify_or_accept_stale_tap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    relay, router, channel, adapter, runtime = await make_relay(tmp_path)
    hook = AsyncMock()
    runtime.adopt(SimpleNamespace(fire_hook=hook))
    entered, deliver, sent = asyncio.Event(), asyncio.Event(), asyncio.Event()
    send_buttons = adapter.send_buttons

    async def blocked_send(chat_id: str, prompt: str, options: list[ButtonOption]) -> str:
        entered.set()
        await deliver.wait()
        token = await send_buttons(chat_id, prompt, options)
        sent.set()
        return token

    monkeypatch.setattr(adapter, "send_buttons", blocked_send)
    task = asyncio.create_task(
        runtime.permission_callback(make_request())
        if kind == "permission"
        else channel.request("question", {"question": "Pick", "options": ["a", "b"]}, default={"answer": None})
    )
    try:
        await asyncio.wait_for(entered.wait(), timeout=5)
        channel.release("buttony:42")
        await asyncio.wait_for(task, timeout=5)
        deliver.set()
        await asyncio.wait_for(sent.wait(), timeout=5)
        await relay.handle_tap(tap(adapter.tokens[0], "allow_always:0" if kind == "permission" else "0"))
        assert relay._pending == {}
        hook.assert_not_awaited()
        assert not (tmp_path / PERMISSIONS_RELATIVE_PATH).exists()
    finally:
        deliver.set()
        await relay.stop()
        await router.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize("index", ["-1", "99", "invalid"])
@pytest.mark.parametrize("kind", ["permission", "question"])
async def test_invalid_button_choice_does_not_select_or_save_a_rule(tmp_path: Path, index: str, kind: str) -> None:
    relay, router, channel, adapter, runtime = await make_relay(tmp_path)
    task = asyncio.create_task(
        runtime.permission_callback(make_request())
        if kind == "permission"
        else channel.request("question", {"question": "Pick", "options": ["a", "b"]}, default={"answer": None})
    )
    try:
        assert await wait_for(lambda: bool(adapter.buttons))
        await relay.handle_tap(tap(adapter.tokens[0], f"allow_always:{index}" if kind == "permission" else index))
        response = await asyncio.wait_for(task, timeout=5)
        if kind == "permission":
            assert isinstance(response, PermissionDecision)
            assert not response.allowed and response.rule is None
        else:
            assert response == {"answer": ""}
        assert not (tmp_path / PERMISSIONS_RELATIVE_PATH).exists()
    finally:
        await relay.stop()
        await router.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize("options", [None, 42, "ab", [], ["only one"]])
async def test_malformed_question_does_not_render_notify_or_stop_the_relay(tmp_path: Path, options: Any) -> None:
    relay, router, channel, adapter, runtime = await make_relay(tmp_path)
    hook = AsyncMock()
    runtime.adopt(SimpleNamespace(fire_hook=hook))
    task = asyncio.create_task(
        channel.request("question", {"question": "Pick", "options": options}, default={"answer": None})
    )
    try:
        assert await wait_for(lambda: bool(channel.pending()))
        await relay._render(
            AgentEvent(
                sender="system",
                event_type=KnownEventType.CONTROL_REQUESTED,
                session_id="s-1",
                content=channel.pending()[0].describe(),
            )
        )
        assert adapter.buttons == []
        hook.assert_not_awaited()
        assert relay._pump_task is not None and not relay._pump_task.done()
        channel.release("buttony:42")
        assert await asyncio.wait_for(task, timeout=5) == {"answer": None}
    finally:
        await relay.stop()
        await router.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize("action", [action for action, _ in PLAN_APPROVAL_OPTIONS] + ["allow_once", "0"])
async def test_full_plan_is_scrubbed_and_chunked_with_only_exact_approval_ids(tmp_path: Path, action: str) -> None:
    relay, router, channel, adapter, runtime = await make_relay(tmp_path)
    adapter.capabilities = replace(adapter.capabilities, text_chunk_limit=128)
    relay._scrub = lambda text: text.replace("fake-private-value", "[redacted]")
    plan = "# Full plan\n\n" + ("- Detailed implementation step.\n" * 150) + "fake-private-value\nTail."
    task = asyncio.create_task(runtime.request_plan_approval(plan))
    try:
        assert await wait_for(lambda: bool(adapter.buttons))
        assert len(adapter.sent) > 1
        assert all(len(part) <= 128 for part in adapter.sent)
        assert "fake-private-value" not in "\n".join(adapter.sent)
        assert "[redacted]" in "\n".join(adapter.sent)
        assert adapter.sent[-1].endswith("Tail.")
        prompt, buttons = adapter.buttons[0][1:]
        assert prompt == "Implement this plan?"
        assert [(button.option_id, button.label) for button in buttons] == list(PLAN_APPROVAL_OPTIONS)
        request = channel.pending()[0]
        assert request.payload["plan"] == plan and request.default == {"action": None}
        await relay.handle_tap(tap(adapter.tokens[0], action))
        result = await asyncio.wait_for(task, 5)
        assert result == (action if action in dict(PLAN_APPROVAL_OPTIONS) else None)
        await relay.handle_tap(tap(adapter.tokens[0], "implement_plan"))
        assert not channel.pending()
    finally:
        await relay.stop()
        await router.stop()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("blocked_at", ["text", "buttons"])
async def test_cancelled_plan_delivery_never_gives_late_tokens_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, blocked_at: str
) -> None:
    relay, router, channel, adapter, runtime = await make_relay(tmp_path)
    entered, release = asyncio.Event(), asyncio.Event()
    send_text, send_buttons = adapter.send_text, adapter.send_buttons

    async def blocked_text(chat_id: str, text: str, **kwargs: Any) -> str:
        entered.set()
        await release.wait()
        return await send_text(chat_id, text, **kwargs)

    async def blocked_buttons(chat_id: str, prompt: str, options: list[ButtonOption]) -> str:
        entered.set()
        await release.wait()
        return await send_buttons(chat_id, prompt, options)

    monkeypatch.setattr(
        adapter,
        "send_text" if blocked_at == "text" else "send_buttons",
        blocked_text if blocked_at == "text" else blocked_buttons,
    )
    task = asyncio.create_task(runtime.request_plan_approval("# Plan\n\nA complete plan."))
    try:
        await asyncio.wait_for(entered.wait(), 5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        release.set()
        assert await wait_for(lambda: bool(adapter.sent if blocked_at == "text" else adapter.tokens))
        if adapter.tokens:
            await relay.handle_tap(tap(adapter.tokens[0], "implement_plan"))
        else:
            assert not adapter.buttons
        assert not channel.pending() and not relay._pending
    finally:
        release.set()
        await relay.stop()
        await router.stop()


@pytest.mark.asyncio
async def test_late_cancelled_plan_token_cannot_answer_a_fresh_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    relay, router, channel, adapter, runtime = await make_relay(tmp_path)
    entered, release = asyncio.Event(), asyncio.Event()
    send_buttons = adapter.send_buttons

    async def blocked_buttons(chat_id: str, prompt: str, options: list[ButtonOption]) -> str:
        if not adapter.tokens:
            entered.set()
            await release.wait()
        return await send_buttons(chat_id, prompt, options)

    monkeypatch.setattr(adapter, "send_buttons", blocked_buttons)
    old = asyncio.create_task(runtime.request_plan_approval("# Original plan\n\nDo not implement this revision."))
    fresh: asyncio.Task[str | None] | None = None
    try:
        await asyncio.wait_for(entered.wait(), 5)
        old_request = channel.pending()[0].request_id
        old.cancel()
        with pytest.raises(asyncio.CancelledError):
            await old
        fresh = asyncio.create_task(runtime.request_plan_approval("# Fresh plan\n\nImplement only this revision."))
        assert await wait_for(lambda: bool(channel.pending()))
        request_id = channel.pending()[0].request_id
        assert request_id != old_request
        release.set()
        assert await wait_for(lambda: any(prompt.request_id == request_id for prompt in relay._pending.values()))
        assert len(adapter.tokens) == 2
        assert adapter.tokens[0] not in relay._pending
        token = next(token for token, prompt in relay._pending.items() if prompt.request_id == request_id)
        await relay.handle_tap(tap(adapter.tokens[0], "implement_plan_clear"))
        assert not fresh.done() and channel.is_pending(request_id)
        await relay.handle_tap(tap(token, "implement_plan"))
        assert await asyncio.wait_for(fresh, 5) == "implement_plan"
        assert not channel.pending()
    finally:
        release.set()
        await relay.stop()
        await router.stop()
        await asyncio.gather(old, *([fresh] if fresh is not None else []), return_exceptions=True)


@pytest.mark.asyncio
async def test_plan_request_does_not_turn_question_taps_into_implementation(tmp_path: Path) -> None:
    relay, router, channel, adapter, runtime = await make_relay(tmp_path)
    question = asyncio.create_task(
        channel.request("question", {"question": "Which?", "options": ["red", "green"]}, default={"answer": None})
    )
    plan = asyncio.create_task(runtime.request_plan_approval("# Plan\n\nImplement the requested change."))
    try:
        assert await wait_for(lambda: len(adapter.tokens) == 2)
        requests = {request.kind: request.request_id for request in channel.pending()}
        assert not runtime.answer_plan_approval(requests["question"], "implement_plan", client_id="buttony:42")
        question_token = next(token for token, prompt in relay._pending.items() if prompt.kind == "question")
        plan_token = next(token for token, prompt in relay._pending.items() if prompt.kind == "plan_approval")
        await relay.handle_tap(tap(question_token, "implement_plan"))
        assert await question == {"answer": ""}
        assert not plan.done()
        await relay.handle_tap(tap(plan_token, "discuss_plan"))
        assert await plan == "discuss_plan"
    finally:
        await relay.stop()
        await router.stop()
        await asyncio.gather(question, plan, return_exceptions=True)
