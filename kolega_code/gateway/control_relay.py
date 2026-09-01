"""Relay control-channel prompts (permissions, questions) to chat buttons.

Permission approvals and ``ask_user_choice`` questions travel the control
channel as ``CONTROL_REQUESTED`` events, which also lands them in the session
recording. The relay is the chat's answerer: it renders each prompt as an
inline button row, remembers the callback token, and translates a tap back
into an answer through ``SessionRuntime.answer_permission``/``answer_question``.

Exactly one client may answer a session's prompts; the relay takes the control
lease with the chat key as its client id and releases it when the session
closes — which resolves any unanswered prompt to its default, so the agent
never waits on a chat that can no longer answer.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Callable, Optional

from kolega_code.events import AgentEvent, KnownEventType
from kolega_code.gateway.adapters.base import ButtonOption, ChatRef, GatewayAdapter, InboundMessage
from kolega_code.permissions import PermissionDecision, PermissionRule
from kolega_code.session.runtime import deserialize_permission_request

logger = logging.getLogger(__name__)

OPTION_ALLOW_ONCE = "allow_once"
OPTION_DENY = "deny"
OPTION_ALLOW_ALWAYS_PREFIX = "allow_always:"
DENIED_REASON = "Denied by the user."
UNKNOWN_OPTION_REASON = "Unknown approval option."
#: Telegram button labels cap at 64 characters.
MAX_BUTTON_LABEL_CHARS = 64

#: Applied to every prompt before it reaches the wire.
Scrub = Callable[[str], str]


@dataclass
class _PendingPrompt:
    """One rendered button prompt awaiting a tap."""

    kind: str
    request_id: str
    token: str
    rule_options: list[dict[str, Any]]
    options: list[str]


class ControlRelay:
    """The chat-side answerer for one session's control channel."""

    def __init__(
        self,
        *,
        chat_ref: ChatRef,
        runtime: Any,
        adapter: GatewayAdapter,
        event_queue: asyncio.Queue[AgentEvent],
        client_id: str,
        scrub: Optional[Scrub] = None,
    ) -> None:
        self._chat_ref = chat_ref
        self._runtime = runtime
        self._adapter = adapter
        self._events = event_queue
        self._client_id = client_id
        self._scrub = scrub or (lambda text: text)
        self._pending: dict[str, _PendingPrompt] = {}
        self._pump_task: Optional[asyncio.Task[None]] = None

    # -- Lifecycle ---------------------------------------------------------

    def start(self) -> None:
        self._runtime.control.acquire(self._client_id)
        self._pump_task = asyncio.create_task(self._pump(), name=f"gateway-relay-{self._client_id}")

    async def stop(self) -> None:
        if self._pump_task is not None:
            self._pump_task.cancel()
            try:
                await self._pump_task
            except asyncio.CancelledError:
                pass
            self._pump_task = None
        # Releasing the lease resolves every outstanding prompt to its default,
        # so a closing session never strands the agent on an unanswerable ask.
        self._runtime.control.release(self._client_id)
        self._pending.clear()

    # -- Prompt rendering --------------------------------------------------

    async def _pump(self) -> None:
        try:
            while True:
                event = await self._events.get()
                if event.event_type == KnownEventType.CONTROL_REQUESTED:
                    await self._render(event)
                elif event.event_type == KnownEventType.CONTROL_RESOLVED:
                    self._forget_request(str(event.content.get("request_id") or ""))
        except asyncio.CancelledError:
            raise

    def _forget_request(self, request_id: str) -> None:
        for token, pending in list(self._pending.items()):
            if pending.request_id == request_id:
                self._pending.pop(token, None)

    async def _render(self, event: AgentEvent) -> None:
        request_id = str(event.content.get("request_id") or "")
        kind = str(event.content.get("kind") or "")
        payload = event.content.get("payload") or {}
        if not request_id:
            return
        if kind == "permission":
            await self._render_permission(request_id, payload)
        elif kind == "question":
            await self._render_question(request_id, payload)
        else:
            logger.debug("gateway relay: ignoring unknown control kind %r", kind)

    async def _render_permission(self, request_id: str, payload: dict[str, Any]) -> None:
        request = deserialize_permission_request(payload.get("request") or {})
        rule_options = [dict(entry) for entry in (payload.get("rule_options") or []) if isinstance(entry, dict)]
        summary = str(request.summary or request.tool_name or "a tool")
        buttons = [ButtonOption(OPTION_ALLOW_ONCE, "Allow once")]
        for index, entry in enumerate(rule_options):
            label = str(entry.get("label") or "allow always")
            buttons.append(ButtonOption(f"{OPTION_ALLOW_ALWAYS_PREFIX}{index}", f"Always: {label}"))
        buttons.append(ButtonOption(OPTION_DENY, "Deny"))
        try:
            token = await self._adapter.send_buttons(
                self._chat_ref.chat_id,
                self._scrub(f"🔐 {summary}"),
                buttons,
            )
        except Exception:  # noqa: BLE001 — a lost prompt falls back to the channel default
            logger.exception("gateway relay: could not render permission prompt")
            return
        self._pending[token] = _PendingPrompt(
            kind="permission", request_id=request_id, token=token, rule_options=rule_options, options=[]
        )

    async def _render_question(self, request_id: str, payload: dict[str, Any]) -> None:
        question = str(payload.get("question") or "Pick an option")
        options = [str(option) for option in (payload.get("options") or [])]
        buttons = [
            ButtonOption(str(index), label[:MAX_BUTTON_LABEL_CHARS] or "\u00a0") for index, label in enumerate(options)
        ]
        if not buttons:
            return
        try:
            token = await self._adapter.send_buttons(self._chat_ref.chat_id, self._scrub(question), buttons)
        except Exception:  # noqa: BLE001
            logger.exception("gateway relay: could not render question prompt")
            return
        self._pending[token] = _PendingPrompt(
            kind="question", request_id=request_id, token=token, rule_options=[], options=options
        )

    # -- Tap handling ------------------------------------------------------

    async def handle_tap(self, message: InboundMessage) -> None:
        token = message.callback_token
        if not token or message.callback_option is None:
            return
        pending = self._pending.pop(token, None)
        if pending is None:
            return
        if pending.kind == "permission":
            self._runtime.answer_permission(
                pending.request_id,
                self._permission_decision(message.callback_option, pending),
                client_id=self._client_id,
            )
        elif pending.kind == "question":
            self._runtime.answer_question(
                pending.request_id,
                {"answer": self._question_answer(message.callback_option, pending)},
                client_id=self._client_id,
            )

    def _permission_decision(self, option_id: str, pending: _PendingPrompt) -> PermissionDecision:
        if option_id == OPTION_ALLOW_ONCE:
            return PermissionDecision(allowed=True)
        if option_id == OPTION_DENY:
            return PermissionDecision(allowed=False, reason=DENIED_REASON)
        if option_id.startswith(OPTION_ALLOW_ALWAYS_PREFIX):
            try:
                entry = pending.rule_options[int(option_id[len(OPTION_ALLOW_ALWAYS_PREFIX) :])]
                rule = PermissionRule.from_dict(entry.get("rule") or {})
            except (ValueError, IndexError, TypeError):
                return PermissionDecision(allowed=False, reason=UNKNOWN_OPTION_REASON)
            return PermissionDecision(allowed=True, reason="Allowed and remembered.", rule=rule)
        return PermissionDecision(allowed=False, reason=UNKNOWN_OPTION_REASON)

    @staticmethod
    def _question_answer(option_id: str, pending: _PendingPrompt) -> str:
        try:
            return pending.options[int(option_id)]
        except (ValueError, IndexError):
            return ""
