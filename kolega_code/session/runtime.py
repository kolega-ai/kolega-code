"""Headless owner of one session's agent.

The terminal UI used to own the agent outright: its lifecycle, its turns, and the
callback that answered permission prompts all lived on a Textual ``App`` subclass,
so an agent could not run at all without a UI attached and no other frontend could
drive one.

``SessionRuntime`` owns those instead. It has no UI dependency, and every
interaction that needs a human answer goes through :class:`ControlChannel`, so a
terminal client, a browser, and an automated harness all drive a session the same
way.

Composition stays with the caller. A CLI assembles an agent from settings, skills,
hooks, prompt extensions, and MCP configuration, and none of that belongs in a
runtime; the caller supplies an ``agent_factory`` and the runtime owns *running*
what it builds.

Permission policy lives here rather than in a frontend. Mode checks and saved-rule
matching are decisions about the session, not about how to draw a dialog, so every
client gets them for free and only genuinely undecidable requests reach a human.
"""

from __future__ import annotations

import asyncio
from dataclasses import asdict
from pathlib import Path
from typing import Any, AsyncIterator, Awaitable, Callable, Optional

from kolega_code.events import AgentEvent
from kolega_code.permissions import (
    PermissionDecision,
    PermissionKind,
    PermissionMode,
    PermissionRequest,
    PermissionRule,
    PermissionStoreError,
    ProjectPermissionStore,
    allow_rule_options,
)

from .control import ControlChannel

#: Builds the agent for this session. Called on start and on every rebuild.
AgentFactory = Callable[[], Awaitable[Any]]

#: Denial is the fallback whenever a permission prompt cannot be answered.
DENIED_BY_DEFAULT = {
    "allowed": False,
    "reason": "No client was able to answer this permission request.",
}


class SessionRuntimeError(RuntimeError):
    """Raised when a runtime operation needs an agent that is not running."""


def serialize_permission_request(request: PermissionRequest) -> dict[str, Any]:
    """Wire form of a permission request, renderable by a client of any kind."""
    payload = asdict(request)
    payload["kind"] = request.kind.value
    payload["summary"] = request.summary
    return payload


def deserialize_permission_request(payload: dict[str, Any]) -> PermissionRequest:
    """Rebuild a request from its wire form."""
    return PermissionRequest(
        kind=PermissionKind(payload.get("kind") or PermissionKind.COMMAND.value),
        tool_name=str(payload.get("tool_name") or ""),
        inputs=dict(payload.get("inputs") or {}),
        command=str(payload.get("command") or ""),
        path=str(payload.get("path") or ""),
        mcp_server=str(payload.get("mcp_server") or ""),
        mcp_tool=str(payload.get("mcp_tool") or ""),
    )


class SessionRuntime:
    """Runs one session's agent, independent of any user interface."""

    def __init__(
        self,
        *,
        session_id: str,
        project_path: Path,
        control: ControlChannel,
        agent_factory: Optional[AgentFactory] = None,
        permission_mode: PermissionMode = PermissionMode.ASK,
        on_notice: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.session_id = session_id
        self.project_path = project_path
        self.control = control
        self.permission_mode = permission_mode
        self._agent_factory = agent_factory
        self._agent: Optional[Any] = None
        # Serialises prompts so two concurrent tool calls cannot both take over
        # the client's single approval surface.
        self._permission_lock = asyncio.Lock()
        self._on_notice = on_notice

    # -- Lifecycle ---------------------------------------------------------

    @property
    def agent(self) -> Optional[Any]:
        """The live agent, or None before start.

        Exposed for introspection a frontend genuinely needs, such as reading
        history or language-server status. Control and lifecycle go through the
        methods below so no client has to reach in to drive a session.
        """
        return self._agent

    @property
    def running(self) -> bool:
        return self._agent is not None

    async def start(self) -> Any:
        """Build the agent for this session using the supplied factory."""
        if self._agent_factory is None:
            raise SessionRuntimeError("This runtime has no agent factory; use adopt() instead")
        self._agent = await self._agent_factory()
        return self._agent

    def adopt(self, agent: Any) -> None:
        """Take ownership of an agent the caller composed itself.

        A host that already assembles agents from its own configuration —
        settings, skills, hooks, prompt extensions — should not have to invert
        that logic to gain a runtime. It composes as before and hands the result
        over, and control still flows through this runtime and its channel.
        """
        self._agent = agent

    async def rebuild(self) -> Any:
        """Replace the agent, disposing of the previous one first."""
        if self._agent_factory is None:
            raise SessionRuntimeError("This runtime has no agent factory; use adopt() instead")
        if self._agent is not None:
            await self._agent.cleanup()
            self._agent = None
        return await self.start()

    async def shutdown(self) -> None:
        if self._agent is None:
            return
        agent = self._agent
        self._agent = None
        await agent.cleanup()

    def _require_agent(self) -> Any:
        if self._agent is None:
            raise SessionRuntimeError("This session has no running agent")
        return self._agent

    # -- Turn execution ----------------------------------------------------

    def send_message(
        self,
        text: str,
        attachments: Optional[list[dict[str, Any]]] = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Run one turn, yielding the agent's response chunks.

        Returns the generator rather than awaiting it, so a caller can consume
        chunks as they arrive and cancel by cancelling the task that consumes it.
        """
        agent = self._require_agent()
        if attachments:
            return agent.process_message_stream(text, attachments)
        return agent.process_message_stream(text)

    # -- Permissions -------------------------------------------------------

    async def permission_callback(self, request: PermissionRequest) -> PermissionDecision:
        """Decide a permission request, asking a client only when necessary.

        Installed on the agent in place of a UI-supplied callback. The ordering is
        deliberate: cheap local policy first, a human last.
        """
        if self.permission_mode != PermissionMode.ASK:
            return PermissionDecision(allowed=True)

        async with self._permission_lock:
            matched = self._matching_rule(request)
            if matched is not None:
                return PermissionDecision(allowed=True, reason=f"Allowed by saved rule {matched.id}.")

            options = allow_rule_options(request)
            response = await self.control.request(
                "permission",
                {
                    "request": serialize_permission_request(request),
                    "rule_options": [
                        {"label": option.label, "description": option.description, "rule": option.rule.to_dict()}
                        for option in options
                    ],
                },
                default=DENIED_BY_DEFAULT,
            )
            return self._decision_from_response(response)

    def _matching_rule(self, request: PermissionRequest) -> Optional[PermissionRule]:
        try:
            return ProjectPermissionStore(self.project_path).first_match(request)
        except PermissionStoreError as exc:
            # An unreadable rule file must not silently grant or deny; fall
            # through to asking, and tell the user why.
            self._notify(str(exc))
            return None

    def _decision_from_response(self, response: dict[str, Any]) -> PermissionDecision:
        rule_data = response.get("rule")
        rule: Optional[PermissionRule] = None
        if isinstance(rule_data, dict):
            try:
                rule = PermissionRule.from_dict(rule_data)
            except Exception:
                rule = None
        return PermissionDecision(
            allowed=bool(response.get("allowed")),
            reason=str(response.get("reason") or ""),
            rule=rule,
        )

    def answer_permission(
        self,
        request_id: str,
        decision: PermissionDecision,
        *,
        client_id: Optional[str] = None,
    ) -> bool:
        """Answer an outstanding permission prompt. False if it is already settled."""
        response: dict[str, Any] = {"allowed": decision.allowed, "reason": decision.reason}
        if decision.rule is not None:
            response["rule"] = decision.rule.to_dict()
        return self.control.respond(request_id, response, client_id=client_id)

    def answer_question(
        self,
        request_id: str,
        answer: dict[str, Any],
        *,
        client_id: Optional[str] = None,
    ) -> bool:
        """Answer an outstanding non-permission prompt."""
        return self.control.respond(request_id, answer, client_id=client_id)

    def set_permission_mode(self, mode: PermissionMode) -> None:
        self.permission_mode = mode
        if self._agent is not None:
            self._agent.set_permission_mode(mode)

    # -- Notices -----------------------------------------------------------

    def _notify(self, message: str) -> None:
        if self._on_notice is None:
            return
        try:
            self._on_notice(message)
        except Exception:
            pass


def control_channel_for(
    session_id: str,
    broadcast: Callable[[AgentEvent, str, str], Awaitable[None]],
    *,
    workspace_id: str,
    thread_id: str,
    timeout: Optional[float] = None,
) -> ControlChannel:
    """Build a control channel that announces on a connection manager.

    Routing announcements through the same transport as every other event is what
    puts prompts into the session recording, so a replay shows where the agent
    stopped to ask.
    """

    async def emit(event: AgentEvent) -> None:
        await broadcast(event, workspace_id, thread_id)

    kwargs: dict[str, Any] = {"session_id": session_id, "emit": emit}
    if timeout is not None:
        kwargs["timeout"] = timeout
    return ControlChannel(**kwargs)
