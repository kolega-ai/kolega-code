"""ACP presentation for runtime-owned control requests."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Callable

from acp.connection import StreamDirection, StreamEvent
from acp.helpers import text_block, tool_content, update_tool_call
from acp.interfaces import Client
from acp.schema import (
    AcceptElicitationResponse,
    ClientCapabilities,
    ElicitationFormSessionMode,
    ElicitationSchema,
    ElicitationStringPropertySchema,
    EnumOption,
    PermissionOption,
)

from kolega_code.acp.agent_factory import _finish_cleanup
from kolega_code.acp.permissions import CONTROL_REQUEST_ID_META_KEY, AcpPermissionBroker
from kolega_code.events import AgentEvent, KnownEventType
from kolega_code.permissions import PermissionDecision
from kolega_code.session.plans import PLAN_APPROVAL_OPTIONS, PlanApprovalAction, plan_approval_action
from kolega_code.session.runtime import deserialize_permission_request, deserialize_permission_rule_options
from kolega_code.tools.core import ToolError

if TYPE_CHECKING:
    from kolega_code.acp.session import AcpSession


class AcpControlAdapter:
    def __init__(
        self,
        session: AcpSession,
        *,
        session_provider: Callable[[], AcpSession | None],
        connection_provider: Callable[[], Client | None],
        capabilities_provider: Callable[[], ClientCapabilities | None],
    ) -> None:
        self.client_id: str = f"acp:{session.session_id}"
        self.tasks: dict[str, asyncio.Task[None]] = {}
        self._session: AcpSession = session
        self._session_provider: Callable[[], AcpSession | None] = session_provider
        self._connection_provider: Callable[[], Client | None] = connection_provider
        self._capabilities_provider: Callable[[], ClientCapabilities | None] = capabilities_provider
        self._broker: AcpPermissionBroker | None = None
        session.question_state["elicit"] = self.request_question

    async def emit(self, event: AgentEvent) -> None:
        session = self._session
        await session.agent.connection_manager.broadcast_event(
            event,
            session.record.workspace_id,
            session.record.thread_id,
        )
        request_id = str(event.content.get("request_id") or "")
        if event.event_type == KnownEventType.CONTROL_RESOLVED:
            task = self.tasks.get(request_id)
            if (
                task is not None
                and not task.cancelling()
                and (event.content.get("reason") != "answered" or event.content.get("kind") == "plan_approval")
            ):
                task.cancel()
            return
        if event.event_type != KnownEventType.CONTROL_REQUESTED or not event.content.get("has_controller"):
            return
        ready = asyncio.Event()

        async def handle() -> None:
            # Eager tasks must be registered before the transport can acknowledge delivery.
            await ready.wait()
            await self._handle_request(event)

        task = asyncio.create_task(handle(), name=f"acp-control-{session.session_id}")
        self.tasks[request_id] = task

        def completed(done: asyncio.Task[None]) -> None:
            if self.tasks.get(request_id) is done:
                self.tasks.pop(request_id)

        task.add_done_callback(completed)
        ready.set()

    async def on_transport_message(self, event: StreamEvent) -> bool:
        if event.direction != StreamDirection.OUTGOING or event.message.get("method") not in {
            "session/request_permission",
            "elicitation/create",
        }:
            return False
        params = event.message.get("params")
        if not isinstance(params, dict):
            return False
        metadata = params.get("_meta")
        if not isinstance(metadata, dict):
            return False
        request_id = metadata.get(CONTROL_REQUEST_ID_META_KEY)
        if not isinstance(request_id, str):
            return False
        session = self._session_provider()
        if request_id not in self.tasks or session is None or session.runtime is None or session.closed:
            return False
        await session.runtime.prompt_delivered(request_id)
        return True

    async def _handle_request(self, event: AgentEvent) -> None:
        session = self._session_provider()
        if session is None or session.runtime is None:
            return
        request_id = str(event.content.get("request_id") or "")
        kind = str(event.content.get("kind") or "")
        payload = event.content.get("payload")
        if not request_id:
            return
        runtime = session.runtime
        if not runtime.control.is_pending(request_id):
            return
        if kind == "plan_approval":
            try:
                action = await self._prompt_plan_approval(payload, request_id=request_id)
            except asyncio.CancelledError:
                if session.plan_approval_task is None or not session.plan_approval_task.cancelling():
                    runtime.answer_plan_approval(request_id, None, client_id=self.client_id)
                raise
            except Exception:
                action = None
            if session.plan_approval_task is not None and session.plan_approval_task.cancelling():
                return
            task = asyncio.current_task()
            if (
                self._session_provider() is not session
                or session.runtime is not runtime
                or session.closed
                or (task is not None and task.cancelling())
            ):
                action = None
            runtime.answer_plan_approval(request_id, action, client_id=self.client_id)
            return
        if not isinstance(payload, dict):
            return
        if kind == "permission":
            request_data = payload.get("request")
            try:
                if not isinstance(request_data, dict):
                    raise ValueError("Malformed permission request")
                request = deserialize_permission_request(request_data)
            except Exception:
                runtime.answer_permission(
                    request_id,
                    PermissionDecision(allowed=False, reason="The permission request was malformed."),
                    client_id=self.client_id,
                )
                return
            if self._broker is None:
                conn = self._connection_provider()
                assert conn is not None
                self._broker = AcpPermissionBroker(conn, session.session_id, lambda: session.agent)
            decision = await self._broker.prompt(
                request,
                deserialize_permission_rule_options(payload.get("rule_options")),
                request_id=request_id,
                on_delivered=lambda: runtime.prompt_delivered(request_id),
            )
            runtime.answer_permission(request_id, decision, client_id=self.client_id)
            return
        if kind == "question":
            question = str(payload.get("question") or "")
            labels = [str(label) for label in (payload.get("options") or [])]
            descriptions = [str(item) for item in (payload.get("descriptions") or [])]
            try:
                answer = await self._elicit_answer(question, labels, descriptions, request_id=request_id)
            except Exception as exc:
                runtime.answer_question(
                    request_id,
                    {"answer": None, "error": str(exc)},
                    client_id=self.client_id,
                )
                return
            runtime.answer_question(request_id, {"answer": answer or None}, client_id=self.client_id)

    async def _prompt_plan_approval(self, payload: object, *, request_id: str) -> PlanApprovalAction | None:
        if not isinstance(payload, dict):
            return None
        plan = payload.get("plan")
        if (
            not isinstance(plan, str)
            or not plan.strip()
            or payload.get("options") != [{"id": action, "label": label} for action, label in PLAN_APPROVAL_OPTIONS]
        ):
            return None
        conn = self._connection_provider()
        if conn is None:
            return None
        session = self._session
        if session.closed:
            return None
        tool_call = update_tool_call(
            f"plan-approval-{session.session_id[:8]}",
            title="Approve implementation plan?",
            kind="other",
            status="pending",
            content=[tool_content(text_block(plan))],
        )
        options = [
            PermissionOption(
                option_id=action,
                name=label,
                kind="reject_once" if action == "discuss_plan" else "allow_once",
            )
            for action, label in PLAN_APPROVAL_OPTIONS
        ]
        response = await conn.request_permission(
            session.session_id, tool_call, options, **{CONTROL_REQUEST_ID_META_KEY: request_id}
        )
        task = asyncio.current_task()
        if (
            self._session_provider() is not session
            or session.runtime is None
            or session.closed
            or (session.plan_approval_task is not None and session.plan_approval_task.cancelling())
            or (task is not None and task.cancelling())
            or not session.runtime.control.is_pending(request_id)
        ):
            return None
        # Embedded clients without a transport observer acknowledge at reply.
        await session.runtime.prompt_delivered(request_id)
        outcome = response.get("outcome") if isinstance(response, dict) else getattr(response, "outcome", None)
        if isinstance(outcome, dict):
            if outcome.get("outcome") != "selected":
                return None
            option_id = outcome.get("optionId")
        else:
            if getattr(outcome, "outcome", None) != "selected":
                return None
            option_id = getattr(outcome, "option_id", None)
        return plan_approval_action(option_id)

    async def request_question(self, question: str, labels: list[str], descriptions: list[str]) -> str:
        session = self._session_provider()
        if session is None or session.runtime is None:
            raise ToolError("The session is no longer active.")
        response = await session.runtime.control.request(
            "question",
            {"question": question, "options": list(labels), "descriptions": list(descriptions)},
            default={"answer": None},
        )
        if response.get("error"):
            raise ToolError(str(response["error"]))
        answer = response.get("answer")
        if not isinstance(answer, str) or not answer.strip():
            raise ToolError("The user declined to answer the question.")
        return answer

    async def _elicit_answer(
        self,
        question: str,
        labels: list[str],
        descriptions: list[str],
        *,
        request_id: str,
    ) -> str:
        capabilities = self._capabilities_provider()
        elicitation = getattr(capabilities, "elicitation", None) if capabilities is not None else None
        if elicitation is None or not getattr(elicitation, "form", None):
            raise ToolError(
                "This editor client does not support interactive question forms; "
                "ask the user directly in the conversation instead."
            )
        conn = self._connection_provider()
        if conn is None:
            raise ToolError("No client connection is available for questions.")
        options = [
            EnumOption(const=label, title=label, description=description or None)
            for label, description in zip(labels, descriptions)
        ]
        schema = ElicitationSchema(
            properties={"answer": ElicitationStringPropertySchema(type="string", title=question, one_of=options)},
            required=["answer"],
        )
        response = await conn.create_elicitation(
            message=question,
            mode=ElicitationFormSessionMode(session_id=self._session.session_id, requested_schema=schema),
            **{CONTROL_REQUEST_ID_META_KEY: request_id},
        )
        session = self._session_provider()
        if session is not None and session.runtime is not None:
            await session.runtime.prompt_delivered(request_id)
        if isinstance(response, AcceptElicitationResponse):
            content = response.content or {}
            answer = content.get("answer")
            if answer:
                return str(answer)
            raise ToolError("The client accepted the question but returned no answer.")
        raise ToolError("The user declined to answer the question.")

    async def shutdown(self) -> None:
        if self._session.runtime is not None:
            self._session.runtime.control.release(self.client_id)
        tasks = list(self.tasks.values())
        for task in tasks:
            if not task.cancelling():
                task.cancel()

        async def join() -> None:
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            self.tasks.clear()
            self._broker = None

        await _finish_cleanup(join())
