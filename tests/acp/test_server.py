"""In-process tests for the ACP server's real turn loop (Phase 1).

No network and no settings: a fake factory hands the server an ``AcpSession``
whose agent runs on a scripted fake LLM. The protocol client is faked, so
these tests assert the full prompt -> stream -> complete (and cancel) flow.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest
from acp.interfaces import Client
from acp.schema import AgentMessageChunk, RequestPermissionResponse

from kolega_code.acp.permissions import OPTION_ALLOW_ONCE
from kolega_code.acp.server import AcpAgent
from kolega_code.acp.session import AcpSession
from kolega_code.cli.connection import CliConnectionManager
from kolega_code.cli.session_store import SessionRecord
from kolega_code.llm.models import Message, TextBlock
from kolega_code.llm.providers.models import TokenCount
from kolega_code.permissions import PermissionKind, PermissionRequest

from tests.agent.compaction_helpers import build_agent


class _FakeConn:
    def __init__(self) -> None:
        self.updates: list[Any] = []
        self.permission_calls: list[tuple[str, Any, list[Any]]] = []

    async def session_update(self, session_id: str, update: Any, source: str = "") -> None:
        self.updates.append(update)

    async def request_permission(self, session_id: str, tool_call: Any, options: list[Any], **kwargs: Any) -> Any:
        self.permission_calls.append((session_id, tool_call, options))
        return RequestPermissionResponse.model_validate(
            {"outcome": {"outcome": "selected", "optionId": OPTION_ALLOW_ONCE}}
        )


class EventStream:
    """LLM stream that yields scripted raw events, then a final message."""

    def __init__(self, events: list[Any], final: Message) -> None:
        self._events = events
        self._final = final

    async def __aenter__(self) -> EventStream:
        return self

    async def __aexit__(self, *exc: Any) -> bool:
        return False

    def __aiter__(self) -> EventStream:
        return self

    async def __anext__(self) -> Any:
        if self._events:
            return self._events.pop(0)
        raise StopAsyncIteration

    async def get_final_message(self) -> Message:
        return self._final


class BlockingStream(EventStream):
    """Stream whose next event blocks until cancelled (for cancel tests)."""

    def __init__(self) -> None:
        self._gate = asyncio.Event()
        super().__init__([], Message(role="assistant", content=[TextBlock(text="ok")], stop_reason="end_turn"))

    async def __anext__(self) -> Any:
        await self._gate.wait()
        raise StopAsyncIteration


class StreamingLLM:
    """Drop-in ``agent.llm`` scripting one turn of stream events."""

    def __init__(self, events: list[Any] | None = None, final_text: str = "done") -> None:
        self._events = list(events or [])
        self._final = Message(role="assistant", content=[TextBlock(text=final_text)], stop_reason="end_turn")
        self.provider = MagicMock(base_url="https://api.test.example/v1")
        self.count_tokens = AsyncMock(return_value=TokenCount(input_tokens=0))
        self.generate = AsyncMock(return_value=self._final)
        self.stream = AsyncMock(side_effect=self._stream)

    async def _stream(self, *args: Any, **kwargs: Any) -> EventStream:
        return EventStream(self._events, self._final)


def _text_event(text: str) -> SimpleNamespace:
    return SimpleNamespace(type="text", text=text)


class _FakeFactory:
    """Factory stub returning one prebuilt session; the server only calls open_session."""

    def __init__(self, session: AcpSession) -> None:
        self._session = session
        self.config_options: list[Any] = []
        self.permission_mode_calls: list[tuple[str, Any]] = []
        self.interaction_mode_calls: list[tuple[str, str]] = []
        self.model_calls: list[tuple[str, str, str]] = []

    async def open_session(
        self,
        cwd: str,
        *,
        session_id: str | None = None,
        permission_callback: Any = None,
    ) -> AcpSession:
        resolved_id = session_id or self._session.session_id
        return AcpSession(
            session_id=resolved_id,
            record=self._session.record,
            agent=self._session.agent,
            manager=self._session.manager,
        )

    def config_options_for(self, session: AcpSession) -> list[Any]:
        return self.config_options

    async def set_permission_mode(self, session: AcpSession, mode: Any) -> None:
        self.permission_mode_calls.append((session.session_id, mode))

    async def set_interaction_mode(self, session: AcpSession, mode: str) -> None:
        self.interaction_mode_calls.append((session.session_id, mode))
        session.record.interaction_mode = mode

    async def apply_model(self, session: AcpSession, provider: str, model: str) -> None:
        self.model_calls.append((session.session_id, provider, model))

    def persist(self, session: AcpSession) -> None:
        pass


def _make_session(tmp_path: Path, llm: Any) -> tuple[AcpSession, CliConnectionManager]:
    manager = CliConnectionManager()
    agent, cm = build_agent(tmp_path, connection_manager=manager, llm=llm)
    record = SessionRecord.create(Path(tmp_path), "cli", {})
    session = AcpSession(session_id=record.session_id, record=record, agent=agent, manager=cm)
    return session, cm


def _make_agent(session: AcpSession) -> tuple[AcpAgent, _FakeConn]:
    conn = _FakeConn()
    agent = AcpAgent(factory=_FakeFactory(session))  # pyright: ignore[reportArgumentType]
    agent.on_connect(cast(Client, conn))
    return agent, conn


@pytest.mark.asyncio
async def test_initialize_echoes_protocol_version_and_advertises_load() -> None:
    agent = AcpAgent(factory=_FakeFactory(cast(AcpSession, None)))  # pyright: ignore[reportArgumentType]
    response = await agent.initialize(protocol_version=1, client_capabilities=None)
    assert response.protocol_version == 1
    capabilities = response.agent_capabilities
    assert capabilities is not None
    assert capabilities.load_session is True
    assert capabilities.session_capabilities is not None
    assert capabilities.session_capabilities.list is not None
    assert capabilities.session_capabilities.close is not None


@pytest.mark.asyncio
async def test_prompt_streams_text_and_completes(tmp_path: Path) -> None:
    llm = StreamingLLM(events=[_text_event("hello " * 12)])  # >50 chars guarantees a chunk yield
    session, _cm = _make_session(tmp_path, llm)
    agent, conn = _make_agent(session)
    new_session = await agent.new_session(cwd=str(tmp_path))

    response = await agent.prompt(session_id=new_session.session_id, prompt=[{"type": "text", "text": "hi"}])  # pyright: ignore[reportArgumentType]

    assert response.stop_reason == "end_turn"
    texts = [u.content.text for u in conn.updates if isinstance(u, AgentMessageChunk)]
    assert texts, "expected streamed agent text"
    assert "hello" in "".join(texts)


@pytest.mark.asyncio
async def test_cancel_interrupts_turn_with_cancelled_stop_reason(tmp_path: Path) -> None:
    blocking = StreamingLLM(events=[])
    blocking.stream = AsyncMock(side_effect=lambda *a, **k: BlockingStream())
    session, _cm = _make_session(tmp_path, blocking)
    agent, _conn = _make_agent(session)
    new_session = await agent.new_session(cwd=str(tmp_path))

    prompt_task = asyncio.create_task(
        agent.prompt(session_id=new_session.session_id, prompt=[{"type": "text", "text": "hi"}])  # pyright: ignore[reportArgumentType],
    )
    await asyncio.sleep(0.05)
    await agent.cancel(session_id=new_session.session_id)
    response = await prompt_task

    assert response.stop_reason == "cancelled"


@pytest.mark.asyncio
async def test_prompt_rejects_unknown_session() -> None:
    from acp.exceptions import RequestError

    session, _cm = _make_session(Path("/tmp"), StreamingLLM())
    agent, _conn = _make_agent(session)
    with pytest.raises(RequestError):
        await agent.prompt(session_id="nope", prompt=[{"type": "text", "text": "hi"}])  # pyright: ignore[reportArgumentType]


@pytest.mark.asyncio
async def test_prompt_rejects_empty_text() -> None:
    from acp.exceptions import RequestError

    session, _cm = _make_session(Path("/tmp"), StreamingLLM())
    agent, _conn = _make_agent(session)
    with pytest.raises(RequestError):
        await agent.prompt(session_id=session.session_id, prompt=[{"type": "text", "text": ""}])  # pyright: ignore[reportArgumentType]


@pytest.mark.asyncio
async def test_new_session_converts_config_errors_to_protocol_errors(tmp_path: Path) -> None:
    from acp.exceptions import RequestError

    from kolega_code.cli.config import CliConfigError

    class _RaisingFactory(_FakeFactory):
        async def open_session(
            self,
            cwd: str,
            *,
            session_id: str | None = None,
            permission_callback: Any = None,
        ) -> AcpSession:
            raise CliConfigError("no model configured")

    session, _cm = _make_session(tmp_path, StreamingLLM())
    agent = AcpAgent(factory=_RaisingFactory(session))  # pyright: ignore[reportArgumentType]
    with pytest.raises(RequestError):
        await agent.new_session(cwd=str(tmp_path))


@pytest.mark.asyncio
async def test_new_session_registers_and_returns_session_id(tmp_path: Path) -> None:
    session, _cm = _make_session(tmp_path, StreamingLLM())
    agent, _conn = _make_agent(session)
    response = await agent.new_session(cwd=str(tmp_path))
    assert response.session_id
    assert response.session_id != session.session_id
    assert agent._sessions[response.session_id].agent is session.agent  # noqa: SLF001


@pytest.mark.asyncio
async def test_permission_callback_prompts_through_client(tmp_path: Path) -> None:
    session, _cm = _make_session(tmp_path, StreamingLLM())
    agent, conn = _make_agent(session)
    new_session = await agent.new_session(cwd=str(tmp_path))
    session.agent.current_tool_call_id = "call-9"

    callback = agent._permission_callback_for(new_session.session_id)  # noqa: SLF001
    decision = await callback(
        PermissionRequest(
            kind=PermissionKind.COMMAND,
            tool_name="exec_command",
            inputs={"command": "echo hi"},
            command="echo hi",
        ),
    )

    assert decision.allowed is True
    assert len(conn.permission_calls) == 1
    prompted_session_id, tool_call, options = conn.permission_calls[0]
    assert prompted_session_id == new_session.session_id
    assert tool_call.tool_call_id == "call-9"
    assert options[0].kind == "allow_once"
