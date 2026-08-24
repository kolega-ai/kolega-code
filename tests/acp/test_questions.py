"""ask_user_choice over ACP elicitation: tool extension + server bridge."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest
from acp.interfaces import Client
from acp.schema import (
    AcceptElicitationResponse,
    ClientCapabilities,
    DeclineElicitationResponse,
    ElicitationCapabilities,
    ElicitationFormCapabilities,
)

from kolega_code.acp.questions import QUESTION_TOOL_NAME, build_question_extension
from kolega_code.acp.server import AcpAgent
from kolega_code.tools import ToolError

from tests.acp.test_server import StreamingLLM, _FakeFactory, _make_session

_QUESTIONS = [
    {
        "question": "Which approach?",
        "header": "approach",
        "options": [
            {"label": "Conservative", "description": "safer"},
            {"label": "Aggressive", "description": "faster"},
        ],
    },
    {
        "question": "Ship it?",
        "header": "",
        "options": [{"label": "Yes", "description": ""}, {"label": "No", "description": ""}],
    },
]


@pytest.mark.asyncio
async def test_question_extension_elicits_each_question_in_order() -> None:
    state: dict[str, Any] = {}
    calls: list[tuple[str, list[str], list[str]]] = []

    async def elicit(question: str, labels: list[str], descriptions: list[str]) -> str:
        calls.append((question, labels, descriptions))
        return labels[len(calls) - 1]

    state["elicit"] = elicit
    extension = build_question_extension(state)

    result = json.loads(await extension.tools[QUESTION_TOOL_NAME](_QUESTIONS))
    assert result == {"approach": "Conservative", "Ship it?": "No"}
    assert [call[0] for call in calls] == ["Which approach?", "Ship it?"]
    assert [call[1] for call in calls] == [["Conservative", "Aggressive"], ["Yes", "No"]]
    assert state.get("pending") is False


@pytest.mark.asyncio
async def test_question_extension_rejects_when_pending() -> None:
    state: dict[str, Any] = {"pending": True, "elicit": _stub_elicit}
    extension = build_question_extension(state)

    with pytest.raises(ToolError, match="already waiting"):
        await extension.tools[QUESTION_TOOL_NAME](_QUESTIONS[:1])


@pytest.mark.asyncio
async def test_question_extension_errors_without_client_support() -> None:
    extension = build_question_extension({})

    with pytest.raises(ToolError, match="does not support"):
        await extension.tools[QUESTION_TOOL_NAME](_QUESTIONS[:1])


@pytest.mark.asyncio
async def test_question_extension_resets_pending_on_decline() -> None:
    state: dict[str, Any] = {}

    async def elicit(question: str, labels: list[str], descriptions: list[str]) -> str:
        raise ToolError("declined")

    state["elicit"] = elicit
    extension = build_question_extension(state)

    with pytest.raises(ToolError, match="declined"):
        await extension.tools[QUESTION_TOOL_NAME](_QUESTIONS[:1])
    assert state.get("pending") is False


class _ElicitConn:
    def __init__(self, responses: list[Any]) -> None:
        self.responses = responses
        self.calls: list[tuple[str, Any]] = []
        self.updates: list[Any] = []

    async def create_elicitation(self, message: str, mode: Any, **kwargs: Any) -> Any:
        self.calls.append((message, mode))
        return self.responses.pop(0)

    async def session_update(self, session_id: str, update: Any, source: str = "") -> None:
        self.updates.append(update)


@pytest.mark.asyncio
async def test_elicit_answer_gates_on_client_capabilities(tmp_path: Path) -> None:
    session, _cm = _make_session(tmp_path, StreamingLLM())
    agent = AcpAgent(factory=cast(Any, _FakeFactory(session)))

    with pytest.raises(ToolError, match="does not support"):
        await agent._elicit_answer(session.session_id, "Q?", ["A", "B"], ["", ""])  # noqa: SLF001


@pytest.mark.asyncio
async def test_elicit_answer_returns_accepted_option(tmp_path: Path) -> None:
    session, _cm = _make_session(tmp_path, StreamingLLM())
    agent = AcpAgent(factory=cast(Any, _FakeFactory(session)))
    await agent.initialize(
        protocol_version=1,
        client_capabilities=ClientCapabilities(elicitation=ElicitationCapabilities(form=ElicitationFormCapabilities())),
    )
    conn = _ElicitConn([AcceptElicitationResponse(action="accept", content={"answer": "B"})])
    agent.on_connect(cast(Client, conn))

    answer = await agent._elicit_answer(session.session_id, "Pick one", ["A", "B"], ["", ""])  # noqa: SLF001

    assert answer == "B"
    message, mode = conn.calls[0]
    assert message == "Pick one"
    assert mode.session_id == session.session_id
    schema = mode.requested_schema
    assert schema.required == ["answer"]
    assert [option.const for option in schema.properties["answer"].one_of or []] == ["A", "B"]


@pytest.mark.asyncio
async def test_elicit_answer_errors_on_decline(tmp_path: Path) -> None:
    session, _cm = _make_session(tmp_path, StreamingLLM())
    agent = AcpAgent(factory=cast(Any, _FakeFactory(session)))
    await agent.initialize(
        protocol_version=1,
        client_capabilities=ClientCapabilities(elicitation=ElicitationCapabilities(form=ElicitationFormCapabilities())),
    )
    conn = _ElicitConn([DeclineElicitationResponse(action="decline")])
    agent.on_connect(cast(Client, conn))

    with pytest.raises(ToolError, match="declined"):
        await agent._elicit_answer(session.session_id, "Q?", ["A", "B"], ["", ""])  # noqa: SLF001


@pytest.mark.asyncio
async def test_new_session_binds_elicit(tmp_path: Path) -> None:
    session, _cm = _make_session(tmp_path, StreamingLLM())
    agent = AcpAgent(factory=cast(Any, _FakeFactory(session)))
    await agent.initialize(
        protocol_version=1,
        client_capabilities=ClientCapabilities(elicitation=ElicitationCapabilities(form=ElicitationFormCapabilities())),
    )
    conn = _ElicitConn([AcceptElicitationResponse(action="accept", content={"answer": "A"})])
    agent.on_connect(cast(Client, conn))
    response = await agent.new_session(cwd=str(tmp_path))
    registered = agent._sessions[response.session_id]  # noqa: SLF001

    answer = await registered.question_state["elicit"]("Q?", ["A", "B"], ["", ""])
    assert answer == "A"
    assert conn.calls[0][1].session_id == response.session_id


async def _stub_elicit(question: str, labels: list[str], descriptions: list[str]) -> str:
    return labels[0]
