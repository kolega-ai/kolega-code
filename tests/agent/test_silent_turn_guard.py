"""The silent-turn guard: re-prompt when the model ends a turn with no tool call
and no visible text, instead of finalizing an empty result.

Hermetic: every test scripts the LLM with ``FakeLLM(message_script=[...])``.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from kolega_code.agent.baseagent import BaseAgent
from kolega_code.cli.session_store import SessionStore
from kolega_code.llm.models import Message, TextBlock, ThinkingBlock, ToolCall, ToolResult

from .compaction_helpers import FakeLLM, build_agent

STAGE_MARKERS = [
    "only internal reasoning",
    "two consecutive turns",
    "FINAL WARNING",
]
EXHAUSTED_MARKER = "terminated by the runtime"
TRUNCATED_MARKERS = [
    "cut off mid-reply",
    "twice in a row",
    "three consecutive replies",
]


def _silent_turn_message(stop_reason: str = "end_turn") -> Message:
    return Message(role="assistant", content=[ThinkingBlock(thinking="pondering...")], stop_reason=stop_reason)


def _truncated_text_message() -> Message:
    return Message(role="assistant", content=[TextBlock(text="The fix is to chan")], stop_reason="max_tokens")


def _end_turn_message(text: str = "done") -> Message:
    return Message(role="assistant", content=[TextBlock(text=text)], stop_reason="end_turn")


def _tool_call_message(tool_call: ToolCall) -> Message:
    return Message(
        role="assistant",
        content=[tool_call],
        stop_reason="tool_use",
        tool_calls=[tool_call],
    )


def _tool_result(tool_call: ToolCall) -> ToolResult:
    return ToolResult(
        tool_use_id=tool_call.id,
        name=tool_call.name,
        content="ok",
        is_error=False,
    )


def _configure_agent(agent, message_script: list[Message]) -> None:
    agent.system_prompt = Message(role="system", content=[TextBlock(text="sys")])
    agent.tool_collection = MagicMock()
    agent.tool_collection.get_tool_list = MagicMock(return_value=[])
    agent.llm = FakeLLM(token_script=[100], message_script=message_script)
    agent.log_info = AsyncMock()
    agent.log_error = AsyncMock()


def _user_texts(agent) -> list[str]:
    return [message.get_text_content() for message in agent.history if message.role == "user"]


def _llm_status_events(connection_manager):
    events = []
    for call in connection_manager.broadcast_event.call_args_list:
        event = call.args[0] if call.args else call.kwargs.get("event")
        if event is not None and getattr(event, "event_type", None) == "llm_status_update":
            events.append(event)
    return events


@pytest.mark.asyncio
async def test_thinking_only_turn_is_nudged_then_recovers(base_agent) -> None:
    _configure_agent(base_agent, [_silent_turn_message(), _end_turn_message()])

    _ = [chunk async for chunk in base_agent.process_message_stream("do X")]

    assert base_agent.llm.stream.await_count == 2
    user_texts = _user_texts(base_agent)
    assert any(STAGE_MARKERS[0] in text for text in user_texts)
    assert not any(STAGE_MARKERS[1] in text for text in user_texts)
    assert base_agent.history[-1].role == "assistant"
    assert base_agent.history[-1].get_text_content() == "done"
    assert base_agent.conversation.is_valid_for_anthropic() is True


@pytest.mark.asyncio
async def test_all_silent_turns_exhaust_cap_and_surface_failure(base_agent, mock_connection_manager) -> None:
    _configure_agent(base_agent, [_silent_turn_message() for _ in range(4)])

    chunks = [chunk async for chunk in base_agent.process_message_stream("do X")]

    # Initial attempt + one per nudge; the fourth silent response exhausts the cap.
    assert base_agent.llm.stream.await_count == 4
    user_texts = _user_texts(base_agent)
    stage_positions = []
    for marker in STAGE_MARKERS:
        matches = [index for index, text in enumerate(user_texts) if marker in text]
        assert len(matches) == 1, f"expected exactly one nudge containing {marker!r}"
        stage_positions.append(matches[0])
    assert stage_positions == sorted(stage_positions)
    assert any(EXHAUSTED_MARKER in text for text in user_texts)
    assert any("[no output]" in chunk.get("content", "") for chunk in chunks)

    statuses = _llm_status_events(mock_connection_manager)
    assert sum(1 for event in statuses if event.content.get("status") == "info") == 3
    assert sum(1 for event in statuses if event.content.get("status") == "warning") == 1


@pytest.mark.asyncio
async def test_text_only_turn_not_nudged(base_agent) -> None:
    _configure_agent(base_agent, [_end_turn_message()])

    _ = [chunk async for chunk in base_agent.process_message_stream("do X")]

    assert base_agent.llm.stream.await_count == 1
    assert not any(STAGE_MARKERS[0] in text for text in _user_texts(base_agent))


@pytest.mark.asyncio
async def test_tool_call_turn_not_nudged(base_agent) -> None:
    tool_call = ToolCall(id="tool-1", name="read_file", input={})
    _configure_agent(base_agent, [_tool_call_message(tool_call), _end_turn_message()])
    base_agent.process_tool_calls = AsyncMock(return_value=[_tool_result(tool_call)])

    _ = [chunk async for chunk in base_agent.process_message_stream("do X")]

    assert base_agent.llm.stream.await_count == 2
    assert not any(STAGE_MARKERS[0] in text for text in _user_texts(base_agent))


@pytest.mark.asyncio
async def test_silent_counter_resets_after_productive_turn(base_agent) -> None:
    tool_call = ToolCall(id="tool-1", name="read_file", input={})
    _configure_agent(
        base_agent,
        [
            _silent_turn_message(),
            _tool_call_message(tool_call),
            _silent_turn_message(),
            _silent_turn_message(),
            _silent_turn_message(),
            _end_turn_message(),
        ],
    )
    base_agent.process_tool_calls = AsyncMock(return_value=[_tool_result(tool_call)])

    chunks = [chunk async for chunk in base_agent.process_message_stream("do X")]

    # Without the reset after the productive tool-call iteration, the fourth
    # silent response overall would have exhausted the cap and terminated.
    assert base_agent.llm.stream.await_count == 6
    assert not any("[no output]" in chunk.get("content", "") for chunk in chunks)
    assert base_agent.history[-1].get_text_content() == "done"


@pytest.mark.asyncio
async def test_max_tokens_silent_turn_is_nudged(base_agent) -> None:
    _configure_agent(base_agent, [_silent_turn_message(stop_reason="max_tokens"), _end_turn_message()])

    _ = [chunk async for chunk in base_agent.process_message_stream("do X")]

    assert base_agent.llm.stream.await_count == 2
    assert any(STAGE_MARKERS[0] in text for text in _user_texts(base_agent))


@pytest.mark.asyncio
async def test_deepseek_shaped_max_tokens_stop_reasoning_only_is_nudged(base_agent) -> None:
    # The exact measured DeepSeek runaway signature (TB2.1 trajectories): one
    # huge reasoning-only stream truncated mid-word at the output cap, no tool
    # call, no visible text. Once the wire cap makes the stop reason an honest
    # "max_tokens", this must land in the guard rather than finalize.
    truncated = Message(
        role="assistant",
        content=[ThinkingBlock(thinking="step 4096 of the proof: therefo")],
        stop_reason="max_tokens",
        usage_metadata={"provider": "deepseek", "prompt_tokens": 10, "completion_tokens": 64000},
    )
    _configure_agent(base_agent, [truncated, _end_turn_message()])

    _ = [chunk async for chunk in base_agent.process_message_stream("do X")]

    assert base_agent.llm.stream.await_count == 2
    assert any(STAGE_MARKERS[0] in text for text in _user_texts(base_agent))


@pytest.mark.asyncio
async def test_max_tokens_stop_with_partial_text_is_nudged_then_recovers(base_agent) -> None:
    # The truncated-turn guard: a truncation that cut off mid-sentence in the
    # VISIBLE answer (honest "max_tokens", text, no tool call) gets a
    # continuation prompt instead of finalizing the partial reply.
    truncated = _truncated_text_message()
    _configure_agent(base_agent, [truncated, _end_turn_message()])

    _ = [chunk async for chunk in base_agent.process_message_stream("do X")]

    assert base_agent.llm.stream.await_count == 2
    assert any(TRUNCATED_MARKERS[0] in text for text in _user_texts(base_agent))
    assert not any(STAGE_MARKERS[0] in text for text in _user_texts(base_agent))
    assert base_agent.history[-1].get_text_content() == "done"


@pytest.mark.asyncio
async def test_truncated_turns_exhaust_cap_and_finalize_with_partial_output(
    base_agent, mock_connection_manager
) -> None:
    # Unlike a silent turn there IS content: after the cap the turn finalizes
    # with the partial output rather than reporting "[no output]".
    _configure_agent(base_agent, [_truncated_text_message() for _ in range(4)])

    chunks = [chunk async for chunk in base_agent.process_message_stream("do X")]

    # Initial attempt + one per nudge; the fourth truncation exhausts the cap.
    assert base_agent.llm.stream.await_count == 4
    user_texts = _user_texts(base_agent)
    for marker in TRUNCATED_MARKERS:
        assert sum(1 for text in user_texts if marker in text) == 1, f"expected exactly one nudge with {marker!r}"
    assert not any("[no output]" in chunk.get("content", "") for chunk in chunks)
    assert base_agent.history[-1].role == "assistant"
    assert base_agent.history[-1].get_text_content() == "The fix is to chan"

    statuses = _llm_status_events(mock_connection_manager)
    assert sum(1 for event in statuses if event.content.get("status") == "warning") == 1


@pytest.mark.asyncio
async def test_truncated_counter_resets_after_productive_turn(base_agent) -> None:
    tool_call = ToolCall(id="tool-1", name="read_file", input={})
    _configure_agent(
        base_agent,
        [
            _truncated_text_message(),
            _tool_call_message(tool_call),
            _truncated_text_message(),
            _truncated_text_message(),
            _truncated_text_message(),
            _end_turn_message(),
        ],
    )
    base_agent.process_tool_calls = AsyncMock(return_value=[_tool_result(tool_call)])

    _ = [chunk async for chunk in base_agent.process_message_stream("do X")]

    # Without the reset after the productive tool-call iteration, the fourth
    # truncation overall would have exhausted the cap and finalized early.
    assert base_agent.llm.stream.await_count == 6
    assert base_agent.history[-1].get_text_content() == "done"


@pytest.mark.asyncio
async def test_empty_content_silent_turn_is_nudged(base_agent) -> None:
    empty = Message(role="assistant", content=[], stop_reason="end_turn")
    _configure_agent(base_agent, [empty, _end_turn_message()])

    _ = [chunk async for chunk in base_agent.process_message_stream("do X")]

    # Detection runs on the streamed message, before append_assistant papers the
    # empty content over with a placeholder TextBlock.
    assert base_agent.llm.stream.await_count == 2
    assert any(STAGE_MARKERS[0] in text for text in _user_texts(base_agent))


@pytest.mark.asyncio
async def test_nudge_journal_ordering_and_replay(base_agent, tmp_path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", {})
    base_agent.session_recorder = store.recorder(session.session_id)

    _configure_agent(base_agent, [_silent_turn_message(), _end_turn_message()])

    _ = [chunk async for chunk in base_agent.process_message_stream("do X")]

    event_types = [event.event_type for event in store.journal(session.session_id).read_events()]
    # Context messages: the volatile-context update at the start of the turn,
    # then the nudge injected after the silent assistant response.
    assert event_types[-7:] == [
        "turn.started",
        "context.system",
        "context.message",
        "assistant.message",
        "context.message",
        "assistant.message",
        "turn.completed",
    ]

    replayed = [Message.from_dict(item) for item in store.load(session.session_id).history]
    assert base_agent.history[0] == base_agent.session_context_message
    assert [message.role for message in replayed] == [message.role for message in base_agent.history[1:]]
    assert any(STAGE_MARKERS[0] in message.get_text_content() for message in replayed if message.role == "user")


@pytest.mark.asyncio
async def test_sub_agent_silent_turn_is_nudged(tmp_path) -> None:
    fake_llm = FakeLLM(token_script=[100], message_script=[_silent_turn_message(), _end_turn_message()])
    agent, _ = build_agent(tmp_path, agent_cls=BaseAgent, sub_agent=True, llm=fake_llm)

    _ = [chunk async for chunk in agent.process_message_stream("do X")]

    assert fake_llm.stream.await_count == 2
    assert any(STAGE_MARKERS[0] in text for text in _user_texts(agent))
    assert await agent.recap_agent_outcome() == "done"
